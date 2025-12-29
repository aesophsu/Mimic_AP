# =========================================================
# 0. Environment & Imports
# =========================================================

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

import shap
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
SAVE_DIR = "../figures/nosofa"
os.makedirs(SAVE_DIR, exist_ok=True)


# =========================================================
# 1. Load Data
# =========================================================

df = pd.read_csv("../data/ap_final_analysis_cohort.csv")

print(f"Total cohort size: {df.shape[0]}")


# =========================================================
# 2. Define Outcome & Feature Set (NO-SOFA)
# =========================================================
# ---- 1. 处理类别变量 (One-Hot Encoding) ----
# 注意：一定要先剔除 ID 和 时间戳，否则 get_dummies 会为每个时间点创建一个特征！
DROP_BEFORE_DUMMIES = ["stay_id", "hadm_id", "subject_id", "intime", "admittime", "dischtime", "deathtime", "dod"]
df_filtered = df.drop(columns=[c for c in DROP_BEFORE_DUMMIES if c in df.columns])

df_numeric = pd.get_dummies(df_filtered, drop_first=True)

# ---- 2. 构建主特征集 (Main Set) ----
TARGET = "pof"
EXCLUDE_COLS = [
    TARGET, "resp_pof", "cv_pof", "renal_pof", "mortality_28d",
    "sofa_score", "apsiii", "sapsii", "oasis", "lods",
    "mechanical_vent_flag", "vaso_flag", "los", "insurance", "race"
]

FEATURE_COLS = [c for c in df_numeric.columns if c not in EXCLUDE_COLS]

X = df_numeric[FEATURE_COLS]
y = df[TARGET] # 标签直接从原始 df 取，或者从 df_numeric 取均可

# ---- 3. 构建敏感性特征集 (Sensitivity Set) ----
SENSITIVITY_EXCLUDE = [
    "creatinine_min", "creatinine_max",
    "bun_min", "bun_max",
    "chloride_min", "chloride_max"
]

FEATURE_COLS_SENS = [c for c in FEATURE_COLS if c not in SENSITIVITY_EXCLUDE]

# 这一步已经不需要 NUMERIC_WHITELIST 检查了，因为 df_numeric 保证全是数值
X_sens = df_numeric[FEATURE_COLS_SENS]

print(f"Main feature count: {X.shape[1]}")
print(f"Sensitivity feature count: {X_sens.shape[1]}")


# =========================================================
# 3. Train / Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_sens, y,
    test_size=0.30,
    stratify=y,
    random_state=RANDOM_STATE
)

print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")


# =========================================================
# 4. Imputation + Scaling (Fit on Train Only)
# =========================================================

preprocess = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

X_train_scaled = pd.DataFrame(
    preprocess.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    preprocess.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)


# =========================================================
# 5. LASSO Feature Selection (Logistic, NO SMOTE)
# =========================================================

print("\n=== LASSO Feature Selection (L1 Logistic) ===")

lasso = LogisticRegressionCV(
    Cs=10,
    cv=5,
    penalty="l1",
    solver="saga",
    scoring="roc_auc",
    max_iter=4000,
    n_jobs=-1,
    random_state=RANDOM_STATE
)

lasso.fit(X_train_scaled, y_train)

coef = pd.Series(lasso.coef_[0], index=X_train.columns)
selected_features = coef[coef != 0].index.tolist()

print(f"LASSO selected {len(selected_features)} features")
print(coef.sort_values(ascending=False).head(10))

X_train_lasso = X_train_scaled[selected_features]
X_test_lasso = X_test_scaled[selected_features]


# =========================================================
# 6. SMOTE (ONLY for model training)
# =========================================================

smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train_lasso, y_train)

print(f"After SMOTE: n={len(y_train_res)}, POF rate={y_train_res.mean():.2f}")


# =========================================================
# 7. Model Training
# =========================================================

models = {
    "Logistic": LogisticRegression(max_iter=2000),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        random_state=RANDOM_STATE
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=RANDOM_STATE
    )
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_res, y_train_res)

    y_prob = model.predict_proba(X_test_lasso)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    results[name] = {
        "model": model,
        "prob": y_prob,
        "auc": auc
    }

    print(f"{name} Test AUC: {auc:.3f}")


# =========================================================
# 8. SOFA Benchmark (NOT a feature)
# =========================================================

sofa_test = df.loc[y_test.index, "sofa_score"]
sofa_auc = roc_auc_score(y_test, sofa_test)

print(f"\nSOFA Benchmark AUC: {sofa_auc:.3f}")

results["SOFA"] = {
    "prob": sofa_test / 24.0,
    "auc": sofa_auc
}


# =========================================================
# 9. ROC Curve
# =========================================================

plt.figure(figsize=(7, 6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["prob"])
    plt.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – No-SOFA ML Model")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "ROC_NoSOFA.png"), dpi=300)
plt.close()


# =========================================================
# 10. SHAP Interpretation (XGBoost)
# =========================================================

best_model = results["XGBoost"]["model"]


explainer = shap.TreeExplainer(best_model) 
shap_values = explainer.shap_values(X_test_lasso)

plt.figure()
shap.summary_plot(
    shap_values,
    X_test_lasso,
    show=False
)
plt.title("SHAP Summary – No-SOFA Model")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "SHAP_Beeswarm_NoSOFA.png"), dpi=300)
plt.close()


# =========================================================
# 11. Save Feature List
# =========================================================

coef[selected_features].sort_values(
    key=np.abs, ascending=False
).to_csv(os.path.join(SAVE_DIR, "LASSO_Selected_Features.csv"))

print("\n=== No-SOFA ML pipeline completed successfully ===")
