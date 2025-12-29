import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ===============================
# 0. 全局配置
# ===============================
RANDOM_STATE = 42
MAX_LASSO_FEATURES = 20

SAVE_DIR = "../figures/nosofa_comparison"
FEATURE_DIR = "../features"
MODEL_DIR = "../models"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ===============================
# 1. 加载数据
# ===============================
df = pd.read_csv("../data/ap_final_analysis_cohort.csv")
print(f"Total cohort size: {df.shape[0]}")

TARGET = "pof"

# ===============================
# 2. 特征排除逻辑
# ===============================
BASE_EXCLUDE = [
    TARGET, "resp_pof", "cv_pof", "renal_pof", "mortality_28d",
    "sofa_score", "apsiii", "sapsii", "oasis", "lods",
    "mechanical_vent_flag", "vaso_flag", "los",
    "stay_id", "hadm_id", "subject_id",
    "intime", "admittime", "dischtime", "deathtime", "dod",
    "race", "insurance", "language"
]

SENSITIVITY_EXCLUDE = [
    "creatinine_min", "creatinine_max",
    "bun_min", "bun_max",
    "chloride_min", "chloride_max"
]

df_filtered = df.drop(columns=[c for c in BASE_EXCLUDE if c in df.columns])
df_numeric = pd.get_dummies(df_filtered, drop_first=True)

all_features = df_numeric.columns.tolist()
sensitivity_features = [c for c in all_features if c not in SENSITIVITY_EXCLUDE]

experiments = {
    "Main_Analysis": all_features,
    "Sensitivity_No_Renal": sensitivity_features
}

# ===============================
# 3. DeLong Test
# ===============================
def delong_auc_test(y_true, p1, p2):
    from scipy.stats import norm
    def auc_var(y, p):
        pos = p[y == 1]
        neg = p[y == 0]
        v = [(pi > nj) + 0.5*(pi == nj) for pi in pos for nj in neg]
        return np.var(v) / (len(pos)*len(neg))

    auc1 = roc_auc_score(y_true, p1)
    auc2 = roc_auc_score(y_true, p2)
    var = auc_var(y_true, p1) + auc_var(y_true, p2)
    z = (auc1 - auc2) / np.sqrt(var)
    p = 2 * (1 - norm.cdf(abs(z)))
    return auc1 - auc2, p

# ===============================
# 4. 主实验循环
# ===============================
results = {}

for name, feature_list in experiments.items():
    print(f"\n{'='*50}")
    print(f"🚀 Running Experiment: {name}")
    print(f"Initial Feature Count: {len(feature_list)}")

    X = df_numeric[feature_list]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )

    preprocess = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    X_train_s = pd.DataFrame(
        preprocess.fit_transform(X_train),
        columns=X_train.columns, index=X_train.index
    )
    X_test_s = pd.DataFrame(
        preprocess.transform(X_test),
        columns=X_test.columns, index=X_test.index
    )

    # ===============================
    # LASSO（balanced + 特征上限）
    # ===============================
    print("--- Running LASSO Selection ---")
    lasso = LogisticRegressionCV(
        Cs=10,
        cv=5,
        penalty="l1",
        solver="saga",
        scoring="roc_auc",
        class_weight="balanced",
        max_iter=5000,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    lasso.fit(X_train_s, y_train)

    coef = pd.Series(lasso.coef_[0], index=X_train.columns)
    selected = (
        coef[coef != 0]
        .sort_values(key=np.abs, ascending=False)
        .head(MAX_LASSO_FEATURES)
    )

    print(f"LASSO selected {len(selected)} features")

    # 保存特征
    selected.to_csv(
        os.path.join(FEATURE_DIR, f"LASSO_Features_{name}.csv")
    )

    X_train_l = X_train_s[selected.index]
    X_test_l = X_test_s[selected.index]

    # ===============================
    # SMOTE（仅 LASSO 后）
    # ===============================
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train_l, y_train)

    # ===============================
    # XGBoost
    # ===============================
    model = XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.02,
        gamma=1.0,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=5,
        reg_lambda=2.0,
        eval_metric="logloss",
        random_state=RANDOM_STATE
    )
    model.fit(X_res, y_res)
    
    if name == "Main_Analysis":
        print(f"--- Saving Objects for {name} ---")
        
        # 1. 保存模型
        joblib.dump(model, os.path.join(MODEL_DIR, 'final_xgb_model.pkl'))
        
        # 2. 保存特征清单 (注意：保存的是 LASSO 最终选择的特征)
        selected_features = selected.index.tolist()
        with open(os.path.join(MODEL_DIR, 'feature_list_main.txt'), 'w') as f:
            for feat in selected_features:
                f.write(f"{feat}\n")
        
        # 3. 保存预处理 Pipeline (包含 Imputer 和 Scaler)
        # 提示：由于 eICU 数据也需要同样的标准化和中位数填补，保存这个对象非常重要
        joblib.dump(preprocess, os.path.join(MODEL_DIR, 'scaler.pkl'))
        
        print(f"✅ Successfully saved model, feature list, and scaler to {MODEL_DIR}")
        
    y_prob = model.predict_proba(X_test_l)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"{name} XGBoost AUC: {auc:.4f}")

    results[name] = {
        "auc": auc,
        "y_true": y_test,
        "y_prob": y_prob,
        "model": model,
        "X_test": X_test_l
    }

# ===============================
# 5. ΔAUC 统计比较
# ===============================
delta_auc, p_value = delong_auc_test(
    results["Main_Analysis"]["y_true"],
    results["Main_Analysis"]["y_prob"],
    results["Sensitivity_No_Renal"]["y_prob"]
)

print("\n=== ΔAUC Comparison ===")
print(f"ΔAUC (Main − Sensitivity): {delta_auc:.4f}")
print(f"P-value (DeLong): {p_value:.4f}")

# ===============================
# 6. SHAP（Main + Sensitivity）
# ===============================
for name, res in results.items():
    print(f"--- Generating SHAP: {name} ---")
    explainer = shap.TreeExplainer(res["model"])
    shap_values = explainer.shap_values(res["X_test"])

    plt.figure()
    shap.summary_plot(shap_values, res["X_test"], show=False)
    plt.title(f"SHAP Summary: {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"SHAP_{name}.png"), dpi=300)

print("\n✅ All analyses and model saving completed successfully.")
