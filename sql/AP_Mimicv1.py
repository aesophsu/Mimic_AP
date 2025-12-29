import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# =========================================================
# 0. 配置与路径
# =========================================================
RANDOM_STATE = 42
SAVE_DIR = "../figures/nosofa_comparison"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================================================
# 1. 加载数据
# =========================================================
df = pd.read_csv("../data/ap_final_analysis_cohort.csv")
print(f"Total cohort size: {df.shape[0]}")

# =========================================================
# 2. 特征定义与严格排除逻辑
# =========================================================
TARGET = "pof"

# 1. 基础排除：结局变量、ID、时间戳、以及非临床干扰变量（Race, Insurance）
BASE_EXCLUDE = [
    TARGET, "resp_pof", "cv_pof", "renal_pof", "mortality_28d",
    "sofa_score", "apsiii", "sapsii", "oasis", "lods",
    "mechanical_vent_flag", "vaso_flag", "los",
    "stay_id", "hadm_id", "subject_id",
    "intime", "admittime", "dischtime", "deathtime", "dod",
    "race", "insurance", "language" 
]

# 2. 敏感性排除列表 (肾功能指标)
SENSITIVITY_EXCLUDE = [
    "creatinine_min", "creatinine_max", "bun_min", "bun_max", "chloride_min", "chloride_max"
]

# ---- 处理类别变量：先删除 ID/时间戳再做 One-Hot，防止特征爆炸 ----
df_filtered = df.drop(columns=[c for c in BASE_EXCLUDE if c in df.columns])
df_numeric = pd.get_dummies(df_filtered, drop_first=True)

# 3. 定义实验组
all_clinical_features = df_numeric.columns.tolist()
sensitivity_features = [c for c in all_clinical_features if c not in SENSITIVITY_EXCLUDE]

experiments = {
    "Main_Analysis": all_clinical_features,
    "Sensitivity_No_Renal": sensitivity_features
}

# =========================================================
# 3. 自动化实验循环
# =========================================================
exp_results = {}

for exp_name, feature_list in experiments.items():
    print(f"\n{'='*50}")
    print(f"🚀 Running Experiment: {exp_name}")
    print(f"Initial Feature Count: {len(feature_list)}")

    X_exp = df_numeric[feature_list]
    y_exp = df[TARGET]

    # 1. 拆分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_exp, y_exp, test_size=0.30, stratify=y_exp, random_state=RANDOM_STATE
    )

    # 2. 预处理：填补与标准化
    preprocess = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    X_train_scaled = pd.DataFrame(preprocess.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(preprocess.transform(X_test), columns=X_test.columns, index=X_test.index)

    # 3. LASSO 特征筛选
    print("--- Running LASSO Selection ---")
    lasso = LogisticRegressionCV(
        Cs=10, cv=5, penalty="l1", solver="saga", scoring="roc_auc", 
        max_iter=5000, n_jobs=-1, random_state=RANDOM_STATE
    )
    lasso.fit(X_train_scaled, y_train)
    
    coef = pd.Series(lasso.coef_[0], index=X_train.columns)
    selected_features = coef[coef != 0].index.tolist()
    print(f"LASSO selected {len(selected_features)} features")

    X_train_lasso = X_train_scaled[selected_features]
    X_test_lasso = X_test_scaled[selected_features]

    # 4. SMOTE (仅针对训练集)
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train_lasso, y_train)

    # 5. 模型训练与评估 (针对小样本微调后的 XGBoost)
    print("--- Training Tuned XGBoost Model ---")
    model = XGBClassifier(
        n_estimators=500,       # 增加树的数量
        max_depth=3,            # 降低深度防止过拟合
        learning_rate=0.02,     # 降低步长提高泛化
        gamma=1.0,              # 增加分裂门槛
        subsample=0.7,          # 样本扰动
        colsample_bytree=0.7,   # 特征扰动
        min_child_weight=5,     # 限制叶子节点最小权重
        reg_lambda=2.0,         # L2 正则化
        eval_metric="logloss",
        random_state=RANDOM_STATE
    )
    model.fit(X_train_res, y_train_res)
    
    y_prob = model.predict_proba(X_test_lasso)[:, 1]
    auc_score = roc_auc_score(y_test, y_prob)
    print(f"{exp_name} XGBoost AUC: {auc_score:.4f}")

    # 存储结果用于对比
    exp_results[exp_name] = {
        "y_true": y_test,
        "y_prob": y_prob,
        "auc": auc_score,
        "selected_features": selected_features,
        "model": model,
        "X_test": X_test_lasso
    }

    # 6. 保存该组实验的特征系数
    coef[selected_features].sort_values(ascending=False).to_csv(
        os.path.join(SAVE_DIR, f"Features_{exp_name}.csv")
    )

# =========================================================
# 4. 最终对比可视化
# =========================================================

# 1. 综合 ROC 曲线
plt.figure(figsize=(8, 7))
for name, data in exp_results.items():
    fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {data['auc']:.3f})")

# 加入 SOFA Benchmark 参照
sofa_test = df.loc[exp_results["Main_Analysis"]["y_true"].index, "sofa_score"]
fpr_s, tpr_s, _ = roc_curve(exp_results["Main_Analysis"]["y_true"], sofa_test)
plt.plot(fpr_s, tpr_s, 'k--', alpha=0.5, label=f"SOFA Benchmark (AUC = {roc_auc_score(exp_results['Main_Analysis']['y_true'], sofa_test):.3f})")

plt.plot([0, 1], [0, 1], color='gray', linestyle=':', lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Comparison: Main vs Sensitivity Analysis")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "Final_Comparison_ROC.png"), dpi=300)

# 2. SHAP 对比 (针对 Main Analysis)
print("\n--- Generating SHAP for Main Analysis ---")
best_data = exp_results["Main_Analysis"]
explainer = shap.TreeExplainer(best_data["model"])
shap_values = explainer.shap_values(best_data["X_test"])

plt.figure()
shap.summary_plot(shap_values, best_data["X_test"], show=False)
plt.title("SHAP Summary: Main Analysis (Tuned Model)")
plt.savefig(os.path.join(SAVE_DIR, "SHAP_Main_Analysis.png"), dpi=300)

print(f"\n✅ All experiments completed. Figures saved to {SAVE_DIR}")
