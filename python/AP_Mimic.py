import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import seaborn as sns
from scipy.stats import norm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, confusion_matrix
from imblearn.over_sampling import SMOTE

# =========================================================
# 0. 全局配置与路径
# =========================================================
RANDOM_STATE = 42
MAX_LASSO_FEATURES = 15
N_BOOTSTRAPS = 1000

BASE_DIR = ".."
DATA_PATH = os.path.join(BASE_DIR, "data/ap_final_analysis_cohort.csv")
SAVE_DIR_FIG = os.path.join(BASE_DIR, "figures/final_robust")
SAVE_DIR_MODEL = os.path.join(BASE_DIR, "models")
SAVE_DIR_DATA = os.path.join(BASE_DIR, "data/cleaned") # 新增清洗后数据存放路径

for d in [SAVE_DIR_FIG, SAVE_DIR_MODEL, SAVE_DIR_DATA]:
    os.makedirs(d, exist_ok=True)

# =========================================================
# 1. 深度数据清洗函数 (分阶段处理)
# =========================================================

def clinical_winsorization(data):
    """阶段1：离群值盖帽 (用于 Table 1)"""
    df = data.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # 排除标签和分类标识符
        if col not in ['pof', 'gender_num', 'subject_id', 'hadm_id', 'stay_id'] and '_flag' not in col:
            upper = df[col].quantile(0.99)
            lower = df[col].quantile(0.01)
            df[col] = df[col].clip(lower, upper)
    return df
def clinical_feature_engineering(data):
    """
    阶段2：单位自动校准、非线性变换与共线性处理
    """
    df = data.copy()
    
    # === 1. 单位自动校准 (针对 Table 1 发现的单位不统一问题) ===
    
    # A. 纤维蛋白原 (Fibrinogen) 校准
    # MIMIC 通常使用 mg/dL (均值约 300-400)，eICU 可能使用 g/L (均值约 2-4)
    # 如果中位数非常小（比如 < 50），则判定为 g/L，乘以 100 转换成 mg/dL
    if 'fibrinogen_max' in df.columns:
        median_val = df['fibrinogen_max'].median()
        if not pd.isna(median_val) and median_val < 50:
            print(f"🔄 检测到 Fibrinogen 单位异常 (Median={median_val:.2f}), 正在从 g/L 转换为 mg/dL...")
            df['fibrinogen_max'] = df['fibrinogen_max'] * 100

    # B. 阴离子间隙 (Anion Gap) 校准
    # 如果发现 eICU 的 Anion Gap 整体偏低，可能是计算公式差异，
    # 临床研究中通常使用 Z-Score 或直接对齐均值（此处建议先检查是否为量纲问题）
    # 如果有明确的倍数关系，在此处添加转换逻辑
    
    # === 2. 偏态指标 Log 转换 (针对模型优化) ===
    # 包含你 SHAP 图中重要的几个连续变量
    skewed_cols = [
        'amylase_max', 'lipase_max', 'crp_max', 'fibrinogen_max', 
        'wbc_max', 'creatinine_min', 'bun_max', 'glucose_max', 'lactate_max'
    ]
    for col in skewed_cols:
        if col in df.columns:
            # 使用 log1p (ln(x+1)) 处理，并进行 clip 防止负值
            df[col] = np.log1p(df[col].astype(float).clip(lower=0))
    
    # === 3. 共线性检测 (保持原有逻辑) ===
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
    # 确保不删掉目标变量和关键 ID
    to_drop = [c for c in to_drop if c not in ['pof', 'subject_id', 'hadm_id', 'stay_id']]
    
    if to_drop:
        print(f"📉 移除高度共线性特征 (>0.9): {to_drop}")
        df = df.drop(columns=to_drop)
        
    return df

# =========================================================
# 2. 数据准备与保存逻辑
# =========================================================
df_raw = pd.read_csv(DATA_PATH)

# --- 生成并保存 Table 1 版本 ---
# 仅截断异常值，保留原始量纲 (mg/dL, mmol/L 等)
df_table1 = clinical_winsorization(df_raw)
df_table1.to_csv(os.path.join(SAVE_DIR_DATA, "mimic_for_table1.csv"), index=False)
print("✅ Table 1 数据已保存 (保留原始单位).")

# --- 生成并保存模型版本 ---
# 执行 Log 转换和共线性剔除
df_model_ready = clinical_feature_engineering(df_table1)
df_model_ready.to_csv(os.path.join(SAVE_DIR_DATA, "mimic_for_model.csv"), index=False)
print("✅ 模型训练数据已保存 (已执行 Log 转换).")

# =========================================================
# 3. 建模流程 (使用 df_model_ready)
# =========================================================
TARGET = "pof"

# A. 临床结局与标识符 (必须排除)
IDENTIFIERS = ["stay_id", "hadm_id", "subject_id", "intime", "admittime", 
               "dischtime", "deathtime", "dod", "race", "insurance"]

# B. 数据泄露指标 (绝对不能出现在预测因子中)
LEAKAGE_METRICS = [
    "los",                   # 住院时长是结果，不是预测因子
    "mortality_28d",         # 死亡结局
    "resp_pof", "cv_pof", "renal_pof" # 结局的子组成部分
]

# C. 治疗干预指标 (因果倒置风险)
# POF 的定义依赖于这些干预，包含它们会让 AUC 虚高
TREATMENT_INTERVENTION = [
    "vaso_flag",             # 升压药使用情况
    "mechanical_vent_flag",  # 机械通气使用情况
]

# D. 评分系统及其强相关子项
# 如果你的结局 POF 是基于 SOFA 定义的，最好排除其对应的生理比值
SCORING_SYSTEMS = [
    "sofa_score", "apsiii", "sapsii", "oasis", "lods",
    "pao2fio2ratio_min"      # 氧合指数是 SOFA 呼吸评分的核心，建议排除
]

# 汇总最终排除列表
BASE_EXCLUDE = [TARGET] + IDENTIFIERS + LEAKAGE_METRICS + TREATMENT_INTERVENTION + SCORING_SYSTEMS

X = pd.get_dummies(df_model_ready.drop(columns=[c for c in BASE_EXCLUDE if c in df_model_ready.columns]), drop_first=True)
y = df_model_ready[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)

# 标准化 (Scaler 仅应用于模型输入)
scaler_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
X_train_s = pd.DataFrame(scaler_pipe.fit_transform(X_train), columns=X_train.columns)
X_test_s = pd.DataFrame(scaler_pipe.transform(X_test), columns=X_test.columns)

# --- LASSO 筛选 ---
lasso = LogisticRegressionCV(Cs=15, cv=5, penalty="l1", solver="liblinear", scoring="roc_auc", random_state=RANDOM_STATE)
lasso.fit(X_train_s, y_train)
selected_feats = pd.Series(lasso.coef_[0], index=X_train_s.columns).abs().sort_values(ascending=False).head(MAX_LASSO_FEATURES).index.tolist()

X_train_l, X_test_l = X_train_s[selected_feats], X_test_s[selected_feats]

# --- SMOTE ---
X_res, y_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train_l, y_train)

# --- 训练与校准 ---
results_store = {}
models = {
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=RANDOM_STATE),
    "Logistic": LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE)
}

print(f"\n{'Model':<12} | {'AUC (95% CI)':<22} | {'Sens':<6} | {'Spec':<6} | {'Brier':<6}")
print("-" * 75)

for name, m in models.items():
    calibrated = CalibratedClassifierCV(m, method='isotonic', cv=3)
    calibrated.fit(X_res, y_res)
    y_prob = calibrated.predict_proba(X_test_l)[:, 1]
    
    auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    ix = np.argmax(tpr - fpr)
    sens, spec = tpr[ix], 1 - fpr[ix]
    
    print(f"{name:<12} | {auc:.3f} | {sens:.3f} | {spec:.3f} | {brier_score_loss(y_test, y_prob):.3f}")
    results_store[name] = {"y_prob": y_prob, "model": calibrated}

# =========================================================
# 4. 可视化
# =========================================================

# --- A. 校准曲线 ---
plt.figure(figsize=(8, 6))
for name in results_store:
    prob_true, prob_pred = calibration_curve(y_test, results_store[name]["y_prob"], n_bins=10)
    plt.plot(prob_pred, prob_true, "s-", label=f"{name}")
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.title("Calibration Curve (Isotonic)")
plt.xlabel("Predicted Probability")
plt.ylabel("Actual Probability")
plt.legend()
plt.savefig(os.path.join(SAVE_DIR_FIG, "Calibration_Curve.png"))

# --- B. SHAP 解释 (针对 XGBoost) ---
print("\n--- Generating SHAP Analysis ---")
try:
    # 兼容性修正：尝试使用 .estimator，如果不行则 fallback 到 .base_estimator
    calibrated_model = results_store["XGBoost"]["model"]
    first_clf = calibrated_model.calibrated_classifiers_[0]
    
    if hasattr(first_clf, 'estimator'):
        best_xgb = first_clf.estimator
    else:
        best_xgb = first_clf.base_estimator

    # 确保 SHAP 能够识别特征名
    # XGBoost 在 SHAP 中有时需要显式指定特征名
    explainer = shap.TreeExplainer(best_xgb)
    shap_values = explainer.shap_values(X_test_l)

    plt.figure(figsize=(12, 8))
    # 使用 beeswarm 图可以更直观地看到特征对结果的正负影响
    shap.summary_plot(shap_values, X_test_l, show=False, max_display=15)
    plt.title("SHAP Feature Importance (XGBoost)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR_FIG, "XGB_SHAP_Beeswarm.png"), bbox_inches='tight')
    print("✅ SHAP Beeswarm plot saved.")

except Exception as e:
    print(f"⚠️ SHAP plotting failed: {e}")

# 保存最终的模型和特征列表
joblib.dump(results_store["XGBoost"]["model"], os.path.join(SAVE_DIR_MODEL, "calibrated_xgb.pkl"))
joblib.dump(scaler_pipe, os.path.join(SAVE_DIR_MODEL, "scaler_pipe.pkl"))
# 记录 LASSO 选中的特征名，方便 eICU 验证时对齐
with open(os.path.join(SAVE_DIR_MODEL, "selected_features.txt"), "w") as f:
    for feat in selected_feats:
        f.write(f"{feat}\n")

print(f"\n✅ 全部完成。数据保存在: {SAVE_DIR_DATA}")
