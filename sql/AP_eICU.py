import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss, 
                             f1_score, precision_score, recall_score, 
                             confusion_matrix)
from sklearn.calibration import calibration_curve

# =========================================================
# 1. 路径与配置 (与训练脚本保持一致)
# =========================================================
DATA_PATH = "../data/ap_eicu_validation.csv" 
MODEL_DIR = "../models"
SAVE_DIR = "../figures/external_validation"
os.makedirs(SAVE_DIR, exist_ok=True)

# 训练集确定的最佳阈值 (根据你上一步 Main_Analysis XGBoost 的输出填写)
BEST_THRESH = 0.480 
RANDOM_STATE = 42
N_BOOTSTRAPS = 1000

# =========================================================
# 2. 工具函数 (与训练脚本对齐)
# =========================================================
def get_auc_ci(y_true, y_prob, n_bootstraps=1000, rng_seed=42):
    rng = np.random.RandomState(rng_seed)
    bootstrapped_scores = []
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_prob), len(y_prob))
        if len(np.unique(y_true[indices])) < 2:
            continue
        bootstrapped_scores.append(roc_auc_score(y_true[indices], y_prob[indices]))
    sorted_scores = np.sort(bootstrapped_scores)
    return sorted_scores[int(0.025 * len(sorted_scores))], sorted_scores[int(0.975 * len(sorted_scores))]

# =========================================================
# 3. 加载模型及附件
# =========================================================
print("\n--- Loading Calibrated Model and Pipeline ---")
model = joblib.load(os.path.join(MODEL_DIR, 'final_xgb_model.pkl'))
scaler_pipeline = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))

with open(os.path.join(MODEL_DIR, 'feature_list_main.txt'), 'r') as f:
    lasso_features = [line.strip() for line in f]

fit_time_features = scaler_pipeline.feature_names_in_

df_eicu = pd.read_csv(DATA_PATH)
print(f"eICU Dataset Loaded: {df_eicu.shape}")

# =========================================================
# 4. 特征对齐映射 (MIMIC Name -> eICU Name)
# =========================================================
mapping = {
    'admission_age': 'age',
    'weight_admit': 'weight',
    'height_admit': 'height',
    'calcium_min': 'lab_calcium_min',
    'respiratory_rate_max': 'resp_rate_max',
    'respiratory_rate_min': 'resp_rate_min',
    'heart_rate_max': 'heart_rate_max',
    'heart_rate_min': 'heart_rate_min',
    'temperature_max': 'temp_max',
    'temperature_min': 'temp_min',
    'platelets_min': 'platelet_min'
}

X_eicu_aligned = pd.DataFrame(index=df_eicu.index)

print("--- Aligning Features ---")
for col in fit_time_features:
    if col in df_eicu.columns:
        X_eicu_aligned[col] = df_eicu[col]
    elif col in mapping and mapping[col] in df_eicu.columns:
        X_eicu_aligned[col] = df_eicu[mapping[col]]
    elif col == 'gender_M' and 'gender' in df_eicu.columns:
        X_eicu_aligned[col] = (df_eicu['gender'] == 1).astype(int)
    else:
        X_eicu_aligned[col] = np.nan

# =========================================================
# 5. 预处理与预测
# =========================================================
# 使用训练集得到的 scaler 进行转换
X_eicu_transformed = scaler_pipeline.transform(X_eicu_aligned)
X_eicu_preprocessed = pd.DataFrame(X_eicu_transformed, columns=fit_time_features)

# 锁定 LASSO 特征
X_final = X_eicu_preprocessed[lasso_features]
y_true = df_eicu['pof_proxy']

# 使用校准模型输出概率
y_probs = model.predict_proba(X_final)[:, 1]
# 使用训练集最佳阈值进行分类
y_pred = (y_probs >= BEST_THRESH).astype(int)

# =========================================================
# 6. 性能指标计算 (全量对齐)
# =========================================================
auc_val = roc_auc_score(y_true, y_probs)
ci_low, ci_high = get_auc_ci(y_true, y_probs, n_bootstraps=N_BOOTSTRAPS)
brier = brier_score_loss(y_true, y_probs)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
sens = tp / (tp + fn)
spec = tn / (tn + fp)
prec = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"\n" + "="*50)
print(f"{'EXTERNAL VALIDATION RESULTS (eICU)':^50}")
print("-" * 50)
print(f"ROC-AUC:       {auc_val:.4f} (95% CI: {ci_low:.3f}-{ci_high:.3f})")
print(f"Brier Score:   {brier:.4f}")
print(f"Threshold:     {BEST_THRESH} (Optimized in MIMIC)")
print(f"Sensitivity:   {sens:.4f}")
print(f"Specificity:   {spec:.4f}")
print(f"Precision:     {prec:.4f}")
print(f"F1-Score:      {f1:.4f}")
print("=" * 50)

# =========================================================
# 7. 可视化 (ROC & Calibration)
# =========================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# ROC 曲线
fpr, tpr, _ = roc_curve(y_true, y_probs)
ax1.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'eICU External (AUC = {auc_val:.3f})')
ax1.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve: eICU Validation')
ax1.legend(loc='lower right')

# 校准曲线 (Reliability Diagram)
prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
ax2.plot(prob_pred, prob_true, marker='s', markersize=4, lw=2, label='Calibrated XGBoost', color='#ff7f0e')
ax2.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.5, label='Perfectly Calibrated')
ax2.set_xlabel('Predicted Probability')
ax2.set_ylabel('Actual Proportion')
ax2.set_title('Calibration Plot: eICU Validation')
ax2.legend(loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "eicu_metrics_combined.png"), dpi=300)

# =========================================================
# 8. SHAP 解释性分析 (修复版)
# =========================================================
print("\n--- Generating SHAP Analysis ---")

# 关键修复：从 CalibratedClassifierCV 中提取已经 fit 过的原始 XGB 模型
# model.calibrated_classifiers_ 是一个列表，我们取第一个
try:
    # 提取内部的具体某个基模型对象
    calibrated_sub_model = model.calibrated_classifiers_[0]
    
    # 不同的 sklearn 版本可能略有不同，尝试提取 base_estimator
    if hasattr(calibrated_sub_model, 'estimator'):
        base_model_for_shap = calibrated_sub_model.estimator
    else:
        base_model_for_shap = calibrated_sub_model.base_estimator
        
    print(f"Successfully extracted base model type: {type(base_model_for_shap)}")
    
    # 初始化 TreeExplainer
    explainer = shap.TreeExplainer(base_model_for_shap)
    shap_values = explainer.shap_values(X_final)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_final, show=False, max_display=15)
    plt.title(f"SHAP Summary (eICU Validation, n={len(df_eicu)})", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "eicu_shap_summary.png"), dpi=300)
    print("✅ SHAP plot saved successfully.")

except Exception as e:
    print(f"❌ SHAP error still persists: {e}")
    print("Hint: If error persists, ensure X_final columns match training features exactly.")
