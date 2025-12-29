import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, f1_score, precision_score, recall_score
from sklearn.calibration import calibration_curve

# ===============================
# 1. 路径与配置
# ===============================
DATA_PATH = "../data/ap_eicu_validation.csv"  # 请确保这是 SQL V8 导出的结果
MODEL_DIR = "../models"
SAVE_DIR = "../figures/external_validation"
os.makedirs(SAVE_DIR, exist_ok=True)

# 初步读取以检查列名
df_check = pd.read_csv(DATA_PATH)
print("--- 1. CSV 真实列名列表 ---")
print(df_check.columns.tolist())

# ===============================
# 2. 加载模型及附件
# ===============================
print("\n--- Loading Model and Preprocessing Objects ---")
model = joblib.load(os.path.join(MODEL_DIR, 'final_xgb_model.pkl'))
scaler_pipeline = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))

# 加载训练时确定的 LASSO 特征列表
with open(os.path.join(MODEL_DIR, 'feature_list_main.txt'), 'r') as f:
    lasso_features = [line.strip() for line in f]

# 获取标准化器训练时看到的原始特征顺序
fit_time_features = scaler_pipeline.feature_names_in_

df_eicu = pd.read_csv(DATA_PATH)
print(f"eICU Dataset Loaded: {df_eicu.shape}")

# ===============================
# 3. 特征对齐映射 (MIMIC Name -> eICU Name)
# ===============================
# 映射原则：左边是模型需要的名称，右边是 SQL V8 导出的真实列名
mapping = {
    'admission_age': 'age',
    'weight_admit': 'weight',
    'height_admit': 'height',
    'calcium_min': 'lab_calcium_min',
    'albumin_min': 'albumin_min',
    'albumin_max': 'albumin_max',
    'respiratory_rate_max': 'resp_rate_max',
    'respiratory_rate_min': 'resp_rate_min',
    'heart_rate_max': 'heart_rate_max',
    'heart_rate_min': 'heart_rate_min',
    'mbp_min': 'mbp_min',
    'sbp_min': 'sbp_min',
    'temperature_max': 'temp_max',
    'temperature_min': 'temp_min',
    'spo2_max': 'spo2_max',
    'glucose_max': 'glucose_max',
    'platelets_min': 'platelet_min',
    'hemoglobin_min': 'hemoglobin_min',
    'aniongap_max': 'aniongap_max',
    'sodium_max': 'sodium_max',
    'potassium_max': 'potassium_max'
}

X_eicu_aligned = pd.DataFrame(index=df_eicu.index)

print("--- Aligning Features ---")
for col in fit_time_features:
    # 1. 直接匹配 (如 ph_min, bun_min, bun_max, creatinine_max, wbc_max, lactate_max, ptt_max)
    if col in df_eicu.columns:
        X_eicu_aligned[col] = df_eicu[col]
    # 2. 映射匹配
    elif col in mapping and mapping[col] in df_eicu.columns:
        X_eicu_aligned[col] = df_eicu[mapping[col]]
    # 3. 性别编码转换
    elif col == 'gender_M' and 'gender' in df_eicu.columns:
        X_eicu_aligned[col] = (df_eicu['gender'] == 1).astype(int)
    # 4. 其他字段
    elif col in ['bmi', 'vaso_flag', 'vent_flag'] and col in df_eicu.columns:
        X_eicu_aligned[col] = df_eicu[col]
    else:
        # 记录缺失特征以便排查
        X_eicu_aligned[col] = np.nan

# ===============================
# 4. 数据完整性审计
# ===============================
print("\n--- Data Integrity Audit (LASSO Features) ---")
missing_info = X_eicu_aligned[lasso_features].isnull().mean() * 100
print(f"Average Missing Rate in Key Features: {missing_info.mean():.2f}%")
highly_missing = missing_info[missing_info > 10].sort_values(ascending=False)
if not highly_missing.empty:
    print("Top Missing Features (>10%):")
    print(highly_missing)

# ===============================
# 5. 预处理与预测
# ===============================
# 执行插补和标准化 (使用训练集的 pipeline)
X_eicu_transformed = scaler_pipeline.transform(X_eicu_aligned)
X_eicu_preprocessed = pd.DataFrame(X_eicu_transformed, columns=fit_time_features)

# 仅保留进入模型的特征
X_final = X_eicu_preprocessed[lasso_features]
y_true = df_eicu['pof_proxy']

# 模型推理
y_probs = model.predict_proba(X_final)[:, 1]
y_pred = (y_probs > 0.5).astype(int)

# ===============================
# 6. 性能指标计算
# ===============================
auc_score = roc_auc_score(y_true, y_probs)
brier = brier_score_loss(y_true, y_probs)
f1 = f1_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print(f"\n" + "="*30)
print(f"ROC-AUC:       {auc_score:.4f}")
print(f"Brier Score:   {brier:.4f}")
print(f"F1-Score:      {f1:.4f}")
print(f"Precision:     {precision:.4f}")
print(f"Recall:        {recall:.4f}")
print("="*30)

# ===============================
# 7. 绘图：结果可视化
# ===============================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# ROC 曲线
fpr, tpr, _ = roc_curve(y_true, y_probs)
ax1.plot(fpr, tpr, color='#2c7da0', lw=2.5, label=f'eICU External (AUC = {auc_score:.4f})')
ax1.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('False Positive Rate (1 - Specificity)')
ax1.set_ylabel('True Positive Rate (Sensitivity)')
ax1.set_title('ROC Curve: eICU External Validation')
ax1.legend(loc='lower right')
ax1.grid(alpha=0.2)

# 校准曲线
prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=8)
ax2.plot(prob_pred, prob_true, marker='s', markersize=4, lw=2, label='XGBoost', color='#f3722c')
ax2.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.5, label='Perfectly Calibrated')
ax2.set_xlabel('Predicted Probability')
ax2.set_ylabel('Actual Proportion')
ax2.set_title('Calibration Plot (Reliability)')
ax2.legend(loc='upper left')
ax2.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "v8_validation_metrics.png"), dpi=300)
plt.show()

# ===============================
# 8. SHAP 解释性分析
# ===============================
print("--- Generating SHAP Analysis ---")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_final)

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_final, show=False, max_display=15)
plt.title(f"SHAP Impact Summary\n(eICU Cohort, n={len(df_eicu)})", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "v8_shap_summary.png"), dpi=300)
plt.show()

print(f"✅ External validation complete. Results saved in: {SAVE_DIR}")
