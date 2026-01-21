import os
import joblib
import numpy as np
import pandas as pd
from tableone import TableOne
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer  # 必须导入以启用 MICE
from sklearn.impute import IterativeImputer

# =========================================================
# 1. 配置与路径映射 (基于 v3.0 目录树)
# =========================================================
BASE_DIR = "../../"
INPUT_PATH = os.path.join(BASE_DIR, "data/cleaned/mimic_raw_scale.csv")
SAVE_DIR = os.path.join(BASE_DIR, "data/cleaned")

# 资产持久化路径
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts/scalers")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "mimic_scaler.joblib")
IMPUTER_PATH = os.path.join(ARTIFACT_DIR, "mimic_mice_imputer.joblib")
SKEW_CONFIG_PATH = os.path.join(ARTIFACT_DIR, "skewed_cols_config.pkl")

REPORT_DIR = os.path.join(BASE_DIR, "results/tables")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

def run_mimic_standardization():
    print("="*70)
    print("🚀 启动模块 03: 亚组划分、Log 转换、MICE 插补与标准化")
    print("="*70)
    
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 错误: 找不到输入文件 {INPUT_PATH}")
        return
    
    df = pd.read_csv(INPUT_PATH)

    # =========================================================
    # 2. 亚组划分 (Subgroup Definition) 保持不变
    # =========================================================
    if 'creatinine_max' in df.columns and 'chronic_kidney_disease' in df.columns:
        df['subgroup_no_renal'] = (
            (df['creatinine_max'] < 1.5) & (df['chronic_kidney_disease'] == 0)
        ).astype(int)
        print(f"✅ 亚组标记完成: '无预存肾损伤' n = {df['subgroup_no_renal'].sum()}")
        
    # LASSO 不接受字符串，必须在此处转换
    if 'gender' in df.columns and df['gender'].dtype == 'object':
        df['gender'] = df['gender'].map({'M': 1, 'F': 0})
        print("✅ 字段 'gender' 已完成数值化映射 (M->1, F->0)")

    # =========================================================
    # 3. 📊 自动化统计分析 (Table 1 & 2) - 基于物理尺度
    # =========================================================
    clinical_features = [
        'admission_age', 'bmi', 'heart_failure', 'chronic_kidney_disease', 
        'malignant_tumor', 'bun_min', 'creatinine_max', 'lactate_max', 
        'pao2fio2ratio_min', 'wbc_max', 'alt_max', 'ast_max'
    ]
    outcome_cols = ['pof', 'mortality_28d', 'composite_outcome']
    cols_for_table = [c for c in (clinical_features + outcome_cols) if c in df.columns]
    categorical = [c for c in ['heart_failure', 'chronic_kidney_disease', 'malignant_tumor', 
                               'mortality_28d', 'composite_outcome', 'subgroup_no_renal'] if c in cols_for_table]
    nonnormal = [c for c in cols_for_table if c not in categorical]

    print("\n📊 正在生成统计报告 (物理尺度)...")
    t1 = TableOne(df, columns=cols_for_table, categorical=categorical, nonnormal=nonnormal, groupby='pof', pval=True)
    t1.to_csv(os.path.join(REPORT_DIR, "table1_baseline.csv"))
    
    if 'subgroup_no_renal' in df.columns:
        t2 = TableOne(df, columns=cols_for_table, categorical=categorical, nonnormal=nonnormal, groupby='subgroup_no_renal', pval=True)
        t2.to_csv(os.path.join(REPORT_DIR, "table2_renal_subgroup.csv"))
    print(f"✅ Table 1 & 2 已存至: {REPORT_DIR}")

    # =========================================================
    # 4. 泄露防护与特征准备 (修正版)
    # =========================================================
    # A. 定义不参与建模的 ID 与时间列
    drop_from_modeling = [
        'subject_id', 'hadm_id', 'stay_id', 'database', 
        'admittime', 'dischtime', 'intime', 'deathtime', 'dod',
        'early_death_24_48h', 'hosp_mortality'
    ]
    
    # B. 定义必须保持原始格式的列 (标签、子结局、亚组标记)
    protected_cols = [
        'pof', 'resp_pof', 'cv_pof', 'renal_pof', 
        'mortality_28d', 'composite_outcome', 'subgroup_no_renal',
        'gender', 'heart_failure', 'chronic_kidney_disease', 
        'malignant_tumor', 'mechanical_vent_flag', 'vaso_flag'
    ]
    
    df_model = df.drop(columns=[c for c in drop_from_modeling if c in df.columns])

    # 强制将保护列转换为整数 (防止标准化污染)
    for col in protected_cols:
        if col in df_model.columns:
            df_model[col] = df_model[col].fillna(0).astype(int)

    # 强制剔除非数值列 (如 Race 等文本)
    remaining_text = df_model.select_dtypes(include=['object']).columns.tolist()
    if remaining_text:
        print(f"⚠️ 警告: 强制剔除非数值列以防报错: {remaining_text}")
        df_model = df_model.drop(columns=remaining_text)
        
    # C. 确定真正需要“数值处理”的特征 (排除保护列)
    numeric_features = [c for c in df_model.select_dtypes(include=[np.number]).columns 
                        if c not in protected_cols]
    
    print(f"✅ 特征分类完成: 数值特征 {len(numeric_features)} 个, 保护列 {len(protected_cols)} 个")

    # =========================================================
    # 5. 🧪 核心增强：动态 Log1p 转换 (处理偏态)
    # =========================================================
    skewed_cols = ['creatinine_max', 'creatinine_min', 'bun_max', 'bun_min',
                   'wbc_max', 'wbc_min', 'glucose_max', 'glucose_min',
                   'lactate_max', 'alt_max', 'ast_max', 'bilirubin_total_max']
    existing_skewed = [c for c in skewed_cols if c in numeric_features]
    
    print(f"🔄 执行 Log1p 转换 (处理 {len(existing_skewed)} 个偏态指标)...")
    for col in existing_skewed:
        df_model[col] = np.log1p(df_model[col].clip(lower=0))
    
    joblib.dump(existing_skewed, SKEW_CONFIG_PATH)

    # =========================================================
    # 6. 🧪 核心增强：MICE 多重插补 (仅针对数值特征)
    # =========================================================
    print("🧪 启动 MICE 多重插补 (仅处理 numeric_features)...")
    imputer = IterativeImputer(max_iter=10, random_state=42, initial_strategy='median')
    
    # 注意：只对数值列进行 fit 和 transform
    df_model[numeric_features] = imputer.fit_transform(df_model[numeric_features])
    joblib.dump(imputer, IMPUTER_PATH)

    # =========================================================
    # 7. ⚖️ Z-score 标准化 (仅针对数值特征)
    # =========================================================
    print("⚖️ 执行 Z-score 标准化 (排除标签和亚组列)...")
    scaler = StandardScaler()
    
    # 关键点：只标准化数值特征，protected_cols 保持原样
    df_model[numeric_features] = scaler.fit_transform(df_model[numeric_features])
    joblib.dump(scaler, SCALER_PATH)

    # =========================================================
    # 8. 检查并保存
    # =========================================================
    if 'subgroup_no_renal' in df_model.columns:
        unique_vals = df_model['subgroup_no_renal'].unique()
        print(f"🚩 最终亚组列状态检查: {unique_vals} (应为 [1 0] 或 [0 1])")
    
    print(f"\n📋 原始数据探测: {df.shape[0]} 行, {df.shape[1]} 列")
    print(f"{'Feature Name':<25} | {'Missing%':<10} | {'Median':<10} | {'Mean':<10} | {'Max':<10}")
    print("-" * 75)
        
    processed_path = os.path.join(SAVE_DIR, "mimic_processed.csv")
    df_model.to_csv(processed_path, index=False)   
    
    print("-" * 70)
    print(f"✅ 模块 03 处理完成！")
    print(f"   - 建模张量: {processed_path}")
    print(f"   - 标准化后的特征数: {len(numeric_features)}")
    print("-" * 70)

if __name__ == "__main__":
    run_mimic_standardization()
