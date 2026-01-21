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
    # 4. 泄露防护与特征准备
    # =========================================================
    drop_from_modeling = [
        'subject_id', 'hadm_id', 'stay_id', 'database', 
        'admittime', 'dischtime', 'intime', 'deathtime', 'dod',
        'early_death_24_48h', 'hosp_mortality'
    ]
    df_model = df.drop(columns=[c for c in drop_from_modeling if c in df.columns])
    
    # 确定需要预处理的数值列 (排除标签和二分类列)
    binary_cols = outcome_cols + categorical
    numeric_features = [c for c in df_model.select_dtypes(include=[np.number]).columns 
                        if c not in binary_cols]

    # =========================================================
    # 5. 🧪 核心增强：动态 Log1p 转换 (处理偏态)
    # =========================================================
    skewed_cols = ['creatinine_max', 'creatinine_min', 'bun_max', 'bun_min',
                   'wbc_max', 'wbc_min', 'glucose_max', 'glucose_min',
                   'lactate_max', 'alt_max', 'ast_max', 'bilirubin_total_max']
    existing_skewed = [c for c in skewed_cols if c in numeric_features]
    
    print(f"\n🔄 执行 Log1p 转换 (处理 {len(existing_skewed)} 个偏态指标)...")
    for col in existing_skewed:
        df_model[col] = np.log1p(df_model[col].clip(lower=0))
    
    # 保存偏态列清单，供 eICU 脚本复用
    joblib.dump(existing_skewed, SKEW_CONFIG_PATH)

    # =========================================================
    # 6. 🧪 核心增强：MICE 多重插补
    # =========================================================
    print("🧪 启动 MICE 多重插补 (链式方程)...")
    # 使用中位数作为初始策略，更具鲁棒性
    imputer = IterativeImputer(max_iter=10, random_state=42, initial_strategy='median')
    df_model[numeric_features] = imputer.fit_transform(df_model[numeric_features])
    
    # 保存 Imputer 资产
    joblib.dump(imputer, IMPUTER_PATH)

    # =========================================================
    # 7. ⚖️ Z-score 标准化
    # =========================================================
    print("⚖️ 执行 Z-score 标准化并保存 Scaler...")
    scaler = StandardScaler()
    df_model[numeric_features] = scaler.fit_transform(df_model[numeric_features])
    
    # 保存 Scaler 资产
    joblib.dump(scaler, SCALER_PATH)

    # =========================================================
    # 8. 持久化建模张量
    # =========================================================
    processed_path = os.path.join(SAVE_DIR, "mimic_processed.csv")
    df_model.to_csv(processed_path, index=False)
    
    print("-" * 70)
    print(f"✅ 模块 03 处理完成！")
    print(f"  - 建模张量: {processed_path}")
    print(f"  - 资产 1 (Scaler): {SCALER_PATH}")
    print(f"  - 资产 2 (Imputer): {IMPUTER_PATH}")
    print(f"  - 资产 3 (Skew Config): {SKEW_CONFIG_PATH}")
    print("-" * 70)

if __name__ == "__main__":
    run_mimic_standardization()
