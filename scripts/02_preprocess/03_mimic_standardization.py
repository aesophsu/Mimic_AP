import os
import joblib
import numpy as np
import pandas as pd
from tableone import TableOne
from sklearn.preprocessing import StandardScaler

# =========================================================
# 1. 配置与路径映射 (基于新目录树)
# =========================================================
BASE_DIR = "../../"
# 输入是 02 步清洗后的物理值数据
INPUT_PATH = os.path.join(BASE_DIR, "data/cleaned/mimic_raw_scale.csv")
SAVE_DIR = os.path.join(BASE_DIR, "data/cleaned")
SCALER_PATH = os.path.join(BASE_DIR, "artifacts/scalers/mimic_scaler.joblib")
REPORT_DIR = os.path.join(BASE_DIR, "results/tables")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

def run_mimic_standardization():
    print("="*70)
    print("🚀 启动模块 03: 亚组划分、Table 1 审计与特征标准化")
    print("="*70)
    
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 错误: 找不到输入文件 {INPUT_PATH}")
        return
    
    df = pd.read_csv(INPUT_PATH)

    # =========================================================
    # 2. 亚组划分 (Subgroup Definition) 
    # =========================================================
    # 临床定义：入院 24h 内肌酐 < 1.5 mg/dL 且无 CKD 史
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
    
    # 过滤掉不存在的列
    cols_for_table = [c for c in (clinical_features + outcome_cols) if c in df.columns]
    categorical = [c for c in ['heart_failure', 'chronic_kidney_disease', 'malignant_tumor', 
                               'mortality_28d', 'composite_outcome', 'subgroup_no_renal'] if c in cols_for_table]
    nonnormal = [c for c in cols_for_table if c not in categorical]

    print("\n📊 正在生成统计报告...")
    t1 = TableOne(df, columns=cols_for_table, categorical=categorical, 
                  nonnormal=nonnormal, groupby='pof', pval=True)
    t1.to_csv(os.path.join(REPORT_DIR, "table1_baseline.csv"))
    print(f"✅ Table 1 已存至: {REPORT_DIR}/table1_baseline.csv")

    if 'subgroup_no_renal' in df.columns:
        print("🔍 正在生成 Table 2: 肾功能亚组对比...")
        t2 = TableOne(df, columns=cols_for_table, categorical=categorical, 
                      nonnormal=nonnormal, groupby='subgroup_no_renal', pval=True)
        t2.to_csv(os.path.join(REPORT_DIR, "table2_renal_subgroup.csv"))
        print(f"✅ Table 2 已存至: {REPORT_DIR}/table2_renal_subgroup.csv")

    # =========================================================
    # 4. 泄露防护：剔除无关 ID 与非预测变量
    # =========================================================
    # 结局标签必须保留，但在标准化时要排除
    drop_from_modeling = [
        'subject_id', 'hadm_id', 'stay_id', 'database', 
        'admittime', 'dischtime', 'intime', 'deathtime', 'dod',
        'early_death_24_48h', 'hosp_mortality'
    ]
    df_model = df.drop(columns=[c for c in drop_from_modeling if c in df.columns])

    # =========================================================
    # 5. 特征标准化 (Standardization)
    # =========================================================
    print("\n⚖️ 执行 Z-score 标准化...")
    
    # 仅对数值型连续变量标准化，排除二进制标签和亚组标记
    binary_cols = outcome_cols + categorical
    numeric_features = [c for c in df_model.select_dtypes(include=[np.number]).columns 
                        if c not in binary_cols]

    scaler = StandardScaler()
    df_model[numeric_features] = scaler.fit_transform(df_model[numeric_features])
    
    # 保存 Scaler 资产，用于后续 eICU 验证集对齐
    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ Scaler 资产已序列化至: {SCALER_PATH}")

    # =========================================================
    # 6. 持久化输出
    # =========================================================
    processed_path = os.path.join(SAVE_DIR, "mimic_processed.csv")
    df_model.to_csv(processed_path, index=False)
    
    print("-" * 70)
    print(f"📊 模块 03 处理完成:")
    print(f"  - 最终张量维度: {df_model.shape}")
    print(f"  - 包含标签: {outcome_cols}")
    print(f"  - 预处理数据存至: {processed_path}")
    print("-" * 70)

if __name__ == "__main__":
    run_mimic_standardization()
