import os
import numpy as np
import pandas as pd

# =========================================================
# 1. 配置与路径
# =========================================================
BASE_DIR = ".."
INPUT_PATH = os.path.join(BASE_DIR, "data/ap_final_analysis_cohort.csv")
SAVE_DIR = os.path.join(BASE_DIR, "data/cleaned")
MISSING_THRESHOLD = 0.3 

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def run_module_01():
    print("🚀 开始运行模块 01 (增强版): 状态标记与数据清洗...")
    
    df = pd.read_csv(INPUT_PATH)
    
    # =========================================================
    # 2. 状态标记 (New: Convert Timestamps to Binary Labels)
    # =========================================================
    # A. 住院死亡标记 (Hospital Mortality)
    # 如果 deathtime 非空，则视为住院期间死亡
    df['hosp_mortality'] = df['deathtime'].notnull().astype(int)
    
    # B. 院外/总死亡标记 (Overall Mortality)
    # 如果 dod (Date of Death) 非空，则代表该患者已死亡（涵盖住院和出院后）
    df['overall_mortality'] = df['dod'].notnull().astype(int)
    
    print(f"✅ 已生成死亡状态标记: 住院死亡 {df['hosp_mortality'].sum()} 例, 总死亡 {df['overall_mortality'].sum()} 例")

    # =========================================================
    # 3. 核心保护白名单 (修正后)
    # =========================================================
    # 我们保留状态标签，而允许删除极高缺失的原始时间列（因为已经提取了有用信息）
    outcome_labels = ['hosp_mortality', 'overall_mortality', 'pof', 'renal_pof', 'resp_pof', 'cv_pof']
    
    clinical_soul_cols = [
        'lactate_max', 'pao2fio2ratio_min', 'crp_max', 
        'lipase_max', 'lab_amylase_max', 'fibrinogen_max', 'creatinine_max'
    ]
    
    id_cols = ['subject_id', 'hadm_id', 'stay_id']
    
    white_list = outcome_labels + clinical_soul_cols + id_cols

    # =========================================================
    # 4. 缺失率过滤
    # =========================================================
    missing_pct = df.isnull().mean()
    high_missing_cols = missing_pct[missing_pct > MISSING_THRESHOLD].index.tolist()
    
    # 此时可以放心让 deathtime 和 dod 被剔除，因为我们已经有了 hosp_mortality 和 overall_mortality
    cols_to_drop = [c for c in high_missing_cols if c not in white_list]
    
    print(f"🗑️ 剔除特征 (含原始时间戳): {cols_to_drop}")
    df_filtered = df.drop(columns=cols_to_drop)

    # =========================================================
    # 5. 单位校准与盖帽 (同前)
    # =========================================================
    # Fibrinogen 校准
    if 'fibrinogen_max' in df_filtered.columns:
        median_fib = df_filtered['fibrinogen_max'].median()
        if not pd.isna(median_fib) and median_fib < 50:
            df_filtered['fibrinogen_max'] = df_filtered['fibrinogen_max'] * 100

    # 盖帽处理 (针对 Table 1)
    df_table1 = df_filtered.copy()
    numeric_cols = df_table1.select_dtypes(include=[np.number]).columns
    # 排除分类变量和结局标签
    skip_clip = white_list + ['gender_num', 'alcoholic_ap', 'biliary_ap', 'hyperlipidemic_ap', 'drug_induced_ap']
    
    for col in numeric_cols:
        if col in df_table1.columns and col not in skip_clip:
            df_table1[col] = df_table1[col].clip(df_table1[col].quantile(0.01), df_table1[col].quantile(0.99))
    
    # 保存结果
    table1_path = os.path.join(SAVE_DIR, "mimic_for_table1.csv")
    df_table1.to_csv(table1_path, index=False)
    
    print(f"✅ 模块 01 (增强版) 完成! 最终保存特征数: {df_table1.shape[1]}")

if __name__ == "__main__":
    run_module_01()
