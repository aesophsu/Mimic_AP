import os
import numpy as np
import pandas as pd

# =========================================================
# 1. 配置与路径
# =========================================================
BASE_DIR = ".."
INPUT_PATH = os.path.join(BASE_DIR, "data/ap_final_analysis_cohort.csv")
SAVE_DIR = os.path.join(BASE_DIR, "data/cleaned")

# 缺失率门槛：超过 30% 则剔除（除非在白名单中）
MISSING_THRESHOLD = 0.3 

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def run_module_01():
    print("="*60)
    print("🚀 运行模块 01: 状态标记、28天死亡定义与特征清洗")
    print("="*60)
    
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 错误: 找不到输入文件 {INPUT_PATH}")
        return
    
    df_raw = pd.read_csv(INPUT_PATH)
    df = df_raw.copy()
    print(f"📊 原始数据读取成功: {df.shape[0]} 行, {df.shape[1]} 列")

    # =========================================================
    # 2. 精准定义 28天死亡率 (Primary Outcome)
    # =========================================================
    # 转换时间格式
    df['intime'] = pd.to_datetime(df['intime'])
    df['deathtime'] = pd.to_datetime(df['deathtime'])
    df['dod'] = pd.to_datetime(df['dod'])

    # 标记住院死亡与总死亡 (基础标记)
    df['hosp_mortality'] = df['deathtime'].notnull().astype(int)
    df['overall_mortality'] = df['dod'].notnull().astype(int)

    # 计算入 ICU 到死亡的天数
    # 取 deathtime 和 dod 中较早的一个作为死亡发生时间
    death_timestamp = df[['deathtime', 'dod']].min(axis=1)
    days_to_death = (death_timestamp - df['intime']).dt.total_seconds() / (24 * 3600)

    # 核心指标：28天死亡率 (1=28天内死亡, 0=存活或28天后死亡)
    df['mortality_28d'] = ((days_to_death >= 0) & (days_to_death <= 28)).astype(int)
    
    print(f"✅ 结局标记完成:")
    print(f"   - 28天内死亡: {df['mortality_28d'].sum()} 例")
    print(f"   - 住院期间死亡: {df['hosp_mortality'].sum()} 例")
    print(f"   - 随访总死亡: {df['overall_mortality'].sum()} 例")

    # =========================================================
    # 3. 核心保护白名单 (White List)
    # =========================================================
    # 即使缺失率高，也必须保留的特征
    outcome_labels = ['mortality_28d', 'hosp_mortality', 'overall_mortality', 
                      'pof', 'renal_pof', 'resp_pof', 'cv_pof']
    
    clinical_soul_cols = [
        'lactate_max', 'pao2fio2ratio_min',
        'lipase_max', 'lab_amylase_max', 'creatinine_max'
    ]
    
    id_time_cols = ['subject_id', 'hadm_id', 'stay_id', 'intime', 'admittime']
    
    white_list = outcome_labels + clinical_soul_cols + id_time_cols

    # =========================================================
    # 4. 缺失率过滤 (Feature Filtering)
    # =========================================================
    missing_pct = df.isnull().mean()
    high_missing_cols = missing_pct[missing_pct > MISSING_THRESHOLD].index.tolist()
    
    # 确定剔除名单：在高缺失名单中，且不在白名单内
    # 注意：原始的 deathtime 和 dod 会在这里被剔除，因为我们已提取了 mortality_28d
    cols_to_drop = [c for c in high_missing_cols if c not in white_list]
    df_filtered = df.drop(columns=cols_to_drop)

    # =========================================================
    # 5. 单位校准与盖帽处理 (Table 1 Ready)
    # =========================================================
    # Fibrinogen 单位校准 (mg/dL)
    if 'fibrinogen_max' in df_filtered.columns:
        median_fib = df_filtered['fibrinogen_max'].median()
        if not pd.isna(median_fib) and median_fib < 50:
            df_filtered['fibrinogen_max'] = df_filtered['fibrinogen_max'] * 100

    df_table1 = df_filtered.copy()
    numeric_cols = df_table1.select_dtypes(include=[np.number]).columns
    
    # 排除不需要截断的列 (ID, 结局, 二元变量)
    skip_clip = white_list + ['gender_num', 'alcoholic_ap', 'biliary_ap', 
                             'hyperlipidemic_ap', 'drug_induced_ap', 'vaso_flag', 'mechanical_vent_flag']
    
    for col in numeric_cols:
        if col in df_table1.columns and col not in skip_clip:
            df_table1[col] = df_table1[col].clip(df_table1[col].quantile(0.01), 
                                                 df_table1[col].quantile(0.99))

    # =========================================================
    # 6. 特征自检报告与清单显示
    # =========================================================
    print("-" * 60)
    print("📋 特征自检报告")
    print("-" * 60)
    print(f"🔹 原始总列数: {len(df_raw.columns)}")
    print(f"🔹 剔除列数: {len(cols_to_drop)}")
    print(f"🔹 最终保留列数: {len(df_table1.columns)}")
    
    if cols_to_drop:
        print(f"🗑️ 已剔除特征 (缺失率 > {MISSING_THRESHOLD*100}%):")
        for c in sorted(cols_to_drop):
            print(f"   - {c:<25} (缺失率: {df_raw[c].isnull().mean():.2%})")

    # 打印最终清单
    final_cols = sorted(df_table1.columns.tolist())
    print("-" * 60)
    print("💎 最终保留特征清单：")
    for i in range(0, len(final_cols), 3):
        row = final_cols[i:i+3]
        print("".join([f"{col:<30}" for col in row]))

    # 保存文件
    table1_path = os.path.join(SAVE_DIR, "mimic_for_table1.csv")
    df_table1.to_csv(table1_path, index=False)
    
    print("-" * 60)
    print(f"✅ 模块 01 完成! 干净数据已存至: {table1_path}")
    print("="*60)

if __name__ == "__main__":
    run_module_01()
