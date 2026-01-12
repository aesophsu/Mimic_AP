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
    print("🚀 运行模块 01: 状态标记、单位校准与特征清洗")
    print("="*60)
    
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 错误: 找不到输入文件 {INPUT_PATH}")
        return
    
    df_raw = pd.read_csv(INPUT_PATH)
    df = df_raw.copy()
    print(f"📊 原始数据读取成功: {df.shape[0]} 行, {df.shape[1]} 列")

    # =========================================================
    # 2. 精准定义结局指标
    # =========================================================
    df['intime'] = pd.to_datetime(df['intime'])
    df['deathtime'] = pd.to_datetime(df['deathtime'])
    df['dod'] = pd.to_datetime(df['dod'])

    # 计算 28天死亡率
    death_timestamp = df[['deathtime', 'dod']].min(axis=1)
    days_to_death = (death_timestamp - df['intime']).dt.total_seconds() / (24 * 3600)
    df['mortality_28d'] = ((days_to_death >= 0) & (days_to_death <= 28)).astype(int)
    
    # 标记其他结局用于分析
    df['hosp_mortality'] = df['deathtime'].notnull().astype(int)
    df['overall_mortality'] = df['dod'].notnull().astype(int)

    # =========================================================
    # 3. 核心保护白名单 (防止关键特征被过滤)
    # =========================================================
    outcome_labels = ['mortality_28d', 'hosp_mortality', 'overall_mortality', 
                      'pof', 'renal_pof', 'resp_pof', 'cv_pof']
    clinical_soul_cols = ['lactate_max', 'pao2fio2ratio_min', 'lipase_max', 
                          'lab_amylase_max', 'creatinine_max', 'ast_max', 'alt_max', 'bun_min']
    id_time_cols = ['subject_id', 'hadm_id', 'stay_id', 'intime', 'admittime']
    white_list = outcome_labels + clinical_soul_cols + id_time_cols

    # =========================================================
    # 4. 缺失率过滤
    # =========================================================
    missing_pct = df.isnull().mean()
    cols_to_drop = [c for c in missing_pct[missing_pct > MISSING_THRESHOLD].index 
                    if c not in white_list]
    df_filtered = df.drop(columns=cols_to_drop)

    # =========================================================
    # 5. ⚠️ 核心修正：物理单位校准 (与 eICU 对齐)
    # =========================================================
    print("\n🩺 正在执行物理单位对齐审计...")
    
    # 校准函数：基于中位数判断当前量级
    def harmonize_mimic_units(data):
        # 1. AST/ALT 校准: 如果中位数 < 10，说明极有可能是 Log 尺度或严重偏离原始 U/L 单位
        for col in ['ast_max', 'alt_max']:
            if col in data.columns:
                med = data[col].median()
                if not pd.isna(med) and med < 10:
                    print(f"  - 发现 {col} 量级偏低 ({med:.2f}), 执行反 Log 还原或单位校准...")
                    # 如果数据已经是 Log1p 后的，尝试还原: e^x - 1
                    data[col] = np.expm1(data[col]) 
        
        # 2. BUN 校准: 确保单位为 mg/dL (eICU 标准)
        if 'bun_min' in data.columns:
            med = data['bun_min'].median()
            if not pd.isna(med) and med < 5:
                print(f"  - 发现 bun_min 量级偏低 ({med:.2f}), 尝试 mmol/L -> mg/dL 转换...")
                data['bun_min'] = data['bun_min'] * 2.8
        
        # 3. Fibrinogen 校准
        if 'fibrinogen_max' in data.columns:
            med = data['fibrinogen_max'].median()
            if not pd.isna(med) and med < 50:
                print(f"  - 发现 fibrinogen_max 量级偏低 ({med:.2f}), 转换为 mg/dL...")
                data['fibrinogen_max'] = data['fibrinogen_max'] * 100
        return data

    df_filtered = harmonize_mimic_units(df_filtered)

    # =========================================================
    # 6. 盖帽处理 (Clipping 1%-99%)
    # =========================================================
    df_table1 = df_filtered.copy()
    numeric_cols = df_table1.select_dtypes(include=[np.number]).columns
    skip_clip = white_list + ['gender_num', 'alcoholic_ap', 'biliary_ap', 
                              'hyperlipidemic_ap', 'drug_induced_ap', 'vaso_flag']
    
    for col in numeric_cols:
        if col in df_table1.columns and col not in skip_clip:
            lower = df_table1[col].quantile(0.01)
            upper = df_table1[col].quantile(0.99)
            df_table1[col] = df_table1[col].clip(lower, upper)

    # =========================================================
    # 7. 特征自检报告与保存
    # =========================================================
    print("-" * 60)
    print(f"🔹 最终保留列数: {len(df_table1.columns)}")
    print(f"🔹 关键指标审计 (Median):")
    for c in ['ast_max', 'bun_min', 'creatinine_max']:
        if c in df_table1.columns:
            print(f"  - {c:<15}: {df_table1[c].median():.2f}")

    table1_path = os.path.join(SAVE_DIR, "mimic_for_table1.csv")
    df_table1.to_csv(table1_path, index=False)
    
    print("-" * 60)
    print(f"✅ 模块 01 完成! 干净原始尺度数据已存至: {table1_path}")

if __name__ == "__main__":
    run_module_01()
