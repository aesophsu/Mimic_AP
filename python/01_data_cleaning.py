import os
import numpy as np
import pandas as pd

# =========================================================
# 1. 配置与路径
# =========================================================
BASE_DIR = ".."
INPUT_PATH = os.path.join(BASE_DIR, "data/ap_final_analysis_cohort.csv")
SAVE_DIR = os.path.join(BASE_DIR, "data/cleaned")
MISSING_THRESHOLD = 0.3  # 30% 缺失率剔除门槛

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def run_module_01():
    print("="*70)
    print("🚀 启动模块 01: 跨库尺度对齐与特征清洗")
    print("="*70)
    
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 错误: 找不到输入文件 {INPUT_PATH}")
        return
    
    df = pd.read_csv(INPUT_PATH)
    
    # =========================================================
    # 2. 特征探测与全清单统计 (Table 1 预审)
    # =========================================================
    print(f"\n📋 原始数据探测: {df.shape[0]} 行, {df.shape[1]} 列")
    print(f"{'Feature Name':<25} | {'Missing%':<10} | {'Median':<10} | {'Mean':<10} | {'Max':<10}")
    print("-" * 75)
    
    initial_stats = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            missing = df[col].isnull().mean() * 100
            med = df[col].median()
            mean = df[col].mean()
            v_max = df[col].max()
            print(f"{col:<25} | {missing:>8.2f}% | {med:>10.2f} | {mean:>10.2f} | {v_max:>10.2f}")
            initial_stats.append(col)
    if 'early_death_24_48h' in df.columns:
        early_death_mask = (df['early_death_24_48h'] == 1)
        correction_count = ((df['early_death_24_48h'] == 1) & (df['pof'] == 0)).sum()
        
        df.loc[early_death_mask, 'pof'] = 1
        df.loc[early_death_mask, 'composite_outcome'] = 1
        
        print(f"\n🩺 标签审计：修正了 {correction_count} 例早亡导致的 POF 假阴性。")
    # =========================================================
    # 3. 核心保护白名单 (强制保留关键变量)
    # =========================================================
    # 结局指标 + 临床灵魂字段 + ID
    outcome_labels = ['pof', 'mortality_28d', 'composite_outcome', 'early_death_24_48h', 
                      'resp_pof', 'cv_pof', 'renal_pof']
    clinical_soul_cols = ['lactate_max', 'pao2fio2ratio_min', 'lipase_max', 
                          'creatinine_max', 'ast_max', 'alt_max', 'bun_min', 'bmi']
    white_list = outcome_labels + clinical_soul_cols + ['subject_id', 'hadm_id', 'stay_id']

    # =========================================================
    # 4. 缺失率过滤 (30% 门槛)
    # =========================================================
    missing_pct = df.isnull().mean()
    cols_to_drop = [c for c in missing_pct[missing_pct > MISSING_THRESHOLD].index if c not in white_list]
    df = df.drop(columns=cols_to_drop)
    print(f"\n🗑️ 基于缺失率 (>30%) 剔除 {len(cols_to_drop)} 个非核心特征。")

    # =========================================================
    # 5. 🩺 物理尺度对齐 (Automated Unit Auditing)
    # =========================================================
    print("\n🩺 正在执行跨库物理单位审计 (MIMIC ➡️ eICU)...")
    
    # A. BUN 转换 (依据 2.801 系数)
    # A. BUN 转换 (依据 2.801 系数，database-aware)
    if 'bun_min' in df.columns and 'database' in df.columns:
        bun_med_by_db = df.groupby('database')['bun_min'].median()

    # 审计输出（一次性）
        print(f"[Audit][BUN] median by database:\n{bun_med_by_db}")

        if bun_med_by_db.get('eicu', 10) < 5:
            print("  - [BUN 校准]: eICU 检测到 mmol/L 量级，应用 2.801 转换")
            for c in ['bun_min', 'bun_max']:
                if c in df.columns:
                    df.loc[df['database'] == 'eicu', c] *= 2.801

    # B. AST/ALT 校准 (检测是否已被 Log 转换)
    for col in ['ast_max', 'alt_max']:
        if col in df.columns:
            med = df[col].median()
            maxv = df[col].max()
            print(f"[Audit][{col}] median={med:.2f}, max={maxv:.2f}")
            if med < 10 and df[col].max() < 50: # 如果中位数极低，执行反 Log 还原
                print(f"  - [{col} 校准]: 检测到量级异常低 ({med:.2f}), 执行反 Log (expm1) 还原...")
                df[col] = np.expm1(df[col])

    # C. Fibrinogen 校准 (g/L -> mg/dL)
    if 'fibrinogen_max' in df.columns:
        med = df['fibrinogen_max'].median()
        if med < 10: 
            print(f"  - [Fibrinogen 校准]: 检测到 g/L 量级 ({med:.2f}), 转换为 mg/dL...")
            df['fibrinogen_max'] = df['fibrinogen_max'] * 100

    # =========================================================
    # 6. 盖帽处理 (Clipping 1%-99%)
    # =========================================================
    print("\n✂️ 执行 1%-99% 盖帽处理以消除离群值...")
    skip_clip = white_list + ['gender_num', 'vaso_flag', 'mechanical_vent_flag']
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col not in skip_clip:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower, upper)

    # =========================================================
    # 7. 最终缺失率与审计统计报告
    # =========================================================
    print("\n" + "-"*70)
    print(f"📊 模块 01 清洗完成总结:")
    print(f"  - 最终样本量: {df.shape[0]}")
    print(f"  - 最终特征数: {df.shape[1]}")
    print(f"  - 关键指标对齐审计 (Median):")
    for c in ['ast_max', 'bun_min', 'creatinine_max', 'bmi']:
        if c in df.columns:
            print(f"    > {c:<18}: {df[c].median():.2f}")
    
    # 缺失率警告
    print("\n🔍 核心白名单字段缺失情况:")
    for c in clinical_soul_cols:
        if c in df.columns:
            m_rate = df[c].isnull().mean() * 100
            print(f"    > {c:<18}: {m_rate:>6.2f}% {'❗' if m_rate > 30 else ''}")

    # 保存结果
    save_path = os.path.join(SAVE_DIR, "mimic_for_table1.csv")
    df.to_csv(save_path, index=False)
    print(f"\n✅ 干净数据已存至: {save_path}")

if __name__ == "__main__":
    run_module_01()
