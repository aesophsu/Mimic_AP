import os
import numpy as np
import pandas as pd
from tableone import TableOne

# =========================================================
# 1. 配置与路径
# =========================================================
BASE_DIR = ".."
INPUT_PATH = os.path.join(BASE_DIR, "data/cleaned/mimic_for_table1.csv")
SAVE_DIR = os.path.join(BASE_DIR, "data/cleaned")

def run_module_02():
    print("="*60)
    print("🚀 运行优化模块 02: 深度特征过滤与亚组划分 (保持原始尺度)")
    print("="*60)
    
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 错误: 找不到输入文件 {INPUT_PATH}")
        return
    
    # 加载模块 01 处理后的干净原始数据
    df = pd.read_csv(INPUT_PATH)

    # =========================================================
    # 2. 泄露防护：剔除治疗干扰、评分变量及冗余 ID
    # =========================================================
    print(f"\n📋 原始数据探测: {df.shape[0]} 行, {df.shape[1]} 列")
    print(f"{'Feature Name':<25} | {'Missing%':<10} | {'Median':<10} | {'Mean':<10} | {'Max':<10}")
    print("-" * 75)
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            series = df[col].dropna() # 排除空值进行计算
            missing = df[col].isnull().mean() * 100
            med = series.median() if not series.empty else 0
            mean = series.mean() if not series.empty else 0
            v_max = series.max() if not series.empty else 0
            print(f"{col:<25} | {missing:>8.2f}% | {med:>10.2f} | {mean:>10.2f} | {v_max:>10.2f}")

    # 逻辑：排除临床评分、入选后治疗及可能干扰预测的冗余结局指标
    # 核心：保留三个研究终点用于不同建模任务
    all_labels = ['pof', 'mortality_28d', 'composite_outcome']
    
    must_drop = [
        # 1. 评分系统 (Data Leakage 重灾区)
        'sofa_score', 'sapsii', 'apsiii', 'oasis', 'lods',
        # 2. 时间与 ID (非生物学特征)
        'admittime', 'dischtime', 'intime', 'subject_id', 'hadm_id', 'stay_id',
        # 3. 器官支持措施 (属于“治疗”而非“入组基线”)
        'mechanical_vent_flag', 'vaso_flag',
        # 4. 其他中间或冲突变量
        'los', 'deathtime', 'dod', 'early_death_24_48h', 
        'hosp_mortality', 'overall_mortality'
    ]
    
    # 仅剔除存在的列
    cols_to_drop = [c for c in must_drop if c in df.columns]
    df_clean = df.drop(columns=cols_to_drop)
    # =========================================================
    # 2.1 🔍 缺失率深度审计 (为模块 03 插补策略做准备)
    # =========================================================
    print("\n🔍 缺失率 Top 10 特征审计:")
    missing_pct = df_clean.isnull().mean() * 100
    print(missing_pct.sort_values(ascending=False).head(10).map("{:.2f}%".format))
    # 如果某些核心变量缺失率 > 50%，这里会给你一个直观的警告
    high_missing = missing_pct[missing_pct > 50].index.tolist()
    if high_missing:
        print(f"⚠️ 警告: 以下特征缺失率超过 50%: {high_missing}")
    
    # 🛡️ 强制审计：确保三个终点指标安全保留
    for label in all_labels:
        assert label in df_clean.columns, f"❌ 严重错误: 终点指标 {label} 被误删！"
    
    print(f"\n🛡️ 泄露防护：已剔除 {len(cols_to_drop)} 个潜在泄露/非预测特征")
    print(f"📊 当前特征维数 (含标签): {df_clean.shape[1]}")

    # =========================================================
    # 3. 亚组划分 (Subgroup Definition) - 使用原始量级
    # =========================================================
    # 临床定义：入院 24h 内肌酐 < 1.5 mg/dL 且 无慢性肾病史 (CKD)
    if 'creatinine_max' in df_clean.columns and 'chronic_kidney_disease' in df_clean.columns:
        df_clean['subgroup_no_renal'] = (
            (df_clean['creatinine_max'] < 1.5) & 
            (df_clean['chronic_kidney_disease'] == 0)
        ).astype(int)
        
        no_renal_count = df_clean['subgroup_no_renal'].sum()
        print(f"✅ 亚组标记完成: '无预存肾损伤' 样本量 = {no_renal_count} (占 {no_renal_count/len(df_clean):.1%})")
    else:
        print("⚠️ 警告: 缺少关键字段，跳过亚组划分。")
    # =========================================================
    # 3.1 🛡️ 跨数据库分层审计 (预防数据偏倚)
    # =========================================================
    print("\n🛡️ 亚组定义审计 (按数据库来源):")
    
    # 检查是否存在 database 标识列
    db_col = 'database' if 'database' in df_clean.columns else ('source' if 'source' in df_clean.columns else None)

    if db_col:
        for db in df_clean[db_col].unique():
            db_mask = df_clean[db_col] == db
            n_total = db_mask.sum()
            n_sub = df_clean.loc[db_mask, 'subgroup_no_renal'].sum()
            pct = (n_sub / n_total) * 100
            print(f"  [Audit] {db:10}: '无预存肾损' 样本数 = {int(n_sub)} / {n_total} ({pct:.1f}%)")
    else:
        # 如果暂无多库列，打印全样本审计
        n_sub = df_clean['subgroup_no_renal'].sum()
        print(f"  [Audit] 单中心模式: '无预存肾损' 总样本数 = {int(n_sub)} / {len(df_clean)}")
    # =========================================================
    # 4. 📊 自动化统计分析 (Table 1 & Table 2)
    # =========================================================
    from tableone import TableOne
    
    # 定义展示变量
    columns_for_table = [
        'admission_age', 'bmi', 'heart_failure', 'chronic_kidney_disease', 
        'malignant_tumor', 'bun_min', 'creatinine_max', 'lactate_max', 
        'pao2fio2ratio_min', 'wbc_max', 'alt_max', 'ast_max',
        'mortality_28d', 'composite_outcome'
    ]
    
    # 自动识别存在的列与分类变量
    columns_for_table = [c for c in columns_for_table if c in df_clean.columns]
    categorical = [c for c in ['heart_failure', 'chronic_kidney_disease', 'malignant_tumor', 
                               'mortality_28d', 'composite_outcome'] if c in columns_for_table]

    # --- 4.1 生成 Table 1 (POF vs Non-POF) ---
    print("\n📊 正在生成 Table 1: 全人群基线特征 (按 POF 分组)...")
    # 识别非正态分布变量（简单逻辑：所有连续变量通常在医学中都按非正态处理）
    non_normal_cols = [c for c in columns_for_table if c not in categorical]

    # 修改 TableOne 调用
    t1 = TableOne(df_clean, columns=columns_for_table, categorical=categorical, 
                  nonnormal=non_normal_cols, # 新增：指定非正态变量
                  groupby='pof', pval=True, missing=True)
    print(t1.tabulate(tablefmt="github"))
    
    # --- 4.2 生成 Table 2 (Subgroup: Renal vs No-Renal) ---
    print("\n🔍 正在生成 Table 2: 肾功能亚组对比 (按 subgroup_no_renal 分组)...")
    t2 = TableOne(df_clean, columns=columns_for_table, categorical=categorical, 
                  nonnormal=non_normal_cols, # <--- 建议在这里也加上这行
                  groupby='subgroup_no_renal', pval=True, missing=True)
    print(t2.tabulate(tablefmt="github"))

    # --- 4.3 统一保存统计报告 ---
    REPORT_DIR = os.path.join(BASE_DIR, "reports")
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    t1_path = os.path.join(REPORT_DIR, "table_1_pof_comparison.csv")
    t2_path = os.path.join(REPORT_DIR, "table_2_renal_subgroup.csv")
    
    t1.to_csv(t1_path)
    t2.to_csv(t2_path)
    
    print(f"\n✅ 统计报告已保存:")
    print(f"   - Table 1 (POF对比): {t1_path}")
    print(f"   - Table 2 (亚组对比): {t2_path}")

    print("\n💡 状态：特征保持原始物理量级 (Raw Scale)，归一化移至模块 03 执行。")
    
    # =========================================================
    # 5. 最终保存
    # =========================================================
    model_ready_path = os.path.join(SAVE_DIR, "mimic_for_model.csv")
    df_clean.to_csv(model_ready_path, index=False)
    
    print("-" * 60)
    print(f"📊 数据就绪统计:")
    print(f"   - 样本总数: {df_clean.shape[0]}")
    print(f"   - 最终特征数 (含标签与亚组标记): {df_clean.shape[1]}")
    print(f"   - 主要结局 (POF) 发生率: {df_clean['pof'].mean():.2%}")
    print("-" * 60)
    print(f"✅ 模块 02 优化完成! 数据存至: {model_ready_path}")
    # --- 在模块末尾 df_clean.to_csv 之后添加 ---
    import gc
    
    # 显式删除不再需要的原始巨大 DataFrame
    if 'df' in locals():
        del df
        
    # 强制进行垃圾回收
    gc.collect()
    
    print("🧹 内存已清理，准备进入下一模块。")
if __name__ == "__main__":
    run_module_02()
