import os
import numpy as np
import pandas as pd

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
            missing = df[col].isnull().mean() * 100
            med = df[col].median()
            mean = df[col].mean()
            v_max = df[col].max()
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
    # 4. 📊 Table 1 自动化统计分析
    # =========================================================
    print("\n📊 正在生成 Table 1 基线特征对比表 (按 POF 分组)...")
    from tableone import TableOne
    
    # 选择要在 Table 1 展示的特征
    columns_for_table1 = [
        'admission_age', 'bmi', 'heart_failure', 'chronic_kidney_disease', 
        'malignant_tumor', 'bun_min', 'creatinine_max', 'lactate_max', 
        'pao2fio2ratio_min', 'wbc_max', 'alt_max', 'ast_max',
        'mortality_28d', 'composite_outcome'
    ]
    
    # 自动过滤不存在的列并识别分类变量
    columns_for_table1 = [c for c in columns_for_table1 if c in df_clean.columns]
    categorical = [c for c in ['heart_failure', 'chronic_kidney_disease', 'malignant_tumor', 
                               'mortality_28d', 'composite_outcome'] if c in columns_for_table1]

    # 执行统计：pval=True 自动进行显著性检验 (T-test/Kruskal-Wallis/Chi-square)
    mytable = TableOne(df_clean, columns=columns_for_table1, categorical=categorical, 
                       groupby='pof', pval=True, missing=True)
    
    print(mytable.tabulate(tablefmt="github"))
    
    # 保存统计报告
    report_path = os.path.join(BASE_DIR, "reports/table_1_baseline.csv")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    mytable.to_csv(report_path)
    print(f"✅ Table 1 已保存至: {report_path}")

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

if __name__ == "__main__":
    run_module_02()
