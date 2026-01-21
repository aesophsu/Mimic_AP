import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tableone import TableOne

# =========================================================
# 1. 配置与路径
# =========================================================
BASE_DIR = "../../"
# 注意：审计使用的是物理值版本，而非标准化后的版本
INPUT_PATH = os.path.join(BASE_DIR, "data/cleaned/mimic_raw_scale.csv")
# 获取模块 03 已经算好的 processed 数据（为了获取亚组标记）
PROCESSED_PATH = os.path.join(BASE_DIR, "data/cleaned/mimic_processed.csv")
RESULT_DIR = os.path.join(BASE_DIR, "results/tables")
FIGURE_DIR = os.path.join(BASE_DIR, "results/figures/audit")

os.makedirs(FIGURE_DIR, exist_ok=True)

def run_mimic_stat_audit():
    print("="*70)
    print("🚀 启动模块 04: 深度统计审计与缺失值可视化")
    print("="*70)

    # 加载数据
    df_raw = pd.read_csv(INPUT_PATH)
    df_proc = pd.read_csv(PROCESSED_PATH)
    
    # 将 03 步生成的亚组标记合并回 raw 数据中以便审计
    if 'subgroup_no_renal' in df_proc.columns:
        df_raw['subgroup_no_renal'] = df_proc['subgroup_no_renal']

    # =========================================================
    # 2. 缺失值热图审计 (Missingness Heatmap)
    # =========================================================
    print("\n🎨 正在绘制缺失值分布热图...")
    plt.figure(figsize=(15, 8))
    # 选取前 50 个特征进行可视化避免图表过挤
    cols_to_plot = [c for c in df_raw.columns if 'id' not in c.lower()][:50]
    sns.heatmap(df_raw[cols_to_plot].isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Data Heatmap (First 50 Features)')
    heatmap_path = os.path.join(FIGURE_DIR, "missingness_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"✅ 热图已保存至: {heatmap_path}")

    # =========================================================
    # 3. 核心统计变量定义
    # =========================================================
    clinical_features = [
        'admission_age', 'bmi', 'heart_failure', 'chronic_kidney_disease', 
        'malignant_tumor', 'bun_min', 'creatinine_max', 'lactate_max', 
        'pao2fio2ratio_min', 'wbc_max', 'alt_max', 'ast_max', 'glucose_max',
        'platelets_min', 'bilirubin_max'
    ]
    outcomes = ['pof', 'mortality_28d']
    
    # 筛选实际存在的列
    all_audit_cols = [c for c in (clinical_features + outcomes) if c in df_raw.columns]
    categorical = [c for c in ['heart_failure', 'chronic_kidney_disease', 'malignant_tumor', 'pof', 'mortality_28d'] if c in all_audit_cols]
    nonnormal = [c for c in all_audit_cols if c not in categorical]

    # =========================================================
    # 4. 单因素分析与 P-value 过滤
    # =========================================================
    print("\n🔬 执行单因素显著性审计 (By POF)...")
    t1 = TableOne(df_raw, columns=all_audit_cols, categorical=categorical, 
                  nonnormal=nonnormal, groupby='pof', pval=True)
    
    # 提取 P-value 小于 0.05 的变量
    # tableone 的 table 存储在 .tableone 属性中
    t1_df = t1.tableone
    
    # 尝试解析 P-Value 列
    try:
        # 寻找 P-Value 列（通常是最后一列）
        pval_col = [c for c in t1_df.columns if 'P-Value' in str(c)][0]
        # 转换并筛选显著变量
        sig_vars = t1_df[t1_df[pval_col].apply(lambda x: '<' in str(x) or (isinstance(x, float) and x < 0.05))]
        
        print(f"\n📢 [统计发现] 以下变量在 POF 组间具有显著差异 (P < 0.05):")
        for idx in sig_vars.index[:10]: # 打印前 10 个
            print(f"  - {idx[0]}")
    except Exception as e:
        print(f"⚠️ 无法自动解析显著变量: {e}")

    # =========================================================
    # 5. 保存审计报告
    # =========================================================
    t1_path = os.path.join(RESULT_DIR, "table_1_detailed_audit.csv")
    t1.to_csv(t1_path)
    
    # 如果存在亚组，产出亚组审计
    if 'subgroup_no_renal' in df_raw.columns:
        t2 = TableOne(df_raw, columns=all_audit_cols, categorical=categorical, 
                      nonnormal=nonnormal, groupby='subgroup_no_renal', pval=True)
        t2_path = os.path.join(RESULT_DIR, "table_2_subgroup_audit.csv")
        t2.to_csv(t2_path)
        print(f"\n✅ 亚组审计报告已更新: {t2_path}")

    print("\n" + "="*70)
    print("📊 深度审计完成！")
    print(f"统计建议：在下一步 LASSO 筛选中，重点关注显著性变量。")
    print("="*70)

if __name__ == "__main__":
    run_mimic_stat_audit()
