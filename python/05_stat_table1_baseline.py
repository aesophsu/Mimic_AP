import os
import pandas as pd
import numpy as np
import joblib
from scipy import stats

# =========================================================
# 配置路径
# =========================================================
BASE_DIR = ".."
INPUT_PATH = os.path.join(BASE_DIR, "data/cleaned/mimic_for_model.csv")
SAVE_DIR = os.path.join(BASE_DIR, "results")
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

def run_module_05():
    print("="*60)
    print("🚀 运行模块 05: 临床 Table 1 自动化统计 (精简版)")
    print("="*60)

    if not os.path.exists(INPUT_PATH):
        print(f"❌ 错误: 找不到输入文件 {INPUT_PATH}")
        return
        
    df = pd.read_csv(INPUT_PATH)
    target = 'pof'

    # 1. 定义要分析的临床维度（排除病因）
    # 人口学与体格指标
    demographics = ['admission_age', 'weight_admit', 'bmi']
    # 既往病史
    comorbidities = ['heart_failure', 'chronic_kidney_disease', 'malignant_tumor']
    # 模块 3 筛选的 Top 12 核心生化指标
    try:
        selected_features = joblib.load(os.path.join(BASE_DIR, "models/selected_features.pkl"))
    except:
        selected_features = []
        print("⚠️ 未找到 selected_features.pkl")

    # 汇总所有需要分析的变量
    continuous_vars = [v for v in (demographics + selected_features) if v in df.columns]
    categorical_vars = [v for v in (['gender'] + comorbidities) if v in df.columns]

    table1_data = []

    # --- A. 连续变量处理 ---
    for var in continuous_vars:
        g0 = df[df[target] == 0][var].dropna()
        g1 = df[df[target] == 1][var].dropna()
        
        if len(g0) == 0 or len(g1) == 0: continue

        # 正态性检验
        _, p_norm = stats.shapiro(df[var].dropna()[:5000])
        
        if p_norm > 0.05:
            # 正态分布: Mean ± SD
            desc0 = f"{g0.mean():.2f} ± {g0.std():.2f}"
            desc1 = f"{g1.mean():.2f} ± {g1.std():.2f}"
            _, p_val = stats.ttest_ind(g0, g1)
            method = "t-test"
        else:
            # 非正态分布: Median [IQR]
            desc0 = f"{g0.median():.2f} [{g0.quantile(0.25):.2f}-{g0.quantile(0.75):.2f}]"
            desc1 = f"{g1.median():.2f} [{g1.quantile(0.25):.2f}-{g1.quantile(0.75):.2f}]"
            _, p_val = stats.mannwhitneyu(g0, g1)
            method = "Mann-Whitney U"
            
        table1_data.append({
            'Variable': var,
            'Non-POF (N=612)': desc0, # 这里的N值根据你之前的输出调整
            'POF (N=577)': desc1,
            'P-value': p_val,
            'Test': method
        })

    # --- B. 分类变量处理 ---
    for var in categorical_vars:
        # 统一映射
        if var == 'gender':
            df[var+'_label'] = df[var].replace({1: 'Male', 0: 'Female'})
        else:
            df[var+'_label'] = df[var].replace({1: 'Yes', 0: 'No'})
            
        contingency = pd.crosstab(df[var+'_label'], df[target])
        if contingency.shape[0] < 2: continue
        
        _, p_chi2, _, _ = stats.chi2_contingency(contingency)
        
        for idx in contingency.index:
            c0, c1 = contingency.loc[idx, 0], contingency.loc[idx, 1]
            n0, n1 = len(df[df[target]==0]), len(df[df[target]==1])
            desc0 = f"{int(c0)} ({c0/n0*100:.1f}%)"
            desc1 = f"{int(c1)} ({c1/n1*100:.1f}%)"
            
            table1_data.append({
                'Variable': f"{var}: {idx}",
                'Non-POF (N=612)': desc0,
                'POF (N=577)': desc1,
                'P-value': p_chi2 if idx == contingency.index[0] else np.nan,
                'Test': "Chi-square"
            })

    # 3. 整理与输出
    table1_df = pd.DataFrame(table1_data)
    table1_df['P-value'] = table1_df['P-value'].apply(lambda x: "<0.001" if x < 0.001 else (f"{x:.4f}" if pd.notna(x) else ""))
    
    output_path = os.path.join(SAVE_DIR, "Table1_Baseline_Characteristics.csv")
    table1_df.to_csv(output_path, index=False)
    
    print("-" * 60)
    print(table1_df.to_string(index=False))
    print("-" * 60)
    print(f"✅ Table 1 已更新并保存至: {output_path}")

if __name__ == "__main__":
    run_module_05()
