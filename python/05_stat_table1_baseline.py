import os
import pandas as pd
import numpy as np
import joblib
from scipy import stats

# =========================================================
# 配置路径
# =========================================================
BASE_DIR = ".."
# 关键修改：读取清洗后、标准化前的原始数据集 (请根据你实际文件名修改)
RAW_DATA_PATH = os.path.join(BASE_DIR, "data/cleaned/mimic_for_model.csv") 
SAVE_DIR = os.path.join(BASE_DIR, "results")
MODEL_DIR = os.path.join(BASE_DIR, "models")

if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

def run_module_05_raw_full():
    print("="*70)
    print("🚀 运行模块 05: 全量原始数值基线统计 (论文 Table 1 标准版)")
    print("="*70)

    # 1. 加载数据
    if not os.path.exists(RAW_DATA_PATH):
        print(f"❌ 错误: 找不到原始数据文件 {RAW_DATA_PATH}")
        return
    
    df = pd.read_csv(RAW_DATA_PATH)
    
    # 2. 核心特征对齐
    try:
        selected_features = joblib.load(os.path.join(MODEL_DIR, "selected_features.pkl"))
        print(f"✅ 已同步模型核心特征: {len(selected_features)} 个")
    except:
        selected_features = []
        print("⚠️ 警告: 未找到 selected_features.pkl，将使用默认列名")

    # 定义目标变量和分组 N 值
    target = 'pof'
    n_total = len(df)
    n_pof = int(df[target].sum())
    n_non_pof = n_total - n_pof
    
    print(f"📊 分析样本总量: {n_total} (Non-POF: {n_non_pof}, POF: {n_pof})")

    # 定义变量分类
    # 连续变量：人口学 + 模型核心指标
    continuous_vars = ['admission_age', 'weight_admit', 'bmi'] + [f for f in selected_features if f not in ['admission_age']]
    # 分类变量：性别 + 既往史
    categorical_vars = ['gender', 'heart_failure', 'chronic_kidney_disease', 'malignant_tumor']

    table1_data = []

    # --- A. 连续变量处理 (原始数值) ---
    for var in [v for v in continuous_vars if v in df.columns]:
        g0 = df[df[target] == 0][var].dropna()
        g1 = df[df[target] == 1][var].dropna()
        
        # 统计描述逻辑
        _, p_norm = stats.shapiro(df[var].dropna()[:5000]) # 全量数据正态检验
        
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
            f'Non-POF (N={n_non_pof})': desc0,
            f'POF (N={n_pof})': desc1,
            'P-value': p_val,
            'Test': method
        })

    # --- B. 分类变量处理 ---
    # --- B. 分类变量处理 (修复空数据报错) ---
    for var in categorical_vars:
        if var not in df.columns:
            print(f"   ⚠️ 跳过分类变量 {var}: 不在列名中")
            continue
            
        # 统计有效样本数（排除该列的缺失值）
        valid_df = df[[var, target]].dropna()
        if len(valid_df) == 0:
            print(f"   ⚠️ 跳过分类变量 {var}: 该列数据全为空")
            continue

        # 标签映射逻辑
        if var == 'gender':
            valid_df[var+'_label'] = valid_df[var].map({1: 'Male', 0: 'Female'})
        else:
            valid_df[var+'_label'] = valid_df[var].map({1: 'Yes', 0: 'No'})
            
        # 生成交叉表
        contingency = pd.crosstab(valid_df[var+'_label'], valid_df[target])
        
        # 健壮性检查：交叉表必须是 2x2 或更大
        if contingency.size == 0 or contingency.shape[0] < 2:
            print(f"   ⚠️ 跳过分类变量 {var}: 数据分布不足以进行卡方检验")
            continue
        
        try:
            _, p_chi2, _, _ = stats.chi2_contingency(contingency)
        except ValueError:
            p_chi2 = np.nan

        first_row = True
        for idx in contingency.index:
            # 动态获取当前变量下的组内样本量
            c0 = contingency.loc[idx, 0] if 0 in contingency.columns else 0
            c1 = contingency.loc[idx, 1] if 1 in contingency.columns else 0
            
            # 使用全量 N 值计算百分比
            desc0 = f"{int(c0)} ({c0/n_non_pof*100:.1f}%)"
            desc1 = f"{int(c1)} ({c1/n_pof*100:.1f}%)"
            
            table1_data.append({
                'Variable': f"{var}: {idx}",
                f'Non-POF (N={n_non_pof})': desc0,
                f'POF (N={n_pof})': desc1,
                'P-value': p_chi2 if first_row else np.nan,
                'Test': "Chi-square"
            })
            first_row = False

    # 3. 输出与格式化
    table1_df = pd.DataFrame(table1_data)
    
    # 严格的 P 值格式化
    def format_p(x):
        if pd.isna(x): return ""
        if x < 0.001: return "<0.001"
        return f"{x:.3f}"

    table1_df['P-value'] = table1_df['P-value'].apply(format_p)
    
    output_path = os.path.join(SAVE_DIR, "Table1_Full_Raw_Data.csv")
    table1_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print("-" * 85)
    print(table1_df.to_string(index=False))
    print("-" * 85)
    print(f"✅ 全量原始数值 Table 1 已生成: {output_path}")

if __name__ == "__main__":
    run_module_05_raw_full()
