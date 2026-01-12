import pandas as pd
import numpy as np
import os
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency

# =========================================================
# 1. 路径配置
# =========================================================
MIMIC_PATH = "../data/ap_final_analysis_cohort.csv"
EICU_PATH = "../data/ap_eicu_validation.csv"
OUTPUT_PATH = "../figures/Table1_Refined_Final.csv"

# =========================================================
# 2. 核心统计工具函数
# =========================================================
def calculate_smd(m_vec, e_vec, is_categorical=False):
    """优化后的 SMD 计算逻辑，确保分类变量不返回 nan"""
    try:
        m_vec = m_vec.astype(float).dropna()
        e_vec = e_vec.astype(float).dropna()
        
        if is_categorical:
            p1 = m_vec.mean()
            p2 = e_vec.mean()
            # 分类变量 SMD 公式
            denom = np.sqrt((p1*(1-p1) + p2*(1-p2)) / 2)
        else:
            m1, m2 = m_vec.mean(), e_vec.mean()
            s1, s2 = m_vec.std(), e_vec.std()
            # 连续变量 SMD 公式
            denom = np.sqrt((s1**2 + s2**2) / 2)
        
        return np.abs(m_vec.mean() - e_vec.mean()) / denom if denom != 0 else 0
    except:
        return np.nan

def generate_table1_final(df_mimic, df_eicu, feature_mapping):
    results = []
    
    for m_col, e_col in feature_mapping.items():
        if m_col not in df_mimic.columns or e_col not in df_eicu.columns:
            print(f"⚠️ 跳过: {m_col}")
            continue
            
        m_vec = pd.to_numeric(df_mimic[m_col], errors='coerce').dropna()
        e_vec = pd.to_numeric(df_eicu[e_col], errors='coerce').dropna()

        if len(m_vec) == 0 or len(e_vec) == 0:
            continue

        # 判定是否为分类变量
        unique_vals = m_vec.unique()
        is_categorical = len(unique_vals) <= 2
        
        if is_categorical:
            target_val = np.max(unique_vals)
            m_c, e_c = (m_vec == target_val).sum(), (e_vec == target_val).sum()
            m_s = f"{int(m_c)} ({m_c/len(m_vec)*100:.1f}%)"
            e_s = f"{int(e_c)} ({e_c/len(e_vec)*100:.1f}%)"
            
            # 卡方检验
            obs = np.array([[m_c, len(m_vec)-m_c], [e_c, len(e_vec)-e_c]])
            try:
                _, p_val, _, _ = chi2_contingency(obs)
            except:
                p_val = 1.0
            
            smd = calculate_smd(m_vec, e_vec, is_categorical=True)
            stat_type = "n (%)"
        else:
            # 连续变量统计
            smd = calculate_smd(m_vec, e_vec, is_categorical=False)
            if abs(m_vec.skew()) > 1.5:
                m_s = f"{m_vec.median():.2f} [{m_vec.quantile(0.25):.2f}-{m_vec.quantile(0.75):.2f}]"
                e_s = f"{e_vec.median():.2f} [{e_vec.quantile(0.25):.2f}-{e_vec.quantile(0.75):.2f}]"
                _, p_val = mannwhitneyu(m_vec, e_vec)
                stat_type = "Median [IQR]"
            else:
                m_s = f"{m_vec.mean():.2f} ({m_vec.std():.2f})"
                e_s = f"{e_vec.mean():.2f} ({e_vec.std():.2f})"
                _, p_val = ttest_ind(m_vec, e_vec)
                stat_type = "Mean (SD)"

        results.append({
            "Characteristic": m_col,
            "Type": stat_type,
            "MIMIC-IV": m_s,
            "eICU": e_s,
            "P-value": f"{p_val:.3f}" if p_val >= 0.001 else "<0.001",
            "SMD": f"{smd:.3f}"
        })
    
    return pd.DataFrame(results)

# =========================================================
# 3. 执行主流程
# =========================================================
if __name__ == "__main__":
    df_mimic = pd.read_csv(MIMIC_PATH)
    df_eicu = pd.read_csv(EICU_PATH)

    # --- 数据深度清洗 (针对单位不统一问题) ---
    
    # 1. 性别预处理
    df_mimic['gender_num'] = df_mimic['gender'].map({'M': 1, 'F': 0, 1: 1, 0: 0})
    df_eicu['gender_num'] = df_eicu['gender'].map({'M': 1, 'F': 0, 1: 1, 0: 0, 'Male': 1, 'Female': 0})
    
    # 2. pH 值清洗: 剔除小于 6.5 或大于 8.0 的非生理值 (解决 eICU 2.84 问题)
    df_eicu.loc[(df_eicu['ph_min'] < 6.5) | (df_eicu['ph_min'] > 8.0), 'ph_min'] = np.nan
    
    # 3. 纤维蛋白原单位对齐: 判定 eICU 是否为 g/L (均值远小于 10 则乘以 100 转换为 mg/dL)
    fib_mean = df_eicu['fibrinogen_max'].mean()
    if fib_mean < 15: # 典型的单位错位阈值
        print(f"🔧 检测到 eICU 纤维蛋白原单位异常 (Mean={fib_mean:.2f}), 正在进行 mg/dL 转换...")
        df_eicu['fibrinogen_max'] = df_eicu['fibrinogen_max'] * 100

    # 4. BMI 异常值清洗
    df_eicu.loc[(df_eicu['bmi'] > 80) | (df_eicu['bmi'] < 12), 'bmi'] = np.nan

    # 映射表
    mapping = {
        'admission_age': 'age',
        'gender_num': 'gender_num',
        'weight_admit': 'weight',
        'bmi': 'bmi',
        'ph_min': 'ph_min',
        'creatinine_max': 'creatinine_max',
        'bun_max': 'bun_max',
        'wbc_max': 'wbc_max',
        'aniongap_max': 'aniongap_max',
        'glucose_max': 'glucose_max',
        'fibrinogen_max': 'fibrinogen_max',
        'ptt_max': 'ptt_max',
        'lactate_max': 'lactate_max',
        'spo2_max': 'spo2_max',
        'vaso_flag': 'vaso_flag',
        'mechanical_vent_flag': 'vent_flag',
        'pof': 'pof_proxy'
    }

    print("--- 正在生成优化后的 Table 1 ---")
    table1 = generate_table1_final(df_mimic, df_eicu, mapping)
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    table1.to_csv(OUTPUT_PATH, index=False)
    
    print(f"✅ 成功! 修正后的文件已保存至: {OUTPUT_PATH}")
    print("\n表格预览:")
    print(table1.to_string(index=False))
