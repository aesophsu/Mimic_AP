import os
import json
import numpy as np
import pandas as pd

# =========================================================
# 1. 配置与路径映射 (基于新目录树)
# =========================================================
BASE_DIR = "../../"
INPUT_PATH = os.path.join(BASE_DIR, "data/raw/mimic_raw_data.csv")
DICT_PATH = os.path.join(BASE_DIR, "artifacts/features/feature_dictionary.json")
SAVE_DIR = os.path.join(BASE_DIR, "data/cleaned")
MISSING_THRESHOLD = 0.3

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

class FeatureAuditor:
    """临床特征审计器：负责根据字典定义校验单位与生理范围"""
    
    def __init__(self, dict_path):
        with open(dict_path, 'r', encoding='utf-8') as f:
            self.feature_dict = json.load(f)

    def audit_units_and_ranges(self, df):
        df_cleaned = df.copy()
        print(f"\n{'Feature':<20} | {'Action':<40} | {'Status'}")
        print("-" * 80)

        for col, config in self.feature_dict.items():
            # 增加两个前置检查：
            # 1. 列必须在 DataFrame 中
            # 2. 列必须是数值类型 (避免处理日期或 ID 字符串)
            if col not in df_cleaned.columns:
                continue
            
            if not pd.api.types.is_numeric_dtype(df_cleaned[col]):
                # 如果字典里定义了但数据里是字符串/日期，跳过审计
                continue

            med = df_cleaned[col].median()
            log_min = config['ref_range']['logical_min']
            log_max = config['ref_range']['logical_max']
            factor = config.get('conversion_factor', 1.0)

            if config.get("apply_expm1", False) and pd.notnull(med):
                if med < 10:  # 只有当量级明显是 Log 时才触发
                    df_cleaned[col] = np.expm1(df_cleaned[col])
                    print(f"{col:<20} | Applied expm1 restoration (Anti-Log) | 🔄")
                    # 更新 med 以便后续的单位转换逻辑使用正确的中值
                    med = df_cleaned[col].median()

            # --- 原有的：单位自动对齐 ---
            if pd.notnull(med) and log_min is not None:
                if med < (log_min * 0.2) and factor != 1.0:
                    df_cleaned[col] = df_cleaned[col] * factor
                    print(f"{col:<20} | Applied conversion factor x{factor:<10} | ✅")
            
            # --- 原有的：生理异常值清洗 ---
            if log_min is not None and log_max is not None:
                mask = (df_cleaned[col] < log_min) | (df_cleaned[col] > log_max)
                if mask.any():
                    df_cleaned.loc[mask, col] = np.nan
                    print(f"{col:<20} | Removed {mask.sum():>3} physiologic outliers | ⚠️")

        return df_cleaned

# =========================================================
# 2. 核心清洗模块
# =========================================================
def run_cross_database_alignment():
    print("="*70)
    print("🚀 启动模块 02: 临床特征空间审计与清洗 (MIMIC-IV)")
    print("="*70)

    if not os.path.exists(INPUT_PATH):
        print(f"❌ 错误: 找不到输入文件 {INPUT_PATH}")
        return

    df = pd.read_csv(INPUT_PATH)
    auditor = FeatureAuditor(DICT_PATH)
    
    # 2.1 结局指标规整与早亡修正
    if 'early_death_24_48h' in df.columns:
        early_death_mask = (df['early_death_24_48h'] == 1)
        df.loc[early_death_mask, 'pof'] = 1
        if 'mortality_28d' in df.columns:
            df.loc[early_death_mask, 'mortality_28d'] = 1

    if 'pof' in df.columns and 'mortality_28d' in df.columns:
        df['composite_outcome'] = ((df['pof'] == 1) | (df['mortality_28d'] == 1)).astype(int)
        print("✅ [Labels] 结局指标对齐完成。")

    # 2.2 核心保护白名单与缺失率过滤
    white_list = [
        'subject_id', 'hadm_id', 'stay_id', 'database',
        'pof', 'mortality_28d', 'composite_outcome', 'early_death_24_48h',
        'lactate_max', 'pao2fio2ratio_min', 'lipase_max', 'creatinine_max', 'bun_min'
    ]
    
    missing_pct = df.isnull().mean()
    cols_to_drop = [c for c in missing_pct[missing_pct > MISSING_THRESHOLD].index if c not in white_list]
    df = df.drop(columns=cols_to_drop)
    print(f"🗑️ [Filter] 已剔除缺失率 >{MISSING_THRESHOLD*100}% 的非核心特征。")

    # 2.3 物理单位审计与生理范围约束
    df = auditor.audit_units_and_ranges(df)

    # 2.4 统计截断 (Statistical Clipping)
    print("\n✂️ [Clipping] 执行 1%-99% 统计盖帽处理...")
    binary_cols = ['gender_num', 'vaso_flag', 'mechanical_vent_flag', 'composite_outcome', 'pof']
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col not in white_list and col not in binary_cols:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower, upper)

    # 2.5 审计总结与持久化
    print("\n" + "-"*70)
    print(f"📊 模块清洗统计: 样本 {df.shape[0]} | 特征 {df.shape[1]}")
    
    # 抽取核心指标进行审计报告
    report_cols = ['bun_min', 'creatinine_max', 'lactate_max', 'pao2fio2ratio_min']
    print("\n🔍 关键特征统计审计:")
    for c in report_cols:
        if c in df.columns:
            print(f"  > {c:<20}: Median={df[c].median():>8.2f} | Missing={df[c].isnull().mean()*100:>6.2f}%")

    save_path = os.path.join(SAVE_DIR, "mimic_raw_scale.csv")
    df.to_csv(save_path, index=False)
    print(f"\n✅ 已生成中间产物: {save_path}")

if __name__ == "__main__":
    run_cross_database_alignment()
