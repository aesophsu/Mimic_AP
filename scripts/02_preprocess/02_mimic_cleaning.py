import os
import json
import numpy as np
import pandas as pd

BASE_DIR = "../../"
INPUT_PATH = os.path.join(BASE_DIR, "data/raw/mimic_raw_data.csv")
DICT_PATH = os.path.join(BASE_DIR, "artifacts/features/feature_dictionary.json")
SAVE_DIR = os.path.join(BASE_DIR, "data/cleaned")
MISSING_THRESHOLD = 0.3

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

class FeatureAuditor:
    def __init__(self, dict_path):
        with open(dict_path, 'r', encoding='utf-8') as f:
            self.feature_dict = json.load(f)

    def audit_units_and_ranges(self, df):
        print(f"\n📋 原始数据探测: {df.shape[0]} 行, {df.shape[1]} 列")
        print(f"{'Feature Name':<25} | {'Missing%':<10} | {'Median':<10} | {'Mean':<10} | {'Max':<10}")
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                series = df[col].dropna()
                missing = df[col].isnull().mean() * 100
                med = series.median() if not series.empty else 0
                mean = series.mean() if not series.empty else 0
                v_max = series.max() if not series.empty else 0
                print(f"{col:<25} | {missing:>8.2f}% | {med:>10.2f} | {mean:>10.2f} | {v_max:>10.2f}")
               
        df_cleaned = df.copy()
        print(f"\n{'Feature':<20} | {'Action':<40} | {'Status'}")
        print("-" * 80)
        
        for col, config in self.feature_dict.items():
            if col not in df_cleaned.columns:
                continue
           
            if not pd.api.types.is_numeric_dtype(df_cleaned[col]):
                continue
            
            ref = config.get('ref_range', {})
            log_min = ref.get('logical_min')
            log_max = ref.get('logical_max')
            factor = config.get('conversion_factor', 1.0)
           
            series_curr = df_cleaned[col].dropna()
            if series_curr.empty:
                continue
           
            med = series_curr.median()
            
            # 1. 单位转换
            if log_min is not None and factor != 1.0:
                if med < (log_min * 0.2):
                    df_cleaned[col] = df_cleaned[col] * factor
                    print(f"{col:<20} | Applied conversion factor x{factor:<10} | ✅")
            
            # 2. 生理范围硬过滤
            if log_min is not None and log_max is not None:
                mask = (df_cleaned[col] < log_min) | (df_cleaned[col] > log_max)
                if mask.any():
                    num_removed = mask.sum()
                    df_cleaned.loc[mask, col] = np.nan
                    print(f"{col:<20} | Removed {num_removed:>3} physiologic outliers | ⚠️")
        
        return df_cleaned

def run_cross_database_alignment():
    print("="*70)
    print("🚀 启动模块 02: 临床特征空间审计与清洗 (MIMIC-IV)")
    print("="*70)
    
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 错误: 找不到输入文件 {INPUT_PATH}")
        return
    
    df = pd.read_csv(INPUT_PATH)
    auditor = FeatureAuditor(DICT_PATH)
   
    # --- 标签对齐逻辑 ---
    if 'early_death_24_48h' in df.columns:
        early_death_mask = (df['early_death_24_48h'] == 1)
        df.loc[early_death_mask, 'pof'] = 1
        if 'mortality_28d' in df.columns:
            df.loc[early_death_mask, 'mortality_28d'] = 1
    
    if 'pof' in df.columns and 'mortality_28d' in df.columns:
        df['composite_outcome'] = ((df['pof'] == 1) | (df['mortality_28d'] == 1)).astype(int)
        print("✅ [Labels] 结局指标对齐完成。")
    
    # --- 新增：统一结局变量名称（与目录树和 artifacts/models/{target} 一致） ---
    if 'mortality_28d' in df.columns:
        df = df.rename(columns={'mortality_28d': 'mortality'})
        print("✅ 已将 'mortality_28d' 重命名为 'mortality'")
    
    if 'composite_outcome' in df.columns:
        df = df.rename(columns={'composite_outcome': 'composite'})
        print("✅ 已将 'composite_outcome' 重命名为 'composite'")
    
    # --- 特征筛选逻辑（同步更新 white_list 中的旧名称） ---
    white_list = [
        'subject_id', 'hadm_id', 'stay_id', 'database',
        'pof', 'mortality', 'composite', 'early_death_24_48h',
        'lactate_max', 'pao2fio2ratio_min', 'lipase_max', 'creatinine_max', 'bun_min'
    ]
   
    missing_pct = df.isnull().mean()
    cols_to_drop = [c for c in missing_pct[missing_pct > MISSING_THRESHOLD].index if c not in white_list]
    df = df.drop(columns=cols_to_drop)
    print(f"🗑️ [Filter] 已剔除缺失率 >{MISSING_THRESHOLD*100}% 的非核心特征。")
    
    # --- 执行审计与逻辑过滤 ---
    df = auditor.audit_units_and_ranges(df)
    
    # --- 执行 1%-99% 统计盖帽 (Clipping) ---
    print("\n✂️ [Clipping] 执行 1%-99% 统计盖帽处理...")
    binary_cols = ['gender_num', 'vaso_flag', 'mechanical_vent_flag', 'composite', 'pof']  # 已更新
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
   
    for col in numeric_cols:
        if col not in white_list and col not in binary_cols:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            if pd.notnull(lower) and pd.notnull(upper):
                df[col] = df[col].clip(lower, upper)
    
    print("\n" + "-"*70)
    print(f"📊 模块清洗统计: 样本 {df.shape[0]} | 特征 {df.shape[1]}")

    report_cols = ['bun_min', 'creatinine_max', 'lactate_max', 'spo2_max', 'pao2fio2ratio_min', 'rdw_max', 'mortality', 'composite']
    print("\n🔍 关键特征统计审计:")
    for c in report_cols:
        if c in df.columns:
            series = df[c].dropna()
            med = series.median() if not series.empty else 0
            miss = df[c].isnull().mean() * 100
            print(f" > {c:<20}: Median={med:>8.2f} | Missing={miss:>6.2f}%")
    
    save_path = os.path.join(SAVE_DIR, "mimic_raw_scale.csv")
    df.to_csv(save_path, index=False)
    print(f"\n✅ 已生成中间产物: {save_path}")

if __name__ == "__main__":
    run_cross_database_alignment()
