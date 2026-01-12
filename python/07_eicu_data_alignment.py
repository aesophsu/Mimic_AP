import pandas as pd
import numpy as np
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

# =========================================================
# 1. 配置与路径
# =========================================================
BASE_DIR = ".."
RAW_EICU_PATH = os.path.join(BASE_DIR, "data/eicu_raw_ap.csv") 
SAVE_DIR = os.path.join(BASE_DIR, "data/cleaned")
FEATURES_PATH = os.path.join(BASE_DIR, "models/selected_features.pkl")

def run_module_07_v2():
    print("="*60)
    print("🚀 模块 07: eICU 多中心对齐、审计与特征分析")
    print("="*60)

    # 1. 加载模型资产
    if not os.path.exists(FEATURES_PATH):
        print("❌ 错误：找不到特征清单。")
        return
    selected_features = joblib.load(FEATURES_PATH)
    df = pd.read_csv(RAW_EICU_PATH)

    # 2. 列名映射 (对齐模块 01-03 的命名契约)
    mapping = {
        'age': 'admission_age',
        'ph_min': 'ph_min',
        'creatinine_max': 'creatinine_max',
        'bun_max': 'bun_max',
        'wbc_max': 'wbc_max',
        'ast_max': 'ast_max',
        'lactate_max': 'lactate_max',
        'albumin_min': 'albumin_min',
        'temp_max': 'temperature_max',
        'mbp_min': 'mean_bp_min',
        'spo2_max': 'spo2_max',
        'gender': 'gender'
    }
    df.rename(columns=mapping, inplace=True)
    
    # 性别转换映射
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'M': 1, 'F': 0, 1: 1, 0: 0})

    # ---------------------------------------------------------
    # 3. 核心：特征缺失率审计报告
    # ---------------------------------------------------------
    print("\n🔍 [1/3] 特征对齐审计报告 (MIMIC Top 12 -> eICU):")
    audit_data = []
    for feat in selected_features:
        if feat in df.columns:
            missing = df[feat].isnull().mean() * 100
            status = "✅ 匹配" if missing < 30 else "⚠️ 高缺失"
        else:
            missing = 100.0
            status = "❌ 完全缺失"
        audit_data.append({'Feature': feat, 'Missing%': f"{missing:.2f}%", 'Status': status})
    
    audit_df = pd.DataFrame(audit_data)
    print(audit_df.to_string(index=False))

    # ---------------------------------------------------------
    # 4. 执行数据变换 (Log1p + Clipping)
    # ---------------------------------------------------------
    print("\n🧪 [2/3] 应用偏态转换 (Log1p) 与 物理裁剪...")
    
    # 需要 Log 的偏态指标 (遵循模块 02)
    skewed_features = ['creatinine_max', 'bun_max', 'wbc_max', 'ast_max', 'lactate_max']
    
    for col in skewed_features:
        if col in df.columns:
            # 记录转换前后的中位数用于验证单位对齐
            pre_med = df[col].median()
            df[col] = np.log1p(df[col].astype(float).clip(lower=0))
            # print(f"   - {col:<15}: 原中位数 {pre_med:.2f} -> Log后 {df[col].median():.2f}")
            
    if 'ph_min' in df.columns:
        df['ph_min'] = df['ph_min'].clip(6.8, 7.8)

    # ---------------------------------------------------------
    # 5. 生成模型就绪矩阵
    # ---------------------------------------------------------
    print("\n🛠️ [3/3] 构建验证矩阵与最终清洗...")
    X_eicu = pd.DataFrame(index=df.index)
    for feat in selected_features:
        if feat in df.columns:
            # 用中位数填补 eICU 的缺失值 (模拟模块 03 的简单插补部分)
            X_eicu[feat] = df[feat].fillna(df[feat].median())
        else:
            # 针对完全缺失列，填补 0 (标准化后的均值)
            X_eicu[feat] = 0.0
            
    # 特征自检：打印转换后的关键统计分布
    print("\n📊 转换后指标分布自检 (验证单位对齐):")
    inspect_cols = [c for c in ['ph_min', 'creatinine_max', 'temperature_max'] if c in X_eicu.columns]
    print(X_eicu[inspect_cols].describe().loc[['min', '50%', 'max']])

    # ---------------------------------------------------------
    # 6. 保存数据
    # ---------------------------------------------------------
    eicu_ready_path = os.path.join(SAVE_DIR, "eicu_for_model.csv")
    df_ready = pd.concat([X_eicu, df[['pof']]], axis=1)
    df_ready.to_csv(eicu_ready_path, index=False)
    
    print("-" * 60)
    print(f"✅ 模块 07 成功完成！共处理 {len(df_ready)} 例 eICU 患者。")
    print(f"📁 验证就绪数据已存至: {eicu_ready_path}")

if __name__ == "__main__":
    run_module_07_v2()
