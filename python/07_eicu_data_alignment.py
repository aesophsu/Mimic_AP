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
MODELS_DIR = os.path.join(BASE_DIR, "models")

def run_module_07(target='pof'):
    print("\n" + "="*70)
    print(f"🚀 模块 07: eICU 多中心对齐 (结局: {target.upper()})")
    print("="*70)

    # 1. 动态加载该结局在 MIMIC 训练阶段产生的资产
    assets_path = os.path.join(MODELS_DIR, f"train_assets_{target}.pkl")
    if not os.path.exists(assets_path):
        print(f"❌ 错误：找不到结局 {target} 的资产文件 {assets_path}。请先运行模块 03。")
        return
        
    train_assets = joblib.load(assets_path)
    selected_features = train_assets['selected_features']
    mimic_medians = train_assets['medians']
    skewed_cols_to_log = train_assets['skewed_cols']
    if not os.path.exists(RAW_EICU_PATH):
        print(f"❌ 错误：找不到 eICU 原始数据 {RAW_EICU_PATH}")
        return
    df = pd.read_csv(RAW_EICU_PATH)
    
    # 2. 列名映射 (保持不变，确保与 MIMIC 变量名对齐)
    mapping = {
        'age': 'admission_age',
        'gender': 'gender',
        'pao2fio2': 'pao2fio2ratio_min', 
        'ph_min': 'ph_min', 'ph_max': 'ph_max',
        'spo2_min': 'spo2_min', 'spo2_max': 'spo2_max',
        'creatinine_max': 'creatinine_max',
        'bun_max': 'bun_max', 'bun_min': 'bun_min',
        'lactate_max': 'lactate_max',
        'aniongap_max': 'aniongap_max', 'aniongap_min': 'aniongap_min',
        'calcium_min': 'lab_calcium_min',
        'glucose_max': 'glucose_lab_max',
        'bicarbonate_min': 'bicarbonate_min',
        'wbc_max': 'wbc_max', 'wbc_min': 'wbc_min',
        'albumin_max': 'albumin_max', 'albumin_min': 'albumin_min',
        'alp_max': 'alp_max', 'ast_max': 'ast_max', 'alt_max': 'alt_max',
        'bilirubin_min': 'bilirubin_total_min',
        'hemoglobin_min': 'hemoglobin_min',
        'ptt_min': 'ptt_min',
        'tumor': 'malignant_tumor'
    }
    df.rename(columns=mapping, inplace=True)
    
    # 性别转换映射
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'M': 1, 'F': 0, 1: 1, 0: 0})

    # 3. 特征缺失率审计
    print(f"\n🔍 [1/3] 审计: {target} 所需的特征在 eICU 中的匹配情况")
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

    # 4. 执行数据变换 (Log1p + Clipping)
    print(f"\n🧪 [2/3] 应用 {target} 专属偏态转换 (Log1p)...")
    for col in skewed_cols_to_log:
        if col in df.columns:
            # 执行 Log1p
            df[col] = np.log1p(df[col].astype(float).clip(lower=0))
            
    if 'ph_min' in df.columns:
        df['ph_min'] = df['ph_min'].clip(6.8, 7.8)

    # 5. 生成模型就绪矩阵

    print("\n🛠️ [3/3] 构建验证矩阵与最终清洗...")
    X_eicu = pd.DataFrame(index=df.index)
    for feat in selected_features:
        if feat in df.columns:
            # 优先用 eICU 自身的中位数填充，若 eICU 缺失该特征，则用 MIMIC 的记忆补全
            X_eicu[feat] = df[feat].fillna(df[feat].median())
        else:
            # 填补 MIMIC 训练集该特征的中位数（或者是对数变换后的中位数）
            X_eicu[feat] = mimic_medians.get(feat, 0.0)
            
    # 特征自检：打印转换后的关键统计分布
    print("\n📊 转换后指标分布自检 (验证单位对齐):")
    inspect_cols = [c for c in ['ph_min', 'creatinine_max', 'temperature_max'] if c in X_eicu.columns]
    print(X_eicu[inspect_cols].describe().loc[['min', '50%', 'max']])

    # 6. 保存数据
    eicu_ready_path = os.path.join(SAVE_DIR, f"eicu_for_model_{target}.csv")
    # 确保目标标签列存在 (eICU 的 csv 里必须有对应结局列)
    if target in df.columns:
        df_ready = pd.concat([X_eicu, df[[target]]], axis=1)
        df_ready.to_csv(eicu_ready_path, index=False)
        print("-" * 60)
        print(f"✅ 结局 {target.upper()} 处理成功！")
        print(f"📁 验证数据保存至: {eicu_ready_path}")
    else:
        print(f"⚠️ 警告：eICU 原始数据中找不到结局列 '{target}'，仅保存特征矩阵。")
        X_eicu.to_csv(eicu_ready_path, index=False)
    
def run_all_eicu_alignment():
    """循环处理所有结局"""
    targets = ['pof', 'composite_outcome', 'mortality_28d']
    for t in targets:
        try:
            run_module_07(t)
        except Exception as e:
            print(f"❌ 处理结局 {t} 时发生意外错误: {e}")
            
if __name__ == "__main__":
    run_all_eicu_alignment()
