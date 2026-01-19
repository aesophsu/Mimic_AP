import pandas as pd
import numpy as np
import joblib
import os

# =========================================================
# 1. 配置与路径
# =========================================================
BASE_DIR = ".."
RAW_EICU_PATH = os.path.join(BASE_DIR, "data/ap_external_validation.csv") 
SAVE_DIR = os.path.join(BASE_DIR, "data/cleaned")
MODELS_DIR = os.path.join(BASE_DIR, "models")

def run_module_07(target='pof'):
    print("\n" + "="*70)
    print(f"🚀 模块 07: eICU 多中心对齐 (结局: {target.upper()})")
    print("="*70)

    # 1. 加载 MIMIC 训练阶段资产
    assets_path = os.path.join(MODELS_DIR, f"train_assets_{target}.pkl")
    if not os.path.exists(assets_path):
        print(f"❌ 错误：找不到结局 {target} 的资产文件。")
        return
        
    train_assets = joblib.load(assets_path)
    selected_features = train_assets['selected_features']
    mimic_medians = train_assets['medians']
    skewed_cols_to_log = train_assets['skewed_cols']

    if not os.path.exists(RAW_EICU_PATH):
        print(f"❌ 错误：找不到 eICU 原始数据 {RAW_EICU_PATH}")
        return
    
    df = pd.read_csv(RAW_EICU_PATH)

    # 2. 结局列名称映射逻辑
    target_col_map = {
        'pof': 'pof',
        'composite_outcome': 'composite_outcome',
        'mortality_28d': 'mortality_28d'
    }
    actual_target_col = target_col_map.get(target)

    # 3. 基础清洗与特定指标填充
    if 'gender' in df.columns:
        # eICU gender 可能为字符串，统一转为数值
        df['gender'] = df['gender'].map({'M': 1, 'F': 0, 1: 1, 0: 0}).fillna(1)
    
    # P/F Ratio 缺失值填充（临床正常值 400）
    if 'pao2fio2ratio_min' in df.columns:
        df['pao2fio2ratio_min'] = df['pao2fio2ratio_min'].fillna(400)
        print("ℹ️ 已将 pao2fio2ratio_min 缺失值填充为 400")

    # 4. 特征缺失率审计
    print(f"\n🔍 [1/3] 特征审计: {target}")
    audit_data = []
    for feat in selected_features:
        if feat in df.columns:
            missing = df[feat].isnull().mean() * 100
            status = "✅ 匹配" if missing < 30 else "⚠️ 高缺失"
        else:
            missing = 100.0
            status = "❌ 缺失"
        audit_data.append({'Feature': feat, 'Missing%': f"{missing:.2f}%", 'Status': status})
    print(pd.DataFrame(audit_data).sort_values('Missing%').to_string(index=False))

    # 5. 执行数据变换 (Log1p + Clipping)
    print(f"\n🧪 [2/3] 应用数据变换与生理剪裁...")
    
    # 【同步更新】pH 值的生理限度裁剪，与 SQL 逻辑保持一致
    if 'ph_min' in df.columns:
        # 注意：pH 不需要 Log 变换，因为它本身就是对数尺度
        df['ph_min'] = df['ph_min'].clip(6.7, 7.8)
        print(f"ℹ️ ph_min 已应用生理剪裁: [6.7, 7.8]")

    # 对偏态分布指标进行 Log1p
    for col in skewed_cols_to_log:
        if col in df.columns and col != 'ph_min': # pH 绝不进行 log
            df[col] = np.log1p(df[col].astype(float).clip(lower=0))
    
    # 6. 构建最终矩阵
    print("\n🛠️ [3/3] 构建验证矩阵并填充剩余缺失值...")
    X_eicu = pd.DataFrame(index=df.index)
    for feat in selected_features:
        if feat in df.columns:
            # 策略：优先用 eICU 局部中位数，次选 MIMIC 全局中值
            local_median = df[feat].median()
            fill_val = local_median if not pd.isna(local_median) else mimic_medians.get(feat, 0)
            X_eicu[feat] = df[feat].fillna(fill_val)
        else:
            # 缺失特征兜底
            X_eicu[feat] = mimic_medians.get(feat, 0)

    # 数据分布自检
    inspect_cols = [c for c in ['ph_min', 'creatinine_max', 'pao2fio2ratio_min'] if c in X_eicu.columns]
    if inspect_cols:
        print("\n📊 关键指标最终分布:")
        # 如果是 log 过的，结果会是 log 后的值
        print(X_eicu[inspect_cols].describe().loc[['min', '50%', 'max']])

    # 7. 保存结果
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    eicu_ready_path = os.path.join(SAVE_DIR, f"eicu_for_model_{target}.csv")
    
    if actual_target_col in df.columns:
        df_ready = pd.concat([X_eicu, df[actual_target_col].rename('target')], axis=1)
        df_ready.to_csv(eicu_ready_path, index=False)
        print(f"\n✅ 结局 {target.upper()} 处理成功！样本数: {len(df_ready)}")
    else:
        print(f"❌ 严重错误：找不到结局列 {actual_target_col}")

def run_all_eicu_alignment():
    targets = ['pof', 'composite_outcome', 'mortality_28d']
    for t in targets:
        run_module_07(t)

if __name__ == "__main__":
    run_all_eicu_alignment()
