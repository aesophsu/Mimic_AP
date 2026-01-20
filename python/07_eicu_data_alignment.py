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
    print("\n" + "="*75)
    print(f"🚀 模块 07: eICU 多中心对齐与量纲修正 (结局: {target.upper()})")
    print("="*75)

    # 1. 加载 MIMIC 训练阶段资产
    assets_path = os.path.join(MODELS_DIR, f"train_assets_{target}.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"scaler_{target}.pkl") 
    
    if not os.path.exists(assets_path) or not os.path.exists(scaler_path):
        print(f"❌ 错误：找不到结局 {target} 的资产或 Scaler 文件。")
        return
        
    train_assets = joblib.load(assets_path)
    scaler = joblib.load(scaler_path)
    
    selected_features = train_assets['selected_features']
    mimic_medians = train_assets['medians']
    skewed_cols_to_log = train_assets['skewed_cols']

    if not os.path.exists(RAW_EICU_PATH):
        print(f"❌ 错误：找不到 eICU 原始数据 {RAW_EICU_PATH}")
        return
    
    df = pd.read_csv(RAW_EICU_PATH)

    # 2. 结局列映射
    target_col_map = {'pof': 'pof', 'composite_outcome': 'composite_outcome', 'mortality_28d': 'mortality_28d'}
    actual_target_col = target_col_map.get(target)

    # 3. 基础清洗
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'M': 1, 'F': 0, 1: 1, 0: 0}).fillna(1)
    
    # P/F Ratio 缺失值填充 (临床正常值 400)
    if 'pao2fio2ratio_min' in df.columns:
        df['pao2fio2ratio_min'] = df['pao2fio2ratio_min'].fillna(400)

    # 4. 🧪 核心优化：动态 Log1p 变换 (自适应阈值补丁)
    print(f"\n🪄 [1/4] 执行对数补丁对齐 (Log1p Alignment)...")
    # 针对 SQL 调整后的量纲，将阈值设为 3 更加稳健（覆盖 BUN, WBC, 转氨酶等）
    for col in skewed_cols_to_log:
        if col in df.columns and col != 'ph_min': # pH 严禁 Log
            current_med = df[col].median()
            # 只有中值大于 3 且在训练集偏态名单中才进行 Log
            if current_med > 3:
                df[col] = np.log1p(df[col].astype(float).clip(lower=0))
                print(f"    ✅ 已对 {col} 完成 Log1p (当前中值: {current_med:.2f})")
            else:
                print(f"    ℹ️ 跳过 {col} (中值 {current_med:.2f} 已在对数尺度或量级较小)")
    # 5. 构建特征矩阵 (修正：构建全维度矩阵以匹配 Scaler)
    print(f"\n🛠️ [2/4] 构建全维度特征矩阵以匹配 Scaler (预期特征数: {scaler.n_features_in_})...")

    X_eicu_full = pd.DataFrame(0.0, index=df.index, columns=range(scaler.n_features_in_))
    
    all_features_at_train = train_assets.get('all_features_before_lasso') 
    
    if all_features_at_train is None:
        print("❌ 错误：train_assets 中缺少 'all_features_before_lasso'。")
        print("💡 请确保模块 03 在保存 train_assets 时包含了所有进入 scaler 的原始列名。")
        return

    X_eicu_templated = pd.DataFrame(index=df.index)
    for feat in all_features_at_train:
        if feat in df.columns:
            # 局部中值填充
            local_med = df[feat].median()
            fill_val = local_med if not pd.isna(local_med) else mimic_medians.get(feat, 0)
            X_eicu_templated[feat] = df[feat].fillna(fill_val)
        else:
            # 缺失列补 0 或训练集的中值
            X_eicu_templated[feat] = mimic_medians.get(feat, 0)

    # 6. 归一化对齐
    print(f"\n⚖️ [3/4] 应用全维度归一化...")
    # 确保列顺序完全一致
    X_eicu_templated = X_eicu_templated[all_features_at_train]
    X_eicu_std_all = scaler.transform(X_eicu_templated)
    X_eicu_std_df = pd.DataFrame(X_eicu_std_all, columns=all_features_at_train, index=df.index)
    
    # 7. 关键步骤：最后只提取 LASSO 选出的那 12 个特征
    X_eicu_final_features = X_eicu_std_df[selected_features]
    print(f"    ✅ 已从 63 维标准化数据中提取出 {len(selected_features)} 个 LASSO 特征")

    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    eicu_ready_path = os.path.join(SAVE_DIR, f"eicu_for_model_{target}.csv")
    
    if actual_target_col in df.columns:
        df_ready = pd.concat([X_eicu_final_features, df[actual_target_col].rename('target')], axis=1)
        df_ready.to_csv(eicu_ready_path, index=False)
        print(f"\n✅ 结局 {target.upper()} 处理成功！样本数: {len(df_ready)}")
        print(f"📍 最终文件位置: {eicu_ready_path}")
        
        # 打印关键特征标准化后的统计信息，确保没有异常值
        check_cols = [c for c in ['bun_max', 'creatinine_max', 'wbc_max'] if c in X_eicu_final_features.columns]
        if check_cols:
            print("\n📊 关键特征标准化后分布审计:")
            print(X_eicu_final_features[check_cols].describe().loc[['mean', 'std', 'min', 'max']])
    else:
        print(f"❌ 严重错误：找不到结局列 {actual_target_col}")

def run_all_eicu_alignment():
    targets = ['pof', 'composite_outcome', 'mortality_28d']
    for t in targets:
        run_module_07(t)

if __name__ == "__main__":
    run_all_eicu_alignment()
