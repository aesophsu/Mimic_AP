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

def run_module_07(df_input, target='pof'):
    """
    优化版模块 07: 
    1. 接收预载的 df_input 以节省内存。
    2. 使用 df.copy() 确保模型预处理（Log/Scale）不污染原始数据。
    3. 返回原始 df_input 以供后续 Table 1 聚合。
    """
    print("\n" + "="*75)
    print(f"🚀 模块 07: eICU 多中心对齐与量纲修正 (结局: {target.upper()})")
    print("="*75)
    
    # 【优化 1】内存保护：使用副本进行后续所有破坏性操作（Log/Standardization）
    df = df_input.copy()

    # 1. 加载 MIMIC 训练阶段资产
    assets_path = os.path.join(MODELS_DIR, f"train_assets_{target}.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"scaler_{target}.pkl") 
    
    if not os.path.exists(assets_path) or not os.path.exists(scaler_path):
        print(f"❌ 错误：找不到结局 {target} 的资产或 Scaler 文件。")
        return None, None
        
    train_assets = joblib.load(assets_path)
    scaler = joblib.load(scaler_path)
    
    selected_features = train_assets['selected_features']
    mimic_medians = train_assets['medians']
    skewed_cols_to_log = train_assets['skewed_cols']

    # 2. 结局列映射
    target_col_map = {'pof': 'pof', 'composite_outcome': 'composite_outcome', 'mortality_28d': 'mortality_28d'}
    actual_target_col = target_col_map.get(target)

    # 3. 基础清洗 (仅在副本上执行)
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'M': 1, 'F': 0, 1: 1, 0: 0}).fillna(1)
    
    if 'pao2fio2ratio_min' in df.columns:
        df['pao2fio2ratio_min'] = df['pao2fio2ratio_min'].fillna(400)

    # 4. 🧪 核心优化：动态 Log1p 变换 (副本上操作，保护 df_input)
    print(f"\n🪄 [1/4] 执行对数补丁对齐 (Log1p Alignment)...")
    for col in skewed_cols_to_log:
        if col in df.columns and col != 'ph_min': 
            current_med = df[col].median()
            if current_med > 3:
                df[col] = np.log1p(df[col].astype(float).clip(lower=0))
                print(f"    ✅ 已对 {col} 完成 Log1p (当前中值: {current_med:.2f})")
            else:
                print(f"    ℹ️ 跳过 {col} (中值 {current_med:.2f} 较小，无需 Log)")

    # 5. 构建特征矩阵
    print(f"\n🛠️ [2/4] 构建全维度特征矩阵以匹配 Scaler...")
    all_features_at_train = train_assets.get('all_features_before_lasso') 
    
    if all_features_at_train is None:
        print("❌ 错误：train_assets 中缺少 'all_features_before_lasso'。")
        return None, None

    X_eicu_templated = pd.DataFrame(index=df.index)
    for feat in all_features_at_train:
        if feat in df.columns:
            local_med = df[feat].median()
            fill_val = local_med if not pd.isna(local_med) else mimic_medians.get(feat, 0)
            X_eicu_templated[feat] = df[feat].fillna(fill_val)
        else:
            X_eicu_templated[feat] = mimic_medians.get(feat, 0)

    # 6. 归一化对齐
    print(f"\n⚖️ [3/4] 应用全维度归一化...")
    X_eicu_templated = X_eicu_templated[all_features_at_train]
    X_eicu_std_all = scaler.transform(X_eicu_templated)
    X_eicu_std_df = pd.DataFrame(X_eicu_std_all, columns=all_features_at_train, index=df.index)
    
    # 7. 提取 LASSO 核心特征
    X_eicu_final_features = X_eicu_std_df[selected_features]

    # 保存模型输入文件
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    eicu_ready_path = os.path.join(SAVE_DIR, f"eicu_for_model_{target}.csv")
    
    if actual_target_col in df.columns:
        df_ready = pd.concat([X_eicu_final_features, df[actual_target_col].rename('target')], axis=1)
        df_ready.to_csv(eicu_ready_path, index=False)
        print(f"\n✅ 结局 {target.upper()} 处理成功！样本数: {len(df_ready)}")
        
        # 审计关键特征
        check_cols = [c for c in ['bun_max', 'creatinine_max', 'wbc_max'] if c in X_eicu_final_features.columns]
        if check_cols:
            print("\n📊 关键特征标准化后审计:")
            print(X_eicu_final_features[check_cols].describe().loc[['mean', 'std', 'min', 'max']])
        
        # 【优化 2】返回：1. 原始输入的 df_input (未 Log), 2. 本次选出的特征
        return df_input, selected_features
    else:
        print(f"❌ 严重错误：找不到结局列 {actual_target_col}")
        return None, None
        
def generate_global_table1_data(snapshots):
    """
    聚合三结局的原始物理尺度数据，生成 Table 1 专用文件。
    snapshots: {target_name: (df_raw_original, selected_features_list)}
    """
    print("\n" + "="*60)
    print("📦 正在执行全局数据聚合 (Table 1 专用)...")
    print("="*60)
    
    global_df = None
    
    for target, (df_raw, features) in snapshots.items():
        # 1. 确定当前结局涉及的列清单 (人口学基础变量 + 结局标签 + 模型特征)
        essential_cols = ['admission_age', 'gender', 'bmi', target] + list(features)
        available_cols = [c for c in essential_cols if c in df_raw.columns]
        
        # 2. 提取子集并【重置索引】，这是防止 concat 错位的核心安全操作
        current_subset = df_raw[available_cols].copy().reset_index(drop=True)
        
        if global_df is None:
            # 第一次循环：初始化 global_df
            global_df = current_subset
            print(f"   ✅ 初始化聚合表 (结局: {target})")
        else:
            # 3. 后续循环：只合并新出现的特征列或结局列
            new_cols = [c for c in current_subset.columns if c not in global_df.columns]
            if new_cols:
                # 显式使用横向合并 axis=1
                global_df = pd.concat([global_df, current_subset[new_cols]], axis=1)
                print(f"   ✅ 合并新变量 (来自结局: {target}): {len(new_cols)} 个")

    # 4. 最终保存
    save_path = os.path.join(SAVE_DIR, "eicu_for_table1.csv")
    if global_df is not None:
        global_df.to_csv(save_path, index=False)
        print("-" * 60)
        print(f"✨ 聚合成功！跨库审计专用原始数据已生成。")
        print(f"📍 文件位置: {save_path}")
        print(f"📊 最终表格维度: {global_df.shape[0]} 行 x {global_df.shape[1]} 列")
        print("-" * 60)

def run_all_eicu_alignment():
    """
    主程序入口：执行内存优化后的预处理管道
    """
    # 【内存优化核心】：全脚本仅在此处执行一次磁盘读取
    if not os.path.exists(RAW_EICU_PATH):
        print(f"❌ 错误：找不到 eICU 原始数据文件: {RAW_EICU_PATH}")
        return
        
    print(f"📖 正在加载 eICU 原始大数据集 (Memory-Optimized Loader)...")
    try:
        df_raw_master = pd.read_csv(RAW_EICU_PATH)
    except Exception as e:
        print(f"❌ 读取 CSV 失败: {e}")
        return

    targets = ['pof', 'composite_outcome', 'mortality_28d']
    snapshots = {} 
    
    # 执行多结局预处理循环
    for t in targets:
        # 将 master 数据副本传入，保护原始数据
        result = run_module_07(df_raw_master, target=t)
        if result is not None:
            # result 包含了 (df_input, selected_features)
            snapshots[t] = result
            
    # 如果有任何结局处理成功，则执行聚合
    if snapshots:
        generate_global_table1_data(snapshots)
    else:
        print("⚠️ 警告: 没有成功生成任何结局的快照，跳过聚合。")

if __name__ == "__main__":
    run_all_eicu_alignment()
