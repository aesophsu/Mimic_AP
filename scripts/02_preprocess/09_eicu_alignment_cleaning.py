import os
import json
import joblib
import numpy as np
import pandas as pd

# =========================================================
# 1. 配置与路径 (遵循 14 步标准目录树)
# =========================================================
BASE_DIR = "../../"
RAW_EICU_PATH = os.path.join(BASE_DIR, "data/raw/eicu_raw_data.csv")  # 08步 SQL 产物
EXTERNAL_DIR = os.path.join(BASE_DIR, "data/external")
MODEL_ROOT = os.path.join(BASE_DIR, "artifacts/models")
SCALER_ROOT = os.path.join(BASE_DIR, "artifacts/scalers")

OUTCOMES = ['pof', 'mortality_28d', 'composite_outcome']

for path in [EXTERNAL_DIR]:
    if not os.path.exists(path):
        os.makedirs(path)

# =========================================================
# 2. 核心清洗逻辑 (物理对齐)
# =========================================================
def physical_alignment(df_raw):
    """
    实现功能 1: 物理尺度对齐
    处理性别、极端异常值、单位统一，保留物理原值。
    """
    df = df_raw.copy()
    print(f"🛠️ [1/4] 执行物理对齐 (Physical Alignment)...")

    # 1.1 性别统一编码 (M/1, F/0)
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'M': 1, 'F': 0, 'Male': 1, 'Female': 0, 1: 1, 0: 0}).fillna(1)
    
    # 1.2 处理 PF Ratio (生理上限锁定)
    if 'pao2fio2ratio_min' in df.columns:
        df['pao2fio2ratio_min'] = df['pao2fio2ratio_min'].clip(lower=20, upper=800).fillna(400)

    # 1.3 处理 pH 极端异常 (根据 SQL 已有的 6.7-7.8 进一步确认)
    ph_cols = [c for c in df.columns if 'ph' in c.lower()]
    for col in ph_cols:
        df[col] = df[col].clip(lower=6.7, upper=7.8)

    # 保存物理原值版 (用于 Step 10: Table 1)
    save_path = os.path.join(EXTERNAL_DIR, "eicu_aligned.csv")
    df.to_csv(save_path, index=False)
    print(f"  ✅ 物理原值资产已保存: {save_path}")
    return df

# =========================================================
# 3. 结局专属处理 (Log1p & Standardization)
# =========================================================
def process_outcome_alignment(df_aligned):
    """
    实现功能 2 & 3: 特征工程投影与标准化映射
    """
    # 加载全局对数配置 (03步产出)
    skewed_config_path = os.path.join(SCALER_ROOT, "skewed_cols_config.pkl")
    skewed_cols = joblib.load(skewed_config_path) if os.path.exists(skewed_config_path) else []

    for target in OUTCOMES:
        print(f"\n🚀 正在处理结局映射: [{target.upper()}]")
        
        # 3.1 加载 06 步保存的资产 Bundle
        bundle_path = os.path.join(SCALER_ROOT, f"train_assets_bundle_{target}.pkl")
        scaler_path = os.path.join(MODEL_ROOT, target, "scaler.pkl")
        
        if not (os.path.exists(bundle_path) and os.path.exists(scaler_path)):
            print(f"  ⚠️ 跳过 {target}: 找不到 Bundle 或 Scaler 资产。")
            continue

        bundle = joblib.load(bundle_path)
        scaler = joblib.load(scaler_path)
        selected_features = bundle['feature_order']  # 训练时的特征顺序
        
        # 3.2 复制副本进行建模预处理
        df_target = df_aligned.copy()

        # 3.3 Log1p Alignment (对数补丁)
        print(f"🪄 [2/4] 执行对数补丁对齐...")
        for col in skewed_cols:
            if col in df_target.columns and col in selected_features and col != 'ph_min':
                # 临床启发式校验：只有中值较大时才执行 Log（同步 02/03 步逻辑）
                if df_target[col].median() > 3:
                    df_target[col] = np.log1p(df_target[col].astype(float).clip(lower=0))

        # 3.4 简化特征矩阵构建 (不再寻找 templated)
        print(f"🛠️ [3/4] 构建选定的 {len(selected_features)} 个特征矩阵...")
        X_eicu_final = pd.DataFrame(index=df_target.index)
        for feat in selected_features:
            if feat in df_target.columns:
                # 缺失值处理：使用 eICU 局部中位数填充
                local_med = df_target[feat].median()
                X_eicu_final[feat] = df_target[feat].fillna(local_med if not pd.isna(local_med) else 0)
            else:
                print(f"  ⚠️ 警告: eICU 缺少特征 [{feat}]，填充 0")
                X_eicu_final[feat] = 0
        
        # 严格对齐特征列顺序
        X_eicu_final = X_eicu_final[selected_features]

        # 3.5 标准化映射 (Standard Scaling)
        print(f"⚖️ [4/4] 应用标准化投影 (Scaler Transformation)...")
        X_eicu_std_all = scaler.transform(X_eicu_final)
        X_eicu_final_features = pd.DataFrame(X_eicu_std_all, columns=selected_features, index=df_target.index)

        # 3.6 合并标签并保存
        target_csv_path = os.path.join(EXTERNAL_DIR, f"eicu_processed_{target}.csv")
        if target in df_target.columns:
            df_final_to_save = pd.concat([X_eicu_final_features, df_target[target].reset_index(drop=True)], axis=1)
            df_final_to_save.to_csv(target_csv_path, index=False)
            print(f"✅ 处理成功！建模张量已保存: {target_csv_path} (Shape: {df_final_to_save.shape})")

# =========================================================
# 4. 主程序入口
# =========================================================
def main():
    if not os.path.exists(RAW_EICU_PATH):
        print(f"❌ 错误: 找不到 eICU 原始数据 {RAW_EICU_PATH}，请确认 08 步 SQL 已运行并导出。")
        return

    print("📖 正在加载 eICU 原始导出数据...")
    df_raw = pd.read_csv(RAW_EICU_PATH)
    
    # 第一步：物理对齐与 Table 1 原始数据准备
    df_aligned = physical_alignment(df_raw)
    
    # 第二步：针对各结局进行 Log 和 Scale 投影
    process_outcome_alignment(df_aligned)
    
    print("\n✨ 09 步任务圆满完成！eICU 验证集已完全就绪。")

if __name__ == "__main__":
    main()
