import os
import json
import joblib
import numpy as np
import pandas as pd

# ===================== 路径配置 =====================
BASE_DIR = "../../"
EICU_RAW_PATH = os.path.join(BASE_DIR, "data/raw/eicu_raw_data.csv")
DICT_PATH = os.path.join(BASE_DIR, "artifacts/features/feature_dictionary.json")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts/scalers")
SAVE_DIR = os.path.join(BASE_DIR, "data/external") 
SELECTED_FEAT_PATH = os.path.join(BASE_DIR, "features/selected_features.json")

os.makedirs(SAVE_DIR, exist_ok=True)

class EICUAuditor:
    def __init__(self, dict_path):
        with open(dict_path, 'r', encoding='utf-8') as f:
            self.feature_dict = json.load(f)

    def apply_clinical_rules(self, df):
        """执行临床审计：单位自动修复 + 生理硬过滤 + 结局对齐 + 亚组标记"""
        df_cleaned = df.copy()
        print(f"\n📋 启动临床空间审计: {df.shape[0]} 行")
        
        # 0. 结局变量重命名对齐
        rename_map = {'mortality_28d': 'mortality', 'composite_outcome': 'composite'}
        df_cleaned = df_cleaned.rename(columns={k: v for k, v in rename_map.items() if k in df_cleaned.columns})

        # --- 新增：亚组标记 (subgroup_no_renal) ---
        if 'creatinine_max' in df_cleaned.columns:
            df_cleaned['subgroup_no_renal'] = (df_cleaned['creatinine_max'] < 1.5).astype(int)
            print("✅ 已计算亚组标记: subgroup_no_renal (Cr < 1.5)")

        print(f"{'Feature':<20} | {'Action':<40} | {'Status'}")
        print("-" * 80)

        for col, config in self.feature_dict.items():
            if col not in df_cleaned.columns or not pd.api.types.is_numeric_dtype(df_cleaned[col]):
                continue
            
            ref = config.get('ref_range', {})
            log_min = ref.get('logical_min')
            log_max = ref.get('logical_max')
            factor = config.get('conversion_factor', 1.0)
            
            series_curr = df_cleaned[col].dropna()
            if series_curr.empty: continue
            
            med = series_curr.median()
            
            # 1. 单位转换探测
            if log_min is not None and factor != 1.0:
                if med < (log_min * 0.2): 
                    df_cleaned[col] = df_cleaned[col] * factor
                    print(f"{col:<20} | Applied unit factor x{factor:<11} | ✅")
            
            # 2. 生理范围硬过滤
            if log_min is not None and log_max is not None:
                mask = (df_cleaned[col] < log_min) | (df_cleaned[col] > log_max)
                if mask.any():
                    df_cleaned.loc[mask, col] = np.nan
                    print(f"{col:<20} | Cleared {mask.sum():>3} physiologic outliers | ⚠️")
        
        return df_cleaned

def generate_eicu_processed(target, df_audited, assets):
    """基于审计后的数据，为特定结局生成模型输入文件"""
    print(f"\n🛠️ 正在对齐目标结局: {target}")

    # 1. 加载特征选择清单
    try:
        with open(SELECTED_FEAT_PATH, 'r', encoding='utf-8') as f:
            selected_all = json.load(f)
        selected_features = selected_all[target]['features']
    except Exception as e:
        print(f"❌ 加载特征清单失败: {e}")
        return
    
    # 2. 强制特征顺序对齐
    imputer = assets['imputer']
    scaler = assets['scaler']
    required_features = imputer.feature_names_in_
    
    df_aligned = pd.DataFrame(index=df_audited.index)
    for col in required_features:
        if col in df_audited.columns:
            df_aligned[col] = df_audited[col]
        else:
            df_aligned[col] = np.nan
            print(f"⚠️ 缺失特征: {col}, 将在插补阶段填充")

    # 3. 应用 Log 变换
    for col in assets['skewed_cols']:
        if col in df_aligned.columns:
            df_aligned[col] = np.log1p(df_aligned[col].clip(lower=0))

    # 4. 复用资产：插补与标准化
    print(f"   应用 MIMIC 资产进行 Transform...")
    df_imputed_raw = imputer.transform(df_aligned)
    df_scaled_raw = scaler.transform(df_imputed_raw)
    
    df_scaled = pd.DataFrame(df_scaled_raw, columns=required_features, index=df_audited.index)

    # --- 新增：标准化后均值检查 ---
    max_mean_abs = df_scaled.mean().abs().max()
    print(f"   📊 标准化后均值检查: {max_mean_abs:.4f} (应接近 0)")

    # 5. 恢复保护列
    protected_cols = ['pof', 'mortality', 'composite', 'gender', 'malignant_tumor', 
                      'mechanical_vent_flag', 'vaso_flag', 'dialysis_flag', 'subgroup_no_renal']
    for col in protected_cols:
        if col in df_audited.columns:
            df_scaled[col] = df_audited[col].fillna(0).astype(int)

    # --- 新增：eICU 对齐后关键特征分布审计 ---
    print(f"\n📈 eICU 对齐后分布审计（{target}）：")
    key_cols = ['creatinine_max', 'lactate_max', 'pao2fio2ratio_min', 'ph_min']
    for col in key_cols:
        if col in df_scaled.columns:
            series = df_scaled[col].dropna()
            print(f"  {col:<20}: Mean={series.mean():.4f} | Std={series.std():.4f}")

    # 6. 保存最终推理集
    save_path = os.path.join(SAVE_DIR, f"eicu_processed_{target}.csv")
    df_scaled.to_csv(save_path, index=False)
    print(f"✅ 完成: {save_path}")

def main():
    print("="*70)
    print("🚀 模块 09: eICU 数据清洗与多结局对齐")
    print("="*70)

    # 1. 【新增】资产加载的错误处理
    try:
        assets = {
            'scaler': joblib.load(os.path.join(ARTIFACT_DIR, "mimic_scaler.joblib")),
            'imputer': joblib.load(os.path.join(ARTIFACT_DIR, "mimic_mice_imputer.joblib")),
            'skewed_cols': joblib.load(os.path.join(ARTIFACT_DIR, "skewed_cols_config.pkl")),
            'bundle': joblib.load(os.path.join(ARTIFACT_DIR, "train_assets_bundle.pkl"))
        }
        print("✅ MIMIC 资产加载成功。")
    except Exception as e:
        print(f"❌ 加载 MIMIC 资产失败: {e}")
        return

    # 2. 初始数据读取
    if not os.path.exists(EICU_RAW_PATH):
        print(f"❌ 错误: 找不到 eICU 原始数据 {EICU_RAW_PATH}")
        return
        
    df_raw = pd.read_csv(EICU_RAW_PATH)
    auditor = EICUAuditor(DICT_PATH)

    # 3. 生成 eicu_raw_scale.csv
    df_audited = auditor.apply_clinical_rules(df_raw)
    
    print("\n✂️ 执行 1%-99% 统计盖帽...")
    numeric_cols = df_audited.select_dtypes(include=[np.number]).columns
    exclude_clipping = ['pof', 'mortality', 'composite', 'gender', 'malignant_tumor', 'subgroup_no_renal']
    for col in numeric_cols:
        if col not in exclude_clipping:
            lower, upper = df_audited[col].quantile(0.01), df_audited[col].quantile(0.99)
            if pd.notnull(lower) and pd.notnull(upper):
                df_audited[col] = df_audited[col].clip(lower, upper)

    scale_save_path = os.path.join(SAVE_DIR, "eicu_raw_scale.csv")
    df_audited.to_csv(scale_save_path, index=False)
    print(f"\n⭐ 临床审计版已生成 (用于 Table 1/漂移分析): {scale_save_path}")

    # 4. 生成 Z-score 变换推理版
    for target in ['pof', 'mortality', 'composite']:
        generate_eicu_processed(target, df_audited, assets)

    print("\n" + "="*70)
    print("✅ 模块 09 运行成功！已准备好统计与预测两套数据。")

if __name__ == "__main__":
    main()
