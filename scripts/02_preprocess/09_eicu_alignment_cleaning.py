import os
import json
import joblib
import numpy as np
import pandas as pd

# ===================== 路径配置 (Paths) =====================
BASE_DIR = "../../"
EICU_RAW_PATH = os.path.join(BASE_DIR, "data/raw/eicu_raw_data.csv")
DICT_PATH = os.path.join(BASE_DIR, "artifacts/features/feature_dictionary.json")
SELECTED_FEAT_PATH = os.path.join(BASE_DIR, "artifacts/features/selected_features.json")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts/scalers")
SAVE_DIR = os.path.join(BASE_DIR, "data/external")

# ===================== 全局常量 (Constants) =====================
# 保护列：不参与 Z-score 变换，需在最后恢复的原样列
PROTECTED_COLS = [
    'pof', 'mortality', 'composite', 'gender', 'malignant_tumor', 
    'mechanical_vent_flag', 'vaso_flag', 'dialysis_flag', 'subgroup_no_renal'
]

# 核心保留列：审计阶段必须保留的列（包含 ID 和结局变量）
ESSENTIAL_COLS = ['patientunitstayid', 'uniquepid', 'los'] + PROTECTED_COLS

# 盖帽排除列：统计盖帽 (Winsorization) 时不需要处理的列
EXCLUDE_CLIPPING = ['patientunitstayid', 'uniquepid', 'los'] + PROTECTED_COLS

os.makedirs(SAVE_DIR, exist_ok=True)

def audit_clinical_limits(df, feature_dict):
    """生理规则审计：执行单位换算与物理极限剔除"""
    df_temp = df.copy()
    for col, config in feature_dict.items():
        if col not in df_temp.columns or not pd.api.types.is_numeric_dtype(df_temp[col]):
            continue
        
        ref = config.get('ref_range', {})
        log_min, log_max = ref.get('logical_min'), ref.get('logical_max')
        factor = config.get('conversion_factor', 1.0)
        
        series_valid = df_temp[col].dropna()
        if series_valid.empty: continue
        
        # 1. 自动单位修复：检测到中值远低于逻辑下限时应用转换系数
        if log_min is not None and factor != 1.0:
            if series_valid.median() < (log_min * 0.2): 
                df_temp[col] *= factor
        
        # 2. 极限值清理：超出医学逻辑范围的数据置为 NaN
        if log_min is not None and log_max is not None:
            mask = (df_temp[col] < log_min) | (df_temp[col] > log_max)
            df_temp.loc[mask, col] = np.nan
            
    return df_temp

def apply_clinical_audit_workflow(df, auditor_config):
    """主审计流：执行列名对齐、特征过滤、亚组识别及生理审计"""
    # 1. 结局变量对齐
    df = df.rename(columns={'mortality_28d': 'mortality', 'composite_outcome': 'composite'})

    # 2. 临床白名单过滤：保留字典特征 + 核心配置列
    allowed_cols = list(auditor_config.keys()) + ESSENTIAL_COLS
    df_cleaned = df[[c for c in allowed_cols if c in df.columns]].copy()
    
    print(f"📋 审计启动: 原始 {df.shape[1]} 列 -> 目标 {df_cleaned.shape[1]} 列")

    # 3. 衍生变量计算：识别非肾损伤亚组 (Scr < 1.5)
    if 'creatinine_max' in df_cleaned.columns:
        df_cleaned['subgroup_no_renal'] = (df_cleaned['creatinine_max'] < 1.5).astype(int)

    # 4. 执行生理规则校验
    return audit_clinical_limits(df_cleaned, auditor_config)

def load_mimic_assets():
    """资产加载：获取 MIMIC 训练阶段保存的 Scaler、Imputer 及偏态配置"""
    try:
        assets = {
            'scaler': joblib.load(os.path.join(ARTIFACT_DIR, "mimic_scaler.joblib")),
            'imputer': joblib.load(os.path.join(ARTIFACT_DIR, "mimic_mice_imputer.joblib")),
            'skewed_cols': joblib.load(os.path.join(ARTIFACT_DIR, "skewed_cols_config.pkl"))
        }
        print(f"✅ MIMIC 资产加载成功 (含 {len(assets['skewed_cols'])} 个偏态特征配置)")
        return assets
    except Exception as e:
        print(f"❌ 资产加载失败: {e}"); return None

def get_union_feature_config():
    """特征并集提取：整合多结局所需的特征清单，并匹配生理审计规则"""
    if not (os.path.exists(SELECTED_FEAT_PATH) and os.path.exists(DICT_PATH)):
        print("❌ 错误: 配置文件缺失"); return None

    # 加载结局特征清单和生理规则字典
    with open(SELECTED_FEAT_PATH, 'r', encoding='utf-8') as f:
        selected_json = json.load(f)
    with open(DICT_PATH, 'r', encoding='utf-8') as f:
        full_physio_dict = json.load(f)

    # 提取所有结局涉及的特征并集
    union_features = {feat for target in selected_json.values() for feat in target['features']}
    
    # 仅保留并集特征的审计规则
    auditor_config = {k: v for k, v in full_physio_dict.items() if k in union_features}
    print(f"🎯 识别到 {len(union_features)} 个唯一特征，已匹配审计规则")
    return auditor_config

def run_clinical_audit(df_raw, auditor_config):
    """临床审计执行：应用生理规则过滤，并执行 1%-99% 统计盖帽"""
    # 1. 调用之前定义的函数式审计流
    df_audited = apply_clinical_audit_workflow(df_raw, auditor_config)
    
    # 2. 统计盖帽 (Winsorization)：消除 1% 极值波动
    clinical_features = [c for c in df_audited.columns if c in auditor_config and c not in EXCLUDE_CLIPPING]
    
    clipped_count = 0
    for col in clinical_features:
        q = df_audited[col].quantile([0.01, 0.99])
        if q.isnull().any(): continue
        
        df_audited[col] = df_audited[col].clip(lower=q[0.01], upper=q[0.99])
        clipped_count += 1
    
    print(f"✅ 盖帽处理完成: {clipped_count} 个临床特征已约束")
    return df_audited

def align_feature_space(df_audited, required_features):
    """空间对齐：确保 eICU 列顺序与 MIMIC 训练时的 imputer 期待完全一致"""
    df_aligned = pd.DataFrame(index=df_audited.index)
    for col in required_features:
        if col in df_audited.columns:
            # 解决重名列防御：若存在重名列则取第一列
            val = df_audited[col]
            df_aligned[col] = val.iloc[:, 0] if isinstance(val, pd.DataFrame) else val
        else:
            # eICU 完全缺失的列填 NaN，后续由 MICE 插补
            df_aligned[col] = np.nan
    return df_aligned

def apply_mimic_transform(df_aligned, assets):
    """变换同步：执行 Log 变换、MICE 插补与 Z-score 标准化"""
    imputer, scaler, skewed_cols = assets['imputer'], assets['scaler'], assets['skewed_cols']
    df_trans = df_aligned.copy()
    
    # 1. 偏态特征同步 Log1p 变换
    for col in skewed_cols:
        if col in df_trans.columns:
            df_trans[col] = np.log1p(df_trans[col].clip(lower=0))
    
    # 2. 执行 MIMIC 沉淀的 Transform 管道
    print("   执行 Transform (MICE + Scaler)...")
    imputed_data = imputer.transform(df_trans)
    return pd.DataFrame(
        scaler.transform(imputed_data),
        columns=imputer.feature_names_in_,
        index=df_aligned.index
    )

def audit_final_distribution(df):
    """分布审计：检查变换后 Z-score 是否接近标准分布 (Mean=0, Std=1)"""
    print("\n📈 关键特征 (Z-score) 审计:")
    for col in ['creatinine_max', 'lactate_max', 'pao2fio2ratio_min', 'ph_min']:
        if col in df.columns:
            print(f"   {col:<20}: Mean={df[col].mean():.3f} | Std={df[col].std():.3f}")

def generate_eicu_processed(target, df_audited, assets):
    """结局生成：针对特定 outcome 生成对齐后的推理数据集"""
    print(f"\n🛠️ 处理结局: {target}")
    
    # 1. 加载对应结局的特征清单 (使用 SELECTED_FEAT_PATH)
    try:
        with open(SELECTED_FEAT_PATH, 'r') as f:
            selected_features = json.load(f)[target]["features"]
    except Exception as e:
        print(f"❌ 加载特征清单失败: {e}"); return

    # 2. 对齐特征空间并执行 MIMIC 转换
    df_aligned = align_feature_space(df_audited, assets['imputer'].feature_names_in_)
    df_scaled = apply_mimic_transform(df_aligned, assets)

    # 3. 筛选特征并恢复保护列 (Labels & Flags)
    df_final = df_scaled[[f for f in selected_features if f in df_scaled.columns]].copy()
    for col in PROTECTED_COLS:
        if col in df_audited.columns:
            source = df_audited[col]
            df_final[col] = (source.iloc[:, 0] if isinstance(source, pd.DataFrame) else source).fillna(0).astype(int).values

    # 4. 分布审计与导出
    audit_final_distribution(df_final)
    save_path = os.path.join(SAVE_DIR, f"eicu_processed_{target}.csv")
    df_final.to_csv(save_path, index=False)
    print(f"✅ 保存推理集: {save_path}")

def main():
    print("="*60)
    print("🚀 Module 09: eICU Preprocessing & Outcome Alignment")
    print("="*60)

    # 1. 初始化：加载 MIMIC 训练资产与并集特征规则
    assets = load_mimic_assets()
    auditor_config = get_union_feature_config()
    if not assets or not auditor_config:
        return

    # 2. 数据读取：加载 eICU 原始队列
    if not os.path.exists(EICU_RAW_PATH):
        print(f"❌ Error: Raw data not found at {EICU_RAW_PATH}")
        return
    
    df_raw = pd.read_csv(EICU_RAW_PATH)
    print(f"📦 Loaded eICU raw data: {df_raw.shape[0]} patients")
    
    # 3. 临床预处理：执行生理审计、单位修复及 1%-99% 盖帽
    df_audited = run_clinical_audit(df_raw, auditor_config)

    # 4. 存档中间态：保存未标准化但已清洗的临床版本（便于后续验证）
    scale_save_path = os.path.join(SAVE_DIR, "eicu_raw_scale.csv")
    df_audited.to_csv(scale_save_path, index=False)
    print(f"⭐ Clinical audited data saved: {scale_save_path}")

    # 5. 多结局对齐循环：生成各结局专属的 Z-score 处理集
    print("\n🔄 Starting alignment & Z-score transformation...")
    for target in ['pof', 'mortality', 'composite']:
        generate_eicu_processed(target, df_audited, assets)

    print("\n" + "="*60)
    print("✅ Module 09 Pipeline Completed Successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
