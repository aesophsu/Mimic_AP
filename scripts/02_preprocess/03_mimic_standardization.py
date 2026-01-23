import os
import joblib
import json
import numpy as np
import pandas as pd
from tableone import TableOne
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

BASE_DIR = "../../"
INPUT_PATH = os.path.join(BASE_DIR, "data/cleaned/mimic_raw_scale.csv")
SAVE_DIR = os.path.join(BASE_DIR, "data/cleaned")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts/scalers")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "mimic_scaler.joblib")
IMPUTER_PATH = os.path.join(ARTIFACT_DIR, "mimic_mice_imputer.joblib")
SKEW_CONFIG_PATH = os.path.join(ARTIFACT_DIR, "skewed_cols_config.pkl")
REPORT_DIR = os.path.join(BASE_DIR, "results/tables")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

def run_mimic_standardization():
    print("="*70)
    print("启动模块 03: Log 转换、MICE 插补与标准化 (MIMIC-IV)")
    print("="*70)

    if not os.path.exists(INPUT_PATH):
        print(f"输入文件不存在: {INPUT_PATH}")
        return

    try:
        with open(os.path.join(BASE_DIR, "artifacts/features/feature_dictionary.json"), 'r', encoding='utf-8') as f:
            feat_dict = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"特征字典读取失败: {e}")
        return

    df = pd.read_csv(INPUT_PATH)

    # 1. 亚组标记
    if 'creatinine_max' in df.columns and 'chronic_kidney_disease' in df.columns:
        df['subgroup_no_renal'] = (
            (df['creatinine_max'] < 1.5) & (df['chronic_kidney_disease'] == 0)
        ).astype(int)

    # 2. gender 已由 02 步完成，这里保留仅作保险（可选删除）
    if 'gender' in df.columns:
        df['gender'] = df['gender'].replace(['M', 'Male', 'MALE', 1, 1.0], 1)\
                                   .replace(['F', 'Female', 'FEMALE', 0, 0.0], 0)\
                                   .fillna(df['gender'].mode()[0] if not df['gender'].dropna().empty else 0)\
                                   .astype(int)

    # 3. TableOne 基线表（物理尺度）
    clinical_features = [
        'admission_age', 'weight_admit', 'gender',                    # 人口统计学
        'sofa_score', 'apsiii', 'sapsii', 'oasis', 'lods',            # 严重程度评分 (全套)
        'heart_failure', 'chronic_kidney_disease', 'malignant_tumor', # 共病 (既往史)
        'mechanical_vent_flag', 'vaso_flag',                          # 治疗干预 (Flag)
        'wbc_max', 'hemoglobin_min', 'platelets_min',                 # 血常规
        'bun_max', 'creatinine_max', 'bilirubin_total_max',           # 肾功能与肝功能
        'alt_max', 'ast_max', 'alp_max',                              # 肝损害指标
        'lactate_max', 'pao2fio2ratio_min', 'spo2_min', 'ph_min',     # 灌注、呼吸与酸碱
        'sodium_max', 'potassium_max', 'bicarbonate_min'              # 电解质与代谢
    ]
    outcome_cols = ['pof', 'mortality', 'composite', 'subgroup_no_renal']
    cols_for_table = [c for c in (clinical_features + outcome_cols) if c in df.columns]
    categorical = [
        c for c in [
            'gender', 'heart_failure', 'chronic_kidney_disease', 'malignant_tumor',
            'mechanical_vent_flag', 'vaso_flag', 'mortality', 'composite', 
            'subgroup_no_renal'
        ] if c in cols_for_table
    ]
    nonnormal = [c for c in cols_for_table if c not in categorical]

    t1 = TableOne(df, columns=cols_for_table, categorical=categorical, nonnormal=nonnormal, groupby='pof', pval=True, htest_name=True)
    t1.to_csv(os.path.join(REPORT_DIR, "table1_baseline.csv"))

    if 'subgroup_no_renal' in df.columns:
        cols_for_t2 = [c for c in cols_for_table if c != 'subgroup_no_renal']
        cat_for_t2 = [c for c in categorical if c != 'subgroup_no_renal']
        nonnormal_for_t2 = [c for c in nonnormal if c != 'subgroup_no_renal']

        t2 = TableOne(df, columns=cols_for_t2, categorical=cat_for_t2, nonnormal=nonnormal_for_t2, groupby='subgroup_no_renal', pval=True)
        t2.to_csv(os.path.join(REPORT_DIR, "table2_renal_subgroup.csv"))
    
    print(f"{'Feature Name':<25} | {'Missing%':<10} | {'Median':<10} | {'Mean':<10} | {'Max':<10}")
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            series = df[col].dropna()
            missing = df[col].isnull().mean() * 100
            med = series.median() if not series.empty else 0
            mean = series.mean() if not series.empty else 0
            v_max = series.max() if not series.empty else 0
            print(f"{col:<25} | {missing:>8.2f}% | {med:>10.2f} | {mean:>10.2f} | {v_max:>10.2f}")

    drop_from_modeling = [
        'subject_id', 'hadm_id', 'stay_id', 
        'admittime', 'dischtime', 'intime', 'deathtime', 'dod',
        'early_death_24_48h', 'hosp_mortality', 'los'
    ]
    
    protected_cols = [
        'pof', 'resp_pof', 'cv_pof', 'renal_pof', 
        'mortality', 'composite', 'subgroup_no_renal',
        'gender', 'heart_failure', 'chronic_kidney_disease', 
        'malignant_tumor', 'mechanical_vent_flag', 'vaso_flag'
    ]

    df_model = df.drop(columns=[c for c in drop_from_modeling if c in df.columns])
    for col in protected_cols:
        if col in df_model.columns:
            df_model[col] = df_model[col].fillna(0).astype(int)

    remaining_text = df_model.select_dtypes(include=['object']).columns.tolist()
    if remaining_text:
        df_model = df_model.drop(columns=remaining_text)

    numeric_features = [c for c in df_model.select_dtypes(include=[np.number]).columns
                        if c not in protected_cols]

    skewed_cols = [
        col for col in numeric_features
        if col in feat_dict and feat_dict[col].get("needs_log_transform", False)
    ]

    if skewed_cols:
        print(f"\n对以下 {len(skewed_cols)} 个特征应用 log1p（保留 NaN 让 MICE 处理）：")
        print(", ".join(skewed_cols))
        for col in skewed_cols:
            df_model[col] = np.log1p(df_model[col].clip(lower=0))

    joblib.dump(skewed_cols, SKEW_CONFIG_PATH)

    raw_medians = df_model[numeric_features].median().to_dict()  # Log 后、标准化前的中位数

    # ==================== MICE 插补 ====================
    print("\n开始 MICE 多重插补（将在 Log 空间进行）...")
    imputer = IterativeImputer(max_iter=15, random_state=42, verbose=2)
    df_model[numeric_features] = imputer.fit_transform(df_model[numeric_features])
    print(f"MICE 完成，实际迭代次数：{imputer.n_iter_}")
    joblib.dump(imputer, IMPUTER_PATH)

    # ==================== 标准化 ====================
    print("开始标准化（StandardScaler）...")
    scaler = StandardScaler()
    df_model[numeric_features] = scaler.fit_transform(df_model[numeric_features])
    joblib.dump(scaler, SCALER_PATH)

    # ==================== 打包训练资产 ====================
    train_assets = {
        'skewed_cols': skewed_cols,
        'medians': raw_medians,        # Log 后、标准化前的真实中位数（最实用）
        'feature_order': numeric_features,
        'n_samples': len(df)
    }
    bundle_path = os.path.join(ARTIFACT_DIR, "train_assets_bundle.pkl")
    joblib.dump(train_assets, bundle_path)

    # ==================== 保存最终张量 ====================
    processed_path = os.path.join(SAVE_DIR, "mimic_processed.csv")
    df_model.to_csv(processed_path, index=False)


    # ==================== 控制台输出检查 (Audit) ====================
    print("\n" + "="*20 + " 产物正确性核查 " + "="*20)
    
    # 检查文件物理存在
    artifacts = {
        "Skew Config": SKEW_CONFIG_PATH,
        "MICE Imputer": IMPUTER_PATH,
        "Standard Scaler": SCALER_PATH,
        "Asset Bundle": bundle_path,
        "Processed Data": processed_path
    }
    for name, path in artifacts.items():
        status = "✅ 存在" if os.path.exists(path) else "❌ 缺失"
        size = f"{os.path.getsize(path)/1024:.1f} KB" if os.path.exists(path) else "N/A"
        print(f"{name:<20} : {status:<5} | 大小: {size}")

    print(f"\n最终建模特征数：{len(numeric_features)} (数值特征)")
    print(f"建模张量维度: {df_model.shape}")
    print(f"已生成中间产物: {processed_path}")
    print(f"已保存核心资产至: {ARTIFACT_DIR}")
    print("\n03 步完成！可以进入 04_mimic_stat_audit.py 或 05_feature_selection_lasso.py")
    print("-"*70)

if __name__ == "__main__":
    run_mimic_standardization()
