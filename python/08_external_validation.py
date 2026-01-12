import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss

# =========================================================
# 1. 配置与路径
# =========================================================
BASE_DIR = ".."
EICU_RAW_CLEANED = os.path.join(BASE_DIR, "data/cleaned/eicu_for_table1.csv") 
MODELS_PATH = os.path.join(BASE_DIR, "models/all_models.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models/scaler.pkl")
IMPUTER_PATH = os.path.join(BASE_DIR, "models/mice_imputer.pkl")
SKEWED_COLS_PATH = os.path.join(BASE_DIR, "models/skewed_cols.pkl")
SELECTED_FEATURES_PATH = os.path.join(BASE_DIR, "models/selected_features.pkl")
TEST_DATA_MIMIC_PATH = os.path.join(BASE_DIR, "models/test_data_main.pkl")
SAVE_DIR = os.path.join(BASE_DIR, "results")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def run_module_08_debug():
    print("="*60)
    print("🏆 模块 08: 增强诊断与强制对齐外部验证")
    print("="*60)

    # 1. 加载模型资产
    try:
        models_dict = joblib.load(MODELS_PATH)
        scaler = joblib.load(SCALER_PATH)
        imputer = joblib.load(IMPUTER_PATH)
        skewed_cols = joblib.load(SKEWED_COLS_PATH)
        selected_features = joblib.load(SELECTED_FEATURES_PATH)
        
        # 获取训练时严格的列清单和顺序
        if hasattr(scaler, 'feature_names_in_'):
            train_cols = list(scaler.feature_names_in_)
        elif hasattr(imputer, 'feature_names_in_'):
            train_cols = list(imputer.feature_names_in_)
        else:
            print("❌ 严重错误：Scaler 和 Imputer 均未保存特征名。请重新运行模块 03 并检查 sklearn 版本。")
            return
        
        print(f"✅ 资产加载成功。训练时特征数: {len(train_cols)}，精选特征数: {len(selected_features)}")
    except Exception as e:
        print(f"❌ 资产加载失败: {e}")
        return

    # 2. 加载 eICU 数据
    df_eicu = pd.read_csv(EICU_RAW_CLEANED)
    print(f"📊 eICU 原始数据加载成功: {df_eicu.shape}")

    # ---------------------------------------------------------
    # 3. 🔍 深度纠错诊断 (关键步骤)
    # ---------------------------------------------------------
    print("\n🔍 [诊断] 特征对齐分析:")
    eicu_cols = set(df_eicu.columns)
    missing_in_eicu = [c for c in train_cols if c not in eicu_cols]
    unseen_in_train = [c for c in eicu_cols if c not in train_cols and c != 'pof']
    
    print(f"  - 训练特征总数: {len(train_cols)}")
    print(f"  - eICU 缺失的特征数: {len(missing_in_eicu)}")
    if missing_in_eicu:
        print(f"  - 缺失示例 (前5个): {missing_in_eicu[:5]}")
    
    # ---------------------------------------------------------
    # 4. 🛠️ 强制对齐特征空间 (逻辑闭环)
    # ---------------------------------------------------------
    print("\n🧪 正在强制重建 eICU 特征空间以匹配模型...")
    X_eicu_aligned = pd.DataFrame(index=df_eicu.index)
    
    for col in train_cols:
        if col in df_eicu.columns:
            # 1. 基础赋值
            val = df_eicu[col].copy()
            # 2. 动态 Log1p 转换 (必须与模块 03 严格一致)
            if col in skewed_cols:
                val = np.log1p(val.fillna(val.median()).clip(lower=0))
            X_eicu_aligned[col] = val
        else:
            # 3. 补全缺失列 (用 0 占位，后续插补器会处理或保持中性)
            X_eicu_aligned[col] = np.nan

    # 🛑 强制列顺序与训练时完全一致
    X_eicu_aligned = X_eicu_aligned[train_cols]
    print("✅ 强制对齐完成。")

    # ---------------------------------------------------------
    # 5. 执行插补与标准化
    # ---------------------------------------------------------
    try:
        X_eicu_imp = imputer.transform(X_eicu_aligned)
        X_eicu_std = scaler.transform(X_eicu_imp)
        
        # 转换为带列名的 DF 方便后续提取
        X_eicu_processed = pd.DataFrame(X_eicu_std, columns=train_cols)
        X_eicu_final = X_eicu_processed[selected_features]
        print("✅ MICE 插补与 StandardScaler 缩放成功。")
    except Exception as e:
        print(f"❌ 预处理失败: {e}")
        return

    # 6. 性能评估
    y_eicu = df_eicu['pof']
    X_mimic_test, y_mimic_test = joblib.load(TEST_DATA_MIMIC_PATH)

    print("\n" + "="*70)
    print(f"{'Algorithm':<20} | {'MIMIC AUC':<12} | {'eICU AUC':<12} | {'Brier'}")
    print("-" * 70)

    plt.figure(figsize=(9, 8), dpi=150)
    for name, clf in models_dict.items():
        y_prob_mimic = clf.predict_proba(X_mimic_test.values)[:, 1]
        y_prob_eicu = clf.predict_proba(X_eicu_final.values)[:, 1]

        auc_m = roc_auc_score(y_mimic_test, y_prob_mimic)
        auc_e = roc_auc_score(y_eicu, y_prob_eicu)
        
        print(f"{name:<20} | {auc_m:.4f}     | {auc_e:.4f}     | {brier_score_loss(y_eicu, y_prob_eicu):.4f}")
        
        fpr, tpr, _ = roc_curve(y_eicu, y_prob_eicu)
        plt.plot(fpr, tpr, label=f'{name} (eICU AUC={auc_e:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Cross-Database Validation (MIMIC -> eICU)')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(SAVE_DIR, "external_validation_debug.png"))
    plt.show()

if __name__ == "__main__":
    run_module_08_debug()
