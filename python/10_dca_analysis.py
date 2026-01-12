import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# =========================================================
# 1. 配置与路径
# =========================================================
BASE_DIR = ".."
MODELS_PATH = os.path.join(BASE_DIR, "models/all_models.pkl")
EICU_RAW_CLEANED = os.path.join(BASE_DIR, "data/cleaned/eicu_for_table1.csv") 
SCALER_PATH = os.path.join(BASE_DIR, "models/scaler.pkl")
IMPUTER_PATH = os.path.join(BASE_DIR, "models/mice_imputer.pkl")
SKEWED_COLS_PATH = os.path.join(BASE_DIR, "models/skewed_cols.pkl")
SELECTED_FEATURES_PATH = os.path.join(BASE_DIR, "models/selected_features.pkl")
SAVE_DIR = os.path.join(BASE_DIR, "results")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# =========================================================
# 2. 核心函数：计算净获益
# =========================================================
def calculate_net_benefit(y_true, y_prob, thresholds):
    net_benefit = []
    n = len(y_true)
    for pt in thresholds:
        if pt <= 0 or pt >= 1:
            net_benefit.append(0)
            continue
        # 根据阈值计算预测结果
        y_pred = (y_prob >= pt).astype(int)
        # 手动计算 tp 和 fp 避免 confusion_matrix 在极端情况下的崩溃
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        # DCA 核心公式
        nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        net_benefit.append(nb)
    return net_benefit

def run_module_09_compatible():
    print("="*60)
    print("📈 运行模块 09: 临床决策曲线分析 (DCA) - 兼容性增强版")
    print("="*60)

    # 1. 加载模型资产并解决 feature_names_in_ 问题
    try:
        models_dict = joblib.load(MODELS_PATH)
        scaler = joblib.load(SCALER_PATH)
        imputer = joblib.load(IMPUTER_PATH)
        skewed_cols = joblib.load(SKEWED_COLS_PATH)
        selected_features = joblib.load(SELECTED_FEATURES_PATH)
        
        # 🛡️ 兼容性修复逻辑
        if hasattr(scaler, 'feature_names_in_'):
            train_cols = list(scaler.feature_names_in_)
        elif hasattr(imputer, 'feature_names_in_'):
            train_cols = list(imputer.feature_names_in_)
        else:
            # 如果都找不到，则需要手动根据模块 02/03 的列提取逻辑推断，
            # 这里的 train_cols 必须是训练时的完整特征清单。
            print("❌ 错误：无法从资产中提取特征名。请检查模块 03 运行时的 sklearn 版本。")
            return
            
        print(f"✅ 资产对齐成功。训练特征数: {len(train_cols)}")
    except Exception as e:
        print(f"❌ 加载资产失败: {e}")
        return

    # 2. 加载并对齐 eICU 数据 (逻辑同模块 08 增强版)
    df_eicu = pd.read_csv(EICU_RAW_CLEANED)
    X_eicu_aligned = pd.DataFrame(index=df_eicu.index)
    
    for col in train_cols:
        if col in df_eicu.columns:
            val = df_eicu[col].copy()
            if col in skewed_cols:
                val = np.log1p(val.fillna(val.median()).clip(lower=0))
            X_eicu_aligned[col] = val
        else:
            X_eicu_aligned[col] = np.nan
            
    # 强制排序并执行转换
    X_eicu_aligned = X_eicu_aligned[train_cols]
    X_eicu_std = scaler.transform(imputer.transform(X_eicu_aligned))
    X_eicu_final = pd.DataFrame(X_eicu_std, columns=train_cols)[selected_features]
    y_eicu = df_eicu['pof'].values

    # 3. 计算与绘制
    thresholds = np.linspace(0.01, 0.99, 100)
    plt.figure(figsize=(10, 8), dpi=150)

    # 绘制基准线
    prevalence = np.mean(y_eicu)
    net_benefit_all = [prevalence - (1 - prevalence) * (pt / (1 - pt)) for pt in thresholds]
    
    plt.plot(thresholds, net_benefit_all, color='gray', linestyle='--', label='Treat All', alpha=0.6)
    plt.axhline(y=0, color='black', linestyle='-', label='Treat None', alpha=0.6)

    # 绘制多模型曲线
    for name, clf in models_dict.items():
        print(f"🧪 计算中: {name}...")
        y_prob = clf.predict_proba(X_eicu_final.values)[:, 1]
        nb = calculate_net_benefit(y_eicu, y_prob, thresholds)
        plt.plot(thresholds, nb, lw=2, label=f'{name}')

    # 4. 图表美化
    plt.xlim(0, 1.0)
    plt.ylim(-0.05, prevalence + 0.1)
    plt.xlabel('Threshold Probability', fontsize=12)
    plt.ylabel('Net Benefit', fontsize=12)
    plt.title('Clinical Decision Curve Analysis (External eICU Data)', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    
    save_path = os.path.join(SAVE_DIR, "dca_final_eicu.png")
    plt.savefig(save_path)
    plt.show()

    print("-" * 60)
    print(f"✅ 模块 09 DCA 运行成功！结果已保存至: {save_path}")

if __name__ == "__main__":
    run_module_09_compatible()
