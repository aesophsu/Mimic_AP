import os
import pandas as pd
import numpy as np
import joblib
import optuna

# 机器学习核心库
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

# 屏蔽警告
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =========================================================
# 1. 配置与路径
# =========================================================
BASE_DIR = ".."
INPUT_PATH = os.path.join(BASE_DIR, "data/cleaned/mimic_for_model.csv")
SAVE_DIR = os.path.join(BASE_DIR, "models")
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def run_module_03_optimized():
    print("="*60)
    print("🚀 运行终极重构模块 03: 5 种模型竞赛 + 动态对数处理")
    print("="*60)
    
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 错误: 找不到输入文件 {INPUT_PATH}")
        return
        
    df = pd.read_csv(INPUT_PATH)

    # =========================================================
    # 2. 特征清洗与预处理 (关键：修复文本列报错)
    # =========================================================
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'M': 1, 'F': 0})
    
    target = 'pof'
    # 排除列表
    drop_list = [
        target, 'mortality_28d', 'renal_pof', 'resp_pof', 'cv_pof', 
        'subgroup_no_renal', 'hosp_mortality', 'overall_mortality',
        'composite_outcome'
    ]
    
    # 🛡️ 自动剔除非数值特征 (处理 ValueError: could not convert string to float)
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    final_drop = list(set(drop_list + text_cols))
    print(f"🗑️ 自动剔除泄露/非数值特征: {text_cols}")
    
    X = df.drop(columns=[c for c in final_drop if c in df.columns])
    y = df[target]
    subgroup_flag = df['subgroup_no_renal']

    # 处理无穷大并确保数值化
    X = X.replace([np.inf, -np.inf], np.nan).astype(float)
    
    # 划分训练/测试集
    X_train, X_test, y_train, y_test, sub_train, sub_test = train_test_split(
        X, y, subgroup_flag, test_size=0.2, random_state=42, stratify=y
    )

    # =========================================================
    # 3. 🧪 核心修正：动态 Log1p 转换 (救赎线性模型)
    # =========================================================
    skewed_cols = ['creatinine_max', 'creatinine_min', 'bun_max', 'bun_min',
                   'wbc_max', 'wbc_min', 'glucose_max', 'glucose_min',
                   'lab_amylase_max', 'lipase_max', 'lactate_max',
                   'alt_max', 'ast_max', 'bilirubin_total_max', 
                   'alp_max', 'inr_max', 'rdw_max']
    
    existing_skewed = [c for c in skewed_cols if c in X_train.columns]
    print(f"🔄 正在执行动态 Log1p 转换 ({len(existing_skewed)} 个变量)...")
    for col in existing_skewed:
        X_train[col] = np.log1p(X_train[col].clip(lower=0))
        X_test[col] = np.log1p(X_test[col].clip(lower=0))

    # =========================================================
    # 4. 增强型多重插补 (MICE) & 标准化
    # =========================================================
    print("🧪 正在执行深度插补 (MICE)...")
    mice_imputer = IterativeImputer(max_iter=20, random_state=42, initial_strategy='median')
    scaler = StandardScaler()

    X_train_imp = mice_imputer.fit_transform(X_train)
    X_train_std = scaler.fit_transform(X_train_imp)

    X_test_imp = mice_imputer.transform(X_test)
    X_test_std = scaler.transform(X_test_imp)

    # 保存资产供跨库验证
    joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.pkl"))
    joblib.dump(mice_imputer, os.path.join(SAVE_DIR, "mice_imputer.pkl"))
    joblib.dump(existing_skewed, os.path.join(SAVE_DIR, "skewed_cols.pkl"))

    # =========================================================
    # 5. LASSO 特征降维 (Top 12)
    # =========================================================
    print("🧪 正在精选极致核心特征 (Top 12)...")
    lasso = LassoCV(cv=5, random_state=42, max_iter=20000).fit(X_train_std, y_train)
    
    coef_abs = np.abs(lasso.coef_)
    indices = np.argsort(coef_abs)[-12:] # 锁定绝对值最大的 12 个特征
    selected_features = X.columns[indices].tolist()
    
    X_train_final = X_train_std[:, indices]
    X_test_final = X_test_std[:, indices]
    print(f"✅ 特征精简完成: {selected_features}")

    # =========================================================
    # 6. XGBoost Optuna 超参数寻优
    # =========================================================
    print("\n🔬 启动 XGBoost 贝叶斯寻优 (Optuna)...")
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'random_state': 42, 'eval_metric': 'logloss'
        }
        model = XGBClassifier(**param)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        return cross_val_score(model, X_train_final, y_train, cv=cv, scoring='roc_auc').mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    best_xgb = XGBClassifier(**study.best_params)

    # =========================================================
    # 7. 🏆 5 种模型算法竞赛 (含概率校准)
    # =========================================================
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=20),
        "SVM": SVC(probability=True, kernel='rbf', C=1.0), 
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42),
        "XGBoost": best_xgb
    }

    # 准备亚组测试索引
    sub_mask = (sub_test == 1).values
    X_test_sub = X_test_final[sub_mask]
    y_test_sub = y_test.iloc[sub_mask]

    print("\n" + "="*70)
    print(f"{'Algorithm':<20} | {'Main AUC':<10} | {'No-Renal AUC':<10} | {'Brier':<10}")
    print("-" * 70)

    calibrated_results = {}
    for name, model in models.items():
        # 使用概率校准优化 Brier Score
        clf = CalibratedClassifierCV(model, cv=3, method='isotonic')
        clf.fit(X_train_final, y_train)
        
        y_prob = clf.predict_proba(X_test_final)[:, 1]
        auc_main = roc_auc_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        
        y_prob_sub = clf.predict_proba(X_test_sub)[:, 1]
        auc_sub = roc_auc_score(y_test_sub, y_prob_sub)
        
        calibrated_results[name] = clf
        print(f"{name:<20} | {auc_main:.4f}     | {auc_sub:.4f}         | {brier:.4f}")

    # =========================================================
    # 8. 全资产保存
    # =========================================================
    joblib.dump(calibrated_results, os.path.join(SAVE_DIR, "all_models.pkl"))
    joblib.dump(selected_features, os.path.join(SAVE_DIR, "selected_features.pkl"))
    
    # 保存测试集 DataFrame 格式供模块 08 使用
    X_test_final_df = pd.DataFrame(X_test_final, columns=selected_features)
    joblib.dump((X_test_final_df, y_test), os.path.join(SAVE_DIR, "test_data_main.pkl"))
    
    print("-" * 60)
    print("✅ 模块 03 成功！线性模型与树模型已完成动态处理并保存。")

if __name__ == "__main__":
    run_module_03_optimized()
