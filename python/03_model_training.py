import os
import pandas as pd
import numpy as np
import joblib

# 机器学习核心库
from sklearn.model_selection import train_test_split
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

# =========================================================
# 1. 配置与路径
# =========================================================
BASE_DIR = ".."
INPUT_PATH = os.path.join(BASE_DIR, "data/cleaned/mimic_for_model.csv")
SAVE_DIR = os.path.join(BASE_DIR, "models")
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def run_module_03():
    print("="*60)
    print("🚀 运行优化模块 03: 极致精简特征 + 亚组算法竞赛")
    print("="*60)
    
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 错误: 找不到输入文件 {INPUT_PATH}")
        return
        
    df = pd.read_csv(INPUT_PATH)

    # =========================================================
    # 2. 特征清洗与预处理
    # =========================================================
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'M': 1, 'F': 0})
    
    target = 'pof'
    drop_list = [
        target, 'mortality_28d', 'renal_pof', 'resp_pof', 'cv_pof', 
        'subgroup_no_renal', 'hosp_mortality', 'overall_mortality',
        'subject_id', 'hadm_id', 'stay_id', 'admittime', 'dischtime', 'intime', 
        'race', 'insurance'
    ]
    
    X = df.drop(columns=[c for c in drop_list if c in df.columns])
    y = df[target]
    
    # 记录亚组标记
    subgroup_flag = df['subgroup_no_renal']

    X = X.replace([np.inf, -np.inf], np.nan).astype(float)
    
    # 划分训练/测试集
    X_train, X_test, y_train, y_test, sub_train, sub_test = train_test_split(
        X, y, subgroup_flag, test_size=0.2, random_state=42, stratify=y
    )

    # =========================================================
    # 3. 增强型多重插补 (MICE)
    # =========================================================
    print("🧪 正在执行深度插补 (MICE)...")
    mice_imputer = IterativeImputer(
        max_iter=40, 
        n_nearest_features=10, 
        tol=1e-3, 
        random_state=42,
        initial_strategy='median'
    )
    scaler = StandardScaler()

    X_train_imp = mice_imputer.fit_transform(X_train)
    X_test_imp = mice_imputer.transform(X_test)
    
    X_train_std = scaler.fit_transform(X_train_imp)
    X_test_std = scaler.transform(X_test_imp)

    # =========================================================
    # 4. LASSO 特征降维 (Top 12)
    # =========================================================
    print("🧪 正在精选极致核心特征 (Top 12)...")
    lasso = LassoCV(cv=5, random_state=42, max_iter=20000).fit(X_train_std, y_train)
    
    coef_series = pd.Series(np.abs(lasso.coef_), index=X.columns).sort_values(ascending=False)
    num_features = min(12, (lasso.coef_ != 0).sum())
    selected_features = coef_series.head(num_features).index.tolist()
    
    feature_idx = [X.columns.get_loc(c) for c in selected_features]
    X_train_final = X_train_std[:, feature_idx]
    X_test_final = X_test_std[:, feature_idx]
    
    print(f"✅ 特征精简完成：保留了 {len(selected_features)} 个指标。")
    print(f"📝 最终指标清单: {selected_features}")

    # =========================================================
    # 5. 亚组算法竞赛：全人群 vs No-Renal
    # =========================================================
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=30),
        "SVM": SVC(probability=True, kernel='rbf', C=1.0), 
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, eval_metric='logloss', random_state=42)
    }

    # 准备亚组测试数据供竞赛使用
    X_test_final_df = pd.DataFrame(X_test_final, columns=selected_features)
    subgroup_mask = (sub_test == 1).values
    X_test_sub = X_test_final_df[subgroup_mask]
    y_test_sub = y_test.iloc[subgroup_mask] if isinstance(y_test, pd.Series) else y_test[subgroup_mask]

    print("\n🏆 算法性能竞赛 (AUC 对比):")
    print(f"{'Algorithm':<20} | {'Main AUC':<10} | {'No-Renal AUC':<10} | {'Brier (Main)':<10}")
    print("-" * 65)

    calibrated_results = {}
    for name, model in models.items():
        clf = CalibratedClassifierCV(model, cv=3, method='isotonic')
        clf.fit(X_train_final, y_train)
        
        # Main Test
        y_prob_main = clf.predict_proba(X_test_final)[:, 1]
        auc_main = roc_auc_score(y_test, y_prob_main)
        brier_main = brier_score_loss(y_test, y_prob_main)
        
        # Subgroup Test
        y_prob_sub = clf.predict_proba(X_test_sub.values)[:, 1]
        auc_sub = roc_auc_score(y_test_sub, y_prob_sub)
        
        calibrated_results[name] = clf
        print(f"{name:<20} | {auc_main:.4f}     | {auc_sub:.4f}         | {brier_main:.4f}")

    # =========================================================
    # 6. 保存核心结果
    # =========================================================
    joblib.dump(calibrated_results, os.path.join(SAVE_DIR, "all_models.pkl"))
    joblib.dump(selected_features, os.path.join(SAVE_DIR, "selected_features.pkl"))
    joblib.dump((X_test_final_df, y_test), os.path.join(SAVE_DIR, "test_data_main.pkl"))
    joblib.dump((X_test_sub, y_test_sub), os.path.join(SAVE_DIR, "test_data_subgroup.pkl"))

    print("-" * 60)
    print(f"✅ 模块 03 运行成功！已保存亚组测试样本: {len(y_test_sub)}")
    print("="*60)

if __name__ == "__main__":
    run_module_03()
