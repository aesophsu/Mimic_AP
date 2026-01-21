import os
import json
import joblib
import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.utils import resample
import warnings

# 基础配置
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 路径管理
BASE_DIR = "../../"
INPUT_PATH = os.path.join(BASE_DIR, "data/cleaned/mimic_processed.csv")
JSON_FEAT_PATH = os.path.join(BASE_DIR, "artifacts/features/selected_features.json")
SAVE_DIR = os.path.join(BASE_DIR, "artifacts/models")
FIG_DIR = os.path.join(BASE_DIR, "results/figures/performance")

for d in [SAVE_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# =========================================================
# 辅助工具函数
# =========================================================
def get_auc_ci(model, X_test, y_test, n_bootstraps=1000):
    scores = []
    # 确保 X_test 是 array
    X_arr = np.array(X_test)
    y_arr = np.array(y_test)
    indices = np.arange(len(y_arr))
    
    for i in range(n_bootstraps):
        resample_idx = resample(indices, random_state=i)
        y_res = y_arr[resample_idx]
        if len(np.unique(y_res)) < 2: continue
        
        prob = model.predict_proba(X_arr[resample_idx])[:, 1]
        scores.append(roc_auc_score(y_res, prob))
    sorted_scores = np.sort(scores)
    if len(sorted_scores) == 0: return 0, 0
    return sorted_scores[int(0.025 * len(sorted_scores))], sorted_scores[int(0.975 * len(sorted_scores))]

# =========================================================
# 核心训练流水线
# =========================================================
def run_model_training_flow():
    # 1. 加载数据与特征配置
    df = pd.read_csv(INPUT_PATH)
    with open(JSON_FEAT_PATH, 'r') as f:
        feature_config = json.load(f)
    
    global_performance = []

    for target in feature_config.keys():
        print(f"\n\n{'='*30} 正在分析结局: {target.upper()} {'='*30}")
        
        # 2. 准备该结局专属特征集
        selected_features = feature_config[target]['features']
        X = df[selected_features].copy()
        y = df[target]
        subgroup = df['subgroup_no_renal']

        # 3. 划分与预处理
        X_train, X_test, y_train, y_test, _, sub_test = train_test_split(
            X, y, subgroup, test_size=0.2, random_state=42, stratify=y
        )
        imputer_pre = IterativeImputer(max_iter=10, random_state=42)
        scaler_pre = StandardScaler()
        X_train_pre = scaler_pre.fit_transform(imputer_pre.fit_transform(X_train))

        # 4. XGBoost Optuna 贝叶斯寻优
        print(f"🔬 启动 {target} 的 XGBoost 寻优...")
        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'random_state': 42, 'eval_metric': 'logloss'
            }
            model = XGBClassifier(**param)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            return cross_val_score(model, X_train_imp, y_train, cv=cv, scoring='roc_auc').mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30) # 生产环境建议 50-100
        best_xgb = XGBClassifier(**study.best_params, random_state=42)

        # 5. 模型竞赛 (含概率校准)
        models = {
            "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, class_weight='balanced')
            "XGBoost": best_xgb,
            "SVM": SVC(probability=True, class_weight='balanced'),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight='balanced')
        }

        calibrated_results = {}
        target_summary = []

        print("\n" + "-"*95)
        print(f"{'Algorithm':<20} | {'Main AUC (95% CI)':<30} | {'Brier':<10}")
        print("-" * 95)

        for name, m in models.items():
            # 概率校准
            clf = CalibratedClassifierCV(m, cv=3, method='isotonic')
            clf.fit(X_train_pre, y_train) 
            calibrated_results[name] = clf
            X_test_pre = scaler_pre.transform(imputer_pre.transform(X_test)) # 严谨：用 train 的参数转 test
            y_prob = clf.predict_proba(X_test_pre)[:, 1]
            sub_mask = (sub_test == 1).values
            if len(np.unique(y_test[sub_mask])) > 1:
                auc_sub = roc_auc_score(y_test[sub_mask], y_prob[sub_mask])
                low_s, high_s = get_auc_ci(clf, X_test_pre[sub_mask], y_test[sub_mask])
            else:
                auc_sub, low_s, high_s = 0, 0, 0

            main_auc_str = f"{auc_main:.3f} ({low_m:.3f}-{high_m:.3f})"
            sub_auc_str = f"{auc_sub:.3f} ({low_s:.3f}-{high_s:.3f})"
            
            print(f"{name:<20} | {main_auc_str:<30} | {brier:.4f}")

            target_summary.append({
                "Outcome": target,
                "Algorithm": name,
                "Main AUC": round(auc_main, 4),
                "Main AUC CI": main_auc_str,
                "No-Renal AUC CI": sub_auc_str,
                "Brier": round(brier, 4)
            })

        # 6. 保存资产
        joblib.dump(calibrated_results, os.path.join(SAVE_DIR, f"models_{target}.pkl"))
        global_performance.extend(target_summary)

        # 7. 绘图
        plot_performance(calibrated_results, X_test_imp, y_test, target)

    # 生成最终汇总表
    pd.DataFrame(global_performance).to_csv(os.path.join(SAVE_DIR, "performance_report.csv"), index=False)
    print(f"\n🚀 训练流程全部完成！报告已存至: {SAVE_DIR}")

def plot_performance(models, X_test, y_test, target):
    """将 ROC 和 Calibration 曲线生成为两个独立的学术图片"""
    
    # --- 1. 绘制并保存独立 ROC 曲线 ---
    plt.figure(figsize=(8, 7), dpi=300) # 高清分辨率
    for name, clf in models.items():
        y_prob = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.3f})")
    
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, alpha=0.7)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title(f"ROC Curves: Predicted {target.upper()}", fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    roc_path = os.path.join(FIG_DIR, f"Figure_ROC_{target}.png")
    plt.savefig(roc_path, bbox_inches='tight')
    plt.show()
    plt.close()

    # --- 2. 绘制并保存独立校准曲线 (Calibration) ---
    plt.figure(figsize=(8, 7), dpi=300)
    for name, clf in models.items():
        y_prob = clf.predict_proba(X_test)[:, 1]
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', lw=2, ms=6, label=name)
    
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Perfectly Calibrated')
    plt.xlabel("Predicted Probability", fontsize=12)
    plt.ylabel("Actual Probability", fontsize=12)
    plt.title(f"Calibration Curves: Predicted {target.upper()}", fontsize=14, fontweight='bold')
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    calib_path = os.path.join(FIG_DIR, f"Figure_Calibration_{target}.png")
    plt.savefig(calib_path, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"✅ 图片已保存:\n   - ROC: {roc_path}\n   - Calib: {calib_path}")

if __name__ == "__main__":
    run_model_training_flow()
