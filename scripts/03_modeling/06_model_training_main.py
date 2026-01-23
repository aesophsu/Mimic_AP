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
from sklearn.exceptions import ConvergenceWarning
import seaborn as sns
import warnings

# 基础配置
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 路径管理
BASE_DIR = "../../"
INPUT_PATH = os.path.join(BASE_DIR, "data/cleaned/mimic_processed.csv")
JSON_FEAT_PATH = os.path.join(BASE_DIR, "artifacts/features/selected_features.json")
MODEL_ROOT = os.path.join(BASE_DIR, "artifacts/models")
RESULT_ROOT = os.path.join(BASE_DIR, "results/figures")
os.makedirs(os.path.join(BASE_DIR, "artifacts/scalers"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "results/tables"), exist_ok=True)

def get_auc_ci(model, X_test, y_test, n_bootstraps=1000):
    scores = []
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

def plot_performance(models, X_test, y_test, target, save_path):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
    plt.rcParams['axes.unicode_minus'] = False
    colors = sns.color_palette("Set1", n_colors=len(models))
    plt.figure(figsize=(7, 7), dpi=300)
    ax = plt.gca()
    for i, (name, clf) in enumerate(models.items()):
        y_prob = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, lw=2.5, color=colors[i],
                 label=f"{name} (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], color='#454545', linestyle='--', lw=1.2, alpha=0.8)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=13, labelpad=10)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=13, labelpad=10)
    plt.title(f"ROC Analysis: {target.upper()}", fontsize=15, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=10, frameon=False)
    plt.grid(color='whitesmoke', linestyle='-', linewidth=1)
    plt.tight_layout()
    roc_path = os.path.join(save_path, f"{target}_ROC.pdf") # 推荐保存为 PDF 矢量图
    plt.savefig(roc_path, bbox_inches='tight', format='pdf')
    plt.savefig(roc_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=600)
    plt.close()
    plt.figure(figsize=(7, 7), dpi=300)
    ax = plt.gca()
    for i, (name, clf) in enumerate(models.items()):
        y_prob = clf.predict_proba(X_test)[:, 1]
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy='uniform')
        plt.plot(prob_pred, prob_true, marker='s', ms=5, lw=2, 
                 color=colors[i], label=name, alpha=0.9)
    plt.plot([0, 1], [0, 1], color='#454545', linestyle=':', lw=1.5, label='Perfectly Calibrated')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel("Predicted Probability", fontsize=13, labelpad=10)
    plt.ylabel("Actual Observed Probability", fontsize=13, labelpad=10)
    plt.title(f"Calibration Analysis: {target.upper()}", fontsize=15, fontweight='bold', pad=20)
    plt.legend(loc="upper left", fontsize=10, frameon=False)
    plt.grid(color='whitesmoke', linestyle='-', linewidth=1)
    plt.tight_layout()
    calib_path = os.path.join(save_path, f"{target}_Calibration.pdf")
    plt.savefig(calib_path, bbox_inches='tight', format='pdf')
    plt.savefig(calib_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=600)
    plt.close()
    print(f"✅ 医学出版级图片已保存 (PDF & PNG):\n - {save_path}")

def optimize_all_models(X_train, y_train):
    """
    统一的 Optuna 寻优模块：涵盖 XGB, RF, SVM, DT
    """
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_models = {}
    # --- 1. XGBoost 寻优 ---
    print("🔬 正在优化 XGBoost (n_trials=100)...")
    def xgb_obj(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'random_state': 42, 'eval_metric': 'logloss'
        }
        model = XGBClassifier(**param)
        return cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='roc_auc').mean()
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(xgb_obj, n_trials=100)
    best_models["XGBoost"] = XGBClassifier(**study_xgb.best_params, random_state=42)
    # --- 2. Random Forest 寻优 ---
    print("🔬 正在优化 Random Forest (n_trials=100)...")
    def rf_obj(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
        }
        model = RandomForestClassifier(**param, class_weight='balanced', random_state=42)
        return cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='roc_auc').mean()
    study_rf = optuna.create_study(direction='maximize')
    study_rf.optimize(rf_obj, n_trials=100)
    best_models["Random Forest"] = RandomForestClassifier(**study_rf.best_params, class_weight='balanced', random_state=42)
    # --- 3. SVM 寻优 ---
    print("🔬 正在优化 SVM (n_trials=50)...")
    def svm_obj(trial):
        param = {
            'C': trial.suggest_float('C', 0.1, 10.0, log=True),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'kernel': 'rbf'
        }
        model = SVC(**param, probability=True, class_weight='balanced', max_iter=2000)
        return cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='roc_auc').mean()
    
    study_svm = optuna.create_study(direction='maximize')
    study_svm.optimize(svm_obj, n_trials=50) # SVM 耗时较长， trials 可略少
    best_models["SVM"] = SVC(**study_svm.best_params, probability=True, class_weight='balanced', max_iter=5000)
    # --- 4. Decision Tree 寻优 ---
    print("🔬 正在优化 Decision Tree (n_trials=50)...")
    def dt_obj(trial):
        param = {
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20)
        }
        model = DecisionTreeClassifier(**param, class_weight='balanced', random_state=42)
        return cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='roc_auc').mean()
    
    study_dt = optuna.create_study(direction='maximize')
    study_dt.optimize(dt_obj, n_trials=50)
    best_models["Decision Tree"] = DecisionTreeClassifier(**study_dt.best_params, class_weight='balanced', random_state=42)

    return best_models

def train_and_calibrate_all(X_train, y_train, best_instances):
    """
    接收寻优后的模型字典，并进行统一的等渗校准（Isotonic Calibration）
    """
    models = {
        **best_instances,  # 解包包含 XGB, RF, SVM, DT 的字典
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=2000)
    }
    calibrated_results = {}
    for name, m in models.items():
        print(f"⚙️ 正在校准模型: {name}...")
        clf = CalibratedClassifierCV(m, cv=3, method='isotonic', n_jobs=-1)
        try:
            clf.fit(X_train, y_train)
            calibrated_results[name] = clf
        except Exception as e:
            print(f"❌ 模型 {name} 校准失败: {e}")
    return calibrated_results

def evaluate_performance(models_dict, X_test, y_test, sub_mask):
    """
    计算全人群及亚组的评估指标与置信区间，并实时打印日志
    """
    summary = []
    ci_stats = {}
    print(f"\n{'Algorithm':<20} | {'Main AUC (95% CI)':<25} | {'Brier':<10}")
    print("-" * 65)
    for name, clf in models_dict.items():
        y_prob = clf.predict_proba(X_test)[:, 1]
        auc_m = roc_auc_score(y_test, y_prob)
        low_m, high_m = get_auc_ci(clf, X_test, y_test)
        brier = brier_score_loss(y_test, y_prob)
        if len(np.unique(y_test[sub_mask])) > 1:
            auc_s = roc_auc_score(y_test[sub_mask], y_prob[sub_mask])
            low_s, high_s = get_auc_ci(clf, X_test[sub_mask], y_test[sub_mask])
        else:
            auc_s, low_s, high_s = 0, 0, 0
        main_ci_str = f"{auc_m:.3f} ({low_m:.3f}-{high_m:.3f})"
        sub_ci_str = f"{auc_s:.3f} ({low_s:.3f}-{high_s:.3f})"
        print(f"{name:<20} | {main_ci_str:<25} | {brier:.4f}")
        summary.append({
            "Algorithm": name,
            "Main AUC": round(auc_m, 4),
            "Main CI": main_ci_str,
            "Sub CI": sub_ci_str,
            "Brier": round(brier, 4)
        })
        ci_stats[name] = {
            "main": [float(low_m), float(high_m)], 
            "sub": [float(low_s), float(high_s)]
        }
    return summary, ci_stats

def save_model_assets(target, target_dir, models_dict, scaler, ci_stats, features, X_train_cols):
    """
    保存模型、标准化器及部署资产包，确保 eICU 外部验证的列对齐
    """
    joblib.dump(models_dict, os.path.join(target_dir, "all_models_dict.pkl"))
    joblib.dump(scaler, os.path.join(target_dir, "scaler.pkl"))
    joblib.dump(ci_stats, os.path.join(target_dir, "bootstrap_ci_stats.pkl"))
    deploy_bundle = {
        'feature_names': list(features),  # 强制转为 list 存储
        'scaler': scaler,
        'target_outcome': target
    }
    joblib.dump(deploy_bundle, os.path.join(target_dir, "deploy_bundle.pkl"))
    imp_list = []
    for name in ["XGBoost", "Random Forest", "Logistic Regression"]:
        if name in models_dict:
            try:
                cal_clf = models_dict[name].calibrated_classifiers_[0]
                base = getattr(cal_clf, 'estimator', getattr(cal_clf, 'base_estimator', None))
                if name == "Logistic Regression":
                    weights = base.coef_.flatten()
                else:
                    weights = base.feature_importances_
                if len(weights) == len(features):
                    imp_df = pd.DataFrame({
                        'feature': features, 
                        'importance': weights, 
                        'algorithm': name, 
                        'outcome': target
                    })
                    imp_list.append(imp_df)
                else:
                    print(f"⚠️ {name} 权重长度({len(weights)})与特征数({len(features)})不匹配，已跳过。")
            except Exception as e:
                print(f"⚠️ 提取 {name} 重要性时发生错误: {e}")
                continue
    if imp_list:
        final_imp_df = pd.concat(imp_list, ignore_index=True)
        final_imp_df.to_csv(os.path.join(target_dir, "feature_importance.csv"), index=False)
        print(f"💾 特征重要性已保存至: {target_dir}")
        
def run_model_training_flow():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"❌ 找不到输入数据: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    with open(JSON_FEAT_PATH, 'r') as f:
        feature_config = json.load(f)
    global_performance = []
    for target, config in feature_config.items():
        if target not in df.columns:
            print(f"⚠️ 跳过结局 {target}: 不在数据列中")
            continue
        print(f"\n\n{'='*20} 启动临床结局分析: {target.upper()} {'='*20}")
        target_model_dir = os.path.join(MODEL_ROOT, target.lower())
        target_fig_dir = os.path.join(RESULT_ROOT, target.lower())
        for d in [target_model_dir, target_fig_dir]: os.makedirs(d, exist_ok=True)
        selected_features = config['features']
        missing_feats = [f for f in selected_features if f not in df.columns]
        if missing_feats:
            raise ValueError(f"❌ 结局 {target} 缺失关键特征: {missing_feats}")
        X_data = df[selected_features].copy()
        y_data = df[target]
        sub_group = df['subgroup_no_renal'] # 亚组分析标签
        X_train, X_test, y_train, y_test, _, sub_test = train_test_split(
            X_data, y_data, sub_group, 
            test_size=0.2, random_state=42, stratify=y_data
        )
        scaler_pre = StandardScaler()
        X_train_pre = scaler_pre.fit_transform(X_train)
        X_test_pre = scaler_pre.transform(X_test)
        joblib.dump(scaler_pre, os.path.join(target_model_dir, "scaler.pkl"))
        sub_mask = (sub_test == 1).values
        best_tuned_instances = optimize_all_models(X_train_pre, y_train)
        calibrated_models = train_and_calibrate_all(X_train_pre, y_train, best_tuned_instances)
        summary_list, ci_stats = evaluate_performance(calibrated_models, X_test_pre, y_test, sub_mask)
        for s in summary_list: 
            s['Outcome'] = target 
        save_model_assets(target, target_model_dir, calibrated_models, scaler_pre, ci_stats, selected_features, X_train.columns)
        X_test_df = pd.DataFrame(X_test_pre, columns=selected_features)
        eval_bundle = {
            'X_test_pre': X_test_df,
            'X_test_raw': X_test,
            'y_test': y_test.values, 
            'sub_mask': sub_mask, 
            'features': selected_features
        }
        joblib.dump(eval_bundle, os.path.join(target_model_dir, "eval_data.pkl"))
        plot_performance(calibrated_models, X_test_pre, y_test, target, target_fig_dir)
        global_performance.extend(summary_list)

    if global_performance:
        perf_df = pd.DataFrame(global_performance)
        report_path = os.path.join(MODEL_ROOT, "performance_report.csv")
        perf_df.to_csv(report_path, index=False)
        print(f"\n✅ 所有结局分析完成！汇总报告见: {report_path}")

if __name__ == "__main__":
    run_model_training_flow()
