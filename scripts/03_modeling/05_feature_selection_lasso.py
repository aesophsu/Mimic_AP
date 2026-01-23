import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = "../../"
INPUT_PATH = os.path.join(BASE_DIR, "data/cleaned/mimic_processed.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts/features")
MODELS_ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts/models")
FIG_DIR = os.path.join(BASE_DIR, "results/figures/lasso")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

def plot_academic_lasso(cv_model, X_columns, target):
    Cs = cv_model.Cs_
    log_Cs = np.log10(Cs)
    pos_class_idx = cv_model.classes_[1]
    scores_mean = cv_model.scores_[pos_class_idx].mean(axis=0)
    scores_se = cv_model.scores_[pos_class_idx].std(axis=0) / np.sqrt(cv_model.scores_[pos_class_idx].shape[0])
    idx_max = np.argmax(scores_mean)
    target_score = scores_mean[idx_max] - scores_se[idx_max]
    eligible_indices = np.where(scores_mean >= target_score)[0]
    idx_1se = eligible_indices[np.argmin(Cs[eligible_indices])]
    plt.figure(figsize=(8, 6), dpi=300)
    ax1 = plt.gca()
    ax1.errorbar(log_Cs, scores_mean, yerr=scores_se, fmt='o', color='red',
                 ecolor='lightgray', elinewidth=1, capsize=2, mfc='red', ms=4, label='CV ROC AUC')
    ax1.axvline(log_Cs[idx_max], color='blue', linestyle='--', 
                label=f'Max AUC (logC={log_Cs[idx_max]:.2f})')
    ax1.axvline(log_Cs[idx_1se], color='black', linestyle='--', 
                label=f'1-SE Rule (logC={log_Cs[idx_1se]:.2f})')
    ax1.set_xlabel(r'$\log_{10}(C)$')
    ax1.set_ylabel('Mean ROC AUC')
    ax1.set_title(f'LASSO Selection (Logistic): {target.upper()}', fontweight='bold', pad=15)
    ax1.legend(loc='lower right', frameon=True)
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xlabel('Regularization Strength')
    ax2.set_xticks([log_Cs.min(), log_Cs.max()])
    ax2.set_xticklabels(['Strong (Sparse)', 'Weak (Dense)'])
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"lasso_diag_{target}.png"), bbox_inches='tight')
    plt.close()

def plot_lasso_trajectories(X, y, target):
    print(f"正在计算 {target.upper()} 的系数路径...")
    alphas, coefs, _ = logistic_regression_path(
        X.values, y, pos_class=1, Cs=100, l1_ratio=1.0, 
        fit_intercept=True, penalty='l1', solver='liblinear'
    )
    log_Cs_path = np.log10(1 / (alphas * X.shape[0]))
    path_coefs = coefs[0] 
    plt.figure(figsize=(10, 7), dpi=300)
    final_coefs = path_coefs[-1, :]
    important_idx = np.argsort(np.abs(final_coefs))[-12:]
    for i in range(path_coefs.shape[1]):
        if i in important_idx:
            plt.plot(log_Cs_path, path_coefs[:, i], label=X.columns[i], linewidth=1.5)
        else:
            plt.plot(log_Cs_path, path_coefs[:, i], color='gray', alpha=0.2, linewidth=0.5)
    plt.axhline(0, color='black', lw=1, ls='-')
    plt.xlabel(r'$\log_{10}(C)$')
    plt.ylabel('Coefficients')
    plt.title(f'LASSO Path: {target.upper()}', fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, title="Top Features")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"lasso_traj_{target}.png"), bbox_inches='tight')
    plt.close()

def run_lasso_selection_flow():
    targets = ['pof', 'mortality', 'composite']
    df = pd.read_csv(INPUT_PATH)
    protected = ['pof', 'mortality', 'composite', 'subgroup_no_renal',
                 'resp_pof', 'cv_pof', 'renal_pof',
                 'sofa_score', 'apsiii', 'sapsii', 'oasis', 'lods',
                 'subject_id', 'hadm_id', 'stay_id', 'los',
                 'mechanical_vent_flag', 'vaso_flag']
    X_cols = [c for c in df.columns if c not in protected]
    X_audit = df[X_cols]
    max_mean = X_audit.mean().abs().max()
    print(f"🔍 [标准化审计] 特征最大绝对均值: {max_mean:.6f}")
    if max_mean > 0.1:
        print("⚠️ 警告: 检测到特征均值显著偏离0，请确认是否已运行 03_standardization.py")
    else:
        print("✅ 审计通过: 特征尺度已对齐")
    all_outcomes_features = {}
    for target in targets:
        print(f"\n>>> 正在精炼: {target.upper()}")
        TARGET_ARTIFACTS = os.path.join(MODELS_ARTIFACTS_DIR, target)
        os.makedirs(TARGET_ARTIFACTS, exist_ok=True)
        X = df[X_cols]
        y = df[target].values
        classes = np.unique(y)
        print(f"📊 [结局审计] 类别分布: {classes}, 阳性样本数: {sum(y==1)}")
        if len(classes) != 2:
            print(f"❌ 错误: {target} 不是二分类结局，跳过。")
            continue
        lasso_cv = LogisticRegressionCV(
            Cs=100, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
            penalty='l1', solver='liblinear', scoring='roc_auc',
            random_state=42, max_iter=1000, n_jobs=-1
        )
        lasso_cv.fit(X, y)
        scores_mean = lasso_cv.scores_[1].mean(axis=0)
        scores_se = lasso_cv.scores_[1].std(axis=0) / np.sqrt(10)
        idx_max = np.argmax(scores_mean)
        target_score = scores_mean[idx_max] - scores_se[idx_max]
        eligible_idx = np.where(scores_mean >= target_score)[0]
        idx_1se = eligible_idx[np.argmin(lasso_cv.Cs_[eligible_idx])]
        best_C = lasso_cv.C_[0]
        coef = lasso_cv.coef_[0]
        selected_idx = np.where(coef != 0)[0]
        selected_features = X.columns[selected_idx].tolist()
        if len(selected_features) > 12:
            coef_abs = np.abs(coef[selected_idx])
            top_idx_relative = np.argsort(coef_abs)[-12:]
            selected_features = [selected_features[i] for i in top_idx_relative]
        plot_academic_lasso(lasso_cv, X.columns, target)
        plot_lasso_trajectories(X, y, target)
        all_outcomes_features[target] = {
            "n_features": len(selected_features),
            "features": selected_features,
            "weights": {f: round(float(coef[X.columns.get_loc(f)]), 4) for f in selected_features},
            "best_C": float(best_C),
            "best_lambda": float(1/best_C)
        }
        print(f"🎯 选定特征: {selected_features}")
        individual_path = os.path.join(TARGET_ARTIFACTS, "selected_features.json")
        with open(individual_path, "w") as f:
            json.dump(all_outcomes_features[target], f, indent=4)
    selected_path = os.path.join(ARTIFACTS_DIR, "selected_features.json")
    with open(selected_path, "w", encoding='utf-8') as f:
        json.dump(all_outcomes_features, f, ensure_ascii=False, indent=4)
    print(f"\n📂 资产已固化至: {selected_path}")
    print("下一步：进入 06_model_training_main.py")

if __name__ == "__main__":
    run_lasso_selection_flow()
