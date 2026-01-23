import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
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
    
def plot_feature_importance(features, weights, target):
    if not features:
        return

    features = np.array(features)
    weights = np.array(weights)
    sorted_idx = np.argsort(np.abs(weights))
    
    # 1. 字体与画布设置：使用无衬线字体，增加边距
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(8, 10 * (len(features)/15)), dpi=300)

    # 2. 颜色选择：医学论文经典的“低饱和度红蓝”
    # 红色 (#d62728): 危险因素/正相关; 蓝色 (#1f77b4): 保护因素/负相关
    colors = ['#d62728' if w > 0 else '#1f77b4' for w in weights[sorted_idx]]
    
    # 3. 绘图：减小条形高度 (height) 使其看起来更精致
    bars = ax.barh(range(len(features)), weights[sorted_idx], color=colors, 
                   edgecolor='white', linewidth=0.5, height=0.7)
    
    # 4. 坐标轴美化
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features[sorted_idx], fontsize=10, fontweight='medium')
    ax.axvline(0, color='black', lw=1.2, zorder=3) # 加粗零线
    
    # 5. 移除上方和右方的边框 (Spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 5)) # 让 Y 轴稍微离开条形
    
    # 6. 数据标注：更精细的控制
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{width:.3f}',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5 if width > 0 else -5, 0),
                    textcoords="offset points",
                    ha='left' if width > 0 else 'right', 
                    va='center', fontsize=9, fontweight='bold',
                    color=bar.get_facecolor())

    # 7. 标签与标题
    ax.set_xlabel('Regression Coefficient (Standardized)', fontsize=11, fontweight='bold')
    # 标题通常在论文中通过 Figure Legend 描述，图内标题建议简洁
    ax.set_title(f'Predictors for {target.upper()}', loc='left', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # 添加轻微的垂直网格线
    ax.grid(axis='x', linestyle='--', alpha=0.4, zorder=0)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"lasso_importance_{target}.png"), 
                bbox_inches='tight', transparent=False, facecolor='white')
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

        # 结局类别审计
        classes = np.unique(y)
        print(f"📊 [结局审计] 类别分布: {classes}, 阳性样本数: {sum(y==1)}")
        if len(classes) != 2:
            print(f"❌ 错误: {target} 不是二分类结局，跳过。")
            continue

        lasso_cv = LogisticRegressionCV(
            Cs=100, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
            penalty='l1', solver='liblinear', scoring='roc_auc',
            random_state=42, max_iter=1000, n_jobs=1
        )
        lasso_cv.fit(X, y)

        # 1-SE 准则
        pos_class = 1
        scores_mean = lasso_cv.scores_[pos_class].mean(axis=0)
        scores_se = lasso_cv.scores_[pos_class].std(axis=0) / np.sqrt(lasso_cv.scores_[pos_class].shape[0])
        idx_max = np.argmax(scores_mean)
        target_score = scores_mean[idx_max] - scores_se[idx_max]
        eligible_indices = np.where(scores_mean >= target_score)[0]
        idx_1se = eligible_indices[np.argmin(lasso_cv.Cs_[eligible_indices])]
        best_C_1se = lasso_cv.Cs_[idx_1se]

        # 严格遵守 1-SE: 使用 best_C_1se 重新 fit 模型获取 coef

        final_lasso = LogisticRegression(
            C=best_C_1se, penalty='l1', solver='liblinear',
            random_state=42, max_iter=1000
        )
        final_lasso.fit(X, y)
        coef = final_lasso.coef_[0]

        selected_idx = np.where(coef != 0)[0]
        selected_features = X.columns[selected_idx].tolist()

        if len(selected_features) > 12:
            coef_abs = np.abs(coef[selected_idx])
            top_idx = np.argsort(coef_abs)[-12:]
            selected_features = [selected_features[i] for i in top_idx]

        # 可视化
        plot_academic_lasso(lasso_cv, X.columns, target)
        plot_feature_importance(selected_features, [all_outcomes_features[target]["weights"][f] for f in selected_features], target)

        all_outcomes_features[target] = {
            "n_features": len(selected_features),
            "features": selected_features,
            "weights": {f: round(float(coef[X.columns.get_loc(f)]), 4) for f in selected_features},
            "best_C": float(best_C_1se),
            "best_lambda": float(1 / best_C_1se)
        }

        print(f"🎯 选定特征 ({len(selected_features)} 个): {', '.join(selected_features)}")

        # 独立保存
        with open(os.path.join(TARGET_ARTIFACTS, "selected_features.json"), "w", encoding='utf-8') as f:
            json.dump(all_outcomes_features[target], f, ensure_ascii=False, indent=4)

    # 全局保存
    selected_path = os.path.join(ARTIFACTS_DIR, "selected_features.json")
    with open(selected_path, "w", encoding='utf-8') as f:
        json.dump(all_outcomes_features, f, ensure_ascii=False, indent=4)

    print(f"\n📂 全局特征清单已固化至: {selected_path}")
    print("下一步：进入 06_model_training_main.py")

if __name__ == "__main__":
    run_lasso_selection_flow()
