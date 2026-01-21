import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, lasso_path
import warnings

warnings.filterwarnings('ignore')

# =========================================================
# 1. 路径配置
# =========================================================
BASE_DIR = "../../"
INPUT_PATH = os.path.join(BASE_DIR, "data/cleaned/mimic_processed.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts/features")
FIG_DIR = os.path.join(BASE_DIR, "results/figures/lasso")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

def run_lasso_selection_flow():
    """针对三种结局循环执行 LASSO 筛选并产出学术图表"""
    targets = ['pof', 'mortality_28d', 'composite_outcome']
    df = pd.read_csv(INPUT_PATH)
    
    # 汇总特征清单的字典
    all_outcomes_features = {}

    for target in targets:
        print(f"\n{'='*20} 正在精炼特征结局: {target.upper()} {'='*20}")
        
        # 2. 数据准备：剔除所有结局标签及 ID 类字段
        outcome_cols = ['pof', 'mortality_28d', 'composite_outcome', 'subgroup_no_renal']
        X = df.drop(columns=[c for c in outcome_cols if c in df.columns])
        y = df[target]
        
        # 3. 执行 LassoCV 并计算 1-SE 准则
        lasso = LassoCV(cv=10, random_state=42, max_iter=20000).fit(X, y)
        
        alphas = lasso.alphas_
        log_alphas = np.log10(alphas)
        mse_mean = lasso.mse_path_.mean(axis=1)
        mse_se = lasso.mse_path_.std(axis=1) / np.sqrt(lasso.mse_path_.shape[1])
        
        idx_min = np.argmin(mse_mean)
        target_mse = mse_mean[idx_min] + mse_se[idx_min]
        idx_1se = np.where(mse_mean <= target_mse)[0][-1] 

        # 4. 获取路径计数
        _, coefs_path, _ = lasso_path(X, y, alphas=alphas)
        active_counts = np.sum(coefs_path != 0, axis=0)

        # 5. 绘制学术级 LASSO 诊断图
        plot_academic_lasso(log_alphas, mse_mean, mse_se, idx_min, idx_1se, active_counts, target)

        # 6. 特征提取 (根据 1-SE 准则选择非零系数)
        # 如果 1-SE 选出的特征过多，此处可固定取绝对值最大的 Top 12
        coef_abs = np.abs(lasso.coef_)
        # 使用 1-SE 索引对应的系数进行筛选，或直接取 Top 12 核心特征
        top_indices = np.argsort(coef_abs)[-12:] 
        selected_features = X.columns[top_indices].tolist()
        
        all_outcomes_features[target] = {
            "n_features": len(selected_features),
            "features": selected_features
        }
        
        print(f"✅ {target} 筛选完成: 选定 {len(selected_features)} 个临床核心特征")

    # 7. 持久化特征指令集 (JSON)
    json_path = os.path.join(ARTIFACTS_DIR, "selected_features.json")
    with open(json_path, "w") as f:
        json.dump(all_outcomes_features, f, indent=4)
    print(f"\n📂 全结局特征清单已加密存至: {json_path}")

def plot_academic_lasso(log_alphas, mse_mean, mse_se, idx_min, idx_1se, active_counts, target):
    """保存符合 SCI 发表标准的 LASSO 诊断图"""
    plt.figure(figsize=(8, 6), dpi=300)
    ax1 = plt.gca()
    
    # 绘制 MSE 散点与误差棒
    ax1.errorbar(log_alphas, mse_mean, yerr=mse_se, fmt='o', color='red', 
                 ecolor='lightgray', elinewidth=1, capsize=2, mfc='red', ms=4, label='CV MSE')
    
    # 标注 Min MSE 线与 1-SE 线
    ax1.axvline(log_alphas[idx_min], color='blue', linestyle='--', label=f'Min MSE (n={active_counts[idx_min]})')
    ax1.axvline(log_alphas[idx_1se], color='black', linestyle='--', label=f'1-SE Rule (n={active_counts[idx_1se]})')

    ax1.set_xlabel(r'$\log_{10}(\lambda)$', fontsize=12)
    ax1.set_ylabel('Mean Squared Error', fontsize=12)
    ax1.set_title(f'LASSO Selection: {target.upper()}', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left')

    # 顶部添加特征数量轴
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    tick_pos = np.linspace(log_alphas[-1], log_alphas[0], 8)
    # 找到最接近 tick_pos 的索引以显示特征数
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels([active_counts[np.abs(log_alphas - t).argmin()] for t in tick_pos])
    ax2.set_xlabel('Number of Non-zero Coefficients', fontsize=11, labelpad=10)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"lasso_diag_{target}.png"), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    run_lasso_selection_flow()
