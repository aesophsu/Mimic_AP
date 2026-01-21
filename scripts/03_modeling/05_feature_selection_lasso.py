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
    all_outcomes_features = {}
     
    for target in targets:
        print(f"\n{'='*20} 正在精炼特征结局: {target.upper()} {'='*20}")
        
        # 2. 数据准备：剔除所有结局标签及 ID 类字段
        outcomes = ['pof', 'mortality_28d', 'composite_outcome', 'subgroup_no_renal',
                    'resp_pof', 'cv_pof', 'renal_pof']
        scores = ['sofa_score', 'apsiii', 'sapsii', 'oasis', 'lods']
        interventions = ['mechanical_vent_flag', 'vaso_flag']
        admin_vars = ['los', 'stay_id', 'hadm_id', 'subject_id']
        drop_cols = outcomes + scores + interventions + admin_vars
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
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
        plot_lasso_trajectories(log_alphas, coefs_path, X.columns, target)
        # 6. 特征提取与排序 (修正解包错误)
        coef_abs = np.abs(lasso.coef_)
        top_indices = np.argsort(coef_abs)[-12:] 
        
        # 显式提取特征名和对应的系数值
        selected_features = X.columns[top_indices].tolist()
        selected_coefs = lasso.coef_[top_indices].tolist() # 转为列表
        
        # 组合并排序：确保 zip 生成的是 (feature_name, weight) 的二元组
        feature_results = sorted(
            zip(selected_features, selected_coefs), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )

        print(f"\n✅ {target.upper()} 筛选完成 | 核心特征贡献度排行:")
        print("-" * 65)
        print(f"{'Rank':<5} | {'Feature Name':<25} | {'Weight':<10} | {'Impact'}")
        print("-" * 65)

        # 安全获取最大权重绝对值用于绘图
        max_w = max([abs(w) for name, w in feature_results]) if feature_results else 1
        
        for idx, (f, w) in enumerate(feature_results, 1):
            symbol = "▲ Risk" if w > 0 else "▼ Prot"
            bar_len = int(abs(w) / max_w * 10)
            bar = "█" * bar_len
            print(f"{idx:02d}   | {f:<25} | {w:>10.4f} | {symbol:<7} {bar}")
        
        print("-" * 65)

        all_outcomes_features[target] = {
            "n_features": len(selected_features),
            "features": [f for f, w in feature_results],
            "weights": {f: round(float(w), 4) for f, w in feature_results}
        }
        
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

def plot_lasso_trajectories(log_alphas, coefs_path, feature_names, target):
    """绘制 LASSO 系数随 Lambda 变化的轨迹图 (SCI 风格)"""
    plt.figure(figsize=(10, 7), dpi=300)
    
    # coefs_path 的形状通常是 (n_features, n_alphas)
    for i in range(coefs_path.shape[0]):
        plt.plot(log_alphas, coefs_path[i, :], label=feature_names[i] if np.max(np.abs(coefs_path[i, :])) > 0.05 else "")

    plt.axvline(log_alphas[0], color='black', linestyle=':', alpha=0.3)
    plt.xlabel(r'$\log_{10}(\lambda)$', fontsize=12)
    plt.ylabel('Coefficients', fontsize=12)
    plt.title(f'LASSO Regression Trajectories: {target.upper()}', fontsize=14, fontweight='bold')
    
    # 只显示最终入选或贡献较大的图例，避免图例过多遮挡图像
    # 如果特征太多，建议不显示 legend 或者只显示 Top 10
    # plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1), fontsize=8)
    
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"lasso_traj_{target}.png"), bbox_inches='tight')
    plt.close()
    
if __name__ == "__main__":
    run_lasso_selection_flow()
