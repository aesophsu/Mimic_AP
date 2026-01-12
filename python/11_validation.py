import os
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

# =========================================================
# 1. 加载资产
# =========================================================
BASE_DIR = ".."
SELECTED_FEATURES_PATH = os.path.join(BASE_DIR, "models/selected_features.pkl")
MIMIC_PATH = os.path.join(BASE_DIR, "data/cleaned/mimic_for_model.csv")

selected_features = joblib.load(SELECTED_FEATURES_PATH)
df_mimic = pd.read_csv(MIMIC_PATH)
X_selected = df_mimic[selected_features].fillna(df_mimic[selected_features].median())

def run_enhanced_collinearity_audit():
    print("="*60)
    print("🔬 核心特征共线性审计报告 (Clinical Feature Audit)")
    print("="*60)

    # ---------------------------------------------------------
    # A. 基础 Pearson 相关性分析
    # ---------------------------------------------------------
    corr_matrix = X_selected.corr()
    
    print("\n🚩 [Step 1] 高度相关特征对 (Pearson r > 0.5):")
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            r_val = corr_matrix.iloc[i, j]
            if abs(r_val) > 0.5:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], r_val))
                print(f"  - {corr_matrix.columns[i]:<15} vs {corr_matrix.columns[j]:<15} | r = {r_val:.4f}")
    
    if not high_corr_pairs:
        print("  ✅ 未发现显著共线性对，特征独立性良好。")

    # ---------------------------------------------------------
    # B. 多重共线性诊断 (VIF)
    # ---------------------------------------------------------
    # VIF > 5 或 10 通常认为存在严重共线性
    print("\n🚩 [Step 2] 多重共线性诊断 (Variance Inflation Factor):")
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_selected.columns
    vif_data["VIF"] = [variance_inflation_factor(X_selected.values, i) for i in range(len(X_selected.columns))]
    vif_data = vif_data.sort_values(by="VIF", ascending=False)
    
    for _, row in vif_data.iterrows():
        status = "⚠️ 高" if row['VIF'] > 5 else "✅ 稳健"
        print(f"  - {row['Feature']:<20} | VIF = {row['VIF']:>6.2f} | {status}")

    # ---------------------------------------------------------
    # C. 可视化：层级聚类热图 (Clustermap)
    # ---------------------------------------------------------
    # 聚类热图能直观显示哪些特征形成了“临床指标簇”
    plt.figure(figsize=(12, 10))
    g = sns.clustermap(corr_matrix, 
                       annot=True, 
                       fmt=".2f", 
                       cmap='RdBu_r', 
                       vmin=-1, vmax=1,
                       figsize=(10, 10))
    plt.title("Hierarchical Clustering of Core Features", y=1.02)
    
    save_path = os.path.join(BASE_DIR, "results/feature_collinearity_clustermap.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"\n📊 聚类热图已保存至: {save_path}")
    plt.show()

    # ---------------------------------------------------------
    # D. 临床解释建议输出
    # ---------------------------------------------------------
    print("\n📝 [Step 3] 论文讨论素材 (Clinical Interpretation Advice):")
    if any(v > 5 for v in vif_data['VIF']):
        print("  💡 提示：存在 VIF > 5 的特征。在讨论中应解释这些变量虽然数学上相关，")
        print("     但捕捉了患者不同生理维度的异常（如肾功能的代偿 vs 损伤）。")
    else:
        print("  💡 提示：所有特征 VIF 均处于理想水平。这增强了模型系数的可信度和解释性。")

if __name__ == "__main__":
    run_enhanced_collinearity_audit()
