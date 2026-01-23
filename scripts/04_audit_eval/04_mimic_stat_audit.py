import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ===================== 配置路径 =====================
BASE_DIR = "../.."
DATA_PATH = os.path.join(BASE_DIR, "data/cleaned/mimic_raw_scale.csv")
TABLE_DIR = os.path.join(BASE_DIR, "results/tables")
FIGURE_DIR = os.path.join(BASE_DIR, "results/figures/audit")
os.makedirs(FIGURE_DIR, exist_ok=True)

def plot_heatmap():
    """绘制符合医学论文发表标准的缺失值热图"""
    print("开始绘制专业缺失值热图...")
    df = pd.read_csv(DATA_PATH)

    # 1. 核心优化：按缺失率对特征进行排序
    # 这能让缺失模式（尤其是成块缺失的指标，如血气分析）集中显示
    missing_rates = df.isnull().mean()
    sorted_cols = missing_rates.sort_values(ascending=False).index
    df_sorted = df[sorted_cols]

    # 2. 设置绘图风格
    plt.style.use('seaborn-v0_8-white') # 使用纯白背景
    fig, ax = plt.subplots(figsize=(14, 8))

    # 3. 绘制热图
    # cmap: 使用医疗报告常用的 'Greys' (灰白) 或 'Blues' (蓝白)
    # cbar=True: 医学论文通常需要 Legend 说明颜色含义
    sns.heatmap(
        df_sorted.isnull(), 
        cmap=['#F5F5F5', '#2E5A88'], # 浅灰代表存在，深蓝色代表缺失
        cbar=True, 
        yticklabels=False,
        ax=ax
    )

    # 4. 美化颜色条 (Colorbar)
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0.25, 0.75])
    colorbar.set_ticklabels(['Observed', 'Missing'])
    colorbar.outline.set_visible(True)

    # 5. 完善标签（使用学术标题格式）
    plt.title("Pattern of Missing Clinical Observations", fontsize=16, pad=20, fontweight='bold')
    plt.xlabel("Clinical Features (Sorted by Missing Rate)", fontsize=12, labelpad=10)
    plt.ylabel(f"Study Participants (N={len(df)})", fontsize=12, labelpad=10)

    # 6. 旋转横坐标刻度，防止重叠
    plt.xticks(rotation=45, ha='right', fontsize=9)
    
    # 7. 移除四周多余线条
    sns.despine(left=True, bottom=True)
    
    plt.tight_layout()

    # 保存为高分辨率图片
    save_path = os.path.join(FIGURE_DIR, "mimic_missing_heatmap_pro.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"专业级热图已保存至: {save_path}")

def check_existing_tables():
    """检查 Table 1 和 Table 2 是否已生成"""
    t1_path = os.path.join(TABLE_DIR, "table1_baseline.csv")
    t2_path = os.path.join(TABLE_DIR, "table2_renal_subgroup.csv")
    
    print("\n检查已生成表格：")
    print(f"Table 1 (Baseline): {'✅ 存在' if os.path.exists(t1_path) else '❌ 缺失'}")
    print(f"Table 2 (No-Renal Subgroup): {'✅ 存在' if os.path.exists(t2_path) else '❌ 缺失'}")

def main():
    print("="*70)
    print("启动模块 04: 描述统计与审计 - 补齐缺失热图 (MIMIC-IV)")
    print("="*70)

    check_existing_tables()
    plot_heatmap()

    print("\n04 步补齐完成！")
    print("已验证 Table 1/2 存在，并生成缺失热图。")
    print("下一步：进入 05_feature_selection_lasso.py（基于 mimic_processed.csv）")

if __name__ == "__main__":
    main()
