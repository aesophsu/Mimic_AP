import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp

# ===================== 配置路径 =====================
BASE_DIR = "../.."
MIMIC_PROCESSED = os.path.join(BASE_DIR, "data/cleaned/mimic_processed.csv")
EICU_PROCESSED_DIR = os.path.join(BASE_DIR, "data/external")
VALIDATION_DIR = os.path.join(BASE_DIR, "validation")
FIGURE_DIR = os.path.join(BASE_DIR, "results/figures/comparison")

os.makedirs(VALIDATION_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# 结局列表（与 06/09 步一致）
TARGETS = ['pof', 'mortality', 'composite']

def load_processed_data(target):
    """加载数据并自动对齐共有特征"""
    mimic_path = MIMIC_PROCESSED
    eicu_path = os.path.join(EICU_PROCESSED_DIR, f"eicu_processed_{target}.csv")
    
    if not os.path.exists(mimic_path) or not os.path.exists(eicu_path):
        raise FileNotFoundError(f"缺失文件: {mimic_path} 或 {eicu_path}")
    
    df_mimic = pd.read_csv(mimic_path)
    df_eicu = pd.read_csv(eicu_path)
    
    # 1. 排除非预测特征
    exclude = [
        'pof', 'mortality', 'composite', 'subgroup_no_renal', 
        'gender', 'malignant_tumor', 'mechanical_vent_flag', 
        'vaso_flag', 'dialysis_flag', 'uniquepid', 'patientunitstayid'
    ]
    
    # 2. 动态寻找共有特征 (Intersection)
    # 这一步会自动过滤掉那些在 eICU 中没被包含的特征（如 sofa_score 等）
    common_features = [c for c in df_eicu.columns if c in df_mimic.columns and c not in exclude]
    
    # 3. 确保是数值型
    features = [c for c in common_features if pd.api.types.is_numeric_dtype(df_mimic[c])]
    
    print(f"  对齐成功: 结局 [{target.upper()}] 共有 {len(features)} 个特征参与漂移分析")
    
    return df_mimic[features], df_eicu[features], features

def ks_drift_test(mimic_series, eicu_series):
    """KS 测试 + 效应量"""
    if len(mimic_series.dropna()) < 5 or len(eicu_series.dropna()) < 5:
        return {"statistic": np.nan, "pvalue": np.nan, "drift": "样本不足"}
    
    ks_stat, p_value = ks_2samp(mimic_series.dropna(), eicu_series.dropna())
    drift_level = "显著" if p_value < 0.05 else "不显著"
    return {
        "ks_statistic": round(ks_stat, 4),
        "p_value": round(p_value, 4),
        "drift_significant": drift_level,
        "max_diff": round(ks_stat, 4)  # KS 统计量本身即最大累积差异
    }

def plot_distribution_comparison(mimic_series, eicu_series, feature_name, target):
    """绘制分布对比图（密度图 + KS 统计）"""
    plt.figure(figsize=(10, 6), dpi=300)
    sns.kdeplot(mimic_series.dropna(), label='MIMIC (Train)', color='#1f77b4', linewidth=2)
    sns.kdeplot(eicu_series.dropna(), label='eICU (Validation)', color='#ff7f0e', linewidth=2)
    
    plt.title(f'Distribution Comparison: {feature_name}\n({target.upper()})', fontsize=14, fontweight='bold')
    plt.xlabel(feature_name, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=10)
    sns.despine()
    
    # 添加 KS 统计文本
    ks_result = ks_drift_test(mimic_series, eicu_series)
    plt.text(0.02, 0.95, f"KS Statistic: {ks_result['ks_statistic']:.4f}\np-value: {ks_result['p_value']:.4f}",
             transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    save_path = os.path.join(FIGURE_DIR, f"dist_drift_{feature_name}_{target}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return save_path

def main():
    print("="*70)
    print("启动模块 10: 跨队列漂移分析 (MIMIC vs eICU)")
    print("="*70)

    drift_summary = {}

    for target in TARGETS:
        print(f"\n分析结局: {target.upper()}")
        try:
            df_mimic, df_eicu, features = load_processed_data(target)
        except Exception as e:
            print(f"  加载失败: {e}")
            continue

        drift_summary[target] = {}
        top_drift_features = []

        for feature in features:
            if feature not in df_mimic.columns or feature not in df_eicu.columns:
                continue

            mimic_vals = df_mimic[feature]
            eicu_vals = df_eicu[feature]

            ks_result = ks_drift_test(mimic_vals, eicu_vals)
            drift_summary[target][feature] = ks_result

            # 记录显著漂移特征（用于绘图）
            if ks_result['p_value'] < 0.05:
                top_drift_features.append((feature, ks_result['ks_statistic']))

            # 绘制 Top 漂移特征的分布对比图（可选：只画 p<0.05 的前 10 个）
            if ks_result['p_value'] < 0.05:
                plot_distribution_comparison(mimic_vals, eicu_vals, feature, target)

        # 保存该结局的漂移结果
        drift_summary[target]['summary'] = {
            "total_features": len(features),
            "significant_drift": len([f for f in drift_summary[target] if drift_summary[target][f]['p_value'] < 0.05]),
            "top_drift": sorted(top_drift_features, key=lambda x: x[1], reverse=True)[:10]
        }

    # 全局保存漂移报告
    drift_json_path = os.path.join(VALIDATION_DIR, "eicu_vs_mimic_drift.json")
    with open(drift_json_path, 'w', encoding='utf-8') as f:
        json.dump(drift_summary, f, ensure_ascii=False, indent=4)

    print(f"\n漂移分析完成！报告保存至: {drift_json_path}")
    print("下一步：进入 11_external_validation_perf.py（eICU 盲测性能评估）")

if __name__ == "__main__":
    main()
