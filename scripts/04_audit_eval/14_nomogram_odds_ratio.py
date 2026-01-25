import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
import matplotlib.ticker as ticker
from feature_utils import FeatureFormatter
from plot_utils import PlotUtils
from plot_config import PlotConfig

# ===================== 配置 =====================
BASE_DIR = "../.."
MODEL_ROOT = os.path.join(BASE_DIR, "artifacts/models")
FIGURE_DIR = os.path.join(BASE_DIR, "results/figures/clinical")
TABLE_DIR = os.path.join(BASE_DIR, "results/tables")

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

TARGETS = ['pof', 'mortality', 'composite']

# 设置绘图风格
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'], # 添加备选字体
    'axes.unicode_minus': False,
    'mathtext.fontset': 'stix'  # 使用类 Times New Roman 的数学字体，支持极好
})

def get_or_with_bootstrap(model, X_test, y_test, features, n_iterations=500):
    """
    真实的 Bootstrap 估算 OR 的 95% CI
    通过对原始测试集进行有放回抽样，计算系数的分布
    """
    print(f"[*] Starting Real Bootstrap for 95% CI (n={n_iterations})...")
    
    # 1. 提取底层逻辑回归模型
    if hasattr(model, 'calibrated_classifiers_'):
        base_model = model.calibrated_classifiers_[0].estimator
    else:
        base_model = model
    
    raw_coefs = base_model.coef_[0]
    raw_ors = np.exp(raw_coefs)
    
    # 2. 准备数据容器
    boot_coefs = []
    X_arr = X_test.values if hasattr(X_test, 'values') else X_test
    y_arr = np.array(y_test)
    
    # 3. 迭代重采样
    for i in range(n_iterations):
        try:
            X_res, y_res = resample(
                X_arr, y_arr,
                random_state=i,
                stratify=y_arr
            )
        except ValueError:
            continue
        
        # 使用与训练一致的参数拟合
        params = base_model.get_params()
        params.update({'max_iter': 2000})

        boot_lr = LogisticRegression(**params)
        
        try:
            boot_lr.fit(X_res, y_res)
            boot_coefs.append(boot_lr.coef_[0])
        except Exception:
            continue
            
        if (i + 1) % 100 == 0:
            print(f"    - Progress: {i + 1}/{n_iterations} iterations completed")

    # 4. 计算 2.5% 和 97.5% 分位数
    boot_coefs = np.array(boot_coefs)
    lower_coefs = np.percentile(boot_coefs, 2.5, axis=0)
    upper_coefs = np.percentile(boot_coefs, 97.5, axis=0)

    if boot_coefs.shape[0] < n_iterations * 0.8:
        print("[!] Warning: Too many bootstrap failures, CI may be unstable.")

    return pd.DataFrame({
        'Feature': features,
        'OR': raw_ors,
        'OR_Lower': np.exp(lower_coefs),
        'OR_Upper': np.exp(upper_coefs),
        'Coef': raw_coefs
    })

def plot_forest_or(or_df, target, formatter, lang='en', show_or_text=True):
    """
    Publication-grade forest plot with OR (95% CI) text annotation
    """
    # ===============================
    # 1. 排序 & 预处理
    # ===============================
    or_df = or_df.sort_values('OR', ascending=True).copy()

    plot_utils = PlotUtils(formatter, lang)
    labels = plot_utils.format_feature_labels(
        or_df['Feature'], with_unit=True
    )
    left_err, right_err = plot_utils.compute_or_error(or_df)
    x_min, x_max = plot_utils.compute_or_xlim(or_df)

    # ===============================
    # 2. 画布与基础 Forest Plot
    # ===============================
    plt.figure(figsize=PlotConfig.FOREST_FIGSIZE, dpi=PlotConfig.FIG_DPI)
    y_pos = np.arange(len(or_df))

    plt.errorbar(
        or_df['OR'],
        y_pos,
        xerr=[left_err, right_err],
        fmt='s',
        markersize=6,
        color=PlotConfig.OR_POINT_COLOR,
        ecolor=PlotConfig.OR_CI_COLOR,
        elinewidth=2,
        capsize=4
    )

    # Reference line (OR = 1)
    plt.axvline(
        1,
        color=PlotConfig.OR_REF_LINE_COLOR,
        linestyle='--',
        lw=1.2
    )

    # ===============================
    # 3. Y / X 轴设置
    # ===============================
    plt.yticks(y_pos, labels, fontsize=PlotConfig.TICK_FONT)
    plt.xscale('log')
    plt.xlabel('Odds Ratio (log scale)', fontsize=PlotConfig.LABEL_FONT)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks(PlotConfig.LOG_OR_TICKS)
    plt.xlim(x_min, x_max)

    # ===============================
    # 4. OR (95% CI) 文本显示（核心新增）
    # ===============================
    if show_or_text:
        # 文本列放在最右侧
        log_max = np.log10(or_df['OR_Upper'].max())
        text_x = 10 ** (log_max + 0.25)  # 向右移动 0.25 log units

        for y, (_, row) in zip(y_pos, or_df.iterrows()):
            or_text = PlotUtils.format_or_ci(
                row['OR'], row['OR_Lower'], row['OR_Upper']
            )
            plt.text(
                text_x, y,
                or_text,
                va='center',
                fontsize=PlotConfig.TICK_FONT,
                color='#222222'
            )
        plt.xlim(x_min, 10 ** (log_max + 0.45))

    # ===============================
    # 5. 标题与美化
    # ===============================
    title = f'Risk Factors for {target.upper()}'
    if lang != 'en':
        title += '（中文版）'

    plt.title(
        title,
        fontsize=PlotConfig.TITLE_FONT,
        fontweight='bold',
        pad=18
    )

    plt.grid(axis='x', alpha=0.2)
    plt.tight_layout()
    if show_or_text:
        plt.text(
            text_x, y_pos[-1] + 1,
            'OR (95% CI)',
            ha='left',
            va='bottom',
            fontsize=PlotConfig.TICK_FONT,
            fontweight='bold'
        )

    # ===============================
    # 6. 保存
    # ===============================
    save_fn = os.path.join(
        FIGURE_DIR,
        f"Forest_Plot_{target}_{lang}"
    )

    plt.savefig(f"{save_fn}.pdf", bbox_inches='tight')
    plt.savefig(
        f"{save_fn}.png",
        dpi=PlotConfig.SAVE_DPI,
        bbox_inches='tight'
    )
    plt.close()

    print(f"[*] Forest Plot saved: {save_fn}")

def plot_nomogram_standard(
    or_df,
    target,
    formatter,
    feature_ranges=None,
    intercept=0.0,
    max_coef=None,
    lang='en',
    top_k=10
):
    """
    Publication-grade clinical nomogram for logistic regression models

    Parameters
    ----------
    or_df : DataFrame
        Must contain: Feature, Coef
    feature_ranges : dict
        {feature: (min, max)} for physical scale ticks
    intercept : float
        Logistic regression intercept
    max_coef : float
        Maximum absolute coefficient for scaling points (if None, inferred)
    """

    # ===============================
    # 1. Feature selection & scaling
    # ===============================
    df = or_df.copy()
    df['abs_coef'] = df['Coef'].abs()

    top_df = (
        df.sort_values('abs_coef', ascending=False)
          .head(top_k)
          .copy()
    )

    if max_coef is None:
        max_coef = top_df['abs_coef'].max()

    # Linear point assignment (standard nomogram convention)
    top_df['Points_max'] = (top_df['abs_coef'] / max_coef) * 100
    top_df = top_df.sort_values('Points_max', ascending=True)

    n_feat = len(top_df)

    # ===============================
    # 2. Canvas
    # ===============================
    fig, ax = plt.subplots(figsize=(14, 12), dpi=300)

    # ===============================
    # 3. Global Points ruler (0–100)
    # ===============================
    main_y = n_feat + 1.2
    ax.hlines(main_y, 0, 100, lw=2, color='black')

    for p in range(0, 101, 10):
        ax.vlines(p, main_y, main_y + 0.25, lw=1.4, color='black')
        if p % 20 == 0:
            ax.text(
                p, main_y + 0.55, f"{p}",
                ha='center', va='bottom',
                fontsize=12, fontweight='bold'
            )
            ax.vlines(
                p, -5, main_y,
                lw=0.6, ls='--',
                color='gray', alpha=0.3, zorder=0
            )

    ax.text(
        -12, main_y + 0.2,
        'Points' if lang == 'en' else '评分',
        ha='right', va='center',
        fontsize=15, fontweight='bold'
    )

    # ===============================
    # 4. Feature-specific axes
    # ===============================
    for i, (_, row) in enumerate(top_df.iterrows()):
        feat = row['Feature']
        coef = row['Coef']
        max_pt = row['Points_max']

        ax.hlines(i, 0, max_pt, lw=1.6, color='black')

        # ---- Physical value ticks ----
        if feature_ranges and feat in feature_ranges:
            v_min, v_max = feature_ranges[feat]
            v_ticks = np.linspace(v_min, v_max, 5)

            p_ticks = (
                np.linspace(max_pt, 0, 5)
                if coef < 0 else
                np.linspace(0, max_pt, 5)
            )

            for v, p in zip(v_ticks, p_ticks):
                ax.vlines(p, i, i + 0.18, lw=1, color='black')
                ax.text(
                    p, i - 0.35, f"{v:.1f}",
                    ha='center', va='top',
                    fontsize=9, color='#333333'
                )

        # ---- Feature label (formatter) ----
        feat_label = formatter.get_label(
            feat, lang=lang, with_unit=True
        )
        ax.text(
            -12, i,
            feat_label,
            ha='right', va='center',
            fontsize=12, fontweight='bold'
        )

    # ===============================
    # 5. Total Points axis
    # ===============================
    tp_y = -2.3
    ax.hlines(tp_y, 0, 100, lw=2.6, color='darkred')

    ax.text(
        -12, tp_y,
        'Total Points' if lang == 'en' else '总评分',
        ha='right', va='center',
        fontsize=14, fontweight='bold',
        color='darkred'
    )

    for p in np.linspace(0, 100, 11):
        ax.vlines(p, tp_y, tp_y - 0.28, lw=2, color='darkred')
        ax.text(
            p, tp_y - 0.9,
            f"{int(p)}",
            ha='center',
            fontsize=11,
            fontweight='bold',
            color='darkred'
        )

    # ===============================
    # 6. Risk probability axis
    # ===============================
    prob_y = -4.8
    ax.hlines(prob_y, 0, 100, lw=2.6, color='darkblue')

    risk_label = (
        f'Risk of {target.upper()}'
        if lang == 'en'
        else f'{target.upper()} 发生风险'
    )

    ax.text(
        -12, prob_y,
        risk_label,
        ha='right', va='center',
        fontsize=14, fontweight='bold',
        color='darkblue'
    )

    probs = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    for j, p in enumerate(probs):
        logit_p = np.log(p / (1 - p))
        total_pt = (logit_p - intercept) / max_coef * 100

        if 0 <= total_pt <= 100:
            ax.vlines(
                total_pt,
                prob_y,
                prob_y + 0.28,
                lw=2,
                color='darkblue'
            )
            ax.text(
                total_pt,
                prob_y - (0.9 if j % 2 == 0 else 1.4),
                f"{p:.2f}",
                ha='center',
                fontsize=10,
                fontweight='bold',
                color='darkblue'
            )

    # ===============================
    # 7. Global styling
    # ===============================
    title = (
        f"Clinical Nomogram for Predicting {target.upper()}"
        if lang == 'en'
        else f"{target.upper()} 预测列线图"
    )

    ax.set_title(
        title,
        fontsize=20,
        fontweight='bold',
        pad=60
    )

    ax.set_xlim(-38, 112)
    ax.set_ylim(prob_y - 2.8, main_y + 2.8)
    ax.axis('off')

    # ===============================
    # 8. Save
    # ===============================
    plt.tight_layout()
    save_path = os.path.join(
        FIGURE_DIR,
        f"Nomogram_{target}_{lang}"
    )

    plt.savefig(f"{save_path}.pdf", bbox_inches='tight', transparent=True)
    plt.savefig(f"{save_path}.png", dpi=600, bbox_inches='tight')
    plt.close()

    print(f"[*] Nomogram ({lang}) saved: {save_path}")    

def load_raw_reference(file_path):
    """加载用于 Nomogram 刻度的原始物理数据"""
    if os.path.exists(file_path):
        raw_df = pd.read_csv(file_path)
        print(f"[*] Raw physical data loaded for Nomogram scales.")
        return raw_df
    else:
        print(f"[!] Warning: {file_path} not found. Scales will be empty.")
        return None

def calculate_feature_physical_ranges(or_results, raw_df):
    """计算 Top 10 特征的 1% - 99% 分位数物理范围"""
    feat_ranges = {}
    if raw_df is not None:
        # 提取系数绝对值最大的前 10 个特征
        top_feats = or_results.assign(abs_c=lambda x: x['Coef'].abs())\
                              .sort_values('abs_c', ascending=False)\
                              .head(10)['Feature']
        for f in top_feats:
            if f in raw_df.columns:
                # 使用分位数避免离群值导致坐标轴过长
                feat_ranges[f] = np.percentile(raw_df[f].dropna(), [1, 99])
    return feat_ranges

def process_single_target(target, raw_df):
    """处理单个结局指标的完整流水线"""
    print(f"\n>>> Processing Outcome: {target.upper()}")
    try:
        # 1. 资产加载
        target_dir = os.path.join(MODEL_ROOT, target.lower())
        models = joblib.load(os.path.join(target_dir, "all_models_dict.pkl"))
        eval_data = joblib.load(os.path.join(target_dir, "eval_data.pkl"))
        formatter = FeatureFormatter()
        model = models.get("Logistic Regression")
        if model is None:
            print(f"  [Skipped] Logistic Regression not found.")
            return

        X_test = eval_data['X_test_pre']
        y_test = eval_data['y_test']
        features = eval_data['features']
        
        # 2. 统计计算 (Bootstrap OR)
        or_results = get_or_with_bootstrap(model, X_test, y_test, features, n_iterations=500)
        
        # 保存 CSV 结果
        # 保存 CSV 和 JSON 结果 (JSON 便于后续论文补充和网页展示)
        table_fn = os.path.join(TABLE_DIR, f"OR_Statistics_{target}.csv")
        json_fn = os.path.join(TABLE_DIR, f"OR_Json_{target}.json")
        or_results.to_csv(table_fn, index=False)
        or_results.to_json(json_fn, orient='records', indent=4)
        
        # 3. 计算物理刻度范围
        feat_ranges = calculate_feature_physical_ranges(or_results, raw_df)
        
        # 4. 绘图 (plot_nomogram_standard 内部已包含 PDF 保存逻辑)
        # 4. 提取精确绘图所需的模型参数 (Intercept 和 Max Coef)
        if hasattr(model, 'calibrated_classifiers_'):
            base_model = model.calibrated_classifiers_[0].estimator
        else:
            base_model = model
            
        model_intercept = base_model.intercept_[0]
        # 获取 OR 结果中绝对值最大的系数，用于标准化点数基准
        max_abs_coef = or_results['Coef'].abs().max()

        # 5. 绘图
        plot_forest_or(or_results, target, formatter=formatter)
        plot_nomogram_standard(
            or_results, 
            target,
            formatter=formatter,
            feature_ranges=feat_ranges,
            intercept=model_intercept,
            max_coef=max_abs_coef
        )
        
        print(f"  [Success] Statistics and Figures generated for {target}.")
        
    except Exception as e:
        print(f"  [Critical Error] Failed for {target}: {str(e)}")
        # import traceback; traceback.print_exc() # 调试时可开启

def main():
    print("="*70)
    print("Module 14: Clinical Translation - Real Bootstrap OR & Nomogram")
    print("="*70)
    
    # 初始化：加载物理数据基准
    RAW_DATA_PATH = os.path.join(BASE_DIR, "data/cleaned/mimic_raw_scale.csv")
    raw_df = load_raw_reference(RAW_DATA_PATH)
    
    # 循环处理每个目标
    for target in TARGETS:
        process_single_target(target, raw_df)

    print("\n" + "="*70)
    print("Project Pipeline Complete! Check /figures/clinical/ and /tables/")
    print("="*70)

if __name__ == "__main__":
    main()
