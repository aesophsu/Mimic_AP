import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dcurves import dca
from sklearn.calibration import calibration_curve
import traceback

# ===================== 配置 =====================
BASE_DIR = "../.."
MODEL_ROOT = os.path.join(BASE_DIR, "artifacts/models")
EICU_DIR = os.path.join(BASE_DIR, "data/external")
FIGURE_DIR = os.path.join(BASE_DIR, "results/figures/clinical")
os.makedirs(FIGURE_DIR, exist_ok=True)

TARGETS = ['pof', 'mortality', 'composite']

# 统一出版风格字体
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'axes.unicode_minus': False
})

def load_model_and_data(target):
    """加载模型和 eICU 数据"""
    target_dir = os.path.join(MODEL_ROOT, target.lower())
    models_path = os.path.join(target_dir, "all_models_dict.pkl")
    # 假设 eICU 验证集已准备好
    eicu_path = os.path.join(EICU_DIR, f"eicu_processed_{target.lower()}.csv")
    
    if not os.path.exists(models_path):
        raise FileNotFoundError(f"模型资产缺失: {target}")
    
    models = joblib.load(models_path)
    # 这里的 df_eicu 应该是你在 Module 11 中生成的外部验证 DataFrame
    df_eicu = pd.read_csv(eicu_path)
    
    y_true = df_eicu[target].values
    # 确保特征列与训练时一致
    eval_data = joblib.load(os.path.join(target_dir, "eval_data.pkl"))
    features = eval_data['features']
    X = df_eicu[features] 
    
    return models, X, y_true

def plot_dca(dca_res, data_df, target):
    """
    医学出版级 DCA 与 Calibration 组合绘图
    功能：展示临床净获益 (Net Benefit) 并叠加模型可靠性 (Calibration)
    """
    # 1. 数据预处理
    plot_df = dca_res.copy()
    plot_df['model'] = plot_df['model'].str.replace('_prob', '', regex=False)
    
    fig, ax1 = plt.subplots(figsize=(8, 7), dpi=300)
    
    # 2. 绘制 DCA 基准线
    sns.lineplot(data=plot_df[plot_df['model'] == 'all'], x='threshold', y='net_benefit', 
                 color='black', linestyle='--', lw=1.2, ax=ax1)
    sns.lineplot(data=plot_df[plot_df['model'] == 'none'], x='threshold', y='net_benefit', 
                 color='black', lw=1.2, ax=ax1)
    
    # 3. 绘制预测模型 DCA 曲线
    pred_models = plot_df[~plot_df['model'].isin(['all', 'none'])]
    sns.lineplot(data=pred_models, x='threshold', y='net_benefit', 
                 hue='model', lw=2.5, palette="Set1", ax=ax1)
    
    # 4. 添加显式文本标注 (末端对齐)
    ax1.text(0.71, plot_df[plot_df['model'] == 'all']['net_benefit'].iloc[-5], 
             'Treat All', fontsize=9, fontweight='bold')
    ax1.text(0.71, 0.01, 'Treat None', fontsize=9, fontweight='bold')
    
    # 5. 叠加校准曲线 (使用次坐标轴)
    # 自动获取第一个主预测模型的概率列进行展示
    main_model_col = [c for c in data_df.columns if '_prob' in c][0]
    prob_true, prob_pred = calibration_curve(data_df['outcome'], data_df[main_model_col], n_bins=10)
    
    ax2 = ax1.twinx()
    ax2.plot(prob_pred, prob_true, 'o--', color='purple', alpha=0.4, label='Calibration (Main Model)')
    ax2.set_ylabel('Observed Fraction (Calibration)', color='purple', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.set_ylim(0, 1)

    # 6. 出版级样式美化
    ax1.set_title(f"Clinical Decision & Reliability: {target.upper()}", fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel("Threshold Probability", fontsize=11)
    ax1.set_ylabel("Net Benefit", fontsize=11)
    ax1.set_xlim(0, 0.75)
    ax1.set_ylim(-0.015, plot_df['net_benefit'].max() * 1.15)
    
    ax1.legend(frameon=False, loc='upper right', bbox_to_anchor=(0.95, 0.95))
    ax1.grid(axis='y', color='whitesmoke', linestyle='--', zorder=0)
    sns.despine(ax=ax1, right=False) # 保留右侧轴以显示校准刻度
    
    # 7. 双格式导出
    base_n = os.path.join(FIGURE_DIR, f"Fig5_DCA_Calibration_{target}")
    plt.savefig(f"{base_n}.pdf", bbox_inches='tight')
    plt.savefig(f"{base_n}.png", bbox_inches='tight', dpi=600)
    plt.close()
    
    print(f"[*] Combined DCA-Calibration plot saved: {base_n}.pdf")

def compute_net_benefit(models, X_scaled, y_true, target):
    """
    执行 DCA 计算
    """
    print(f"[*] Calculating Net Benefit for {target.upper()}...")
    
    data_df = pd.DataFrame({'outcome': y_true})
    predictor_cols = []
    X_input = X_scaled.values if hasattr(X_scaled, 'values') else X_scaled
    
    for name, model in models.items():
        prob_col = f"{name}_prob"
        # 使用 X_input (numpy array) 进行预测，消除警告
        data_df[prob_col] = model.predict_proba(X_input)[:, 1]
        predictor_cols.append(prob_col)

    thresholds = np.arange(0, 0.76, 0.01)

    # 位置参数调用 (保持之前成功的逻辑)
    try:
        dca_res = dca(data_df, 'outcome', predictor_cols, thresholds)
    except Exception:
        try:
            dca_res = dca(data=data_df, outcome='outcome', model_names=predictor_cols, thresholds=thresholds)
        except:
            dca_res = dca(data=data_df, outcome='outcome', predictors=predictor_cols, thresholds=thresholds)
    
    return dca_res, data_df


def main():
    print("="*70)
    print("Module 13: Clinical Utility (DCA) & External Validation")
    print("="*70)
    
    for target in TARGETS:
        print(f"\n>>> Analyzing Outcome: {target.upper()}")
        try:
            models, X_scaled, y_true = load_model_and_data(target)
            dca_results, eval_df = compute_net_benefit(models, X_scaled, y_true, target)
            plot_dca(dca_results, eval_df, target)
            
            # 保存数据以便后续分析
            dca_results.to_csv(os.path.join(FIGURE_DIR, f"DCA_Data_{target}.csv"), index=False)
            
        except Exception:
            print(f"  [Critical Error] Failed for {target}:")
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()
