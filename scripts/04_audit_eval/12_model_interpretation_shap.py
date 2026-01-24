import os
import joblib
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import shap
import warnings

# 忽略 SHAP 与 Matplotlib 交互时的版本警告
warnings.filterwarnings("ignore")

# ===================== 配置 =====================
BASE_DIR = "../.."
MODEL_ROOT = os.path.join(BASE_DIR, "artifacts/models")
FIGURE_DIR = os.path.join(BASE_DIR, "results/figures/interpretation")
INTERP_DIR = os.path.join(FIGURE_DIR, "shap_values")

os.makedirs(INTERP_DIR, exist_ok=True)

TARGETS = ['pof', 'mortality', 'composite']
RANDOM_STATE = 42

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'axes.unicode_minus': False,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
})

def compute_shap_values(X, model, explainer_type):
    """计算 SHAP 值：统一处理 Tree 和 Linear 解释器"""
    print(f"[*] Calculating SHAP values ({explainer_type})...")
    if explainer_type == "tree":
        explainer = shap.TreeExplainer(model)
        # 兼容不同版本的 SHAP 返回格式
        shap_res = explainer(X)
        if len(shap_res.shape) == 3: # 针对某些二分类输出
            shap_res = shap_res[:, :, 1]
    else:
        explainer = shap.Explainer(model, X)
        shap_res = explainer(X)
    return shap_res

def save_shap_data(shap_values, X, target):
    """保存 SHAP 值及对应的原始特征值，便于后续统计建模"""
    shap_df = pd.DataFrame(shap_values.values, columns=X.columns)
    raw_df = X.reset_index(drop=True)
    raw_df.columns = [f"raw_{col}" for col in raw_df.columns]
    combined_df = pd.concat([shap_df, raw_df], axis=1)
    save_path = os.path.join(INTERP_DIR, f"SHAP_Data_Export_{target}.csv")
    combined_df.to_csv(save_path, index=False)
    base_val = shap_values.base_values[0] if isinstance(shap_values.base_values, np.ndarray) else shap_values.base_values
    with open(os.path.join(INTERP_DIR, f"SHAP_BaseValue_{target}.txt"), "w") as f:
        f.write(str(base_val))
    print(f"  -> SHAP 原始数据已导出: {save_path}")

def plot_shap_summary(shap_values, X, target):
    """医学蜂群图：全局特征贡献排序"""
    plt.figure(figsize=(9, 7), dpi=300)
    
    # 使用自定义配色：蓝色(低值)到红色(高值)
    shap.summary_plot(
        shap_values, X, plot_type="dot", max_display=15,
        show=False, color_bar=True, plot_size=None
    )
    
    # 细节微调
    plt.title(f"Impact on {target.upper()} Risk", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=11, labelpad=10)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # 自动保存 PDF 和 PNG
    base_n = os.path.join(INTERP_DIR, f"Fig4A_Summary_{target}")
    plt.savefig(f"{base_n}.pdf", bbox_inches='tight')
    plt.savefig(f"{base_n}.png", bbox_inches='tight', dpi=600)
    plt.close()

def plot_shap_force(shap_values, X, target):
    """个体决策图：解析高风险患者的特征贡献"""
    # 选取预测对数几率最高的样本
    risk_scores = shap_values.values.sum(axis=1)
    idx = np.argmax(risk_scores)
    
    # 渲染个体预测解释
    plt.figure(figsize=(14, 3))
    shap.force_plot(
        shap_values[idx].base_values,
        shap_values[idx].values,
        X.iloc[idx].round(2),
        matplotlib=True, show=False,
        text_rotation=15, contribution_threshold=0.03
    )
    
    plt.title(f"Individual Prediction Logic: High-Risk {target.upper()}", fontsize=12, pad=25)
    
    base_n = os.path.join(INTERP_DIR, f"Fig4B_Force_{target}")
    plt.savefig(f"{base_n}.pdf", bbox_inches='tight')
    plt.savefig(f"{base_n}.png", bbox_inches='tight', dpi=600)
    plt.close()

def plot_shap_dependence(shap_values, X, target):
    """依赖关系图：分析关键特征的非线性趋势与交互影响"""
    # 选取 SHAP 绝对值最大的前 3 个特征
    imp = np.abs(shap_values.values).mean(0)
    top_indices = np.argsort(imp)[-3:][::-1]
    top_feats = X.columns[top_indices]

    for feat in top_feats:
        # 创建画布，模仿 Seaborn 风格
        fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)
        
        # 绘制依赖图：自动寻找最强交互特征作为颜色区分
        shap.dependence_plot(
            feat, shap_values.values, X, show=False, ax=ax,
            interaction_index='auto', alpha=0.7, dot_size=25
        )
        
        ax.set_title(f"Non-linear Risk Trend: {feat}", fontsize=12, fontweight='bold')
        ax.grid(color='whitesmoke', linestyle='-', linewidth=0.5, zorder=0)
        ax.set_facecolor('white')
        
        # 保存
        f_clean = str(feat).replace("/", "_").replace(" ", "_").replace(">", "gt").replace("<", "lt")
        base_n = os.path.join(INTERP_DIR, f"Fig4C_Dep_{target}_{f_clean}")
        plt.savefig(f"{base_n}.png", bbox_inches='tight', dpi=600)
        plt.close()

def load_eval_and_model(target):
    """加载测试集和最佳模型，适配 CalibratedClassifierCV 结构"""
    target_dir = os.path.join(MODEL_ROOT, target.lower())
    
    eval_path = os.path.join(target_dir, "eval_data.pkl")
    models_path = os.path.join(target_dir, "all_models_dict.pkl")
    
    if not all(os.path.exists(p) for p in [eval_path, models_path]):
        raise FileNotFoundError(f"缺少资产: {target}")
    
    eval_data = joblib.load(eval_path)
    models = joblib.load(models_path)
    
    # 优先使用 XGBoost（TreeExplainer 性能最佳）
    if "XGBoost" in models:
        cal_model = models["XGBoost"]
        # [关键修复]：从校准容器中提取原始 XGBoost 实例
        if hasattr(cal_model, "calibrated_classifiers_"):
            model = cal_model.calibrated_classifiers_[0].estimator
        else:
            model = cal_model
        explainer_type = "tree"
    else:
        print(f"{target} 无 XGBoost，使用 Logistic 替代")
        model = models.get("Logistic Regression")
        explainer_type = "linear"
    
    X_test = eval_data['X_test_raw']  # 使用原始尺度进行解释，提高临床可读性
    y_test = eval_data['y_test']
    
    return X_test, y_test, model, explainer_type

def main():
    print("="*70)
    print("启动模块 12: SHAP 模型解释性分析 (适配校准模型)")
    print("="*70)
    
    for target in TARGETS:
        print(f"\n>>> 正在生成结局解释: {target.upper()}")
        try:
            # 1. 加载资产并处理嵌套的校准模型
            X_test, y_test, model, explainer_type = load_eval_and_model(target)
            
            # 2. 计算 SHAP 值
            shap_vals = compute_shap_values(X_test, model, explainer_type)
            save_shap_data(shap_vals, X_test, target)
            # 3. 生成三大标准解释图
            plot_shap_summary(shap_vals, X_test, target)
            plot_shap_force(shap_vals, X_test, target)
            plot_shap_dependence(shap_vals, X_test, target)
            
        except Exception as e:
            print(f"  [失败] {target}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*70)
    print("12 步完成！所有结局的 SHAP 临床解释已保存至结果目录。")
    print("下一步：执行 13_clinical_calibration_dca.py 进行获益评估。")

if __name__ == "__main__":
    main()
