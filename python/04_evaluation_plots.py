import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, brier_score_loss
import shap
import warnings

# 屏蔽不必要的 UserWarning 干扰
warnings.filterwarnings('ignore', category=UserWarning)

# 配置路径
MODEL_DIR = "../models"
FIG_DIR = "../figures"
if not os.path.exists(FIG_DIR): os.makedirs(FIG_DIR)

def calculate_net_benefit(y_true, y_prob, thresh):
    y_pred = (y_prob >= thresh).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    n = len(y_true)
    if thresh >= 1.0 or thresh <= 0: return 0
    return (tp / n) - (fp / n) * (thresh / (1 - thresh))

def run_module_04_debug_version():
    print("="*70)
    print("🚀 运行模块 04 增强审计版: 可视化与 SHAP 解释")
    print("="*70)

    # 1. 加载资产并打印审计信息
    print("📂 [Step 1/4] 正在加载模型与数据资产...")
    try:
        all_models = joblib.load(os.path.join(MODEL_DIR, "all_models.pkl"))
        selected_features = joblib.load(os.path.join(MODEL_DIR, "selected_features.pkl"))
        
        X_test, y_test = joblib.load(os.path.join(MODEL_DIR, "test_data_main.pkl"))
        X_test_np = X_test.values if hasattr(X_test, 'values') else X_test

        # [新增] 加载亚组数据用于对比审计
        X_sub, y_sub = joblib.load(os.path.join(MODEL_DIR, "test_data_sub.pkl"))
        X_sub_np = X_sub.values if hasattr(X_sub, 'values') else X_sub

        print(f"   ✅ 加载成功: 包含 {len(all_models)} 个模型")
        print(f"   ✅ 特征列表: {selected_features}")
        print(f"   ✅ 测试集维度: {X_test_np.shape}, POF 流行率: {np.mean(y_test):.2%}")
    except Exception as e:
        print(f"   ❌ 加载失败: {e}")
        return

    # --------------------------------------------------------
    # [图 1] 全模型 ROC 对比
    # --------------------------------------------------------
    print("\n📈 [Step 2/4] 正在绘制多模型 ROC 曲线并审计 AUC...")
    plt.figure(figsize=(9, 8))
    # --------------------------------------------------------
    # [Step 2] 同步模块 03 的审计数据
    # --------------------------------------------------------
    # 填入模块 03 打印的 Main AUC (95% CI)
    ci_data = {
        "XGBoost": "0.831 (0.771-0.882)",
        "SVM": "0.839 (0.782-0.888)",
        "Random Forest": "0.834 (0.777-0.885)",
        "Logistic Regression": "0.833 (0.775-0.884)",
        "Decision Tree": "0.818 (0.760-0.873)"
    }

    # [新增] 填入模块 03 打印的 No-Renal AUC (95% CI)
    sub_ci_data = {
        "XGBoost": "0.752 (0.645-0.853)",
        "SVM": "0.774 (0.656-0.880)",
        "Random Forest": "0.760 (0.647-0.861)",
        "Logistic Regression": "0.768 (0.656-0.862)",
        "Decision Tree": "0.745 (0.631-0.852)"
    }

    for name, clf in all_models.items():
        # 强制使用 numpy 数组预测，消除警告
        y_prob = clf.predict_proba(X_test_np)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(*roc_curve(y_test, y_prob)[:2])
    
        # [新增] 计算亚组性能
        y_prob_sub = clf.predict_proba(X_sub_np)[:, 1]
        roc_auc_sub = auc(*roc_curve(y_sub, y_prob_sub)[:2])
    
        # 修改打印信息，增加 Sub-AUC 审计
        print(f"   🔍 模型审计: {name:<20} | Test AUC: {roc_auc:.4f} | Sub-AUC: {roc_auc_sub:.4f}")
    
        display_label = f"{name}: {ci_data.get(name, f'{roc_auc:.3f}')}"
        plt.plot(fpr, tpr, lw=2, label=display_label)

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.2)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Predictive Performance Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(alpha=0.2)
    plt.savefig(os.path.join(FIG_DIR, "01_ROC_Comparison.png"), dpi=300)
    plt.close()

    # --------------------------------------------------------
    # [图 2] SHAP 解释 (针对 XGBoost)
    # --------------------------------------------------------
    print("\n🧪 [Step 3/4] 正在生成 XGBoost SHAP 解释 (针对修正标签后的审计)...")
    try:
        xgb_calibrated = all_models['XGBoost']
        xgb_raw = xgb_calibrated.calibrated_classifiers_[0].estimator
        
        explainer = shap.TreeExplainer(xgb_raw)
        # 使用 Numpy 数组以确保特征对应正确
        shap_values = explainer.shap_values(X_test_np)
        
        print(f"   ✅ SHAP 计算完成。SHAP 数组维度: {np.shape(shap_values)}")
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, 
            X_test_np, 
            feature_names=selected_features, 
            show=False,
            plot_type="dot"
        )
        plt.title('SHAP Feature Importance: Drivers of POF Risk', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "02_SHAP_Summary.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"   ⚠️ SHAP 生成跳过: {e}")

    # --------------------------------------------------------
    # Step 4: 全模型 DCA 临床价值审计 (修复索引错误并全量化)
    # --------------------------------------------------------
    print("\n⚖️ [Step 4/4] 正在执行全模型 DCA 临床价值审计...")
    plt.figure(figsize=(10, 8))
    thresholds = np.arange(0.01, 0.81, 0.01)
    
    # 基础参照线: Treat All (所有人都视为高危)
    prev = np.mean(y_test)
    nb_all = [prev - (1 - prev) * (t / (1 - t)) for t in thresholds]
    
    model_windows = {}
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']

    for (name, clf), color in zip(all_models.items(), colors):
        y_prob = clf.predict_proba(X_test_np)[:, 1]
        nb_model = [calculate_net_benefit(y_test, y_prob, t) for t in thresholds]
        
        # 精确计算获益窗口: Net Benefit > Treat All 且 Net Benefit > 0
        better_than_all = [t for t, nb, nba in zip(thresholds, nb_model, nb_all) if nb > nba and nb > 0]
        
        if better_than_all:
            win_min, win_max = min(better_than_all), max(better_than_all)
            window_str = f"{win_min:.1%} - {win_max:.1%}"
            model_windows[name] = window_str
            print(f"   ✅ {name:<20} | 获益窗口: {window_str}")
        else:
            model_windows[name] = "No Benefit"
            print(f"   ⚠️ {name:<20} | 未检测到获益区间")

        plt.plot(thresholds, nb_model, lw=2, color=color, label=f"{name} ({model_windows[name]})")

    # 绘制参考虚线
    plt.plot(thresholds, nb_all, color='black', linestyle=':', alpha=0.4, label='Treat All')
    plt.axhline(y=0, color='gray', lw=1, label='Treat None')
    
    plt.ylim(-0.05, prev + 0.1)
    plt.xlim(0, 0.8)
    plt.xlabel('Risk Threshold Probability (Cut-off)')
    plt.ylabel('Net Benefit')
    plt.title('Decision Curve Analysis: Comparative Utility', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(alpha=0.2)
    plt.savefig(os.path.join(FIG_DIR, "03_DCA_Full_Comparison.png"), dpi=300)
    plt.close()

    # --------------------------------------------------------
    # 总结输出 (Table 2 终极版)
    # --------------------------------------------------------
    print("\n" + "="*115)
    print(f"{'Algorithm':<20} | {'Main AUC (95% CI)':<25} | {'No-Renal AUC (95% CI)':<25} | {'DCA Window':<15}")
    print("-" * 115)
    for name in all_models.keys():
        main_val = ci_data.get(name, "N/A")
        sub_val = sub_ci_data.get(name, "N/A")
        window = model_windows.get(name, "N/A")
        print(f"{name:<20} | {main_val:<25} | {sub_val:<25} | {window:<15}")
    print("="*115)
    print(f"🎉 模块 04 运行成功！图表位于: {FIG_DIR}")
    
if __name__ == "__main__":
    run_module_04_debug_version()
