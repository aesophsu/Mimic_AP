import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import shap

# 配置路径
MODEL_DIR = "../models"
SAVE_DIR = "../models/plots"
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

def calculate_net_benefit(y_true, y_prob, thresh):
    y_pred = (y_prob >= thresh).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    n = len(y_true)
    if thresh >= 1.0: return 0
    return (tp / n) - (fp / n) * (thresh / (1 - thresh))

def run_module_04():
    print("="*60)
    print("🚀 运行模块 04: 修复 SHAP 排序错误与可视化")
    print("="*60)

    # 1. 加载数据
    all_models = joblib.load(os.path.join(MODEL_DIR, "all_models.pkl"))
    selected_features = joblib.load(os.path.join(MODEL_DIR, "selected_features.pkl"))
    # 将特征名转为 numpy 数组以修复 TypeError
    feature_names_arr = np.array(selected_features)
    
    X_main, y_main = joblib.load(os.path.join(MODEL_DIR, "test_data_main.pkl"))
    X_sub, y_sub = joblib.load(os.path.join(MODEL_DIR, "test_data_subgroup.pkl"))

    # --------------------------------------------------------
    # 图 1: SVM 在人群间的性能稳健性
    # --------------------------------------------------------
    plt.figure(figsize=(8, 7))
    model_svm = all_models['SVM']
    
    for (X, y), label, color in [((X_main, y_main), 'Main Cohort', 'darkblue'), 
                                 ((X_sub, y_sub), 'No-Renal Subgroup', 'crimson')]:
        X_np = X.values if hasattr(X, 'values') else X
        y_prob = model_svm.predict_proba(X_np)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=3, label=f'{label} (AUC = {roc_auc:.3f})')
        print(f"📊 SVM - {label} AUC: {roc_auc:.4f}")

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.title('Best Model (SVM) Performance Comparison')
    plt.savefig(os.path.join(SAVE_DIR, "01_SVM_Performance.png"), dpi=300)
    print("✅ 已保存: 01_SVM_Performance.png")

    # --------------------------------------------------------
    # 图 2: SHAP 解释 (修正 Feature Names 索引问题)
    # --------------------------------------------------------
    print("🧪 正在生成 SHAP Bee-swarm Plot...")
    # 使用 Random Forest 提取 SHAP (性能与 SVM 极度接近，且具有原生 TreeExplainer)
    rf_calibrated = all_models['Random Forest']
    rf_raw = rf_calibrated.calibrated_classifiers_[0].estimator
    
    explainer = shap.TreeExplainer(rf_raw)
    X_main_np = X_main.values if hasattr(X_main, 'values') else X_main
    shap_values = explainer.shap_values(X_main_np)
    
    # 针对二分类 RandomForest：shap_values 是一个列表 [class0_values, class1_values]
    # 我们关注正类 (POF发生)
    if isinstance(shap_values, list):
        target_shap = shap_values[1]
    else:
        target_shap = shap_values

    plt.figure(figsize=(10, 8))
    # 显式传递 numpy 格式的 feature_names
    shap.summary_plot(
        target_shap, 
        X_main_np, 
        feature_names=feature_names_arr, 
        plot_type="dot", 
        show=False
    )
    plt.title('Feature Impact on Organ Failure Risk (SHAP Values)')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "02_SHAP_Summary.png"), dpi=300)
    print("✅ 已保存: 02_SHAP_Summary.png")

    # --------------------------------------------------------
    # 图 3: DCA 决策曲线
    # --------------------------------------------------------
    print("🧪 正在生成 DCA 曲线...")
    plt.figure(figsize=(8, 7))
    thresholds = np.arange(0, 0.8, 0.01)
    
    # 获取 SVM 概率
    X_main_np = X_main.values if hasattr(X_main, 'values') else X_main
    y_prob_svm = model_svm.predict_proba(X_main_np)[:, 1]
    
    nb_model = [calculate_net_benefit(y_main, y_prob_svm, t) for t in thresholds]
    prevalence = np.mean(y_main)
    nb_all = [prevalence - (1 - prevalence) * (t / (1 - t)) for t in thresholds]
    
    plt.plot(thresholds, nb_model, color='red', lw=2, label='Proposed SVM Model')
    plt.plot(thresholds, nb_all, color='black', linestyle=':', label='Treat All')
    plt.axhline(y=0, color='gray', label='Treat None')
    
    plt.ylim(-0.05, prevalence + 0.1)
    plt.xlim(0, 0.7)
    plt.xlabel('Risk Threshold')
    plt.ylabel('Net Benefit')
    plt.legend()
    plt.title('Decision Curve Analysis for Clinical Utility')
    plt.savefig(os.path.join(SAVE_DIR, "03_DCA_Curve.png"), dpi=300)
    print("✅ 已保存: 03_DCA_Curve.png")

    print("="*60)
    print(f"🎉 模块 04 运行成功！图表位于: {SAVE_DIR}")

if __name__ == "__main__":
    run_module_04()
