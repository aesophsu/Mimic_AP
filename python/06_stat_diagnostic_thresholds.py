import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, precision_score, recall_score, f1_score

# =========================================================
# 配置路径
# =========================================================
BASE_DIR = ".."
MODEL_DIR = os.path.join(BASE_DIR, "models")
SAVE_DIR = os.path.join(BASE_DIR, "results")
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

def run_module_06():
    print("="*60)
    print("🚀 运行模块 06: 计算最佳截断值与临床诊断效能")
    print("="*60)

    # 1. 加载模型与测试数据
    # 我们选择表现最好的 SVM 模型
    all_models = joblib.load(os.path.join(MODEL_DIR, "all_models.pkl"))
    model = all_models['SVM']
    
    # 加载测试集
    X_test, y_test = joblib.load(os.path.join(MODEL_DIR, "test_data_main.pkl"))
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test

    # 2. 获取预测概率
    y_probs = model.predict_proba(X_test_np)[:, 1]

    # 3. 计算 ROC 曲线数据
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)

    # 4. 寻找约登指数最大值的索引 (Youden Index = Sensitivity + Specificity - 1)
    # 约登指数最大点即为最佳截断值
    youden_index = tpr + (1 - fpr) - 1
    best_idx = np.argmax(youden_index)
    best_threshold = thresholds[best_idx]
    
    # 5. 基于最佳截断值生成二分类预测
    y_pred = (y_probs >= best_threshold).astype(int)

    # 6. 计算详细性能指标
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    sensitivity = tp / (tp + fn)      # 灵敏度 (Recall)
    specificity = tn / (tn + fp)      # 特异度
    ppv = tp / (tp + fp)              # 阳性预测值 (Precision)
    npv = tn / (tn + fn)              # 阴性预测值
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = f1_score(y_test, y_pred)

    # 7. 打印结果
    print(f"🎯 最佳概率截断值 (Best Threshold): {best_threshold:.4f}")
    print("-" * 30)
    print(f"✅ 灵敏度 (Sensitivity):   {sensitivity:.4f}")
    print(f"✅ 特异度 (Specificity):   {specificity:.4f}")
    print(f"✅ 阳性预测值 (PPV):       {ppv:.4f}")
    print(f"✅ 阴性预测值 (NPV):       {npv:.4f}")
    print(f"✅ 总准确率 (Accuracy):    {accuracy:.4f}")
    print(f"✅ F1 分数 (F1-Score):     {f1:.4f}")

    # 8. 保存诊断指标到 CSV
    performance_metrics = {
        'Metric': ['Best Threshold', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'Accuracy', 'F1-Score'],
        'Value': [best_threshold, sensitivity, specificity, ppv, npv, accuracy, f1]
    }
    metrics_df = pd.DataFrame(performance_metrics)
    metrics_path = os.path.join(SAVE_DIR, "diagnostic_performance_svm.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # --------------------------------------------------------
    # 绘制带 Cut-off 点的 ROC 曲线
    # --------------------------------------------------------
    plt.figure(figsize=(8, 7))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {best_threshold:.2f})')
    plt.scatter(fpr[best_idx], tpr[best_idx], color='red', marker='o', s=100, 
                label=f'Best Cut-off: {best_threshold:.2f}')
    
    plt.annotate(f'Cut-off: {best_threshold:.2f}\n(Sen: {sensitivity:.2f}, Spe: {specificity:.2f})',
                 xy=(fpr[best_idx], tpr[best_idx]), xytext=(fpr[best_idx]+0.1, tpr[best_idx]-0.2),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=10)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curve with Optimal Cut-off')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plot_path = os.path.join(BASE_DIR, "models/plots/05_ROC_with_Cutoff.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\n📈 ROC 截断值可视化图表已保存至: {plot_path}")
    print(f"✅ 指标数据已保存至: {metrics_path}")

if __name__ == "__main__":
    run_module_06()
