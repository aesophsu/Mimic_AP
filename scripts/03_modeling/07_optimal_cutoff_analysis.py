import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, f1_score, roc_auc_score

# =========================================================
# 1. 基础配置
# =========================================================
BASE_DIR = "../../"
MODEL_ROOT = os.path.join(BASE_DIR, "artifacts/models")
FIG_ROOT = os.path.join(BASE_DIR, "results/figures")
TABLE_ROOT = os.path.join(BASE_DIR, "results/tables")
OUTCOMES = ['pof', 'mortality_28d', 'composite_outcome']

for path in [FIG_ROOT, TABLE_ROOT]:
    if not os.path.exists(path):
        os.makedirs(path)

def calculate_detailed_metrics(y_true, y_prob, threshold):
    """基于特定阈值计算临床诊断指标"""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # 计算点估计
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics = {
        "Threshold": round(threshold, 4),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "Sensitivity": round(sensitivity, 4),
        "Specificity": round(specificity, 4),
        "PPV": round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0,
        "NPV": round(tn / (tn + fn), 4) if (tn + fn) > 0 else 0,
        "F1_Score": round(f1_score(y_true, y_pred), 4),
        "Accuracy": round((tp + tn) / (tp + tn + fp + fn), 4)
        # "Sen_CI": "N/A"  # 如果不跑 Bootstrap，建议先注释掉或设为 N/A
    }
    return metrics

def plot_diagnostic_viz(y_true, y_prob, threshold, name, target, save_dir):
    """生成科研级 ROC 标注图和概率分布图"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # 1. 带截断点的 ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    ax1.plot(fpr, tpr, label=f'ROC (AUC={auc_val:.3f})', color='darkorange', lw=2)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    
    # 计算当前阈值下的 Sen/Spe 用于标注
    perf = calculate_detailed_metrics(y_true, y_prob, threshold)
    ax1.scatter(1-perf['Specificity'], perf['Sensitivity'], color='red', s=100, 
                label=f'Best Cutoff: {threshold:.3f}\n(Sen:{perf["Sensitivity"]:.2f}, Spe:{perf["Specificity"]:.2f})')
    # 在 ax1.scatter 之后添加，增强科研感
    ax1.annotate(f'Opt: {threshold:.3f}\nSen: {perf["Sensitivity"]:.2f}\nSpe: {perf["Specificity"]:.2f}',
                 xy=(1-perf['Specificity'], perf['Sensitivity']), 
                 xytext=(1-perf['Specificity']+0.1, perf['Sensitivity']-0.2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
    ax1.set_xlabel('1 - Specificity')
    ax1.set_ylabel('Sensitivity')
    ax1.set_title(f'{name} ROC Analysis ({target})')
    ax1.legend(loc='lower right')

    # 2. 概率分布图 (展示风险分离度)
    df_prob = pd.DataFrame({'prob': y_prob, 'target': y_true})
    ax2.hist(df_prob[df_prob['target'] == 0]['prob'], bins=40, alpha=0.5, label='Normal', color='blue', density=True)
    ax2.hist(df_prob[df_prob['target'] == 1]['prob'], bins=40, alpha=0.5, label='Outcome(+)', color='red', density=True)
    ax2.axvline(threshold, color='black', linestyle='--', lw=2, label=f'Cutoff: {threshold:.3f}')
    
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Density')
    ax2.set_title('Risk Separation Distribution')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"07_Diagnostic_{name}.png"))
    plt.close()

# =========================================================
# 2. 核心执行逻辑
# =========================================================
def run_cutoff_optimization_flow():
    print("🚀 启动 07 步：自动化阈值寻优与临床效能审计...")
    global_summary = []

    for target in OUTCOMES:
        target_dir = os.path.join(MODEL_ROOT, target)
        fig_save_dir = os.path.join(FIG_ROOT, target)
        
        if not os.path.exists(target_dir):
            print(f"⚠️ 跳过 {target}: 找不到资产目录")
            continue

        print(f"\n--- 正在处理终点: [{target.upper()}] ---")
        
        try:
            # 1. 加载核心资产
            models_dict = joblib.load(os.path.join(target_dir, "all_models_dict.pkl"))
            eval_data = joblib.load(os.path.join(target_dir, "eval_data.pkl"))
            X_test_pre = eval_data['X_test_pre']
            y_test = eval_data['y_test']
            
            # 2. 增强功能 1：特征对齐逻辑 (修复缩进与逻辑)
            feat_path = os.path.join(target_dir, "selected_features.json")
            if os.path.exists(feat_path):
                with open(feat_path, 'r') as f:
                    selected_features = json.load(f)
                
                # 确保 X_test 的列顺序与训练时一致
                if isinstance(X_test_pre, pd.DataFrame):
                    X_test_pre = X_test_pre[selected_features]
                    print(f"  ✅ 特征强制对齐成功 (n={len(selected_features)})")
                else:
                    print(f"  ⚠️ 警告: {target} 的测试集不是 DataFrame 格式，无法自动排序特征。")
            
        except Exception as e:
            print(f"❌ 加载 {target} 资产失败: {e}")
            continue

        target_thresholds = {}
        target_perf_report = []

        # 3. 模型遍历循环
        for name, clf in models_dict.items():
            X_eval = X_test_pre.values if hasattr(X_test_pre, 'values') else X_test_pre
            y_prob = clf.predict_proba(X_eval)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            
            # 功能 2：Youden Index 寻优与异常处理
            if len(thresholds) <= 1:
                print(f"  ⚠️ 模型 {name} 预测无区分度，设置默认阈值 0.5")
                best_th = 0.5
            else:
                youden_index = tpr + (1 - fpr) - 1
                best_idx = np.argmax(youden_index)
                best_th = float(thresholds[best_idx])
            
            # 修正 sklearn 有时会生成阈值 > 1.0 的情况
            if best_th > 1.0: best_th = 1.0

            # 功能 3：全维度效能审计 (包含混淆矩阵计数)
            perf = calculate_detailed_metrics(y_test, y_prob, best_th)
            perf['Algorithm'] = name
            target_perf_report.append(perf)
            target_thresholds[name] = best_th

            # 功能 5：可视化诊断图 (ROC + 概率分布)
            plot_diagnostic_viz(y_test, y_prob, best_th, name, target, fig_save_dir)

            # 记录到全局汇总清单
            global_summary.append({
                "Outcome": target,
                "Algorithm": name,
                "AUC": round(roc_auc_score(y_test, y_prob), 4),
                **perf
            })

        # =========================================================
        # 4. 资产持久化 (位置：结局循环内，模型循环外)
        # =========================================================
        # 功能 4.1: 保存阈值 JSON (供 eICU 外部验证直接调用)
        th_json_path = os.path.join(target_dir, "thresholds.json")
        with open(th_json_path, 'w') as f:
            json.dump(target_thresholds, f, indent=4)
        
        # 功能 4.2: 保存效能报告 (Table 3 内容)
        perf_df = pd.DataFrame(target_perf_report)
        # 存入模型子目录
        perf_df.to_csv(os.path.join(target_dir, "internal_diagnostic_perf.csv"), index=False)
        # 存入全局 Table 汇总目录
        perf_df.to_csv(os.path.join(TABLE_ROOT, f"Table3_Internal_Perf_{target}.csv"), index=False)
        
        print(f"  ✅ 阈值资产已绑定: {th_json_path}")
        best_model_info = max(target_perf_report, key=lambda x: x['F1_Score'])
        print(f"  ✅ 审计完成。最优算法: {best_model_info['Algorithm']} (F1: {best_model_info['F1_Score']})")

    # 5. 保存全结局全局汇总表 (位置：所有循环结束后)
    if global_summary:
        summary_df = pd.DataFrame(global_summary)
        summary_df.to_csv(os.path.join(MODEL_ROOT, "global_diagnostic_summary.csv"), index=False)
        print(f"\n📊 任务圆满完成！全局汇总报告已生成: {MODEL_ROOT}/global_diagnostic_summary.csv")

if __name__ == "__main__":
    run_cutoff_optimization_flow()
