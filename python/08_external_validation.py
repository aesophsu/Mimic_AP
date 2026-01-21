import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss, precision_recall_curve, auc

# =========================================================
# 1. 配置与路径
# =========================================================
BASE_DIR = ".."
DATA_DIR = os.path.join(BASE_DIR, "data/cleaned")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAVE_DIR = os.path.join(BASE_DIR, "results/validation")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def run_module_08_multi_model_eval():
    print("="*85)
    print("🏆 模块 08: 多模型外部验证 (MIMIC -> eICU)")
    print("="*85)

    targets = ['pof', 'composite_outcome', 'mortality_28d']
    
    # 设置绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')
    # 每一行代表一个结局，绘制其所有模型的 ROC
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), dpi=150)

    # 用于汇总所有指标的列表
    performance_metrics = []

    for i, target in enumerate(targets):
        print(f"\n🚀 正在验证结局: {target.upper()}")
        
        # 1. 加载 eICU 数据
        eicu_path = os.path.join(DATA_DIR, f"eicu_for_model_{target}.csv")
        if not os.path.exists(eicu_path):
            print(f"  ⚠️ 跳过: 找不到数据 {eicu_path}")
            continue
            
        df_eicu = pd.read_csv(eicu_path)
        X_eicu = df_eicu.drop('target', axis=1)
        y_eicu = df_eicu['target']

        # 2. 加载模型字典
        model_dict_path = os.path.join(MODELS_DIR, f"all_models_{target}.pkl")
        if not os.path.exists(model_dict_path):
            print(f"  ⚠️ 跳过: 找不到模型包 {model_dict_path}")
            continue
            
        models_dict = joblib.load(model_dict_path)
        
        # 3. 遍历字典中的每一个模型进行评估
        for algo_name, model in models_dict.items():
            try:
                # 预测概率
                y_prob = model.predict_proba(X_eicu.values)[:, 1]
                
                # 计算指标
                auc_score = roc_auc_score(y_eicu, y_prob)
                brier = brier_score_loss(y_eicu, y_prob)
                
                # 计算 PR-AUC
                prec, rec, _ = precision_recall_curve(y_eicu, y_prob)
                pr_auc = auc(rec, prec)
                
                # 保存结果
                performance_metrics.append({
                    'Target': target,
                    'Algorithm': algo_name,
                    'ROC-AUC': auc_score,
                    'PR-AUC': pr_auc,
                    'Brier': brier
                })
                
                # 4. 绘制 ROC 曲线
                fpr, tpr, _ = roc_curve(y_eicu, y_prob)
                axes[i].plot(fpr, tpr, lw=1.5, label=f'{algo_name} (AUC={auc_score:.3f})')
                
                print(f"  ✅ {algo_name:<20} | AUC: {auc_score:.4f} | Brier: {brier:.4f}")
                
            except Exception as e:
                print(f"  ❌ 评估 {algo_name} 时出错: {e}")

        # 设置子图格式
        axes[i].plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5)
        axes[i].set_title(f'Outcome: {target.upper()}', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('False Positive Rate')
        axes[i].set_ylabel('True Positive Rate')
        axes[i].legend(loc="lower right", fontsize=9, frameon=True)

    # 5. 保存指标汇总表
    summary_df = pd.DataFrame(performance_metrics)
    summary_path = os.path.join(SAVE_DIR, "external_validation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "multi_model_external_roc.png"))
    print(f"\n📊 所有模型验证完成！")
    print(f"📂 指标汇总已保存至: {summary_path}")
    print(f"🖼️ ROC 曲线图已保存至: {os.path.join(SAVE_DIR, 'multi_model_external_roc.png')}")
    plt.show()

if __name__ == "__main__":
    run_module_08_multi_model_eval()
