import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================================================
# 1. 配置与路径
# =========================================================
BASE_DIR = ".."
DATA_DIR = os.path.join(BASE_DIR, "data/cleaned")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAVE_DIR = os.path.join(BASE_DIR, "results/dca")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR, exist_ok=True)

# =========================================================
# 2. 核心函数：净获益计算引擎
# =========================================================
def calculate_net_benefit(y_true, y_prob, thresholds):
    net_benefit = []
    n = len(y_true)
    for pt in thresholds:
        if pt <= 0 or pt >= 1:
            net_benefit.append(0)
            continue
        y_pred = (y_prob >= pt).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        net_benefit.append(nb)
    return net_benefit

# =========================================================
# 3. 运行增强版 DCA 评估
# =========================================================
def run_module_13_enhanced_dca():
    print("="*85)
    print("📈 模块 13: 临床决策曲线分析 (DCA) - 临床实用性审计版")
    print("="*85)

    targets = ['pof', 'composite_outcome', 'mortality_28d']
    thresholds = np.linspace(0.01, 0.99, 100)
    # 定义审计阈值：临床上最受关注的决策点
    audit_pts = [0.1, 0.2, 0.5] 

    for target in targets:
        print(f"\n🚀 正在分析结局: {target.upper()}")
        print("-" * 45)
        
        eicu_path = os.path.join(DATA_DIR, f"eicu_for_model_{target}.csv")
        model_dict_path = os.path.join(MODELS_DIR, f"all_models_{target}.pkl")
        
        if not (os.path.exists(eicu_path) and os.path.exists(model_dict_path)):
            print(f"⚠️ 跳过: 缺少数据或模型。")
            continue

        df_eicu = pd.read_csv(eicu_path)
        y_eicu = df_eicu['target'].values
        X_eicu_values = df_eicu.drop('target', axis=1).values
        models_dict = joblib.load(model_dict_path)
        
        prevalence = np.mean(y_eicu)
        print(f"📊 外部数据流行率 (Prevalence): {prevalence:.2%}")

        # 绘图初始化
        plt.figure(figsize=(10, 8), dpi=150)
        
        # 策略 A: Treat All
        nb_all = [prevalence - (1 - prevalence) * (pt / (1 - pt)) for pt in thresholds]
        plt.plot(thresholds, nb_all, color='gray', linestyle='--', label='Treat All', alpha=0.5)
        
        # 策略 B: Treat None
        plt.axhline(y=0, color='black', linestyle='-', label='Treat None', alpha=0.5)

        # 审计汇总
        audit_results = []

        # 策略 C: 多模型评估
        for name, model in models_dict.items():
            try:
                y_prob = model.predict_proba(X_eicu_values)[:, 1]
                nb_curve = calculate_net_benefit(y_eicu, y_prob, thresholds)
                plt.plot(thresholds, nb_curve, lw=2, label=f'{name}')
                
                # 提取特定点的审计信息
                nb_at_pts = []
                for pt in audit_pts:
                    val = calculate_net_benefit(y_eicu, y_prob, [pt])[0]
                    nb_at_pts.append(val)
                
                audit_results.append([name] + nb_at_pts)
                print(f"  ✅ 已计算: {name:<20} | NB@20%: {nb_at_pts[1]:.4f}")
                
            except Exception as e:
                print(f"  ❌ 无法预测 {name}: {e}")

        # 输出审计表格 (用于 Discussion 写作)
        print("\n🚩 关键阈值净获益汇总 (Net Benefit Table):")
        audit_df = pd.DataFrame(audit_results, columns=['Algorithm'] + [f'Pt={p:.0%}' for p in audit_pts])
        # 计算 Treat All 的基准线
        nb_all_audit = [prevalence - (1 - prevalence) * (pt / (1 - pt)) for pt in audit_pts]
        audit_df.loc[len(audit_df)] = ['[Base] Treat All'] + nb_all_audit
        print(audit_df.to_string(index=False))

        # 图表细节
        plt.xlim(0, 0.7)
        plt.ylim(-0.05, prevalence + 0.1)
        plt.xlabel('Risk Threshold Probability', fontsize=12)
        plt.ylabel('Net Benefit', fontsize=12)
        plt.title(f'DCA: {target.upper()} (eICU Validation)', fontsize=14)
        plt.legend(loc='upper right', frameon=True)
        plt.grid(alpha=0.3)
        
        save_path = os.path.join(SAVE_DIR, f"dca_final_{target}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    print("\n" + "="*85)
    print("✅ 模块 13 增强分析完成！")

if __name__ == "__main__":
    run_module_13_enhanced_dca()
