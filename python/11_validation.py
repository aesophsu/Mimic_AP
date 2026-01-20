import os
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# =========================================================
# 1. 配置与路径
# =========================================================
BASE_DIR = ".."
DATA_DIR = os.path.join(BASE_DIR, "data/cleaned")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAVE_DIR = os.path.join(BASE_DIR, "results/calibration")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR, exist_ok=True)

def run_module_12_enhanced_audit(target='pof'):
    print("="*75)
    print(f"🔬 模块 12: 临床校准审计与比值比 (OR) 分析 | 结局: {target.upper()}")
    print("="*75)

    # 1. 加载 eICU 外部验证数据与模型字典
    eicu_path = os.path.join(DATA_DIR, f"eicu_for_model_{target}.csv")
    model_dict_path = os.path.join(MODELS_DIR, f"all_models_{target}.pkl")
    
    if not (os.path.exists(eicu_path) and os.path.exists(model_dict_path)):
        print(f"❌ 错误：缺少 {target} 的验证数据或模型包。")
        return

    df_eicu = pd.read_csv(eicu_path)
    X_eicu = df_eicu.drop('target', axis=1)
    y_eicu = df_eicu['target']
    models_dict = joblib.load(model_dict_path)

    # ---------------------------------------------------------
    # A. 概率校准审计 (Calibration Curve)
    # ---------------------------------------------------------
    print("\n🚩 [Step 1] 正在执行多模型校准审计 (Probability Calibration):")
    plt.figure(figsize=(9, 8), dpi=150)
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)", alpha=0.5)
    
    calibration_metrics = []

    for name, model in models_dict.items():
        # 获取外部验证集预测概率
        # 兼容性处理：如果是 Pipeline 则使用 .values
        X_input = X_eicu.values if hasattr(model, 'named_steps') else X_eicu
        y_prob = model.predict_proba(X_input)[:, 1]
        
        # 计算校准曲线与 Brier 分数
        prob_true, prob_pred = calibration_curve(y_eicu, y_prob, n_bins=10)
        brier = brier_score_loss(y_eicu, y_prob)
        
        # 绘图
        plt.plot(prob_pred, prob_true, "s-", markersize=4, label=f"{name} (Brier: {brier:.4f})")
        calibration_metrics.append((name, brier))
        print(f"  - {name:<20} | Brier Score = {brier:.4f}")

    plt.title(f"External Calibration Curve: {target.upper()}", fontsize=14)
    plt.xlabel("Predicted Risk (Expected Probability)")
    plt.ylabel("Observed Outcome (Actual Probability)")
    plt.legend(loc="lower right", frameon=True)
    plt.grid(alpha=0.3)
    
    cal_img_path = os.path.join(SAVE_DIR, f"calibration_audit_{target}.png")
    plt.savefig(cal_img_path, bbox_inches='tight')
    print(f"\n📊 校准审计图已保存至: {cal_img_path}")

    # ---------------------------------------------------------
    # B. 比值比分析 (Odds Ratio for Nomogram)
    # ---------------------------------------------------------
    if "Logistic Regression" in models_dict:
        print(f"\n🚩 [Step 2] 提取 {target.upper()} 临床风险权重 (Odds Ratios):")
        lr_wrapper = models_dict["Logistic Regression"]
        
        # --- 修复代码开始 ---
        # 1. 处理 CalibratedClassifierCV 包装
        if hasattr(lr_wrapper, 'calibrated_classifiers_'):
            # 提取第一个交叉验证折叠中的基模型
            raw_model = lr_wrapper.calibrated_classifiers_[0].estimator
        else:
            raw_model = lr_wrapper

        # 2. 处理 Pipeline 包装
        if hasattr(raw_model, 'named_steps'):
            final_lr = raw_model.named_steps['model']
        else:
            final_lr = raw_model

        # 3. 提取系数 (确保它有 coef_ 属性)
        if hasattr(final_lr, 'coef_'):
            coefs = final_lr.coef_[0]
            # --- 修复代码结束 ---
            
            or_values = np.exp(coefs)
            
            or_df = pd.DataFrame({
                'Feature': X_eicu.columns,
                'Beta_Coef': coefs,
                'Odds_Ratio': or_values
            }).sort_values(by='Odds_Ratio', ascending=False)

            # 保存并打印结果
            or_path = os.path.join(SAVE_DIR, f"odds_ratio_{target}.csv")
            or_df.to_csv(or_path, index=False)
            
            for _, row in or_df.iterrows():
                impact = "🚩 危险因素" if row['Odds_Ratio'] > 1 else "✅ 保护因素"
                print(f"  - {row['Feature']:<20} | OR = {row['Odds_Ratio']:>6.2f} | {impact}")
        else:
            print("  ⚠️ 无法提取系数：模型不包含 coef_ 属性。")

    # ---------------------------------------------------------
    # C. 临床解释建议
    # ---------------------------------------------------------
    print("\n📝 [Step 3] 临床解释笔记 (Audit Notes):")
    best_brier = min(calibration_metrics, key=lambda x: x[1])
    print(f"  💡 预测可靠性：{best_brier[0]} 具有最低的 Brier 分数，代表其概率估计最精准。")
    print("  💡 诺莫图转化：Logistic Regression 的 OR 值反映了单单位特征变化对发病胜算的贡献。")
    print("  💡 风险校准：若曲线在理想线上方，代表模型在外部人群中倾向于低估风险（Under-prediction）。")

    plt.show()

if __name__ == "__main__":
    # 针对所有结局执行审计
    for t in ['pof', 'composite_outcome', 'mortality_28d']:
        run_module_12_enhanced_audit(t)
