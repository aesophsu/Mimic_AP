import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, brier_score_loss
import shap
import warnings

# 忽略模型预测时的版本兼容性警告
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
    print("🚀 启动模块 04：多模型性能对比与临床可解释性分析")
    print("="*70)

    # 1. 环境准备：加载模型、特征列表及外部验证集
    print("📂 [Step 1/4] 正在检索特定终点的序列化模型与数据资产...")
    endpoints = ['pof', 'death_28d', 'combined'] 
    for target in endpoints:
        print(f"\n" + "="*70)
        print(f"🚀 正在处理研究终点: {target.upper()}")
        print("="*70)

        # 动态加载对应 target 的模型和数据
        try:
            all_models = joblib.load(os.path.join(MODEL_DIR, f"all_models_{target}.pkl"))
            selected_features = joblib.load(os.path.join(MODEL_DIR, f"selected_features_{target}.pkl"))
            # 注意：此处文件名需与你模块03保存的 test_data_main_{target}.pkl 一致
            X_test, y_test = joblib.load(os.path.join(MODEL_DIR, f"test_data_main_{target}.pkl"))
            X_sub, y_sub = joblib.load(os.path.join(MODEL_DIR, f"test_data_sub_{target}.pkl"))
            X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
            X_sub_np = X_sub.values if hasattr(X_sub, 'values') else X_sub
            # 读取模块 03 预计算的 95% 置信区间 (CI) 统计数据
            ci_path = os.path.join(MODEL_DIR, f"ci_main_{target}.pkl")
            sub_ci_path = os.path.join(MODEL_DIR, f"ci_sub_{target}.pkl")
            if os.path.exists(ci_path):
                ci_data = joblib.load(ci_path)
                sub_ci_data = joblib.load(sub_ci_path)
            else:
                # 若无 CI 缓存文件，则仅展示单次点估计结果
                ci_data = {} 
                sub_ci_data = {}
            print(f"   ✅ 加载成功: 包含 {len(all_models)} 个模型")
            print(f"   ✅ 特征列表: {selected_features}")
            print(f"   ✅ 测试集维度: {X_test_np.shape}, POF 流行率: {np.mean(y_test):.2%}")
        except Exception as e:
            print(f"   ❌ {target} 加载失败: {e}")
            continue # 跳过当前结局，继续下一个

        # --------------------------------------------------------
        # [图 1] 全模型 ROC 对比
        # --------------------------------------------------------
        print("\n📈 [Step 2/4] 区分度评价：生成受试者工作特征曲线 (ROC)")
        plt.figure(figsize=(9, 8))
        # --------------------------------------------------------
        # [Step 2] 同步模块 03 的审计数据
        # --------------------------------------------------------
        for name, clf in all_models.items():
            # 强制使用 numpy 数组预测，消除警告
            y_prob = clf.predict_proba(X_test_np)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(*roc_curve(y_test, y_prob)[:2])
    
            # 敏感性分析：计算非肾源性 (No-Renal) 亚组的区分度
            y_prob_sub = clf.predict_proba(X_sub_np)[:, 1]
            roc_auc_sub = auc(*roc_curve(y_sub, y_prob_sub)[:2])
    
            # 性能汇总：对比全样本与亚组的 AUC 表现
            print(f"   🔍 模型审计: {name:<20} | Test AUC: {roc_auc:.4f} | Sub-AUC: {roc_auc_sub:.4f}")
    
            display_label = f"{name}: {ci_data.get(name, f'{roc_auc:.3f}')}"
            plt.plot(fpr, tpr, lw=2, label=display_label)

        plt.plot([0, 1], [0, 1], 'k--', alpha=0.2)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'Predictive Performance Comparison: {target.upper()}', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=9)
        plt.grid(alpha=0.2)
        plt.savefig(os.path.join(FIG_DIR, f"01_ROC_{target}.png"), dpi=300)
        plt.close()

        # --------------------------------------------------------
        # [图 2] SHAP 解释 (针对 SVM - 基于当前 target 动态区分)
        # --------------------------------------------------------
        print(f"\n🧪 [Step 3/4] 临床解释性：基于 SHAP 值的特征贡献度分析")
        
        # 策略：为不同终点维护独立的解释模型缓存以提高效率
        SHAP_CACHE_PATH = os.path.join(MODEL_DIR, f"svm_shap_values_{target}.pkl")

        try:
            # 1. 尝试加载现有的缓存
            if os.path.exists(SHAP_CACHE_PATH):
                # 修正 2: 修复缩进
                print(f"   ♻️ 检测到缓存，正在加载 {target} 预计算的 SHAP 值...")
                shap_values = joblib.load(SHAP_CACHE_PATH)
            else:
                print(f"   ⏳ 未检测到缓存，启动 {target} 的 SVM SHAP 计算...")
                print("   📢 提示：全量样本蒙特卡洛计算较为耗时，正在生成高精度解释图...")
                # 动态特征对齐：提取当前模型最关键的临床预测因子
                current_features = X_test.columns.tolist() 
                print(f"   📊 当前模型特征数: {len(current_features)}")
                svm_model = all_models['SVM']
                
                # 定义预测概率函数
                def svm_predict(data):
                    return svm_model.predict_proba(data)[:, 1]

                # 使用当前 target 对应的测试集背景
                masker = shap.maskers.Independent(X_test_np) 
                explainer = shap.Explainer(svm_predict, masker)
                
                # 执行计算
                shap_values = explainer(X_test_np, silent=True)
                
                # 保存结果
                joblib.dump(shap_values, SHAP_CACHE_PATH)
                print(f"   💾 {target} 的 SHAP 计算完成并已保存。")

            # 2. 绘图
            plt.figure(figsize=(12, 10))
            shap.plots.beeswarm(shap_values, max_display=12, show=False)
            
            # 修正 3: 标题区分 target
            plt.title(f'SVM SHAP Summary: Impact on {target.upper()} (Full Audit)', fontsize=14, fontweight='bold')
            plt.xlabel(f"SHAP Value (Impact on {target.upper()} Probability)")
        
            plt.tight_layout()
            
            # 修正 4: 保存文件名区分 target
            save_path = os.path.join(FIG_DIR, f"02_SHAP_Summary_SVM_{target}.png")
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"   ✅ {target} 的 SHAP 摘要图已生成: {os.path.basename(save_path)}")

        except Exception as e:
            print(f"   ⚠️ {target} 的 SHAP 模块运行失败: {e}")

        # --------------------------------------------------------
        # Step 4: 全模型 DCA 临床价值审计 (针对当前 target 动态区分)
        # --------------------------------------------------------
        print(f"\n⚖️ [Step 4/4] 临床应用价值：决策曲线分析 (DCA) 与净获益评价")
        plt.figure(figsize=(10, 8))
        
        # 阈值优化：根据各终点实际流行率调整风险截断点范围
        # 通常 DCA 观察范围在 0 到 患病率的 2-3 倍之间最有意义
        thresholds = np.arange(0.01, 0.81, 0.01)
    
        # 基础参照线: Treat All (所有人都视为高危)
        prev = np.mean(y_test)
        nb_all = [prev - (1 - prev) * (t / (1 - t)) for t in thresholds]
    
        model_windows = {}
        colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']

        for (name, clf), color in zip(all_models.items(), colors):
            # 确保使用 numpy 数组预测
            y_prob = clf.predict_proba(X_test_np)[:, 1]
            nb_model = [calculate_net_benefit(y_test, y_prob, t) for t in thresholds]
        
            # 获益区间审计：确定模型优于“全干预”或“不干预”策略的临床范围
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
    
        # 视觉优化：自适应调整纵轴以完整呈现各模型的净获益曲线
        plt.ylim(-0.05, max(prev + 0.1, 0.2)) 
        plt.xlim(0, 0.8)
        plt.xlabel('Risk Threshold Probability (Cut-off)')
        plt.ylabel('Net Benefit')
        
        # 修正 3: 标题动态包含 target 名称
        plt.title(f'Decision Curve Analysis: {target.upper()} Comparative Utility', fontsize=14, fontweight='bold')
        
        plt.legend(loc='upper right', fontsize=9)
        plt.grid(alpha=0.2)
        
        # --- 修正后的 Step 4 DCA 保存示例 ---
        save_path = os.path.join(FIG_DIR, f"03_DCA_{target}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ DCA 临床价值图已生成: {os.path.basename(save_path)}")
        # --------------------------------------------------------
        # 结果汇总：生成学术论文标准表 (Table 2 - 模型性能对比汇总)
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
