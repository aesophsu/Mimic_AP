import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================================================
# 1. 配置与路径
# =========================================================
BASE_DIR = ".."
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAVE_DIR = os.path.join(BASE_DIR, "results/nomograms")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR, exist_ok=True)

# 临床标签映射（保持不变）
LABEL_MAP = {
    'creatinine_max': 'Max Creatinine',
    'bun_min': 'Min BUN',
    'admission_age': 'Age',
    'wbc_max': 'Max WBC',
    'ph_min': 'Min pH',
    'ph_max': 'Max pH',
    'spo2_max': 'Max SpO2',
    'spo2_min': 'Min SpO2',
    'lactate_max': 'Max Lactate',
    'pao2fio2ratio_min': 'Min PaO2/FiO2',
    'albumin_max': 'Max Albumin',
    'albumin_min': 'Min Albumin',
    'ast_max': 'Max AST',
    'alt_max': 'Max ALT',
    'glucose_lab_max': 'Max Glucose',
    'malignant_tumor': 'Malignant Tumor',
    'bilirubin_total_min': 'Min Total Bilirubin',
    'lab_calcium_min': 'Min Calcium',
    'alp_max': 'Max ALP'
}

def generate_all_nomograms_refined():
    targets = ['pof', 'composite_outcome', 'mortality_28d']
    
    print("="*80)
    print("🎨 模块 14: 临床诺莫图权重分析 (结局对齐版)")
    print("="*80)

    for target in targets:
        print(f"\n🚀 正在构建结局权重系统: {target.upper()}")
        
        # --- 核心修复：动态加载对应结局的特征列表 ---
        features_path = os.path.join(MODELS_DIR, f"selected_features_{target}.pkl")
        model_path = os.path.join(MODELS_DIR, f"all_models_{target}.pkl")
        
        if not os.path.exists(features_path) or not os.path.exists(model_path):
            print(f"  ⚠️ 跳过: 找不到 {target} 的特征列表或模型文件")
            continue
            
        # 加载数据
        selected_features = joblib.load(features_path)
        models_dict = joblib.load(model_path)
        lr_model = models_dict.get("Logistic Regression")
        
        # 提取系数 (多层拆箱)
        curr_step = lr_model
        if hasattr(curr_step, 'calibrated_classifiers_'):
            curr_step = curr_step.calibrated_classifiers_[0].estimator
        if hasattr(curr_step, 'named_steps'):
            final_lr = curr_step.named_steps['model']
        else:
            final_lr = curr_step

        coefs = final_lr.coef_[0]
        
        # 确保系数与特征名对齐
        if len(coefs) != len(selected_features):
            print(f"  ❌ 严重错误: {target} 系数数量({len(coefs)}) 与特征名数量({len(selected_features)}) 不匹配！")
            continue

        # 应用标签转换
        features_to_use = [LABEL_MAP.get(f, f.replace('_', ' ').title()) for f in selected_features]

        # 计算 Nomogram 分值 (以最大影响因子为100分)
        max_impact = np.max(np.abs(coefs))
        scaling_factor = 100 / max_impact
        points = coefs * scaling_factor
        
        nomo_df = pd.DataFrame({
            'Feature': features_to_use,
            'Nomogram_Points': points
        }).sort_values(by='Nomogram_Points', key=abs, ascending=True)

        # 绘图
        plt.figure(figsize=(12, 8), dpi=150)
        # 经典的临床红蓝配色
        colors = ['#E64B35' if x > 0 else '#4DBBD5' for x in nomo_df['Nomogram_Points']]
        
        bars = plt.barh(nomo_df['Feature'], nomo_df['Nomogram_Points'], color=colors, alpha=0.8)
        plt.axvline(0, color='black', linewidth=1.2)
        
        # 数值标注
        for bar in bars:
            width = bar.get_width()
            plt.text(width + (2 if width > 0 else -2), bar.get_y() + bar.get_height()/2, 
                     f'{width:.1f}', va='center', ha='left' if width > 0 else 'right',
                     fontsize=10, fontweight='bold')

        plt.title(f"Clinical Nomogram Weights: {target.upper()}", fontsize=15, pad=20)
        plt.xlabel("Points contribution (Nomogram Scale)", fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.3)
        
        # 动态调整横轴范围，确保标签不被遮挡
        limit = max(abs(points)) + 20
        plt.xlim(-limit, limit)
        
        img_path = os.path.join(SAVE_DIR, f"nomogram_refined_{target}.png")
        plt.savefig(img_path, bbox_inches='tight')
        plt.close()
        
        # 保存得分 CSV
        nomo_df.to_csv(os.path.join(SAVE_DIR, f"nomogram_points_{target}.csv"), index=False)
        print(f"  ✅ 优化后的图像与得分表已生成。")

    print("\n" + "="*80)
    print("✨ 任务完成！请在 results/nomograms 查看最新结果。")

if __name__ == "__main__":
    generate_all_nomograms_refined()
