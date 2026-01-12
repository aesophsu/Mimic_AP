import os
import numpy as np
import pandas as pd

# =========================================================
# 1. 配置与路径
# =========================================================
BASE_DIR = ".."
INPUT_PATH = os.path.join(BASE_DIR, "data/cleaned/mimic_for_table1.csv")
SAVE_DIR = os.path.join(BASE_DIR, "data/cleaned")

def run_module_02():
    print("="*60)
    print("🚀 运行优化模块 02: 深度特征过滤与亚组划分 (保持原始尺度)")
    print("="*60)
    
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 错误: 找不到输入文件 {INPUT_PATH}")
        return
    
    # 加载模块 01 处理后的干净原始数据
    df = pd.read_csv(INPUT_PATH)

    # =========================================================
    # 2. 泄露防护：剔除治疗干扰、评分变量及冗余 ID
    # =========================================================
    # 逻辑：排除临床评分、入选后才产生的治疗手段及结局信息，防止 AUC 虚高
    must_drop = [
        # 1. 评分系统 (包含结局信息，是 Data Leakage 的重灾区)
        'sofa_score', 'sapsii', 'apsiii', 'oasis', 'lods',
        # 2. 时间与 ID (非生物学预测因子)
        'admittime', 'dischtime', 'intime', 'subject_id', 'hadm_id', 'stay_id',
        # 3. 器官支持措施 (属于“治疗”而非“基线状态”)
        'mechanical_vent_flag', 'vaso_flag',
        # 4. 其他结局指标 (防止模型直接学习到结果)
        'los', 'deathtime', 'dod', 'hosp_mortality', 'overall_mortality'
    ]
    
    # 仅剔除存在的列
    cols_to_drop = [c for c in must_drop if c in df.columns]
    df_clean = df.drop(columns=cols_to_drop)
    
    print(f"🛡️ 泄露防护：已剔除 {len(cols_to_drop)} 个潜在泄露/非预测特征")
    print(f"📊 当前特征维数: {df_clean.shape[1]}")

    # =========================================================
    # 3. 亚组划分 (Subgroup Definition) - 使用原始量级
    # =========================================================
    # 临床定义：入院 24h 内肌酐 < 1.5 mg/dL 且 无慢性肾病史 (CKD)
    # 此时 creatinine_max 是模块 01 修正后的原始值 (如 1.2)，直接对比 1.5
    
    if 'creatinine_max' in df_clean.columns and 'chronic_kidney_disease' in df_clean.columns:
        # 定义无预存肾损伤亚组
        df_clean['subgroup_no_renal'] = (
            (df_clean['creatinine_max'] < 1.5) & 
            (df_clean['chronic_kidney_disease'] == 0)
        ).astype(int)
        
        no_renal_count = df_clean['subgroup_no_renal'].sum()
        print(f"✅ 亚组标记完成: '无预存肾损伤' 样本量 = {no_renal_count} (占 {no_renal_count/len(df_clean):.1%})")
    else:
        print("⚠️ 警告: 缺少 creatinine_max 或 chronic_kidney_disease，跳过亚组划分。")

    # =========================================================
    # 4. ⚠️ 特别说明：Log1p 转换说明
    # =========================================================
    # 注意：我们在此处不进行 Log1p 转换。
    # 理由：为了保证 Table 1 的数据描述是原始临床数值，
    # 且为了让模块 03 的训练管道能从原始值开始学习标准化逻辑。
    print("💡 状态：特征保持原始物理量级 (Raw Scale)，Log 转换将移至模块 03 执行。")

    # =========================================================
    # 5. 最终自检与保存
    # =========================================================
    model_ready_path = os.path.join(SAVE_DIR, "mimic_for_model.csv")
    df_clean.to_csv(model_ready_path, index=False)
    
    print("-" * 60)
    print(f"📊 数据就绪统计:")
    print(f"   - 样本总数: {df_clean.shape[0]}")
    print(f"   - 最终进入模型候选的特征数: {df_clean.shape[1]}")
    print(f"   - 结局 (POF) 发生率: {df_clean['pof'].mean():.2%}")
    print("-" * 60)
    print(f"✅ 模块 02 优化完成! 数据存至: {model_ready_path}")

if __name__ == "__main__":
    run_module_02()
