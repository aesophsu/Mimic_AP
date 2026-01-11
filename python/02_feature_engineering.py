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
    print("🚀 运行优化模块 02: 深度特征过滤、转换与亚组划分")
    print("="*60)
    
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 错误: 找不到输入文件 {INPUT_PATH}")
        return
    
    df = pd.read_csv(INPUT_PATH)

    # =========================================================
    # 2. 必须剔除的特征 (泄漏防护与临床无关项)
    # =========================================================
    # 逻辑：排除总分、时间戳、器官支持手段（防止 AUC 虚高）
    must_drop = [
        # 评分系统 (包含结局信息)
        'sofa_score', 'sapsii', 'apsiii', 'oasis', 'lods',
        # 时间与 ID (非预测因子)
        'admittime', 'dischtime', 'intime', 'subject_id', 'hadm_id', 'stay_id',
        # 器官支持 (属于“治疗”而非“基线”)
        'mechanical_vent_flag', 'vaso_flag',
        # 其他不相关的时效特征
        'los' 
    ]
    
    # 临时保留结局变量 (Label)，但从特征集中移除
    df_clean = df.drop(columns=[c for c in must_drop if c in df.columns])
    print(f"🛡️ 泄露防护：已剔除 {len(must_drop)} 个潜在泄露/干扰特征")

    # =========================================================
    # 3. Log1p 转换 (处理偏态分布)
    # =========================================================
    # 根据模块 1 报告，针对波动大、分布偏斜的生化指标进行转换
    skewed_features = [
        'creatinine_max', 'creatinine_min', 'bun_max', 'bun_min',
        'wbc_max', 'wbc_min', 'glucose_max', 'glucose_min',
        'lab_amylase_max', 'lipase_max', 'lactate_max',
        'alt_max', 'ast_max', 'bilirubin_total_max', 
        'alp_max', 'inr_max', 'rdw_max'
    ]
    
    existing_skewed = [c for c in skewed_features if c in df_clean.columns]
    for col in existing_skewed:
        # np.log1p(x) = ln(1+x)
        df_clean[col] = np.log1p(df_clean[col].astype(float).clip(lower=0))
    
    print(f"✅ 转换完成：对 {len(existing_skewed)} 个实验室指标执行了 Log1p 处理")

    # =========================================================
    # 4. 亚组划分 (Subgroup Definition)
    # =========================================================
    # 临床定义：入院 24h 内肌酐 < 1.5 mg/dL 且 无慢性肾病史
    # 注意：此时 df_clean 里的 creatinine 已经是 log 后的
    log_cre_threshold = np.log1p(1.5) 
    
    if 'creatinine_max' in df_clean.columns:
        df_clean['subgroup_no_renal'] = (
            (df_clean['creatinine_max'] < log_cre_threshold) & 
            (df_clean['chronic_kidney_disease'] == 0)
        ).astype(int)
    
    # =========================================================
    # 5. 最终自检与保存
    # =========================================================
    model_ready_path = os.path.join(SAVE_DIR, "mimic_for_model.csv")
    df_clean.to_csv(model_ready_path, index=False)
    
    print("-" * 60)
    print(f"📊 数据就绪统计:")
    print(f"   - 最终特征总数: {df_clean.shape[1]}")
    print(f"   - 无肾损伤亚组 (No-Renal): {df_clean['subgroup_no_renal'].sum()} 例")
    print(f"   - 24h后 POF 发生率: {df_clean['pof'].mean():.1%}")
    print("-" * 60)
    print(f"✅ 模块 02 优化完成! 模型就绪数据存至: {model_ready_path}")

if __name__ == "__main__":
    run_module_02()
