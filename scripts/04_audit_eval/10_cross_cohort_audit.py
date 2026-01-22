import os
import pandas as pd
import numpy as np
from tableone import TableOne

# =========================================================
# 1. 配置与路径
# =========================================================
BASE_DIR = "../../"
MIMIC_RAW_PATH = os.path.join(BASE_DIR, "data/cleaned/mimic_raw_scale.csv") 
EICU_ALIGNED_PATH = os.path.join(BASE_DIR, "data/external/eicu_aligned.csv") 
SAVE_DIR = os.path.join(BASE_DIR, "results/tables")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR, exist_ok=True)

def run_step_10_cross_audit():
    print("🚀 开始跨库基线审计 (MIMIC-IV vs eICU)...")

    # 1. 加载数据
    if not (os.path.exists(MIMIC_RAW_PATH) and os.path.exists(EICU_ALIGNED_PATH)):
        print("❌ 错误：找不到原始物理尺度数据，请确认 02 步和 09 步已成功运行。")
        return

    df_mimic = pd.read_csv(MIMIC_RAW_PATH)
    df_eicu = pd.read_csv(EICU_ALIGNED_PATH)

    # 2. 选取审计特征
    audit_features = [
        'admission_age', 'gender', 'bmi', 
        'creatinine_max', 'bun_max', 'wbc_max', 
        'glucose_lab_max', 'hematocrit_max', 'respiratory_rate_max',
        'pof', 'mortality_28d'
    ]
    
    common_cols = [c for c in audit_features if c in df_mimic.columns and c in df_eicu.columns]
    
    # 3. 提取子集并标记队列
    df_mimic_sub = df_mimic[common_cols].copy()
    df_mimic_sub['cohort'] = 'MIMIC-IV (Dev)'
    
    df_eicu_sub = df_eicu[common_cols].copy()
    df_eicu_sub['cohort'] = 'eICU (External)'

    # ---------------------------------------------------------
    # 【修复核心】统一分类变量类型，防止 TableOne 排序报错
    # ---------------------------------------------------------
    categorical = ['gender', 'pof', 'mortality_28d']
    existing_cat = [c for c in categorical if c in common_cols]

    print("🛠️ 正在对齐分类变量编码...")
    for col in existing_cat:
        # 1. 将所有值转为字符串，避免 int 与 str 混合
        # 2. 处理可能存在的编码差异（统一为 0/1）
        for df_temp in [df_mimic_sub, df_eicu_sub]:
            # 统一性别映射示例（如果 eICU 是 'M'/'F' 而 MIMIC 是 1/0，这里强制统一）
            if col == 'gender':
                df_temp[col] = df_temp[col].map({'M': '1', 'F': '0', 1: '1', 0: '0', '1': '1', '0': '0'})
            
            # 强制转为 String 并处理缺失值
            df_temp[col] = df_temp[col].astype(str).replace({'nan': np.nan, 'None': np.nan, 'unknown': np.nan})
        
        print(f"  ✅ {col} 类型对齐完成")

    # 合并数据
    df_total = pd.concat([df_mimic_sub, df_eicu_sub], axis=0, ignore_index=True)

    # 4. 执行 TableOne 统计
    print("📊 正在计算统计指标与 SMD (Standardized Mean Difference)...")
    
    nonnormal = [c for c in common_cols if c not in categorical]

    try:
        mytable = TableOne(
            df_total, 
            columns=common_cols, 
            categorical=existing_cat, 
            nonnormal=nonnormal,
            groupby='cohort', 
            pval=True, 
            smd=True,
            overall=False # 重点对比两库差异，无需 Overall
        )

        # 5. 保存资产
        table_path = os.path.join(SAVE_DIR, "Table1_MIMIC_vs_eICU_SMD.csv")
        mytable.to_csv(table_path)
        
        print("\n" + "="*60)
        print(f"✨ 跨库基线表已生成：{table_path}")
        print("="*60)
        
        # 6. 人群漂移分析 (SMD > 0.1 表示存在临床分布不一致)
        # 注意：不同版本 tableone 获取 SMD 的方式略有不同
        print("\n🚨 人群漂移预警 (Population Drift Analysis):")
        # 尝试从 mytable.tableone 获取
        try:
            # 访问 MultiIndex 中的 SMD 列
            smd_data = mytable.tableone['SMD']
            for feat in smd_data.index:
                val = smd_data.loc[feat]
                # 有些特征可能有多个 level，取最大值
                val_max = val.max() if isinstance(val, pd.Series) else val
                
                if pd.isna(val_max): continue
                
                if val_max > 0.1:
                    status = "🔴 显著差异" if val_max > 0.2 else "🟡 轻微偏移"
                    print(f"  - {feat:<20}: SMD = {val_max:.3f} | {status}")
        except Exception as e:
            print(f"  ⚠️ 自动审计 SMD 失败 (可能库版本不同)，请手动检查 CSV 文件中的 SMD 列。")

    except Exception as e:
        print(f"❌ TableOne 执行失败: {e}")

if __name__ == "__main__":
    run_step_10_cross_audit()
