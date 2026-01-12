import os
import pandas as pd
import numpy as np
from tableone import TableOne

# =========================================================
# 1. 配置与路径
# =========================================================
BASE_DIR = ".."
MIMIC_PATH = os.path.join(BASE_DIR, "data/cleaned/mimic_for_model.csv")
EICU_PATH = os.path.join(BASE_DIR, "data/cleaned/eicu_for_table1.csv")
SAVE_DIR = os.path.join(BASE_DIR, "results")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def run_module_09_table_one():
    print("="*60)
    print("📊 模块 09: 跨库基线对比 (MIMIC-IV vs eICU)")
    print("="*60)

    # 加载数据
    if not (os.path.exists(MIMIC_PATH) and os.path.exists(EICU_PATH)):
        print("❌ 错误：找不到输入 CSV 文件。")
        return

    df_m = pd.read_csv(MIMIC_PATH)
    df_e = pd.read_csv(EICU_PATH)

    # 1. 标记来源（用于分组对比）
    df_m['Cohort'] = 'MIMIC-IV'
    df_e['Cohort'] = 'eICU'

    # 2. 统一特征工程逻辑（1/0 -> Yes/No, M/F -> Male/Female）
    # 确保性别列统一
    def unify_gender(x):
        if x in [1, 'M', 'Male']: return 'Male'
        if x in [0, 'F', 'Female']: return 'Female'
        return np.nan

    df_m['gender'] = df_m['gender'].apply(unify_gender)
    df_e['gender'] = df_e['gender'].apply(unify_gender)

    # 3. 选取展示变量
    # 这里建议展示最核心的 12 个 LASSO 特征和人口学指标
    common_vars = [
        'admission_age', 'gender', 'bmi', 
        'creatinine_max', 'bun_min', 'ast_max', 'alt_max', 
        'wbc_max', 'ph_min', 'potassium_max', 'spo2_max', 'pof'
    ]
    
    # 剔除两个库中都不存在的变量
    existing_vars = [v for v in common_vars if v in df_m.columns and v in df_e.columns]

    # 4. 合并数据并重置索引 (修复报错的关键)
    df_combined = pd.concat([df_m[existing_vars + ['Cohort']], 
                             df_e[existing_vars + ['Cohort']]], 
                            axis=0, ignore_index=True)

    # 5. 标签映射映射
    df_combined['pof'] = df_combined['pof'].map({1: 'Yes', 0: 'No'})

    # 6. 定义统计属性
    categorical = ['gender', 'pof']
    # 危重症指标通常呈偏态分布，建议使用中位数(四分位数)展示
    nonnormal = [v for v in existing_vars if v not in categorical and v != 'admission_age']

    print(f"⏳ 正在计算 {len(df_combined)} 例患者的统计数据...")
    try:
        mytable = TableOne(
            df_combined, 
            columns=existing_vars, 
            categorical=categorical, 
            groupby='Cohort', 
            nonnormal=nonnormal, 
            pval=True, 
            smd=True
        )
        
        # 打印到控制台
        print("\n" + mytable.tabulate(tablefmt="github"))
        
        # 保存到本地
        save_path = os.path.join(SAVE_DIR, "Table1_MIMIC_vs_eICU.csv")
        mytable.to_csv(save_path)
        print(f"\n✅ Table 1 已成功生成并保存至: {save_path}")

    except Exception as e:
        print(f"❌ TableOne 运行出错: {e}")

if __name__ == "__main__":
    run_module_09_table_one()
