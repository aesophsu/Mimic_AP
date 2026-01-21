import os
import pandas as pd
import numpy as np
import joblib
from tableone import TableOne
from scipy import stats

# =========================================================
# 1. 配置与路径
# =========================================================
BASE_DIR = ".."
MIMIC_PATH = os.path.join(BASE_DIR, "data/cleaned/mimic_for_table1.csv")
EICU_PATH = os.path.join(BASE_DIR, "data/cleaned/eicu_for_table1.csv")

SAVE_DIR = os.path.join(BASE_DIR, "results/tables")
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

# 学术命名映射表
FEATURE_MAP = {
    'admission_age': 'Age, years',
    'gender': 'Sex, Male (%)',
    'bmi': 'BMI, kg/m²',
    'creatinine_max': 'Max Creatinine, mg/dL',
    'bun_max': 'Max BUN, mg/dL',
    'wbc_max': 'Max WBC, 10⁹/L',
    'lactate_max': 'Max Lactate, mmol/L',
    'pao2fio2ratio_min': 'Min PaO2/FiO2',
    'alt_max': 'Max ALT, U/L',
    'ast_max': 'Max AST, U/L',
    'alp_max': 'Max ALP, U/L',
    'pof': 'Persistent Organ Failure',
    'composite_outcome': 'Composite Outcome',
    'mortality_28d': '28-day Mortality',
    'Renal_Group': 'Renal Subgroup'
}

def run_module_03_audit():
    print("\n" + "="*85)
    print("📊 模块 03: 基线特征描述与跨库人群深度审计 (SMD Standardized)")
    print("="*85)

    # 1. 数据加载
    if not os.path.exists(MIMIC_PATH) or not os.path.exists(EICU_PATH):
        print("❌ 错误：输入文件丢失。")
        return

    df_m = pd.read_csv(MIMIC_PATH)
    df_e = pd.read_csv(EICU_PATH)

    # 2. 预处理与对齐
    df_m['Cohort'] = 'MIMIC-IV (Derivation)'
    df_e['Cohort'] = 'eICU (External Val)'
    
    # 修复替换警告与下推行为
    for df in [df_m, df_e]:
        if 'gender' in df.columns:
            df['gender'] = df['gender'].replace({'M': 1, 'F': 0, 'Male': 1, 'Female': 0})
            df['gender'] = df['gender'].infer_objects(copy=False) # 显式调用以消除警告

    # 3. 自动探测共同变量
    common_vars = [v for v in df_e.columns if v in df_m.columns and v not in ['Cohort', 'id', 'stay_id']]
    categorical = ['gender', 'pof', 'composite_outcome', 'mortality_28d']
    categorical = [c for c in categorical if c in common_vars]

    # 4. 🧠 深度审计：正态性探测
    print("🧪 正在执行正态性检验...")
    nonnormal = []
    # 这里的 concat 仅用于探测分布，不需要 reset_index
    df_combined_temp = pd.concat([df_m[common_vars], df_e[common_vars]], axis=0)
    for var in [v for v in common_vars if v not in categorical]:
        data_sample = df_combined_temp[var].dropna()
        if len(data_sample) > 20:
            stat, p = stats.normaltest(data_sample.sample(min(len(data_sample), 1000)))
            if p < 0.05:
                nonnormal.append(var)
                
    # 5. 🚀 任务 A: 生成跨库对比表 (MIMIC vs eICU)
    print("\n⏳ 任务 A: 正在计算跨库 SMD 审计表...")
    df_cross = pd.concat([df_m, df_e], axis=0).reset_index(drop=True)
    table1 = TableOne(
        df_cross, columns=common_vars, categorical=categorical,
        groupby='Cohort', nonnormal=nonnormal, 
        pval=True, smd=True, 
        rename=FEATURE_MAP, display_all=True
    )
    table1.to_csv(os.path.join(SAVE_DIR, "Table1_Cross_Cohort_Audit.csv"))
    # --- 新增输出 ---
    print("\n📊 Table 1 核心内容预览 (MIMIC vs eICU):")
    print(table1.tableone.head(15)) # 展示前15行，涵盖人口学和结局

    # 6. 🚀 任务 B: 生成 MIMIC 内部单因素分析 (POF 分组)
    print("\n⏳ 任务 B: 正在计算 MIMIC 内部 POF 相关性分析...")
    internal_vars = [v for v in common_vars if v not in ['composite_outcome', 'mortality_28d']]
    table2 = TableOne(
        df_m, columns=internal_vars, categorical=[c for c in categorical if c == 'gender'],
        groupby='pof', nonnormal=nonnormal, 
        pval=True, rename=FEATURE_MAP
    )
    table2.to_csv(os.path.join(SAVE_DIR, "Table2_Internal_POF_Analysis.csv"))
    # --- 新增输出 ---
    print("\n📊 Table 2 核心内容预览 (POF vs Non-POF):")
    print(table2.tableone.head(10))

    # 7. 🚀 任务 C: 肾功能亚组对比 (Renal Subgroup)
    print("\n⏳ 任务 C: 正在计算 MIMIC 内部肾功能亚组分析...")
    df_m['Renal_Group'] = np.where(df_m['creatinine_max'] > 1.2, 'Renal Injury', 'Normal')
    renal_vars = [v for v in internal_vars if v != 'creatinine_max'] + ['Renal_Group']
    table3 = TableOne(
        df_m, columns=renal_vars, categorical=['gender', 'pof'],
        groupby='Renal_Group', nonnormal=nonnormal,
        pval=True, rename=FEATURE_MAP
    )
    table3.to_csv(os.path.join(SAVE_DIR, "Table3_Renal_Subgroup_Analysis.csv"))
    # --- 新增输出 ---
    print("\n📊 Table 3 核心内容预览 (Renal Subgroup):")
    print(table3.tableone.head(10))

    # 8. 🚀 任务 D: 三结局发生率对比审计
    print("\n📊 任务 D: 正在审计多结局发生率 (Incidence Analysis)...")
    outcomes = ['pof', 'composite_outcome', 'mortality_28d']
    mimic_inc = (df_m[outcomes].mean() * 100).rename('MIMIC-IV (%)')
    eicu_inc = (df_e[outcomes].mean() * 100).rename('eICU (%)')
    incidence_table = pd.concat([mimic_inc, eicu_inc], axis=1)
    incidence_table.index = [FEATURE_MAP.get(i, i) for i in incidence_table.index]
    
    print("-" * 50)
    print(incidence_table.round(2)) # 这里本身已有打印输出
    print("-" * 50)
    incidence_table.to_csv(os.path.join(SAVE_DIR, "Table4_Outcome_Incidence_Compare.csv"))
    
    print(f"\n✅ 审计完成！四张表格已保存至: {SAVE_DIR}")

if __name__ == "__main__":
    run_module_03_audit()
