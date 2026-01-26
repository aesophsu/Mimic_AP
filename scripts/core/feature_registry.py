# scripts/core/feature_registry.py

from typing import Dict
from .spec import FeatureSpec

FEATURE_REGISTRY: Dict[str, FeatureSpec] = {

    # =====================
    # ID / Indexing (Identifiers)
    # =====================
    "subject_id": FeatureSpec(
        name="subject_id",
        display_en="Subj_ID",
        display_cn="患者ID",
        clinical_domain="other",
        allow_in_model=False,
        allow_in_selection=False,
        table_role="id",
    ),

    "hadm_id": FeatureSpec(
        name="hadm_id",
        display_en="HADM_ID",
        display_cn="住院ID",
        clinical_domain="other",
        allow_in_model=False,
        allow_in_selection=False,
        table_role="id",
    ),

    "stay_id": FeatureSpec(
        name="stay_id",
        display_en="Stay_ID",
        display_cn="ICU留滞ID",
        clinical_domain="other",
        allow_in_model=False,
        allow_in_selection=False,
        table_role="id",
    ),
    # =====================
    # Demographics & Baseline
    # =====================
    "gender": FeatureSpec(
        name="gender",
        display_en="Gender",
        display_cn="性别",
        clinical_domain="demographics",
        table_role="group",            # 作为分组变量，常用于亚组分析
        allow_in_selection=False,      # 通常作为固定基线调整，不参与特征筛选
    ),

    "admission_age": FeatureSpec(
        name="admission_age",
        display_en="Age",
        display_cn="年龄",
        unit="years",
        zscore=True,
        clinical_domain="demographics",
        table_role="feature",
        time_anchor="hospital_admit",
    ),

    "weight_admit": FeatureSpec(
        name="weight_admit",
        display_en="Weight",
        display_cn="体重",
        unit="kg",
        zscore=True,
        clinical_domain="demographics",
        table_role="feature",
        time_anchor="icu_admit",
    ),

    "los": FeatureSpec(
        name="los",
        display_en="LOS",              # Length of Stay 缩写
        display_cn="住院时长",
        unit="days",
        clinical_domain="demographics",
        table_role="outcome",          # 倾向于作为结局指标
        time_aggregation="sum",
    ),
    # =====================
    # Outcomes: Primary & Composite
    # =====================
    "pof": FeatureSpec(
        name="pof",
        display_en="POF",              # Persistent Organ Failure
        display_cn="持续性器官衰竭",
        clinical_domain="outcome",
        table_role="outcome",
        allow_in_model=False,
        allow_in_selection=False,
    ),

    "composite": FeatureSpec(
        name="composite",
        display_en="Composite",        # 复合结局
        display_cn="复合结局",
        clinical_domain="outcome",
        table_role="outcome",
        allow_in_model=False,
        allow_in_selection=False,
    ),

    # =====================
    # Outcomes: Organ-specific POF
    # =====================
    "resp_pof": FeatureSpec(
        name="resp_pof",
        display_en="Resp-POF",         # 呼吸系统持续衰竭
        display_cn="呼吸功能持续衰竭",
        clinical_domain="outcome",
        table_role="outcome",
        allow_in_model=False,
    ),

    "cv_pof": FeatureSpec(
        name="cv_pof",
        display_en="CV-POF",           # Cardiovascular 简写
        display_cn="心血管功能持续衰竭",
        clinical_domain="outcome",
        table_role="outcome",
        allow_in_model=False,
    ),

    "renal_pof": FeatureSpec(
        name="renal_pof",
        display_en="Renal-POF",
        display_cn="肾脏功能持续衰竭",
        clinical_domain="outcome",
        table_role="outcome",
        allow_in_model=False,
    ),

    # =====================
    # Outcomes: Mortality
    # =====================
    "mortality": FeatureSpec(
        name="mortality",
        display_en="Mortality",
        display_cn="病死率",
        clinical_domain="outcome",
        table_role="outcome",
        allow_in_model=False,
    ),

    "early_death_24_48h": FeatureSpec(
        name="early_death_24_48h",
        display_en="Early Death",      # 24-48h 早期死亡
        display_cn="早期死亡",
        clinical_domain="outcome",
        table_role="outcome",
        allow_in_model=False,
    ),
    # =====================
    # Interventions (Window-summary flags)
    # =====================
    "mechanical_vent_flag": FeatureSpec(
        name="mechanical_vent_flag",
        display_en="MV",                # Mechanical Ventilation 缩写
        display_cn="机械通气",
        clinical_domain="intervention",
        table_role="confounder",        # 默认为混杂因素，常用于逻辑回归校正
        impute_method="constant_zero",  # 缺失通常意味着未进行该治疗，填补为0
        allow_in_model=True,
    ),

    "vaso_flag": FeatureSpec(
        name="vaso_flag",
        display_en="Vaso",              # Vasopressors 缩写
        display_cn="血管活性药",
        clinical_domain="intervention",
        table_role="confounder",
        impute_method="constant_zero",  # 同上，缺失视为未使用
        allow_in_model=True,
    ),
    # =====================
    # Severity Scores
    # =====================
    "sofa_score": FeatureSpec(
        name="sofa_score",
        display_en="SOFA",
        display_cn="SOFA评分",
        zscore=True,
        clinical_domain="severity",
        table_role="feature",
        time_aggregation=None,  # 已是聚合指标
    ),

    "apsiii": FeatureSpec(
        name="apsiii",
        display_en="APS III",
        display_cn="APS III评分",
        zscore=True,
        clinical_domain="severity",
        table_role="feature",
        time_aggregation=None,
    ),

    "sapsii": FeatureSpec(
        name="sapsii",
        display_en="SAPS II",
        display_cn="SAPS II评分",
        zscore=True,
        clinical_domain="severity",
        table_role="feature",
        time_aggregation=None,
    ),

    "oasis": FeatureSpec(
        name="oasis",
        display_en="OASIS",
        display_cn="OASIS评分",
        zscore=True,
        clinical_domain="severity",
        table_role="feature",
        time_aggregation=None,
    ),

    "lods": FeatureSpec(
        name="lods",
        display_en="LODS",
        display_cn="LODS评分",
        zscore=True,
        clinical_domain="severity",
        table_role="feature",
        time_aggregation=None,
    ),
    # =====================
    # Labs: Hematology (Blood Routine)
    # =====================
    "wbc_min": FeatureSpec(
        name="wbc_min",
        display_en="WBC (min)",
        display_cn="白细胞计数(最小)",
        latex=r"WBC_{min}",
        unit="10^9/L",
        zscore=True,
        log_transform=True,            # 感染指标通常呈偏态分布
        clinical_domain="lab",
        time_aggregation="min",
        time_window_hr=24,
    ),

    "wbc_max": FeatureSpec(
        name="wbc_max",
        display_en="WBC (max)",
        display_cn="白细胞计数(最大)",
        latex=r"WBC_{max}",
        unit="10^9/L",
        zscore=True,
        log_transform=True,
        clinical_domain="lab",
        time_aggregation="max",
        time_window_hr=24,
    ),

    "hematocrit_min": FeatureSpec(
        name="hematocrit_min",
        display_en="Hct (min)",
        display_cn="红细胞压积(最小)",
        latex=r"Hct_{min}",
        unit="%",
        zscore=True,
        clinical_domain="lab",
        time_aggregation="min",
    ),

    "hematocrit_max": FeatureSpec(
        name="hematocrit_max",
        display_en="Hct (max)",
        display_cn="红细胞压积(最大)",
        latex=r"Hct_{max}",
        unit="%",
        zscore=True,
        clinical_domain="lab",
        time_aggregation="max",
    ),

    "hemoglobin_min": FeatureSpec(
        name="hemoglobin_min",
        display_en="Hb (min)",
        display_cn="血红蛋白(最小)",
        latex=r"Hb_{min}",
        unit="g/dL",
        zscore=True,
        clinical_domain="lab",
        time_aggregation="min",
    ),

    "hemoglobin_max": FeatureSpec(
        name="hemoglobin_max",
        display_en="Hb (max)",
        display_cn="血红蛋白(最大)",
        latex=r"Hb_{max}",
        unit="g/dL",
        zscore=True,
        clinical_domain="lab",
        time_aggregation="max",
    ),

    "platelets_min": FeatureSpec(
        name="platelets_min",
        display_en="PLT (min)",
        display_cn="血小板计数(最小)",
        latex=r"PLT_{min}",
        unit="10^9/L",
        zscore=True,
        clinical_domain="lab",
        time_aggregation="min",
    ),

    "platelets_max": FeatureSpec(
        name="platelets_max",
        display_en="PLT (max)",
        display_cn="血小板计数(最大)",
        latex=r"PLT_{max}",
        unit="10^9/L",
        zscore=True,
        clinical_domain="lab",
        time_aggregation="max",
    ),

    "rdw_max": FeatureSpec(
        name="rdw_max",
        display_en="RDW (max)",
        display_cn="红细胞分布宽度(最大)",
        latex=r"RDW_{max}",
        unit="%",
        zscore=True,
        clinical_domain="lab",
        time_aggregation="max",
    ),
    # =====================
    # Labs: Renal / Metabolic
    # =====================
    "bun_min": FeatureSpec(
        name="bun_min",
        display_en="BUN (min)",
        display_cn="尿素氮(最小)",
        latex=r"BUN_{min}",
        unit="mg/dL",
        unit_si="mmol/L",
        convert="bun_mgdl_to_mmol",
        log_transform=True,
        zscore=True,
        clinical_domain="renal",
        time_aggregation="min",
    ),

    "bun_max": FeatureSpec(
        name="bun_max",
        display_en="BUN (max)",
        display_cn="尿素氮(最大)",
        latex=r"BUN_{max}",
        unit="mg/dL",
        unit_si="mmol/L",
        convert="bun_mgdl_to_mmol",
        log_transform=True,
        zscore=True,
        clinical_domain="renal",
        time_aggregation="max",
    ),

    "creatinine_min": FeatureSpec(
        name="creatinine_min",
        display_en="Cr (min)",
        display_cn="肌酐(最小)",
        latex=r"Cr_{min}",
        unit="mg/dL",
        unit_si="µmol/L",
        convert="creatinine_mgdl_to_umol",
        log_transform=True,
        zscore=True,
        clinical_domain="renal",
        time_aggregation="min",
    ),

    "creatinine_max": FeatureSpec(
        name="creatinine_max",
        display_en="Cr (max)",
        display_cn="肌酐(最大)",
        latex=r"Cr_{max}",
        unit="mg/dL",
        unit_si="µmol/L",
        convert="creatinine_mgdl_to_umol",
        log_transform=True,
        zscore=True,
        clinical_domain="renal",
        time_aggregation="max",
    ),

    "phosphate_min": FeatureSpec(
        name="phosphate_min",
        display_en="Phos (min)",
        display_cn="血磷(最小)",
        latex=r"P_{min}",
        unit="mg/dL",
        unit_si="mmol/L",
        zscore=True,
        clinical_domain="renal",
        time_aggregation="min",
    ),
    # =====================
    # Labs: Electrolytes & Acid-Base
    # =====================
    "sodium_min": FeatureSpec(
        name="sodium_min",
        display_en="Na (min)",
        display_cn="血钠(最小)",
        latex=r"Na^{+}_{min}",
        unit="mEq/L",
        unit_si="mmol/L",
        zscore=True,
        clinical_domain="electrolyte",
        time_aggregation="min",
    ),

    "sodium_max": FeatureSpec(
        name="sodium_max",
        display_en="Na (max)",
        display_cn="血钠(最大)",
        latex=r"Na^{+}_{max}",
        unit="mEq/L",
        unit_si="mmol/L",
        zscore=True,
        clinical_domain="electrolyte",
        time_aggregation="max",
    ),

    "potassium_min": FeatureSpec(
        name="potassium_min",
        display_en="K (min)",
        display_cn="血钾(最小)",
        latex=r"K^{+}_{min}",
        unit="mEq/L",
        unit_si="mmol/L",
        zscore=True,
        clinical_domain="electrolyte",
        time_aggregation="min",
    ),

    "potassium_max": FeatureSpec(
        name="potassium_max",
        display_en="K (max)",
        display_cn="血钾(最大)",
        latex=r"K^{+}_{max}",
        unit="mEq/L",
        unit_si="mmol/L",
        zscore=True,
        clinical_domain="electrolyte",
        time_aggregation="max",
    ),

    "chloride_min": FeatureSpec(
        name="chloride_min",
        display_en="Cl (min)",
        display_cn="血氯(最小)",
        latex=r"Cl^{-}_{min}",
        unit="mEq/L",
        unit_si="mmol/L",
        zscore=True,
        clinical_domain="electrolyte",
        time_aggregation="min",
    ),

    "chloride_max": FeatureSpec(
        name="chloride_max",
        display_en="Cl (max)",
        display_cn="血氯(最大)",
        latex=r"Cl^{-}_{max}",
        unit="mEq/L",
        unit_si="mmol/L",
        zscore=True,
        clinical_domain="electrolyte",
        time_aggregation="max",
    ),

    "bicarbonate_min": FeatureSpec(
        name="bicarbonate_min",
        display_en="HCO3 (min)",
        display_cn="碳酸氢盐(最小)",
        latex=r"HCO_{3min}^{-}",
        unit="mEq/L",
        unit_si="mmol/L",
        zscore=True,
        clinical_domain="electrolyte",
        time_aggregation="min",
    ),

    "bicarbonate_max": FeatureSpec(
        name="bicarbonate_max",
        display_en="HCO3 (max)",
        display_cn="碳酸氢盐(最大)",
        latex=r"HCO_{3max}^{-}",
        unit="mEq/L",
        unit_si="mmol/L",
        zscore=True,
        clinical_domain="electrolyte",
        time_aggregation="max",
    ),

    "aniongap_min": FeatureSpec(
        name="aniongap_min",
        display_en="AG (min)",
        display_cn="阴离子间隙(最小)",
        latex=r"AG_{min}",
        unit="mEq/L",
        zscore=True,
        clinical_domain="electrolyte",
        time_aggregation="min",
    ),

    "aniongap_max": FeatureSpec(
        name="aniongap_max",
        display_en="AG (max)",
        display_cn="阴离子间隙(最大)",
        latex=r"AG_{max}",
        unit="mEq/L",
        zscore=True,
        clinical_domain="electrolyte",
        time_aggregation="max",
    ),

    "ph_min": FeatureSpec(
        name="ph_min",
        display_en="pH (min)",
        display_cn="pH值(最小)",
        latex=r"pH_{min}",
        zscore=True,
        clinical_domain="acid-base",
        time_aggregation="min",
    ),

    "ph_max": FeatureSpec(
        name="ph_max",
        display_en="pH (max)",
        display_cn="pH值(最大)",
        latex=r"pH_{max}",
        zscore=True,
        clinical_domain="acid-base",
        time_aggregation="max",
    ),
    # =====================
    # Labs: Liver Function
    # =====================
    "albumin_min": FeatureSpec(
        name="albumin_min",
        display_en="Alb (min)",
        display_cn="白蛋白(最小)",
        latex=r"Alb_{min}",
        unit="g/dL",
        unit_si="g/L",
        zscore=True,
        clinical_domain="liver",
        time_aggregation="min",
    ),

    "albumin_max": FeatureSpec(
        name="albumin_max",
        display_en="Alb (max)",
        display_cn="白蛋白(最大)",
        latex=r"Alb_{max}",
        unit="g/dL",
        unit_si="g/L",
        zscore=True,
        clinical_domain="liver",
        time_aggregation="max",
    ),

    "bilirubin_total_min": FeatureSpec(
        name="bilirubin_total_min",
        display_en="TBil (min)",
        display_cn="总胆红素(最小)",
        latex=r"TBil_{min}",
        unit="mg/dL",
        unit_si="µmol/L",
        log_transform=True,
        zscore=True,
        clinical_domain="liver",
        time_aggregation="min",
    ),

    "bilirubin_total_max": FeatureSpec(
        name="bilirubin_total_max",
        display_en="TBil (max)",
        display_cn="总胆红素(最大)",
        latex=r"TBil_{max}",
        unit="mg/dL",
        unit_si="µmol/L",
        log_transform=True,
        zscore=True,
        clinical_domain="liver",
        time_aggregation="max",
    ),

    "alt_min": FeatureSpec(
        name="alt_min",
        display_en="ALT (min)",
        display_cn="谷丙转氨酶(最小)",
        latex=r"ALT_{min}",
        unit="IU/L",
        log_transform=True,
        zscore=True,
        clinical_domain="liver",
        time_aggregation="min",
    ),

    "alt_max": FeatureSpec(
        name="alt_max",
        display_en="ALT (max)",
        display_cn="谷丙转氨酶(最大)",
        latex=r"ALT_{max}",
        unit="IU/L",
        log_transform=True,
        zscore=True,
        clinical_domain="liver",
        time_aggregation="max",
    ),

    "ast_min": FeatureSpec(
        name="ast_min",
        display_en="AST (min)",
        display_cn="谷草转氨酶(最小)",
        latex=r"AST_{min}",
        unit="IU/L",
        log_transform=True,
        zscore=True,
        clinical_domain="liver",
        time_aggregation="min",
    ),

    "ast_max": FeatureSpec(
        name="ast_max",
        display_en="AST (max)",
        display_cn="谷草转氨酶(最大)",
        latex=r"AST_{max}",
        unit="IU/L",
        log_transform=True,
        zscore=True,
        clinical_domain="liver",
        time_aggregation="max",
    ),

    "alp_min": FeatureSpec(
        name="alp_min",
        display_en="ALP (min)",
        display_cn="碱性磷酸酶(最小)",
        latex=r"ALP_{min}",
        unit="IU/L",
        log_transform=True,
        zscore=True,
        clinical_domain="liver",
        time_aggregation="min",
    ),

    "alp_max": FeatureSpec(
        name="alp_max",
        display_en="ALP (max)",
        display_cn="碱性磷酸酶(最大)",
        latex=r"ALP_{max}",
        unit="IU/L",
        log_transform=True,
        zscore=True,
        clinical_domain="liver",
        time_aggregation="max",
    ),

    "tbar": FeatureSpec(
        name="tbar",
        display_en="T-Bil (avg)",      # 假设为总胆红素相关
        display_cn="总胆红素(均值/标量)",
        unit="mg/dL",
        log_transform=True,
        zscore=True,
        clinical_domain="liver",
    ),
    # =====================
    # Labs: Coagulation
    # =====================
    "inr_min": FeatureSpec(
        name="inr_min",
        display_en="INR (min)",
        display_cn="国际标准化比值(最小)",
        latex=r"INR_{min}",
        unit=None,                     # INR 为比值，无单位
        zscore=True,
        clinical_domain="coagulation",
        time_aggregation="min",
    ),

    "inr_max": FeatureSpec(
        name="inr_max",
        display_en="INR (max)",
        display_cn="国际标准化比值(最大)",
        latex=r"INR_{max}",
        zscore=True,
        clinical_domain="coagulation",
        time_aggregation="max",
    ),

    "pt_min": FeatureSpec(
        name="pt_min",
        display_en="PT (min)",
        display_cn="凝血酶原时间(最小)",
        latex=r"PT_{min}",
        unit="sec",
        zscore=True,
        clinical_domain="coagulation",
        time_aggregation="min",
    ),

    "pt_max": FeatureSpec(
        name="pt_max",
        display_en="PT (max)",
        display_cn="凝血酶原时间(最大)",
        latex=r"PT_{max}",
        unit="sec",
        zscore=True,
        clinical_domain="coagulation",
        time_aggregation="max",
    ),

    "ptt_min": FeatureSpec(
        name="ptt_min",
        display_en="aPTT (min)",       # 临床更常缩写为 aPTT
        display_cn="活化部分凝血活酶时间(最小)",
        latex=r"aPTT_{min}",
        unit="sec",
        zscore=True,
        clinical_domain="coagulation",
        time_aggregation="min",
    ),

    "ptt_max": FeatureSpec(
        name="ptt_max",
        display_en="aPTT (max)",
        display_cn="活化部分凝血活酶时间(最大)",
        latex=r"aPTT_{max}",
        unit="sec",
        zscore=True,
        clinical_domain="coagulation",
        time_aggregation="max",
    ),
    # =====================
    # Labs: Perfusion & Inflammation
    # =====================
    "lactate_max": FeatureSpec(
        name="lactate_max",
        display_en="Lactate (max)",
        display_cn="乳酸(最大)",
        latex=r"Lac_{max}",
        unit="mmol/L",
        log_transform=True,            # 乳酸呈偏态分布，严重时指数级上升
        zscore=True,
        clinical_domain="perfusion",
        time_aggregation="max",
    ),

    # =====================
    # Labs: Pancreatic
    # =====================
    "lipase_max": FeatureSpec(
        name="lipase_max",
        display_en="Lipase (max)",
        display_cn="脂肪酶(最大)",
        latex=r"Lip_{max}",
        unit="IU/L",
        log_transform=True,            # 胰腺炎时脂肪酶成倍增加
        zscore=True,
        clinical_domain="pancreas",
        time_aggregation="max",
    ),
    # =====================
    # Labs: Glucose (Mixed Sources)
    # =====================
    "glucose_min": FeatureSpec(
        name="glucose_min",
        display_en="Glc (min)",        # 泛指来源的血糖
        display_cn="血糖(最小)",
        latex=r"Glc_{min}",
        unit="mg/dL",
        unit_si="mmol/L",
        convert="glucose_mgdl_to_mmol",
        zscore=True,
        clinical_domain="metabolic",
        time_aggregation="min",
    ),

    "glucose_max": FeatureSpec(
        name="glucose_max",
        display_en="Glc (max)",
        display_cn="血糖(最大)",
        latex=r"Glc_{max}",
        unit="mg/dL",
        unit_si="mmol/L",
        convert="glucose_mgdl_to_mmol",
        zscore=True,
        clinical_domain="metabolic",
        time_aggregation="max",
    ),

    # =====================
    # Labs: Glucose (Lab-only)
    # =====================
    "glucose_lab_min": FeatureSpec(
        name="glucose_lab_min",
        display_en="Glc-Lab (min)",    # 明确标注为实验室来源
        display_cn="血糖-实验室(最小)",
        latex=r"Glc_{lab.min}",
        unit="mg/dL",
        unit_si="mmol/L",
        convert="glucose_mgdl_to_mmol",
        zscore=True,
        clinical_domain="metabolic",
        time_aggregation="min",
    ),

    "glucose_lab_max": FeatureSpec(
        name="glucose_lab_max",
        display_en="Glc-Lab (max)",
        display_cn="血糖-实验室(最大)",
        latex=r"Glc_{lab.max}",
        unit="mg/dL",
        unit_si="mmol/L",
        convert="glucose_mgdl_to_mmol",
        zscore=True,
        clinical_domain="metabolic",
        time_aggregation="max",
    ),
    # =====================
    # Vitals: Oxygenation
    # =====================
    "spo2_min": FeatureSpec(
        name="spo2_min",
        display_en="SpO2 (min)",
        display_cn="血氧饱和度(最小)",
        latex=r"SpO_{2min}",
        unit="%",
        zscore=True,
        clinical_domain="vital",
        time_aggregation="min",
    ),

    "spo2_max": FeatureSpec(
        name="spo2_max",
        display_en="SpO2 (max)",
        display_cn="血氧饱和度(最大)",
        latex=r"SpO_{2max}",
        unit="%",
        zscore=True,
        clinical_domain="vital",
        time_aggregation="max",
    ),
    # =====================
    # Temporal Dynamics: Slopes
    # =====================
    "glucose_slope": FeatureSpec(
        name="glucose_slope",
        display_en="Glc Slope",        # 血糖变化斜率
        display_cn="血糖变化率",
        latex=r"\Delta Glc",           # 使用 Delta 符号表示变化率
        unit="mg/dL/hr",
        unit_si="mmol/L/hr",
        zscore=True,
        clinical_domain="metabolic",
        time_aggregation="slope",      # 明确指定为斜率聚合
        time_anchor="icu_admit",
        time_window_hr=24,             # 建议根据实际窗口调整
    ),

    "spo2_slope": FeatureSpec(
        name="spo2_slope",
        display_en="SpO2 Slope",       # 血氧变化斜率
        display_cn="血氧变化率",
        latex=r"\Delta SpO_2",
        unit="%/hr",
        zscore=True,
        clinical_domain="vital",
        time_aggregation="slope",
        time_anchor="icu_admit",
        time_window_hr=24,
    ),
    # =====================
    # Labs: Oxygenation Derivatives
    # =====================
    "pao2fio2ratio_min": FeatureSpec(
        name="pao2fio2ratio_min",
        display_en="P/F Ratio (min)",
        display_cn="氧合指数(最小)",
        latex=r"PaO_2/FiO_{2min}",
        unit="mmHg",
        zscore=True,
        log_transform=False,           # 通常 P/F 比例分布较为均匀，但可视情况调整
        clinical_domain="respiratory",
        table_role="feature",          # 虽缺失率高，但临床价值极大
        time_aggregation="min",
        missing_rate=0.0,
        allow_in_selection=True,       # 建议在敏感性分析中保留
    ),
    # =====================
    # Comorbidities (Baseline History)
    # =====================
    "heart_failure": FeatureSpec(
        name="heart_failure",
        display_en="HF",                # Heart Failure
        display_cn="心力衰竭",
        clinical_domain="comorbidity",
        table_role="confounder",        # 设为混杂因素，用于固定校正
        allow_in_selection=False,       # 不参与自动筛选，由临床经验决定保留
        impute_method="constant_zero",  # 缺失通常代表无该病史
    ),

    "chronic_kidney_disease": FeatureSpec(
        name="chronic_kidney_disease",
        display_en="CKD",               # Chronic Kidney Disease
        display_cn="慢性肾脏病",
        clinical_domain="comorbidity",
        table_role="confounder",
        allow_in_selection=False,
        impute_method="constant_zero",
    ),

    "malignant_tumor": FeatureSpec(
        name="malignant_tumor",
        display_en="Malignancy",        # 恶性肿瘤简写
        display_cn="恶性肿瘤",
        clinical_domain="comorbidity",
        table_role="confounder",
        allow_in_selection=False,
        impute_method="constant_zero",
    ),
}
