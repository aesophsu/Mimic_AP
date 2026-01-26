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
        # 显式保持默认
        log_transform=False,
        zscore=False,
        impute_method=None,
        time_aggregation=None,
    ),

    "hadm_id": FeatureSpec(
        name="hadm_id",
        display_en="HADM_ID",
        display_cn="住院ID",
        clinical_domain="other",
        allow_in_model=False,
        allow_in_selection=False,
        table_role="id",
        log_transform=False,
        zscore=False,
        impute_method=None,
        time_aggregation=None,
    ),

    "stay_id": FeatureSpec(
        name="stay_id",
        display_en="Stay_ID",
        display_cn="ICU留滞ID",
        clinical_domain="other",
        allow_in_model=False,
        allow_in_selection=False,
        table_role="id",
        log_transform=False,
        zscore=False,
        impute_method=None,
        time_aggregation=None,
    ),
    
    # =====================
    # Demographics & Baseline
    # =====================
    "gender": FeatureSpec(
        name="gender",
        display_en="Gender",
        display_cn="性别",
        clinical_domain="demographics",
        table_role="group",
        allow_in_model=True,
        allow_in_selection=False,
        log_transform=False,
        zscore=False,
        impute_method="constant_zero",          # 补齐：通常使用众数填补
        ref_range=(0, 1),              # 补齐：以男性(0)为参考基准
    ),

    "admission_age": FeatureSpec(
        name="admission_age",
        display_en="Age",
        display_cn="年龄",
        unit="years",
        zscore=True,
        log_transform=False,
        clinical_domain="demographics",
        table_role="feature",
        time_anchor="hospital_admit",
        time_aggregation=None,         # 静态基线，无聚合过程
        clip_bounds=(18, 100),         # 剔除极端高龄或错误的录入
    ),

    "weight_admit": FeatureSpec(
        name="weight_admit",
        display_en="Weight",
        display_cn="体重",
        unit="kg",
        zscore=True,
        log_transform=False,
        clinical_domain="demographics",
        table_role="feature",
        time_anchor="icu_admit",
        time_aggregation=None,
        clip_bounds=(30, 250),         # 30kg - 250kg 是成人ICU合理范围
        ref_range=(50, 90),            # 参考范围仅供分布参考
    ),

    "los": FeatureSpec(
        name="los",
        display_en="LOS",
        display_cn="住院时长",
        unit="days",
        clinical_domain="demographics",
        table_role="outcome",
        allow_in_model=False,
        allow_in_selection=False,
        time_aggregation=None,         # 修正语法冒号
        time_anchor=None,
        time_window_hr=None,
        zscore=False,
        clip_bounds=(0, 100),
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
        log_transform=False,
        zscore=False,
    ),

    "composite": FeatureSpec(
        name="composite",
        display_en="Composite Outcome",
        display_cn="复合结局理论",
        clinical_domain="outcome",
        table_role="outcome",
        allow_in_model=False,
        allow_in_selection=False,
        log_transform=False,
        zscore=False,
    ),

    # =====================
    # Outcomes: Organ-specific POF
    # =====================
    "resp_pof": FeatureSpec(
        name="resp_pof",
        display_en="Resp-POF",          # Respiratory POF
        display_cn="呼吸功能持续衰竭",
        clinical_domain="outcome",
        table_role="outcome",
        allow_in_model=False,
        allow_in_selection=False,
        log_transform=False,
        zscore=False,
    ),

    "cv_pof": FeatureSpec(
        name="cv_pof",
        display_en="CV-POF",            # Cardiovascular POF
        display_cn="心血管功能持续衰竭",
        clinical_domain="outcome",
        table_role="outcome",
        allow_in_model=False,
        allow_in_selection=False,
        log_transform=False,
        zscore=False,
    ),

    "renal_pof": FeatureSpec(
        name="renal_pof",
        display_en="Renal-POF",         # Renal POF
        display_cn="肾脏功能持续衰竭",
        clinical_domain="outcome",
        table_role="outcome",
        allow_in_model=False,
        allow_in_selection=False,
        log_transform=False,
        zscore=False,
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
        allow_in_selection=False,
        log_transform=False,
        zscore=False,
    ),

    "early_death_24_48h": FeatureSpec(
        name="early_death_24_48h",
        display_en="Early Death",      # 24-48h 早期死亡
        display_cn="早期死亡",
        clinical_domain="outcome",
        table_role="outcome",
        allow_in_model=False,
        allow_in_selection=False,
        log_transform=False,
        zscore=False,
    ),

    # =====================
    # Interventions (Window-summary flags)
    # =====================
    "mechanical_vent_flag": FeatureSpec(
        name="mechanical_vent_flag",
        display_en="MV",                # Mechanical Ventilation 缩写
        display_cn="机械通气",
        clinical_domain="intervention",
        table_role="confounder",        # 作为混杂因素进入模型校正
        impute_method="constant_zero",  # 缺失通常意味着未进行该治疗
        allow_in_model=True,
        allow_in_selection=False,       # 强制入模校正，不参与筛选
        time_aggregation="max",         # 窗口内只要有一次即为 1
        time_window_hr=24.0,            # 预设观察窗口
    ),

    "vaso_flag": FeatureSpec(
        name="vaso_flag",
        display_en="Vaso",              # Vasopressors 缩写
        display_cn="血管活性药",
        clinical_domain="intervention",
        table_role="confounder",
        impute_method="constant_zero",
        allow_in_model=True,
        allow_in_selection=False,       # 强制入模校正
        time_aggregation="max",
        time_window_hr=24.0,
    ),
    
    # =====================
    # Severity Scores
    # =====================
    "sofa_score": FeatureSpec(
        name="sofa_score",
        display_en="SOFA",
        display_cn="SOFA评分",
        latex=r"SOFA",
        zscore=True,
        log_transform=False,
        clinical_domain="severity",
        table_role="feature",
        time_aggregation=None,         # 评分本身即为时间窗口内的聚合值
        time_anchor="icu_admit",
        time_window_hr=24.0,           # 通常基于入库前24h数据计算
        clip_bounds=(0, 24),           # SOFA 理论最大值 24
    ),

    "apsiii": FeatureSpec(
        name="apsiii",
        display_en="APS III",
        display_cn="APS III评分",
        latex=r"APS\ III",
        zscore=True,
        log_transform=False,
        clinical_domain="severity",
        table_role="feature",
        time_aggregation=None,
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(0, 299),
        ref_range=(0, 40),
    ),

    "sapsii": FeatureSpec(
        name="sapsii",
        display_en="SAPS II",
        display_cn="SAPS II评分",
        latex=r"SAPS\ II",
        zscore=True,
        log_transform=False,
        clinical_domain="severity",
        table_role="feature",
        time_aggregation=None,
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(0, 163),
        ref_range=(0, 29),
    ),

    "oasis": FeatureSpec(
        name="oasis",
        display_en="OASIS",
        display_cn="OASIS评分",
        latex=r"OASIS",
        zscore=True,
        log_transform=False,
        clinical_domain="severity",
        table_role="feature",
        time_aggregation=None,
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(0, 70),
        ref_range=(0, 20),
    ),

    "lods": FeatureSpec(
        name="lods",
        display_en="LODS",
        display_cn="LODS评分",
        latex=r"LODS",
        zscore=True,
        log_transform=False,
        clinical_domain="severity",
        table_role="feature",
        time_aggregation=None,
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(0, 22),
        ref_range=(0, 3),
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
        log_transform=True,            # 感染指标通常呈偏态分布
        zscore=True,
        clinical_domain="lab",
        table_role="feature",
        time_aggregation="min",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(0.1, 100),        # 100*10^9/L 以上极罕见且可能是白血病/错误
        ref_range=(4.0, 10.0),
    ),

    "wbc_max": FeatureSpec(
        name="wbc_max",
        display_en="WBC (max)",
        display_cn="白细胞计数(最大)",
        latex=r"WBC_{max}",
        unit="10^9/L",
        log_transform=True,
        zscore=True,
        clinical_domain="lab",
        table_role="feature",
        time_aggregation="max",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(0.1, 100),        # 100*10^9/L 以上极罕见且可能是白血病/错误
        ref_range=(4.0, 10.0),
    ),

"hematocrit_min": FeatureSpec(
        name="hematocrit_min",
        display_en="Hct (min)",
        display_cn="红细胞压积(最小)",
        latex=r"Hct_{min}",
        unit="%",
        zscore=True,
        clinical_domain="lab",
        table_role="feature",
        time_aggregation="min",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(10.0, 70.0),       # 极度贫血至极度红细胞增多
        ref_range=(36.0, 50.0),
    ),

    "hematocrit_max": FeatureSpec(
        name="hematocrit_max",
        display_en="Hct (max)",
        display_cn="红细胞压积(最大)",
        latex=r"Hct_{max}",
        unit="%",
        zscore=True,
        clinical_domain="lab",
        table_role="feature",
        time_aggregation="max",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(10.0, 70.0),
        ref_range=(36.0, 50.0),
    ),

    "hemoglobin_min": FeatureSpec(
        name="hemoglobin_min",
        display_en="Hb (min)",
        display_cn="血红蛋白(最小)",
        latex=r"Hb_{min}",
        unit="g/dL",
        unit_si="g/L",
        convert="gdl_to_gl_multiply_10", # 单位换算：1 g/dL = 10 g/L
        zscore=True,
        clinical_domain="lab",
        table_role="feature",
        time_aggregation="min",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(3.0, 25.0),        # 低于3.0需紧急输血，高于25.0极罕见
        ref_range=(12.0, 17.5),
    ),
    
    "hemoglobin_max": FeatureSpec(
        name="hemoglobin_max",
        display_en="Hb (max)",
        display_cn="血红蛋白(最大)",
        latex=r"Hb_{max}",
        unit="g/dL",
        unit_si="g/L",
        convert="gdl_to_gl_multiply_10",
        zscore=True,
        clinical_domain="lab",
        table_role="feature",
        time_aggregation="max",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(3.0, 25.0),
        ref_range=(12.0, 17.5),
    ),

    "platelets_min": FeatureSpec(
        name="platelets_min",
        display_en="PLT (min)",
        display_cn="血小板计数(最小)",
        latex=r"PLT_{min}",
        unit="10^9/L",
        log_transform=True,             # 建议开启：PLT在危重症下波动跨度大且呈偏态
        zscore=True,
        clinical_domain="lab",
        table_role="feature",
        time_aggregation="min",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(5, 2000),          # 排除录入错误的0
        ref_range=(150, 450),
    ),

    "platelets_max": FeatureSpec(
        name="platelets_max",
        display_en="PLT (max)",
        display_cn="血小板计数(最大)",
        latex=r"PLT_{max}",
        unit="10^9/L",
        log_transform=True,
        zscore=True,
        clinical_domain="lab",
        table_role="feature",
        time_aggregation="max",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(5, 2000),
        ref_range=(150, 450),
    ),

    "rdw_max": FeatureSpec(
        name="rdw_max",
        display_en="RDW (max)",
        display_cn="红细胞分布宽度(最大)",
        latex=r"RDW_{max}",
        unit="%",
        zscore=True,
        clinical_domain="lab",
        table_role="feature",
        time_aggregation="max",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(10.0, 35.0),
        ref_range=(11.0, 15.0),
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
        table_role="feature",
        time_aggregation="min",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        # 临床上限通常在150-200左右，排除200以上的极极端异常
        clip_bounds=(1.0, 200.0),      
        ref_range=(7.0, 20.0),
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
        table_role="feature",
        time_aggregation="max",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(1.0, 200.0),
        ref_range=(7.0, 20.0),
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
        table_role="feature",
        time_aggregation="min",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        # 成人肌酐极少超过30 mg/dL，即使是透析患者
        clip_bounds=(0.1, 30.0),       
        ref_range=(0.6, 1.3),
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
        table_role="feature",
        time_aggregation="max",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(0.1, 30.0),
        ref_range=(0.6, 1.3),
    ),

    "phosphate_min": FeatureSpec(
        name="phosphate_min",
        display_en="Phos (min)",
        display_cn="血磷(最小)",
        latex=r"P_{min}",
        unit="mg/dL",
        unit_si="mmol/L",
        convert="phosphate_mgdl_to_mmol", # 1 mg/dL ≈ 0.323 mmol/L
        log_transform=False,
        zscore=True,
        clinical_domain="renal",
        table_role="feature",
        time_aggregation="min",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(0.5, 15.0),
        ref_range=(2.5, 4.5),
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
        table_role="feature",
        time_aggregation="min",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(100.0, 180.0),      # 极度低钠至重度高钠
        ref_range=(135.0, 145.0),
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
        table_role="feature",
        time_aggregation="max",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(100.0, 180.0),
        ref_range=(135.0, 145.0),
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
        table_role="feature",
        time_aggregation="min",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(1.5, 10.0),         # 极低钾/极高钾
        ref_range=(3.5, 5.0),
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
        table_role="feature",
        time_aggregation="max",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(1.5, 10.0),
        ref_range=(3.5, 5.0),
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
        table_role="feature",
        time_aggregation="min",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(60.0, 160.0),
        ref_range=(98.0, 107.0),
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
        table_role="feature",
        time_aggregation="max",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(60.0, 160.0),
        ref_range=(98.0, 107.0),
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
        table_role="feature",
        time_aggregation="min",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(2.0, 60.0),         # 低于2表示极重度酸中毒
        ref_range=(22.0, 28.0),
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
        table_role="feature",
        time_aggregation="max",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(2.0, 60.0),
        ref_range=(22.0, 28.0),
    ),

    "aniongap_min": FeatureSpec(
        name="aniongap_min",
        display_en="AG (min)",
        display_cn="阴离子间隙(最小)",
        latex=r"AG_{min}",
        unit="mEq/L",
        zscore=True,
        clinical_domain="electrolyte",
        table_role="feature",
        time_aggregation="min",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(1.0, 60.0),
        ref_range=(8.0, 16.0),
    ),

    "aniongap_max": FeatureSpec(
        name="aniongap_max",
        display_en="AG (max)",
        display_cn="阴离子间隙(最大)",
        latex=r"AG_{max}",
        unit="mEq/L",
        zscore=True,
        clinical_domain="electrolyte",
        table_role="feature",
        time_aggregation="max",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(1.0, 60.0),
        ref_range=(8.0, 16.0),
    ),

    "ph_min": FeatureSpec(
        name="ph_min",
        display_en="pH (min)",
        display_cn="pH值(最小)",
        latex=r"pH_{min}",
        zscore=True,
        log_transform=False,           # pH 本身就是对数尺度，无需再次 log
        clinical_domain="acid-base",
        table_role="feature",
        time_aggregation="min",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        # 理论生理极限通常在 6.5 - 8.0 之间
        clip_bounds=(6.5, 8.0),        
        ref_range=(7.35, 7.45),
    ),

    "ph_max": FeatureSpec(
        name="ph_max",
        display_en="pH (max)",
        display_cn="pH值(最大)",
        latex=r"pH_{max}",
        zscore=True,
        log_transform=False,
        clinical_domain="acid-base",
        table_role="feature",
        time_aggregation="max",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(6.5, 8.0),
        ref_range=(7.35, 7.45),
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
        convert="gdl_to_gl_multiply_10", 
        zscore=True,
        clinical_domain="liver",
        table_role="feature",
        time_aggregation="min",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(1.0, 6.0),           # 低于1.0极罕见，高于6.0多为输注后
        ref_range=(3.5, 5.2),
    ),

    "albumin_max": FeatureSpec(
        name="albumin_max",
        display_en="Alb (max)",
        display_cn="白蛋白(最大)",
        latex=r"Alb_{max}",
        unit="g/dL",
        unit_si="g/L",
        convert="gdl_to_gl_multiply_10",
        zscore=True,
        clinical_domain="liver",
        table_role="feature",
        time_aggregation="max",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(1.0, 6.0),
        ref_range=(3.5, 5.2),
    ),

    "bilirubin_total_min": FeatureSpec(
        name="bilirubin_total_min",
        display_en="TBil (min)",
        display_cn="总胆红素(最小)",
        latex=r"TBil_{min}",
        unit="mg/dL",
        unit_si="µmol/L",
        convert="bilirubin_mgdl_to_umol",  # 1 mg/dL = 17.1 umol/L
        log_transform=True,
        zscore=True,
        clinical_domain="liver",
        table_role="feature",
        time_aggregation="min",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(0.1, 70.0),          # 重度黄疸上限可达 50-70
        ref_range=(0.3, 1.2),
    ),

    "bilirubin_total_max": FeatureSpec(
        name="bilirubin_total_max",
        display_en="TBil (max)",
        display_cn="总胆红素(最大)",
        latex=r"TBil_{max}",
        unit="mg/dL",
        unit_si="µmol/L",
        convert="bilirubin_mgdl_to_umol",
        log_transform=True,
        zscore=True,
        clinical_domain="liver",
        table_role="feature",
        time_aggregation="max",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(0.1, 70.0),
        ref_range=(0.3, 1.2),
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
        table_role="feature",
        time_aggregation="max",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(1.0, 10000.0),       # 急性肝衰竭可破万
        ref_range=(7.0, 55.0),
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
        table_role="feature",
        time_aggregation="max",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(1.0, 10000.0),
        ref_range=(8.0, 48.0),
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
        table_role="feature",
        time_aggregation="max",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(10.0, 4000.0),
        ref_range=(40.0, 130.0),
    ),

    "tbar": FeatureSpec(
        name="tbar",
        display_en="B/A Ratio",
        display_cn="胆红素/白蛋白比值",
        latex=r"\frac{TBil}{Alb}",
        zscore=True,
        log_transform=True, 
        clinical_domain="liver",
        table_role="feature",
        time_aggregation=None,
        time_anchor=None,
        unit=None,                        # 无量纲
        clip_bounds=(0.01, 50.0),         # 经验性区间
        ref_range=(0.05, 0.5),            # 估算值，随计算方法变化
    ),
    
    # =====================
    # Labs: Coagulation
    # =====================
    "inr_min": FeatureSpec(
        name="inr_min",
        display_en="INR (min)",
        display_cn="国际标准化比值(最小)",
        latex=r"INR_{min}",
        unit=None,                     
        log_transform=True,            # 建议开启：严重凝血障碍时呈指数级增长
        zscore=True,
        clinical_domain="coagulation",
        table_role="feature",
        time_aggregation="min",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(0.5, 15.0),       # 排除极端错误值，INR > 10 已极度危险
        ref_range=(0.8, 1.1),          # 未服用抗凝药的正常水平
    ),

    "inr_max": FeatureSpec(
        name="inr_max",
        display_en="INR (max)",
        display_cn="国际标准化比值(最大)",
        latex=r"INR_{max}",
        unit=None,
        log_transform=True,
        zscore=True,
        clinical_domain="coagulation",
        table_role="feature",
        time_aggregation="max",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(0.5, 15.0),
        ref_range=(0.8, 1.1),
    ),

    "pt_min": FeatureSpec(
        name="pt_min",
        display_en="PT (min)",
        display_cn="凝血酶原时间(最小)",
        latex=r"PT_{min}",
        unit="sec",
        log_transform=True,
        zscore=True,
        clinical_domain="coagulation",
        table_role="feature",
        time_aggregation="min",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(5.0, 150.0),      # PT上限通常不超150s
        ref_range=(11.0, 13.5),
    ),

    "pt_max": FeatureSpec(
        name="pt_max",
        display_en="PT (max)",
        display_cn="凝血酶原时间(最大)",
        latex=r"PT_{max}",
        unit="sec",
        log_transform=True,
        zscore=True,
        clinical_domain="coagulation",
        table_role="feature",
        time_aggregation="max",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(5.0, 150.0),
        ref_range=(11.0, 13.5),
    ),

    "ptt_min": FeatureSpec(
        name="ptt_min",
        display_en="aPTT (min)",
        display_cn="活化部分凝血活酶时间(最小)",
        latex=r"aPTT_{min}",
        unit="sec",
        log_transform=True,
        zscore=True,
        clinical_domain="coagulation",
        table_role="feature",
        time_aggregation="min",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(10.0, 150.0),
        ref_range=(25.0, 35.0),
    ),

    "ptt_max": FeatureSpec(
        name="ptt_max",
        display_en="aPTT (max)",
        display_cn="活化部分凝血活酶时间(最大)",
        latex=r"aPTT_{max}",
        unit="sec",
        log_transform=True,
        zscore=True,
        clinical_domain="coagulation",
        table_role="feature",
        time_aggregation="max",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(10.0, 150.0),
        ref_range=(25.0, 35.0),
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
        log_transform=True,            
        zscore=True,
        clinical_domain="perfusion",
        table_role="feature",
        time_aggregation="max",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        # 乳酸高于 20 通常提示极高死亡率或标本污染
        clip_bounds=(0.3, 30.0),       
        ref_range=(0.5, 2.0),
    ),

    # =====================
    # Labs: Pancreas
    # =====================
    "lipase_max": FeatureSpec(
        name="lipase_max",
        display_en="Lipase (max)",
        display_cn="脂肪酶(最大)",
        latex=r"Lip_{max}",
        unit="IU/L",
        log_transform=True,            
        zscore=True,
        clinical_domain="pancreas",
        table_role="feature",
        time_aggregation="max",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        # 严重急性胰腺炎时可高达数千
        clip_bounds=(5.0, 10000.0),    
        ref_range=(10.0, 160.0),
    ),

    # =====================
    # Labs: Glucose (Mixed Sources)
    # =====================
    "glucose_min": FeatureSpec(
        name="glucose_min",
        display_en="Glc (min)",
        display_cn="血糖(最小)",
        latex=r"Glc_{min}",
        unit="mg/dL",
        unit_si="mmol/L",
        convert="glucose_mgdl_to_mmol",
        log_transform=False,
        zscore=True,
        clinical_domain="metabolic",
        table_role="feature",
        time_aggregation="min",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        # 低于 20 为致死性低血糖，高于 1000 为极重度高血糖（如 HHS）
        clip_bounds=(20.0, 1000.0),    
        ref_range=(70.0, 100.0),
    ),

    "glucose_max": FeatureSpec(
        name="glucose_max",
        display_en="Glc (max)",
        display_cn="血糖(最大)",
        latex=r"Glc_{max}",
        unit="mg/dL",
        unit_si="mmol/L",
        convert="glucose_mgdl_to_mmol",
        log_transform=False,
        zscore=True,
        clinical_domain="metabolic",
        table_role="feature",
        time_aggregation="max",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(20.0, 1000.0),
        ref_range=(70.0, 100.0),
    ),

 # =====================
    # Labs: Glucose (Lab-only)
    # =====================
    "glucose_lab_min": FeatureSpec(
        name="glucose_lab_min",
        display_en="Glc-Lab (min)",
        display_cn="血糖-实验室(最小)",
        latex=r"Glc_{lab.min}",
        unit="mg/dL",
        unit_si="mmol/L",
        convert="glucose_mgdl_to_mmol",
        log_transform=False,
        zscore=True,
        clinical_domain="metabolic",
        table_role="feature",
        time_aggregation="min",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        # 对应 mmol/L 约 (1.1, 55.5)
        clip_bounds=(20.0, 1000.0),    
        ref_range=(70.0, 100.0),
    ),

    "glucose_lab_max": FeatureSpec(
        name="glucose_lab_max",
        display_en="Glc-Lab (max)",
        display_cn="血糖-实验室(最大)",
        latex=r"Glc_{lab.max}",
        unit="mg/dL",
        unit_si="mmol/L",
        convert="glucose_mgdl_to_mmol",
        log_transform=False,
        zscore=True,
        clinical_domain="metabolic",
        table_role="feature",
        time_aggregation="max",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(20.0, 1000.0),
        ref_range=(70.0, 100.0),
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
        log_transform=False,
        clinical_domain="vital",
        table_role="feature",
        time_aggregation="min",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        # 低于 50% 读数通常被视为伪影或濒死
        clip_bounds=(50.0, 100.0),     
        ref_range=(94.0, 100.0),
    ),

    "spo2_max": FeatureSpec(
        name="spo2_max",
        display_en="SpO2 (max)",
        display_cn="血氧饱和度(最大)",
        latex=r"SpO_{2max}",
        unit="%",
        zscore=True,
        log_transform=False,
        clinical_domain="vital",
        table_role="feature",
        time_aggregation="max",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        clip_bounds=(50.0, 100.0),
        ref_range=(94.0, 100.0),
    ),
    # =====================
    # Temporal Dynamics: Slopes
    # =====================
    "glucose_slope": FeatureSpec(
        name="glucose_slope",
        display_en="Glc Slope",
        display_cn="血糖变化率",
        latex=r"\frac{\Delta Glc}{\Delta t}",
        unit="mg/dL/hr",
        unit_si="mmol/L/hr",
        convert="glucose_mgdl_to_mmol",
        zscore=True,
        clinical_domain="metabolic",
        table_role="feature",
        time_aggregation="slope",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        # 限制每小时变化不超过 200 mg/dL (约 11 mmol/L)
        clip_bounds=(-200.0, 200.0),   
        ref_range=(-5.0, 5.0),         # 理想状态应波动极小
        impute_method="constant_zero", # 变化率为缺失通常意味着数值平稳（slope=0）
    ),

    "spo2_slope": FeatureSpec(
        name="spo2_slope",
        display_en="SpO2 Slope",
        display_cn="血氧变化率",
        latex=r"\frac{\Delta SpO_2}{\Delta t}",
        unit="%/hr",
        zscore=True,
        clinical_domain="vital",
        table_role="feature",
        time_aggregation="slope",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        # 限制每小时血氧升降不超过 20%
        clip_bounds=(-20.0, 20.0),      
        ref_range=(-1.0, 1.0),
        impute_method="constant_zero",
    ),

    # =====================
    # Labs: Oxygenation Derivatives
    # =====================
    "pao2fio2ratio_min": FeatureSpec(
        name="pao2fio2ratio_min",
        display_en="P/F Ratio (min)",
        display_cn="氧合指数(最小)",
        latex=r"PaO_2/FiO_{2min}",
        unit=None,                     # 虽然常用 mmHg，但严格意义上是比值
        zscore=True,
        log_transform=False,           # 临床上更关注原始线性区间（100/200/300断点）
        clinical_domain="respiratory",
        table_role="feature",
        time_aggregation="min",
        time_anchor="icu_admit",
        time_window_hr=24.0,
        allow_in_selection=True,
        clip_bounds=(20.0, 800.0),     
        ref_range=(400.0, 500.0),      # 正常成人参考范围
    ),
    
    # =====================
    # Comorbidities (Baseline History)
    # =====================
    "heart_failure": FeatureSpec(
        name="heart_failure",
        display_en="Heart Failure",
        display_cn="心力衰竭",
        latex=r"I_{HF}",
        clinical_domain="comorbidity",
        table_role="confounder",
        allow_in_selection=False,
        impute_method="constant_zero",
        clip_bounds=None,              # 二分类变量不使用截断
        ref_range=(0, 0),              # 参考值为无该病
    ),

    "chronic_kidney_disease": FeatureSpec(
        name="chronic_kidney_disease",
        display_en="CKD",
        display_cn="慢性肾脏病",
        latex=r"I_{CKD}",
        clinical_domain="comorbidity",
        table_role="confounder",
        allow_in_selection=False,
        impute_method="constant_zero",
        clip_bounds=None,
        ref_range=(0, 0),
    ),

    "malignant_tumor": FeatureSpec(
        name="malignant_tumor",
        display_en="Malignancy",
        display_cn="恶性肿瘤",
        latex=r"I_{Malignancy}",
        clinical_domain="comorbidity",
        table_role="confounder",
        allow_in_selection=False,
        impute_method="constant_zero",
        clip_bounds=None,
        ref_range=(0, 0),
    ),
}
