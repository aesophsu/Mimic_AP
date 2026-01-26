# scripts/core/feature_registry.py

from typing import Dict
from .spec import FeatureSpec

FEATURE_REGISTRY: Dict[str, FeatureSpec] = {

    # =====================
    # Demographics
    # =====================
    "age": FeatureSpec(
        name="age",
        display_en="Age",
        display_cn="年龄",
        unit="years",
        zscore=True,
        clinical_domain="demographic",
        table_role="feature",
    ),

    "sex": FeatureSpec(
        name="sex",
        display_en="Sex",
        display_cn="性别",
        clinical_domain="demographic",
        table_role="group",
        allow_in_selection=False,
    ),

    # =====================
    # Renal function
    # =====================
    "creatinine": FeatureSpec(
        name="creatinine",
        display_en="Creatinine",
        display_cn="肌酐",
        latex="Cr",
        unit="mg/dL",
        unit_si="µmol/L",
        convert="creatinine_mgdl_to_umol",
        log_transform=True,
        zscore=True,
        clinical_domain="renal",
        ref_range=(0.3, 15),
        time_aggregation="max",
        time_anchor="icu_admit",
        time_window_hr=24,
    ),

    "bun": FeatureSpec(
        name="bun",
        display_en="BUN",
        display_cn="尿素氮",
        latex="BUN",
        unit="mg/dL",
        unit_si="mmol/L",
        convert="bun_mgdl_to_mmol",
        log_transform=True,
        zscore=True,
        clinical_domain="renal",
        ref_range=(1, 50),
        time_aggregation="max",
        time_anchor="icu_admit",
        time_window_hr=24,
    ),

    # =====================
    # Outcomes
    # =====================
    "pof": FeatureSpec(
        name="pof",
        display_en="Persistent organ failure",
        display_cn="持续性器官衰竭",
        table_role="outcome",
        allow_in_selection=False,
        allow_in_model=False,
    ),
}
