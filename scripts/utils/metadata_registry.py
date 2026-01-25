from dataclasses import dataclass
from typing import Optional, Tuple, Literal


@dataclass(frozen=True)
class FeatureSpec:
    # =====================
    # Identity
    # =====================
    name: str
    display_en: str
    display_cn: str

    # =====================
    # Representation
    # =====================
    latex: Optional[str] = None
    unit: Optional[str] = None
    unit_si: Optional[str] = None
    convert: Optional[str] = None  # name of conversion function

    # =====================
    # Preprocessing
    # =====================
    log_transform: bool = False
    zscore: bool = False
    impute_method: Optional[
        Literal["median", "mean", "mice", "constant_zero", "normal"]
    ] = "median"

    # =====================
    # Modeling control
    # =====================
    clinical_domain: str = "other"
    allow_in_model: bool = True
    allow_in_selection: bool = True  # feature selection (e.g., LASSO)
    table_role: Literal[
        "feature",   # model input
        "outcome",   # primary / secondary outcome
        "id",        # identifiers (subject_id, hadm_id)
        "group"      # grouping variables (sex, comorbidity)
    ] = "feature"

    # =====================
    # Clinical constraints
    # =====================
    ref_range: Optional[Tuple[float, float]] = None
    clip_bounds: Optional[Tuple[float, float]] = None

    # =====================
    # Metadata (cohort-specific)
    # =====================
    missing_rate: Optional[float] = None
    # NOTE:
    # - cohort- and database-specific
    # - populated post-profiling
    # - NOT used in modeling logic
