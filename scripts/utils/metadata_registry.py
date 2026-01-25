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
    convert: Optional[str] = None  # name of unit conversion function

    # =====================
    # Preprocessing
    # =====================
    log_transform: bool = False
    zscore: bool = False

    # ⚠️ 默认 None，由 pipeline 根据 table_role 决定
    impute_method: Optional[
        Literal["median", "mean", "mice", "constant_zero", "normal"]
    ] = None

    # =====================
    # Temporal semantics
    # =====================
    time_aggregation: Optional[
        Literal[
            "first",
            "last",
            "min",
            "max",
            "mean",
            "median",
            "slope",
            "trend",
            "count"     # NEW: measurement frequency
        ]
    ] = None

    time_anchor: Optional[
        Literal[
            "icu_admit",
            "hospital_admit",
            "event_onset"
        ]
    ] = None

    # NEW: observation window length (in hours)
    time_window_hr: Optional[float] = None
    # Example: 24, 48, 72

    # =====================
    # Modeling control
    # =====================
    clinical_domain: str = "other"
    allow_in_model: bool = True
    allow_in_selection: bool = True

    table_role: Literal[
        "feature",      # model input
        "outcome",      # primary / secondary outcome
        "id",           # identifiers
        "group",        # grouping variable (sex, CKD)
        "confounder"    # NEW: adjusted but not selected
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
    # populated post-profiling, never drives preprocessing
