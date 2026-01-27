# scripts/pipeline/unit_inference.py

from typing import Dict, Any
import numpy as np
import pandas as pd

from scripts.core.spec import FeatureSpec
from scripts.core.conversion import apply_feature_conversion


# =============================================================================
# Utility
# =============================================================================
def _fraction_in_range(
    values: pd.Series,
    ref_range,
) -> float:
    """
    Compute fraction of non-missing values within ref_range.
    """
    if ref_range is None:
        return np.nan

    x = values.dropna()
    if x.empty:
        return np.nan

    low, high = ref_range
    return ((x >= low) & (x <= high)).mean()


# =============================================================================
# Main API
# =============================================================================
def infer_feature_unit(
    series: pd.Series,
    spec: FeatureSpec,
    *,
    delta_threshold: float = 0.30,
    min_fraction: float = 0.70,
) -> Dict[str, Any]:
    """
    Infer dominant unit scale for a feature.

    Returns
    -------
    dict with keys:
        inferred_unit: "unit" | "unit_si" | "ambiguous"
        P_raw
        P_inv
        delta_P
        confidence: "high" | "medium" | "low"
    """
    # ---------------------
    # Guard conditions
    # ---------------------
    if spec.convert is None or spec.ref_range is None:
        return {
            "inferred_unit": "unit",
            "P_raw": np.nan,
            "P_inv": np.nan,
            "delta_P": np.nan,
            "confidence": "high",
        }

    # ---------------------
    # Raw scale
    # ---------------------
    P_raw = _fraction_in_range(series, spec.ref_range)

    # ---------------------
    # Inverse converted scale (SI -> unit)
    # ---------------------
    converted_inv = apply_feature_conversion(series, spec.convert)
    P_inv = _fraction_in_range(converted_inv, spec.ref_range)

    delta_P = P_inv - P_raw

    # ---------------------
    # Decision logic (explicit, auditable)
    # ---------------------
    if (
        delta_P >= delta_threshold
        and P_inv >= min_fraction
    ):
        inferred = "unit_si"
        confidence = "high"
    elif (
        delta_P <= -delta_threshold
        and P_raw >= min_fraction
    ):
        inferred = "unit"
        confidence = "high"
    else:
        inferred = "ambiguous"
        confidence = "medium"

    return {
        "inferred_unit": inferred,
        "P_raw": P_raw,
        "P_inv": P_inv,
        "delta_P": delta_P,
        "confidence": confidence,
    }


# =============================================================================
# Batch inference over dataframe
# =============================================================================
def infer_all_units(
    df: pd.DataFrame,
    registry: Dict[str, FeatureSpec],
) -> Dict[str, Dict[str, Any]]:
    """
    Infer units for all features in the registry.

    Returns
    -------
    Dict[col_name, inference_result]
    """
    results = {}

    for name, spec in registry.items():
        if spec.table_role != "feature":
            continue

        results[name] = infer_feature_unit(df[name], spec)

    return results
