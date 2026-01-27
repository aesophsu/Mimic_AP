# scripts/pipeline/patching.py

from typing import Dict, Tuple
import numpy as np
import pandas as pd

from scripts.core.spec import FeatureSpec
from scripts.core.conversion import apply_feature_conversion


# =============================================================================
# Internal helpers
# =============================================================================
def _in_ref_range(x: pd.Series, ref_range: Tuple[float, float]) -> pd.Series:
    low, high = ref_range
    return (x >= low) & (x <= high)


# =============================================================================
# Main API
# =============================================================================
def apply_unit_patching(
    series: pd.Series,
    spec: FeatureSpec,
    inference: Dict,
) -> Dict[str, object]:
    """
    Apply chained unit harmonization (A/B/C paths).

    Parameters
    ----------
    series:
        Raw feature values.
    spec:
        FeatureSpec.
    inference:
        Output from unit_inference.infer_feature_unit.

    Returns
    -------
    dict with:
        cleaned_series
        audit: {
            global_converted: bool
            inverse_patched: int
            forward_patched: int
        }
    """
    x = series.copy()
    audit = {
        "global_converted": False,
        "inverse_patched": 0,
        "forward_patched": 0,
    }

    # ---------------------
    # Guards
    # ---------------------
    if spec.convert is None or spec.ref_range is None:
        return {"cleaned_series": x, "audit": audit}

    ref_range = spec.ref_range
    inferred_unit = inference["inferred_unit"]

    # Track which values have already been patched
    patched_inverse = pd.Series(False, index=x.index)
    patched_forward = pd.Series(False, index=x.index)

    # ==========================================================
    # Path A: Global realignment (SI -> unit)
    # ==========================================================
    if inferred_unit == "unit_si":
        x = apply_feature_conversion(x, spec.convert)
        audit["global_converted"] = True

    # ==========================================================
    # Path B: Local inverse patching
    # (assume majority correct, minority in SI)
    # ==========================================================
    mask_out = ~_in_ref_range(x, ref_range)
    if mask_out.any():
        candidate = apply_feature_conversion(x[mask_out], spec.convert)
        mask_fixable = _in_ref_range(candidate, ref_range)

        x.loc[mask_out[mask_out].index[mask_fixable]] = candidate[mask_fixable]
        patched_inverse.loc[mask_out[mask_out].index[mask_fixable]] = True
        audit["inverse_patched"] += int(mask_fixable.sum())

    # ==========================================================
    # Path C: Forward compensation (only after global A)
    # ==========================================================
    if audit["global_converted"]:
        mask_out = ~_in_ref_range(x, ref_range)
        mask_out &= ~patched_inverse  # idempotent guard

        if mask_out.any():
            candidate = apply_feature_conversion(
                x[mask_out],
                spec.convert,
            )
            mask_fixable = _in_ref_range(candidate, ref_range)

            x.loc[mask_out[mask_out].index[mask_fixable]] = candidate[mask_fixable]
            patched_forward.loc[mask_out[mask_out].index[mask_fixable]] = True
            audit["forward_patched"] += int(mask_fixable.sum())

    return {
        "cleaned_series": x,
        "audit": audit,
    }
