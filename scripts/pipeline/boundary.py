# scripts/pipeline/boundary.py

from typing import Dict
import pandas as pd

from scripts.core.spec import FeatureSpec


# =============================================================================
# Hard boundary scrubbing
# =============================================================================
def apply_hard_bounds(
    series: pd.Series,
    spec: FeatureSpec,
) -> Dict[str, object]:
    """
    Apply hard physical bounds (clip_bounds).

    Behavior
    --------
    - Values outside clip_bounds → NaN
    - No statistical outlier logic
    - Non-destructive otherwise

    Returns
    -------
    dict with:
        cleaned_series
        hard_dropped: int
    """
    x = series.copy()
    hard_dropped = 0

    if spec.clip_bounds is None:
        return {
            "cleaned_series": x,
            "hard_dropped": 0,
        }

    low, high = spec.clip_bounds
    mask_invalid = (x < low) | (x > high)

    hard_dropped = int(mask_invalid.sum())
    x.loc[mask_invalid] = pd.NA

    return {
        "cleaned_series": x,
        "hard_dropped": hard_dropped,
    }
