# scripts/pipeline/modeling_transform.py

from typing import Dict, Tuple
import numpy as np
import pandas as pd

from scripts.core.spec import FeatureSpec


# =============================================================================
# Fit scalers
# =============================================================================
def fit_scalers(
    df_train: pd.DataFrame,
    registry: Dict[str, FeatureSpec],
) -> Dict[str, Tuple[float, float]]:
    """
    Fit mean/std for Z-score scaling on training data only.

    Returns
    -------
    Dict[col_name, (mean, std)]
    """
    scalers = {}

    for name, spec in registry.items():
        if (
            spec.table_role == "feature"
            and spec.zscore
            and name in df_train.columns
        ):
            mean = df_train[name].mean()
            std = df_train[name].std()
            scalers[name] = (mean, std)

    return scalers


# =============================================================================
# Apply transforms
# =============================================================================
def apply_modeling_transforms(
    df: pd.DataFrame,
    registry: Dict[str, FeatureSpec],
    scalers: Dict[str, Tuple[float, float]],
) -> pd.DataFrame:
    """
    Apply log-transform and Z-score scaling.
    """
    df_out = df.copy()

    for name, spec in registry.items():
        if name not in df_out.columns:
            continue

        # ---------------------
        # Log transform
        # ---------------------
        if spec.table_role == "feature" and spec.log_transform:
            df_out[name] = np.log1p(df_out[name])

        # ---------------------
        # Z-score
        # ---------------------
        if spec.table_role == "feature" and spec.zscore:
            mean, std = scalers[name]
            if std > 0:
                df_out[name] = (df_out[name] - mean) / std

    return df_out
