# scripts/pipeline/imputation.py

from typing import Dict, Tuple
import pandas as pd

from sklearn.impute import SimpleImputer, IterativeImputer

from scripts.core.spec import FeatureSpec


# =============================================================================
# Imputer factory
# =============================================================================
def build_imputer(method: str):
    if method == "median":
        return SimpleImputer(strategy="median")
    if method == "mean":
        return SimpleImputer(strategy="mean")
    if method == "constant_zero":
        return SimpleImputer(strategy="constant", fill_value=0)
    if method == "mice":
        return IterativeImputer(random_state=42)
    if method is None:
        return None
    raise ValueError(f"Unknown impute_method: {method}")


# =============================================================================
# Fit / transform API
# =============================================================================
def fit_imputers(
    df_train: pd.DataFrame,
    registry: Dict[str, FeatureSpec],
) -> Dict[str, object]:
    """
    Fit imputers on training data only.

    Returns
    -------
    Dict[col_name, fitted_imputer]
    """
    imputers = {}

    for name, spec in registry.items():
        if spec.table_role not in {"feature", "confounder"}:
            continue

        if spec.impute_method is None:
            continue

        imputer = build_imputer(spec.impute_method)
        imputer.fit(df_train[[name]])
        imputers[name] = imputer

    return imputers


def apply_imputers(
    df: pd.DataFrame,
    imputers: Dict[str, object],
) -> pd.DataFrame:
    """
    Apply fitted imputers to dataframe.
    """
    df_out = df.copy()

    for name, imputer in imputers.items():
        df_out[name] = imputer.transform(df_out[[name]]).ravel()

    return df_out
