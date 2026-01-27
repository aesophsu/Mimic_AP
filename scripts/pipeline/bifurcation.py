# scripts/pipeline/bifurcation.py

from typing import Dict, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.core.spec import FeatureSpec
from scripts.core.conversion import apply_feature_conversion


# =============================================================================
# Main API
# =============================================================================
def bifurcate_table1_and_modeling(
    df: pd.DataFrame,
    registry: Dict[str, FeatureSpec],
    *,
    outcome_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split cleaned dataframe into:
        - Table 1 dataframe (SI units, no imputation)
        - Train dataframe (ML branch)
        - Test dataframe (ML branch)

    Returns
    -------
    df_table1, df_train, df_test
    """

    # ---------------------
    # Identify columns by role
    # ---------------------
    table1_cols = []
    model_cols = []

    for name, spec in registry.items():
        if spec.table_role in {"feature", "confounder", "outcome", "group"}:
            table1_cols.append(name)

        if spec.table_role in {"feature", "confounder", "outcome"} and spec.allow_in_model:
            model_cols.append(name)

    df_table1 = df[table1_cols].copy()
    df_model = df[model_cols].copy()

    # ---------------------
    # Table 1: convert to SI units
    # ---------------------
    for name, spec in registry.items():
        if (
            name in df_table1.columns
            and spec.table_role == "feature"
            and spec.convert is not None
        ):
            df_table1[name] = apply_feature_conversion(
                df_table1[name],
                spec.convert,
            )

    # ---------------------
    # Train / Test split (anti-leakage)
    # ---------------------
    y = df_model[outcome_col]

    df_train, df_test = train_test_split(
        df_model,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    return df_table1, df_train, df_test
