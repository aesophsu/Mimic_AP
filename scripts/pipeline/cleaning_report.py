# scripts/pipeline/cleaning_report.py

from typing import Dict, List
import pandas as pd
from scripts.core.spec import FeatureSpec


def build_cleaning_report(
    registry: Dict[str, FeatureSpec],
    decision_log: Dict[str, Dict],
) -> pd.DataFrame:
    """
    Parameters
    ----------
    registry : feature registry
    decision_log : runtime decisions collected during pipeline

    Returns
    -------
    pd.DataFrame
    """

    rows: List[dict] = []

    for name, spec in registry.items():
        log = decision_log.get(name, {})

        rows.append({
            "feature": name,
            "role": spec.table_role,
            "primary_unit": log.get("primary_unit"),
            "conversion_action": log.get("conversion_action"),
            "smart_patch_applied": log.get("smart_patch", False),
            "hard_clip": spec.clip_bounds is not None,
            "log_transform": getattr(spec, "log_transform", False),
            "zscore": getattr(spec, "zscore", False),
            "imputation": spec.impute_method,
            "final_unit_table1": spec.unit_si if spec.convert else spec.unit,
        })

    return pd.DataFrame(rows)


def export_cleaning_report(
    df_report: pd.DataFrame,
    out_prefix: str,
):
    df_report.to_csv(f"{out_prefix}_cleaning_report.csv", index=False)
    df_report.to_json(
        f"{out_prefix}_cleaning_report.json",
        orient="records",
        indent=2,
    )
