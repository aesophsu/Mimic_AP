# run_pipeline.py

from scripts.core.registry import load_registry
from scripts.pipeline.stage1_profiling import run_profiling
from scripts.pipeline.stage2_cleaning import run_unit_cleaning
from scripts.pipeline.stage3_scrubbing import run_hard_bounds
from scripts.pipeline.bifurcation import bifurcate_table1_and_modeling
from scripts.pipeline.imputation import fit_imputers, apply_imputers
from scripts.pipeline.modeling_transform import fit_scalers, apply_modeling_transforms
from scripts.pipeline.cleaning_report import build_cleaning_report, export_cleaning_report

import pandas as pd


def main():
    # ---------------------
    # Load
    # ---------------------
    registry = load_registry()
    df_raw = pd.read_csv("data/raw/mimic_features.csv")

    # ---------------------
    # Stage 1–2: unit logic
    # ---------------------
    df_unit_cleaned, decision_log = run_unit_cleaning(
        df_raw,
        registry,
    )

    # ---------------------
    # Stage 3: hard bounds
    # ---------------------
    df_cleaned = run_hard_bounds(
        df_unit_cleaned,
        registry,
    )

    # ---------------------
    # Stage 4: bifurcation
    # ---------------------
    df_table1, df_train, df_test = bifurcate_table1_and_modeling(
        df_cleaned,
        registry,
        outcome_col="mortality",
    )

    # ---------------------
    # Modeling branch
    # ---------------------
    imputers = fit_imputers(df_train, registry)
    df_train = apply_imputers(df_train, imputers)
    df_test = apply_imputers(df_test, imputers)

    scalers = fit_scalers(df_train, registry)
    df_train = apply_modeling_transforms(df_train, registry, scalers)
    df_test = apply_modeling_transforms(df_test, registry, scalers)

    df_train = df_train.dropna(subset=["mortality"])
    df_test = df_test.dropna(subset=["mortality"])

    # ---------------------
    # Export
    # ---------------------
    df_table1.to_csv("outputs/df_table1.csv", index=False)
    df_train.to_csv("outputs/df_train.csv", index=False)
    df_test.to_csv("outputs/df_test.csv", index=False)

    report = build_cleaning_report(registry, decision_log)
    export_cleaning_report(report, "outputs/pipeline")

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
