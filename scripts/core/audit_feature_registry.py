# scripts/core/audit_feature_registry.py

"""
Audit script for FEATURE_REGISTRY in the AP-POF study.
Checks:
- FeatureSpec type consistency
- Unit validity
- Time aggregation validity
- log_transform feasibility
- clip_bounds vs ref_range consistency
- Summarizes issues and exports Excel report
"""

import pandas as pd
from scripts.core.feature_registry import FEATURE_REGISTRY
from scripts.core.spec import FeatureSpec
from scripts.core import conversion
import pathlib


def audit_feature_registry():
    records = []
    
    # 允许的聚合方法白名单（对应 Spec 定义）
    ALLOWED_AGG = ["first", "last", "min", "max", "mean", "median", "slope", "trend", "count", None]
    
    for fname, spec in FEATURE_REGISTRY.items():
        # 1. 基本校验逻辑
        type_ok = isinstance(spec, FeatureSpec)
        unit_ok = (spec.unit is None) or (spec.unit in getattr(conversion, "ALL_UNITS", []))
        convert_ok = (spec.convert is None) or (spec.convert == "identity") or \
                     (spec.convert in conversion.CONVERSION_REGISTRY)
        agg_ok = spec.time_aggregation in ALLOWED_AGG
        
        # 2. 逻辑校验：Log Transform 安全性
        log_ok = True
        if spec.log_transform and spec.clip_bounds:
            if spec.clip_bounds[0] <= 0:
                log_ok = False # Log 无法处理非正值
        
        # 3. 逻辑校验：数值范围合理性
        clip_vs_ref_ok = True
        if spec.clip_bounds and spec.ref_range:
            # clip_bounds 必须包含（大于等于）ref_range
            if not (spec.clip_bounds[0] <= spec.ref_range[0] <= spec.ref_range[1] <= spec.clip_bounds[1]):
                clip_vs_ref_ok = False

        # 4. 汇总全量字段 (对齐 FeatureSpec 结构)
        records.append({
            # Identity
            "feature": fname,
            "display_en": spec.display_en,
            "display_cn": spec.display_cn,
            
            # Representation & Checks
            "latex": spec.latex,
            "unit": spec.unit,
            "unit_si": spec.unit_si,
            "unit_ok": unit_ok,
            "convert": spec.convert,
            "convert_ok": convert_ok,
            
            # Preprocessing
            "log_transform": spec.log_transform,
            "log_ok": log_ok,
            "zscore": spec.zscore,
            "impute_method": spec.impute_method,
            
            # Temporal semantics
            "time_aggregation": spec.time_aggregation,
            "agg_ok": agg_ok,
            "time_anchor": spec.time_anchor,
            "time_window_hr": spec.time_window_hr,
            
            # Modeling control
            "clinical_domain": spec.clinical_domain,
            "table_role": spec.table_role,
            "allow_in_model": spec.allow_in_model,
            "allow_in_selection": spec.allow_in_selection,
            
            # Clinical constraints & Checks
            "clip_bounds": spec.clip_bounds,
            "ref_range": spec.ref_range,
            "clip_vs_ref_ok": clip_vs_ref_ok,
            
            # Metadata
            "missing_rate": spec.missing_rate,
            "type_ok": type_ok
        })

    df = pd.DataFrame.from_records(records)
    
    # 定义核心 Issue 过滤条件
    issues = df[
        (~df["type_ok"]) | (~df["unit_ok"]) | (~df["convert_ok"]) | 
        (~df["agg_ok"]) | (~df["log_ok"]) | (~df["clip_vs_ref_ok"])
    ]

    # --- 打印报告 ---
    print("\n" + "="*40)
    print("      FEATURE REGISTRY FULL AUDIT")
    print("="*40)
    print(f"Total Registered Features: {len(df)}")
    print(f"Features with Issues:      {len(issues)}")
    
    if not issues.empty:
        print("\nISSUE DETAILS:")
        print(issues[["feature", "unit_ok", "convert_ok", "agg_ok", "log_ok", "clip_vs_ref_ok"]])

    # --- 导出全量 Excel ---
    try:
        # 获取当前脚本所在的绝对路径
        current_dir = pathlib.Path(__file__).parent.absolute()
        output_path = current_dir / "feature_registry_full_audit.xlsx"
        
        df.to_excel(output_path, index=False)
        print(f"\nFull audit report saved to: {output_path}")
    except Exception as e:
        print(f"\nFailed to save Excel: {e}")
    
    return df, issues

if __name__ == "__main__":
    audit_feature_registry()
