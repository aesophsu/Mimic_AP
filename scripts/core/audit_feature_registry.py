# scripts/core/audit_feature_registry.py

import pandas as pd
from scripts.core.feature_registry import FEATURE_REGISTRY
from scripts.core.spec import FeatureSpec
from scripts.core import conversion

# 可选：定义允许的时间聚合方法
ALLOWED_AGG = ["min", "max", "mean", "median", "count", "first", "last", "slope", None]

def audit_feature_registry():
    records = []
    
    for fname, spec in FEATURE_REGISTRY.items():
        # 类型检查
        type_ok = isinstance(spec, FeatureSpec)
        
        # 单位检查
        unit_ok = (spec.unit is None) or (spec.unit in conversion.ALL_UNITS)
        
        # 时间聚合检查
        agg_ok = spec.time_aggregation in ALLOWED_AGG
        
        # log_transform 检查
        log_ok = True
        if spec.log_transform:
            if hasattr(spec, "clip_bounds") and spec.clip_bounds is not None:
                if spec.clip_bounds[0] <= 0:
                    log_ok = False  # log无法处理非正值
        
        # clip_bounds vs ref_range 合理性
        clip_vs_ref_ok = True
        if hasattr(spec, "clip_bounds") and hasattr(spec, "ref_range"):
            if spec.clip_bounds and spec.ref_range:
                if not (spec.clip_bounds[0] <= spec.ref_range[0] <= spec.ref_range[1] <= spec.clip_bounds[1]):
                    clip_vs_ref_ok = False
        
        # 汇总每个特征
        records.append({
            "feature": fname,
            "type_ok": type_ok,
            "unit": spec.unit,
            "unit_ok": unit_ok,
            "time_aggregation": spec.time_aggregation,
            "agg_ok": agg_ok,
            "log_transform": spec.log_transform,
            "log_ok": log_ok,
            "clip_bounds": spec.clip_bounds,
            "ref_range": getattr(spec, "ref_range", None),
            "clip_vs_ref_ok": clip_vs_ref_ok,
            "allow_in_model": getattr(spec, "allow_in_model", None),
            "table_role": getattr(spec, "table_role", None),
        })
    
    df = pd.DataFrame.from_records(records)
    
    # 高亮不合理项
    issues = df[
        (~df["type_ok"]) |
        (~df["unit_ok"]) |
        (~df["agg_ok"]) |
        (~df["log_ok"]) |
        (~df["clip_vs_ref_ok"])
    ]
    
    print("===== FEATURE REGISTRY AUDIT =====")
    print(f"Total features: {len(df)}")
    print(f"Issues found: {len(issues)}")
    if len(issues) > 0:
        print(issues[["feature","unit","unit_ok","time_aggregation","agg_ok","log_transform","log_ok","clip_bounds","ref_range","clip_vs_ref_ok"]])
    
    # 输出 Excel 供科研 QC 使用
    df.to_excel("feature_registry_audit.xlsx", index=False)
    print("Audit report saved to feature_registry_audit.xlsx")
    
    return df, issues

if __name__ == "__main__":
    audit_feature_registry()
