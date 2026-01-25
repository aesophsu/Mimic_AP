"""
conversion.py

Central registry for all unit conversion functions used in the study.

Design principles
-----------------
- Single source of truth for physical unit transformations
- Explicit, named conversions referenced by FeatureSpec.convert
- Safe for scalars, NumPy arrays, and Pandas Series
- Vectorized, NaN-safe, and reproducible

Intended for:
- Feature engineering pipelines
- Table 1 / baseline characteristic generation
- External validation (unit harmonization across databases)
"""

from typing import Callable, Dict, Union, Optional
import warnings
import numpy as np
import pandas as pd

# =============================================================================
# Public API for import
# =============================================================================
__all__ = [
    "Numeric",
    "ConversionFunc",
    "CONVERSION_REGISTRY",
    "apply_feature_conversion",
    "register_conversion",
]

# =============================================================================
# Type definitions
# =============================================================================
Numeric = Union[float, int, np.ndarray, pd.Series]
ConversionFunc = Callable[[Numeric], Numeric]

# =============================================================================
# Internal registry
# =============================================================================
_CONVERSION_REGISTRY: Dict[str, ConversionFunc] = {}

def register_conversion(name: str) -> Callable[[ConversionFunc], ConversionFunc]:
    """
    Decorator to register a unit conversion function.

    Args:
        name: Key used in FeatureSpec.convert

    Returns:
        Decorated function (unchanged)
    """
    def decorator(func: ConversionFunc) -> ConversionFunc:
        if name in _CONVERSION_REGISTRY:
            raise KeyError(f"Duplicate conversion key registered: '{name}'")
        _CONVERSION_REGISTRY[name] = func
        return func
    return decorator

# =============================================================================
# Utility: Safe multiplication (vectorized & tolerant)
# =============================================================================
def _safe_mul(x: Numeric, factor: float) -> Numeric:
    """
    Multiply by factor, safely handling non-numeric input.
    """
    try:
        return x * factor
    except Exception:
        return x

# =============================================================================
# 1. Renal & Metabolic Conversions
# =============================================================================
@register_conversion("bun_mgdl_to_mmol")
def bun_mgdl_to_mmol(x: Numeric) -> Numeric:
    """Blood urea nitrogen: mg/dL → mmol/L (Factor: 0.357)"""
    return _safe_mul(x, 0.357)

@register_conversion("creatinine_mgdl_to_umol")
def creatinine_mgdl_to_umol(x: Numeric) -> Numeric:
    """Serum creatinine: mg/dL → µmol/L (Factor: 88.4)"""
    return _safe_mul(x, 88.4)

@register_conversion("glucose_mgdl_to_mmol")
def glucose_mgdl_to_mmol(x: Numeric) -> Numeric:
    """Glucose: mg/dL → mmol/L (Factor: 0.0555)"""
    return _safe_mul(x, 0.0555)

@register_conversion("calcium_mgdl_to_mmol")
def calcium_mgdl_to_mmol(x: Numeric) -> Numeric:
    """Total calcium: mg/dL → mmol/L (Factor: 0.2495)"""
    return _safe_mul(x, 0.2495)

@register_conversion("bilirubin_mgdl_to_umol")
def bilirubin_mgdl_to_umol(x: Numeric) -> Numeric:
    """Total bilirubin: mg/dL → µmol/L (Factor: 17.1)"""
    return _safe_mul(x, 17.1)

# =============================================================================
# 2. Vital Signs & Identity Transforms
# =============================================================================
@register_conversion("fahrenheit_to_celsius")
def fahrenheit_to_celsius(x: Numeric) -> Numeric:
    """Body temperature: Fahrenheit → Celsius"""
    try:
        return (x - 32.0) * 5.0 / 9.0
    except Exception:
        return x

@register_conversion("wbc_to_si")
def wbc_to_si(x: Numeric) -> Numeric:
    """
    WBC count: K/µL → 10^9/L
    Numerically identical, semantic clarity only.
    """
    return x

@register_conversion("identity")
def identity(x: Numeric) -> Numeric:
    """Explicit no-op conversion."""
    return x

# =============================================================================
# 3. Public registry view
# =============================================================================
CONVERSION_REGISTRY: Dict[str, ConversionFunc] = _CONVERSION_REGISTRY

# =============================================================================
# 4. Helper API for pipeline integration
# =============================================================================
def apply_feature_conversion(
    data: Numeric,
    convert_name: Optional[str],
    *,
    strict: bool = False,
) -> Numeric:
    """
    Apply unit conversion based on FeatureSpec.convert.

    Args:
        data: Scalar, NumPy array, or Pandas Series
        convert_name: Key from FeatureSpec.convert
        strict: 
            True  → raise KeyError if conversion not found
            False → warn and return raw data

    Returns:
        Converted data (same type as input)
    """
    if convert_name is None or convert_name == "identity":
        return data

    if convert_name not in CONVERSION_REGISTRY:
        msg = f"Conversion method '{convert_name}' not found in CONVERSION_REGISTRY."
        if strict:
            raise KeyError(msg)
        warnings.warn(msg + " Returning raw data.")
        return data

    return CONVERSION_REGISTRY[convert_name](data)
