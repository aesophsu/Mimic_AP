"""
feature_manager.py

The central engine for metadata-driven feature engineering.
Acts as the bridge between static FeatureSpec definitions and dynamic DataFrame operations.

Responsibilities:
1. Semantic Selection: Query features by role, domain, or modeling eligibility.
2. Data Alignment: Enforce unit consistency using conversion.py.
3. Pipeline Configuration: Generate config dicts for imputation and scaling.
4. Viz Support: Provide display names and LaTeX labels for plotting.
"""

from typing import List, Dict, Optional, Union, Literal, Any
import logging
import pandas as pd
import numpy as np

# Internal Core Imports
from scripts.core.spec import FeatureSpec
from scripts.core.conversion import apply_feature_conversion, unit_plausibility_check

# Lazy import to avoid circular dependency if registry imports manager (unlikely but safe)
try:
    from scripts.core.feature_registry import FEATURE_REGISTRY
except ImportError:
    FEATURE_REGISTRY = {}
    import warnings
    warnings.warn("FEATURE_REGISTRY could not be imported. FeatureManager will be empty.")

# Setup Logger
logger = logging.getLogger(__name__)

class FeatureManager:
    """
    Manager class to interface with FEATURE_REGISTRY.
    Use this instead of accessing the dictionary directly.
    """

    def __init__(self, registry: Optional[Dict[str, FeatureSpec]] = None):
        """
        Initialize the manager.
        Args:
            registry: Optional override for testing. Defaults to global FEATURE_REGISTRY.
        """
        self._registry = registry if registry is not None else FEATURE_REGISTRY
        self._validate_registry()

    def _validate_registry(self):
        """Internal sanity check for the registry."""
        if not self._registry:
            logger.warning("FeatureManager initialized with empty registry.")
        # Future: Add checks for duplicate names or invalid role assignments here.

    # =========================================================================
    # 1. Semantic Selection (Query API)
    # =========================================================================

    def get_spec(self, feature_name: str) -> FeatureSpec:
        """Retrieve the full Spec object for a given feature."""
        if feature_name not in self._registry:
            raise KeyError(f"Feature '{feature_name}' not found in registry.")
        return self._registry[feature_name]

    def select_features(
        self,
        roles: Union[str, List[str], None] = None,
        domains: Union[str, List[str], None] = None,
        model_ready: bool = False,
        allow_selection: bool = False,
        time_window: Optional[float] = None,
        exclude: Optional[List[str]] = None
    ) -> List[str]:
        """
        Master filter to retrieve feature names based on metadata criteria.

        Args:
            roles: Filter by table_role (e.g., 'feature', 'outcome', 'id').
            domains: Filter by clinical_domain (e.g., 'renal', 'vitals').
            model_ready: If True, only return features with allow_in_model=True.
            allow_selection: If True, only return features with allow_in_selection=True.
            time_window: Filter by specific time_window_hr.
            exclude: List of specific names to exclude explicitly.

        Returns:
            List of feature names (strings).
        """
        selected = []
        
        # Normalize inputs to lists
        if isinstance(roles, str): roles = [roles]
        if isinstance(domains, str): domains = [domains]
        if exclude is None: exclude = []

        for name, spec in self._registry.items():
            # 1. Exclusion check
            if name in exclude:
                continue

            # 2. Role check
            if roles and spec.table_role not in roles:
                continue

            # 3. Domain check
            if domains and spec.clinical_domain not in domains:
                continue

            # 4. Modeling flags
            if model_ready and not spec.allow_in_model:
                continue
            
            if allow_selection and not spec.allow_in_selection:
                continue

            # 5. Time window check
            if time_window is not None:
                if spec.time_window_hr != time_window:
                    continue

            selected.append(name)

        return selected

    def get_model_features(self) -> List[str]:
        """Shortcut: Get all features ready for model training (X matrix)."""
        return self.select_features(roles="feature", model_ready=True)

    def get_outcomes(self) -> List[str]:
        """Shortcut: Get all outcome variables (Y targets)."""
        return self.select_features(roles="outcome")

    def get_identifiers(self) -> List[str]:
        """Shortcut: Get ID columns (e.g., subject_id, stay_id)."""
        return self.select_features(roles="id")

    # =========================================================================
    # 2. Data Alignment & Execution (Engine API)
    # =========================================================================

    def align_to_si(
        self, 
        df: pd.DataFrame, 
        strict: bool = False, 
        log: bool = True
    ) -> pd.DataFrame:
        """
        In-place unit conversion to align DataFrame with Spec definitions.
        Iterates through columns, checks for 'convert' function in Spec, and applies it.

        Args:
            df: Input DataFrame.
            strict: If True, raise error if conversion fails.
            log: Print conversion details.
        
        Returns:
            The modified DataFrame (SI units).
        """
        df_aligned = df.copy()
        converted_count = 0

        for col in df_aligned.columns:
            if col not in self._registry:
                continue
            
            spec = self._registry[col]
            
            # Skip if no conversion logic defined
            if not spec.convert or spec.convert == "identity":
                continue

            if log:
                logger.info(f"Converting '{col}': {spec.unit} -> {spec.unit_si} via {spec.convert}")

            # Execute conversion from conversion.py
            try:
                df_aligned[col] = apply_feature_conversion(
                    df_aligned[col], 
                    spec.convert, 
                    strict=strict,
                    log=False # avoid spamming logs per row
                )
                converted_count += 1
            except Exception as e:
                msg = f"Failed to convert feature '{col}': {str(e)}"
                if strict:
                    raise RuntimeError(msg)
                logger.error(msg)

        if log and converted_count > 0:
            logger.info(f"Successfully converted {converted_count} features to SI units.")
        
        return df_aligned

    def clip_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Winsorization (clipping) based on Spec 'clip_bounds'.
        """
        df_clipped = df.copy()
        
        for col in df_clipped.columns:
            if col not in self._registry:
                continue
                
            spec = self._registry[col]
            if spec.clip_bounds:
                lower, upper = spec.clip_bounds
                # Use numpy clip for speed
                df_clipped[col] = df_clipped[col].clip(lower=lower, upper=upper)
        
        return df_clipped

    def audit_data(self, df: pd.DataFrame, strict: bool = False):
        """
        Run plausibility checks on the DataFrame.
        Wraps conversion.unit_plausibility_check.
        """
        logger.info("Starting Data Plausibility Audit...")
        issues_found = False
        
        for col in df.columns:
            if col not in self._registry:
                continue
            
            spec = self._registry[col]
            
            # Only check if we have a reference range
            if spec.ref_range:
                try:
                    unit_plausibility_check(
                        data=df[col],
                        feature_name=col,
                        unit=spec.unit_si if spec.unit_si else spec.unit,
                        ref_range=spec.ref_range,
                        strict=strict,
                        log=True
                    )
                except ValueError:
                    issues_found = True
                    if strict: raise

        if not issues_found:
            logger.info("Audit passed: All features within expected physiological ranges.")

    # =========================================================================
    # 3. Pipeline Configuration Generators (Strategy API)
    # =========================================================================

    def get_impute_strategies(self) -> Dict[str, str]:
        """
        Generate a configuration dictionary for imputation.
        Returns: { 'creatinine': 'median', 'fio2': 'mean', ... }
        """
        strategies = {}
        features = self.get_model_features() # Only care about model features
        
        for name in features:
            spec = self._registry[name]
            # Default fallback logic if not specified in spec
            method = spec.impute_method
            if not method:
                # Intelligent default: continuous -> median, binary -> mode (not handled here yet)
                method = "median" 
            strategies[name] = method
            
        return strategies

    def get_scaling_config(self) -> Dict[str, List[str]]:
        """
        Group features by their required transformation.
        Returns:
            {
                'log': ['bilirubin', 'crp'],
                'zscore': ['age', 'hr'],
                'passthrough': ['gender']
            }
        """
        config = {"log": [], "zscore": [], "passthrough": []}
        features = self.get_model_features()

        for name in features:
            spec = self._registry[name]
            
            # Priority: Log takes precedence over simple Z-score in many pipelines, 
            # or they might be sequential. Here we categorize for the primary transformer.
            if spec.log_transform:
                config["log"].append(name)
            elif spec.zscore:
                config["zscore"].append(name)
            else:
                config["passthrough"].append(name)
                
        return config

    # =========================================================================
    # 4. Visualization & Reporting Helpers
    # =========================================================================

    def get_display_dict(self, lang: Literal["en", "cn"] = "en", use_latex: bool = True) -> Dict[str, str]:
        """
        Create a mapping from raw feature names to pretty display names.
        Useful for axis labels in plotting.
        
        Format: "Display Name (Unit)" or "LaTeX Name (Unit)"
        """
        mapping = {}
        for name, spec in self._registry.items():
            # 1. Base Name
            if use_latex and spec.latex:
                base = spec.latex
            else:
                base = spec.display_cn if lang == "cn" else spec.display_en
            
            # 2. Unit suffix
            # Prefer SI unit if available (assuming data is aligned), else raw unit
            unit = spec.unit_si if spec.unit_si else spec.unit
            
            if unit and unit != "None":
                label = f"{base} ({unit})"
            else:
                label = base
            
            mapping[name] = label
            
        return mapping

# =============================================================================
# Singleton Instance (Optional convenience)
# =============================================================================
# Usage: from scripts.core.feature_manager import fm
fm = FeatureManager()
