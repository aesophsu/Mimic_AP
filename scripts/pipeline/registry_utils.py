# scripts/pipeline/registry_utils.py

import hashlib
import json
from typing import Dict
from dataclasses import asdict

from scripts.core.spec import FeatureSpec


# =============================================================================
# Registry hashing (anti-leakage contract)
# =============================================================================
def hash_registry(registry: Dict[str, FeatureSpec]) -> str:
    """
    Generate a deterministic hash for the feature registry.

    Notes
    -----
    - Only FeatureSpec definitions are hashed
    - Order-invariant
    - Used for audit & reproducibility
    """
    serializable = {
        name: asdict(spec)
        for name, spec in sorted(registry.items(), key=lambda x: x[0])
    }
    payload = json.dumps(serializable, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# =============================================================================
# Role-based column routing
# =============================================================================
def route_columns_by_role(
    df,
    registry: Dict[str, FeatureSpec],
):
    """
    Split dataframe columns by FeatureSpec.table_role.

    Returns
    -------
    dict with keys:
        - id
        - feature
        - confounder
        - outcome
        - group
    """
    routed = {
        "id": [],
        "feature": [],
        "confounder": [],
        "outcome": [],
        "group": [],
    }

    for col in df.columns:
        if col not in registry:
            continue  # already assumed whitelist-filtered upstream

        role = registry[col].table_role
        routed[role].append(col)

    return routed
