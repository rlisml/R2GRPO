from __future__ import annotations
import pandas as pd
from typing import Dict, Any, List, Optional
import math

def _to_str(x: Any) -> str:
    if x is None:
        return "-"
    try:
        if isinstance(x, float) and math.isnan(x):
            return "-"
    except Exception:
        pass
    s = str(x).strip()
    return s if s else "-"

def validate_unknown_fields(
    df: pd.DataFrame,
    schema_cols: List[str],
    *,
    meta_prefix: str = "_",
) -> List[Dict[str, Any]]:
    """
    Objective constraint: Do not allow generation of new fields (columns/entities).
    Convention: Columns starting with meta_prefix are meta columns and skipped (e.g., _op_id).
    """
    schema_set = set(schema_cols)
    cols_non_meta = [c for c in df.columns if not str(c).startswith(meta_prefix)]
    unknown = sorted(list(set(cols_non_meta) - schema_set))
    if unknown:
        return [{"type": "unknown_field", "unknown_fields": unknown, "count": len(unknown)}]
    return []

def validate_value_in_pool(
    df: pd.DataFrame,
    value_pools: Dict[str, Any],
    *,
    article_col: str = "Article",
    missing_token: str = "-"
) -> List[Dict[str, Any]]:
    errs = []
    for c in df.columns:
        if c == article_col:
            continue
        if c not in value_pools:
            continue
        allowed = set(map(str, value_pools[c].get("allowed_values", [])))
        # Always allow missing token and empty string
        allowed.add(missing_token)
        allowed.add("")
        
        if not allowed:
            continue
            
        bad = ~df[c].map(_to_str).isin(allowed)
        if bad.any():
            errs.append({"type": "value_oov", "entity": c, "count": int(bad.sum())})
    return errs

def validate_numeric_in_pool(
    df: pd.DataFrame,
    value_pools: Dict[str, Any],
    numeric_cols: List[str],
) -> List[Dict[str, Any]]:
    errs = []
    for c in numeric_cols:
        if c not in df.columns or c not in value_pools:
            continue
        allowed = set(map(str, value_pools[c].get("allowed_values", [])))
        bad = ~df[c].map(_to_str).isin(allowed)
        if bad.any():
            errs.append({"type": "numeric_not_in_pool", "entity": c, "count": int(bad.sum())})
    return errs

def validate_multirow_quota(
    df: pd.DataFrame,
    quota: Dict[int, int],
    *,
    article_col: str = "Article"
) -> List[Dict[str, Any]]:
    if not quota:
        return []
    counts = df.groupby(article_col).size().value_counts().to_dict()
    errs = []
    for k, tgt in quota.items():
        cur = int(counts.get(int(k), 0))
        if cur < int(tgt):
            errs.append({"type": "multirow_quota", "k": int(k), "target_articles": int(tgt), "current_articles": cur})
    return errs

def validate_all(
    df: pd.DataFrame,
    *,
    schema_cols: List[str],
    value_pools: Dict[str, Any],
    numeric_cols: Optional[List[str]] = None,
    multirow_quota: Optional[Dict[int, int]] = None,
    article_col: str = "Article",
    meta_prefix: str = "_",
    missing_token: str = "-"
) -> Dict[str, Any]:
    numeric_cols = numeric_cols or []
    multirow_quota = multirow_quota or {}

    errs: List[Dict[str, Any]] = []
    errs += validate_unknown_fields(df, schema_cols=schema_cols, meta_prefix=meta_prefix)
    errs += validate_value_in_pool(df, value_pools=value_pools, article_col=article_col, missing_token=missing_token)
    errs += validate_numeric_in_pool(df, value_pools=value_pools, numeric_cols=numeric_cols)
    errs += validate_multirow_quota(df, quota=multirow_quota, article_col=article_col)

    counts: Dict[str, int] = {}
    for e in errs:
        counts[e["type"]] = counts.get(e["type"], 0) + 1
    return {"errors": errs, "error_counts": counts}