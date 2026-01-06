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

def build_value_pools(
    df: pd.DataFrame,
    *,
    article_col: str="Article",
    numeric_cols: Optional[List[str]]=None,
    numeric_bin_rules: Optional[Dict[str,float]]=None,
    low_freq_threshold: int=2,
    head_k: int=3
) -> Dict[str, Any]:
    numeric_cols = numeric_cols or []
    numeric_bin_rules = numeric_bin_rules or {}
    pools: Dict[str, Any] = {}
    cols = [c for c in df.columns if c != article_col]
    for c in cols:
        s = df[c].map(_to_str)
        vc = s.value_counts()
        allowed = list(vc.index.astype(str))
        low_freq = list(vc[vc <= low_freq_threshold].index.astype(str))
        head_vals = list(vc.head(head_k).index.astype(str))
        col = {"allowed_values": allowed, "low_freq_values": low_freq, "head_values": head_vals}
        if c in numeric_cols and float(numeric_bin_rules.get(c, 0) or 0) > 0:
            bin_size = float(numeric_bin_rules[c])
            bin_to_values: Dict[str, List[str]] = {}
            for v in allowed:
                try:
                    fv = float(str(v).replace('%','').strip())
                    b = math.floor(fv/bin_size)*bin_size
                    key = f"[{b},{b+bin_size})"
                except Exception:
                    key = "NON_NUMERIC"
                bin_to_values.setdefault(key, []).append(v)
            col["bin_rule"] = bin_size
            col["bin_to_values"] = bin_to_values
        pools[c] = col
    return pools