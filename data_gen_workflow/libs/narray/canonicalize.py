from __future__ import annotations
import re
from typing import Dict, Any, Optional

_ws = re.compile(r"\s+")

def norm_key(s: str) -> str:
    """
    Objective normalization: lowercase, trim, compress whitespace, remove all whitespace (optional).
    Does not introduce domain knowledge, only used to align different spellings of the same token.
    """
    s = s.strip().lower()
    s = _ws.sub(" ", s)
    # Aggressive alignment: remove all spaces (e.g., Ti O2 -> TiO2)
    s2 = s.replace(" ", "")
    return s2

def build_canonical_map(allowed_values: list[str]) -> Dict[str, str]:
    mp: Dict[str, str] = {}
    for v in allowed_values:
        k = norm_key(str(v))
        # Keep the first value when multiple values map to the same key
        mp.setdefault(k, str(v))
    return mp

def canonicalize_value(v: Any, canonical_map: Dict[str, str]) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    k = norm_key(s)
    return canonical_map.get(k)