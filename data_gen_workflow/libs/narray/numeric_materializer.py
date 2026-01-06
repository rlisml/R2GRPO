import random
import re
import math
from typing import List, Union, Dict, Optional, Any

def _extract_number(v: Any) -> Optional[float]:
    """
    Robust numeric extraction:
    1. Attempt direct float conversion.
    2. Attempt regex extraction of the first valid number (handling units, inequalities, etc.).
    """
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    
    s = str(v).strip()
    if not s or s.lower() in ['-', 'na', 'n/a', 'nan', 'null', 'no', 'none']:
        return None
        
    # 1. Attempt direct float conversion
    try:
        return float(s)
    except ValueError:
        pass
        
    # 2. Regex extraction: matches integer, float, or scientific notation
    # Example: "approx 10.5 min" -> 10.5, "< 0.05%" -> 0.05
    match = re.search(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", s)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
            
    return None

def materialize_numeric_value(
    value_in: Any,
    allowed_values: List[str],
    bin_rule: float,
    bin_to_values: Dict[str, List[float]],
    rng: random.Random
) -> Union[str, float]:
    """
    Converts LLM-generated values or bin labels into concrete numeric values.
    """
    val_str = str(value_in).strip()
    
    # 0. Preprocessing: Handle explicit missing markers
    if val_str in ["-", "nan", "None", "", "MISSING"]:
        return "-"

    # 1. Priority: Check if it is a bin label (e.g., "BIN:[10,20)")
    # This usually indicates the LLM followed strict instructions.
    if val_str.startswith("BIN:") or val_str in bin_to_values:
        key = val_str
        if key.startswith("BIN:"):
            key = key[4:].strip()
        
        candidates = bin_to_values.get(key, [])
        if candidates:
            return rng.choice(candidates)

    # 2. Attempt direct number extraction
    # Even if LLM did not generate a BIN label but a concrete value (possibly with units), try to preserve it.
    extracted_num = _extract_number(val_str)
    if extracted_num is not None:
        return extracted_num

    # 3. Match against discrete value pool
    if val_str in allowed_values:
        return val_str

    # 4. Unable to parse, return "-" (to be handled by subsequent validation)
    return "-"