import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import math

def load_table(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path, engine="openpyxl")
    return pd.read_csv(path)

def multirow_distribution(df: pd.DataFrame, article_col: str="Article") -> Dict[int, Dict[str, Any]]:
    counts = df[article_col].value_counts()
    total = len(counts)
    if total == 0:
        return {}
    vc = counts.value_counts().sort_index()
    return {
        int(k): {"num_articles": int(n), "ratio": float(n) / total}
        for k, n in vc.items()
    }

def entity_distributions(df: pd.DataFrame, article_col: str="Article") -> Dict[str, Dict[str, Dict[str, Any]]]:
    out = {}
    cols = [c for c in df.columns if c != article_col]
    total_rows = len(df)
    for c in cols:
        s = df[c].astype(str).replace({'nan': '-', 'None': '-', '<NA>': '-'})
        vc = s.value_counts(dropna=False)
        out[c] = {
            str(v): {"count": int(cnt), "ratio": float(cnt) / total_rows if total_rows else 0.0}
            for v, cnt in vc.items()
        }
    return out

def numeric_summary(df: pd.DataFrame, numeric_cols: List[str], bin_rules: Dict[str,float]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for c in numeric_cols:
        if c not in df.columns: continue
        s_clean = df[c].astype(str).str.replace(r'[^\d\.\-eE]', '', regex=True)
        s_num = pd.to_numeric(s_clean, errors='coerce')
        unique_count = df[c].nunique()
        freq1 = (df[c].value_counts() == 1).sum()
        
        bin_dist = {}
        bin_size = float(bin_rules.get(c, 0) or 0)
        if bin_size > 0:
            valid_nums = s_num.dropna()
            if not valid_nums.empty:
                floored = np.floor(valid_nums / bin_size) * bin_size
                counts = floored.value_counts()
                for val, count in counts.items():
                    key = f"[{val:g},{val + bin_size:g})"
                    bin_dist[key] = int(count)
            non_numeric_count = len(df) - len(valid_nums)
            if non_numeric_count > 0:
                bin_dist["NON_NUMERIC"] = int(non_numeric_count)
        out[c] = {"unique": int(unique_count), "freq1": int(freq1), "bin_distribution": bin_dist}
    return out

def analyze_head_tail_values(
    df: pd.DataFrame, 
    target_entity_constraints: Dict[str, Any],
    exclude_cols: List[str] = None
) -> Dict[str, Dict[str, List[str]]]:
    """
    1. Parse nested structures in target_entity_constraints:
       - head_ratio (Global high-frequency threshold)
       - tail_ratio_dict (Entity: Low-frequency threshold)
    2. Analyze only the entity columns present in tail_ratio_dict.
    """
    exclude_cols = set(exclude_cols or ["Article", "Gen_ID"])
    analysis = {}
    
    # Extract parameters
    head_threshold = float(target_entity_constraints.get("head_ratio", 0.5))
    tail_map = target_entity_constraints.get("tail_ratio_dict", {})
    
    # Analyze only columns defined in tail_ratio_dict
    target_cols = [c for c in df.columns if c in tail_map and c not in exclude_cols]
    
    for col in target_cols:
        # Get column-specific low-frequency threshold
        try:
            tail_threshold = float(tail_map[col])
        except (ValueError, TypeError):
            tail_threshold = 0.1 # Fallback
            
        vc = df[col].astype(str).value_counts(normalize=True)

        head = vc[vc > head_threshold].sort_values(ascending=False).index.tolist()
        tail = vc[vc < tail_threshold].sort_values(ascending=True).index.tolist()
        
        analysis[col] = {
            "head_values": head,
            "tail_values": tail
        }
    return analysis

def build_distribution_stats(
    df: pd.DataFrame,
    *,
    article_col: str="Article",
    numeric_cols: Optional[List[str]]=None,
    numeric_bin_rules: Optional[Dict[str,float]]=None,
    target_entity_constraints: Optional[Dict[str, Any]] = None,
    cooccur_pairs: Optional[List[Tuple[str,str]]]=None
) -> Dict[str, Any]:
    numeric_cols = numeric_cols or []
    numeric_bin_rules = numeric_bin_rules or {}
    target_entity_constraints = target_entity_constraints or {}
    
    stats = {
        "multirow_distribution": multirow_distribution(df, article_col=article_col),
        "entity_distributions": entity_distributions(df, article_col=article_col),
        "numeric_summary": numeric_summary(df, numeric_cols=numeric_cols, bin_rules=numeric_bin_rules),
        "value_frequency_analysis": analyze_head_tail_values(
            df, 
            target_entity_constraints=target_entity_constraints,
            exclude_cols=[article_col]
        )
    }
    if cooccur_pairs:
        stats["cooccurrence_pairs"] = cooccur_pairs
    return stats

def calculate_distribution_score(
    current_stats: Dict[str, Any],
    target_multirow: Dict[str, float],
    target_entity: Dict[str, Any]
) -> float:
    """
    Calculate distribution score (lower is better).
    Includes:
    1. Multi-row Distribution Score (Fit of multi-row distribution)
    2. Entity Threshold Score (Adherence to entity frequency thresholds): 
       Checks if the frequency of each entity value meets the minimum threshold defined in tail_ratio_dict.
       If frequency < threshold, a penalty is applied.
    """
    score = 0.0
    
    # --- 1. Multi-row Distribution Score ---
    curr_multi = current_stats.get("multirow_distribution", {})
    total_articles = sum(v["num_articles"] for v in curr_multi.values())
    curr_ratios = {}
    if total_articles > 0:
        curr_ratios = {str(k): v["num_articles"] / total_articles for k, v in curr_multi.items()}
            
    all_k = set(curr_ratios.keys()) | set([str(k) for k in target_multirow.keys()])
    for k in all_k:
        t_val = float(target_multirow.get(str(k), 0.0))
        c_val = float(curr_ratios.get(str(k), 0.0))
        # Weight 10.0
        score += abs(t_val - c_val) * 10.0 
    
    # --- 2. Entity Threshold Score ---
    # Goal: Measure if values of each entity defined in target_entity_constraints meet the low-frequency threshold.
    # Logic: For all values in a specific column, if (actual_freq < defined_tail_threshold), it's a violation.
    # Penalty = (threshold - actual_freq) * weight.
    
    tail_map = target_entity.get("tail_ratio_dict", {})
    curr_ent = current_stats.get("entity_distributions", {})
    
    for col, threshold in tail_map.items():
        if col not in curr_ent:
            # Major penalty if column is missing
            score += 5.0
            continue
            
        try:
            thresh_val = float(threshold)
        except:
            continue
            
        col_dist = curr_ent[col] # {"ValueA": {"ratio": 0.3, ...}, ...}
        
        for val_str, info in col_dist.items():
            real_ratio = info.get("ratio", 0.0)
            # If actual ratio is below threshold, the value is considered "too rare" and penalized.
            # The goal is to make all values either "Head" or at least satisfy "Tail Threshold".
            if real_ratio < thresh_val:
                penalty = (thresh_val - real_ratio) * 10.0 # Adjustable weight
                score += penalty

    return score