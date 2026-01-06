from __future__ import annotations
import json
import re
import ast
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Match <think>...</think> tags
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.S)
# Match <json>...</json> tags
_JSON_TAG_PATTERN = re.compile(r"<json>\s*(.*?)\s*</json>", re.S)
# Match Markdown code blocks
_MARKDOWN_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)

@dataclass
class ParseIssue:
    stage: str
    message: str
    details: Optional[Dict[str, Any]] = None

class HardParseError(RuntimeError):
    def __init__(self, issue: ParseIssue):
        super().__init__(f"[{issue.stage}] {issue.message}")
        self.issue = issue

# ==========================================
#  Helper Functions
# ==========================================

def remove_think_content(text: str) -> str:
    if not text: return ""
    return _THINK_PATTERN.sub("", text).strip()

def _try_fix_truncated_json(text: str) -> str:
    """Attempt to fix truncated JSON strings."""
    text = text.strip()
    # Simple stack balance check (for object closing)
    if text.rfind('}') < text.rfind('{'):
        # Missing closing brace
        text += '}'
    if text.rfind(']') < text.rfind('['):
        text += ']'
    # Quote completion
    if text.count('"') % 2 != 0:
        text += '"'
    return text

def _extract_json_objects_stream(text: str) -> List[Any]:
    """
    Stream scan for all JSON objects using JSONDecoder.raw_decode.
    """
    decoder = json.JSONDecoder()
    results = []
    text_len = len(text)
    idx = 0
    
    while idx < text_len:
        while idx < text_len and text[idx].isspace():
            idx += 1
        if idx >= text_len: break
            
        if text[idx] in ('{', '['):
            try:
                obj, end_idx = decoder.raw_decode(text, idx)
                results.append(obj)
                idx = end_idx
                continue
            except json.JSONDecodeError:
                pass
        idx += 1
    return results

def _extract_json_via_regex(text: str) -> List[Any]:
    """
    Best-Effort: Use regex to extract strings that look like JSON objects.
    """
    results = []
    # Strategy: Process line by line, trying to extract {...}
    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        if not line: continue
        # Find first { and last }
        start = line.find('{')
        end = line.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = line[start : end+1]
            try:
                obj = json.loads(candidate)
                results.append(obj)
            except:
                # Try fixing truncation
                try:
                    fixed = _try_fix_truncated_json(candidate)
                    obj = json.loads(fixed)
                    results.append(obj)
                except:
                    pass
    return results

def _enforce_schema(obj: Any) -> Any:
    if isinstance(obj, list):
        return [_enforce_schema(item) for item in obj]
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            val = _enforce_schema(v)
            if k in ["k", "column", "entity", "value", "id"]:
                val = str(val)
            elif k in ["target_ratio", "current_ratio", "gap_ratio", "min_non_empty_ratio", "non_empty_ratio", "add_articles", "add_rows"]:
                try: val = float(val) 
                except: pass
            
            if isinstance(val, list) and k not in ["multirow_gap_table", "sparse_gap_table", "numeric_gap_table", "operations"]:
                 val = [str(item) for item in val]
            new_obj[k] = val
        return new_obj
    return obj

# ==========================================
#  Core Parsing Logic
# ==========================================

def extract_gap_tables_json(md_text: str) -> Dict[str, Any]:
    m = _JSON_TAG_PATTERN.search(md_text)
    candidate_text = m.group(1) if m else md_text
    clean_text = remove_think_content(candidate_text)
    
    objects = _extract_json_objects_stream(clean_text)
    for obj in objects:
        if isinstance(obj, dict):
             if any(k in obj for k in ["multirow_gap_table", "stop_signal", "strategy_analysis"]):
                 return _enforce_schema(obj)
    
    if objects and isinstance(objects[0], dict):
        return _enforce_schema(objects[0])

    raise HardParseError(ParseIssue(
        stage="analyze_gap_tables",
        message="Gap_Tables parsing failed",
        details={"extracted": clean_text[:200]}
    ))

def parse_strategy_json(text: str) -> Dict[str, Any]:
    text = remove_think_content(text)
    m = _JSON_TAG_PATTERN.search(text)
    if m: text = m.group(1)
    
    objects = _extract_json_objects_stream(text)
    for obj in objects:
        if isinstance(obj, dict) and ("targeted_narray" in obj or "strategy" in obj):
            return _enforce_schema(obj)
            
    if objects and isinstance(objects[0], dict):
        return _enforce_schema(objects[0])

    raise HardParseError(ParseIssue(
        stage="strategy_json",
        message="Failed to extract valid Strategy JSON",
        details={"snippet": text[:500]}
    ))

def parse_jsonl_strict(text: str) -> List[Dict[str, Any]]:
    """
    Enhanced Parser: Supports streaming, regex fallback, and truncation repair.
    """
    text = remove_think_content(text)
    
    # 1. Prioritize content inside <json> tags
    m = _JSON_TAG_PATTERN.search(text)
    scope_text = m.group(1) if m else text

    valid_rows = []

    # 2. Strategy A: Standard streaming parsing
    try:
        raw_objects = _extract_json_objects_stream(scope_text)
        for obj in raw_objects:
            if isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict): valid_rows.append(_enforce_schema(item))
            elif isinstance(obj, dict):
                valid_rows.append(_enforce_schema(obj))
    except Exception:
        pass

    # 3. Strategy B: Regex line-by-line extraction (Best-Effort)
    if not valid_rows:
        fallback_objs = _extract_json_via_regex(scope_text)
        for obj in fallback_objs:
             if isinstance(obj, dict):
                valid_rows.append(_enforce_schema(obj))

    # 4. Strategy C: Python AST (Last resort)
    if not valid_rows:
        try:
            clean_t = scope_text.replace("```json", "").replace("```", "")
            val = ast.literal_eval(clean_t)
            if isinstance(val, list):
                valid_rows = [_enforce_schema(x) for x in val if isinstance(x, dict)]
            elif isinstance(val, dict):
                valid_rows = [_enforce_schema(val)]
        except:
            pass

    if not valid_rows:
        raise HardParseError(ParseIssue(
            stage="jsonl", 
            message="No valid rows parsed (even after Best-Effort attempts)",
            details={"snippet": text[:200]}
        ))
        
    return valid_rows