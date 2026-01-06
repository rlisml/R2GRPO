from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import re

class PromptStore:
    def __init__(self, prompts_dir: str):
        self.prompts_dir = Path(prompts_dir)

    def load(self, name: str) -> str:
        # Read template file content
        return (self.prompts_dir / name).read_text(encoding="utf-8")

    def render(self, name: str, variables: Dict[str, Any]) -> str:
        tmpl = self.load(name)
        
        # Use regex substitution instead of format_map.
        # Reason: Prompts often contain JSON examples with braces {},
        # which causes ValueError in native string formatting.
        # 
        # Regex explanation:
        # 1. \{ ... \} : Matches content inside curly braces
        # 2. ([a-zA-Z0-9_]+) : Matches variable names (alphanumeric + underscore only)
        # 
        # Result:
        # - "{epoch}" -> Matched, replaced by variables['epoch']
        # - "{ "key": "val" }" -> No match due to quotes/spaces, preserved as is (Safe for JSON)
        pattern = re.compile(r'\{([a-zA-Z0-9_]+)\}')
        
        def replace_func(match):
            key = match.group(1) # Get variable name
            val = variables.get(key)
            
            # If variable exists, replace with string value
            if val is not None:
                return str(val)
            
            # If not found, keep original (Simulating SafeDict behavior)
            return match.group(0)
            
        return pattern.sub(replace_func, tmpl)