from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional, Dict, Any

@dataclass
class LLMResponse:
    text: str
    raw: Optional[Dict[str, Any]] = None

class LLMClient(Protocol):
    def generate(self, prompt: str, *, system: Optional[str] = None, temperature: float = 0.2) -> LLMResponse:
        ...

# class MockLLMClient:
#     """离线联调用 mock：返回最小占位输出，保证管线可跑通。生产环境请替换。"""
#     def generate(self, prompt: str, *, system: Optional[str] = None, temperature: float = 0.2) -> LLMResponse:
#         if "Gap_Tables" in prompt:
#             return LLMResponse(
#                 "## Summary\nmock\n\n## Gap_Tables (JSON)\n```json\n"
#                 "{\"multirow_gap_table\":[],\"entity_gap_table\":{},\"sparse_gap_table\":[],\"numeric_gap_table\":[]}"
#                 "\n```\n"
#             )
#         if "strategy_i" in prompt or "operations" in prompt:
#             return LLMResponse(json.dumps({
#                 "epoch": 0,
#                 "summary": {"top_deficits": [], "intent": "mock"},
#                 "targets": {"multirow_ratio_target": {}, "entity_target_mode": "custom", "numeric_bin_rules": {}},
#                 "operations": [],
#                 "delta_from_prev": {"added_operations": [], "removed_operations": [], "modified_operations": []},
#                 "stopping_criteria": {"multirow_ratio_within": 0.01, "entity_oov_zero": False, "max_epoch": 1},
#                 "risk_control": {"anti_drift": [], "rollback_policy": "mock"}
#             }, ensure_ascii=False))
#         if "domain_norms" in prompt and "entity_description" in prompt:
#             return LLMResponse("## domain_norms\n- RuleID: R1\n  Type: Hard\n  Desc: mock\n\n## entity_description\n### ETL\n- mock\n")
#         return LLMResponse("mock")
