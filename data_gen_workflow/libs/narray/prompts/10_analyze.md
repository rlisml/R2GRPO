# Input
- Current-round data statistics (program-generated, string-based): {distribution_stats_json}
- Target distribution: {target_multirow_ratio}
- Target entity constraints: {target_entity_constraints}
- Previous-round strategy: {strategy_i_minus_1}

# Task
1. Generate the current-round distribution analysis epoch_i_distribution, which must include structured gap tables Gap_Tables (JSON):
  - multirow_gap_table: for each k (row-count group), provide target/current/gap (ratios and article counts)
  - entity_gap_table: for each entity column, provide the list of values to be supplemented / suppressed (at least top 20)
  - sparse_gap_table: the difference between the sparse-column non_empty_ratio and the target min_non_empty_ratio
2. Decide whether early stopping of the iteration is needed (Early Stopping).
   - If all multi-tuple distributions and entity distributions are already close to the targets target_multirow_ratio and target_entity_constraints, then recommend stopping.


# Output requirements (strictly follow)
1. **XML wrapping**: **MUST** and **ONLY** output one standard JSON object to describe the gap tables. The JSON data must be strictly wrapped in the `<json>` and `</json>` tags.
2. **JSON syntax**:
   - Must use double quotes.
   - Trailing commas are forbidden.
   - Comments are forbidden.


# Output format (strictly follow)
<json>
{
  "stop_signal": false,  // [boolean] Whether to recommend early stopping
  "stop_reason": "Because..., need to continue generating", // [str] 
  "summary": "...",
  "gap_analysis": {
    "multirow_gap_table": [],
    "entity_gap_table": {},
    "numeric_gap_table": []
  },

}
</json>