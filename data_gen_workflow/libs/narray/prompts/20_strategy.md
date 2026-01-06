# Input
- epoch_i_distribution：{epoch_i_distribution}
- Target distribution：{target_multirow_ratio} (Multi-row)
- Target entity constraints：{target_entity_constraints}
- Resource budget (maximum new rows for this round)：{max_total_new_rows} rows
- Domain knowledge (Norms & Entities)：{domain_norms}

# Task
Please analyze the gap between `multirow_distribution` (current) and `target_multirow_ratio` (target).
Generate a **multi-row data supplementation plan (`targeted_narray`)**.

# Logical requirements
1. **Pure quantity planning**：You only need to plan **“how many are missing, how many to add”**。
2. **Budget control**：Ensure `sum(target_k * count)` does not exceed this round’s resource budget。
3. **Prioritize making up the shortfall**：Prioritize supplementing those k groups whose ratios are far below the target。

# Output Schema (must be strictly followed)
Please wrap the output in `<json>` and `</json>` tags.

<json>
{
  "epoch": {epoch},
  "summary": "Analyze the gap...",
  "stop_signal": false, 
  "stop_reason": "...",
  "targeted_narray": [
    {
      "target_k": 5,
      "count": 3,
      "rationale": "For the k=5 group, the target is 10%, the current is only 2%, and the total budget allows it, so add 3 groups."
    },
    {
      "target_k": 1,
      "count": 10,
      "rationale": "..."
    }
  ]
}
</json>