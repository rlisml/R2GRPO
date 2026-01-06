# 输入
- epoch_i_distribution：{epoch_i_distribution}
- 目标分布：{target_multirow_ratio} (Multi-row)
- 目标实体约束：{target_entity_constraints}
- 资源预算 (本轮最大新增行数)：{max_total_new_rows} 行
- 领域知识 (Norms & Entities)：{domain_norms}

# 任务
请分析 `multirow_distribution` (当前) 与 `target_multirow_ratio` (目标) 的差距。
生成一份**多行数据补充计划 (`targeted_narray`)**。

# 逻辑要求
1. **纯数量规划**：只需要规划**“缺多少、补多少”**。
2. **预算控制**：确保 `sum(target_k * count)` 不超过本轮资源预算。
3. **优先补短板**：优先补充那些比例远低于目标的 k 值组别。

# 输出 Schema (必须严格遵循)
请将输出包裹在 `<json>` 和 `</json>` 标签中。

<json>
{
  "epoch": {epoch},
  "summary": "分析差距...",
  "stop_signal": false, 
  "stop_reason": "...",
  "targeted_narray": [
    {
      "target_k": 5,
      "count": 3,
      "rationale": "k=5 组别目标为 10%，当前仅 2%，且总预算允许，故补充 3 组。"
    },
    {
      "target_k": 1,
      "count": 10,
      "rationale": "..."
    }
  ]
}
</json>