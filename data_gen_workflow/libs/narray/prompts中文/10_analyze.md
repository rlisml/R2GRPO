# 输入
- 当前轮数据统计（程序生成，按字符串口径）：{distribution_stats_json}
- 目标分布：{target_multirow_ratio}
- 实体目标约束：{target_entity_constraints}
- 上一轮策略：{strategy_i_minus_1}

# 任务
1. 生成本轮分布分析 epoch_i_distribution，必须包含结构化缺口表 Gap_Tables(JSON)：
  - multirow_gap_table：逐 k（行数组别）给出 target/current/gap（比例与文章数）
  - entity_gap_table：逐实体列给出需补齐/需抑制的 value 列表（至少 top 20）
  - sparse_gap_table：稀疏列 non_empty_ratio 与目标 min_non_empty_ratio 的差值
2. 决策是否需要提前终止迭代（Early Stopping）。
   - 如果所有多元组分布与实体分布均已接近目标target_multirow_ratio和target_entity_constraints，则建议停止。


# 输出要求（严格遵循）
1. **XML 封装**：**必须**且**只能**输出一个标准的 JSON 对象来描述缺口表。JSON 数据必须严格包裹在 `<json>` 和 `</json>` 标签中。
2. **JSON 语法**：
   - 必须使用双引号。
   - 禁止尾随逗号。
   - 禁止注释。


# 输出格式（严格遵循）
<json>
{
  "stop_signal": false,  // [boolean] 是否建议提前终止
  "stop_reason": "因为...，需继续生成", // [str] 
  "summary": "...",
  "gap_analysis": {
    "multirow_gap_table": [],
    "entity_gap_table": {},
    "numeric_gap_table": []
  },

}
</json>
