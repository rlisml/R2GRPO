# 任务
你现在是数据修饰与融合专家。
输入是一组**拼接的数据 (`Mixed Source Group`)**。这组数据可能来自不同的原始文献。
你的目标是生成一个新的、自洽的实验数据组，Article ID 必须统一为 `{target_new_id}`。

# 输入信息
## 1. 领域知识 (Norms & Entities)
{domain_norms}

## 2. 分布优化指引 (Head/Tail Analysis)
请重点关注每列的高频取值集合"head_values"与低频取值集合"tail_values"：
<json>
{head_tail_values_dict}
</json>

## 3. 拼接源数据
<json>
{source_rows_json}
</json>

# 修改步骤
1. **融合 (Harmonize)**：
   - 消除拼接带来的不一致。如果源数据来自两个不同的实验，请调整参数（如环境、退火条件），使整组数据看起来像是在同一个实验设定下完成的对比实验。
   - 确保每两行数据之间存在差异。如果存在两行/多行数据除了`Article`和`PCEstabilized (%)`字段（实体）之外所有列的取值均相同，则必须修改`Article`和`PCEstabilized (%)`字段外的至少一列的值为该列 `tail_values` 中的某个合理值。
   - **Structure Preserving**：保持行数不变。

2. **长尾优化 (Head-to-Tail Optimization)**：
   - 扫描数据，如果发现某列的值属于 `head_values`（高频），则将其强制替换为该列 `tail_values` 中的某个合理值。
   - 例如：若 "ETL" 是 "TiO2" (Head)，改为 "SnO2 Sb" (Tail)。

3. **相互约束**：
   - 根据 **领域知识**，如果修改了 A 列，必须检查并连带修改 其他 列以保持科学合理性。
   - **PCE 预测**：如果改变了任何一个实体（列）的取值，则必须合理预估 `PCEstabilized (%)` 的变化（变大/变小），并给出具体数值。

# 输出要求 (严格遵循)
1. **思考过程 (Thinking Process)**：首先，请在 `<think>...</think>` 标签中详细分析当前的融合策略、长尾替换选择以及科学合理性检查。
2. **JSONL 数据**：然后，**必须**且**只能**输出 JSONL 格式的数据。所有数据必须严格包裹在 `<json>` 和 `</json>` 标签中。每行一个 JSON 对象。
3. **ID 统一**：所有行的 `Article` 字段必须设为 `{target_new_id}`。
4. **日志**：在每行数据中增加 `_meta_change_log` 字段，简述修改了什么及其原因。

# 输出示例
<think>
分析源数据发现... 决定将 ETL 从 TiO2 修改为 SnO2...
</think>
<json>
{"Article": "Gen_New_ID...", "ETL": "SnO2 Sb", "PCE": 21.5, "_meta_change_log": "融合了两组数据；检测到高频 TiO2，替换为低频 SnO2 Sb；因此 PCE 从 20.1 提升至 21.5"}
{"Article": "Gen_New_ID...", "ETL": "SnO2 Sb", "PCE": 20.8, "_meta_change_log": "..."}
</json>