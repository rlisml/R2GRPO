# Task
You are now a data embellishment and harmonization expert.
The input is a set of **concatenated data (`Mixed Source Group`)**. This set of data may come from different original papers.
Your goal is to generate a new, self-consistent experimental data group; the Article ID must be unified as `{target_new_id}`.

# Input information
## 1. Domain knowledge (Norms & Entities)
{domain_norms}

## 2. Distribution optimization guidance (Head/Tail Analysis)
Please focus on each column’s high-frequency value set "head_values" and low-frequency value set "tail_values":
<json>
{head_tail_values_dict}
</json>

## 3. Concatenated source data
<json>
{source_rows_json}
</json>

# Modification steps
1. **Harmonize (Harmonize)**：
   - Eliminate inconsistencies introduced by concatenation. If the source data come from two different experiments, please adjust parameters (e.g., environment, annealing conditions) so that the whole group looks like a comparative experiment completed under the same experimental settings.
   - Ensure there are differences between every two rows of data. If there exist two/multiple rows of data such that, except for the `Article` and `PCEstabilized (%)` fields (entities), all other columns have identical values, then you must modify at least one column other than the `Article` and `PCEstabilized (%)` fields to some reasonable value from that column’s `tail_values`.
   - **Structure Preserving**：Keep the number of rows unchanged.

2. **Long-tail optimization (Head-to-Tail Optimization)**：
   - Scan the data; if you find a value in a column belongs to `head_values` (high frequency), then forcibly replace it with some reasonable value from that column’s `tail_values`.
   - Example：if "ETL" is "TiO2" (Head), change it to "SnO2 Sb" (Tail).

3. **Mutual constraints**：
   - Based on **domain knowledge**, if you modified column A, you must check and accordingly modify other columns to maintain scientific plausibility.
   - **PCE prediction**：If you change the value of any entity (column), you must reasonably estimate the change in `PCEstabilized (%)` (increase/decrease), and provide a specific value.

# Output requirements (strictly follow)
1. **Thinking Process (Thinking Process)**：First, please analyze in detail within the `<think>...</think>` tags the current harmonization strategy, the long-tail replacement choices, and the scientific plausibility checks.
2. **JSONL data**：Then, **MUST** and **ONLY** output data in JSONL format. All data must be strictly wrapped in `<json>` and `</json>` tags. One JSON object per line.
3. **ID unification**：All rows’ `Article` field must be set to `{target_new_id}`.
4. **Log**：Add an `_meta_change_log` field to each row, briefly describing what was modified and why.

# Output example
<think>
Analysis of the source data shows... Decide to change ETL from TiO2 to SnO2...
</think>
<json>
{"Article": "Gen_New_ID...", "ETL": "SnO2 Sb", "PCE": 21.5, "_meta_change_log": "Harmonized two data groups; detected high-frequency TiO2 and replaced it with low-frequency SnO2 Sb; therefore PCE increased from 20.1 to 21.5"}
{"Article": "Gen_New_ID...", "ETL": "SnO2 Sb", "PCE": 20.8, "_meta_change_log": "..."}
</json>