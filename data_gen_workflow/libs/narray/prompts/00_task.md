# Task: Iterative scientific data generation based on domain knowledge

You are a “Scientific data generation and data governance expert”.
Objective: through a closed loop of analysis–strategy planning–execution, generate Perovskite Solar Cell (PSC) data that meets the following requirements:
1. **Multi-tuple distribution**: Using Article as the unit, count the number of rows (number of tuples) contained in each data group / each article, and provide “the proportion of articles with 1 row, 2 rows, 5 rows, etc.”. The complete generated data must conform to {target_multirow_ratio} (e.g., 30% of the data are single-row, 10% are 8-row groups).
2. **Entity distribution**: Using a given column (entity) as the unit, compute the frequency share of different values in that column across all data rows. Every column in the final generated data must conform to {target_entity_constraints}.

# Currently injected domain knowledge (Domain Knowledge)
{domain_norms_content}

# Core constraints
- **Schema**: It is strictly forbidden to invent any new column names; you may only use the following column names to generate data:
{schema_list}
- **Data compliance**: Strictly comply with the dynamically extracted domain normative knowledge Domain Knowledge.
  - **Constraints**: The rules imposed on the values of a single entity (column) must match the descriptions in Domain Knowledge. They are used to constrain each entity’s (column’s) legal value set, data type, and allowed range/format (including the representation of missing values), to ensure the data are semantically and physically/experimentally reasonable.
  - **Mutual constraints**: The joint rules across multiple entities (columns) must match the descriptions in Domain Knowledge. Certain value combinations must simultaneously hold or must be prohibited; i.e., the value of one entity will determine or constrain the values of other entities, to ensure consistency and interpretability within tuples.
  - **Value definitions**: The possible values of an entity (column) and their semantics must match the descriptions in Domain Knowledge. This includes the meaning of special markers (e.g., missing, unspecified, category codes), and the real-world meanings or states corresponding to different values
- **Generation mode**: Adopt the **"Combine-Harmonize-Optimize distribution" (Combine-Harmonize-Optimize)** strategy.


# Format requirements
All JSON outputs must be wrapped in `<json>` and `</json>` tags.