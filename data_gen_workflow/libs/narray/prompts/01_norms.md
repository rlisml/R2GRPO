# Task: Domain knowledge extraction

Please read all domain knowledge files provided under the {path2knowledge} path as well as the seed data Schema ({seed_schema_and_examples}).
Extract and summarize the following two aspects, and merge them into a single **"Domain Knowledge Document"**:

1. **Domain Norms (Domain Norms)**: Extract hard constraints about material properties, fabrication processes, and device structures.
   - **Constraints**: Rules imposed on the values of a single entity (column), used to constrain its legal value set, data type, and allowed range/format (including the representation of missing values), to ensure the data are semantically and physically/experimentally reasonable.
   - **Mutual constraints**: Joint rules across multiple entities (columns), specifying that certain value combinations must simultaneously hold or must be prohibited; i.e., the value of one entity will determine or constrain the values of other entities, to ensure consistency and interpretability within tuples.
   - **Value definitions**: Provide explicit explanations of the possible values of an entity (column) and their semantics, including the meaning of special markers (e.g., missing, unspecified, category codes), and the real-world meanings or states corresponding to different values, thereby ensuring the “value–meaning” mapping is clear and reusable.

2. **Entity Descriptions (Entity Descriptions)**:
   - Explain the meaning of each column in the Schema.
   - **Special attention**：Clearly distinguish the domain meanings of special values (e.g., "no" vs "0").
   - Explain the physical meaning of different entity values.



# Output requirements
- Directly output the summary content, using a clear Markdown heading structure.
- No additional separator tags are needed.