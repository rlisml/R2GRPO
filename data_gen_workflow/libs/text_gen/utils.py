import json

BASE_SYSTEM_PROMPT = """You are an expert scientific researcher and writer. 
Your task is to assist in generating, expanding, or refining scientific papers based on specific instructions.
Always maintain a formal, academic tone.
"""

# Reusable literalism guardrail (kept short so it can be embedded into multiple prompts).
LITERALISM_ASCII_ONLY = """
**Data Literalism (Critical)**:
1. **ASCII only**: Do NOT introduce unicode subscripts/superscripts (e.g., keep "FA0.83", not "FA₀.₈₃"; keep "Li+", not "Li⁺").
2. **No LaTeX**: Do NOT convert formulas/units to LaTeX unless it already exists in the input.
3. **Verbatim**: If an input value appears (especially inside <v>...</v> tags), keep it character-for-character.
"""

# [Note] This string is NOT formatted using .format(), so single braces are fine here.
TAGGING_INSTRUCTION = """
IMPORTANT OUTPUT FORMAT REQUIREMENTS:
You must strictly structure your response into three specific XML blocks:

1. <think>
   In this block, describe your reasoning process.
</think>

2. <paper>
   This block contains the **content of the current chapter/section**.
   - **Data Tagging**: When describing a value from the provided data table, you MUST use the <v>{{col_name}}::{{value}}</v> tag.
   - **DATA LITERALISM (CRITICAL)**: 
     - **DO NOT** convert chemical formulas or units to LaTeX (e.g., KEEP "TiO2", DO NOT write "$\\mathrm{TiO}_{2}$").
     - **DO NOT** use unicode subscripts/superscripts (e.g., KEEP "FA0.83", DO NOT write "FA₀.₈₃").
     - **COPY VERBATIM**: The value inside the tag MUST match the input data string exactly, character for character.
   - **No Markdown**: Do not include markdown code blocks.
</paper>

3. <other>
   Any other comments or notes.
</other>
"""

PHRASE_PROMPT_TEMPLATE = """
You are a professional academic editor. 

**Source Text**:
"{text_segment}"
(Note: Some values have been replaced with tags like <v>Column::Value</v>. Keep the tag structure strictly.)

**Reference Context**:
{ref_content}

**Task**:
Under the premise of ensuring the semantic meaning remains unchanged, rewrite Source Text sentence by sentence. 
Employ the following techniques:
1. **Synonym Replacement**: Use academic equivalents for key verbs and adjectives.
2. **Part-of-Speech Transformation**: Change noun-heavy phrases to verbal phrases (or vice versa).
3. **Voice Alternation**: Switch between active and passive voice where appropriate.
4. **Structural Reordering**: Alter the Subject-Verb-Object order or move prepositional phrases.
5. **Clause Reconstruction**: Convert simple sentences into complex ones using relative/adverbial clauses.

**Constraints**:
1. **Verbatim Terms**: DO NOT change the spelling or formatting of chemical formulas, material names, or specific scientific acronyms. Use them exactly as they appear in the Source Text.
2. **Tag Integrity**: Ensure all <v>...</v> tags are preserved strictly. Do not alter the content inside the tags.
3. **No LaTeX**: Do not introduce LaTeX formatting for chemical formulas unless it was already present in the Source Text outside of tags.
4. **ASCII only**: Do not introduce unicode subscripts/superscripts anywhere (e.g., keep "SnO2" not "SnO₂", keep "Li+" not "Li⁺").

**Output Format**: 
{tagging_instruction}
"""

SCALE_REWRITE_PROMPT_TEMPLATE = """
You are a scientific writer expanding a specific section of a paper.

**Original Section Content**:
"{text_segment}"

**Context**:
This section needs to be updated with new experimental data (marked with <v>...</v> tags) and potentially expanded to fit a larger paper structure.

**Task**:
1. Rewrite the section to incorporate the new data tags naturally.
2. Expand the logical flow if the original content is too brief.
3. **Data Literalism**: Do not format the data values. Use the exact string provided in the tags.
4. **ASCII only**: Do not introduce unicode subscripts/superscripts.

**Output Format**:
{tagging_instruction}
"""

MISSING_DATA_DESC_TEMPLATE = """
You are a scientific data interpreter.

**Task**: 
Convert the following isolated data points into a concise, grammatically correct academic sentence or short paragraph.

**Data Points**:
{missing_data_json}

**Context**: 
These parameters describe a Perovskite Solar Cell (or related scientific experiment).

**Constraint**:
{literalism_ascii_only}

**Output**: 
Return ONLY the descriptive text.
"""

GENERATE_CHAPTER_TEMPLATE = """
You are writing a specific chapter for a scientific paper.

**Current Chapter Title**: {chapter_title}
**Target Word Count**: Approx. {word_limit} words

**Context - Paper Summary**:
The paper so far contains:
"{generated_summary}"

**Reference Style/Content**:
(Use this as a reference for tone and writing style)
"{reference_content}"

**Task**:
Use {word_limit} words to write the content for the chapter "{chapter_title}".
1. Ensure the content flows logically from the summary provided.
2. Adhere to the word limit ({word_limit} words).
3. **Data Usage**: When inserting specific data values, use the <v>...</v> format if available in your context, otherwise use plain text. Do not hallucinate values.

{tagging_instruction}
"""

GENERATE_DISCUSSION_TEMPLATE = """
You are writing the "Discussion" section of a scientific paper.
**Task Goal**: Detailed description and causal analysis of the experimental data.

**Input Data (Row Data)**:
{row_data_description}

**Reference Style**:
"{reference_content}"

**Constraints**:
1. **Target Word Count**: Approx. {word_limit} words.
2. **Structure Requirement** (Strictly follow this 3-phase logic):
   - **Description (30%)**: Detailed description of specific values.
   - **Causal analysis (30%)**: Analyze relationships (e.g., PCE vs variables).
   - **Comparative analysis (40%)**: Compare different groups/rows.
3. **Data Literalism**: You MUST use the provided values exactly as shown in the Input Data. Do not convert to LaTeX or Unicode.
   - Correct: "Li-TBP"
   - Incorrect: "Li$^{{+}}$" or "Li⁺"

{tagging_instruction}
"""

SUMMARIZE_CONTENT_TEMPLATE = """
Please summarize the following scientific paper content into a concise paragraph no more than 500 words.
Focus on the key findings and logical flow established so far.

**Constraints**:
{literalism_ascii_only}
Return plain text only (no XML blocks, no markdown code blocks).

**Content to Summarize**:
"{current_content}"
"""

# --- Prompt Getters ---

def get_chapter_prompt(chapter, word_limit, summary, ref_content, tagging_instr):
    return GENERATE_CHAPTER_TEMPLATE.format(
        chapter_title=chapter,
        word_limit=word_limit,
        generated_summary=summary or "This is the beginning of the paper.",
        reference_content=ref_content, 
        tagging_instruction=tagging_instr
    )

def get_discussion_prompt(row_data, word_limit, ref_content, tagging_instr):
    data_desc = json.dumps(row_data, indent=2, ensure_ascii=False)
    return GENERATE_DISCUSSION_TEMPLATE.format(
        row_data_description=data_desc,
        word_limit=word_limit,
        reference_content=ref_content, 
        tagging_instruction=tagging_instr
    )

def get_summary_prompt(content):
    return SUMMARIZE_CONTENT_TEMPLATE.format(
        current_content=content[:30000],
        literalism_ascii_only=LITERALISM_ASCII_ONLY.strip()
    )

def get_phrase_prompt(text_segment, ref_content):
    return PHRASE_PROMPT_TEMPLATE.format(
        text_segment=text_segment,
        ref_content=ref_content, 
        tagging_instruction=TAGGING_INSTRUCTION
    )

def get_scale_rewrite_prompt(text_segment):
    return SCALE_REWRITE_PROMPT_TEMPLATE.format(
        text_segment=text_segment,
        tagging_instruction=TAGGING_INSTRUCTION
    )

def get_missing_data_desc_prompt(missing_data):
    return MISSING_DATA_DESC_TEMPLATE.format(
        missing_data_json=json.dumps(missing_data, indent=2, ensure_ascii=False),
        literalism_ascii_only=LITERALISM_ASCII_ONLY.strip()
    )