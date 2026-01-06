import re
import pandas as pd

class TextAnalyzer:
    def process_generated_paper(self, text, row_data_df, title):
        """
        Process LLM-generated tagged text:
        1. Pre-cleaning: Remove structural noise like <think>, <paper>, <other>.
        2. Remove <v>...</v> tags but preserve the values.
        3. Calculate word position ranges (Start-End) for the preserved values in the Clean Text.
        4. Return Clean Text, Word Count, Tags (List), Position Row (DataFrame).
        """
        
        # === Step 0: Sanitization (Deep Cleaning) ===
        # Must perform this before index calculation to avoid offsets
        
        # 1. Remove paired blocks (greedy match)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'<other>.*?</other>', '', text, flags=re.DOTALL)
        
        # 2. Remove residual isolated tags
        text = re.sub(r'</?(think|paper|other)>', '', text)
        
        # === Step 1: Split & Analyze Indices ===
        
        # Use loose regex to split, allowing spaces around ::
        parts = re.split(r'(<v>.*?</v>)', text, flags=re.DOTALL)
        
        clean_tokens = []
        position_records = {} # {Column: [list of "start-end"]}
        
        current_word_idx = 0
        all_tags = []
        
        # Pre-compile loose regex for <v>Col :: Val</v>
        tag_pattern = re.compile(r'<v>\s*(.*?)\s*::\s*(.*?)\s*</v>', re.DOTALL)
        
        for part in parts:
            if not part: continue
            
            # Check if it is a tag
            tag_match = tag_pattern.match(part)
            
            if tag_match:
                col_name = tag_match.group(1).strip()
                val_content = tag_match.group(2).strip()
                
                # Record Tag Info
                all_tags.append({"col": col_name, "val": val_content})
                
                # Analyze word count of the Value content
                val_tokens = val_content.split()
                if not val_tokens:
                    continue
                    
                start_idx = current_word_idx
                end_idx = start_idx + len(val_tokens) - 1
                
                # Record "Start-End" position
                rec_str = f"{start_idx}-{end_idx}"
                if col_name not in position_records:
                    position_records[col_name] = []
                position_records[col_name].append(rec_str)
                
                clean_tokens.extend(val_tokens)
                current_word_idx += len(val_tokens)
                
            else:
                # Normal Text / Sanitize invalid tags
                part = re.sub(r'</?v\b[^>]*>', '', part)
                sub_tokens = part.split()
                current_word_idx += len(sub_tokens)
                
        # === Step 2: Generate Clean Text ===
        # Use the same regex for replacement
        def replace_func(m):
            return m.group(2).strip() # group(2) is value
            
        clean_text = tag_pattern.sub(replace_func, text)

        # Fallback: remove any residual <v> / </v> tags
        clean_text = re.sub(r'</?v\b[^>]*>', '', clean_text)
        
        # Calculate total word count
        final_word_count = len(clean_text.split())
        
        # === Step 3: Generate Position Table ===
        pos_df = row_data_df.copy()
        for col in pos_df.columns:
            locs = position_records.get(col, [])
            if locs:
                pos_df[col] = "; ".join(locs)
            else:
                pos_df[col] = "-"
                
        if 'Article' in pos_df.columns:
            pos_df['Article'] = title

        return clean_text, final_word_count, all_tags, pos_df