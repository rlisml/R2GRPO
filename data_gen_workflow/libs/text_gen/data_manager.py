import pandas as pd
import json
import os
import re
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple

class DataManager:
    def __init__(self, excel_path: str, md_dir: str):
        self.excel_path = excel_path
        self.md_dir = Path(md_dir)
        self.df = self._load_data()

    def _load_data(self):
        if self.excel_path.endswith('.xlsx'):
            return pd.read_excel(self.excel_path)
        elif self.excel_path.endswith('.csv'):
            return pd.read_csv(self.excel_path)
        else:
            raise ValueError("Unsupported file format")

    def get_grouped_data(self):
        # Group data by the 'Article' column
        if 'Article' not in self.df.columns:
            raise ValueError("Column 'Article' not found in data.")
        return {name: group for name, group in self.df.groupby('Article')}

    def _fuzzy_match_paper_files(self, article_id: str) -> List[Path]:
        """
        Fuzzy match files based on Article ID (e.g., Gen_1_PartA_PartB).
        Rule: Split by '_', and match substrings with length > 5 against filenames.
        """
        parts = article_id.split('_')
        valid_parts = [p for p in parts if len(p) > 5]
        
        if not valid_parts:
            # If no long segments exist, try matching exactly
            valid_parts = [article_id]

        matched_files = set()
        all_md_files = list(self.md_dir.glob("*.md"))
        
        for part in valid_parts:
            for f in all_md_files:
                if part.lower() in f.name.lower():
                    matched_files.add(f)
        
        return list(matched_files)

    def load_matched_papers_content(self, article_id: str) -> List[Tuple[str, str]]:
        """
        Returns a list of [(filename, content), ...]
        """
        files = self._fuzzy_match_paper_files(article_id)
        results = []
        for f in files:
            try:
                txt = f.read_text(encoding='utf-8', errors='ignore')
                results.append((f.name, txt))
            except:
                pass
        return results

    def load_random_papers(self, n: int) -> List[str]:
        all_files = list(self.md_dir.glob("*.md"))
        if not all_files:
            return []
        selected = random.sample(all_files, min(n, len(all_files)))
        return [f.read_text(encoding='utf-8', errors='ignore') for f in selected]

    # ==========================================
    # Section Extraction & Manipulation
    # ==========================================
    
    def parse_sections(self, content: str) -> Dict[str, str]:
        """
        Robustly parse Markdown level 1 headers using regex.
        """
        sections = {}
        # Regex explanation:
        # (?m) : Multiline mode, ^ and $ match each line
        # ^#\s+ : Line starts with # followed by at least one whitespace
        # (.*)$ : Capture title content until end of line
        matches = list(re.finditer(r"(?m)^#\s+(.*)$", content))
        
        if not matches:
            return {"Content": content} # Treat as full content if no headers found

        # Iterate through matches
        for i, m in enumerate(matches):
            title = m.group(1).strip()
            start_pos = m.end() # Position where title line ends
            
            # Content ends at the start of the next header or EOF
            if i + 1 < len(matches):
                end_pos = matches[i+1].start()
            else:
                end_pos = len(content)
            
            section_content = content[start_pos:end_pos].strip()
            sections[title] = section_content
            
        return sections

    def fuzzy_get_section(self, sections: Dict[str, str], keywords: List[str]) -> Optional[str]:
        """
        Find section content where the header contains any of the keywords.
        """
        for title, content in sections.items():
            for kw in keywords:
                if kw.lower() in title.lower():
                    return content
        return None

    def fuzzy_get_section_title(self, sections: Dict[str, str], keywords: List[str]) -> Optional[str]:
        """
        Return the matching header key.
        """
        for title in sections.keys():
            for kw in keywords:
                if kw.lower() in title.lower():
                    return title
        return None

    def construct_architecture(self, sections: Dict[str, str]) -> Dict[str, str]:
        """
        Return an ordered dictionary representing structure.
        """
        return sections.copy()

    def count_formulas(self, text: str) -> int:
        # Simple count of LaTeX markers ($$ or $)
        return text.count('$$') + text.count('$ ')