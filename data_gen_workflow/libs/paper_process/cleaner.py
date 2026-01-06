import re
import logging

logger = logging.getLogger(__name__)

class MarkdownCleaner:
    def __init__(self):
        pass

    def clean(self, text: str) -> str:
        """
        Clean Markdown text based on specific rules:
        1. Keep the first Level 1 header with more than 5 words (header line only).
        2. Start from Level 1 header containing "Abstract" or "Introduction",
           save until the Level 1 header containing "Conclusion" and its content.
           If Abstract/Introduction is not found, retain the full text.
        3. Keep Level 1 headers and content containing "Supplement" or "appendices".
        """
        # 1. Split text into sections (Level 1 header as delimiter)
        matches = list(re.finditer(r"(?m)^#\s+(.*)$", text))
        
        # If no headers found, return full text directly
        if not matches:
            logger.info("[Cleaner] No Level 1 headings found. Retaining full text.")
            return text

        sections = []
        for i, m in enumerate(matches):
            header_text = m.group(1).strip()
            start_pos = m.start()
            end_pos = matches[i+1].start() if i + 1 < len(matches) else len(text)
            
            content = text[start_pos:end_pos]
            sections.append({
                "header": header_text,
                "content": content, 
                "index": i
            })

        final_parts = []
        
        # --- Rule 1: Header (First > 5 words) ---
        for sec in sections:
            word_count = len(sec["header"].split())
            if word_count > 5:
                final_parts.append(f"# {sec['header']}\n")
                break
        
        # --- Rule 2: Body (Abstract/Intro -> Conclusion) ---
        start_idx = -1
        end_idx = -1

        # 2.1 Find start point
        for i, sec in enumerate(sections):
            h = sec["header"].lower()
            if "abstract" in h or "introduction" in h:
                start_idx = i
                break
        
        # If start point not found, retain full text (starting from first section)
        if start_idx == -1:
            logger.info("[Cleaner] No 'Abstract' or 'Introduction' found. Retaining all sections.")
            # Treat all sections as body
            for sec in sections:
                final_parts.append(sec["content"])
            return "\n".join(final_parts)

        # 2.2 Find end point (Execute only if start point found)
        for i in range(start_idx, len(sections)):
            h = sections[i]["header"].lower()
            if "conclusion" in h:
                end_idx = i
                break 
        
        if end_idx == -1:
            end_idx = len(sections) - 1

        # 2.3 Add body interval
        for i in range(start_idx, end_idx + 1):
            final_parts.append(sections[i]["content"])

        # --- Rule 3: Supplementary Material (Supplement / Appendices) ---
        for i, sec in enumerate(sections):
            h = sec["header"].lower()
            if "supplement" in h or "appendices" in h:
                # Deduplication check
                is_duplicate = False
                if start_idx != -1 and end_idx != -1:
                    if start_idx <= i <= end_idx:
                        is_duplicate = True
                
                if not is_duplicate:
                    final_parts.append(sec["content"])

        cleaned_text = "\n".join(final_parts)
        return cleaned_text