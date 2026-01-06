import pandas as pd
import json
import uuid
import random
import re
import asyncio
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from .utils import (
    BASE_SYSTEM_PROMPT, TAGGING_INSTRUCTION,
    get_phrase_prompt, get_scale_rewrite_prompt, get_missing_data_desc_prompt,
    get_chapter_prompt, get_discussion_prompt, get_summary_prompt
)

STOP_MATCH_VALUES = {str(i) for i in range(10)} | {"no", "na", "n/a", "ms"}

class AugmentationStrategy:
    def __init__(self, llm, data_manager, text_analyzer):
        self.llm = llm
        self.dm = data_manager
        self.analyzer = text_analyzer
        self.value_sets = self._build_value_sets(data_manager.df)

    def _generate_summary(self, current_full_text: str) -> str:
        if not current_full_text.strip():
            return ""
        prompt = get_summary_prompt(current_full_text)
        resp = self.llm.generate(prompt, system=BASE_SYSTEM_PROMPT, temperature=0.2)
        return self._normalize_unicode_chem(resp.text.strip())
    
    # Async wrapper for summary generation (needed for run_generate_async)
    async def _generate_summary_async(self, current_full_text: str) -> str:
        if not current_full_text.strip():
            return ""
        prompt = get_summary_prompt(current_full_text)
        resp = await self.llm.generate_async(prompt, system=BASE_SYSTEM_PROMPT, temperature=0.2)
        return self._normalize_unicode_chem(resp.text.strip())

    def _sanitize_ref_content(self, text):
        clean = re.sub(r'<.*?>', '', text)
        return clean.strip()

    def _build_value_sets(self, df):
        sets = {}
        exclude_cols = ["Article", "_row_id", "Gen_ID", "_row_hash"]
        for col in df.columns:
            if col in exclude_cols: continue
            try:
                values = df[col].dropna().astype(str).unique()
                valid_vals = [v for v in values if len(v.strip()) >= 1]
                valid_vals = [v for v in valid_vals if v.strip().lower() not in STOP_MATCH_VALUES]
                valid_vals.sort(key=lambda x: len(x), reverse=True)
                if valid_vals:
                    sets[col] = valid_vals
            except Exception:
                pass
        return sets

    # ... (Keep ASCII Post-processing logic unchanged) ...
    _SUB_MAP = str.maketrans({
        "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
        "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
        "₊": "+", "₋": "-",
        "ₓ": "x", "ₐ": "a", "ₑ": "e", "ₕ": "h", "ₖ": "k", "ₗ": "l",
        "ₘ": "m", "ₙ": "n", "ₒ": "o", "ₚ": "p", "ₛ": "s", "ₜ": "t",
    })

    _SUP_TO_ASCII = {
        "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
        "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
        "⁺": "+", "⁻": "-",
    }

    _SUP_BLOCK_RE = re.compile(r"([A-Za-z0-9\)\]\}])([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)")
    _TAG_RE = re.compile(r"<v>\s*(.*?)\s*::\s*(.*?)\s*</v>", re.DOTALL)

    def _normalize_unicode_chem(self, text: str) -> str:
        if not text:
            return text
        text = text.translate(self._SUB_MAP)
        def sup_repl(m: re.Match) -> str:
            prefix = m.group(1)
            block = m.group(2)
            ascii_block = "".join(self._SUP_TO_ASCII.get(ch, "") for ch in block)
            if any(ch.isdigit() for ch in ascii_block):
                return f"{prefix}^{ascii_block}"
            return f"{prefix}{ascii_block}"
        return self._SUP_BLOCK_RE.sub(sup_repl, text)

    def _build_canonical_map(self, group_df: pd.DataFrame) -> Dict[str, str]:
        canonical = {}
        exclude_cols = {"Article", "_row_id", "Gen_ID", "_row_hash"}
        for col in group_df.columns:
            if col in exclude_cols: continue
            vals = group_df[col].dropna().astype(str).tolist()
            vals = [v for v in vals if v.lower() not in ["nan", "none", "-", ""]]
            if not vals: continue
            uniq = []
            seen = set()
            for v in vals:
                if v not in seen:
                    uniq.append(v)
                    seen.add(v)
            canonical[col] = ", ".join(uniq)
        return canonical

    def _enforce_tag_values(self, text: str, canonical: Dict[str, str], strict_fill: bool = True) -> str:
        if not text: return text
        def repl(m: re.Match) -> str:
            col = (m.group(1) or "").strip()
            val = (m.group(2) or "").strip()
            if strict_fill and col in canonical:
                return f"<v>{col}::{canonical[col]}</v>"
            return f"<v>{col}::{self._normalize_unicode_chem(val)}</v>"
        return self._TAG_RE.sub(repl, text)

    def _postprocess_ascii_literalism(self, text: str, group_df: pd.DataFrame, strict_fill: bool = True) -> str:
        if not text: return text
        canonical = self._build_canonical_map(group_df)
        text = self._normalize_unicode_chem(text)
        text = self._enforce_tag_values(text, canonical, strict_fill=strict_fill)
        return text

    # ... (Keep Helper Functions unchanged) ...
    def _normalize_text_with_mapping(self, text: str) -> Tuple[str, List[int]]:
        normalized = []
        mapping = []
        i = 0
        n = len(text)
        while i < n:
            char = text[i]
            if char == '\\':
                i += 1
                while i < n and text[i].isalpha():
                    i += 1
                continue
            if char in ['{', '}', '_', '^', '$', ' ']:
                i += 1
                continue
            normalized.append(char)
            mapping.append(i)
            i += 1
        return "".join(normalized), mapping

    def _is_boundary_valid(self, full_text: str, start: int, end: int, keyword: str = "") -> bool:
        prev_char = full_text[start - 1] if start > 0 else " "
        next_char = full_text[end] if end < len(full_text) else " "
        if prev_char.isalnum(): return False
        if next_char.isalnum(): return False
        kw = (keyword or "").strip()
        if kw.isdigit() and len(kw) == 1:
            if prev_char.isdigit() or next_char.isdigit() or prev_char == "." or next_char == ".":
                return False
        return True

    def _find_matches_robust(self, full_text: str, keyword: str) -> List[Tuple[int, int]]:
        norm_text, mapping = self._normalize_text_with_mapping(full_text)
        norm_keyword, _ = self._normalize_text_with_mapping(keyword)
        if not norm_keyword: return []
        matches = []
        start_search = 0
        while True:
            idx = norm_text.find(norm_keyword, start_search)
            if idx == -1: break
            norm_end = idx + len(norm_keyword)
            if idx < len(mapping) and (norm_end - 1) < len(mapping):
                orig_start = mapping[idx]
                orig_end_char_idx = mapping[norm_end - 1]
                orig_end = orig_end_char_idx + 1
                if self._is_boundary_valid(full_text, orig_start, orig_end, keyword):
                    matches.append((orig_start, orig_end))
            start_search = norm_end
        return matches

    # Helper to generate missing data description async
    async def _generate_missing_data_desc_async(self, missing_data: Dict) -> str:
        if not missing_data: return ""
        print(f"      [Data Supplement] Generating description for {len(missing_data)} missing cols...")
        desc_prompt = get_missing_data_desc_prompt(missing_data)
        desc_resp_obj = await self.llm.generate_async(desc_prompt, system=BASE_SYSTEM_PROMPT, temperature=0.2)
        desc_resp = desc_resp_obj.text
        desc_clean = self._normalize_unicode_chem(desc_resp.strip())
        return f"\n\n[Additional Context Parameters]: {desc_clean}"
    
    # Logic to inject data values and describe missing data via LLM
    async def _inject_data_and_get_missing_async(self, text: str, group_df: pd.DataFrame) -> str:
        all_replacements = []
        replaced_cols = set()
        
        # Phase 1: Search & Replace (CPU Bound)
        for col, known_values in self.value_sets.items():
            current_vals = group_df[col].dropna().astype(str).unique().tolist()
            current_vals = [v for v in current_vals if v.lower() not in ["nan", "none", "-", ""]]
            if not current_vals: continue
            val_str = ", ".join(current_vals)
            target_str = f"<v>{col}::{val_str}</v>"
            for old_val in known_values:
                matches = self._find_matches_robust(text, old_val)
                for start, end in matches:
                    is_overlap = False
                    for item in all_replacements:
                        r_start, r_end = item[0], item[1]
                        if max(start, r_start) < min(end, r_end):
                            is_overlap = True
                            break
                    if not is_overlap:
                        all_replacements.append((start, end, target_str))
                        replaced_cols.add(col)
        all_replacements.sort(key=lambda x: x[0], reverse=True)
        new_text_list = list(text)
        for start, end, target_str in all_replacements:
            new_text_list[start:end] = list(target_str)
        final_text = "".join(new_text_list)

        # Phase 2: Missing Data (May involve LLM)
        missing_data = {}
        exclude_sys_cols = ["Article", "_row_id", "Gen_ID", "_row_hash"]
        for col in group_df.columns:
            if col in exclude_sys_cols or col in replaced_cols: continue
            vals = group_df[col].dropna().astype(str).unique().tolist()
            vals = [v for v in vals if v.lower() not in ["nan", "none", "-", ""]]
            if vals: missing_data[col] = vals
        
        if missing_data:
            suffix = await self._generate_missing_data_desc_async(missing_data)
            final_text += suffix

        return final_text

    def _extract_xml_blocks(self, full_text: str) -> Tuple[str, str, str]:
        think = ""
        paper = ""
        other = ""
        m_think = re.search(r'<think>(.*?)(?:</think>|$)', full_text, re.DOTALL)
        if m_think: think = m_think.group(1).strip()
        m_paper = re.search(r'<paper>(.*?)</paper>', full_text, re.DOTALL)
        if m_paper:
            paper = m_paper.group(1).strip()
        else:
            m_paper_open = re.search(r'<paper>(.*)$', full_text, re.DOTALL)
            if m_paper_open:
                paper = m_paper_open.group(1).strip()
            else:
                clean = re.sub(r'<think>.*?</think>', '', full_text, flags=re.DOTALL)
                clean = re.sub(r'<other>.*?</other>', '', clean, flags=re.DOTALL)
                clean = clean.replace('<paper>', '').replace('</paper>', '')
                paper = clean.strip()
        m_other = re.search(r'<other>(.*?)(?:</other>|$)', full_text, re.DOTALL)
        if m_other: other = m_other.group(1).strip()
        return think, paper, other

    # ==================== Async Strategies ====================
    
    async def run_phrase_async(self, article_name, group_df, num_copies, semaphore: asyncio.Semaphore):
        # Batch generation tasks
        tasks = []
        for i in range(num_copies):
            tasks.append(self._run_single_phrase_task(article_name, group_df, semaphore))
        
        # Concurrent execution and result collection
        results = await asyncio.gather(*tasks)
        
        # Filter None
        return [r for r in results if r]

    async def _run_single_phrase_task(self, article_name, group_df, semaphore):
        # 1. Preparation (CPU-bound)
        matched_papers = self.dm.load_matched_papers_content(article_name)
        if not matched_papers: return None
        
        main_paper_name, main_paper_content = random.choice(matched_papers)
        other_discussions = []
        for name, content in matched_papers:
            if name == main_paper_name: continue
            sects = self.dm.parse_sections(content)
            disc = self.dm.fuzzy_get_section(sects, ["Discussion"])
            if disc: other_discussions.append(disc)
        ref_context = "\n\n".join(other_discussions)
        sections = self.dm.parse_sections(main_paper_content)
        
        new_id = f"Phrase_{article_name}_{uuid.uuid4().hex[:6]}"
        new_df = group_df.copy()
        new_df['Article'] = new_id
        
        # 2. Parallel Section Rewrite
        section_tasks = []
        section_titles = list(sections.keys())
        
        for title in section_titles:
            content = sections[title]
            section_tasks.append(self._rewrite_section_async(title, content, new_df, ref_context, semaphore))
            
        # Wait for all sections to complete
        processed_sections = await asyncio.gather(*section_tasks)
        
        # 3. Assemble Results
        final_sections_content = []
        full_think_log = []
        
        for idx, (p_content, p_think) in enumerate(processed_sections):
            title = section_titles[idx]
            if not p_content: p_content = sections[title] # Fallback
            full_think_log.append(f"--- Section: {title} ---\n{p_think}")
            
            if title == "Content":
                final_sections_content.append(p_content)
            else:
                final_sections_content.append(f"# {title}\n\n{p_content}")
                
        full_text = "\n\n".join(final_sections_content)
        full_text = self._postprocess_ascii_literalism(full_text, new_df, strict_fill=True)
        combined_think = "\n\n".join(full_think_log)

        final_title = f"{new_id}_Rephrased"
        clean_text, word_count, tags, pos_df = self.analyzer.process_generated_paper(full_text, new_df, final_title)
        
        return {
            "title": final_title, "text": clean_text, "data": pos_df, 
            "tags": tags, "word_count": word_count, "think": combined_think
        }

    async def _rewrite_section_async(self, title, content, new_df, ref_context, semaphore):
        # Inject data (Async call because of potential missing data description LLM call)
        # Note: ref_context is same for all sections
        injected_text = await self._inject_data_and_get_missing_async(content, new_df)
        user_prompt = get_phrase_prompt(injected_text, ref_context)
        
        async with semaphore:
            raw_response_obj = await self.llm.generate_async(user_prompt, system=BASE_SYSTEM_PROMPT, temperature=0.4)
            raw_response = raw_response_obj.text
            
        think, paper_content, _ = self._extract_xml_blocks(raw_response)
        return paper_content, think

    async def run_scale_async(self, article_name, group_df, min_words, max_words, semaphore: asyncio.Semaphore):
        # 1. Preparation
        matched_papers = self.dm.load_matched_papers_content(article_name)
        if not matched_papers: return []
        
        base_name, base_content = matched_papers[0]
        base_sections = self.dm.parse_sections(base_content)
        base_headers = list(base_sections.keys())

        for i in range(1, len(matched_papers)):
            _, other_content = matched_papers[i]
            other_sections = self.dm.parse_sections(other_content)
            for h in other_sections.keys():
                is_exist = any(h.lower() in bh.lower() or bh.lower() in h.lower() for bh in base_headers)
                if not is_exist:
                    base_headers.append(h)
                    base_sections[h] = other_sections[h]

        row_data = group_df.to_dict(orient='records')
        new_id = f"{article_name}_Scale_{uuid.uuid4().hex[:4]}"
        new_df = group_df.copy()
        new_df['Article'] = new_id

        # 2. Task Planning
        final_structure_map = {} # header -> task OR static_content
        ref_disc = ""
        
        tasks_map = {} # key -> coroutine
        
        for h in base_headers:
            h_lower = h.lower()
            original_text = base_sections.get(h, "")
            if "reference" in h_lower or "acknowledge" in h_lower: continue
            
            if "discussion" in h_lower:
                ref_disc += original_text + "\n"
                final_structure_map[h] = "__GEN_DISC__"
            else:
                # Prepare rewrite task
                tasks_map[h] = self._scale_rewrite_section_async(original_text, new_df, semaphore)
                final_structure_map[h] = "__TASK__"

        has_disc_placeholder = any(v == "__GEN_DISC__" for v in final_structure_map.values())
        if not has_disc_placeholder:
            final_structure_map["Discussion"] = "__GEN_DISC__"
        
        # 3. Concurrent Execution: Generate Discussion + Rewrite Sections
        # Discussion Task
        disc_limit = int(0.3 * max_words)
        disc_prompt = get_discussion_prompt(row_data, disc_limit, self._sanitize_ref_content(ref_disc), TAGGING_INSTRUCTION)
        
        async def gen_disc_task():
            async with semaphore:
                raw = await self.llm.generate_async(disc_prompt, system=BASE_SYSTEM_PROMPT)
                _, txt, _ = self._extract_xml_blocks(raw.text)
                return txt

        # Aggregate all tasks
        all_futures = {}
        all_futures["Discussion"] = gen_disc_task()
        for h, coro in tasks_map.items():
            all_futures[h] = coro
            
        # Concurrent wait
        keys = list(all_futures.keys())
        coros = [all_futures[k] for k in keys]
        results = await asyncio.gather(*coros)
        results_map = dict(zip(keys, results))
        
        # 4. Backfill Content
        full_text_str = f"# {new_id}\n\n"
        # Preserve original order
        ordered_keys = [k for k in base_headers if k in final_structure_map]
        if "Discussion" in final_structure_map and "Discussion" not in ordered_keys:
            ordered_keys.append("Discussion")

        for h in ordered_keys:
            if h == "Discussion":
                content = results_map["Discussion"]
            elif h in tasks_map:
                content = results_map[h]
            else:
                continue # Ref/Ack
            
            if not content: content = "Generation Failed"
            full_text_str += f"# {h}\n{content}\n\n"

        full_text_str = self._postprocess_ascii_literalism(full_text_str, new_df, strict_fill=True)
        final_title = f"{new_id}_Scaled"
        clean_text, word_count, tags, pos_df = self.analyzer.process_generated_paper(full_text_str, new_df, final_title)
        
        return [{
            "title": final_title, "text": clean_text, "data": pos_df, 
            "tags": tags, "word_count": word_count, "think": ""
        }]

    async def _scale_rewrite_section_async(self, old_text, new_df, semaphore):
        injected = await self._inject_data_and_get_missing_async(old_text, new_df)
        rewrite_prompt = get_scale_rewrite_prompt(injected)
        async with semaphore:
            raw = await self.llm.generate_async(rewrite_prompt, system=BASE_SYSTEM_PROMPT)
        _, txt, _ = self._extract_xml_blocks(raw.text)
        return txt if txt else old_text

    async def run_generate_async(self, article_name, group_df, ref_num, min_words, max_words, semaphore: asyncio.Semaphore):
        # Chapters have dependencies (Summary iteration), so internal sections are serial.
        # But the whole run_generate_async can be called concurrently from outside.
        
        results = []
        ref_contents = self.dm.load_random_papers(ref_num)
        if not ref_contents: return []
        
        # (Structure setup logic same as sync)
        header_counts = {}
        all_parsed = [self.dm.parse_sections(c) for c in ref_contents]
        for p in all_parsed:
            for h in p.keys(): header_counts[h] = header_counts.get(h, 0) + 1
        
        base_order = list(all_parsed[0].keys())
        article_structure = []
        for h in base_order:
            if header_counts.get(h, 0) >= max(1, len(ref_contents) // 2):
                article_structure.append(h)

        if not any("abstract" in h.lower() for h in article_structure): article_structure.insert(0, "Abstract")
        if not any("discussion" in h.lower() for h in article_structure): article_structure.append("Discussion")
        if not any("conclusion" in h.lower() for h in article_structure): article_structure.append("Conclusion")
        
        row_data = group_df.to_dict(orient='records')
        ref_disc_text = ""
        for p in all_parsed:
            d = self.dm.fuzzy_get_section(p, ["Discussion"])
            if d: ref_disc_text += d + "\n"

        disc_word_limit = int(0.35 * max_words)
        disc_prompt = get_discussion_prompt(row_data, disc_word_limit, self._sanitize_ref_content(ref_disc_text), TAGGING_INSTRUCTION)
        
        async with semaphore:
            raw_disc = await self.llm.generate_async(disc_prompt, system=BASE_SYSTEM_PROMPT)
        _, generated_discussion, _ = self._extract_xml_blocks(raw_disc.text)
        if not generated_discussion: generated_discussion = "Failed."

        generated_sections = {}
        generated_summary = ""
        
        # Serial Generation Loop
        for chapter in article_structure:
            if "discussion" in chapter.lower():
                generated_sections[chapter] = generated_discussion
                generated_summary = await self._generate_summary_async(generated_summary + "\n" + generated_discussion)
                continue

            if "abstract" in chapter.lower(): limit = 300
            elif "conclusion" in chapter.lower(): limit = 500
            else:
                remaining = (0.65 * max_words) - 800
                count = max(1, len(article_structure) - 3)
                limit = max(int(remaining / count), min_words)

            ref_chapter_text = ""
            for p in all_parsed:
                c = self.dm.fuzzy_get_section(p, [chapter])
                if c: ref_chapter_text += c + "\n"
            
            chap_prompt = get_chapter_prompt(chapter, limit, generated_summary, self._sanitize_ref_content(ref_chapter_text), TAGGING_INSTRUCTION)
            
            async with semaphore:
                raw_chap = await self.llm.generate_async(chap_prompt, system=BASE_SYSTEM_PROMPT)
            _, chap_text, _ = self._extract_xml_blocks(raw_chap.text)
            if not chap_text: chap_text = raw_chap.text

            generated_sections[chapter] = chap_text
            new_full = "\n".join(generated_sections.values())
            generated_summary = await self._generate_summary_async(new_full)

        new_id = f"Gen_{article_name}_{uuid.uuid4().hex[:6]}"
        new_df = group_df.copy()
        new_df['Article'] = new_id
        
        full_paper_str = f"# {new_id}\n\n"
        for h in article_structure:
            full_paper_str += f"# {h}\n{generated_sections.get(h, '')}\n\n"

        full_paper_str = self._postprocess_ascii_literalism(full_paper_str, new_df, strict_fill=True)
        final_title = f"{new_id}_Generated"
        clean_text, word_count, tags, pos_df = self.analyzer.process_generated_paper(full_paper_str, new_df, final_title)
            
        results.append({
            "title": final_title, "text": clean_text, "data": pos_df, 
            "tags": tags, "word_count": word_count, "think": ""
        })
        return results