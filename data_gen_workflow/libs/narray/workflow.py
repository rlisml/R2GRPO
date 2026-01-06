from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import random
import pandas as pd
import yaml
import re
import os
import asyncio

from .distribution import load_table, build_distribution_stats, calculate_distribution_score, analyze_head_tail_values
from .prompts import PromptStore
from .strict_parsing import extract_gap_tables_json, parse_strategy_json, parse_jsonl_strict

try:
    from paper_process.converter import DocumentConverter
    from paper_process.cleaner import MarkdownCleaner
except ImportError:
    DocumentConverter = None
    MarkdownCleaner = None

@dataclass
class IterativeConfig:
    max_epoch: int = 5
    article_col: str = "Article"
    meta_prefix: str = "_"
    missing_token: str = "-"
    random_seed: int = 7
    numeric_cols: Optional[List[str]] = None
    numeric_bin_rules: Optional[Dict[str, float]] = None
    target_multirow_ratio: Optional[Dict[str, float]] = None
    
    # Structure: { "head_ratio": 0.5, "tail_ratio_dict": {"col": 0.2, ...} }
    target_entity_constraints: Dict[str, Any] = field(default_factory=dict)
    
    max_workers: int = 5 

    max_total_new_rows: int = 300

    path2knowledge: Optional[str] = None       
    domain_knowledge_path: Optional[str] = None 
    mineru_config: Dict[str, Any] = field(default_factory=dict) 

class IterativeGenerator:
    def __init__(self, config: IterativeConfig, llm: Any, prompts: PromptStore):
        self.config = config
        self.llm = llm
        self.prompts = prompts
        self.rng = random.Random(config.random_seed)
        self._think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)

    def _write_json(self, path: Path, obj: Any):
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    def _extract_json_content(self, text: str) -> str:
        pattern = r"<json>(.*?)</json>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text
        
    def _extract_think_and_clean(self, text: str) -> Tuple[str, str]:
        """
        Robust extraction logic to handle cases where output is truncated 
        and the closing </think> tag is missing.
        """
        think_content = ""
        cleaned_text = text.strip()

        start_tag = "<think>"
        end_tag = "</think>"
        
        s_idx = text.find(start_tag)
        if s_idx != -1:
            e_idx = text.find(end_tag)
            if e_idx != -1:
                # Normal case: closed tag found
                think_content = text[s_idx + len(start_tag) : e_idx].strip()
                # Remove the think block, keep parts before and after
                cleaned_text = (text[:s_idx] + text[e_idx + len(end_tag):]).strip()
            else:
                # Truncated case: start tag exists but no end tag
                # Assume all subsequent content is think content (or truncated data)
                think_content = text[s_idx + len(start_tag):].strip()
                cleaned_text = text[:s_idx].strip()
        
        return cleaned_text, think_content

    def _smart_source_selection(self, df_all: pd.DataFrame, targeted_narray: List[Dict], global_seq_start: int) -> Tuple[List[Dict], int]:
        k_map = {} 
        usage_counts = {} 
        counts = df_all[self.config.article_col].value_counts()
        for article_id, k in counts.items():
            k = int(k)
            if k not in k_map: k_map[k] = []
            k_map[k].append(article_id)
            usage_counts[article_id] = 0
        selection_plan = []
        current_global_seq = global_seq_start
        for target in targeted_narray:
            try:
                t_k = int(target.get("target_k", 0))
                count = int(target.get("count", 0))
            except: continue
            if t_k <= 0 or count <= 0: continue
            for _ in range(count):
                current_global_seq += 1
                components = [] 
                current_sum = 0
                while current_sum < t_k:
                    needed = t_k - current_sum
                    candidate_k = None
                    for try_k in range(needed, 0, -1):
                        if try_k in k_map and len(k_map[try_k]) > 0:
                            candidate_k = try_k
                            break
                    if not candidate_k: break 
                    candidates = k_map[candidate_k]
                    candidates.sort(key=lambda x: usage_counts[x])
                    best_id = candidates[0]
                    usage_counts[best_id] += 1
                    components.append(best_id)
                    current_sum += candidate_k
                if current_sum == t_k:
                    suffix_parts = []
                    for uid in components:
                        s_uid = str(uid)
                        part = s_uid[-25:] if len(s_uid) >= 25 else s_uid
                        suffix_parts.append(part)
                    suffix = "_".join(suffix_parts)
                    new_id = f"Gen_{current_global_seq}_{suffix}"
                    selection_plan.append({
                        "target_k": t_k, "source_articles": components, "new_article_id": new_id, "rationale": target.get("rationale", "")
                    })
        return selection_plan, current_global_seq

    def _resolve_domain_norms(self, out_dir: Path, schema_examples_json: str) -> str:
        if self.config.domain_knowledge_path:
            p = Path(self.config.domain_knowledge_path)
            if p.exists() and p.is_file():
                print(f"  [Knowledge] Loading existing domain norms from: {p}")
                return p.read_text(encoding="utf-8").strip()
        if self.config.path2knowledge and Path(self.config.path2knowledge).exists():
            print(f"  [Knowledge] Processing source documents in: {self.config.path2knowledge}")
            if DocumentConverter is None:
                print("  [Error] paper_process libs not found.")
                return ""
            try:
                converter = DocumentConverter(self.config.mineru_config)
                cleaner = MarkdownCleaner()
            except Exception as e:
                print(f"  [Error] Init MinerU failed: {e}")
                return ""
            src_dir = Path(self.config.path2knowledge)
            all_files = list(src_dir.glob("*.*"))
            file_groups = {}
            for f in all_files:
                stem = f.stem
                if stem not in file_groups: file_groups[stem] = []
                file_groups[stem].append(f)
            combined_context = []
            for stem, group_files in file_groups.items():
                md_file = next((f for f in group_files if f.suffix.lower() == '.md'), None)
                content = ""
                file_to_process = None
                if md_file:
                    file_to_process = md_file
                    content = md_file.read_text(encoding='utf-8', errors='ignore')
                else:
                    raw_file = next((f for f in group_files if f.suffix.lower() in ['.pdf', '.docx', '.doc']), None)
                    if raw_file:
                        file_to_process = raw_file
                        md_path = converter.convert(str(raw_file))
                        if md_path and Path(md_path).exists():
                            content = Path(md_path).read_text(encoding='utf-8')
                if content and file_to_process:
                    cleaned_text = cleaner.clean(content)
                    combined_context.append(f"\n--- Document: {stem} ---\n{cleaned_text}")
            if not combined_context: return ""
            full_knowledge_text = "\n".join(combined_context)
            print("  [Knowledge] Generating Domain Norms via LLM...")
            norm_prompt = self.prompts.render("01_norms.md", {
                "path2knowledge": full_knowledge_text, "seed_schema_and_examples": schema_examples_json
            })
            generated_norms = self.llm.generate(norm_prompt).text.strip()
            norms_clean, _ = self._extract_think_and_clean(generated_norms)
            (out_dir / "domain_norms.md").write_text(norms_clean, encoding="utf-8")
            return norms_clean
        return ""

    async def _process_single_plan_async(self, plan: Dict, df_all: pd.DataFrame, domain_norms_txt: str, val_freq: Dict, system_prompt: str):
        source_ids = plan["source_articles"]
        source_df = df_all[df_all[self.config.article_col].isin(source_ids)].copy()
        source_rows = source_df.to_dict(orient="records")
        gen_prompt = self.prompts.render("30_generate_jsonl.md", {
            "domain_norms": domain_norms_txt,
            "head_tail_values_dict": json.dumps(val_freq, ensure_ascii=False),
            "tail_values_json": "See head_tail_values_dict structure",
            "source_rows_json": json.dumps(source_rows, ensure_ascii=False),
            "target_new_id": plan["new_article_id"]
        })
        
        # Call asynchronous generation interface
        if hasattr(self.llm, "generate_async"):
            gen_resp_raw_obj = await self.llm.generate_async(gen_prompt, system=system_prompt, temperature=0.7)
            gen_resp_raw = gen_resp_raw_obj.text
        else:
             gen_resp_raw_obj = await asyncio.to_thread(self.llm.generate, gen_prompt, system=system_prompt, temperature=0.7)
             gen_resp_raw = gen_resp_raw_obj.text

        gen_clean, gen_think = self._extract_think_and_clean(gen_resp_raw)
        return {"plan_id": plan["new_article_id"], "raw": gen_resp_raw, "clean": gen_clean, "think": gen_think}

    async def _run_batch_generation(self, selection_plans, df_all, domain_norms_txt, val_freq, system_prompt):
        tasks = []
        # Use Semaphore to control concurrency
        sem = asyncio.Semaphore(self.config.max_workers)
        
        async def sem_task(plan):
            async with sem:
                return await self._process_single_plan_async(plan, df_all, domain_norms_txt, val_freq, system_prompt)

        for plan in selection_plans:
            tasks.append(sem_task(plan))
        
        return await asyncio.gather(*tasks)

    def _process_single_plan(self, plan: Dict, df_all: pd.DataFrame, domain_norms_txt: str, val_freq: Dict, system_prompt: str):
        # Wraps the async logic for synchronous compatibility
        return asyncio.run(self._process_single_plan_async(plan, df_all, domain_norms_txt, val_freq, system_prompt))

    def run(self, *, seed_path: str, out_dir: str, path2knowledge: Optional[str] = None) -> Dict[str, Any]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        df_all = load_table(seed_path)
        schema_cols = list(df_all.columns)
        (out / "schema_cols.json").write_text(json.dumps(schema_cols, ensure_ascii=False), encoding="utf-8")

        schema_examples = {c: df_all[c].astype(str).dropna().head(5).tolist() for c in df_all.columns}
        schema_examples_json = json.dumps(schema_examples, ensure_ascii=False)
        
        domain_norms_txt = self._resolve_domain_norms(out, schema_examples_json)

        system_prompt = self.prompts.render("00_task.md", {
            "schema_list": json.dumps(schema_cols, ensure_ascii=False),
            "target_multirow_ratio": json.dumps(self.config.target_multirow_ratio or {}),
            "domain_norms_content": domain_norms_txt,
        })

        best_score = float('inf')
        global_gen_counter = 0 

        for epoch in range(1, self.config.max_epoch + 1):
            print(f"\n>>> Epoch {epoch}/{self.config.max_epoch}")
            
            stats = build_distribution_stats(
                df_all,
                article_col=self.config.article_col,
                numeric_cols=self.config.numeric_cols or [],
                numeric_bin_rules=self.config.numeric_bin_rules or {},
                target_entity_constraints=self.config.target_entity_constraints
            )
            self._write_json(out / f"epoch_{epoch:02d}_stats.json", stats)
            
            val_freq = stats.get("value_frequency_analysis", {})
            
            # Calculate Score
            current_score = calculate_distribution_score(
                stats, 
                self.config.target_multirow_ratio or {},
                self.config.target_entity_constraints 
            )
            print(f"  > Current Score: {current_score:.4f} (Best: {best_score:.4f})")
            if current_score < best_score:
                best_score = current_score
                df_all.to_excel(out / "best.xlsx", index=False)
                self._write_json(out / "best_distribution.json", stats)

            # Analyze
            analyze_prompt = self.prompts.render("10_analyze.md", {
                "epoch": epoch,
                "distribution_stats_json": json.dumps(stats, ensure_ascii=False),
                "target_multirow_ratio": json.dumps(self.config.target_multirow_ratio),
                "pending_soft_issues": "[]", 
                "prev_forced_issues": "[]",
                "strategy_i_minus_1": "{}", 
                "numeric_bin_rules": "{}"
            })
            analyze_text_raw = self.llm.generate(analyze_prompt, system=system_prompt).text
            analyze_clean, analyze_think = self._extract_think_and_clean(analyze_text_raw)
            
            if analyze_think:
                with open(out / f"epoch_{epoch:02d}_thoughts.log", 'a', encoding='utf-8') as f_think:
                     f_think.write(f"\n=== Analysis Phase ===\n{analyze_think}\n")
            
            analyze_text = self._extract_json_content(analyze_clean)
            (out / f"epoch_{epoch:02d}_analysis.md").write_text(analyze_clean, encoding="utf-8")
            
            stop_epoch = False
            try:
                if "{" in analyze_text: 
                    analysis_json = json.loads(analyze_text)
                else:
                    analysis_json = extract_gap_tables_json(analyze_clean)

                if analysis_json.get("stop_signal") is True:
                    print(f"  [Stop] LLM suggests stopping: {analysis_json.get('stop_reason')}")
                    stop_epoch = True
            except Exception:
                pass
            if stop_epoch: break

            # Strategy
            strat_prompt = self.prompts.render("20_strategy.md", {
                "epoch": epoch,
                "epoch_i_distribution": analyze_clean,
                "target_multirow_ratio": json.dumps(self.config.target_multirow_ratio),
                "max_total_new_rows": self.config.max_total_new_rows,
                "domain_norms": domain_norms_txt,
            })
            strat_text_raw = self.llm.generate(strat_prompt, system=system_prompt).text
            strat_clean, strat_think = self._extract_think_and_clean(strat_text_raw)
            if strat_think:
                with open(out / f"epoch_{epoch:02d}_thoughts.log", 'a', encoding='utf-8') as f_think:
                     f_think.write(f"\n=== Strategy Phase ===\n{strat_think}\n")

            strat_text = self._extract_json_content(strat_clean)
            (out / f"epoch_{epoch:02d}_strategy.json").write_text(strat_text, encoding="utf-8")
            
            try:
                strategy_obj = json.loads(strat_text)
            except:
                try:
                    strategy_obj = parse_strategy_json(strat_clean)
                except:
                    print("  [Fatal] Strategy parse failed.")
                    continue
            
            # Generate (Parallel)
            targeted_narray = strategy_obj.get("targeted_narray", [])
            selection_plans, new_global_seq = self._smart_source_selection(
                df_all, targeted_narray, global_gen_counter
            )
            global_gen_counter = new_global_seq
            
            if not selection_plans:
                print("  [Info] No generation plans created.")
                continue

            new_rows_buffer = []
            epoch_log_path = out / f"epoch_{epoch:02d}_generations.jsonl"
            epoch_think_path = out / f"epoch_{epoch:02d}_thoughts.log"
            if epoch_log_path.exists(): epoch_log_path.unlink()

            print(f"  > Starting ASYNC generation with {self.config.max_workers} workers...")
            
            # Start async event loop for concurrent generation
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            results = loop.run_until_complete(
                self._run_batch_generation(selection_plans, df_all, domain_norms_txt, val_freq, system_prompt)
            )

            # Process Results
            for result in results:
                plan_id = result["plan_id"]
                if result["think"]:
                    with open(epoch_think_path, 'a', encoding='utf-8') as f_think:
                        f_think.write(f"\n# --- ID: {plan_id} ---\n{result['think']}\n")
                with open(epoch_log_path, 'a', encoding='utf-8') as f_log:
                    f_log.write(f"\n# --- ID: {plan_id} ---\n{result['clean']}\n")
                try:
                    generated_data = parse_jsonl_strict(result["clean"])
                    if generated_data:
                        valid_rows = []
                        for row in generated_data:
                            if not row or not isinstance(row, dict): continue
                            row[self.config.article_col] = plan_id
                            valid_rows.append(row)
                        new_rows_buffer.extend(valid_rows)
                        print(f"    > {plan_id}: Parsed {len(valid_rows)} rows")
                except Exception as e:
                    print(f"    [Error] Parse failed for {plan_id}: {e}")

            if new_rows_buffer:
                df_new = pd.DataFrame(new_rows_buffer)
                for c in schema_cols:
                    if c not in df_new.columns:
                        df_new[c] = self.config.missing_token
                df_new = df_new[schema_cols]
                print(f"  > Generated {len(df_new)} new rows in total.")
                df_all = pd.concat([df_all, df_new], ignore_index=True)

        # === End of Loop ===
        final_path = out / "final_merged.xlsx"
        df_all.to_excel(final_path, index=False)
        print(f"\n[Done] Saved final merged data to: {final_path}")

        # Final Analysis
        print(">>> Running Final Analysis on Merged Data...")
        final_stats = build_distribution_stats(
            df_all,
            article_col=self.config.article_col,
            numeric_cols=self.config.numeric_cols or [],
            numeric_bin_rules=self.config.numeric_bin_rules or {},
            target_entity_constraints=self.config.target_entity_constraints
        )
        self._write_json(out / "final_distribution.json", final_stats)

        try:
            final_analyze_prompt = self.prompts.render("10_analyze.md", {
                "epoch": "Final",
                "distribution_stats_json": json.dumps(final_stats, ensure_ascii=False),
                "target_multirow_ratio": json.dumps(self.config.target_multirow_ratio),
                "pending_soft_issues": "[]", 
                "prev_forced_issues": "[]",
                "strategy_i_minus_1": "{}", 
                "numeric_bin_rules": "{}"
            })
            final_resp = self.llm.generate(final_analyze_prompt, system=system_prompt).text
            final_clean, _ = self._extract_think_and_clean(final_resp)
            (out / "final_analysis.md").write_text(final_clean, encoding="utf-8")
            print(f"[Done] Saved final analysis report to: {out / 'final_analysis.md'}")
        except Exception as e:
            print(f"[Warning] Failed to generate final analysis report: {e}")

        return {"out_dir": str(out), "final_path": str(final_path)}


def load_config(path: str) -> IterativeConfig:
    import yaml
    print(f"\n[Config] Loading configuration from: {path}")
    try:
        yaml_content = Path(path).read_text(encoding="utf-8")
        full_config = yaml.safe_load(yaml_content)
    except Exception as e:
        raise ValueError(f"Failed to parse YAML file: {e}")

    config_data = full_config.get('narray', {}).copy()
    if 'mineru_config' in full_config:
        config_data['mineru_config'] = full_config['mineru_config']
    
    valid_keys = set(IterativeConfig.__annotations__.keys())
    filtered_data = {}
    
    for k, v in config_data.items():
        if k in valid_keys:
            filtered_data[k] = v
        else:
            print(f"  [Config Warning] Parameter '{k}' is unused or unknown in current version.")
            
    return IterativeConfig(**filtered_data)