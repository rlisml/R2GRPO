import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import yaml
import argparse
import json
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from types import SimpleNamespace
import datetime
import asyncio # [New]

sys.path.append(str(Path(__file__).parent / "libs"))

from core.llm import CoreLLM
from narray.workflow import IterativeGenerator, IterativeConfig
from narray.prompts import PromptStore
from narray.llm import LLMResponse

from text_gen.data_manager import DataManager
from text_gen.text_analyzer import TextAnalyzer
from text_gen.strategies import AugmentationStrategy

def load_config(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# ... (Keep NArrayLLMAdapter unchanged) ...
class NArrayLLMAdapter:
    def __init__(self, core_llm: CoreLLM):
        self.core_llm = core_llm
    def generate(self, prompt: str, *, system: str = None, temperature: float = 0.2) -> LLMResponse:
        sys_p = system if system else "You are a helpful scientific data assistant."
        raw_response = self.core_llm.generate(prompt=prompt, system=sys_p, temperature=temperature)
        text_content = raw_response.text if hasattr(raw_response, 'text') else str(raw_response)
        return LLMResponse(text=text_content)
    # [Async support]
    async def generate_async(self, prompt: str, *, system: str = None, temperature: float = 0.2) -> LLMResponse:
        sys_p = system if system else "You are a helpful scientific data assistant."
        raw_response = await self.core_llm.generate_async(prompt=prompt, system=sys_p, temperature=temperature)
        text_content = raw_response.text if hasattr(raw_response, 'text') else str(raw_response)
        return LLMResponse(text=text_content)

class TextGenLLMAdapter:
    def __init__(self, core_llm: CoreLLM):
        self.core_llm = core_llm
    
    def generate(self, prompt: str, system: str = None, **kwargs):
        return self.core_llm.generate(prompt=prompt, system=system, **kwargs)

    # [Async support]
    async def generate_async(self, prompt: str, system: str = None, **kwargs):
        return await self.core_llm.generate_async(prompt=prompt, system=system, **kwargs)

# ... (Keep run_stage_1_narray unchanged) ...
def run_stage_1_narray(config: dict, llm: CoreLLM, seed_path: str, output_root: str):
    print("\n" + "="*60)
    print(">>> Stage 1: Running nArray (Structure Generation)")
    print("="*60)
    narray_cfg_dict = config['narray']
    valid_keys = IterativeConfig.__init__.__code__.co_varnames
    filtered_cfg = {k: v for k, v in narray_cfg_dict.items() if k in valid_keys}
    if 'mineru_config' in config:
        filtered_cfg['mineru_config'] = config['mineru_config']
    iter_config = IterativeConfig(**filtered_cfg)
    stage1_out = Path(output_root) / "stage1_structure"
    stage1_out.mkdir(parents=True, exist_ok=True)
    prompts_path = Path(__file__).parent / "libs" / "narray" / "prompts"
    prompts = PromptStore(str(prompts_path))
    narray_adapter = NArrayLLMAdapter(llm)
    generator = IterativeGenerator(config=iter_config, llm=narray_adapter, prompts=prompts)
    result = generator.run(seed_path=seed_path, out_dir=str(stage1_out), path2knowledge=narray_cfg_dict.get('path2knowledge'))
    final_excel = result['final_path']
    print(f"  [Stage 1 Done] Generated Data: {final_excel}")
    return final_excel

# [Refactored Async Stage 2]
def run_stage_2_textgen(config: dict, llm: CoreLLM, input_excel: str, output_root: str):
    print("\n" + "="*60)
    print(">>> Stage 2: Running TextGen (Content Generation) [Async Mode]")
    print("="*60)

    tg_cfg = config['text_gen']
    
    # 1. 确定输入数据源
    target_input = input_excel
    if not target_input or not Path(target_input).exists():
        fallback_path = tg_cfg.get('based_narray_data')
        if fallback_path:
            print(f"  [Info] Stage 1 output not present/requested. Using 'based_narray_data': {fallback_path}")
            target_input = fallback_path
        else:
            print("  [Error] No input data available for Stage 2.")
            return

    output_dir = Path(output_root) / "stage2_content"
    output_dir.mkdir(parents=True, exist_ok=True)
    md_dir = tg_cfg['md_dir']
    
    aug_level = str(tg_cfg.get('aug_level', 'phrase')).lower()
    
    print(f"  > Input Excel: {target_input}")
    print(f"  > MD Library:  {md_dir}")
    print(f"  > Output Dir:  {output_dir}")
    print(f"  > Aug Level:   {aug_level}")

    tg_llm_adapter = TextGenLLMAdapter(llm)
    dm = DataManager(str(target_input), md_dir)
    analyzer = TextAnalyzer()
    strategy = AugmentationStrategy(tg_llm_adapter, dm, analyzer)

    grouped_data = dm.get_grouped_data()
    print(f"  > Loaded {len(grouped_data)} groups.")

    all_final_data = []
    all_tags = []
    all_lengths = []
    
    num_aug = tg_cfg.get('num_aug', 1)
    min_sec_words = tg_cfg.get('min_section_words', 200)
    max_art_words = tg_cfg.get('max_article_words', 3000)
    gen_ref_num = tg_cfg.get('generate_ref_article_num', 3)
    
    # 允许配置文件设置 max_workers，默认 5
    max_workers = tg_cfg.get('max_workers', 5)
    print(f"  > Max Concurrency: {max_workers}")

    # Async Main Logic
    async def _async_main():
        semaphore = asyncio.Semaphore(max_workers)
        
        # 1. 创建所有任务
        tasks = []
        
        for article_name, group_df in grouped_data.items():
            if '_row_id' not in group_df.columns:
                group_df = group_df.copy()
                group_df['_row_id'] = range(len(group_df))
            
            if aug_level == "phrase":
                # [Fix] 传递 num_aug 参数
                tasks.append(strategy.run_phrase_async(article_name, group_df, num_aug, semaphore))
                
            elif aug_level == "scale":
                for _ in range(num_aug):
                    tasks.append(strategy.run_scale_async(article_name, group_df, min_sec_words, max_art_words, semaphore))
                    
            elif aug_level == "generate":
                for _ in range(num_aug):
                    tasks.append(strategy.run_generate_async(article_name, group_df, gen_ref_num, min_sec_words, max_art_words, semaphore))
            
            else:
                print(f"  [Warning] Unknown aug_level: {aug_level}")

        if not tasks:
            print("  [Info] No tasks created.")
            return

        print(f"  > Scheduled {len(tasks)} tasks.")

        # 2. 执行并使用进度条
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Async Generating"):
            try:
                # results is List[Dict]
                results = await f
                if not results: continue
                
                # 实时保存结果
                for res in results:
                    # [Async I/O Offload] 再次确认这里使用了 run_in_executor
                    await asyncio.get_running_loop().run_in_executor(
                        None, 
                        save_result,
                        output_dir, res, all_final_data, all_tags, all_lengths
                    )
                    
            except Exception as e:
                print(f"  [Error] Task failed: {e}")
                import traceback
                traceback.print_exc()

    # 启动 Event Loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    loop.run_until_complete(_async_main())

    if all_final_data:
        df_final = pd.DataFrame(all_final_data)
        if '_row_id' in df_final.columns:
            df_final.drop(columns=['_row_id'], inplace=True)
        
        final_path = output_dir / "augmented_data.xlsx"
        df_final.to_excel(final_path, index=False)
        
        pd.DataFrame(all_tags).to_excel(output_dir / "augmented_tags.xlsx", index=False)
        pd.DataFrame(all_lengths).to_excel(output_dir / "article_stats.xlsx", index=False)
        print(f"  [Stage 2 Done] Results saved to {output_dir}")
    else:
        print("  [Stage 2 Warning] No data generated.")

def save_result(output_dir, res, data_list, tags_list, length_list):
    import re
    safe_title = re.sub(r'[\\/*?:"<>|]', "_", res['title'])
    if len(safe_title) > 150: safe_title = safe_title[:150]
    
    # 1. Save Paper Content (.md)
    with open(output_dir / f"{safe_title}.md", 'w', encoding='utf-8') as f:
        f.write(res['text'])
        
    # 2. Save Thoughts (.txt)
    if res.get('think'):
        with open(output_dir / f"{safe_title}_thoughts.txt", 'w', encoding='utf-8') as f:
            f.write(res['think'])
    
    # 3. Save Data Positions
    df = res['data']
    for _, row in df.iterrows():
        d = row.to_dict()
        d['Article'] = safe_title
        data_list.append(d)
        
    tags_list.extend(res['tags'])
    length_list.append({
        "Article": safe_title,
        "Word_Count": res.get('word_count', 0),
        "File_Name": f"{safe_title}.md"
    })

def main():
    parser = argparse.ArgumentParser(description="Unified Data Gen Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    global_cfg = cfg['pipeline']
    
    print(">>> Initializing Unified LLM Core...")
    core_llm = CoreLLM(cfg['llm'])
    
    current_data_path = global_cfg.get('seed_data_path')
    workspace = global_cfg['workspace_dir']
    ws_path = Path(workspace)
    if ws_path.exists():
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name = f"{ws_path.name}_{timestamp}"
        workspace = str(ws_path.parent / new_name)
        print(f"  [Init] Workspace '{global_cfg['workspace_dir']}' exists.")
        print(f"  [Init] Auto-renamed output directory to: {workspace}")

    # Stage 1
    if global_cfg.get('run_narray', False):
        try:
            current_data_path = run_stage_1_narray(cfg, core_llm, current_data_path, workspace)
        except Exception as e:
            print(f"Stage 1 Error: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        current_data_path = ""
    
    # Stage 2
    if global_cfg.get('run_textgen', False):
        try:
            run_stage_2_textgen(cfg, core_llm, current_data_path, workspace)
        except Exception as e:
            print(f"Stage 2 Error: {e}")
            import traceback
            traceback.print_exc()
            return

    print("\n>>> All Stages Completed.")

if __name__ == "__main__":
    main()