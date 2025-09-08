# -*- coding: utf-8 -*-
import argparse
# Argument parser setup
parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5 models with multi-task SFT")
parser.add_argument('--cuda_device', type=str, default="1", help="CUDA device number(s) to use (e.g., '6' or '0,1')")
parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-0.5B-Instruct_sft_multi_task_16_2",
                   help="Model name or path from the available qwen_models list")
parser.add_argument('--train_epochs', type=int, default=3)
args = parser.parse_args()
# Set environment variables and configurations based on arguments
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
import torch
import json
import re
import numpy as np
import logging
from datetime import datetime
from typing import List, Dict, Set, Tuple
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from prompt_background import ner_background, re_background
# Model configuration
max_seq_length = 2500
num_train_epochs = args.train_epochs
lora_rank = 64
lora_alpha = 128
factor = 1  # Scale the F1
per_device_train_batch_size = 16
num_generations = 16
gradient_accumulation_steps = 1
model_name = args.model_name
load_in_4bit = '4bit' in model_name
dtype = torch.bfloat16
dataset_name = ''


# SYSTEM_PROMPT = """
#         Respond in the following format:
#         <think>
#         Provide step-by-step reasoning to solve the task based on the given instructions and sentence.
#         </think>
#         <answer>
#         Provide the final answer in JSON format as specified in the instruction.
#         </answer>
#         """

#for R2GRPO
SYSTEM_PROMPT = """
    Respond in the following format:
    <reasoning>
    Provide step-by-step reasoning to solve the task based on the given instructions and sentence.
    </reasoning>
    <think>
    Cite the specific sentence part (e.g., phrase, verb, or structure) supporting the relation.
    Articulate a symbolic pattern you discovered (e.g., "The verb 'achieves' suggests a Method is applied to a Task, implying a relation").
    Explain how this pattern leads to the predicted relation, referencing the relationship definition.
    Use concise, logical chains (e.g., "X performs Y → relation Z because of definition").
    </think>
    <answer>
    Provide the final answer in JSON format as specified in the instruction.
    </answer>"""
    
# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    local_files_only=True,
    fast_inference=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=lora_alpha,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Configure logging to file
log_dir = f'logs/grpotraininglog/{model_name}_2tagnew{dataset_name}'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=f'{log_dir}/training_responses_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filemode='a'
)
logger = logging.getLogger(__name__)

# Load dataset
dataset = load_dataset("json", data_files={
    "train": f"SciER/SciER/LLM/train{dataset_name}.jsonl",
})
train_dataset = dataset["train"]
max_steps = (num_train_epochs * len(train_dataset) // (gradient_accumulation_steps))
logging_steps = int(0.001 * max_steps)
logger_steps = int(0.001 * max_steps)

# Prompt formatting function
def format_ner_re_end2end_prompt(example: Dict) -> Dict:
    sentence = example["sentence"]
    ner = example["ner"]
    rel = example["rel"]

    prompt = f"""{ner_background}

{re_background}

Given the sentence: "{sentence}"

Extract entities and their relations.

### Instruction:
- Think step-by-step to identify entities and their relationships.
- Return the results in JSON format with:
  - "ner": a list of [entity, type] pairs.
  - "rel": a list of [subject, relation, object] triples.
"""

    response_dict = {"ner": ner, "rel": rel}
    response = f"""<think>
    ...
</think>
<answer>
{json.dumps(response_dict, ensure_ascii=False)}
</answer>"""
    return {"prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}], "answer": response}

# Prepare dataset
def prepare_grpo_dataset(dataset: Dataset) -> Dataset:
    return dataset.map(format_ner_re_end2end_prompt, batched=False)

train_grpo_dataset = prepare_grpo_dataset(train_dataset)
train_grpo_dataset = train_grpo_dataset.map(
    lambda x: {"difficulty": len(x["sentence"].split()) + len(x["rel"])},
    desc="Computing difficulty"
)
train_grpo_dataset = train_grpo_dataset.sort("difficulty")
train_grpo_dataset = train_grpo_dataset.remove_columns("difficulty")

# Pre-compile regex patterns for efficiency
SYMBOLIC_PATTERN = re.compile(r"implies|suggests|because|maps to|indicates|pattern|rule|therefore|hence|thus|consequently|leads to", re.IGNORECASE)
QUOTE_PATTERN = re.compile(r'[\'"]([^\'"]+)[\'"]')
STEP_PATTERN = re.compile(r"(step \d+|\d\.|[a-z]\)|-\s|•\s)", re.IGNORECASE)
EXPLICIT_PATTERN_DEF_PATTERN = re.compile(r"(pattern|rule)\s*.*?(implies|suggests|indicates|means|defines)", re.IGNORECASE)
RELATION_EXPLAIN_PATTERN = re.compile(r"(relation|relationship)\s*.*? (because|due to|as)", re.IGNORECASE)

def extract_xml_tag(text: str, tag: str) -> str:
    try:
        match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""
    except (TypeError, AttributeError):
        return ""

def extract_xml_think(text: str) -> str:
    return extract_xml_tag(text, "think")

def extract_xml_answer(text: str) -> str:
    return extract_xml_tag(text, "answer")

def reasoning_reward(prompts: List[Dict], completions: List[List[Dict]], answer, sentence: List[str], max_reasoning_reward: float = 0.5, **kwargs) -> List[float]:
    rewards = []

    if not isinstance(sentence, list):
        if isinstance(sentence, str) and len(completions) > 0:
             sentence = [sentence] * len(completions)
        else:
             sentence = [""] * len(completions)

    if len(sentence) != len(completions):
         sentence = (sentence + [""] * len(completions))[:len(completions)]

    try:
        is_relevant_task = prompts and len(prompts[0]) > 1 and "Extract entities and their relations" in prompts[0][1].get("content", "")
    except (IndexError, TypeError, KeyError):
        is_relevant_task = False

    if not is_relevant_task:
        return [0.0] * len(completions)

    original_total_max_score = 0.8 + 1.5 + 0.7
    scaling_factor = max_reasoning_reward / original_total_max_score if original_total_max_score > 0 else 0.0

    scaled_max_evidence_contribution = 0.8 * scaling_factor
    scaled_max_symbolic_contribution = 1.5 * scaling_factor
    scaled_max_consistency_contribution = 0.7 * scaling_factor

    scaled_max_possible_citation_score = 0.6 * scaling_factor
    scaled_citation_bonus_base = 0.1 * scaling_factor
    scaled_citation_bonus_specificity = 0.1 * scaling_factor
    scaled_fallback_overlap_cap = 0.2 * scaling_factor

    scaled_symbolic_bonuses = {
        "SYMBOLIC_PATTERN": 0.5 * scaling_factor,
        "EXPLICIT_PATTERN_DEF_PATTERN": 0.3 * scaling_factor,
        "RELATION_EXPLAIN_PATTERN": 0.4 * scaling_factor,
        "STEP_PATTERN": 0.3 * scaling_factor,
    }

    scaled_max_possible_consistency_score = 0.7 * scaling_factor

    for i, completion_list in enumerate(completions):
        current_sentence = sentence[i] if i < len(sentence) else ""
        current_sentence_lower_stripped = current_sentence.lower().strip() if current_sentence else ""
        score = 0.0

        try:
            completion_content = ""
            if completion_list and isinstance(completion_list, list) and completion_list[0]:
                 completion_content = completion_list[0].get("content", "")

            think_text = extract_xml_think(completion_content)
            answer_text = extract_xml_answer(completion_content)

            if not think_text:
                rewards.append(0.0)
                continue

            evidence_score = 0.0
            accumulated_citation_reward = 0.0
            found_quotes = False

            if current_sentence_lower_stripped:
                quoted_phrases = QUOTE_PATTERN.findall(think_text)

                if quoted_phrases:
                    found_quotes = True
                    for phrase in quoted_phrases:
                        phrase_lower_stripped = phrase.lower().strip()

                        if not phrase_lower_stripped: continue

                        if phrase_lower_stripped == current_sentence_lower_stripped:
                            continue

                        if phrase_lower_stripped in current_sentence_lower_stripped:
                            specificity_bonus = max(0, 1.0 - (len(phrase_lower_stripped) / len(current_sentence_lower_stripped)))
                            accumulated_citation_reward += scaled_citation_bonus_base + (scaled_citation_bonus_specificity * specificity_bonus)

                evidence_score = min(accumulated_citation_reward, scaled_max_possible_citation_score)

            score += min(evidence_score, scaled_max_evidence_contribution)

            symbolic_score = 0.0

            if SYMBOLIC_PATTERN.search(think_text):
                symbolic_score += scaled_symbolic_bonuses["SYMBOLIC_PATTERN"]
            if EXPLICIT_PATTERN_DEF_PATTERN.search(think_text):
                symbolic_score += scaled_symbolic_bonuses["EXPLICIT_PATTERN_DEF_PATTERN"]
            if RELATION_EXPLAIN_PATTERN.search(think_text):
                symbolic_score += scaled_symbolic_bonuses["RELATION_EXPLAIN_PATTERN"]
            if STEP_PATTERN.search(think_text):
                symbolic_score += scaled_symbolic_bonuses["STEP_PATTERN"]

            score += min(symbolic_score, scaled_max_symbolic_contribution)

            consistency_score = 0.0
            num_rels = 0
            try:
                expected = extract_xml_answer(answer[0])
                expected_rel = json.loads(expected)["rel"]

                if isinstance(expected_rel, list):
                    if not isinstance(expected_rel, list): expected_rel = []

                    num_rels = len(expected_rel)
                    rel_mentions = 0
                    if num_rels > 0:
                        think_text_lower = think_text.lower()
                        for rel_triple in expected_rel:
                           try:
                               if isinstance(rel_triple, (list, tuple)) and len(rel_triple) == 3 and \
                                  all(isinstance(part, str) for part in rel_triple):
                                   if all(str(part).lower() in think_text_lower for part in rel_triple):
                                       rel_mentions += 1
                               elif isinstance(rel_triple, (list, tuple)) and len(rel_triple) == 3:
                                   try:
                                        if all(str(part).lower() in think_text_lower for part in rel_triple):
                                             rel_mentions += 1
                                   except Exception: pass
                           except Exception: continue

                        consistency_score = (rel_mentions / num_rels) * scaled_max_possible_consistency_score

                    elif think_text and "no relation" in think_text.lower():
                         if not expected_rel:
                              consistency_score = 0.5 * scaled_max_possible_consistency_score
                else:
                     consistency_score = 0.0

            except (json.JSONDecodeError, TypeError, AttributeError):
                 consistency_score = 0.0
            except Exception as e:
                 consistency_score = 0.0

            score += min(consistency_score, scaled_max_consistency_contribution)

            final_score = max(0.0, min(score, max_reasoning_reward))
            rewards.append(float(final_score))

        except Exception as e:
            rewards.append(0.0)

    return rewards

def compute_overlap_reward(expected_span, predicted_span):
    if not isinstance(expected_span, str) or not isinstance(predicted_span, str):
        return 0.0

    try:
        expected_words = set(expected_span.split())
        predicted_words = set(predicted_span.split())
        intersection = len(expected_words & predicted_words)
        union = len(expected_words | predicted_words)
        return intersection / union if union > 0 else 0.0
    except AttributeError:
        return 0.0

def compute_entity_overlap(expected_ner, predicted_ner):
    if not expected_ner or not predicted_ner:
        return 0.0

    predicted_entities = []
    for entry in predicted_ner:
        if isinstance(entry, (list, tuple)) and len(entry) >= 1:
            entity = entry[0] if isinstance(entry[0], str) else str(entry[0])
            predicted_entities.append(entity)
        elif isinstance(entry, dict):
            if "entity" in entry and isinstance(entry["entity"], str):
                predicted_entities.append(entry["entity"])
            elif "text" in entry and isinstance(entry["text"], str):
                predicted_entities.append(entry["text"])
        elif isinstance(entry, str):
            predicted_entities.append(entry)

    expected_entities = []
    for entry in expected_ner:
        if isinstance(entry, (list, tuple)) and len(entry) >= 1:
            entity = entry[0] if isinstance(entry[0], str) else str(entry[0])
            expected_entities.append(entity)
        elif isinstance(entry, dict):
            if "entity" in entry and isinstance(entry["entity"], str):
                expected_entities.append(entry["entity"])
            elif "text" in entry and isinstance(entry["text"], str):
                expected_entities.append(entry["text"])
        elif isinstance(entry, str):
            expected_entities.append(entry)

    overlap_scores = []
    for e_entity in expected_entities:
        max_overlap = 0.0
        for p_entity in predicted_entities:
            try:
                current_overlap = compute_overlap_reward(e_entity, p_entity)
                if current_overlap > max_overlap:
                    max_overlap = current_overlap
            except:
                continue
        overlap_scores.append(max_overlap)

    return sum(overlap_scores) / len(expected_entities) if expected_entities else 0.0

def compute_f1(expected: List, predicted: List) -> float:
    if not expected and not predicted:
        return 1.0
    if not expected or not predicted:
        return 0.0
    expected_set = set(tuple(x) for x in expected)
    predicted_set = set(tuple(x) for x in predicted)
    tp = len(expected_set & predicted_set)
    fp = len(predicted_set - expected_set)
    fn = len(expected_set - predicted_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1

log_global_step = -1

def ner_reward(prompts, completions, answer, **kwargs) -> List[float]:
    global log_global_step
    log_global_step += 1

    if "Extract entities and their relations" not in prompts[0][1]["content"]:
        return [0.0] * len(completions)
    
    responses = [extract_xml_answer(completion[0]["content"]) for completion in completions]
    expected = extract_xml_answer(answer[0])
    
    try:
        expected_dict = json.loads(expected) if expected else {}
        expected_ner = expected_dict.get("ner", [])
    except (json.JSONDecodeError, TypeError):
        expected_ner = []

    if log_global_step % logger_steps == 0:
        logger.info(f"\nStep {log_global_step}")
        example_idx = 0
        generated_response = completions[example_idx][0]['content']
        expected_answer = answer[example_idx]
        logger.info(f"Generated Response:\n{generated_response}")
        logger.info(f"Expected Answer:\n{expected_answer}")
        logger.info("--------------------------------------------------")

    rewards = []
    for resp in responses:
        if not resp:
            rewards.append(0.0)
            continue
            
        try:
            resp_dict = json.loads(resp) if resp else {}
            if isinstance(resp_dict, dict):
                resp_ner = resp_dict.get("ner", [])
            else:
                resp_ner = []
            
            f1_score = compute_f1(expected_ner, resp_ner) if expected_ner else 0.0
            overlap_score = compute_entity_overlap(expected_ner, resp_ner)
            combined_score = (0.7 * f1_score) + (0.3 * overlap_score)
            rewards.append(factor * combined_score)
            
        except (json.JSONDecodeError, TypeError):
            rewards.append(0.0)

    return rewards

def rel_correctness_reward(prompts, completions, answer, **kwargs) -> List[float]:
    if "Extract entities and their relations" not in prompts[0][1]["content"]:
        return [0.0] * len(completions)
    responses = [extract_xml_answer(completion[0]["content"]) for completion in completions]
    expected = extract_xml_answer(answer[0])
    try:
        expected_rel = json.loads(expected)["rel"]
        rewards = []
        for resp in responses:
            if not resp:
                resp_rel = []
            else:
                try:
                    resp_dict = json.loads(resp) if resp else {}
                    if isinstance(resp_dict, dict):
                        resp_rel = resp_dict.get("rel", [])
                    else:
                        resp_rel = []
                except (json.JSONDecodeError, TypeError):
                    resp_rel = []
            f1 = compute_f1(expected_rel, resp_rel)
            rewards.append(2 * factor * f1)
        return rewards
    except (json.JSONDecodeError, TypeError):
        return [0.0] * len(responses)

def length_reward(prompts, completions, answer, **kwargs) -> List[float]:
    target_length = 500
    max_reward = 0.2

    rewards = []
    for completion in completions:
        think_text = extract_xml_think(completion[0]["content"])
        if not think_text:
            rewards.append(0.0)
            continue
        
        word_count = len([w for w in think_text.split() if w.strip()])
        reward = max_reward * (word_count) / (target_length)
        rewards.append(float(reward))

    return rewards

def format_reward(completions, **kwargs) -> List[float]:
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

# GRPO Training setup
training_args = GRPOConfig(
    use_vllm=True,
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    logging_steps=logging_steps,
    num_generations=num_generations,
    num_train_epochs=num_train_epochs,
    max_prompt_length=1100,
    save_steps=int(0.1 * max_steps),
    max_completion_length=1400,
    max_grad_norm=0.1,
    report_to="tensorboard",
    output_dir=f'loras/grpo_training/grpoori/{model_name}_{lora_rank}{dataset_name}_newrule2',
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        ner_reward,
        rel_correctness_reward,
        reasoning_reward,
        format_reward,
    ],
    args=training_args,
    train_dataset=train_grpo_dataset,
)

# Train the model
trainer.train()

merged_path = f"merged_models/grpoori/{model_name}_grpo_{lora_rank}{dataset_name}_casual"
lora_save_path = f"loras/grpo_final_grpoori/{model_name}_grpo_{lora_rank}{dataset_name}_casual"
# Save the LoRA adapters
model.save_pretrained(lora_save_path)
tokenizer.save_pretrained(lora_save_path)

# Merge and save the model in 16-bit format
model.save_pretrained_merged(
    merged_path,
    tokenizer,
    save_method="merged_16bit",
)

print(f"GRPO training completed. LoRA saved to {lora_save_path}")
print(f"Merged 16-bit model saved to {merged_path}")