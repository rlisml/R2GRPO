# Scientific Entity and Relation Extraction (SciER)

This repository contains code for training and evaluating models for scientific entity recognition and relation extraction tasks.

## Overview

The project implements a two-stage training approach:
1. mimic-Supervised Fine-tuning (SFT)
2. R2GRPO optimization

## Requirements

See `requirements.txt` for complete dependencies. Key requirements:
- Python 3.8+
- PyTorch 2.5.1
- Transformers 4.49.0
- Unsloth 2025.3.14
- VLLM 0.7.2

## Training Pipeline

### Stage 1: mimic-Supervised Fine-tuning

The SFT stage uses `newreasonallsft.py` to perform instruction tuning with multiple tasks:

- Named Entity Recognition (NER)
- Relation Extraction (RE) 
- End-to-end Entity and Relation Extraction

To run SFT training:
```bash
python mimicsft.py --model_name "Qwen/Qwen2.5-7B-Instruct" --tasks ner re re_plus --train_epochs 3 --cuda_device 0
python mimicsft.py --model_name "Qwen/Qwen2.5-7B-Instruct" --tasks end2end --train_epochs 3 --cuda_device 0
```

### Stage 2: R^2GRPO

To run R^2GRPO training:
```bash
python r2grpo.py --model_path [sft_model_path] --cuda_device 0
```

### Evaluation
```bash
python evaluate.py --model_path [model_path] --cuda_device 0
```

### Data format
Training data should follow this JSON format:
{
    "sentence": "text...",
    "ner": [["entity1", "type1"], ["entity2", "type2"]],
    "rel": [["subject", "relation", "object"]]
}


Our code is based on the [unsloth](https://github.com/unslothai/unsloth) package.

The original dataset can be found in [SciER](https://github.com/edzq/SciER)


# Scientific Data Generation Workflow
The "data_gen_workflow" contains code for the automated generation of structured n-array data and corresponding academic paper using Large Language Models (LLMs).

## Requirements
- Python 3.10+
- PyTorch (CUDA 12.x)
- vllm==0.11.2 (for local inference acceleration)

## Stage 1: N-Array generation
To run the full workflow (starting with Stage 1): python run_workflow.py --config yaml/test_qwen3-32b.yaml. If you only want to generate n-array data, please set 'run_textgen' to 'false' in the YAML file.

## Stage 2: Scientific papers generation
If you only want to generate scientific papers, please set 'run_narray' to 'false' and 'run_narray' is' true 'in the YAML file. At the same time, you need to specify the path of' ased_narray_data '.

### Data
The benchmark data used for data generation workflow comes from:
Odabaşı Ç, Yıldırım R. Performance analysis of perovskite solar cells in 2013–2018 using machine-learning tools[J]. Nano Energy, 2019, 56: 770-791.


