#!/bin/bash
# OpenRouter benchmark completion
# Run locally with OpenRouter API access

python benchmarks/run.py --tasks all --models Qwen/Qwen3.6-27B --modes unconstrained --backend openrouter --out unified/raw.jsonl --resume --tries 1 --parallel-tasks 24

python benchmarks/run.py --tasks all --models Qwen/Qwen3.6-27B --modes unconstrained_raw --backend openrouter --out unified/raw.jsonl --resume --tries 1 --parallel-tasks 24

python benchmarks/run.py --tasks all --models google/gemma-4-26B-A4B-it --modes unconstrained --backend openrouter --out unified/raw.jsonl --resume --tries 1 --parallel-tasks 24

python benchmarks/run.py --tasks all --models google/gemma-4-26B-A4B-it --modes unconstrained_raw --backend openrouter --out unified/raw.jsonl --resume --tries 1 --parallel-tasks 24

python benchmarks/run.py --tasks all --models google/gemma-4-E4B-it --modes unconstrained --backend openrouter --out unified/raw.jsonl --resume --tries 1 --parallel-tasks 24

python benchmarks/run.py --tasks all --models google/gemma-4-E4B-it --modes unconstrained_raw --backend openrouter --out unified/raw.jsonl --resume --tries 1 --parallel-tasks 24

python benchmarks/run.py --tasks all --models google/gemma-4-31b-it --modes unconstrained_raw --backend openrouter --out unified/raw.jsonl --resume --tries 1 --parallel-tasks 24

python benchmarks/run.py --tasks all --models openai/gpt-oss-20b --modes unconstrained --backend openrouter --out unified/raw.jsonl --resume --tries 1 --parallel-tasks 24

python benchmarks/run.py --tasks all --models openai/gpt-oss-20b --modes unconstrained_raw --backend openrouter --out unified/raw.jsonl --resume --tries 1 --parallel-tasks 24

python benchmarks/run.py --tasks all --models openai/gpt-5.4-mini --modes unconstrained_raw --backend openrouter --out unified/raw.jsonl --resume --tries 1 --parallel-tasks 24

python benchmarks/run.py --tasks all --models deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --modes unconstrained_raw --backend openrouter --out unified/raw.jsonl --resume --tries 1 --parallel-tasks 24

python benchmarks/run.py --tasks all --models deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --modes unconstrained --backend openrouter --out unified/raw.jsonl --resume --tries 1 --parallel-tasks 24


