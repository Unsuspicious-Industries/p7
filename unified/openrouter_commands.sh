#!/bin/bash
# OpenRouter benchmark completion
# Run locally with OpenRouter API access

python benchmarks/run.py --tasks all --models Qwen/Qwen3.6-27B --modes unconstrained --backend openrouter --out unified/raw.jsonl --resume

python benchmarks/run.py --tasks all --models Qwen/Qwen3.6-27B --modes unconstrained_raw --backend openrouter --out unified/raw.jsonl --resume

python benchmarks/run.py --tasks all --models google/gemma-4-26B-A4B-it --modes unconstrained --backend openrouter --out unified/raw.jsonl --resume

python benchmarks/run.py --tasks all --models google/gemma-4-26B-A4B-it --modes unconstrained_raw --backend openrouter --out unified/raw.jsonl --resume

python benchmarks/run.py --tasks all --models google/gemma-4-E4B-it --modes unconstrained --backend openrouter --out unified/raw.jsonl --resume

python benchmarks/run.py --tasks all --models google/gemma-4-E4B-it --modes unconstrained_raw --backend openrouter --out unified/raw.jsonl --resume

python benchmarks/run.py --tasks all --models google/gemma-4-31b-it --modes unconstrained_raw --backend openrouter --out unified/raw.jsonl --resume

python benchmarks/run.py --tasks all --models openai/gpt-oss-20b --modes unconstrained --backend openrouter --out unified/raw.jsonl --resume

python benchmarks/run.py --tasks all --models openai/gpt-oss-20b --modes unconstrained_raw --backend openrouter --out unified/raw.jsonl --resume

python benchmarks/run.py --tasks all --models openai/gpt-5.4-mini --modes unconstrained_raw --backend openrouter --out unified/raw.jsonl --resume

python benchmarks/run.py --tasks all --models openai/gpt-5.3-codex --modes unconstrained_raw --backend openrouter --out unified/raw.jsonl --resume

python benchmarks/run.py --tasks all --models qwen/qwen3.5-35b-a3b --modes unconstrained_raw --backend openrouter --out unified/raw.jsonl --resume