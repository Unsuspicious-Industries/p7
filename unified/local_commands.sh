#!/bin/bash
# Local benchmark completion (constrained modes)
# Run on GPU machine

python benchmarks/run.py --tasks all --models Qwen/Qwen3.5-0.8B --modes constrained_direct --backend local --out unified/raw.jsonl --resume

python benchmarks/run.py --tasks all --models Qwen/Qwen3.5-0.8B --modes constrained_mixed --backend local --out unified/raw.jsonl --resume

python benchmarks/run.py --tasks all --models Qwen/Qwen3.5-2B --modes constrained_mixed --backend local --out unified/raw.jsonl --resume

python benchmarks/run.py --tasks all --models Qwen/Qwen3.5-4B --modes constrained_direct --backend local --out unified/raw.jsonl --resume

python benchmarks/run.py --tasks all --models Qwen/Qwen3.5-4B --modes constrained_mixed --backend local --out unified/raw.jsonl --resume

python benchmarks/run.py --tasks all --models Qwen/Qwen3.5-4B-Base --modes constrained_direct --backend local --out unified/raw.jsonl --resume

python benchmarks/run.py --tasks all --models Qwen/Qwen3.5-4B-Base --modes constrained_mixed --backend local --out unified/raw.jsonl --resume

python benchmarks/run.py --tasks all --models Qwen/Qwen3.6-27B --modes constrained_direct --backend local --out unified/raw.jsonl --resume

python benchmarks/run.py --tasks all --models Qwen/Qwen3.6-27B --modes constrained_mixed --backend local --out unified/raw.jsonl --resume