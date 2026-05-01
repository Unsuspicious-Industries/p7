#!/bin/bash
# Local benchmark completion (constrained modes)
# Run on GPU machine with --device cuda and --parallel-tasks auto

python benchmarks/run.py --tasks all --models Qwen/Qwen3.5-0.8B --modes constrained_direct --backend local --device cuda --parallel-tasks auto --out final/unified/raw.jsonl --resume

python benchmarks/run.py --tasks all --models Qwen/Qwen3.5-0.8B --modes constrained_mixed --backend local --device cuda --parallel-tasks auto --out final/unified/raw.jsonl --resume

python benchmarks/run.py --tasks all --models Qwen/Qwen3.5-2B --modes constrained_mixed --backend local --device cuda --parallel-tasks auto --out final/unified/raw.jsonl --resume

python benchmarks/run.py --tasks all --models Qwen/Qwen3.5-4B --modes constrained_direct --backend local --device cuda --parallel-tasks auto --out final/unified/raw.jsonl --resume

python benchmarks/run.py --tasks all --models Qwen/Qwen3.5-4B --modes constrained_mixed --backend local --device cuda --parallel-tasks auto --out final/unified/raw.jsonl --resume

python benchmarks/run.py --tasks all --models Qwen/Qwen3.5-4B-Base --modes constrained_direct --backend local --device cuda --parallel-tasks auto --out final/unified/raw.jsonl --resume

python benchmarks/run.py --tasks all --models Qwen/Qwen3.5-4B-Base --modes constrained_mixed --backend local --device cuda --parallel-tasks auto --out final/unified/raw.jsonl --resume

python benchmarks/run.py --tasks all --models Qwen/Qwen3.6-27B --modes constrained_direct --backend local --device cuda --parallel-tasks auto --out final/unified/raw.jsonl --resume

python benchmarks/run.py --tasks all --models Qwen/Qwen3.6-27B --modes constrained_mixed --backend local --device cuda --parallel-tasks auto --out final/unified/raw.jsonl --resume