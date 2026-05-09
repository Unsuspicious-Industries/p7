#!/bin/bash
# Local benchmark completion (constrained modes)
# Run on GPU machine
python3 benchmarks/run.py --tasks all --models Qwen/Qwen3.5-4B-Base --modes constrained_mixed --backend local --out unified/raw.jsonl --resume  --parallel-tasks auto --think-budget 2048 --vram 48 --tries 2 --timeout 300

python3 benchmarks/run.py --tasks all --models Qwen/Qwen3.5-4B-Base --modes constrained_direct --backend local --out unified/raw.jsonl --resume  --parallel-tasks auto --vram 48 --tries 2 --timeout 300

python3 benchmarks/run.py --tasks all --models Qwen/Qwen3.5-4B --modes constrained_mixed --backend local --out unified/raw.jsonl --resume  --parallel-tasks auto --think-budget 2048 --vram 48 --tries 2 --timeout 300

python3 benchmarks/run.py --tasks all --models Qwen/Qwen3.5-4B --modes constrained_direct --backend local --out unified/raw.jsonl --resume  --parallel-tasks auto --vram 48 --tries 2 --timeout 300

python3 benchmarks/run.py --tasks all --models Qwen/Qwen3.5-2B --modes constrained_mixed --backend local --out unified/raw.jsonl --resume --parallel-tasks auto --think-budget 2048 --vram 48 --tries 2 --timeout 300

python3 benchmarks/run.py --tasks all --models Qwen/Qwen3.5-0.8B --modes constrained_mixed --backend local --out unified/raw.jsonl --resume --parallel-tasks auto --think-budget 2048 --vram 48 --tries 2 --timeout 300

python3 benchmarks/run.py --tasks all --models Qwen/Qwen3.5-0.8B --modes constrained_direct --backend local --out unified/raw.jsonl --resume --parallel-tasks auto --vram 48 --tries 2 --timeout 300

python3 benchmarks/run.py --tasks all --models zai-org/GLM-Z1-9B-0414 --modes constrained_direct --backend local --out unified/raw.jsonl --resume --parallel-tasks auto --vram 48 --tries 2 --timeout 300

python3 benchmarks/run.py --tasks all --models zai-org/GLM-Z1-9B-0414 --modes constrained_mixed --backend local --out unified/raw.jsonl --resume --parallel-tasks auto --vram 48 --tries 2 --timeout 300
