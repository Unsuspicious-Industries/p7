#!/bin/bash
# Benchmark completion script
# Run on remote with OpenRouter API access

# === OPENROUTER (unconstrained) ===
# Missing: 94
python benchmarks/run.py --tasks all --models Qwen/Qwen3.6-27B --modes unconstrained --backend openrouter --out final/unified/raw.jsonl --resume

# Missing: 94
python benchmarks/run.py --tasks all --models Qwen/Qwen3.6-27B --modes unconstrained_raw --backend openrouter --out final/unified/raw.jsonl --resume

# Missing: 94
python benchmarks/run.py --tasks all --models google/gemma-4-26B-A4B-it --modes unconstrained --backend openrouter --out final/unified/raw.jsonl --resume

# Missing: 94
python benchmarks/run.py --tasks all --models google/gemma-4-26B-A4B-it --modes unconstrained_raw --backend openrouter --out final/unified/raw.jsonl --resume

# Missing: 94
python benchmarks/run.py --tasks all --models google/gemma-4-E4B-it --modes unconstrained --backend openrouter --out final/unified/raw.jsonl --resume

# Missing: 94
python benchmarks/run.py --tasks all --models google/gemma-4-E4B-it --modes unconstrained_raw --backend openrouter --out final/unified/raw.jsonl --resume

# Missing: 94
python benchmarks/run.py --tasks all --models google/gemma-4-31b-it --modes unconstrained_raw --backend openrouter --out final/unified/raw.jsonl --resume

# Missing: 94
python benchmarks/run.py --tasks all --models openai/gpt-oss-20b --modes unconstrained --backend openrouter --out final/unified/raw.jsonl --resume

# Missing: 65
python benchmarks/run.py --tasks all --models openai/gpt-oss-20b --modes unconstrained_raw --backend openrouter --out final/unified/raw.jsonl --resume

# Missing: 94
python benchmarks/run.py --tasks all --models openai/gpt-5.4-mini --modes unconstrained_raw --backend openrouter --out final/unified/raw.jsonl --resume

# Missing: 94
python benchmarks/run.py --tasks all --models openai/gpt-5.3-codex --modes unconstrained_raw --backend openrouter --out final/unified/raw.jsonl --resume

# Missing: 94
python benchmarks/run.py --tasks all --models qwen/qwen3.5-35b-a3b --modes unconstrained_raw --backend openrouter --out final/unified/raw.jsonl --resume

# === LOCAL (constrained) - run locally ===
# Missing: 26
python benchmarks/run.py --tasks all --models Qwen/Qwen3.5-0.8B --modes constrained_direct --backend local --out final/unified/raw.jsonl --resume

# Missing: 69
python benchmarks/run.py --tasks all --models Qwen/Qwen3.5-0.8B --modes constrained_mixed --backend local --out final/unified/raw.jsonl --resume

# Missing: 21
python benchmarks/run.py --tasks all --models Qwen/Qwen3.5-2B --modes constrained_mixed --backend local --out final/unified/raw.jsonl --resume

# Missing: 16
python benchmarks/run.py --tasks all --models Qwen/Qwen3.5-4B --modes constrained_direct --backend local --out final/unified/raw.jsonl --resume

# Missing: 16
python benchmarks/run.py --tasks all --models Qwen/Qwen3.5-4B --modes constrained_mixed --backend local --out final/unified/raw.jsonl --resume

# Missing: 15
python benchmarks/run.py --tasks all --models Qwen/Qwen3.5-4B-Base --modes constrained_direct --backend local --out final/unified/raw.jsonl --resume

# Missing: 15
python benchmarks/run.py --tasks all --models Qwen/Qwen3.5-4B-Base --modes constrained_mixed --backend local --out final/unified/raw.jsonl --resume

# Missing: 94
python benchmarks/run.py --tasks all --models Qwen/Qwen3.6-27B --modes constrained_direct --backend local --out final/unified/raw.jsonl --resume

# Missing: 90
python benchmarks/run.py --tasks all --models Qwen/Qwen3.6-27B --modes constrained_mixed --backend local --out final/unified/raw.jsonl --resume

