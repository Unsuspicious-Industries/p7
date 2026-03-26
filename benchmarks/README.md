# p7 Benchmark Suite

This directory contains a paper-oriented benchmark workflow to answer:

`Is p7 useful for generation tasks in typed formal languages?`

It benchmarks `stlc`, `fun`, and `imp` with 100 tasks each, plus a special stress test (`spec`) where the model is fed full grammar text in the prompt.

## Files

- `mk.py`: generate CSV task suites in `benchmarks/data/`
- `run.py`: execute constrained vs unconstrained runs on multiple models/tries
- `agg.py`: aggregate error rates and constrained-vs-unconstrained deltas

Constrained decoding defaults to the existing completion-based API.
Feed-only constrained decoding is available as an opt-in mode via `--feed-only`.

## Task CSV format

Each CSV row has:

- `task_id`, `language`, `grammar`, `category`
- `instruction`, `initial`, `expected`
- `max_tokens`
- `feed`: `hints` or `full`

`feed=hints` means compact syntax hints are sent in prompt.
`feed=full` means full grammar specification text (`p7.get_grammar`) is included in prompt.

## Default model set

The runner defaults to 5 small models:

- `gpt2`
- `EleutherAI/pythia-160m`
- `EleutherAI/pythia-410m`
- `Qwen/Qwen3.5-0.8B`
- `Qwen/Qwen3.5-2B`

Override with `--models`.

## Quickstart

```bash
python benchmarks/mk.py
python benchmarks/run.py --tries 3 --device cpu
python benchmarks/agg.py
```

## Dry run

```bash
python benchmarks/run.py --dry --max-tasks 5
```

## Remote GPU run

Use the helper script to sync this repo to a remote GPU server, run the full suite,
and (by default) pull results back locally:

```bash
./scripts/remote_benchmark.sh user@gpu-host
```

Useful flags:

- `--detached`: run inside a remote tmux session and return immediately
- `--device cuda`: force GPU device (default is already `cuda`)
- `--models "microsoft/Phi-3.5-mini-instruct,Qwen/Qwen3.5-2B"`: custom model set
- `--tries 1 --max-tasks 20`: fast smoke benchmark
- `--bootstrap`: attempt remote dependency setup (apt + rustup)
- `-c "ssh -p 19241 root@host -L 8080:localhost:8080"`: pass custom SSH params/tunnel setup
- `--batch`: force non-interactive SSH (useful in CI; local default is interactive)

## Output

- `benchmarks/out/raw.jsonl`: one row per attempt
- `benchmarks/out/trace.jsonl`: per-attempt token trace (`step`, `token`, `text`)
- `benchmarks/out/summary.csv`: error-rate table by model/mode/language
- `benchmarks/out/delta.csv`: constrained minus unconstrained deltas
- `benchmarks/out/report.md`: concise narrative summary

## Error taxonomy

- `ok`: exact match
- `parse_error`: output not parseable by grammar
- `incomplete`: parseable but incomplete
- `semantic_mismatch`: parseable and complete, but not expected program

This separation is useful in the paper: p7 should reduce `parse_error` strongly and improve exactness especially on grammar-dense tasks. This also means p7 reduces costs and latency for generation tasks in formal workloads.
