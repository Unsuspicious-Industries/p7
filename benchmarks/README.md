# proposition7 Benchmark Suite

This suite is built for reproducible local/OpenRouter benchmark runs from one TOML config. The artifact submission path is the Docker image built by `scripts/build_artifact_image.sh`. Modal is not part of the benchmark path.

## Task Format

Each task is one TOML file in `benchmarks/data/`.

```toml
id = "stlc_apply_twice_int"
grammar = "stlc"
category = "stlc:logic_repetition"
max_tokens = 72

[prompt]
text = "Apply an Int endofunction twice."

[initial]
text = "λf:(Int->Int)."

[expected]
text = "λf:(Int->Int).λx:Int.(f (f x))"

[resolution]
mode = "equivalence"
type = "(Int -> Int) -> Int -> Int"
normalization = ["alpha", "beta"]
```

Tasks are pure task specifications. They do not specify benchmark mode, backend, model, prompt style, or constraint engine.

## Resolution

Resolution is implemented per grammar because the languages do not have the same semantics.

- `stlc`: `mode = "equivalence"`, with alpha/beta normalization and type checking against `resolution.type`.
- `fun`: `mode = "value"`, using the Fun interpreter and comparing `resolution.value`.
- `imp`: `mode = "env"`, using the Imp interpreter and comparing selected environment variables.
- `lamb`: currently exact/equivalence text comparison after scoped grammar parse.
- other grammars: exact/equivalence text fallback.

All checked-in TOML tasks are validated by tests: expected outputs must parse and pass their own resolution.

## Benchmark Modes

Benchmark modes are code-owned, not task-owned.

- `unconstrained`: plain generation evaluated exactly as emitted.
- `unconstrained_cleaned`: plain generation followed by post-hoc program extraction before evaluation.
- `unconstrained_thinking`: same two-phase reasoning structure as `constrained_mixed`, but the formal output block remains unconstrained.
- `constrained_direct`: proposition7/Aufbau direct constrained generation.
- `constrained_mixed`: proposition7 reasoning environment plus constrained formal generation.
- `outlines`: direct constrained generation using the Outlines CFG backend. This intentionally drops Aufbau type/context rules.
- `outlines_mixed`: same two-phase reasoning structure as `constrained_mixed`, but the formal block uses the Outlines CFG backend.
- OpenRouter models use the same runner through `backend = "openrouter"` matrix entries.

Outlines modes are constraint engine modes, not benchmark backends. The benchmark backend remains `local` for open models.

## Artifact Container

Build the reviewable Docker image tarball:

```bash
./scripts/build_artifact_image.sh
```

Run the full paper suite on a GPU host:

```bash
docker run --rm --gpus all \
  -v "$PWD/artifact-output:/workspace/benchmarks/out" \
  -v "$PWD/.env:/workspace/.env:ro" \
  proposition7-benchmark-artifact:latest paper --resume
```

Mount `.env` only if the config includes an OpenRouter matrix. It must define `OPENROUTER_API_KEY`.

Run the preferred deploy split at the same time in separate containers:
constrained modes on the GPU host, and closed-model unconstrained baselines
through OpenRouter. These commands use separate run directories, so they do not
share output files.

```bash
docker run --rm --gpus all \
  -v "$PWD/artifact-output:/workspace/benchmarks/out" \
  proposition7-benchmark-artifact:latest deploy --resume
```

```bash
docker run --rm \
  -v "$PWD/artifact-output:/workspace/benchmarks/out" \
  -v "$PWD/.env:/workspace/.env:ro" \
  proposition7-benchmark-artifact:latest openrouter --resume
```

`benchmarks/configs/vast_preferred.toml` records both halves of this intended
evaluation matrix and includes the same instructions in comments. Running that
combined config directly is valid, but matrices execute sequentially; use the
split `deploy` and `openrouter` entrypoints above when you want two containers
running at the same time.

The OpenRouter path uses the same task prompts, output extraction, parsing, and semantic resolution as local unconstrained benchmark modes. It does not run constrained or Outlines modes because closed models do not expose token-level logits.

Run the full small-Qwen Modal A10G benchmark using the same artifact container definition:

```bash
pip install -e ".[modal]"
modal setup
python scripts/modal_sandbox_run.py --config benchmarks/configs/modal_qwen_full.toml
```

That launcher copies `raw.jsonl` and `results.json` back to `dist/modal-qwen-full/`.

## Source Runner

Install dependencies:

```bash
pip install -e ".[transformers]"
pip install -e ".[outlines]"  # only needed when a config includes outlines modes
```

Run the full paper-oriented suite:

```bash
python benchmarks/run.py --config benchmarks/configs/paper.toml
```

Benchmarks are repository tools rather than part of the published PyPI package, so run them from a checkout:

```bash
python benchmarks/run.py --config benchmarks/configs/paper.toml
```

Resume an interrupted run without overwriting data:

```bash
python benchmarks/run.py --config path/to/benchmark.toml --resume
```

Every fresh run creates a dedicated run directory under `run.output_root/run.name`. If that directory already exists and `--resume` is not set, the runner creates a new timestamped sibling directory instead of overwriting the old one.

Process a finished run into CSV summaries:

```bash
python benchmarks/agg.py --in benchmarks/out/paper/raw.jsonl --out-dir benchmarks/out/paper/processed
```

## Config Format

```toml
schema_version = 1

[run]
name = "paper"
output_root = "benchmarks/out"
save_traces = false

[tasks]
selectors = ["all"]
ids = []
max_tasks = 0
max_tokens_override = 0

[execution]
tries = 1
seed = 7
timeout = 600.0
think_budget = 128
model_concurrency = "auto"
low_space = true

[local]
device = "cuda"
torch_dtype = "auto"
device_map = "auto"

[local.model_kwargs]

[openrouter]
env_file = ".env"

[[matrix]]
name = "open-models"
backend = "local"
models = ["Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"]
modes = ["constrained_direct", "constrained_mixed", "unconstrained", "unconstrained_cleaned"]

[[matrix]]
name = "closed-baseline"
backend = "openrouter"
models = ["openai/gpt-5.4-mini", "anthropic/claude-4.5-haiku"]
modes = ["unconstrained", "unconstrained_cleaned"]
```

`execution.model_concurrency` only controls concurrent runs for one model at a
time. The runner does not execute multiple models or matrices concurrently in a
single process.

For deployable split runs, use `benchmarks/configs/deploy_constrained.toml` for GPU-backed constrained modes and `benchmarks/configs/openrouter_closed.toml` for closed-model unconstrained modes.

## Resume Safety

Resume is opt-in via `--resume`.

The resume key is hash-aware:

```text
(backend, model, task_id, task_hash, resolution_hash, mode, try)
```

Changing a TOML task or resolution will not reuse stale rows. The raw JSONL line format is unchanged, and aggregation keeps the latest duplicate record for the same key.

## Output

- `<run>/config.toml`: exact copied config used for the run.
- `<run>/raw.jsonl`: append-only benchmark ledger. Record format is unchanged.
- `<run>/results.json`: canonical artifact with metadata, config, summaries, and deduped records.
- `<run>/traces.jsonl`: optional token traces when `run.save_traces = true`.
- `<processed>/cleaned_raw.jsonl`: cleaned and deduped benchmark records.
- `<processed>/benchmark_records.csv`: one row per cleaned benchmark attempt.
- `<processed>/summary_overall.csv`: overall benchmark metrics.
- `<processed>/summary_by_mode.csv`: pass/error rates by mode.
- `<processed>/summary_by_task_type.csv`: pass/error rates by task type.
- `<processed>/summary_by_mode_task_type.csv`: mode-by-task-type summaries.
- `<processed>/summary_by_task.csv`: per-task benchmark summaries.
- `<processed>/comparison_mixed_vs_unconstrained_thinking_by_model_language.csv`: constrained-vs-unconstrained formal block comparison for the reasoning modes.
- `<processed>/comparison_mixed_vs_unconstrained_thinking_by_task_type.csv`: same comparison grouped by task type.

The processing pipeline is CSV-first. No LaTeX/table generation is part of the tracked benchmark workflow.

## Validation

```bash
pytest -q tests/benchmarks_api.py
pytest -q
```
