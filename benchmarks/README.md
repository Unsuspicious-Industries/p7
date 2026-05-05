# p7 Benchmark Suite

This suite is built for reproducible local/OpenRouter benchmark runs from one TOML config. Modal is not part of the benchmark path.

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

- `unconstrained`: plain local Hugging Face generation plus post-hoc program extraction for legacy comparison.
- `unconstrained_raw`: plain local Hugging Face generation evaluated exactly as emitted, without markdown/newline/code-block extraction helpers.
- `constrained_direct`: p7/Aufbau direct constrained generation.
- `constrained_mixed`: p7 reasoning environment plus constrained formal generation.
- `outlines`: syntax-only Outlines CFG constraint engine. This intentionally drops type/context information.
- OpenRouter models use the same runner through `backend = "openrouter"` matrix entries.

Outlines is a constraint engine mode, not a backend. The backend remains local for open models.

## Single Runner

Install dependencies:

```bash
pip install -e ".[transformers]"
pip install outlines  # only needed when a config includes outlines mode
```

Run the bundled smoke config:

```bash
python benchmarks/run.py --config benchmarks/configs/smoke.toml
```

Run the full paper-oriented suite:

```bash
python benchmarks/run.py --config benchmarks/configs/paper.toml
```

Or, after installation:

```bash
p7-benchmark --config benchmarks/configs/smoke.toml
```

Resume an interrupted run without overwriting data:

```bash
python benchmarks/run.py --config path/to/benchmark.toml --resume
```

Every fresh run creates a dedicated run directory under `run.output_root/run.name`. If that directory already exists and `--resume` is not set, the runner creates a new timestamped sibling directory instead of overwriting the old one.

## Config Format

```bash
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
parallel_tasks = "auto"
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
modes = ["constrained_direct", "constrained_mixed", "unconstrained_raw", "unconstrained"]

[[matrix]]
name = "closed-baseline"
backend = "openrouter"
models = ["openai/gpt-5.4-mini", "anthropic/claude-4.5-haiku"]
modes = ["unconstrained"]
```

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

## Validation

```bash
pytest -q tests/benchmarks_api.py
pytest -q
```
