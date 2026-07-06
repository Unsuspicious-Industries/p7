# proposition7 Benchmark Suite

This suite is built for reproducible local/OpenRouter benchmark runs from one TOML config. The artifact Docker image contains this runner, every config under `benchmarks/configs/` at build time, and all Python dependencies, but not model weights or local benchmark outputs.

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

- `constrained_direct`: proposition7/Aufbau direct constrained generation.
- `constrained_mixed`: proposition7 reasoning environment plus constrained formal generation.

- `outlines`: direct constrained generation using the Outlines CFG backend. This intentionally drops Aufbau type/context rules.
- `outlines_mixed`: same two-phase reasoning structure as `constrained_mixed`, but the formal block uses the Outlines CFG backend.
- OpenRouter models use the same runner through `backend = "openrouter"` matrix entries.

Outlines modes are constraint engine modes, not benchmark backends. The benchmark backend remains `local` for open models.

## Pruning Study

Measures how much the *semantic* layer of an Aufbau grammar (its typing rules)
prunes during constrained decoding, versus the bare *syntactic* grammar. Three
regimes, on the exact same task corpus:

- `unconstrained` on `<g>` — no grammar pruning (raw model)
- `constrained_direct` on `<g>_syntactic` — syntactic pruning only (typing rules stripped)
- `constrained_direct` on `<g>` — full semantic pruning

Set `tasks.pruning_twins = true` in the config. For every selected task on a
study grammar the runner emits a twin task whose grammar is the typing-stripped
`<g>_syntactic` (same prompt, initial, expected and oracle). The only thing that
differs between a record and its twin is the pruning. Every rejected completion
carries a `reject_reason` (`non_completable`, `parse_error:…`, `incomplete:…`,
`type_or_value:…`); `type_or_value` is the class only the type system catches.

Three pieces:

1. **GPU run** (relates pruning to model size, goal 3):
   ```bash
   python benchmarks/run.py --config benchmarks/configs/pruning_study.toml --resume
   ```
   Three models spanning the SAS capability range, on `imp` + `fun` (`toy` has no
   corpus tasks, `ml` is out of scope). Constrained records also store per-step
   `step_pre_entropies` / `step_entropies` (mask entropy before/after pruning).

2. **Report** (pass rate per regime + rejection breakdown):
   ```bash
   python benchmarks/pruning_report.py benchmarks/out/pruning-study --json report.json
   ```

3. **Static measurement** (model-free, no GPU; relates pruning to a language's
   typing power, goals 1–2). Walks the same corpus and counts, per prefix, how
   many candidates each grammar admits (`Synthesizer.mask`); semantic admits are
   always a subset of syntactic. Emits per-grammar aggregates plus typing-power
   descriptors and a per-position dataset:
   ```bash
   python benchmarks/pruning_static.py
   # -> benchmarks/out/pruning-study/static/{summary.json,positions.jsonl}
   ```

## Artifact Docker

Build the reviewable Docker image tarball and source bundle:

```bash
make artifact
```

The image has no config-specific entrypoint. Invoke the runner explicitly:

```bash
docker load -i dist/proposition7-benchmark-artifact.tar
docker run --rm proposition7-benchmark-artifact:latest \
  python benchmarks/run.py --config benchmarks/configs/sas26_reproduction.toml --dry-run
```

Run the SAS reproduction on a GPU host with output and model-cache mounts:

```bash
printf 'OPENROUTER_API_KEY=...\n' > .env
mkdir -p artifact-output hf-cache
docker run --rm --gpus all \
  --env-file .env \
  -v "$PWD/artifact-output:/workspace/benchmarks/out" \
  -v "$PWD/hf-cache:/cache/huggingface" \
  proposition7-benchmark-artifact:latest \
  python benchmarks/run.py --config benchmarks/configs/sas26_reproduction.toml --resume
```

Hugging Face models are downloaded into the mounted `hf-cache/` directory at run
time. They are intentionally not baked into the Docker image.

Run any included config by changing `--config`. For local-only configs, omit
`--env-file .env`; for OpenRouter-only configs, omit `--gpus all`.


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

## Validation

```bash
pytest -q tests/benchmarks_api.py
pytest -q
```
