# p7 Benchmark Suite

This suite is built for large Vast.ai/local-GPU runs. Modal is not part of the benchmark path.

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
- `closed_unconstrained`: OpenRouter closed-model unconstrained phase, launched by `benchmarks/models.py` through the `openrouter` backend.

Outlines is a constraint engine mode, not a backend. The backend remains local for open models.

## Vast.ai Run

Install dependencies:

```bash
pip install -e ".[transformers]"
pip install outlines  # only needed for --modes outlines / default matrix
```

Run the full default matrix:

```bash
python benchmarks/models.py --tries 1
```

The default matrix includes local open-model phases for `unconstrained`, `unconstrained_raw`, `constrained_direct`, `constrained_mixed`, and `outlines`, plus the OpenRouter closed-model unconstrained phase. Disable expensive/optional phases as needed:

```bash
python benchmarks/models.py --without-closed --without-mixed --without-outlines
```

Smoke run:

```bash
python benchmarks/models.py --models gpt2 --max-tasks 2 --max-tokens-override 16 --without-closed
```

Use a model file for large runs:

```bash
python benchmarks/models.py --models-file models.txt --tries 1
```

## Direct Runner

Run selected modes directly:

```bash
python benchmarks/run.py \
  --tasks stlc \
  --models gpt2 \
  --modes constrained_direct,outlines,unconstrained_raw,unconstrained \
  --device cuda \
  --device-map auto \
  --resume \
  --out benchmarks/out/manual/raw.jsonl
```

Run OpenRouter closed models directly:

```bash
python benchmarks/run.py \
  --backend openrouter \
  --models openai/gpt-4o-mini \
  --modes unconstrained \
  --openrouter-env .env \
  --resume
```

## Resume Safety

Resume is enabled by default in `benchmarks/models.py` and opt-in in `benchmarks/run.py` via `--resume`.

The resume key is hash-aware:

```text
(backend, model, task_id, task_hash, resolution_hash, mode, try)
```

Changing a TOML task or resolution will not reuse stale rows. Aggregation uses the same hash-aware key and keeps the latest duplicate record.

## Output

- `<phase>/raw.jsonl`: one record per benchmark attempt.
- `<phase>/summary.csv`: metrics by backend, model, mode, and grammar.
- `<phase>/summary_by_category.csv`: metrics by category.
- `combined/raw.jsonl`: merged phase records.
- `combined/delta.csv`: direct p7 constrained vs unconstrained deltas, preferring `unconstrained_raw` when present.
- `combined/report.md`: concise markdown summary.

## Validation

```bash
pytest -q tests/benchmarks_api.py
pytest -q
```
