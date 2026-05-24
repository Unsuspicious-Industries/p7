# proposition7

The `proposition7` package is a constrained decoding interface for the `aufbau` prefix parsing engine. 
It implements type-aware constrained generation and the benchmark runner used to
reproduce the reported artifact results.

This README covers local installation,
local generation, Docker artifact checks, benchmark reproduction, expected
outputs, and validation commands.

Benchmark runs are configured by TOML files, typically stored in `benchmarks/configs`. More information about the config format is given in the benchmark [README](./benchmarks/README.md)
Configs are used for ensuring the reproducibility of benchmarks runs. Running a config twice should yield the same results (without taking into account runtime numerical difference and potentially non-deterministic sampling).
The benchmarks outputs are by default in the `benchmarks/out/<run-name>` folder, where run name depends on the config used. Config name is defined in the top of the TOML file. 

## What Is Included

- Python package `proposition7` with constrained local Hugging Face generation.
- Built-in typed grammars: `stlc`, `fun`, `imp`, and `toy`.
- Benchmark task suite under `benchmarks/data/`.
- TOML benchmark configs under `benchmarks/configs/`.
- Docker artifact build scripts under `artifact/` and `scripts/`.
- Tests that exercise the README-facing public API and benchmark runner.

The artifact does not include API keys, model weights, Hugging Face caches,
benchmark outputs, or local backups.

## Requirements

### Python

Use Python `>=3.9,<3.13`. Python 3.13 is not supported.

Recommended reviewer setup:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
```

### Optional Runtime Requirements

| Use case | Requires GPU | Requires API key | Notes |
| --- | --- | --- | --- |
| Import package and load grammars | No | No | `pip install -e .` is enough. |
| Run tests without real HF models | No | No | Use `make test` in the artifact environment. |
| Run local `gpt2` CPU example | No | No | Requires `.[transformers]`; under 2 GB RAM. |
| Docker dry-run | No | No | Validates benchmark planning only. |
| Small local benchmark | Recommended | No | Use `mini-4b.toml`; CPU is possible but slow. |
| Full SAS reproduction | Yes | Yes | Local model rows need GPU and OpenRouter rows need `OPENROUTER_API_KEY`. |

For local Hugging Face generation, memory usage is dominated by the selected
model. Plan for roughly `2.2 * model_size ` RAM or VRAM. 
The full reproduction includes models much larger than 4B parameters, so
it is intended for a GPU server rather than a laptop. Only CUDA has been tested, so MPS
and CPU-only full reproduction are not recommended.

## Install From Source

From the repository root:

```bash
pip install -e .
```

This is sufficient for grammar loading and non-generation package checks:

```bash
python - <<'PY'
import proposition7

print(proposition7.list_grammars())
print(len(proposition7.get_grammar("fun")) > 100)
PY
```

Expected output:

```text
['stlc', 'imp', 'fun', 'toy']
True
```

For local Hugging Face generation, install the `transformers` extra:

```bash
pip install -e ".[transformers]"
```

For Outlines comparison modes, install the `outlines` extra:

```bash
pip install -e ".[outlines]"
```

Outlines is still in experimental phase. It has not been thouroughly tested and will probaly fail.

## Local Generation

The model object owns one grammar spec. Pass the grammar once when loading the
model. Generation calls should not pass a separate `grammar_name`.

### CPU Example

This is the reviewer-facing smoke test. It runs on CPU with `gpt2`; the exact
program text is model-dependent, but the command should not raise an exception.

```python
import proposition7

grammar = proposition7.get_grammar("fun")
model = proposition7.ConstrainedModel.from_pretrained(
    "gpt2",
    grammar=grammar,
    device="cpu",
)

result = model.generate_constrained(
    prompt="Define inc:Int->Int and call it on 1. Output only program text.",
    initial="let inc: Int -> Int = (n: Int) =>",
    max_tokens=64,
)

print(result.text)
print(result.is_complete)
print(result.stopped_reason)
```

Expected behavior:

- `result.text` starts with the provided `initial` prefix.
- `result.stopped_reason` is one of `complete`, `max_tokens`, `no_valid`, or a
  `stop_token:...` reason.

Because `gpt2` is a tiny non-code model, do not use this example to judge paper
performance. It is a local installation and constrained-decoding smoke test.

### GPU Example

Use this only on a CUDA GPU host with enough VRAM for the selected model.

```python
import proposition7

model = proposition7.ConstrainedModel.from_pretrained(
    "Qwen/Qwen3.5-4B",
    grammar=proposition7.get_grammar("stlc"),
    device="cuda",
    torch_dtype="auto",
)

result = model.generate_constrained(
    prompt="Write the identity function in STLC",
    initial="λx:Int.",
    max_tokens=64,
)

print(result.text)
print(result.is_complete)
print(result.stopped_reason)
```

### Unconstrained Generation

The same model object exposes plain generation:

```python
result = model.generate_unconstrained(
    prompt="Write a short typed function example.",
    max_tokens=64,
    top_k=50,
    temperature=0.8,
)
```

### High-Level Helper

The convenience API loads a model, runs constrained generation, and returns a
small `Result` object:

```python
import proposition7

result = proposition7.generate(
    "identity function",
    model="gpt2",
    grammar="stlc",
    initial="λx:Int.",
    max_tokens=20,
)

print(result.text)
print(result.complete)
```

## Docker Artifact

The canonical image is on GHCR. See also the
[`sas-artifact-v2` release page](https://github.com/Unsuspicious-Industries/p7/releases/tag/sas-artifact-v2).

```bash
docker pull ghcr.io/unsuspicious-industries/proposition7-benchmark-artifact:latest
docker tag  ghcr.io/unsuspicious-industries/proposition7-benchmark-artifact:latest \
            proposition7-benchmark-artifact:latest
```

The retag lets the `docker run` commands below use the short name.

To build from source instead:

```bash
make artifact
docker load -i dist/proposition7-benchmark-artifact.tar
```

The image contains the repository snapshot, benchmark configs, and Python
dependencies. It does not contain API keys, model weights, model caches,
benchmark outputs, or local backups.

Replace `docker` with `podman` if needed.

## Dry-Run The Paper Config

The dry-run validates that the benchmark config, task selection, matrix
expansion, and resume planning work. It does not download models or execute any
generation jobs.

```bash
docker run --rm proposition7-benchmark-artifact:latest \
  python benchmarks/run.py --config benchmarks/configs/sas26_reproduction.toml --dry-run
```

Expected output for a fresh run directory:

```text
[matrix] fig3-core-local: backend=local models=7 modes=constrained_direct,constrained_mixed,unconstrained planned=5922 pending=5922
[matrix] fig7-frontier-constrained: backend=local models=4 modes=constrained_mixed planned=1128 pending=1128
[matrix] fig7-openrouter-raw: backend=openrouter models=3 modes=unconstrained planned=846 pending=846
[matrix] text-reference-openrouter-raw: backend=openrouter models=1 modes=unconstrained planned=282 pending=282
[plan] run=sas26-reproduction tasks=94 matrices=4 total_jobs=8178 pending_jobs=8178 run_dir=benchmarks/out/sas26-reproduction
```

If you run with `--resume` after a partial run, `pending` and `pending_jobs` may
be smaller because completed records are skipped.

Matrix meaning:

- `fig3-core-local`: local open-model rows used for the Figure 3 core comparison
  and Figure 4 aggregate comparison.
- `fig7-frontier-constrained`: larger local constrained rows used in the Figure
  7 leaderboard.
- `fig7-openrouter-raw`: closed/frontier OpenRouter raw baseline rows for Figure
  7 leaderboard.
- `text-reference-openrouter-raw`: reference for 100% pass rate with GPT-5.3-codex

## Run A Small GPU Benchmark

Before launching the full reproduction, run a smaller local-only config on a GPU
server. This checks real model loading, constrained generation, benchmark output,
and aggregation without needing an API key.

```bash
mkdir -p artifact-output hf-cache
docker run --rm --gpus all \
  -v "$PWD/artifact-output:/workspace/benchmarks/out" \
  -v "$PWD/hf-cache:/cache/huggingface" \
  proposition7-benchmark-artifact:latest \
  python benchmarks/run.py --config benchmarks/configs/mini-4b.toml --resume
```

Process the smoke run:

```bash
docker run --rm \
  -v "$PWD/artifact-output:/workspace/benchmarks/out" \
  proposition7-benchmark-artifact:latest \
  python benchmarks/agg.py \
    --in benchmarks/out/mini-4b/raw.jsonl \
    --out-dir benchmarks/out/mini-4b/processed
```

Expected output files after the run and aggregation:

```text
artifact-output/mini-4b/config.toml
artifact-output/mini-4b/raw.jsonl
artifact-output/mini-4b/results.json
artifact-output/mini-4b/processed/summary_by_model_mode.csv
artifact-output/mini-4b/processed/summary_by_task.csv
artifact-output/mini-4b/processed/benchmark_records.csv
artifact-output/mini-4b/processed/process_stats.json
```

## Run The Full SAS Reproduction

The full reproduction contains local GPU rows and OpenRouter rows.

### OpenRouter API Key

Only `sas26_reproduction.toml` has OpenRouter rows and needs an API key.
`mini-4b.toml`, `small-models.toml`, and `simple.toml` are local-only; run them
without `--env-file .env`.

```bash
printf 'OPENROUTER_API_KEY=YOUR_KEY_HERE\n' > .env
mkdir -p artifact-output hf-cache
```

The runner reads `.env` via the `--env-file` flag (path configurable in
`[openrouter] env_file`).

### Run

Run the full reproduction on a GPU host:

```bash
docker run --rm --gpus all \
  --env-file .env \
  -v "$PWD/artifact-output:/workspace/benchmarks/out" \
  -v "$PWD/hf-cache:/cache/huggingface" \
  proposition7-benchmark-artifact:latest \
  python benchmarks/run.py --config benchmarks/configs/sas26_reproduction.toml --resume
```

Expected primary outputs:

```text
artifact-output/sas26-reproduction/config.toml
artifact-output/sas26-reproduction/raw.jsonl
artifact-output/sas26-reproduction/results.json
```

Aggregate the completed run into CSV summaries:

```bash
docker run --rm \
  -v "$PWD/artifact-output:/workspace/benchmarks/out" \
  proposition7-benchmark-artifact:latest \
  python benchmarks/agg.py \
    --in benchmarks/out/sas26-reproduction/raw.jsonl \
    --out-dir benchmarks/out/sas26-reproduction/processed
```

Expected processed outputs include:

```text
artifact-output/sas26-reproduction/processed/summary_by_model_mode.csv
artifact-output/sas26-reproduction/processed/summary_by_model_language.csv
artifact-output/sas26-reproduction/processed/summary_by_mode_language.csv
artifact-output/sas26-reproduction/processed/summary_by_task.csv
artifact-output/sas26-reproduction/processed/delta_by_model_language.csv
artifact-output/sas26-reproduction/processed/benchmark_records.csv
artifact-output/sas26-reproduction/processed/process_stats.json
```

## Configs

Included configs:

- `benchmarks/configs/sas26_reproduction.toml`: full paper-reproduction plan.
- `benchmarks/configs/mini-4b.toml`: smaller local GPU smoke config.
- `benchmarks/configs/small-models.toml`: local small-model sweep.
- `benchmarks/configs/simple.toml`: broader local comparison config.

Run any included config by changing only `--config`. For local-only configs,
omit `--env-file .env`. For OpenRouter-only configs, omit `--gpus all`.

## Validation

The fastest full artifact validation is:

```bash
make test
make artifact
docker load -i dist/proposition7-benchmark-artifact.tar
docker run --rm proposition7-benchmark-artifact:latest python -m pytest -q
docker run --rm proposition7-benchmark-artifact:latest \
  python benchmarks/run.py --config benchmarks/configs/sas26_reproduction.toml --dry-run
```

If you are validating from a source checkout instead of the artifact image:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q tests/api.py tests/grammar.py tests/benchmarks_api.py
python -m pip install -e ".[transformers]"
python -m pytest -q tests/generation_smoke.py
python benchmarks/run.py --config benchmarks/configs/sas26_reproduction.toml --dry-run
```

The generation smoke tests in `tests/generation_smoke.py` exercise the full
constrained decoding stack with a real `gpt2` model on CPU. This verifies model
loading, grammar constraining, token sampling, and output validation without
needing a GPU.

## Built-In Grammars

| Name | Language |
| --- | --- |
| `stlc` | Simply typed lambda calculus |
| `fun` | ML-style functional expressions |
| `imp` | Typed imperative programs |
| `toy` | Small typed toy grammar |

Pass a grammar name through high-level APIs such as `proposition7.generate()`,
or pass a raw grammar spec to `ConstrainedModel.from_pretrained()`.

## Public API Summary

- `proposition7.ConstrainedModel.from_pretrained(model, grammar=...)`: load a
  local Hugging Face model with one grammar spec.
- `model.generate_constrained(prompt=..., initial=..., max_tokens=...)`: run
  grammar-constrained decoding.
- `model.generate_unconstrained(prompt=..., max_tokens=..., top_k=...)`: run
  standard sampling.
- `proposition7.Session(...)`: reusable high-level session.
- `proposition7.generate(...)`: one-shot high-level helper.

The grammar is attached to the model/session at construction time. Generation
calls do not take `grammar_name`.

## Project Layout

```text
src/
  proposition7/
    __init__.py            # public API exports
    api.py                 # high-level generate() and Session
    llm.py                 # ConstrainedModel
    inference.py           # GenerationResult and low-level types
    environment.py         # optional reasoning environment
    grammars/              # bundled .auf grammar specs
    models/                # model-specific adapters
benchmarks/
  api.py                   # backend-neutral benchmark interaction API
  run.py                   # TOML-driven benchmark artifact runner
  configs/                 # benchmark configs
  data/                    # benchmark tasks
  agg.py                   # result aggregation
artifact/                  # Docker artifact image files
scripts/
  build_artifact_image.sh  # builds dist/proposition7-benchmark-artifact.tar
tests/                     # pytest suite
```

## Troubleshooting

`pip install` on Python 3.13 fails:

Use Python 3.12 or another supported version in `>=3.9,<3.13`.

`ImportError: Local generation requires torch`:

Install the local generation extra with `pip install -e ".[transformers]"`.

`OPENROUTER_API_KEY is required`:

The selected config includes OpenRouter rows. Create `.env` with
`OPENROUTER_API_KEY=...`, or run a local-only config such as `mini-4b.toml`.

Docker reports no GPU:

Check that the NVIDIA container runtime is installed and that
`docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi`
works on the host.

Out-of-memory during a local model row:

Use a smaller config/model, reduce concurrency, or run on a larger GPU. The full
SAS reproduction is intended for a GPU server, not a laptop.
