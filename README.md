# proposition7

Type-aware constrained generation for language models. The package is exported as
`proposition7` and keeps `p7` as a compatibility import.

## Install

```bash
pip install -e .
pip install -e ".[transformers]"  # local Hugging Face generation
pip install -e ".[modal]"         # Modal sandbox launcher
```

## Local Generation

```python
import proposition7 as p7

model = p7.ConstrainedModel.from_pretrained(
    "gpt2",
    grammar=p7.get_grammar("fun"),
    device="cpu",
)

result = model.generate_constrained(
    prompt="Define inc:Int->Int and call it on 1. Output only program text.",
    initial="let inc: Int -> Int = (n: Int) =>",
    max_tokens=64,
)

print(result.text)
print(result.is_complete)
```

The same model object also exposes unconstrained generation:

```python
result = model.generate_unconstrained(
    prompt="Write a short typed function example.",
    max_tokens=64,
    top_k=50,
    temperature=0.8,
)
```

The high-level helper remains available:

```python
result = p7.generate(
    "identity function",
    model="gpt2",
    grammar="stlc",
    initial="λx:Int.",
    max_tokens=20,
)
```

## Modal Sandbox Runs

Modal supports container-based execution directly from the artifact Dockerfile.
This repository uses `modal.Image.from_dockerfile(...)` plus `modal.Sandbox.create(...)`
instead of a custom Modal app/function wrapper.

After `modal setup`, launch the small Qwen smoke suite on one A10G with:

```bash
python scripts/modal_sandbox_run.py --config benchmarks/configs/modal_qwen_smoke.toml
```

That script copies `raw.jsonl` and `results.json` back to a local output directory.

## Benchmarks

Build the artifact Docker image tarball:

```bash
./scripts/build_artifact_image.sh
```

Run the containerized smoke suite:

```bash
docker load -i dist/p7-benchmark-artifact.tar
docker run --rm -v "$PWD/artifact-output:/workspace/benchmarks/out" p7-benchmark-artifact:latest smoke
```

Run the paper suite on a GPU host:

```bash
docker run --rm --gpus all \
  -v "$PWD/artifact-output:/workspace/benchmarks/out" \
  -v "$PWD/.env:/workspace/.env:ro" \
  p7-benchmark-artifact:latest paper --resume
```

Run directly from a source checkout:

```bash
python benchmarks/run.py --config benchmarks/configs/smoke.toml --dry-run
python benchmarks/run.py --config benchmarks/configs/smoke.toml
```

The runner creates a dedicated run directory, keeps the append-only `raw.jsonl`
format unchanged for compatibility, and writes a standardized `results.json`
artifact for reproducibility. Resume an interrupted run with:

```bash
python benchmarks/run.py --config path/to/benchmark.toml --resume
```

Benchmark tasks are one TOML file per task in `benchmarks/data/`; benchmark run
parameters live in the run config TOML. See `benchmarks/README.md` for the full
config schema and output layout.

## Grammars

Built-in grammars:

| Name | Language |
| --- | --- |
| `stlc` | Simply typed lambda calculus |
| `fun` | ML-style functional expressions |
| `imp` | Typed imperative programs |
| `toy` | Small typed toy grammar |

Pass a grammar name through high-level APIs, or pass a raw grammar spec to
`ConstrainedModel.from_pretrained`.

## Public API

- `proposition7.ConstrainedModel`: local Hugging Face model wrapper.
- `generate_constrained(...)`: constrained decoding, returning `GenerationResult`.
- `generate_unconstrained(...)`: standard sampling, returning `GenerationResult`.
- `proposition7.generate(...)`: high-level convenience function returning `Result`.

## Project Layout

```text
src/
  __init__.py              # p7 compatibility package
  api.py                   # high-level generate() and Session
  llm.py                   # ConstrainedModel
  sampler.py               # typed token sampler
  inference.py             # low-level constrained loop
  environment.py           # optional reasoning environment
  grammars/                # bundled .auf grammar specs
  models/                  # model-specific adapters
  proposition7/            # exported proposition7 alias package
benchmarks/
  api.py                   # backend-neutral benchmark interaction API
  run.py                   # TOML-driven benchmark artifact runner
  configs/                 # smoke and paper benchmark configs
  agg.py                   # result aggregation
artifact/                   # Docker artifact image files
scripts/
  build_artifact_image.sh   # builds dist/p7-benchmark-artifact.tar
  modal_sandbox_run.py      # launches artifact container on Modal A10G
tests/                     # pytest suite
```

## Test

```bash
pytest -q
```
