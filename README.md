# proposition7

Type-aware constrained generation for language models. Install and import the package as `proposition7`.

## Install

```bash
pip install -e .
pip install -e ".[transformers]"  # local Hugging Face generation
```

## Local Generation

```python
import proposition7

model = proposition7.ConstrainedModel.from_pretrained(
    "gpt2",
    grammar=proposition7.get_grammar("fun"),
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
result = proposition7.generate(
    "identity function",
    model="gpt2",
    grammar="stlc",
    initial="λx:Int.",
    max_tokens=20,
)
```

## Benchmarks

Build the review artifact through the unified Makefile:

```bash
make artifact
```

This produces a reusable Docker image tarball, a source bundle, and a manifest
under `dist/`. The image contains the repository snapshot, all benchmark configs,
and Python dependencies, but it does not contain API keys, model weights, model
caches, benchmark outputs, or `backup/`.

The image has no config-specific entrypoint. Run the benchmark command you want
explicitly with `python benchmarks/run.py --config ...`.

Dry-run the SAS 2026 paper-reproduction config without GPU or API access:

```bash
docker load -i dist/proposition7-benchmark-artifact.tar
docker run --rm proposition7-benchmark-artifact:latest \
  python benchmarks/run.py --config benchmarks/configs/sas26_reproduction.toml --dry-run
```

Run the SAS 2026 paper-reproduction config on a GPU host:

```bash
docker load -i dist/proposition7-benchmark-artifact.tar
mkdir -p artifact-output hf-cache
docker run --rm --gpus all \
  --env-file .env \
  -v "$PWD/artifact-output:/workspace/benchmarks/out" \
  -v "$PWD/hf-cache:/cache/huggingface" \
  proposition7-benchmark-artifact:latest \
  python benchmarks/run.py --config benchmarks/configs/sas26_reproduction.toml --resume
```

Create `.env` with `OPENROUTER_API_KEY=...` before running configs that include
OpenRouter rows. Hugging Face models are downloaded at run time into the mounted
`hf-cache/` directory, not baked into the Docker image.

Run any other included config by changing only `--config`. To test an edited
config or source checkout, mount it over `/workspace` and keep the output/cache
mounts.

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
  proposition7/            # published Python package
    __init__.py            # public API exports
    api.py                 # high-level generate() and Session
    llm.py                 # ConstrainedModel
    inference.py           # low-level constrained loop
    environment.py         # optional reasoning environment
    grammars/              # bundled .auf grammar specs
    models/                # model-specific adapters
benchmarks/
  api.py                   # backend-neutral benchmark interaction API
  run.py                   # TOML-driven benchmark artifact runner
  configs/                 # benchmark configs
  agg.py                   # result aggregation
artifact/                  # Docker artifact image files
scripts/
  build_artifact_image.sh  # builds dist/proposition7-benchmark-artifact.tar
tests/                     # pytest suite
```

## Test

```bash
nix develop path:. -c make build
nix develop path:. -c make test
```
