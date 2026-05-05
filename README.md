# proposition7

Type-aware constrained generation for language models. The package is exported as
`proposition7` and keeps `p7` as a compatibility import.

## Install

```bash
pip install -e .
pip install -e ".[transformers]"  # local Hugging Face generation
pip install -e ".[modal]"         # Modal remote execution
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

## Modal Remote Generation

Create a `.env` file in the project root or current working directory:

```dotenv
MODAL_TOKEN_ID=your-token-id
MODAL_TOKEN_SECRET=your-token-secret
```

Deploy the Modal app once:

```bash
modal deploy -m proposition7.modal
```

Then call the deployed model through the same generation interface:

```python
import proposition7 as p7

remote = p7.ModalDeployment(
    model_name="gpt2",
    grammar="fun",
    gpu="T4",
)

result = remote.generate_constrained(
    prompt="Define inc:Int->Int and call it on 1. Output only program text.",
    initial="let inc: Int -> Int = (n: Int) =>",
    max_tokens=64,
)

print(result.text)
```

`ModalDeployment` loads `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` from `.env`
automatically. You can pass `env_path=...` if the file lives elsewhere.

## Benchmarks

Run a local smoke pass from one reproducible config:

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
- `proposition7.ModalDeployment`: remote Modal client with the same generation methods.
- `generate_constrained(...)`: constrained decoding, returning `GenerationResult`.
- `generate_unconstrained(...)`: standard sampling, returning `GenerationResult`.
- `proposition7.generate(...)`: high-level convenience function returning `Result`.

## Project Layout

```text
src/
  __init__.py              # p7 compatibility package
  api.py                   # high-level generate() and Session
  llm.py                   # ConstrainedModel
  modal_deployment.py      # Modal app and ModalDeployment
  sampler.py               # typed token sampler
  inference.py             # low-level constrained loop
  environment.py           # optional reasoning environment
  grammars/                # bundled .auf grammar specs
  models/                  # model-specific adapters
  proposition7/            # exported proposition7 alias package
benchmarks/
  api.py                   # backend-neutral benchmark interaction API
  run.py                   # Vast.ai/local benchmark runner
  models.py                # Vast.ai/local-GPU benchmark matrix runner
  agg.py                   # result aggregation
tests/                     # pytest suite
```

## Test

```bash
pytest -q
```
