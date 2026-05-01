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

Run a local smoke pass and aggregate results:

```bash
python benchmarks/run.py --dry --max-tasks 5
python benchmarks/run.py --tasks stlc --models gpt2 --modes constrained_direct,unconstrained_raw,unconstrained --tries 1 --device cpu --out benchmarks/out/smoke/raw.jsonl
python benchmarks/agg.py --in benchmarks/out/smoke/raw.jsonl --out-dir benchmarks/out/smoke
```

Run the full open-model matrix on a Vast.ai/local GPU instance:

```bash
python benchmarks/models.py \
  --tasks all \
  --models-file models.txt \
  --tries 1
```

`benchmarks/models.py` defaults to local CUDA execution, writes phase outputs to
`benchmarks/out/vast_model_matrix/`, writes the cross-mode report to
`benchmarks/out/vast_model_matrix/combined/`, and resumes by default. Resume
skips raw records with the same `(backend, model, task_id, task_hash,
resolution_hash, mode, try)` key and aggregation de-duplicates accidental
duplicate appends. Benchmark tasks are one TOML file per task in
`benchmarks/data/`; execution modes are code-owned.

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
