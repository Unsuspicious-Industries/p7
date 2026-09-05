# proposition7

Type-aware constrained generation for language models. The PyPI package is
`proposition7` (release 0.3.0) and supports Python 3.10 and newer.

## Install

```bash
pip install proposition7
pip install "proposition7[transformers]"  # local Hugging Face generation
```

For a source checkout, use `pip install -e .` (or `pip install -e
".[transformers]"`). The base package depends only on `aufbau-rs>=0.5,<0.6`;
model integrations remain optional through the `transformers` extra.

## Publishing 0.3.0

Build and publish the `aufbau-rs` 0.5 wheels first. From the sibling Rust
checkout, maturin builds the extension wheels from the Rust crate:

```bash
cd ../aufbau
maturin build --release --features extension-module -o dist/
# publish the resulting aufbau-rs 0.4 wheels to PyPI
```

Only after those wheels are available on PyPI should the `proposition7` 0.3.0
sdist and wheel be built and published. `import proposition7` stays lazy: the
optional model stack is loaded only when its exports are used. This release
changes no WIRT frozen API.

## Local Generation

```python
import proposition7

model = proposition7.ConstrainedModel.from_pretrained(
    "gpt2",
    grammar=proposition7.get_grammar("ml"),
    device="cpu",
)

result = model.generate_constrained(
    prompt="Define inc over ints and apply it to 1. Output only program text.",
    initial="let inc (n : int) : int =",
    max_tokens=64,
)

print(result.text)
print(result.is_complete)
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

## Grammars

Built-in grammars:

| Name | Language |
| --- | --- |
| `stlc` | Simply typed lambda calculus |
| `ml` | Typed OCaml subset |
| `c` | Typed C subset |
| `toy` | Small typed toy grammar |

Pass a grammar name through high-level APIs, or pass a raw grammar spec to
`ConstrainedModel.from_pretrained`.

## Public API

- `proposition7.ConstrainedModel`: local Hugging Face model wrapper.
- `generate_constrained(...)`: constrained decoding, returning `GenerationResult`.
- `proposition7.generate(...)`: high-level convenience function returning `Result`.

## Project Layout

```text
src/
  proposition7/            # published Python package
    api.py                 # high-level generation API
    llm.py                 # ConstrainedModel
    inference.py           # low-level constrained loop
    grammars/              # bundled .auf grammar specs
    models/                # model-specific adapters
examples/                  # small usage examples
tests/                     # pytest suite
```

## Test

```bash
nix develop path:. -c make build
PYTHONPATH=.:src ./.venv/bin/pytest tests/ -q
```

Torch-backed tests skip when the `transformers` extra is absent.
