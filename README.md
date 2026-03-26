# p7 (Python bindings)

**Truth by Construction: From Natural Language to Logic Machines Through Formal Constraints.**

Python bindings for the [Proposition 7](https://unsuspicious.org/blog/proposition-7) constrained generation system, built on the [Aufbau Rust core](https://github.com/Unsuspicious-Industries/aufbau).

## Installation

### Using pip (from source)

```bash
pip install maturin
maturin develop --target-dir target
```

Note: avoid `maturin develop --inplace` so the extension stays in `target/` instead of the package directory.

### Using Nix

```bash
nix develop
maturin develop --target-dir target
```

## Quick Start

```python
import p7

# Load a model with a built-in typed grammar
model = p7.ConstrainedModel.from_pretrained(
    "gpt2",
    grammar=p7.get_grammar("fun"),
)

# Generate constrained tokens one at a time
gen = model.iter_constrained(
    prompt="Build a function that adds two numbers:\n",
    initial="let add: Int -> Int -> Int = (x: Int) =>",
    max_tokens=30,
    stop_on_complete=True,
    grammar_name="fun",
)
for token in gen:
    print(token, end="", flush=True)
```

## Demo (Visualization)

The `visualization/` app provides a grammar editor, debugger, and constrained generation UI.

### Quick launch

```bash
cd visualization
./start.sh
```

### Manual setup

```bash
maturin develop --target-dir target
pip install flask flask-cors transformers torch

cd visualization/demo
npm install

cd ../api
python app.py
```

In a second terminal:

```bash
cd visualization/demo
npm start
```

The backend runs on `http://localhost:5001` and the demo UI on `http://localhost:3000`.

## Using Built-in Grammars

All built-in grammars include **typing rules** - this is what distinguishes P7 from CFG-only approaches.

```python
import p7

# List available typed grammars
print(p7.list_grammars())  # ['stlc', 'imp', 'fun', 'toy', 'json']

# FUN: typed functional expressions (ML-style)
fun_engine = p7.CompletionEngine(p7.get_grammar("fun"))
fun_engine.feed("let x: Int = 1; x +")
print(fun_engine.get_completions()[:5])

# IMP: typed imperative programs
imp_engine = p7.CompletionEngine(p7.get_grammar("imp"))
imp_engine.feed("{ let x: Int = 1; if (x < 5) { let y: Int = x + 1; } else { let y: Int =")
print(imp_engine.get_completions()[:5])

# Available grammars:
# - stlc: simply typed lambda calculus
# - fun:  typed functional language (let/lambda/application)
# - imp:  typed imperative language (assign/if/while)
# - toy:  toy typed language (beep/boop)
```

### Low-level API

```python
import p7

# Direct grammar + synthesizer access
grammar = p7.Grammar("...")
synth = p7.Synthesizer(grammar, "")

synth.set_input("42")
print(synth.get_completions())  # List of valid next patterns
print(synth.is_complete())      # Check if parse is complete
print(synth.current_text())     # Text fed so far
```

## Grammar Format

The p7 grammar format supports:

- **Productions**: `Name ::= symbol1 symbol2 | alternative`
- **Literals**: `'hello'` or `"world"`
- **Regex patterns**: `/[a-z]+/`
- **Bindings**: `Expr[x]` to name subexpressions
- **Typing rules**: Inference-style rules for type checking

### Example: Simply Typed Lambda Calculus

```
Identifier ::= /[a-z][a-zA-Z0-9]*/
Variable(dec) ::= Identifier[x]
Lambda(lambda) ::= 'λ' Identifier[x] ':' Type[τ] '.' Expression[e]
Application(app) ::= Expression[e1] Expression[e2]

BaseType ::= Identifier[τ]
FunctionType ::= Type[τ1] '->' Type[τ2]
Type ::= BaseType | FunctionType | '(' Type ')'

Expression ::= Variable | Lambda | Application | '(' Expression ')'

x ∈ Γ
----------- (dec)
Γ ⊢ Γ(x)

Γ[x:τ1] ⊢ e : τ2
----------------------- (lambda)
Γ ⊢ τ1 → τ2

Γ ⊢ e1 : τ1 → τ2,   Γ ⊢ e2 : τ1
-------------------------------- (app)
Γ ⊢ τ2
```

## HuggingFace Transformers Integration

The `ConstrainedModel` class handles HuggingFace integration automatically:

```python
import p7

# GPT-2 (default, small)
model = p7.ConstrainedModel.from_pretrained("gpt2", grammar=p7.get_grammar("fun"))

# Phi-3.5-mini
model = p7.ConstrainedModel.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct",
    grammar=p7.get_grammar("stlc"),
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)

# Any causal LM
model = p7.ConstrainedModel.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    grammar=p7.get_grammar("imp"),
)
```

### Generating

```python
import p7

model = p7.ConstrainedModel.from_pretrained("gpt2", grammar=p7.get_grammar("fun"))

# Generate up to max_tokens, streaming tokens via on_token callback
result = model.generate(
    prompt="Write a function:\n",
    initial="let f: Int -> Int = (x: Int) =>",
    max_tokens=30,
    greedy_k=1,         # k=1 is pure greedy; k>1 samples among top-k valid
    pre_top_k=100,      # candidate pool size before grammar filtering
    grammar_name="fun",
    on_token=lambda tok, step: print(tok, end="", flush=True),
)
print()
print(f"Complete: {result.is_complete}, reason: {result.stopped_reason}")

# Generate until the parse is complete (or max_tokens)
result = model.until_complete(
    prompt="Write a function:\n",
    initial="let f: Int -> Int = (x: Int) =>",
    max_tokens=100,
    grammar_name="fun",
)

# Iterate tokens manually
gen = model.iter_constrained(
    prompt="Write a function:\n",
    initial="(x: Int) =>",
    max_tokens=20,
    stop_on_complete=True,
    grammar_name="fun",
)
for token in gen:
    print(token, end="", flush=True)
```

### Reasoning Environment (CoT + Grammar)

```python
import p7

model = p7.ConstrainedModel.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct",
    grammar=p7.get_grammar("stlc"),
    device_map="auto",
    trust_remote_code=True,
)

env = p7.ReasoningEnvironment(
    model=model,
    grammar_name="stlc",
    think_budget=150,   # max free-form reasoning tokens per block
    formal_budget=50,   # max constrained grammar tokens per block
)

result = env.generate(
    prompt="Create the identity function for Int",
    initial="λx:",
    max_blocks=4,
    start_thinking=True,
)
print(result.final_output.content)  # e.g. λx:Int.x
print(f"Think blocks: {len(result.think_blocks)}")
print(f"Grammar blocks: {len(result.grammar_blocks)}")
```

## API Reference

### `Grammar`

```python
grammar = p7.Grammar(spec_string)
```

Create a grammar from a spec string.

### `Synthesizer`

```python
synth = p7.Synthesizer(grammar, initial_text)
synth.set_input(text)             # Replace current text with a full prefix
synth.extend(token)               # Extend by one token
synth.get_completions()           # List[str] of valid next patterns
synth.is_complete()               # True if parse is at a complete state
synth.current_text()              # Text fed so far
```

### `CompletionEngine`

High-level wrapper around `Synthesizer`:

```python
engine = p7.CompletionEngine(grammar_str)
engine.feed(text)           # Replace current text with a full prefix
engine.reset()              # Reset to empty state
engine.get_completions()    # List[str] of valid next patterns
engine.current_text()       # Text fed so far
```

### `TypedSampler`

Filters LLM logits to only grammar-valid tokens:

```python
sampler = p7.TypedSampler(
    grammar=grammar_str,
    vocab=list_of_token_strings,
    logit_fn=callable_returning_list_of_floats,
)
sampler.set_input(text)                    # Replace current text with a full prefix
sampler.feed(token)                        # Extend by one token
sampler.reset()                            # Reset state
sampler.is_complete()                      # Check completion
sampler.infer(pre_top_k=None)              # Masked logits (-inf for invalid)
sampler.infer_text(k=10, pre_top_k=None)  # Top-k valid token strings
sampler.infer_greedy(k=1, pre_top_k=None) # Pick one token (greedy or sampled)
sampler.current_text()                     # Text fed so far
```

### `ConstrainedModel`

```python
model = p7.ConstrainedModel.from_pretrained(
    model_name,      # HuggingFace model ID
    grammar,         # Grammar spec string (use p7.get_grammar(...))
    device="cpu",    # Device (overridden by device_map if provided)
    **model_kwargs,  # Forwarded to AutoModelForCausalLM.from_pretrained
)

# Constrained generation methods
result = model.generate(prompt, initial, max_tokens, greedy_k, pre_top_k,
                        on_token, grammar_name, logit_filter)
result = model.until_complete(prompt, initial, max_tokens, greedy_k, pre_top_k,
                               on_token, grammar_name, logit_filter)
gen    = model.iter_constrained(prompt, initial, max_tokens, greedy_k, pre_top_k,
                                 stop_on_complete, grammar_name, logit_filter)

# Unconstrained generation (raw model output)
result = model.generate_unconstrained(prompt, initial, max_tokens, top_k,
                                       temperature, on_token, stop_tokens, grammar_name)
gen    = model.iter_unconstrained(prompt, initial, max_tokens, top_k,
                                   temperature, stop_tokens, grammar_name)
```

`GenerationResult` fields:

```python
result.text             # str: full generated text (initial + generated)
result.is_complete      # bool: parse reached a complete state
result.tokens_generated # int: number of tokens generated
result.stopped_reason   # str: "complete" | "max_tokens" | "no_valid" | "type_error: ..."
```

### Grammar Utilities

```python
p7.list_grammars()              # List[str] of built-in grammar names
p7.get_grammar(name)            # str: grammar spec for the given name
p7.get_grammar_info(name)       # dict with spec, name, description, examples
p7.GRAMMARS                     # dict mapping name -> grammar info dict
```

### Utility Functions

```python
p7.regex_matches(pattern, text)        # Check if text matches regex
p7.regex_prefix_valid(pattern, prefix) # Check if prefix is valid for regex
```

## Practical Takeaways

- **Practitioners**: Use constraints to eliminate invalid states; devote model capacity to style and strategy.
- **Decision-makers**: Expect better efficiency (less wasted compute) and better interpretability and traceability of the process.
- **Researchers**: Treat interesting-ness as a learnable value function over valid moves; compare signals (structural, graph, human preference).

## Related Work

This approach draws inspiration from:

- [Type-Constrained Code Generation with Language Models](https://arxiv.org/abs/2504.09246) (Mündler et al., 2025) — type-aware constraint generation for LLM code synthesis
- Attribute grammars (Knuth, 1968) and bidirectional type checking (Pierce & Turner, 2000)
- Martin-Löf's Intuitionistic Type Theory
- Wittgenstein's *Tractatus Logico-Philosophicus*

| Feature | proposition-7 | Type-Constrained (arXiv:2504.09246) | Outlines/.txt |
|---------|----------------|-------------------------------------|---------------|
| Type-aware | ✅ | ✅ | ❌ |
| Custom languages | ✅ Any `.auf` grammar | ❌ TypeScript/STLC only | ❌ |
| Declarative rules | ✅ Inference-style | ❌ Hardcoded | ❌ |
| Regex tokens | ✅ | Partial | ✅ |
| Persistent state | ✅ | ✅ | ❌ |

## References

### Philosophy & Foundations
- Carnap, R. (1937). *The Logical Syntax of Language*.
- Wittgenstein, L. (1921). *Tractatus Logico-Philosophicus*.
- Kant, I. (1790). *Critique of Judgment*.

### Core Type Theory
- Martin-Löf, P. (1984). Intuitionistic Type Theory.
- Girard, J.-Y. (1972). System F / cut elimination.
- Barendregt, H. (1992). Lambda calculi with types.
- Wadler, P. (2015). Propositions as Types.

### Constrained Generation & Synthesis
- Muendler, N., et al. (2025). Type-aware constraint generation for LLM code synthesis.
- Willard, B., & Louf, R. (2023). Efficient guided generation for large language models.
- Polikarpova, Kuraj, & Solar-Lezama (2016). Refinement type synthesis.
- Knuth, D. E. (1968). Semantics of context-free languages.
- Pierce, B. C., & Turner, D. N. (2000). Local type inference.

## License

MIT

---

*The project is named after proposition 7 in Wittgenstein's Tractatus: "Whereof one cannot speak, thereof one must be silent" (Wovon man nicht sprechen kann, darüber muß man schweigen). Our system aims to clearly delineate what can be formally expressed versus what belongs to the realm of the unsayable.*
