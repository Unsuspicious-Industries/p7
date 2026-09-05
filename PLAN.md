# p7, plan

**Charter: p7 is generic.** It is a library for building constrained-generation
systems. It ships no agent language, no evaluator, no Γ, no session, no
capabilities, and no wire protocol.

`../ARCHITECTURE.md` is canonical. This file is what p7 does about it.

**Benchmark boundary:** p7 ships no benchmark package and no TASK suite. Runs,
provenance, paper reproduction, and canonical summaries belong to the standalone
`../benchmarks` harness, which also renders figures and tables from those
summaries. It is not a p7 runtime dependency.

> **Priority: wirt ships first, and p7 must not break it.**
>
> wirt owns the only shippable artifact and is green today. Its entire p7
> dependency, verified 2026-08-19, is `proposition7.api` (`generate`,
> `verify`, `generate_pair`, `Result`) plus the `proposition7.runtime` protocol.
> `api._via_runtime` is that path; its docstring already says "this is how wirt
> serves".
>
> **Treat that surface as frozen while the demo is in flight.** The work below
> is disjoint from it by construction: satz is not imported by wirt, and
> `api.Session` (§2b) is a separate object from `_via_runtime`. If a change
> starts to touch `generate`/`verify`/`Result` or the `Runtime` protocol, stop
> and consult wirt first, that is the one coupling between these tracks.

## 1. What p7 is after this revision

Two halves, both generic:

```
proposition7            logits ↔ aufbau: decode loop, tokenizer, SPG cache,
                        generate() / verify() over raw .auf
proposition7.scheme     declaration → .auf. The generic grammar builder.
```

`proposition7.scheme` knows the *shape* of a primitive declaration. It does not
know which primitives exist, what they mean, or what they do. Those are gamma's
(`../gamma/PLAN.md`).

## 2. No turn logic in proposition7

**The rule, stated once and without exception: p7 has no concept of a turn.**

A turn is a session-level idea, it needs Γ, history, a prompt layout, a notion
of what came before and what commits after. p7 decodes one constrained
completion and returns it. Everything that decides *what to decode next*, or
*what a completion means to a conversation*, is gamma's.

Concretely, p7 must not contain: a session that accumulates state across calls,
a conversation history, Γ, a prompt layout, a think-then-formal staging policy,
a retry-and-continue loop above the token level, or a "mode" describing how a
turn is shaped.

This is broader than the satz split, and it is the whole point of this revision.
Three separate violations exist today.

### 2a. `proposition7.satz`, four things under one name

| current | lines | becomes |
|---|---|---|
| `satz/scheme.py` | 140 | `proposition7/scheme/declaration.py`, stays, generic |
| `satz/grammar.py` | 136 | `proposition7/scheme/compose.py`, stays, generic |
| `satz/evaluator.py` | 247 | → `gamma/runtime/evaluator.py` |
| `satz/turn.py` | 133 | → `gamma/runtime/session.py` |
| `satz/context.py` | 51 | → `gamma/runtime/context.py` (Γ) |
| `satz/result.py` | 44 | → `gamma/runtime/result.py` |

276 lines stay and are the deliverable. 475 lines move. The `satz` name is
retired entirely, it named a bundle that should never have been one.

**Why the evaluator cannot stay.** It interprets a language that does not exist:
D3 is unresolved, so `LanguageBinding.core_source` is a hole every caller fills.
Keeping an interpreter for an unwritten grammar inside a generic library is the
exact coupling this revision removes.

`turn.py` is the clearest case: it owns session identity, history, Γ, the prompt
layout and approval. Every one of those is a turn concern.

### 2b. `api.py::Session`, a session that is not one

`Session` (`api.py:90`) holds a model and a grammar and calls itself a session,
but carries no Γ and no history. It is a **model handle** with a convenience
`generate()`. The name claims turn semantics the class does not have, which is
worse than either being honest.

It also smuggles in turn shaping: `generate(reason=True, think_budget=…)`
branches into `ReasoningEnvironment`, a two-stage think-then-formal policy,
which is exactly a decision about how to shape a turn.

- [ ] Rename to say what it is (a configured model handle), or delete it in
      favour of the module-level `generate()`/`verify()`.
- [ ] Remove the `reason=` branch. Staging is a caller's policy, not p7's.

### 2c. `environment.py`, 450 lines of turn policy

`ReasoningEnvironment` is the largest violation and the least obvious, because
it predates the agent language and reads as infrastructure.

It is a **two-stage turn**: think unconstrained to a budget, then emit
constrained formal output, tracking `Mode.THINK`/`Mode.FORMAT`, `ThinkBlock`,
`FormalBlock`, `all_thoughts`, `stopped_reason`. That is a policy about how a
model should be driven across stages, a turn shape. This policy does not belong
in p7; callers that need staged generation own it.

What stays in p7 after this: the decode loop, tokenizer integration, the
`Runtime` protocol, SPG caching, `generate()`, `verify()`, and
`proposition7.scheme`. One call in, one result out, no memory between calls.

## 3. Workstreams

### W1, carve out `proposition7.scheme`

- [ ] Create `src/proposition7/scheme/` with `declaration.py` (from
      `satz/scheme.py`) and `compose.py` (from `satz/grammar.py`).
- [ ] Export `Primitive`, `Param`, `Effect`, `Scheme`, `LanguageBinding`,
      `compose`, `validate_binding`, `nonterminal`, `CompositionError`.
- [ ] `LanguageBinding` keeps only what composition needs: `core_source`,
      `expression_nt`, `result_type`, `rule_prefix`. **The node-name fields
      (`statement_nt`, `identifier_nt`, `variable_nt`, `literal_nts`,
      `todo_nt`) are evaluator concerns and move to gamma**, they describe how
      to *walk* a tree, not how to build a grammar.
- [ ] Delete `src/proposition7/satz/` entirely. No shim, no re-export, no
      deprecation path.
- [ ] `proposition7.scheme` must import nothing from `proposition7.llm`,
      `.api`, `.models`, torch, or a runtime. Composition is pure text.

### W2, make the composer worth calling generic

Today's composer emits exactly one shape: `name(arg, arg)` with a flat typing
rule. That is enough for one language, which is why it was never generic in
practice. Making it a real utility is the actual deliverable of this revision.

- [ ] **Arity is not special-cased.** Verify against the engine that generated
      productions typecheck at 0, 1, 2, 3 and 4 parameters. The parked
      experiment reported 3-argument calls `dead`; STATE recorded this as an
      experiment bug (hardcoded `Call1`/`Call2` over a binary-product domain),
      not an engine limit, and `c.auf` typechecks 1–4 on the same engine.
      **Settle this with a property test before anything is built on top.**
- [ ] `Result[A, E]` must parse in a generated typing rule. The experiment used
      `Result[A;E]` and it is unknown whether the obstacle is authoring or
      engine. Determine which, and fix or document it.
- [ ] Composition is byte-deterministic: name-sorted primitives, key-sorted
      JSON. Assert it with a test that composes a shuffled scheme twice.
- [ ] `validate_binding()` returns every problem, not the first.
- [ ] Property-test over generated schemes: arbitrary names, arities, and type
      sources compose to source that aufbau loads, and every generated label
      has a matching typing rule (a label without one silently disables
      enforcement, see ARCHITECTURE D3).

### W3, inhabitation as a CI assertion (I5)

- [ ] Expose a checked helper so callers do not hand-roll it:
      `scheme.completeness(source) -> ('inhabited', []) | ('sound', [sorts])`.
- [ ] Any grammar p7 hands to a masking decoder asserts `('inhabited', [])` in
      CI. Not a review opinion.

### W4, decode loop and runtime protocol

Unchanged in shape; this revision does not touch the hot path.

- [ ] `Runtime` protocol stays p7's. wirt implements it (tinygrad); a future
      Rust runtime is a sibling implementation, not a rewrite of p7.
- [ ] torch stays an optional extra, imported where used, never at module
      scope. `decode.py` stays numpy.
- [ ] A stop token is honoured only at a complete prefix. Regression-tested.

### W5, the honest telemetry name

- [ ] `step_entropies` is not entropy over grammar-valid tokens. It is the
      distribution left after removing the candidates the retry loop actually
      refused, an upper bound, often *higher* than the pre-mask figure. Either
      rename it to say so or document it at the field. Do not let the demo quote
      it as "entropy after masking".

### W7, housekeeping

- [x] Evaluation harness, artifacts, and generated caches deleted; p7 retains
      only its library and unit/property tests.
- [x] `experiments/satz-lang/` deleted 2026-08-19. Its two findings (arity-3
      `dead`, `Result[A, E]` in a typing rule) are recorded in ARCHITECTURE D3
      and `STATE.md`, and are now gamma's to settle.
- [ ] `__version__` comes from package metadata, not a hardcoded string. It
      currently reports `0.1.0` against different package metadata.
- [x] Tests requiring optional torch extras remain skipped when torch is absent;
      the library suite has no evaluation-dependent collection failures.

## 4. Benchmark boundary

p7 supplies only the generic decode/runtime library. It does not own benchmark
runs, task data, grading, provenance, or summaries. The external harness invokes
WIRT over HTTP and owns those artifacts; LMPL only renders canonical summaries.
The stack-wide acceptance gate is [`../ARCHITECTURE.md` §8](../ARCHITECTURE.md);
keep p7's runtime surface compatible with WIRT while that gate is exercised.

## 5. Required properties

**Statelessness, the turn rule, made testable:**

- **Two identical `generate()` calls return identical results.** No memory
  between calls. This is the single property that makes "no turn logic"
  checkable rather than aspirational; assert it directly.
- No object in `proposition7` accumulates conversation state, history, Γ, or a
  binding environment across calls.
- No module defines a generation *mode*, *stage*, or *phase*.
- `grep -rn "history\|think_budget\|Mode\." src/proposition7/` returns nothing.

**Genericity:**

- `proposition7.scheme` imports no model, runtime, torch, or network module.
- Composition is a pure function of (scheme, binding) and byte-stable.
- Generated grammars load in aufbau at every arity 0–4.
- Every generated rule label has a matching typing rule.
- Every shipped grammar reports `('inhabited', [])`.
- Nothing under `src/proposition7/` mentions a specific primitive by name.
- No module in p7 imports gamma. The dependency runs one way.

## 6. Done when

`import proposition7` gives a constrained-generation library with a grammar
builder, **no session, no turn, and no opinion about agents**.

```bash
grep -ri "read_file\|write_file\|satz" src/          # nothing
grep -rn "class Session\|history\|Mode\." src/       # nothing
```

One call in, one result out, nothing remembered.
