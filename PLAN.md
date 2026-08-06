# Proposition7 Plan: Minimal Backend API v1

Status: proposed integration and release plan.

Cross-plan revision: `semantic-v1-minimal-2026-08-03`

Related plans:

- [`../aufbau/PLAN.md`](../aufbau/PLAN.md)
- [`../provider7/PLAN.md`](../provider7/PLAN.md)
- [`../code7/PLAN.md`](../code7/PLAN.md)

## 1. Scope

P7 owns model-adjacent constrained generation: model/tokenizer integration,
candidate masking, accepted-token feeding, internal grammar caching, and final
verification.

P7 does not own tools, permissions, language authorization, sessions, HTTP, or
execution. Provider7 supplies any tool or language grammar as ordinary `.auf`
source. Generating a code artifact is just another call to the same backend.

The next release should expose the smallest stable interface provider7 needs:

```text
generate(grammar_source, prompt, options)
verify(grammar_source, text, context, goal)
```

## 2. Shared Release Contract

| Layer | Stable identifier | Planned release |
|---|---|---|
| Aufbau engine | `aufbau.engine/v1` | `aufbau` / `aufbau-rs 0.4.0` |
| P7 backend | `proposition7.backend/v1` | `proposition7 0.3.0` |
| Provider wire | `provider7.constraints/v1` | `provider7 0.2.0` |
| S-expression | `provider7.sexpr/v1` | `provider7 0.2.0` |

Dependency and release order:

```text
aufbau-rs 0.4.x
    -> proposition7 0.3.x requires aufbau-rs>=0.4,<0.5
    -> provider7 0.2.x requires proposition7>=0.3,<0.4
    -> code7 0.2.x consumes provider7.constraints/v1
```

Alpha and RC integration uses exact pins. Final releases use the bounded ranges
above. Release order is Aufbau, p7, provider7, code7.

## 3. Current Problems

Current package metadata version: `0.2.0`.

- Runtime `proposition7.__version__` incorrectly reports `0.1.0`.
- Dependency `aufbau-rs>=0.3.1` has no upper bound.
- `ConstrainedModel` stores mutable prompt, KV, and pending-token state on the
  shared object.
- `generate_constrained()` mixes grammar resolution, state setup, decoding,
  and verification in one method.
- The `grammar_name` argument may actually contain complete grammar source.
- Context is added immediately before mask and is affected by Aufbau's stale
  context behavior.
- Completion relies on `status()` instead of strict all-root goal verification.
- There is no stable backend contract for provider7.
- Proposition7 is not yet available as a normal published PyPI dependency.

## 4. Minimal Public API

Stable import path:

```python
from proposition7.backends.v1 import ...
```

### 4.1 Module identity

```python
BACKEND_API = "proposition7.backend/v1"
```

No BackendInfo or feature set is needed. The Backend API identifier guarantees
semantic mask/feed, typed context, goal verification, and request-local state.

### 4.2 Reuse Aufbau verification

```python
from aufbau import Verification
```

P7 returns the engine's immutable `Verification` directly. It does not copy it
into a second p7-specific result type.

`GenerationResult` already exists and remains the generation result. Add one
field:

```python
verification: Verification | None = None
```

Existing token IDs, entropy, retries, and stop fields remain directly on
`GenerationResult`; no trace class is added.

No PreparedConstraint, p7 Verification, GrammarSpec, ContextSpec,
SamplingSpec, GenerationRequest, GenerationResponse, RuntimeInfo, custom
backend exception, or artifact-specific class is needed.

### 4.3 Backend protocol

```python
class ConstraintBackend(Protocol):
    def generate(
        self,
        grammar_source: str,
        prompt: str,
        *,
        initial: str = "",
        context: Mapping[str, str] | None = None,
        max_tokens: int = 64,
        temperature: float = 0.0,
        seed: int | None = None,
        expected_type: str | None = None,
    ) -> GenerationResult: ...

    def verify(
        self,
        grammar_source: str,
        text: str,
        *,
        context: Mapping[str, str] | None = None,
        expected_type: str | None = None,
    ) -> aufbau.Verification: ...
```

That is the full cross-repository Python API.

### 4.4 Contract

- The backend hashes raw `.auf` source and caches its compiled SPG internally.
- Grammar cache handles never cross the repository boundary.
- `generate()` creates request-local model and Synthesizer state.
- Context is applied atomically before the first mask. P7 treats every context
  type string as opaque and lets the grammar's `Type*` parser,
  pattern-matching rules, unification, and rewrites define its meaning.
- Every candidate accepted by mask is fed transactionally.
- `generate()` always performs a fresh final `verify()`.
- `is_complete` is true only when final verification is typed, unambiguous, and
  satisfies `expected_type` when one was supplied.
- `verify()` is state-free relative to model generation.
- Invalid grammar/arguments use `ValueError`; backend execution failures use
  `RuntimeError`. V1 adds no exception hierarchy.

### 4.5 Compatibility

`ConstrainedModel.generate_constrained()` remains throughout 0.3.x as a wrapper
over the backend path. Its removal cannot occur before 0.5.0.

`ConstrainedModel` itself implements `ConstraintBackend`. No concrete adapter
class is added. The Protocol exists only for provider injection, fakes, and
static checks.

Existing grammar-name lookup remains a convenience outside Backend API v1.
Backend API v1 accepts raw source only, eliminating name-versus-source
branching at the cross-repository boundary.

## 5. Internal State Simplification

Split the current model object into:

```text
shared: model, tokenizer, immutable configuration, bounded SPG cache by source hash
request-local: prompt IDs, KV state, pending IDs, Synthesizer, trace
```

No session abstraction is added to p7. Provider7 owns sessions. If a backend
cannot safely run simultaneous requests, it serializes them internally and
documents that operational limit operationally; it does not leak mutable state
across calls. Model and worker identity are deployment configuration, not part
of the backend protocol.

## 6. Workstreams

### Workstream A: backend contract

Can start now without Aufbau 0.4.

Files:

```text
src/proposition7/backends/__init__.py
src/proposition7/backends/v1.py
tests/backend_api_v1.py
```

- [ ] Define the Protocol and module constant; re-export Aufbau Verification.
- [ ] Add a fake backend used by provider7 integration tests.
- [ ] Ensure importing the contract does not import Torch.
- [ ] Add API signature and immutability tests.

### Workstream B: request-local generation state

Can start now in parallel with A.

Files:

```text
src/proposition7/llm.py
src/proposition7/inference.py
tests/llm_*.py
```

- [ ] Move prompt/KV/pending-token state into one internal per-call object.
- [ ] Bound the internal grammar cache size.
- [ ] Route legacy generation through the same internal decode function.
- [ ] Preserve current output and trace behavior.
- [ ] Property-test arbitrary interleavings with fake model state.

### Workstream C: Engine API integration

Blocked by `aufbau-rs 0.4.0a1`.

Files:

```text
src/proposition7/llm.py
tests/backend_aufbau_v1.py
```

- [ ] Require `aufbau.engine/v1`.
- [ ] Compile/cache raw grammar source internally by content hash.
- [ ] Implement atomic context, transactional mask/feed, and strict verify.
- [ ] Make `ConstrainedModel` implement the Protocol directly.
- [ ] Return Aufbau `Verification` directly without a mapping or adapter layer.
- [ ] Fail clearly instead of degrading behavior.

### Workstream D: package release

Metadata work can start now; final is blocked by C.

- [ ] Make package metadata the single version source.
- [ ] Replace hardcoded `__version__` with `importlib.metadata.version()`.
- [ ] Pin alphas exactly; final range is `aufbau-rs>=0.4,<0.5`.
- [ ] Build universal wheel and sdist in clean environments.
- [ ] Test base import without Torch and optional model extras with Torch.
- [ ] Publish to TestPyPI/PyPI.
- [ ] Add changelog, migration notes, and tag `v0.3.0`.

Grammar registries, language metadata, tool schemas, permission systems, and
artifact classes are explicitly not part of p7 0.3.0. Provider7 can use any
language grammar through the same raw-source API.

P7 must not maintain a type-name table or branch on constructors such as
`Int`, `Code`, arrows, capabilities, or products. Its context responsibility is
only to pass the complete name-to-type-source mapping atomically to the grammar
used for that call.

## 7. Required Properties

- Fake and real backends satisfy one shared Protocol property suite.
- Internal grammar source hashes and cache lookup are deterministic.
- Context-sensitive generation agrees with standalone verification.
- Arbitrary context type terms generated from each grammar's `Type*` production
  pass through p7 unchanged and retain Aufbau pattern-matching behavior.
- Accepted mask candidates feed successfully.
- Failed feed never corrupts subsequent generation.
- Every completed result passes fresh goal verification with at most one unique
  root type.
- Request-local state is isolated across arbitrary call schedules.
- Legacy and backend APIs agree for equivalent supported calls.
- Missing Aufbau Engine API v1 fails at backend construction/startup.

## 8. Release Gate

Release sequence:

1. `0.3.0a1`: minimal API types, fake backend, request-local state.
2. `0.3.0a2`: Aufbau 0.4 alpha adapter and provider7 feedback.
3. `0.3.0rc1`: API freeze against Aufbau 0.4 RC.
4. `0.3.0`: publish after Aufbau 0.4.0 final.

`proposition7 0.3.0` is ready when:

- `proposition7.backend/v1` has exactly the minimal API above.
- Real generation uses `aufbau.engine/v1` correctly.
- Version metadata, dependency range, wheel/sdist, and `v0.3.0` tag agree.
- Provider7 contract tests pass against the published release candidate.
