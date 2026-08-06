"""End-to-end wiring tests.

These assert the invariants from ARCHITECTURE, not incidental behaviour. Each
test names the invariant it protects, so a failure says which architectural
property broke rather than which function returned the wrong tuple.
"""

from __future__ import annotations

import pytest

from proposition7.satz import (
    Bind,
    Call,
    CoreRequest,
    CoreResponse,
    Dispatch,
    DispatchError,
    Err,
    Gamma,
    LanguageBinding,
    Lit,
    Ok,
    Param,
    Primitive,
    PrimitiveFailure,
    Program,
    Scheme,
    Todo,
    Var,
    audit,
    build,
    compose,
    validate_binding,
)
from proposition7.satz.grammar import PRIMITIVE_MARKER
from proposition7.satz.scheme import Effect

# ── Fixtures ─────────────────────────────────────────────────────────────

# A stand-in for D3. Not a real language — just enough structure to satisfy the
# composition contract, so the wiring is testable before the core language
# exists.
CORE = f"""// toy core language (stand-in for D3)
Identifier ::= /[a-z_][a-z0-9_]*/
Type* ::= TAtom
TAtom ::= 'PathSet' | 'Text' | 'IoError' | Result
Result ::= 'Result' '[' Type ',' Type ']'
Variable(var) ::= Identifier[x]
// NOTE for D3: an `.auf` regex literal cannot contain an escaped forward
// slash — it terminates the regex early and the loader reports "Unclosed
// quotes". Keep character classes to plain alphanumerics and underscores.
StringLit(str_lit) ::= /"[a-zA-Z0-9_]*"/
Todo(todo) ::= 'todo'
Expression ::= Variable | StringLit | Todo | {PRIMITIVE_MARKER}
Stmt(decl) ::= Identifier[name] '=' Expression[value] ';'
StatementList ::= Stmt StatementList | Stmt
Program ::= StatementList

// ===================== Typing rules =====================
// Every labelled nonterminal above needs a rule here. A label without a
// matching rule silently disables enforcement — the grammar still loads and
// still masks, it just stops checking anything.

x ∈ Γ
----------- (var)
Γ(x)

----------- (str_lit)
'Text'

// The universal inhabitant (ARCHITECTURE §2.2): a metavariable conclusion
// means `todo` inhabits every sort, so the mask can never trap the model in a
// live-but-uncompletable prefix.
----------- (todo)
?A

// Right recursion carries this effect down the whole statement list, which is
// what makes flat same-scope extension work (ARCHITECTURE §2).
Γ ⊢ value : ?t
----------------------- (decl)
Γ → Γ[name:?t] ⊢ 'void'
"""

GLOB = Primitive(
    name="glob",
    params=(Param("pattern", "Text"),),
    returns="PathSet",
    raises="IoError",
    effects=(Effect("read", "**"),),
)

WRITE = Primitive(
    name="write_file",
    params=(Param("path", "Text"), Param("content", "Text")),
    returns="Text",
    raises="IoError",
    effects=(Effect("write", "**"),),
)

COUNT = Primitive(name="count", params=(Param("p", "PathSet"),), returns="Text")

# Total: no `raises`, so it returns a bare `PathSet` rather than a `Result`.
LIST_DIR = Primitive(
    name="list_dir", params=(Param("path", "Text"),), returns="PathSet"
)

SCHEME = Scheme(primitives=(WRITE, GLOB, COUNT, LIST_DIR))
BINDING = LanguageBinding(core_source=CORE)


def hosts(**overrides):
    base = {
        "glob": lambda pattern: [f"{pattern}/a.py"],
        "write_file": lambda path, content: f"wrote {len(content)}B to {path}",
        "count": lambda paths: str(len(paths)),
        "list_dir": lambda path: [f"{path}/a.py", f"{path}/b.py"],
    }
    base.update(overrides)
    return base


class FakeClient:
    """Stands in for provider7. Records what crossed the wire."""

    def __init__(self, completion: str = ""):
        self.completion = completion
        self.requests: list[CoreRequest] = []

    def generate(self, request: CoreRequest) -> CoreResponse:
        self.requests.append(request)
        return CoreResponse(
            completion=self.completion,
            updated_aufbau_context=dict(request.aufbau_context),
        )


# ── Scheme ───────────────────────────────────────────────────────────────


def test_scheme_json_roundtrip():
    assert Scheme.from_json(SCHEME.to_json()).ordered() == SCHEME.ordered()


def test_scheme_rejects_duplicate_names():
    with pytest.raises(ValueError, match="duplicate"):
        Scheme(primitives=(GLOB, GLOB))


def test_scheme_is_the_policy():
    """ARCHITECTURE I7: revoking a capability means removing the primitive."""
    readonly = SCHEME.without_effects({"write"})
    assert readonly.get("write_file") is None
    assert readonly.get("glob") is not None
    # And the grammar follows, so the mask cannot emit the call at all.
    assert "write_file" not in compose(readonly, BINDING)


# ── Grammar composition ──────────────────────────────────────────────────


def test_composition_is_byte_deterministic():
    """ARCHITECTURE §3.1: an unstable grammar recompiles every request."""
    shuffled = Scheme(primitives=(COUNT, LIST_DIR, WRITE, GLOB))
    assert compose(SCHEME, BINDING) == compose(shuffled, BINDING)
    assert Scheme.from_json(SCHEME.to_json()).to_json() == SCHEME.to_json()


def test_fallible_return_is_wrapped():
    """ARCHITECTURE I1a: a fallible primitive cannot return a bare value."""
    source = compose(SCHEME, BINDING)
    assert "Result[PathSet, IoError]" in source
    # The total primitive is not wrapped.
    assert "\nText\n" in source or source.rstrip().endswith("Text")


def test_multi_arg_separator_is_a_quoted_literal():
    """A bare comma silently makes every multi-argument call dead."""
    source = compose(SCHEME, BINDING)
    assert "Expression[arg0] ',' Expression[arg1]" in source


def test_empty_scheme_composes_to_valid_grammar():
    source = compose(Scheme(), BINDING)
    assert PRIMITIVE_MARKER not in source
    assert "| PrimitiveCall" not in source


def test_missing_marker_is_rejected():
    bad = LanguageBinding(core_source="Expression ::= Variable\n")
    with pytest.raises(Exception, match="splice point"):
        compose(SCHEME, bad)


def test_validate_binding_accepts_the_contract():
    assert validate_binding(BINDING) == []


def test_validate_binding_reports_missing_result_constructor():
    binding = LanguageBinding(core_source=CORE.replace("Result", "Either"))
    problems = validate_binding(binding)
    assert any("Result" in p for p in problems)


# ── I2: one table, two projections ───────────────────────────────────────


def test_dispatch_rejects_unimplemented_primitive():
    with pytest.raises(DispatchError, match="no implementation"):
        Dispatch(scheme=SCHEME, hosts={"glob": lambda p: []})


def test_dispatch_rejects_unreachable_implementation():
    with pytest.raises(DispatchError, match="no scheme entry"):
        Dispatch(scheme=SCHEME, hosts=hosts(orphan=lambda: None))


# ── Γ: the two-layer split ───────────────────────────────────────────────


def test_gamma_wire_projection_carries_no_values():
    """ARCHITECTURE I1: values never cross."""
    gamma = Gamma()
    gamma.bind("files", "PathSet", ["secret.py"])
    wire = gamma.types()
    assert wire == {"files": "PathSet"}
    assert "secret.py" not in repr(wire)


# ── Evaluation ───────────────────────────────────────────────────────────


def session(completion: str, **kw):
    client = FakeClient(completion)
    return client, build(
        client,
        BINDING,
        SCHEME,
        hosts(**kw.pop("host_overrides", {})),
        kw.pop("lower"),
        model="test",
        **kw,
    )


def test_fallible_success_wraps_in_ok():
    program = Program((Bind("files", Call("glob", (Lit("src", "Text"),))),))
    _, s = session("", lower=lambda _: program)
    outcome = s.evaluator.run(program, s.gamma)
    assert outcome.ok
    type_source, value = outcome.updates["files"]
    assert isinstance(value, Ok)
    assert type_source == "Result[PathSet, IoError]"


def test_fallible_failure_still_binds():
    """ARCHITECTURE I1a: failure is a value, so Γ′ commits unconditionally."""

    def boom(pattern):
        raise PrimitiveFailure("permission denied")

    program = Program((Bind("files", Call("glob", (Lit("src", "Text"),))),))
    _, s = session("", lower=lambda _: program, host_overrides={"glob": boom})
    outcome = s.evaluator.run(program, s.gamma)

    assert outcome.ok, "an expected failure must not abort the turn"
    type_source, value = outcome.updates["files"]
    assert isinstance(value, Err) and value.error == "permission denied"
    # Same type the mask constrained the call to — the binding exists either way.
    assert type_source == "Result[PathSet, IoError]"


def test_host_bug_is_not_swallowed():
    def broken(pattern):
        raise KeyError("host bug")

    program = Program((Bind("f", Call("glob", (Lit("x", "Text"),))),))
    _, s = session("", lower=lambda _: program, host_overrides={"glob": broken})
    with pytest.raises(KeyError):
        s.evaluator.run(program, s.gamma)


def test_todo_aborts_before_any_effect():
    """ARCHITECTURE §2.2: the model-failure channel runs nothing."""
    written: list[str] = []
    program = Program(
        (
            Bind("a", Call("write_file", (Lit("f", "Text"), Lit("x", "Text")))),
            Bind("b", Todo(note="cannot proceed")),
        )
    )
    _, s = session(
        "",
        lower=lambda _: program,
        host_overrides={"write_file": lambda p, c: written.append(p)},
    )
    outcome = s.evaluator.run(program, s.gamma)

    assert outcome.aborted and outcome.reason == "todo"
    assert outcome.note == "cannot proceed"
    assert written == [], "todo must abort before the earlier statement runs"
    assert outcome.updates == {}


def test_statements_see_earlier_bindings():
    """ARCHITECTURE §2: a flat sequence threads bindings forward.

    Note the chain uses `list_dir` (total, returns a bare `PathSet`) rather
    than `glob` (fallible, returns `Result[PathSet, IoError]`). Feeding a
    `Result` to a primitive expecting its payload is a type error the *mask*
    rejects at decode time — the evaluator deliberately does not re-check it
    (the grammar is the type checker), so such a program can only arise from a
    lowering bug and shows up as a host crash. That asymmetry is the design,
    and this test exists partly to record it.
    """
    program = Program(
        (
            Bind("files", Call("list_dir", (Lit("src", "Text"),))),
            Bind("n", Call("count", (Var("files"),))),
        )
    )
    _, s = session("", lower=lambda _: program)
    outcome = s.evaluator.run(program, s.gamma)
    assert outcome.ok
    assert outcome.updates["n"] == ("Text", "2")


# ── Effects ──────────────────────────────────────────────────────────────


def test_audit_sees_whole_turn_before_execution():
    """ARCHITECTURE §2.1: atomicity is what makes this knowable up front."""
    program = Program(
        (
            Bind("files", Call("glob", (Lit("src", "Text"),))),
            Bind("r", Call("write_file", (Lit("o", "Text"), Lit("c", "Text")))),
        )
    )
    dispatch = Dispatch(scheme=SCHEME, hosts=hosts())
    result = audit(program, dispatch)
    assert result.kinds() == {"read", "write"}
    assert set(result.primitives) == {"glob", "write_file"}


def test_denied_approval_runs_nothing():
    written: list[str] = []
    program = Program(
        (Bind("r", Call("write_file", (Lit("o", "Text"), Lit("c", "Text")))),)
    )
    _, s = session(
        "",
        lower=lambda _: program,
        approve=lambda a: "write" not in a.kinds(),
        host_overrides={"write_file": lambda p, c: written.append(p)},
    )
    outcome = s.evaluator.run(program, s.gamma)
    assert outcome.aborted and outcome.reason == "denied"
    assert written == []


# ── Session ──────────────────────────────────────────────────────────────


def test_turn_commits_and_threads_gamma():
    program = Program((Bind("files", Call("glob", (Lit("src", "Text"),))),))
    client, s = session("files = glob(\"src\");", lower=lambda _: program)

    assert s.gamma.types() == {}
    record = s.turn("find the python files")

    assert record.ok
    assert s.gamma.types() == {"files": "Result[PathSet, IoError]"}

    # Second turn carries the binding into the request's aufbau_context.
    s.turn("count them")
    assert client.requests[-1].aufbau_context == {
        "files": "Result[PathSet, IoError]"
    }


def test_gamma_renders_last_in_the_prompt():
    """ARCHITECTURE §3.1: Γ changes every turn, so it must not sit in the
    cache-stable prefix."""
    program = Program((Bind("files", Call("glob", (Lit("src", "Text"),))),))
    client, s = session("", lower=lambda _: program)
    s.layout.system = "SYSTEM"
    s.layout.task = "TASK"
    s.turn("INSTRUCTION")

    rendered = client.requests[-1].model_context
    assert rendered.index("SYSTEM") < rendered.index("Primitives:")
    assert rendered.index("TASK") < rendered.index("Context:")
    assert rendered.index("Context:") > rendered.index("Primitives:")


def test_incomplete_decode_leaves_gamma_untouched():
    class Incomplete(FakeClient):
        def generate(self, request):
            self.requests.append(request)
            return CoreResponse(
                completion="files = glo",
                updated_aufbau_context=dict(request.aufbau_context),
                is_complete=False,
                stopped_reason="budget",
            )

    client = Incomplete()
    s = build(
        client,
        BINDING,
        SCHEME,
        hosts(),
        lambda _: Program(),
        model="test",
    )
    record = s.turn()
    assert not record.ok
    assert s.gamma.types() == {}


def test_build_wires_result_template_from_binding():
    """The evaluator's reported type must match the grammar's conclusion."""
    binding = LanguageBinding(core_source=CORE, result_type="Either[{ok}, {err}]")
    s = build(
        FakeClient(), binding, SCHEME, hosts(), lambda _: Program(), model="test"
    )
    assert s.dispatch.result_type == "Either[{ok}, {err}]"
    program = Program((Bind("f", Call("glob", (Lit("s", "Text"),))),))
    outcome = s.evaluator.run(program, s.gamma)
    assert outcome.updates["f"][0] == "Either[PathSet, IoError]"
