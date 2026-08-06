"""satz against the real engine. Each test names the invariant it protects."""

from __future__ import annotations

import pytest

from proposition7.satz import (
    Dispatch, DispatchError, Effect, Err, Gamma, LanguageBinding, Ok, Param,
    Primitive, PrimitiveFailure, Scheme, Session, compose, prompt, validate_binding,
)
from proposition7.satz.evaluator import Evaluator

aufbau = pytest.importorskip("aufbau")

# Stand-in for D3: the minimum shape satisfying the composition contract.
# NB an .auf regex literal cannot contain an escaped forward slash.
CORE = """Identifier ::= /[a-z_][a-z0-9_]*/
Type* ::= TAtom
TAtom ::= 'PathSet' | 'Text' | 'IoError' | Result
Result ::= 'Result' '[' Type ',' Type ']'
Variable(var) ::= Identifier[x]
StringLit(str_lit) ::= /"[a-zA-Z0-9_]*"/
Todo(todo) ::= 'todo'
Expression ::= Variable | StringLit | Todo | @PRIMITIVES@
Stmt(decl) ::= Identifier[name] '=' Expression[value] ';'
StatementList ::= Stmt StatementList | Stmt
Program ::= StatementList

x ∈ Γ
----------- (var)
Γ(x)

----------- (str_lit)
'Text'

----------- (todo)
?A

Γ ⊢ value : ?t
----------------------- (decl)
Γ → Γ[name:?t] ⊢ 'void'
"""

LIST = Primitive("list_dir", (Param("path", "Text"),), returns="PathSet")
GLOB = Primitive("glob", (Param("p", "Text"),), returns="PathSet", raises="IoError",
                 effects=(Effect("read", "**"),))
WRITE = Primitive("write_file", (Param("p", "Text"), Param("c", "Text")),
                  returns="Text", raises="IoError", effects=(Effect("write", "**"),))
SCHEME = Scheme((WRITE, GLOB, LIST))
BINDING = LanguageBinding(core_source=CORE)


def hosts(**over):
    base = {"list_dir": lambda p: ["a.py"], "glob": lambda p: ["c.py"],
            "write_file": lambda p, c: "wrote"}
    base.update(over)
    return base


def evaluator(scheme=SCHEME, approve=None, **over):
    spg = aufbau.SPG(compose(scheme, BINDING))
    d = Dispatch(scheme, {k: v for k, v in hosts(**over).items()
                          if scheme.get(k) is not None})
    return Evaluator(d, BINDING, spg, approve=approve)


# ── scheme ──────────────────────────────────────────────────────────────

def test_json_roundtrip():
    assert Scheme.from_json(SCHEME.to_json()).to_json() == SCHEME.to_json()


def test_duplicate_names_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        Scheme((GLOB, GLOB))


# ── grammar ─────────────────────────────────────────────────────────────

def test_composition_is_byte_deterministic():
    """ARCHITECTURE §3.1: an unstable grammar recompiles every request."""
    assert compose(SCHEME, BINDING) == compose(Scheme((LIST, WRITE, GLOB)), BINDING)


def test_fallible_return_is_wrapped():
    """ARCHITECTURE I1a: a fallible primitive cannot return a bare value."""
    src = compose(SCHEME, BINDING)
    assert "Result[PathSet, IoError]" in src


def test_multi_arg_separator_is_quoted():
    """A bare comma silently makes every multi-argument call dead."""
    assert "Expression[arg0] ',' Expression[arg1]" in compose(SCHEME, BINDING)


def test_composition_preserves_the_start_symbol():
    """aufbau takes the last-declared nonterminal as start; appending the
    fragment made PrimitiveCall the root, so a bare call parsed as a program."""
    core_only = aufbau.SPG(CORE.replace(" | @PRIMITIVES@", ""))
    assert aufbau.SPG(compose(SCHEME, BINDING)).start == core_only.start


def test_empty_scheme_composes():
    assert aufbau.SPG(compose(Scheme(), BINDING)).start


def test_validate_binding_accepts_the_contract():
    assert validate_binding(BINDING) == []


def test_validate_binding_reports_missing_result_ctor():
    bad = LanguageBinding(core_source=CORE.replace("Result", "Either"))
    assert any("Result" in p for p in validate_binding(bad))


def test_todo_discharges_inhabitation():
    """ARCHITECTURE §2.2/I5: without a universal inhabitant the mask can trap
    the model in a live-but-uncompletable prefix."""
    assert aufbau.SPG(compose(SCHEME, BINDING)).completeness() == ("inhabited", [])
    concrete = CORE.replace("----------- (todo)\n?A", "----------- (todo)\n'Text'")
    kind, sorts = aufbau.SPG(
        compose(SCHEME, LanguageBinding(core_source=concrete))
    ).completeness()
    assert (kind, sorts) != ("inhabited", [])


# ── I2 / I7 ─────────────────────────────────────────────────────────────

def test_dispatch_rejects_unimplemented():
    with pytest.raises(DispatchError, match="no implementation"):
        Dispatch(SCHEME, {"glob": lambda p: []})


def test_dispatch_rejects_unreachable():
    with pytest.raises(DispatchError, match="no scheme entry"):
        Dispatch(SCHEME, hosts(orphan=lambda: None))


def test_revoking_a_capability_removes_it_from_the_grammar():
    """ARCHITECTURE I7: the scheme is the policy."""
    ro = SCHEME.without_effects({"write"})
    assert ro.get("write_file") is None
    assert "write_file" not in compose(ro, BINDING)


def test_ungranted_primitive_is_masked_out():
    spg = aufbau.SPG(compose(SCHEME.without_effects({"write"}), BINDING))
    synth = aufbau.Synthesizer.from_grammar(spg, "")
    for tok in ["x", "="]:
        synth.feed(tok)
    assert not synth.mask(["write_file"])[0]


# ── Γ ───────────────────────────────────────────────────────────────────

def test_wire_projection_carries_no_values():
    """ARCHITECTURE I1: values never cross."""
    g = Gamma()
    g.bind("files", "PathSet", ["secret.py"])
    assert g.types() == {"files": "PathSet"}


# ── evaluation ──────────────────────────────────────────────────────────

def test_binds_name_type_and_value():
    out = evaluator().run('files = list_dir("src");', Gamma())
    assert out.ok
    assert out.updates["files"] == ("PathSet", ["a.py"])


def test_fallible_success_wraps_in_ok():
    out = evaluator().run('f = glob("s");', Gamma())
    assert isinstance(out.updates["f"][1], Ok)


def test_fallible_failure_still_binds():
    """ARCHITECTURE I1a: failure is a value, so Γ′ commits unconditionally."""
    def boom(p):
        raise PrimitiveFailure("denied")

    out = evaluator(glob=boom).run('f = glob("s");', Gamma())
    assert out.ok, "expected failure must not abort the turn"
    assert isinstance(out.updates["f"][1], Err)


def test_host_bug_propagates():
    def broken(p):
        raise KeyError("host bug")

    with pytest.raises(KeyError):
        evaluator(list_dir=broken).run('f = list_dir("s");', Gamma())


def test_statements_see_earlier_bindings():
    out = evaluator().run('a = list_dir("src"); b = a;', Gamma())
    assert out.updates["b"][1] == ["a.py"]


def test_todo_aborts_before_any_effect():
    """ARCHITECTURE §2.2: the model-failure channel runs nothing."""
    written = []
    out = evaluator(write_file=lambda p, c: written.append(p)).run(
        'a = write_file("f","x"); b = todo;', Gamma()
    )
    assert out.aborted and out.reason == "todo"
    assert written == []
    assert out.updates == {}


def test_audit_sees_whole_turn_before_execution():
    """ARCHITECTURE §2.1: atomicity is what makes this knowable up front."""
    ev = evaluator()
    audit = ev.audit(ev.parse('a = glob("s"); b = write_file("f","x");'))
    assert audit.kinds() == {"read", "write"}


def test_denied_approval_runs_nothing():
    written = []
    out = evaluator(approve=lambda a: "write" not in a.kinds(),
                    write_file=lambda p, c: written.append(p)).run(
        'a = write_file("f","x");', Gamma()
    )
    assert out.aborted and out.reason == "denied"
    assert written == []


# ── session ─────────────────────────────────────────────────────────────

class FakeClient:
    def __init__(self, completion):
        self.completion, self.requests = completion, []

    def generate(self, request):
        self.requests.append(request)
        from proposition7.satz import Response
        return Response(self.completion, dict(request.aufbau_context))


def session(completion, **over):
    spg = aufbau.SPG(compose(SCHEME, BINDING))
    client = FakeClient(completion)
    return client, Session(client, BINDING, Dispatch(SCHEME, hosts(**over)), spg,
                           model="test", system="SYS", task="TASK")


def test_turn_commits_and_threads_gamma():
    client, s = session('files = list_dir("src");')
    assert s.gamma.types() == {}
    assert s.turn("go").ok
    assert s.gamma.types() == {"files": "PathSet"}
    s.turn("again")
    assert client.requests[-1].aufbau_context == {"files": "PathSet"}


def test_gamma_renders_last():
    """ARCHITECTURE §3.1: Γ changes every turn; it must not sit in the prefix."""
    pairs = prompt(SCHEME, Gamma(), system="SYS", task="TASK", instruction="GO")
    contents = [c for _, c in pairs]
    assert contents.index("SYS") < next(i for i, c in enumerate(contents) if "Primitives" in c)
    assert next(i for i, c in enumerate(contents) if "Context:" in c) == len(contents) - 2
