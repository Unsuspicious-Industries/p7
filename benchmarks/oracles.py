from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class OracleResult:
    ok: bool
    reason: str = ""
    observed: Any = None
    expected: Any = None


def canonical_grammar(grammar: str) -> str:
    """Reduce a grammar name to its grading base.

    The aufbau stack is *arbitrary-grammar* by design: any name that is not a
    registered grammar is handled by the default oracle, so a new grammar is
    declared in the registry (``GRAMMAR_SPECS``) and works everywhere — no
    call-site branches. A syntactic twin (``<base>_syntactic``) strips the
    typing rules but is still graded by its semantic base.
    """
    try:
        import proposition7

        return proposition7.base_grammar(grammar)
    except Exception:
        if grammar.endswith("_syntactic"):
            return grammar[: -len("_syntactic")]
        return grammar


# ── Oracle registry ─────────────────────────────────────────────────
#
# The single table that defines how every grammar is graded and validated.
# A new grammar is added by registering one GrammarSpec (its allowed modes,
# per-mode required resolution fields, and per-mode handler) — nothing else
# in the benchmark reads grammar names directly.


@dataclass(frozen=True)
class GrammarSpec:
    """How one grammar resolves/grades output: allowed modes, per-mode
    required resolution keys, list-typed keys, and a handler per mode."""

    name: str
    modes: frozenset[str]
    required: dict[str, frozenset[str]] = field(default_factory=dict)
    list_fields: frozenset[str] = frozenset()
    type_fields: frozenset[str] = frozenset()  # keys that must be non-empty when present
    handlers: dict[str, Callable[[dict, str, str, str], OracleResult]] = field(
        default_factory=dict
    )

    def handler(self, mode: str):
        return self.handlers.get(mode)

    def mode_error(self, mode: str) -> OracleResult:
        return OracleResult(False, f"unsupported_resolution_mode_for_{self.name}: {mode}")


def _wrap(name: str):
    """Guard a handler so an oracle failure is reported, never raised."""

    def wrapper(fn):
        def checked(resolution, output, expected, grammar):
            try:
                return fn(resolution, output, expected, grammar)
            except Exception as error:  # noqa: BLE001
                return OracleResult(False, f"{name}_resolution_error: {error}")

        return checked

    return wrapper


def _exact(resolution, output, expected, grammar) -> OracleResult:
    del resolution, grammar
    ok = _normalize_text(output) == _normalize_text(expected)
    return OracleResult(ok, "" if ok else "exact_mismatch", output, expected)


def _default_mismatch(resolution, output, expected, grammar) -> OracleResult:
    del resolution, grammar
    ok = _normalize_text(output) == _normalize_text(expected)
    return OracleResult(ok, "" if ok else "equivalence_mismatch", output, expected)


def check_resolution(task: Any, output: str, grammar: Optional[str] = None) -> OracleResult:
    resolution = dict(getattr(task, "resolution", {}) or {})
    mode = str(resolution.get("mode") or "exact")
    expected = str(getattr(task, "expected", ""))
    if grammar is None:
        grammar = str(getattr(task, "grammar", ""))
        grammar = canonical_grammar(grammar)

    spec = GRAMMAR_SPECS.get(grammar, GRAMMAR_SPECS[_DEFAULT_GRAMMAR])
    handler = spec.handler(mode)
    if handler is None:
        return spec.mode_error(mode)
    return handler(resolution, output, expected, grammar)


def check_episode_resolution(task: Any, result: Any) -> OracleResult:
    """Grade a multi-turn agent episode (a `proposition7.agents.AgentResult`)
    against a task's `resolution.expected_value`. Task accomplishment, not
    typedness: `result.success` only means the episode mechanically reached
    a well-typed `return` within the turn budget (a one-turn `return
    "hello";` is `success=True` too) -- the actual pass/fail criterion here
    is whether the *executed* return value matches what the task asked for
    (lmpl-plan.md section 5.2)."""
    resolution = dict(getattr(task, "resolution", {}) or {})
    expected_value = resolution.get("expected_value")
    if not bool(getattr(result, "success", False)):
        reason = str(getattr(result, "reason", ""))
        return OracleResult(False, f"episode_failed:{reason}", None, expected_value)
    observed = getattr(result, "return_value", None)
    ok = observed == expected_value
    reason = "" if ok else f"wrong_value:{observed!r}_vs_{expected_value!r}"
    return OracleResult(ok, reason, observed, expected_value)


_ML_SPG = None


def _ml_spg():
    """The compiled ml grammar, built once. ml is a strict OCaml subset, so the
    engine that constrains generation is also the oracle that grades it."""
    global _ML_SPG
    if _ML_SPG is None:
        import aufbau
        import proposition7

        _ML_SPG = aufbau.SPG(proposition7.get_grammar("ml"))
    return _ML_SPG


def _ml_status(spg, program: str) -> str:
    import aufbau

    return aufbau.Synthesizer.from_grammar(spg, program).status()


def _ml_eval(program: str, expected_type: str) -> Optional[str]:
    """Evaluate an ML program with OCaml and return the output, or None on
    failure.  The program must be a valid OCaml expression in aufbau's
    monomorphic ML subset."""
    # Build a wrapper program with appropriate printer for the expected type.
    # The wrapper defines helper printers for lists and pairs, then evaluates
    # the program and prints its value.
    helpers = """
let rec string_of_int_list = function [] -> "" | h :: t -> string_of_int h ^ (match t with [] -> "" | _ -> "; " ^ string_of_int_list t);;
let string_of_int_pair (a, b) = "(" ^ string_of_int a ^ ", " ^ string_of_int b ^ ")";;
let string_of_int_bool_pair (a, b) = "(" ^ string_of_int a ^ ", " ^ string_of_bool b ^ ")";;
"""

    printer_by_type = {
        "int": 'print_endline (string_of_int (%s));;',
        "bool": 'print_endline (string_of_bool (%s));;',
        "unit": 'print_endline "()";;',
        "int list": 'print_endline ("[" ^ string_of_int_list (%s) ^ "]");;',
        "int * int": 'print_endline (string_of_int_pair (%s));;',
        "int * bool": 'print_endline (string_of_int_bool_pair (%s));;',
        "assert_false": 'let _ = (%s) in print_endline "ok";;',  # will crash
    }
    # For function types, any well-typed function evaluates to a closure.
    if expected_type and expected_type.startswith("int ->") or expected_type.startswith("bool ->"):
        printer = 'let _ = (%s) in print_endline "<fun>";;'
    else:
        printer = printer_by_type.get(expected_type, 'print_endline (match (%s) with _ -> "<fun>");;')

    try:
        src = helpers + "\n" + printer % program
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ml", delete=False, prefix="ml_eval_"
        ) as f:
            f.write(src)
            f.flush()
            fname = f.name
        try:
            result = subprocess.run(
                ["ocaml", fname],
                capture_output=True,
                text=True,
                timeout=10,
            )
            stdout = result.stdout.strip()
            if result.returncode != 0:
                # If the program crashed (e.g. Assert_failure for assert false),
                # return a sentinel so callers can check for expected crash.
                if expected_type == "assert_false":
                    return "assert_failure"
                return None
            return stdout
        except subprocess.TimeoutExpired:
            return None
        finally:
            try:
                os.unlink(fname)
            except OSError:
                pass
    except Exception:
        return None


def _ml_eval_handler(resolution, output, expected, grammar) -> OracleResult:
    """`eval` mode (recommended): check well-typedness via aufbau, then evaluate
    the program with the OCaml toplevel and compare the output to
    `resolution["expected_value"]`."""
    expected_value = resolution.get("expected_value", "")
    expected_type = resolution.get("type", "")

    # First, verify the program is well-typed (the grammar's own check)
    spg = _ml_spg()
    status = _ml_status(spg, output)
    if status != "typed":
        return OracleResult(
            False, f"not_well_typed:{status}", output, expected_value
        )

    if expected_type and expected_type != "assert_false":
        ascription = f"let chk : {expected_type} = {output} in chk"
        if _ml_status(spg, ascription) != "typed":
            return OracleResult(False, "type_mismatch", output, expected_type)

    # Evaluate with OCaml and compare
    observed = _ml_eval(output, expected_type)
    if observed is None:
        return OracleResult(False, "eval_crashed_or_timeout", output, expected_value)

    ok = observed == expected_value
    reason = "" if ok else f"wrong_value:{observed!r}_vs_{expected_value!r}"
    return OracleResult(ok, reason, observed, expected_value)


def _ml_legacy(resolution, output, expected, grammar) -> OracleResult:
    """Legacy circular mode (`type`/`equivalence`): kept for early-branch
    compatibility but deprecated — new tasks should use `eval`."""
    spg = _ml_spg()
    status = _ml_status(spg, output)
    if status != "typed":
        return OracleResult(
            False, f"not_well_typed:{status}", output, expected
        )
    wanted = resolution.get("type")
    if wanted:
        ascription = f"let chk : {wanted} = {output} in chk"
        if _ml_status(spg, ascription) != "typed":
            return OracleResult(False, "type_mismatch", output, str(wanted))
        return OracleResult(True, "", output, str(wanted))
    # No declared type: well-typedness plus a text match on the program.
    ok = _normalize_text(output) == _normalize_text(expected)
    return OracleResult(
        ok, "" if ok else "equivalence_mismatch", output, expected
    )


_TOOL_SPGS: dict[str, Any] = {}


def _tool_spg(grammar: str):
    """The compiled tool grammar, built once per variant (`tool`/`tool_sexpr`
    share one type registry in two different concrete syntaxes). Both are
    invented DSLs with no external ground-truth compiler, so aufbau's own
    type system is the oracle — the same self-consistency role ml.auf's
    checker plays before its separate OCaml cross-check."""
    spg = _TOOL_SPGS.get(grammar)
    if spg is None:
        import aufbau
        import proposition7

        spg = aufbau.SPG(proposition7.get_grammar(grammar))
        _TOOL_SPGS[grammar] = spg
    return spg


def _tool_type_handler(resolution, output, expected, grammar) -> OracleResult:
    """`type` mode for the invented tool DSLs: well-typedness against the
    fixed tool registry (search/summarize/count/format)."""
    import aufbau

    status = aufbau.Synthesizer.from_grammar(_tool_spg(grammar), output).status()
    ok = status == "typed"
    return OracleResult(ok, "" if ok else f"not_well_typed:{status}", output, expected)


def _tool_value_handler(resolution, output, expected, grammar) -> OracleResult:
    """`value` mode: run the program against the same deterministic mock
    registry (benchmarks/tool_registry) and compare the *executed* return
    value to ``resolution["expected_value"]`` — task accomplishment, not
    typedness."""
    from benchmarks.tool_registry import InterpretError, run_program

    wanted = resolution.get("expected_value")
    try:
        value, _env = run_program(output, syntax=grammar)
    except InterpretError as error:
        return OracleResult(False, f"unexecutable:{error}", output, expected)
    ok = value == wanted
    reason = "" if ok else f"wrong_value:{value!r}_vs_{wanted!r}"
    return OracleResult(ok, reason, str(value), str(wanted))


def _c_type_handler(resolution, output, expected, grammar) -> OracleResult:
    """Grade c output against a real C compiler: the c grammar's concrete
    syntax is real, compilable C, so `cc -fsyntax-only` is the oracle (the C
    analogue of aufbau's OCaml differential certification harness)."""
    from benchmarks.c_oracle import compiles

    ok, reason = compiles(output)
    return OracleResult(ok, reason, output, expected)


def _stlc_equiv_handler(resolution, output, expected, grammar) -> OracleResult:
    expected_type = resolution.get("type")
    if expected_type:
        observed_type = stlc_type_of(output)
        normalized_expected_type = format_stlc_type(
            parse_stlc_type_text(str(expected_type))
        )
        if observed_type != normalized_expected_type:
            return OracleResult(
                False,
                "type_mismatch",
                observed_type,
                normalized_expected_type,
            )
    norms = {str(x).lower() for x in resolution.get("normalization", [])}
    if "beta" in norms or "alpha" in norms:
        ok = stlc_equivalent(output, expected)
        return OracleResult(ok, "" if ok else "beta_alpha_mismatch")
    ok = _normalize_text(output) == _normalize_text(expected)
    return OracleResult(
        ok, "" if ok else "equivalence_mismatch", output, expected
    )


_DEFAULT_GRAMMAR = "<default>"


GRAMMAR_SPECS: dict[str, GrammarSpec] = {
    "stlc": GrammarSpec(
        name="stlc",
        modes=frozenset({"exact", "equivalence"}),
        list_fields=frozenset({"normalization"}),
        type_fields=frozenset({"type"}),
        handlers={
            "exact": _wrap("stlc")(_exact),
            "equivalence": _wrap("stlc")(_stlc_equiv_handler),
        },
    ),
    "ml": GrammarSpec(
        name="ml",
        modes=frozenset({"exact", "eval", "type", "equivalence"}),
        required={"eval": frozenset({"expected_value"})},
        type_fields=frozenset({"type"}),
        handlers={
            "exact": _wrap("ml")(_exact),
            "eval": _wrap("ml")(_ml_eval_handler),
            "type": _wrap("ml")(_ml_legacy),
            "equivalence": _wrap("ml")(_ml_legacy),
        },
    ),
    "c": GrammarSpec(
        name="c",
        modes=frozenset({"exact", "type"}),
        handlers={
            "exact": _wrap("c")(_exact),
            "type": _wrap("c")(_c_type_handler),
        },
    ),
    "tool": GrammarSpec(
        name="tool",
        modes=frozenset({"exact", "type", "value"}),
        required={"value": frozenset({"expected_value"})},
        handlers={
            "type": _wrap("tool")(_tool_type_handler),
            "value": _wrap("tool")(_tool_value_handler),
        },
    ),
    "tool_sexpr": GrammarSpec(
        name="tool_sexpr",
        modes=frozenset({"exact", "type", "value"}),
        required={"value": frozenset({"expected_value"})},
        handlers={
            "type": _wrap("tool_sexpr")(_tool_type_handler),
            "value": _wrap("tool_sexpr")(_tool_value_handler),
        },
    ),
    _DEFAULT_GRAMMAR: GrammarSpec(
        name="default",
        modes=frozenset({"exact", "equivalence"}),
        handlers={
            "exact": _exact,
            "equivalence": _default_mismatch,
        },
    ),
}


def validate_resolution(
    path: Any,
    task_id: str,
    grammar: str,
    resolution: dict[str, Any],
    kind: str = "single_shot",
) -> None:
    """Validate a task's [resolution] against the registry, so a new grammar
    is declared once (in GRAMMAR_SPECS) and validation follows automatically.

    `kind == "agent"` is the only non-grammar rule: agent episodes always
    resolve by `episode` mode + expected_value, regardless of grammar.
    """
    mode = resolution.get("mode")
    if not isinstance(mode, str) or not mode.strip():
        raise ValueError(
            f"{path}: {task_id}: resolution.mode must be a non-empty string"
        )

    if kind == "agent":
        if mode != "episode":
            raise ValueError(
                f"{path}: {task_id}: agent tasks require resolution.mode = "
                f"'episode', got {mode!r}"
            )
        if "expected_value" not in resolution:
            raise ValueError(
                f"{path}: {task_id}: episode mode requires resolution.expected_value"
            )
        return

    base = canonical_grammar(grammar)
    spec = GRAMMAR_SPECS.get(base, GRAMMAR_SPECS[_DEFAULT_GRAMMAR])
    if mode not in spec.modes:
        raise ValueError(
            f"{path}: {task_id}: unsupported {grammar} resolution mode {mode!r}"
        )
    for required in spec.required.get(mode, frozenset()):
        if not str(resolution.get(required) or "").strip():
            raise ValueError(
                f"{path}: {task_id}: {mode} mode requires resolution.{required}"
            )
    for field in spec.list_fields:
        if field in resolution and not isinstance(resolution[field], list):
            raise ValueError(f"{path}: {task_id}: resolution.{field} must be a list")
    for field in spec.type_fields:
        if field in resolution and not str(resolution.get(field) or "").strip():
            raise ValueError(
                f"{path}: {task_id}: resolution.{field} must be non-empty"
            )


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# STLC beta/alpha equivalence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class Lam:
    name: str
    body: Any


@dataclass(frozen=True)
class App:
    func: Any
    arg: Any


@dataclass(frozen=True)
class TyBase:
    name: str


@dataclass(frozen=True)
class TyFun:
    left: Any
    right: Any


@dataclass(frozen=True)
class TLam:
    name: str
    arg_type: Any
    body: Any


@dataclass(frozen=True)
class TVar:
    name: str


@dataclass(frozen=True)
class TApp:
    func: Any
    arg: Any


class TokenStream:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.index = 0

    def peek(self) -> Optional[str]:
        if self.index >= len(self.tokens):
            return None
        return self.tokens[self.index]

    def pop(self, expected: Optional[str] = None) -> str:
        token = self.peek()
        if token is None:
            raise ValueError("unexpected end of input")
        if expected is not None and token != expected:
            raise ValueError(f"expected {expected!r}, got {token!r}")
        self.index += 1
        return token

    def done(self) -> bool:
        return self.peek() is None


def stlc_equivalent(left: str, right: str) -> bool:
    left_key = alpha_key(beta_normalize(parse_stlc(left)))
    right_key = alpha_key(beta_normalize(parse_stlc(right)))
    return left_key == right_key


def stlc_type_of(text: str) -> str:
    stream = TokenStream(_tokenize_lambda(text))
    term = _parse_typed_stlc_expr(stream)
    if not stream.done():
        raise ValueError(f"unexpected token {stream.peek()!r}")
    return format_stlc_type(_infer_stlc_type(term, {}))


def parse_stlc_type_text(text: str) -> Any:
    stream = TokenStream(_tokenize_lambda(text))
    ty = _parse_stlc_type(stream)
    if not stream.done():
        raise ValueError(f"unexpected type token {stream.peek()!r}")
    return ty


def format_stlc_type(ty: Any) -> str:
    if isinstance(ty, TyBase):
        return ty.name
    if isinstance(ty, TyFun):
        left = format_stlc_type(ty.left)
        if isinstance(ty.left, TyFun):
            left = f"({left})"
        return f"{left} -> {format_stlc_type(ty.right)}"
    raise TypeError(ty)


def _parse_typed_stlc_expr(stream: TokenStream) -> Any:
    if stream.peek() == "λ":
        return _parse_typed_stlc_lambda(stream)
    term = _parse_typed_stlc_atom(stream)
    while _starts_stlc_atom(stream.peek()):
        term = TApp(term, _parse_typed_stlc_atom(stream))
    return term


def _parse_typed_stlc_atom(stream: TokenStream) -> Any:
    token = stream.peek()
    if token == "(":
        stream.pop("(")
        term = _parse_typed_stlc_expr(stream)
        stream.pop(")")
        return term
    if token == "λ":
        return _parse_typed_stlc_lambda(stream)
    if token and re.match(r"[A-Za-z_]", token):
        return TVar(stream.pop())
    raise ValueError(f"expected typed STLC atom, got {token!r}")


def _parse_typed_stlc_lambda(stream: TokenStream) -> Any:
    stream.pop("λ")
    name = stream.pop()
    stream.pop(":")
    arg_type = _parse_stlc_type(stream)
    stream.pop(".")
    return TLam(name, arg_type, _parse_typed_stlc_expr(stream))


def _parse_stlc_type(stream: TokenStream) -> Any:
    left = _parse_stlc_type_atom(stream)
    if stream.peek() in {"->", "→"}:
        stream.pop()
        return TyFun(left, _parse_stlc_type(stream))
    return left


def _parse_stlc_type_atom(stream: TokenStream) -> Any:
    token = stream.peek()
    if token == "(":
        stream.pop("(")
        ty = _parse_stlc_type(stream)
        stream.pop(")")
        return ty
    if token and re.match(r"[A-Za-z_]", token):
        return TyBase(stream.pop())
    raise ValueError(f"expected type atom, got {token!r}")


def _infer_stlc_type(term: Any, env: dict[str, Any]) -> Any:
    if isinstance(term, TVar):
        if term.name not in env:
            raise ValueError(f"unbound variable {term.name}")
        return env[term.name]
    if isinstance(term, TLam):
        new_env = dict(env)
        new_env[term.name] = term.arg_type
        return TyFun(term.arg_type, _infer_stlc_type(term.body, new_env))
    if isinstance(term, TApp):
        func_type = _infer_stlc_type(term.func, env)
        arg_type = _infer_stlc_type(term.arg, env)
        if not isinstance(func_type, TyFun):
            raise ValueError(
                f"application of non-function type {format_stlc_type(func_type)}"
            )
        if func_type.left != arg_type:
            raise ValueError(
                f"argument type mismatch: expected {format_stlc_type(func_type.left)}, got {format_stlc_type(arg_type)}"
            )
        return func_type.right
    raise TypeError(term)


def parse_stlc(text: str) -> Any:
    stream = TokenStream(_tokenize_lambda(text))
    term = _parse_stlc_expr(stream)
    if not stream.done():
        raise ValueError(f"unexpected token {stream.peek()!r}")
    return term


def _tokenize_lambda(text: str) -> list[str]:
    return re.findall(r"λ|->|[().:,]|[A-Za-z_][A-Za-z0-9_]*", text)


def _parse_stlc_expr(stream: TokenStream) -> Any:
    if stream.peek() == "λ":
        return _parse_stlc_lambda(stream)
    term = _parse_stlc_atom(stream)
    while _starts_stlc_atom(stream.peek()):
        term = App(term, _parse_stlc_atom(stream))
    return term


def _starts_stlc_atom(token: Optional[str]) -> bool:
    return token == "(" or token == "λ" or bool(token and re.match(r"[A-Za-z_]", token))


def _parse_stlc_atom(stream: TokenStream) -> Any:
    token = stream.peek()
    if token == "(":
        stream.pop("(")
        term = _parse_stlc_expr(stream)
        stream.pop(")")
        return term
    if token == "λ":
        return _parse_stlc_lambda(stream)
    if token and re.match(r"[A-Za-z_]", token):
        return Var(stream.pop())
    raise ValueError(f"expected atom, got {token!r}")


def _parse_stlc_lambda(stream: TokenStream) -> Any:
    stream.pop("λ")
    name = stream.pop()
    stream.pop(":")
    _skip_type(stream)
    stream.pop(".")
    return Lam(name, _parse_stlc_expr(stream))


def _skip_type(stream: TokenStream) -> None:
    depth = 0
    while True:
        token = stream.peek()
        if token is None:
            raise ValueError("unterminated type annotation")
        if token == "." and depth == 0:
            return
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
        stream.pop()


def free_vars(term: Any) -> set[str]:
    if isinstance(term, Var):
        return {term.name}
    if isinstance(term, App):
        return free_vars(term.func) | free_vars(term.arg)
    if isinstance(term, Lam):
        return free_vars(term.body) - {term.name}
    return set()


def subst(term: Any, name: str, value: Any) -> Any:
    if isinstance(term, Var):
        return value if term.name == name else term
    if isinstance(term, App):
        return App(subst(term.func, name, value), subst(term.arg, name, value))
    if isinstance(term, Lam):
        if term.name == name:
            return term
        if term.name in free_vars(value):
            fresh = _fresh_name(
                term.name, free_vars(term.body) | free_vars(value) | {name}
            )
            renamed = subst(term.body, term.name, Var(fresh))
            return Lam(fresh, subst(renamed, name, value))
        return Lam(term.name, subst(term.body, name, value))
    return term


def _fresh_name(base: str, used: set[str]) -> str:
    index = 0
    while True:
        candidate = f"{base}_{index}"
        if candidate not in used:
            return candidate
        index += 1


def beta_normalize(term: Any, fuel: int = 1000) -> Any:
    current = term
    for _ in range(fuel):
        nxt = _beta_step(current)
        if nxt == current:
            return current
        current = nxt
    raise ValueError("beta reduction did not terminate")


def _beta_step(term: Any) -> Any:
    if isinstance(term, App) and isinstance(term.func, Lam):
        return subst(term.func.body, term.func.name, term.arg)
    if isinstance(term, App):
        new_func = _beta_step(term.func)
        if new_func != term.func:
            return App(new_func, term.arg)
        new_arg = _beta_step(term.arg)
        if new_arg != term.arg:
            return App(term.func, new_arg)
        return term
    if isinstance(term, Lam):
        new_body = _beta_step(term.body)
        if new_body != term.body:
            return Lam(term.name, new_body)
    return term


def alpha_key(term: Any, env: Optional[list[str]] = None) -> Any:
    env = env or []
    if isinstance(term, Var):
        if term.name in env:
            return ("bound", len(env) - 1 - env[::-1].index(term.name))
        return ("free", term.name)
    if isinstance(term, App):
        return ("app", alpha_key(term.func, env), alpha_key(term.arg, env))
    if isinstance(term, Lam):
        return ("lam", alpha_key(term.body, env + [term.name]))
    raise TypeError(term)
