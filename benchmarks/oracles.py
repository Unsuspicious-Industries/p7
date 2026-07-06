from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class OracleResult:
    ok: bool
    reason: str = ""
    observed: Any = None
    expected: Any = None


def check_resolution(task: Any, output: str) -> OracleResult:
    resolution = dict(getattr(task, "resolution", {}) or {})
    mode = str(resolution.get("mode") or "exact")
    expected = str(getattr(task, "expected", ""))
    grammar = str(getattr(task, "grammar", ""))

    # A syntactic twin (e.g. imp_syntactic) constrains generation with the
    # typing rules stripped, but is still graded by its semantic base oracle:
    # the study question is whether syntactic-only pruning yields programs the
    # full checker accepts.
    try:
        import proposition7

        grammar = proposition7.base_grammar(grammar)
    except Exception:
        if grammar.endswith("_syntactic"):
            grammar = grammar[: -len("_syntactic")]

    if grammar == "stlc":
        return _check_stlc_resolution(mode, resolution, output, expected)
    if grammar == "ml":
        return _check_ml_resolution(mode, resolution, output, expected)
    if grammar == "c":
        return _check_c_resolution(mode, resolution, output, expected)
    if grammar in {"tool", "tool_sexpr"}:
        return _check_tool_resolution(mode, resolution, output, expected, grammar)
    return _check_default_resolution(mode, resolution, output, expected)


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


def _check_ml_resolution(
    mode: str,
    resolution: dict[str, Any],
    output: str,
    expected: str,
) -> OracleResult:
    """Grade ml output with the type system itself. `type` mode requires the
    output to be well-typed and to carry the declared type. `exact` is text
    match.

    The type check is done by *ascription*, not by rendering the inferred type
    and comparing strings: it asks the engine whether `let chk : T = <output>
    in chk` type-checks. This routes the comparison through the grammar's own
    checker (so it is exact up to the rewrite theory) and sidesteps a known
    rendering bug where a function-type domain loses its parentheses
    (`(int -> int) -> int` would print as `int -> int -> int`, a different
    type)."""
    try:
        if mode == "exact":
            ok = _normalize_text(output) == _normalize_text(expected)
            return OracleResult(ok, "" if ok else "exact_mismatch", output, expected)

        if mode in {"type", "equivalence"}:
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
    except Exception as error:  # noqa: BLE001 - oracle must never crash the run
        return OracleResult(False, f"ml_resolution_error: {error}")

    return OracleResult(False, f"unsupported_resolution_mode_for_ml: {mode}")


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


def _check_tool_resolution(
    mode: str,
    resolution: dict[str, Any],
    output: str,
    expected: str,
    grammar: str = "tool",
) -> OracleResult:
    """Grade tool-DSL output. `type` mode checks only well-typedness against
    the fixed tool registry (search/summarize/count/format) -- informative
    for the *unconstrained* arm's failure decomposition, but circular as a
    pass/fail criterion for the constrained arm, which is decoded under this
    exact checker (see lmpl-plan.md section 5.1): a completed constrained
    output is well-typed by construction, so `type` mode there would measure
    nothing. `value` mode is the real grading criterion: run the program
    against the same deterministic mock registry (benchmarks/tool_registry)
    and compare the *executed* return value to `resolution["expected_value"]`
    -- task accomplishment, not typedness."""
    try:
        if mode == "exact":
            ok = _normalize_text(output) == _normalize_text(expected)
            return OracleResult(ok, "" if ok else "exact_mismatch", output, expected)

        if mode == "type":
            import aufbau

            status = aufbau.Synthesizer.from_grammar(_tool_spg(grammar), output).status()
            ok = status == "typed"
            return OracleResult(ok, "" if ok else f"not_well_typed:{status}", output, expected)

        if mode == "value":
            from benchmarks.tool_registry import InterpretError, run_program

            wanted = resolution.get("expected_value")
            try:
                value, _env = run_program(output, syntax=grammar)
            except InterpretError as error:
                return OracleResult(False, f"unexecutable:{error}", output, expected)
            ok = value == wanted
            reason = "" if ok else f"wrong_value:{value!r}_vs_{wanted!r}"
            return OracleResult(ok, reason, str(value), str(wanted))
    except Exception as error:  # noqa: BLE001 - oracle must never crash the run
        return OracleResult(False, f"tool_resolution_error: {error}")

    return OracleResult(False, f"unsupported_resolution_mode_for_tool: {mode}")


def _check_default_resolution(
    mode: str,
    resolution: dict[str, Any],
    output: str,
    expected: str,
) -> OracleResult:
    del resolution
    if mode in {"exact", "equivalence"}:
        ok = _normalize_text(output) == _normalize_text(expected)
        reason = (
            ""
            if ok
            else ("exact_mismatch" if mode == "exact" else "equivalence_mismatch")
        )
        return OracleResult(ok, reason, output, expected)
    return OracleResult(False, f"unsupported_resolution_mode_for_default: {mode}")


def _check_stlc_resolution(
    mode: str,
    resolution: dict[str, Any],
    output: str,
    expected: str,
) -> OracleResult:
    try:
        if mode == "exact":
            ok = _normalize_text(output) == _normalize_text(expected)
            return OracleResult(ok, "" if ok else "exact_mismatch", output, expected)

        if mode == "equivalence":
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
    except Exception as error:
        return OracleResult(False, f"stlc_resolution_error: {error}")

    return OracleResult(False, f"unsupported_resolution_mode_for_stlc: {mode}")


def _check_c_resolution(
    mode: str,
    resolution: dict[str, Any],
    output: str,
    expected: str,
) -> OracleResult:
    """Grade c output against a real C compiler: the c grammar's concrete
    syntax is real, compilable C, so `cc -fsyntax-only` is the oracle (the C
    analogue of aufbau's OCaml differential certification harness)."""
    from benchmarks.c_oracle import compiles

    try:
        if mode == "exact":
            ok = _normalize_text(output) == _normalize_text(expected)
            return OracleResult(ok, "" if ok else "exact_mismatch", output, expected)

        if mode == "type":
            ok, reason = compiles(output)
            return OracleResult(ok, reason, output, expected)
    except Exception as error:  # noqa: BLE001 - oracle must never crash the run
        return OracleResult(False, f"c_resolution_error: {error}")

    return OracleResult(False, f"unsupported_resolution_mode_for_c: {mode}")


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
