from __future__ import annotations

import math
import hashlib
import json
import random
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

    if grammar == "stlc":
        return _check_stlc_resolution(mode, resolution, output, expected)
    if grammar == "fun":
        return _check_fun_resolution(mode, resolution, output, expected)
    if grammar == "imp":
        return _check_imp_resolution(mode, resolution, output, expected)
    if grammar == "lamb":
        return _check_lamb_resolution(mode, resolution, output, expected)
    return _check_default_resolution(mode, resolution, output, expected)


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


def _check_fun_resolution(
    mode: str,
    resolution: dict[str, Any],
    output: str,
    expected: str,
) -> OracleResult:
    try:
        if mode == "samples":
            ok, reason, observed, expected_observed = _check_fun_samples(
                output_code=output,
                expected_code=expected,
                samples_spec=resolution.get("samples"),
                fn_name=resolution.get("fn"),
                hidden_cases=resolution.get("hidden_samples", 16),
                seed_hint=resolution.get("seed_hint"),
            )
            if ok and resolution.get("structure"):
                expr = parse_fun(output)
                ok_struct, struct_reason = _check_fun_structure(
                    expr, resolution["structure"]
                )
                if not ok_struct:
                    return OracleResult(False, f"structure_mismatch:{struct_reason}")
            return OracleResult(ok, reason, observed, expected_observed)

        if mode in {"value", "equivalence"}:
            samples = resolution.get("samples")
            if samples:
                ok, reason, observed, expected_observed = _check_fun_samples(
                    output_code=output,
                    expected_code=expected,
                    samples_spec=samples,
                    fn_name=resolution.get("fn"),
                    hidden_cases=resolution.get("hidden_samples", 16),
                    seed_hint=resolution.get("seed_hint"),
                )
                if ok and resolution.get("structure"):
                    expr = parse_fun(output)
                    ok_struct, struct_reason = _check_fun_structure(
                        expr, resolution["structure"]
                    )
                    if not ok_struct:
                        return OracleResult(
                            False,
                            f"structure_mismatch:{struct_reason}",
                            observed,
                            expected_observed,
                        )
                return OracleResult(ok, reason, observed, expected_observed)
            observed = eval_fun(output)
            observed_json = _value_to_json(observed)
            expected_value = resolution.get("value")
            if expected_value is not None:
                expected_json = _value_to_json(expected_value)
                ok = _values_equal(observed_json, expected_json)
                reason = (
                    ""
                    if ok
                    else (
                        "value_mismatch" if mode == "value" else "equivalence_mismatch"
                    )
                )
            else:
                try:
                    expected_eval = eval_fun(expected)
                    expected_json = _value_to_json(expected_eval)
                    if isinstance(observed, Closure) or isinstance(
                        expected_eval, Closure
                    ):
                        ok = fun_alpha_equivalent(
                            output,
                            expected,
                            fn_name=resolution.get("fn"),
                        )
                        reason = "" if ok else "equivalence_mismatch"
                        observed_json = "<fun-alpha>"
                        expected_json = "<fun-alpha>"
                    else:
                        ok = _values_equal(observed_json, expected_json)
                        reason = "" if ok else "equivalence_mismatch"
                except Exception:
                    ok = observed_json != "<closure>"
                    reason = "" if ok else "result_is_closure_expected_value"
            if ok and resolution.get("structure"):
                expr = parse_fun(output)
                ok_struct, struct_reason = _check_fun_structure(
                    expr, resolution["structure"]
                )
                if not ok_struct:
                    return OracleResult(
                        False,
                        f"structure_mismatch:{struct_reason}",
                        observed_json,
                        expected_value,
                    )
                reason = (
                    ""
                    if ok
                    else (
                        "value_mismatch" if mode == "value" else "equivalence_mismatch"
                    )
                )
            return OracleResult(ok, reason, observed_json, expected_value)

        if mode == "exact":
            ok = _normalize_text(output) == _normalize_text(expected)
            return OracleResult(ok, "" if ok else "exact_mismatch", output, expected)
    except Exception as error:
        return OracleResult(False, f"fun_resolution_error: {error}")

    return OracleResult(False, f"unsupported_resolution_mode_for_fun: {mode}")


def _check_imp_resolution(
    mode: str,
    resolution: dict[str, Any],
    output: str,
    expected: str,
) -> OracleResult:
    try:
        if mode == "samples":
            ok, reason, observed, expected_observed = _check_imp_samples(
                output_code=output,
                expected_code=expected,
                resolution=resolution,
            )
            if ok and resolution.get("structure"):
                ok_struct, struct_reason = _check_imp_structure(
                    output, resolution["structure"]
                )
                if not ok_struct:
                    return OracleResult(
                        False,
                        f"structure_mismatch:{struct_reason}",
                        observed,
                        expected_observed,
                    )
            return OracleResult(ok, reason, observed, expected_observed)

        if mode == "env":
            observed_env = eval_imp(output)
            expected_env = resolution.get("env") or eval_imp(expected)
            variables = resolution.get("vars") or sorted(expected_env)
            observed = {name: observed_env.get(name) for name in variables}
            expected_selected = {name: expected_env.get(name) for name in variables}
            ok = _values_equal(observed, expected_selected)
            return OracleResult(
                ok, "" if ok else "env_mismatch", observed, expected_selected
            )

        if mode == "exact":
            ok = _normalize_text(output) == _normalize_text(expected)
            return OracleResult(ok, "" if ok else "exact_mismatch", output, expected)
    except Exception as error:
        return OracleResult(False, f"imp_resolution_error: {error}")

    return OracleResult(False, f"unsupported_resolution_mode_for_imp: {mode}")


def _check_lamb_resolution(
    mode: str,
    resolution: dict[str, Any],
    output: str,
    expected: str,
) -> OracleResult:
    del resolution
    if mode in {"equivalence", "exact"}:
        ok = _normalize_text(output) == _normalize_text(expected)
        reason = (
            ""
            if ok
            else ("exact_mismatch" if mode == "exact" else "equivalence_mismatch")
        )
        return OracleResult(ok, reason, output, expected)
    return OracleResult(False, f"unsupported_resolution_mode_for_lamb: {mode}")


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _value_to_json(value: Any) -> Any:
    if isinstance(value, Closure):
        return "<closure>"
    if isinstance(value, float):
        return round(value, 8)
    if isinstance(value, dict):
        return {key: _value_to_json(val) for key, val in sorted(value.items())}
    return value


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        return math.isclose(float(left), float(right), rel_tol=1e-7, abs_tol=1e-7)
    return left == right


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


# ---------------------------------------------------------------------------
# FUN interpreter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Closure:
    name: str
    body: Any
    env: dict[str, Any]


@dataclass(frozen=True)
class FInt:
    value: int


@dataclass(frozen=True)
class FFloat:
    value: float


@dataclass(frozen=True)
class FBool:
    value: bool


@dataclass(frozen=True)
class FVar:
    name: str


@dataclass(frozen=True)
class FLet:
    name: str
    value: Any
    body: Any


@dataclass(frozen=True)
class FLam:
    name: str
    body: Any


@dataclass(frozen=True)
class FApp:
    func: Any
    arg: Any


@dataclass(frozen=True)
class FBin:
    op: str
    left: Any
    right: Any


def parse_fun(text: str) -> Any:
    stream = TokenStream(_tokenize_fun(text))
    expr = _parse_fun_expr(stream)
    if not stream.done():
        raise ValueError(f"unexpected token {stream.peek()!r}")
    return expr


def fun_alpha_equivalent(left: str, right: str, fn_name: Any = None) -> bool:
    left_expr = _extract_fun_target_expr(parse_fun(left), fn_name)
    right_expr = _extract_fun_target_expr(parse_fun(right), fn_name)
    return _fun_alpha_key(left_expr) == _fun_alpha_key(right_expr)


def eval_fun(text: str) -> Any:
    expr = parse_fun(text)
    return _eval_fun(expr, {})


def _check_fun_structure(expr: Any, structure: dict[str, Any]) -> tuple[bool, str]:
    if "let_bindings" in structure:
        if _count_fun_lets(expr) != int(structure["let_bindings"]):
            return False, "let_bindings_mismatch"
    if "applications" in structure:
        if _count_fun_apps(expr) != int(structure["applications"]):
            return False, "applications_mismatch"
    if "lambdas" in structure:
        if _count_fun_lambdas(expr) != int(structure["lambdas"]):
            return False, "lambdas_mismatch"
    if "operations" in structure:
        ops = structure["operations"]
        found_ops = _fun_expr_ops(expr)
        if not set(ops) <= found_ops:
            return False, "operations_mismatch"
    return True, ""


def _extract_fun_target_expr(expr: Any, fn_name: Any) -> Any:
    requested_name = str(fn_name).strip() if fn_name is not None else ""
    if requested_name:
        found = _find_fun_let_value(expr, requested_name)
        if found is None:
            raise ValueError(f"requested function '{requested_name}' not found")
        return found

    if isinstance(expr, FLet):
        tail = expr.body
        while isinstance(tail, FLet):
            tail = tail.body
        if isinstance(tail, FVar):
            found = _find_fun_let_value(expr, tail.name)
            if found is not None:
                return found
    return expr


def _find_fun_let_value(expr: Any, name: str) -> Any:
    cur = expr
    while isinstance(cur, FLet):
        if cur.name == name:
            return cur.value
        cur = cur.body
    return None


def _fun_alpha_key(expr: Any, env: Optional[list[str]] = None) -> Any:
    env = env or []
    if isinstance(expr, FInt):
        return ("int", expr.value)
    if isinstance(expr, FFloat):
        return ("float", round(expr.value, 8))
    if isinstance(expr, FBool):
        return ("bool", expr.value)
    if isinstance(expr, FVar):
        if expr.name in env:
            return ("bound", len(env) - 1 - env[::-1].index(expr.name))
        return ("free", expr.name)
    if isinstance(expr, FBin):
        return (
            "bin",
            expr.op,
            _fun_alpha_key(expr.left, env),
            _fun_alpha_key(expr.right, env),
        )
    if isinstance(expr, FApp):
        return ("app", _fun_alpha_key(expr.func, env), _fun_alpha_key(expr.arg, env))
    if isinstance(expr, FLam):
        return ("lam", _fun_alpha_key(expr.body, env + [expr.name]))
    if isinstance(expr, FLet):
        return (
            "let",
            _fun_alpha_key(expr.value, env),
            _fun_alpha_key(expr.body, env + [expr.name]),
        )
    raise TypeError(expr)


@dataclass(frozen=True)
class FunSampleCase:
    args: tuple[Any, ...]
    expected: Any


def validate_fun_samples_schema(samples_spec: Any) -> list[FunSampleCase]:
    if not isinstance(samples_spec, list) or not samples_spec:
        raise ValueError("samples must be a non-empty list")

    normalized: list[FunSampleCase] = []
    arity: Optional[int] = None

    for idx, raw_case in enumerate(samples_spec):
        if not isinstance(raw_case, dict):
            raise ValueError(f"samples[{idx}] must be a table/object")
        if "v" not in raw_case:
            raise ValueError(f"samples[{idx}] is missing expected value key 'v'")

        if "args" in raw_case:
            extra_keys = [key for key in raw_case if key not in {"args", "v"}]
            if extra_keys:
                extras = ", ".join(sorted(extra_keys))
                raise ValueError(
                    f"samples[{idx}] uses args-form and cannot contain extra keys: {extras}"
                )
            args_raw = raw_case["args"]
            if not isinstance(args_raw, list):
                raise ValueError(f"samples[{idx}].args must be a list")
            args = tuple(args_raw)
        else:
            arg_keys = [key for key in raw_case if key != "v"]
            if not arg_keys:
                raise ValueError(f"samples[{idx}] must provide args or named inputs")
            args = tuple(raw_case[key] for key in arg_keys)

        if arity is None:
            arity = len(args)
        elif len(args) != arity:
            raise ValueError(
                f"samples[{idx}] arity mismatch: expected {arity}, got {len(args)}"
            )

        for arg_index, arg_value in enumerate(args):
            if not _is_fun_scalar(arg_value):
                raise ValueError(
                    f"samples[{idx}] arg {arg_index} has unsupported type {type(arg_value).__name__}"
                )
        if not _is_fun_scalar(raw_case["v"]):
            raise ValueError(
                f"samples[{idx}] expected value 'v' has unsupported type {type(raw_case['v']).__name__}"
            )

        normalized.append(FunSampleCase(args=args, expected=raw_case["v"]))

    return normalized


def _check_fun_samples(
    *,
    output_code: str,
    expected_code: str,
    samples_spec: Any,
    fn_name: Any,
    hidden_cases: Any,
    seed_hint: Any,
) -> tuple[bool, str, Any, Any]:
    cases = validate_fun_samples_schema(samples_spec)
    candidate = _extract_fun_callable(output_code, fn_name)
    reference = _extract_fun_callable(expected_code, fn_name)

    explicit_observed: list[Any] = []
    explicit_expected: list[Any] = []
    for idx, case in enumerate(cases):
        reference_value = _apply_fun_callable(reference, case.args)
        if not _fun_values_equal(reference_value, case.expected):
            return (
                False,
                f"sample_spec_mismatch:case_{idx}",
                _value_to_json(reference_value),
                case.expected,
            )

        observed_value = _apply_fun_callable(candidate, case.args)
        explicit_observed.append(_value_to_json(observed_value))
        explicit_expected.append(_value_to_json(reference_value))
        if not _fun_values_equal(observed_value, reference_value):
            return (
                False,
                f"sample_fail:case_{idx}",
                _value_to_json(observed_value),
                _value_to_json(reference_value),
            )

    hidden_count = _normalize_hidden_case_count(hidden_cases)
    hidden_args = _generate_hidden_fun_args(
        cases,
        count=hidden_count,
        seed_hint=seed_hint,
        expected_code=expected_code,
    )

    for idx, args in enumerate(hidden_args):
        reference_value = _apply_fun_callable(reference, args)
        observed_value = _apply_fun_callable(candidate, args)
        if not _fun_values_equal(observed_value, reference_value):
            return (
                False,
                f"hidden_sample_fail:case_{idx}",
                _value_to_json(observed_value),
                _value_to_json(reference_value),
            )

    observed_meta = {
        "explicit": explicit_observed,
        "hidden_checked": len(hidden_args),
    }
    expected_meta = {
        "explicit": explicit_expected,
        "hidden_checked": len(hidden_args),
    }
    return True, "", observed_meta, expected_meta


def _is_fun_scalar(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return True
    return False


def _normalize_hidden_case_count(raw_value: Any) -> int:
    if raw_value is None:
        return 16
    try:
        count = int(raw_value)
    except Exception as error:
        raise ValueError(f"hidden_samples must be an int: {error}") from error
    if count < 0:
        raise ValueError("hidden_samples must be >= 0")
    return min(count, 256)


def _extract_fun_callable(code: str, fn_name: Any) -> Closure:
    expr = parse_fun(code)
    env: dict[str, Any] = {}
    found_names: list[str] = []
    found_values: dict[str, Closure] = {}
    cur = expr

    while isinstance(cur, FLet):
        value = _eval_fun(cur.value, env)
        env[cur.name] = value
        if isinstance(value, Closure):
            found_names.append(cur.name)
            found_values[cur.name] = value
        cur = cur.body

    tail_value: Any = None
    try:
        tail_value = _eval_fun(cur, env)
    except Exception:
        tail_value = None

    requested_name = str(fn_name).strip() if fn_name is not None else ""
    if requested_name:
        chosen = found_values.get(requested_name)
        if chosen is None:
            raise ValueError(f"requested function '{requested_name}' not found")
        return chosen

    if isinstance(tail_value, Closure):
        return tail_value

    if len(found_names) == 1:
        return found_values[found_names[0]]
    if len(found_names) > 1:
        return found_values[found_names[-1]]
    raise ValueError("no callable function found in program")


def _apply_fun_callable(func: Closure, args: tuple[Any, ...]) -> Any:
    value: Any = func
    for arg in args:
        if not isinstance(value, Closure):
            raise ValueError("function_arity_mismatch")
        call_env = dict(value.env)
        call_env[value.name] = arg
        value = _eval_fun(value.body, call_env)
    return value


def _fun_values_equal(left: Any, right: Any) -> bool:
    left_json = _value_to_json(left)
    right_json = _value_to_json(right)
    if isinstance(left_json, float) or isinstance(right_json, float):
        return math.isclose(
            float(left_json), float(right_json), rel_tol=1e-7, abs_tol=1e-7
        )
    return left_json == right_json


def _generate_hidden_fun_args(
    cases: list[FunSampleCase],
    *,
    count: int,
    seed_hint: Any,
    expected_code: str,
) -> list[tuple[Any, ...]]:
    if count <= 0:
        return []

    arity = len(cases[0].args)
    existing = {tuple(case.args) for case in cases}
    kinds = _infer_fun_arg_kinds(cases)
    seed_material = {
        "seed_hint": seed_hint,
        "arity": arity,
        "kinds": kinds,
        "cases": [list(case.args) + [case.expected] for case in cases],
        "expected_code": expected_code,
    }
    payload = json.dumps(seed_material, sort_keys=True, ensure_ascii=True)
    seed = int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)

    hidden: list[tuple[Any, ...]] = []
    max_attempts = max(count * 20, 100)
    attempts = 0
    while len(hidden) < count and attempts < max_attempts:
        attempts += 1
        candidate = tuple(_draw_hidden_value(kind, rng) for kind in kinds)
        if candidate in existing:
            continue
        existing.add(candidate)
        hidden.append(candidate)
    return hidden


def _infer_fun_arg_kinds(cases: list[FunSampleCase]) -> list[str]:
    arity = len(cases[0].args)
    kinds: list[str] = []
    for arg_index in range(arity):
        values = [case.args[arg_index] for case in cases]
        if all(isinstance(value, bool) for value in values):
            kinds.append("bool")
            continue
        if all(
            isinstance(value, int) and not isinstance(value, bool) for value in values
        ):
            kinds.append("int")
            continue
        if all(
            (isinstance(value, int) and not isinstance(value, bool))
            or isinstance(value, float)
            for value in values
        ):
            kinds.append("float")
            continue
        raise ValueError(
            f"unsupported mixed argument types in samples at position {arg_index}"
        )
    return kinds


def _draw_hidden_value(kind: str, rng: random.Random) -> Any:
    if kind == "bool":
        return bool(rng.randint(0, 1))
    if kind == "int":
        anchors = [-64, -17, -5, -1, 0, 1, 2, 7, 13, 42, 64]
        if rng.random() < 0.35:
            return anchors[rng.randrange(len(anchors))]
        return rng.randint(-128, 128)
    if kind == "float":
        anchors = [-10.0, -1.5, -0.5, 0.0, 0.5, 1.25, 3.5, 10.0, 42.25]
        if rng.random() < 0.35:
            return anchors[rng.randrange(len(anchors))]
        return round(rng.uniform(-128.0, 128.0), 3)
    raise ValueError(f"unknown hidden sample kind: {kind}")


@dataclass(frozen=True)
class ImpSampleCase:
    inputs: dict[str, Any]
    outputs: dict[str, Any]


def validate_imp_samples_schema(
    samples_spec: Any, input_vars: Any, output_vars: Any
) -> list[ImpSampleCase]:
    if (
        not isinstance(input_vars, list)
        or not input_vars
        or not all(isinstance(name, str) and name for name in input_vars)
    ):
        raise ValueError("input_vars must be a non-empty list of variable names")
    if (
        not isinstance(output_vars, list)
        or not output_vars
        or not all(isinstance(name, str) and name for name in output_vars)
    ):
        raise ValueError("vars must be a non-empty list of output variable names")
    if set(input_vars) & set(output_vars):
        raise ValueError("input_vars and vars must be disjoint")
    if not isinstance(samples_spec, list) or not samples_spec:
        raise ValueError("samples must be a non-empty list")

    normalized: list[ImpSampleCase] = []
    for idx, raw_case in enumerate(samples_spec):
        if not isinstance(raw_case, dict):
            raise ValueError(f"samples[{idx}] must be a table/object")
        missing_inputs = [name for name in input_vars if name not in raw_case]
        missing_outputs = [name for name in output_vars if name not in raw_case]
        if missing_inputs:
            raise ValueError(
                f"samples[{idx}] missing input vars: {', '.join(missing_inputs)}"
            )
        if missing_outputs:
            raise ValueError(
                f"samples[{idx}] missing output vars: {', '.join(missing_outputs)}"
            )
        inputs = {name: raw_case[name] for name in input_vars}
        outputs = {name: raw_case[name] for name in output_vars}
        for name, value in {**inputs, **outputs}.items():
            if not _is_imp_scalar(value):
                raise ValueError(
                    f"samples[{idx}] variable {name} has unsupported type {type(value).__name__}"
                )
        normalized.append(ImpSampleCase(inputs=inputs, outputs=outputs))
    return normalized


def _is_imp_scalar(value: Any) -> bool:
    return isinstance(value, bool) or (
        isinstance(value, int) and not isinstance(value, bool)
    )


def _check_imp_samples(
    *, output_code: str, expected_code: str, resolution: dict[str, Any]
) -> tuple[bool, str, Any, Any]:
    input_vars = resolution.get("input_vars")
    output_vars = resolution.get("vars")
    cases = validate_imp_samples_schema(
        resolution.get("samples"), input_vars, output_vars
    )

    explicit_observed: list[Any] = []
    explicit_expected: list[Any] = []
    for idx, case in enumerate(cases):
        expected_env = eval_imp(
            expected_code, overrides=case.inputs, required_inputs=input_vars
        )
        expected_selected = {name: expected_env.get(name) for name in output_vars}
        if not _values_equal(expected_selected, case.outputs):
            return (
                False,
                f"sample_spec_mismatch:case_{idx}",
                expected_selected,
                case.outputs,
            )
        observed_env = eval_imp(
            output_code, overrides=case.inputs, required_inputs=input_vars
        )
        observed_selected = {name: observed_env.get(name) for name in output_vars}
        explicit_observed.append(observed_selected)
        explicit_expected.append(expected_selected)
        if not _values_equal(observed_selected, expected_selected):
            return (
                False,
                f"sample_fail:case_{idx}",
                observed_selected,
                expected_selected,
            )

    hidden_inputs = _generate_hidden_imp_inputs(
        cases,
        input_vars=input_vars,
        count=_normalize_hidden_case_count(resolution.get("hidden_samples", 16)),
        seed_hint=resolution.get("seed_hint"),
        expected_code=expected_code,
        input_domains=resolution.get("input_domains"),
    )
    for idx, inputs in enumerate(hidden_inputs):
        expected_env = eval_imp(
            expected_code, overrides=inputs, required_inputs=input_vars
        )
        observed_env = eval_imp(
            output_code, overrides=inputs, required_inputs=input_vars
        )
        expected_selected = {name: expected_env.get(name) for name in output_vars}
        observed_selected = {name: observed_env.get(name) for name in output_vars}
        if not _values_equal(observed_selected, expected_selected):
            return (
                False,
                f"hidden_sample_fail:case_{idx}",
                observed_selected,
                expected_selected,
            )
    return (
        True,
        "",
        {"explicit": explicit_observed, "hidden_checked": len(hidden_inputs)},
        {"explicit": explicit_expected, "hidden_checked": len(hidden_inputs)},
    )


def _generate_hidden_imp_inputs(
    cases: list[ImpSampleCase],
    *,
    input_vars: list[str],
    count: int,
    seed_hint: Any,
    expected_code: str,
    input_domains: Any,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    kinds = _infer_imp_input_kinds(cases, input_vars)
    existing = {tuple(case.inputs[name] for name in input_vars) for case in cases}
    seed_material = {
        "seed_hint": seed_hint,
        "input_vars": input_vars,
        "kinds": kinds,
        "cases": [case.inputs for case in cases],
        "expected_code": expected_code,
        "input_domains": input_domains,
    }
    payload = json.dumps(seed_material, sort_keys=True, ensure_ascii=True)
    seed = int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = max(100, count * 20)
    while len(out) < count and attempts < max_attempts:
        attempts += 1
        tup = tuple(
            _draw_hidden_imp_value(kind, rng, _domain_for_input(input_domains, name))
            for name, kind in zip(input_vars, kinds)
        )
        if tup in existing:
            continue
        existing.add(tup)
        out.append({name: value for name, value in zip(input_vars, tup)})
    return out


def _infer_imp_input_kinds(
    cases: list[ImpSampleCase], input_vars: list[str]
) -> list[str]:
    kinds: list[str] = []
    for name in input_vars:
        values = [case.inputs[name] for case in cases]
        if all(isinstance(value, bool) for value in values):
            kinds.append("bool")
        elif all(
            isinstance(value, int) and not isinstance(value, bool) for value in values
        ):
            kinds.append("int")
        else:
            raise ValueError(f"unsupported mixed input types for {name}")
    return kinds


def _domain_for_input(input_domains: Any, name: str) -> Any:
    if not isinstance(input_domains, dict):
        return None
    return input_domains.get(name)


def _draw_hidden_imp_value(kind: str, rng: random.Random, domain: Any) -> Any:
    if kind == "bool":
        return bool(rng.randint(0, 1))
    if kind == "int":
        if isinstance(domain, dict):
            low = int(domain.get("min", -25))
            high = int(domain.get("max", 25))
            if low > high:
                raise ValueError("invalid imp input domain")
            return rng.randint(low, high)
        anchors = [-20, -5, -1, 0, 1, 2, 3, 4, 5, 7, 10, 12, 42]
        if rng.random() < 0.35:
            return anchors[rng.randrange(len(anchors))]
        return rng.randint(-25, 25)
    raise ValueError(f"unknown imp hidden sample kind: {kind}")


def _check_imp_structure(text: str, structure: dict[str, Any]) -> tuple[bool, str]:
    stats = inspect_imp(text)
    for key in ["let_bindings", "assignments", "ifs", "whiles"]:
        if key in structure and stats.get(key) != int(structure[key]):
            return False, f"{key}_mismatch"
    if "operations" in structure:
        found_ops = set(stats.get("operations", []))
        if not set(structure["operations"]) <= found_ops:
            return False, "operations_mismatch"
    return True, ""


def _count_fun_lets(expr: Any) -> int:
    if isinstance(expr, FLet):
        return 1 + _count_fun_lets(expr.value) + _count_fun_lets(expr.body)
    if isinstance(expr, FApp):
        return _count_fun_lets(expr.func) + _count_fun_lets(expr.arg)
    if isinstance(expr, FBin):
        return _count_fun_lets(expr.left) + _count_fun_lets(expr.right)
    if isinstance(expr, FLam):
        return _count_fun_lets(expr.body)
    return 0


def _count_fun_apps(expr: Any) -> int:
    if isinstance(expr, FApp):
        return 1 + _count_fun_apps(expr.func) + _count_fun_apps(expr.arg)
    if isinstance(expr, FBin):
        return _count_fun_apps(expr.left) + _count_fun_apps(expr.right)
    if isinstance(expr, FLet):
        return _count_fun_apps(expr.value) + _count_fun_apps(expr.body)
    if isinstance(expr, FLam):
        return _count_fun_apps(expr.body)
    return 0


def _count_fun_lambdas(expr: Any) -> int:
    if isinstance(expr, FLam):
        return 1 + _count_fun_lambdas(expr.body)
    if isinstance(expr, FApp):
        return _count_fun_lambdas(expr.func) + _count_fun_lambdas(expr.arg)
    if isinstance(expr, FBin):
        return _count_fun_lambdas(expr.left) + _count_fun_lambdas(expr.right)
    if isinstance(expr, FLet):
        return _count_fun_lambdas(expr.value) + _count_fun_lambdas(expr.body)
    return 0


def _fun_expr_ops(expr: Any) -> set[str]:
    if isinstance(expr, FBin):
        return {expr.op} | _fun_expr_ops(expr.left) | _fun_expr_ops(expr.right)
    if isinstance(expr, FApp):
        return _fun_expr_ops(expr.func) | _fun_expr_ops(expr.arg)
    if isinstance(expr, FLet):
        return _fun_expr_ops(expr.value) | _fun_expr_ops(expr.body)
    if isinstance(expr, FLam):
        return _fun_expr_ops(expr.body)
    return set()


def _tokenize_fun(text: str) -> list[str]:
    pattern = r"=>|->|==|!=|<=|>=|\+\.|-\.|\*\.|/\.|[()+\-*/;:=<>]|[0-9]+\.[0-9]+|[0-9]+|[A-Za-z_][A-Za-z0-9_]*"
    return re.findall(pattern, text)


def _parse_fun_expr(stream: TokenStream, min_prec: int = 0) -> Any:
    if stream.peek() == "let":
        return _parse_fun_let(stream)
    left = _parse_fun_app(stream)
    while True:
        op = stream.peek()
        prec = _fun_prec(op)
        if prec < min_prec:
            break
        stream.pop()
        right = _parse_fun_expr(stream, prec + 1)
        left = FBin(op, left, right)
    return left


def _parse_fun_let(stream: TokenStream) -> Any:
    stream.pop("let")
    name = stream.pop()
    stream.pop(":")
    _skip_until(stream, "=")
    stream.pop("=")
    value = _parse_fun_expr(stream)
    stream.pop(";")
    body = _parse_fun_expr(stream)
    return FLet(name, value, body)


def _parse_fun_app(stream: TokenStream) -> Any:
    expr = _parse_fun_atom(stream)
    while stream.peek() == "(":
        stream.pop("(")
        arg = _parse_fun_expr(stream)
        stream.pop(")")
        expr = FApp(expr, arg)
    return expr


def _parse_fun_atom(stream: TokenStream) -> Any:
    token = stream.peek()
    if token == "(":
        if _is_fun_lambda(stream):
            return _parse_fun_lambda(stream)
        stream.pop("(")
        expr = _parse_fun_expr(stream)
        stream.pop(")")
        return expr
    if token == "true" or token == "false":
        return FBool(stream.pop() == "true")
    if token and re.match(r"[0-9]+\.[0-9]+$", token):
        return FFloat(float(stream.pop()))
    if token and token.isdigit():
        return FInt(int(stream.pop()))
    if token and re.match(r"[A-Za-z_]", token):
        return FVar(stream.pop())
    raise ValueError(f"expected FUN atom, got {token!r}")


def _is_fun_lambda(stream: TokenStream) -> bool:
    return (
        stream.index + 3 < len(stream.tokens)
        and stream.tokens[stream.index] == "("
        and re.match(r"[A-Za-z_]", stream.tokens[stream.index + 1])
        and stream.tokens[stream.index + 2] == ":"
    )


def _parse_fun_lambda(stream: TokenStream) -> Any:
    stream.pop("(")
    name = stream.pop()
    stream.pop(":")
    _skip_until(stream, ")")
    stream.pop(")")
    stream.pop("=>")
    return FLam(name, _parse_fun_expr(stream))


def _skip_until(stream: TokenStream, target: str) -> None:
    depth = 0
    while True:
        token = stream.peek()
        if token is None:
            raise ValueError(f"expected {target!r}")
        if token == target and depth == 0:
            return
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
        stream.pop()


def _fun_prec(op: Optional[str]) -> int:
    if op in {"*", "/", "*.", "/."}:
        return 20
    if op in {"+", "-", "+.", "-."}:
        return 10
    if op in {"==", "!=", "<", "<=", ">", ">="}:
        return 5
    return -1


def _eval_fun(expr: Any, env: dict[str, Any]) -> Any:
    if isinstance(expr, (FInt, FFloat, FBool)):
        return expr.value
    if isinstance(expr, FVar):
        if expr.name not in env:
            raise ValueError(f"unbound variable {expr.name}")
        return env[expr.name]
    if isinstance(expr, FLet):
        new_env = dict(env)
        new_env[expr.name] = _eval_fun(expr.value, env)
        return _eval_fun(expr.body, new_env)
    if isinstance(expr, FLam):
        return Closure(expr.name, expr.body, dict(env))
    if isinstance(expr, FApp):
        func = _eval_fun(expr.func, env)
        arg = _eval_fun(expr.arg, env)
        if not isinstance(func, Closure):
            raise ValueError("application of non-function")
        new_env = dict(func.env)
        new_env[func.name] = arg
        return _eval_fun(func.body, new_env)
    if isinstance(expr, FBin):
        left = _eval_fun(expr.left, env)
        right = _eval_fun(expr.right, env)
        return _eval_bin(expr.op, left, right)
    raise TypeError(expr)


def _eval_bin(op: str, left: Any, right: Any) -> Any:
    if op in {"+", "+."}:
        return left + right
    if op in {"-", "-."}:
        return left - right
    if op in {"*", "*."}:
        return left * right
    if op in {"/", "/."}:
        return left / right
    if op == "==":
        return left == right
    if op == "!=":
        return left != right
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    if op == ">=":
        return left >= right
    raise ValueError(f"unknown operator {op}")


# ---------------------------------------------------------------------------
# IMP interpreter
# ---------------------------------------------------------------------------


def eval_imp(
    text: str,
    overrides: Optional[dict[str, Any]] = None,
    required_inputs: Optional[list[str]] = None,
) -> dict[str, Any]:
    parser = ImpParser(
        _tokenize_imp(text), overrides=overrides, required_inputs=required_inputs
    )
    env: dict[str, Any] = {}
    parser.parse_block(env)
    if not parser.done():
        raise ValueError(f"unexpected token {parser.peek()!r}")
    parser.validate_required_inputs()
    return _value_to_json(env)


def inspect_imp(text: str) -> dict[str, Any]:
    tokens = _tokenize_imp(text)
    return {
        "let_bindings": sum(1 for tok in tokens if tok == "let"),
        "assignments": max(
            tokens.count("=") - sum(1 for tok in tokens if tok == "let"), 0
        ),
        "ifs": sum(1 for tok in tokens if tok == "if"),
        "whiles": sum(1 for tok in tokens if tok == "while"),
        "operations": sorted(
            {
                tok
                for tok in tokens
                if tok in {"+", "-", "*", "/", "==", "!=", "<", "<=", ">", ">="}
            }
        ),
    }


def _tokenize_imp(text: str) -> list[str]:
    pattern = r"==|!=|<=|>=|[{}()+\-*/;:=<>]|[0-9]+|[A-Za-z_][A-Za-z0-9_]*"
    return re.findall(pattern, text)


class ImpParser(TokenStream):
    def __init__(
        self,
        tokens: list[str],
        overrides: Optional[dict[str, Any]] = None,
        required_inputs: Optional[list[str]] = None,
    ):
        super().__init__(tokens)
        self.overrides = dict(overrides or {})
        self.required_inputs = list(required_inputs or [])
        self.seen_inputs: set[str] = set()
        self.stats = {
            "let_bindings": 0,
            "assignments": 0,
            "ifs": 0,
            "whiles": 0,
            "operations": set(),
        }

    def validate_required_inputs(self) -> None:
        missing = [
            name for name in self.required_inputs if name not in self.seen_inputs
        ]
        if missing:
            raise ValueError(
                f"missing required input declarations: {', '.join(missing)}"
            )

    def _merge_stats(self, child: "ImpParser") -> None:
        self.stats["let_bindings"] += child.stats["let_bindings"]
        self.stats["assignments"] += child.stats["assignments"]
        self.stats["ifs"] += child.stats["ifs"]
        self.stats["whiles"] += child.stats["whiles"]
        self.stats["operations"].update(child.stats["operations"])

    def parse_block(self, env: dict[str, Any]) -> None:
        self.pop("{")
        while self.peek() != "}":
            if self.peek() is None:
                raise ValueError("unterminated block")
            self.parse_statement(env)
        self.pop("}")

    def parse_statement(self, env: dict[str, Any]) -> None:
        token = self.peek()
        if token == "let":
            self.stats["let_bindings"] += 1
            self.pop("let")
            name = self.pop()
            self.pop(":")
            _skip_until(self, "=")
            self.pop("=")
            if name in self.overrides:
                self.seen_inputs.add(name)
                self._skip_imp_expr_to_semicolon()
                env[name] = self.overrides[name]
            else:
                env[name] = self.parse_expr(env)
            self.pop(";")
            return
        if token == "if":
            self.stats["ifs"] += 1
            self.pop("if")
            self.pop("(")
            cond = self.parse_expr(env)
            self.pop(")")
            then_tokens = self.collect_block_tokens()
            else_tokens = None
            if self.peek() == "else":
                self.pop("else")
                else_tokens = self.collect_block_tokens()
            branch = then_tokens if cond else else_tokens
            if branch is not None:
                child = ImpParser(
                    branch,
                    overrides=self.overrides,
                    required_inputs=self.required_inputs,
                )
                child.parse_block(env)
                self._merge_stats(child)
            return
        if token == "while":
            self.stats["whiles"] += 1
            self.pop("while")
            self.pop("(")
            cond_tokens = self.collect_until_matching_paren()
            body_tokens = self.collect_block_tokens()
            for _ in range(1000):
                cond_parser = ImpParser(
                    cond_tokens,
                    overrides=self.overrides,
                    required_inputs=self.required_inputs,
                )
                cond_value = cond_parser.parse_expr(env)
                self._merge_stats(cond_parser)
                if not cond_value:
                    return
                body_parser = ImpParser(
                    body_tokens,
                    overrides=self.overrides,
                    required_inputs=self.required_inputs,
                )
                body_parser.parse_block(env)
                self._merge_stats(body_parser)
            raise ValueError("while loop exceeded fuel")
        self.stats["assignments"] += 1
        name = self.pop()
        self.pop("=")
        env[name] = self.parse_expr(env)
        self.pop(";")

    def _skip_imp_expr_to_semicolon(self) -> None:
        depth = 0
        while True:
            token = self.peek()
            if token is None:
                raise ValueError("unterminated expression")
            if token == ";" and depth == 0:
                return
            if token == "(":
                depth += 1
            elif token == ")":
                depth -= 1
            self.pop()

    def collect_block_tokens(self) -> list[str]:
        start = self.index
        depth = 0
        while True:
            token = self.pop()
            if token == "{":
                depth += 1
            elif token == "}":
                depth -= 1
                if depth == 0:
                    return self.tokens[start : self.index]

    def collect_until_matching_paren(self) -> list[str]:
        start = self.index
        depth = 0
        while True:
            token = self.peek()
            if token is None:
                raise ValueError("unterminated condition")
            if token == "(":
                depth += 1
            elif token == ")":
                if depth == 0:
                    out = self.tokens[start : self.index]
                    self.pop(")")
                    return out
                depth -= 1
            self.pop()

    def parse_expr(self, env: dict[str, Any], min_prec: int = 0) -> Any:
        left = self.parse_atom(env)
        while True:
            op = self.peek()
            prec = _fun_prec(op)
            if prec < min_prec:
                break
            self.pop()
            self.stats["operations"].add(op)
            right = self.parse_expr(env, prec + 1)
            left = _eval_bin(op, left, right)
        return left

    def parse_atom(self, env: dict[str, Any]) -> Any:
        token = self.peek()
        if token == "(":
            self.pop("(")
            value = self.parse_expr(env)
            self.pop(")")
            return value
        if token == "true" or token == "false":
            return self.pop() == "true"
        if token and token.isdigit():
            return int(self.pop())
        if token and re.match(r"[A-Za-z_]", token):
            name = self.pop()
            if name not in env:
                raise ValueError(f"unbound variable {name}")
            return env[name]
        raise ValueError(f"expected IMP atom, got {token!r}")
