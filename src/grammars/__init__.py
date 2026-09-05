"""Built-in grammars with typing rules (context-dependent generation)."""

from pathlib import Path
from typing import Any, Dict, List


def _base_library():
    """`aufbau.examples`, the shared base grammar library, or None.

    aufbau is already a hard dependency -- `proposition7.grammar` cannot build a
    Synthesizer without it -- so reading the grammars from it costs p7 nothing.
    """
    try:
        from aufbau import examples
        return examples
    except ImportError:
        pass
    # Monorepo checkout, when the installed wheel predates `aufbau.examples`.
    # `src/grammars/` -> p7 -> congen; the loader there needs no import to work.
    local = Path(__file__).resolve().parents[3] / "aufbau" / "examples"
    if local.is_dir():
        import types
        shim = types.SimpleNamespace()
        shim.spec = lambda name: (local / f"{name}.auf").read_text(encoding="utf-8")
        shim.names = lambda: sorted(q.stem for q in local.glob("*.auf"))
        return shim
    return None


def _load_spec(name: str) -> str:
    """The source text of grammar `name`.

    Base grammars come from `aufbau.examples`, which holds the one copy of each.
    p7 used to bundle its own snapshots beside this module; `c.auf` drifted sixty
    lines behind -- missing the return-type constraint and the `ForInit` split --
    and nothing noticed, because nothing compares two copies. Grammars that are
    genuinely p7's own (`typescript.auf`) still live here.
    """
    library = _base_library()
    if library is not None:
        try:
            return library.spec(name)
        except (FileNotFoundError, OSError):
            pass
    local = Path(__file__).resolve().parent / f"{name}.auf"
    if local.is_file():
        return local.read_text(encoding="utf-8")
    have = ", ".join(library.names()) if library is not None else "<no aufbau>"
    raise FileNotFoundError(
        f"grammar spec {name!r} is neither in the aufbau base library ({have}) "
        f"nor beside {Path(__file__).parent}"
    )


# Unified grammar information: spec content + metadata for prompt construction.
GRAMMARS: Dict[str, Dict[str, Any]] = {
    "stlc": {
        "spec": _load_spec("stlc"),
        "name": "Simply Typed Lambda Calculus",
        "short": "typed lambda calculus terms",
        "description": "Simply typed lambda calculus with explicit type annotations",
        "summary": (
            "Simply typed lambda calculus. Produce one term. Syntax: variables x, "
            "lambdas λx:T.body, application f arg (left-associative), and grouping "
            "(expr). Types are Int, Bool, or other type names; function types A->B "
            "are right-associative, so parenthesize function arguments like "
            "(Int->Bool). Bound variables are only available inside their lambda. "
            "Examples: λx:Int.x ; λf:(Int->Bool).λx:Int.(f x)."
        ),
        "syntax_hints": [
            "λx:T.e - lambda abstraction",
            "(f x) - function application",
            "Types use -> and are right-associative: Int->Bool->Int",
            "Parenthesize function arguments and nested types when needed",
        ],
        "examples": [
            ("identity", "λx:Int.x"),
            ("const", "λx:Int.λy:Bool.x"),
            ("apply", "λf:(Int->Bool).λx:Int.(f x)"),
        ],
    },
    "ml": {
        "spec": _load_spec("ml"),
        "name": "ML",
        "short": "well-typed OCaml-subset programs",
        "description": "Monomorphic ML (strict OCaml subset): functions, pairs, lists, match, let rec",
        "summary": (
            "Monomorphic ML, a strict OCaml subset: every program is also valid OCaml. "
            "One expression per program. Lambda: fun (x : int) -> body. Application: "
            "f(arg). Pairs: (a, b) with fst/snd. Lists: [] and h :: t, typed t list. "
            "match xs with [] -> e1 | h :: t -> e2. let x : t = v in body and "
            "let rec f : t -> u = fun (x : t) -> ... in body. Types: int, bool, "
            "t list, t * u, t -> u. Int ops + -, comparisons = < <= > >= return bool. "
            "assert false inhabits any type."
        ),
        "syntax_hints": [
            "Lambda: fun (x : int) -> expr",
            "Application: f(arg)",
            "Lists: 1 :: 2 :: [] of type int list",
            "match xs with [] -> e1 | h :: t -> e2",
            "let x : int = 5 in body / let rec f : int -> int = fun (n : int) -> ... in body",
            "assert false has any type (divergence)",
        ],
        "examples": [
            ("identity", "fun (x : int) -> x"),
            ("let", "let a : int = 5 in a + 1"),
            (
                "sum",
                "let rec sum : int list -> int = fun (xs : int list) -> match xs with [] -> 0 | h :: t -> h + sum(t) in sum(1 :: 2 :: [])",
            ),
        ],
    },
    "toy": {
        "spec": _load_spec("toy"),
        "name": "Toy: Beep Boop",
        "short": "typed nonsense",
        "description": "Meaningless but funny typed expressions",
        "summary": (
            "Tiny typed toy expression language. Values are beep:Fizz, boop:Fizz, "
            "blorp:Fizz or the same words with :Buzz. Expressions can be grouped "
            "with parentheses. Concatenation expr + expr is valid only when both sides "
            "have the same type. Echo syntax expr x ha, expr x ho, or repeated laughs "
            "like expr x ha ho keeps the expression type."
        ),
        "syntax_hints": [
            "Typed value: beep:Fizz",
            "Concatenation: beep:Fizz + boop:Fizz",
        ],
        "examples": [
            ("single", "beep:Fizz"),
            ("concat", "beep:Fizz + boop:Fizz"),
        ],
    },
    "c": {
        "spec": _load_spec("c"),
        "name": "C",
        "short": "well-typed, compilable C functions",
        "description": "A typed C subset: real, compilable C, checked against cc as an external oracle",
        "summary": (
            "A typed C subset: every accepted program is real, compilable C, checked "
            "against `cc -fsyntax-only` as an external oracle. A program is one or "
            "more function definitions, each of arbitrary arity, e.g. `int add(int x, "
            "int y) { return x + y; }`. Base types: int, float, char, void, and "
            "pointers (T*, T**, ...). Pointers: &e (address-of), *e (dereference). "
            "Explicit casts: (T) e. Arithmetic is + - / only (no *, to keep pointer "
            "syntax unambiguous). Control flow: if/else, while, for. A function may "
            "call any function defined earlier in the same program (not itself: no "
            "self-recursion). return is not checked against the declared return type."
        ),
        "syntax_hints": [
            "Function: int add(int x, int y) { return x + y; }",
            "Zero-arg: int f() { ... }",
            "Pointers: int *p = &x; return *p;",
            "Cast: int y = (int) x;",
            "Arithmetic ops are + - / only, never * (reserved for pointers)",
            "A function may call any earlier-defined function, never itself",
        ],
        "examples": [
            ("add", "int add(int x, int y) { return x + y; }"),
            ("deref", "int deref(int *p) { return *p; }"),
            (
                "helper",
                "int double_it(int x) { return x + x; }\nint sum_doubled(int a, int b) { return double_it(a) + double_it(b); }",
            ),
        ],
    },
}


def strip_typing_rules(spec: str) -> str:
    """Return a grammar spec with its typing rules removed.

    Every bundled grammar separates its syntactic productions from its typing
    rules with a section header whose comment, once ``//``, whitespace, and
    any ``=`` fence characters are stripped, reads exactly "Typing Rules"
    (case-insensitive) -- e.g. ``// Typing Rules`` or
    ``// ===== Typing rules =====``. Matching on that exact content (not a
    substring) matters: c.auf's header comment mentions "typing rules" in
    prose (describing what a nonterminal is used for) well before its real
    section divider, and a substring match cut the syntax-only grammar down
    to almost nothing there. Cutting from the real header onward yields a
    purely syntactic grammar."""
    out: List[str] = []
    for line in spec.splitlines():
        stripped = line.strip()
        if stripped.startswith("//"):
            content = stripped[2:].strip().strip("=").strip()
            if content.lower() == "typing rules":
                break
        out.append(line)
    return "\n".join(out).rstrip() + "\n"


def base_grammar(name: str) -> str:
    """Strip a trailing `_syntactic` suffix, if present. A syntactic twin
    (e.g. `imp_syntactic`) constrains generation with typing rules stripped,
    but is still graded by its semantic base grammar's oracle."""
    suffix = "_syntactic"
    return name[: -len(suffix)] if name.endswith(suffix) else name


def list_grammars() -> List[str]:
    """List all available grammar names."""
    return list(GRAMMARS.keys())


def get_grammar(name: str) -> str:
    """Get the spec content for a grammar."""
    if name not in GRAMMARS:
        available = ", ".join(GRAMMARS.keys())
        raise ValueError(f"Unknown grammar '{name}'. Available: {available}")
    return GRAMMARS[name]["spec"]


def get_grammar_summary(name: str) -> str:
    """Get a compact prompt summary for a grammar."""
    info = get_grammar_info(name)
    return str(info.get("summary") or info["description"])


def get_grammar_info(grammar_name: str) -> Dict[str, Any]:
    """Get info about a grammar, with fallback for unknown grammars."""
    if grammar_name in GRAMMARS:
        return GRAMMARS[grammar_name]
    # Fallback for unknown grammars
    return {
        "spec": "",
        "name": grammar_name,
        "short": f"{grammar_name} expressions",
        "description": f"Grammar: {grammar_name}",
        "summary": f"Grammar: {grammar_name}",
        "syntax_hints": [],
        "examples": [],
    }
