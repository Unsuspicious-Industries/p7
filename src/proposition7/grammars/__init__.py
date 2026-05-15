"""Built-in grammars with typing rules (context-dependent generation)."""

from pathlib import Path
from typing import Any, Dict, List


def _load_spec(name: str) -> str:
    """Load a grammar spec from bundled specs, falling back to aufbau repo."""
    # 1. Bundled in package (works in Docker / pip installs)
    bundled = Path(__file__).resolve().parent /  f"{name}.auf"
    if not bundled.exists():
        raise FileNotFoundError(
            f"Grammar spec '{name}' not found. "
            f"Checked: {bundled}"
        )
    return bundled.read_text(encoding="utf-8")

    


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
    "imp": {
        "spec": _load_spec("imp"),
        "name": "IMP",
        "short": "typed imperative programs",
        "description": "Typed imperative language with assignments, conditionals, and loops",
        "summary": (
            "Typed imperative block language. A program is { statements }. Statements: "
            "let x: Int = expr; or Bool, assignment x = expr;, if (cond) { ... } "
            "else { ... }, and while (cond) { ... }. Expressions include integers, "
            "booleans, variables, Int arithmetic + - * /, and Int comparisons "
            "== != < <= > >= returning Bool. Declare variables before use and assign "
            "values of the same type."
        ),
        "syntax_hints": [
            "Programs are wrapped in { ... }",
            "Assignment: let x: Type = value;",
            "Arithmetic values: x + y, x - 1, a * b",
            "Conditionals: if (cond) { ... } else { ... }",
            "Loops: while (cond) { ... }",
        ],
        "examples": [
            ("assignment", "{ let x: Int = 5; }"),
            ("sequence", "{ let x: Int = 1; let y: Int = x + 2; }"),
            ("if_else", "{ let x: Int = 1; if (x < 5) { let y: Int = x + 1; } else { let y: Int = 0; } }"),
            ("while", "{ let counter: Int = 0; while (counter < 3) { counter = counter + 1; } }"),
        ],
    },
    "fun": {
        "spec": _load_spec("fun"),
        "name": "Fun",
        "short": "typed functional expressions",
        "description": "ML-style functional language with let bindings and typed lambdas",
        "summary": (
            "Typed ML-style expression language. Top level is one expression. Literals: "
            "Int, Float, Bool. Let: let x: Type = value; body. Lambda: "
            "(x: Type) => body. Application: f(arg), with repeated calls allowed. "
            "Types include Int, Float, Bool, and monomorphic function types A->B. "
            "Use + - * / for Int and +. -. *. /. for Float. Variables are scoped by "
            "lets and lambdas."
        ),
        "syntax_hints": [
            "Lambda: (x: Type) => expr",
            "Let binding: let x: Type = value; body",
            "Function application: f(arg)",
            "Int ops: + - * /, Float ops: +. -. *. /.",
            "Literals include Int, Float, and Bool",
        ],
        "examples": [
            ("identity", "(x: Int) => x"),
            ("let_int", "let x: Int = 1; x + 2"),
            ("apply_lambda", "((x: Int) => x + 1)(41)"),
            ("float_math", "let f: Float = 1.5; f +. 2.0"),
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
}


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
