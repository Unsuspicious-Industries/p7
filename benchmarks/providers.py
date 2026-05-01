from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional, Union

from p7.inference import GenerationResult


EnvPath = Optional[Union[str, os.PathLike[str]]]


def load_env(path: EnvPath) -> None:
    if not path:
        return
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        key, value = stripped.split("=", 1)
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


def merge_initial(initial: str, generated: str) -> str:
    if not initial or generated.startswith(initial):
        return generated
    return initial + generated


class OpenRouterModel:
    """OpenRouter chat-completions adapter for unconstrained benchmark runs."""

    def __init__(
        self,
        model_name: str,
        *,
        api_key: Optional[str] = None,
        env_path: EnvPath = ".env",
        base_url: str = "https://openrouter.ai/api/v1",
        site_url: str = "https://github.com/Unsuspicious-Industries/p7",
        app_title: str = "proposition7-benchmarks",
    ):
        load_env(env_path)
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.site_url = site_url
        self.app_title = app_title
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for --backend openrouter")

    def generate_constrained(self, *args: Any, **kwargs: Any) -> GenerationResult:
        raise NotImplementedError("OpenRouter backend is unconstrained-only")

    def generate_unconstrained(
        self,
        prompt: str,
        *,
        initial: str = "",
        max_tokens: int = 50,
        top_k: Optional[int] = None,
        temperature: float = 1.0,
        grammar_name: Optional[str] = None,
        on_token: Optional[Callable[[str, int], None]] = None,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        del top_k, grammar_name
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": self._prompt(prompt, initial)}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if seed is not None:
            payload["seed"] = seed

        if on_token is not None:
            text, count, reason = self._stream_chat(payload, on_token)
        else:
            text, count, reason = self._chat(payload)

        return GenerationResult(
            text=merge_initial(initial, text),
            is_complete=False,
            tokens_generated=count,
            stopped_reason=reason,
        )

    def _prompt(self, prompt: str, initial: str) -> str:
        if not initial:
            return prompt
        return (
            f"{prompt}\n\n"
            "Continue this exact prefix. Return only the completed program text, including the prefix.\n"
            f"Prefix:\n{initial}"
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.app_title,
        }

    def _request(self, payload: dict[str, Any]) -> urllib.request.Request:
        return urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )

    def _chat(self, payload: dict[str, Any]) -> tuple[str, int, str]:
        request = self._request(payload)
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP {error.code}: {body}") from error

        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        usage = data.get("usage", {})
        count = int(usage.get("completion_tokens") or len(content.split()) or bool(content))
        return str(content), count, str(choice.get("finish_reason") or "stop")

    def _stream_chat(
        self, payload: dict[str, Any], on_token: Callable[[str, int], None]
    ) -> tuple[str, int, str]:
        payload = {**payload, "stream": True}
        request = self._request(payload)
        chunks: list[str] = []
        finish_reason = "stop"
        step = 0
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    choice = event.get("choices", [{}])[0]
                    finish_reason = str(choice.get("finish_reason") or finish_reason)
                    delta = choice.get("delta", {}).get("content") or ""
                    if not delta:
                        continue
                    chunks.append(delta)
                    on_token(delta, step)
                    step += 1
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP {error.code}: {body}") from error
        return "".join(chunks), step, finish_reason


OUTLINES_LARK: dict[str, str] = {
    "stlc": r"""
        start: expr
        expr: lam | app
        lam: "λ" NAME ":" type "." expr
        app: atom atom+
           | atom
        atom: NAME | "(" expr ")"

        type: type_atom "->" type
            | type_atom
        type_atom: TYPE_NAME | "(" type ")"

        NAME: /[A-Za-z_][A-Za-z0-9_]*/
        TYPE_NAME: /[A-Za-z_][A-Za-z0-9_]*/
        %import common.WS
        %ignore WS
    """,
    "fun": r"""
        start: expr
        expr: let_expr | binary
        let_expr: "let" NAME ":" type "=" expr ";" expr
        binary: application (OP application)*
        application: atom ("(" expr ")")*
        atom: NAME | INT | FLOAT | BOOL | lam | "(" expr ")"
        lam: "(" NAME ":" type ")" "=>" expr

        type: type_atom "->" type
            | type_atom
        type_atom: TYPE_NAME | "(" type ")"

        OP: "+" | "-" | "*" | "/" | "+." | "-." | "*." | "/." | "==" | "!=" | "<" | "<=" | ">" | ">="
        NAME: /[a-z][a-z0-9_]*/
        TYPE_NAME: /[A-Z][A-Za-z0-9_]*/
        BOOL: "true" | "false"
        INT: /[0-9]+/
        FLOAT: /[0-9]+\.[0-9]+/
        %import common.WS
        %ignore WS
    """,
    "imp": r"""
        start: block
        block: "{" stmt* "}"
        stmt: decl | assign | if_stmt | if_else_stmt | while_stmt
        decl: "let" NAME ":" type "=" expr ";"
        assign: NAME "=" expr ";"
        if_stmt: "if" "(" expr ")" block
        if_else_stmt: "if" "(" expr ")" block "else" block
        while_stmt: "while" "(" expr ")" block

        expr: atom (OP atom)*
        atom: NAME | INT | BOOL | "(" expr ")"

        type: type_atom "|" type
            | type_atom
        type_atom: "Int" | "Bool" | "(" type ")"

        OP: "+" | "-" | "*" | "/" | "==" | "!=" | "<" | "<=" | ">" | ">="
        NAME: /[a-z][a-z0-9_]*/
        BOOL: "true" | "false"
        INT: /[0-9]+/
        %import common.WS
        %ignore WS
    """,
    "toy": r"""
        start: expr
        expr: atom | "(" expr ")" | expr "x" chorus | expr "+" expr
        atom: WORD ":" TYPE
        chorus: LAUGH+
        WORD: "beep" | "boop" | "blorp"
        TYPE: "Fizz" | "Buzz"
        LAUGH: "ha" | "ho" | "hee"
        %import common.WS
        %ignore WS
    """,
    "lamb": r"""
        start: definition+
        definition: GLOBAL "=" term

        term: application
            | atom
        application: atom "(" args ")"
                   | application "(" args ")"
        args: term "," args
            | term
        atom: GLOBAL | NAME | "λ" NAME "." term | "(" term ")"

        GLOBAL: "@" NAME
        NAME: /[A-Za-z_][A-Za-z0-9_]*/
        %import common.WS
        %ignore WS
    """,
}

class OutlinesSyntaxModel:
    """Best-effort Outlines CFG baseline for syntax-only constrained decoding."""

    def __init__(self, model_name: str, *, grammar_name: str, device: str = "cpu"):
        if grammar_name not in OUTLINES_LARK:
            raise RuntimeError(f"No Outlines syntax grammar registered for {grammar_name!r}")
        self.model_name = model_name
        self.grammar_name = grammar_name
        self.device = device
        self._generator: Any = None

    def generate_unconstrained(self, *args: Any, **kwargs: Any) -> GenerationResult:
        raise NotImplementedError("Outlines engine is constrained-only")

    def generate_constrained(
        self,
        prompt: str,
        *,
        initial: str = "",
        max_tokens: int = 50,
        grammar_name: Optional[str] = None,
        stop_on_complete: bool = True,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        del stop_on_complete, seed
        generator = self._load_generator(grammar_name or self.grammar_name)
        generated = str(generator(self._prompt(prompt, initial), max_tokens=max_tokens))
        text = merge_initial(initial, generated)
        return GenerationResult(text=text, is_complete=False, tokens_generated=1, stopped_reason="outlines")

    def _prompt(self, prompt: str, initial: str) -> str:
        return prompt if not initial else f"{prompt}\n\nContinue this prefix exactly:\n{initial}"

    def _load_generator(self, grammar_name: str):
        if self._generator is not None:
            return self._generator
        try:
            import outlines
        except ImportError as error:
            raise RuntimeError("Outlines engine requires `pip install outlines`.") from error

        grammar = OUTLINES_LARK[grammar_name]
        try:
            model = outlines.models.transformers(self.model_name, device=self.device)
            self._generator = outlines.generate.cfg(model, grammar)
        except Exception as error:
            raise RuntimeError(
                "Unable to initialize Outlines CFG backend with this installed Outlines version."
            ) from error
        return self._generator
