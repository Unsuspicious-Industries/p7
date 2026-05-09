from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional, Union

import proposition7
from proposition7.inference import GenerationResult
from proposition7.llm import set_generation_seed


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
        site_url: str = "https://unsuspicious.org/blog/proposition-7",
        app_title: str = "proposition7-benchmarks",
    ):
        load_env(env_path)
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.site_url = site_url
        self.app_title = app_title
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is required for --backend openrouter"
            )

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
        count = int(
            usage.get("completion_tokens") or len(content.split()) or bool(content)
        )
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
    """Local HF model wrapper that uses Outlines' CFG backend constraints.

    Outlines v1 delegates CFG masking to LLGuidance. We drive that matcher
    directly so benchmark `initial` prefixes advance the grammar state before
    sampling; generator-level logits processors only see newly generated tokens.
    """

    def __init__(
        self,
        model_name: str,
        *,
        grammar_name: str,
        device: str = "cpu",
        **model_kwargs: Any,
    ):
        if grammar_name not in OUTLINES_LARK:
            raise RuntimeError(
                f"No Outlines syntax grammar registered for {grammar_name!r}"
            )
        self.model_name = model_name
        self.grammar_name = grammar_name
        self.device = device
        self.inner = proposition7.get_model_class(model_name).from_pretrained(
            model_name,
            grammar=proposition7.get_grammar(grammar_name),
            device=device,
            **model_kwargs,
        )
        self._llg_tokenizer: Any = None
        self._grammar_specs: dict[str, str] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    @property
    def grammar(self) -> str:
        return self.inner.grammar

    @grammar.setter
    def grammar(self, value: str) -> None:
        self.inner.grammar = value

    def generate_unconstrained(self, *args: Any, **kwargs: Any) -> GenerationResult:
        return self.inner.generate_unconstrained(*args, **kwargs)

    def generate_constrained(
        self,
        prompt: str,
        *,
        initial: str = "",
        max_tokens: int = 50,
        grammar_name: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        import torch

        set_generation_seed(seed)
        grammar_name = grammar_name or self.grammar_name
        if grammar_name not in OUTLINES_LARK:
            raise RuntimeError(
                f"No Outlines syntax grammar registered for {grammar_name!r}"
            )

        self.inner._set_prompt(
            prompt,
            initial,
            self.inner.start_tokens_constrained(grammar_name),
        )
        stop_tokens = set(self.inner.stop_tokens_constrained(grammar_name))
        stop_token_ids = set(self.inner._stop_token_ids(list(stop_tokens)))
        matcher = self._new_matcher(grammar_name)

        if initial:
            try:
                self._consume_initial(matcher, initial)
            except RuntimeError as error:
                return GenerationResult(initial, False, 0, f"type_error: {error}")

        generated_ids: list[int] = []
        tokens_generated = 0
        stopped_reason = "max_tokens"

        for _ in range(max_tokens):
            logits = self.inner._get_logits_tensor().float().clone()
            self._mask_logits(matcher, logits)
            finite = torch.isfinite(logits)
            if not finite.any():
                stopped_reason = "complete" if self._is_complete(matcher) else "no_valid"
                break

            probs = torch.softmax(logits, dim=-1)
            if not torch.isfinite(probs).any() or probs.sum() <= 0:
                stopped_reason = "complete" if self._is_complete(matcher) else "no_valid"
                break

            token_id = int(torch.multinomial(probs, num_samples=1).item())
            token = self.inner._decode_token_id(token_id)

            if token_id in stop_token_ids or token in stop_tokens:
                stopped_reason = (
                    "complete"
                    if self._is_complete(matcher)
                    else f"stop_token:{self.inner._stop_token_label(token_id, token)}"
                )
                break

            if not self._consume_token(matcher, token_id):
                stopped_reason = self._matcher_error(matcher) or "no_valid"
                break

            generated_ids.append(token_id)
            tokens_generated += 1
            self.inner._append_token_id(token_id)

        return GenerationResult(
            text=initial + self._decode(generated_ids),
            is_complete=self._is_complete(matcher),
            tokens_generated=tokens_generated,
            stopped_reason=stopped_reason,
        )

    def _decode(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        try:
            return self.inner.tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return self.inner.tokenizer.decode(token_ids)

    def _load_llguidance(self) -> tuple[Any, Any]:
        if self._llg_tokenizer is not None:
            import llguidance

            return llguidance, self._llg_tokenizer
        try:
            import outlines  # noqa: F401
            import llguidance
            import llguidance.hf
        except ImportError as error:
            raise RuntimeError(
                "Outlines CFG mode requires `pip install -e '.[outlines]'` "
                "or `pip install 'outlines[llguidance]'`."
            ) from error

        try:
            self._llg_tokenizer = llguidance.hf.from_tokenizer(self.inner.tokenizer)
        except Exception as error:
            raise RuntimeError(
                "Outlines CFG mode requires a fast Hugging Face tokenizer supported by LLGuidance."
            ) from error
        return llguidance, self._llg_tokenizer

    def _grammar_spec(self, grammar_name: str) -> str:
        llguidance, _ = self._load_llguidance()
        if grammar_name not in self._grammar_specs:
            self._grammar_specs[grammar_name] = llguidance.grammar_from(
                "lark", OUTLINES_LARK[grammar_name]
            )
        return self._grammar_specs[grammar_name]

    def _new_matcher(self, grammar_name: str) -> Any:
        llguidance, llg_tokenizer = self._load_llguidance()
        return llguidance.LLMatcher(llg_tokenizer, self._grammar_spec(grammar_name))

    def _consume_initial(self, matcher: Any, initial: str) -> None:
        token_ids = self.inner.tokenizer.encode(initial, add_special_tokens=False)
        for token_id in token_ids:
            if not self._consume_token(matcher, int(token_id)):
                raise RuntimeError(self._matcher_error(matcher) or "invalid prefix")

    def _consume_token(self, matcher: Any, token_id: int) -> bool:
        try:
            return bool(matcher.consume_token(token_id))
        except Exception:
            return False

    def _matcher_error(self, matcher: Any) -> str:
        get_error = getattr(matcher, "get_error", None)
        if not callable(get_error):
            return ""
        try:
            return str(get_error() or "")
        except Exception:
            return ""

    def _is_complete(self, matcher: Any) -> bool:
        try:
            return bool(matcher.is_accepting())
        except Exception:
            return False

    def _mask_logits(self, matcher: Any, logits: Any) -> None:
        import llguidance.torch

        bitmask = llguidance.torch.allocate_token_bitmask(
            1, self._llg_tokenizer.vocab_size
        )
        llguidance.torch.fill_next_token_bitmask(matcher, bitmask, 0)
        mask = bitmask[0].to(logits.device)
        llguidance.torch.apply_token_bitmask_inplace(logits, mask)
