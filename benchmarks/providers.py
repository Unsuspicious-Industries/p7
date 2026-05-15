from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional, Union

import proposition7
from proposition7.inference import GenerationResult


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
        temperature: float = 0.0,
        top_k: int = 0,
        on_token: Optional[Callable[[str, int], None]] = None,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        del top_k
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


class OutlinesSyntaxModel:
    """Local HF model wrapper that delegates constrained generation to proposition7."""

    def __init__(
        self,
        model_name: str,
        *,
        grammar_name: str,
        device: str = "cpu",
        temperature=0.0,
        **model_kwargs: Any,
    ):
        self.model_name = model_name
        self.grammar_name = grammar_name
        self.device = device
        self.inner = proposition7.get_model_class(model_name).from_pretrained(
            model_name,
            grammar=proposition7.get_grammar(grammar_name),
            device=device,
            **model_kwargs,
        )
        self.temperature = temperature

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
        seed: Optional[int] = None,
        temperature: float = 0.0,
    ) -> GenerationResult:
        return self.inner.generate_constrained(
            prompt,
            initial=initial,
            max_tokens=max_tokens,
            seed=seed,
            temperature=temperature,
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
