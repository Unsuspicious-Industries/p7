from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union

from .grammars import GRAMMARS, get_grammar
from .inference import GenerationResult


DEFAULT_APP_NAME = "proposition7-generation"
DEFAULT_GPU = os.environ.get("PROPOSITION7_MODAL_GPU", "T4")
MODAL_ENV_KEYS = ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")
EnvPath = Optional[Union[str, os.PathLike[str]]]


try:
    import modal as _modal
except ImportError:
    _modal = None


def load_modal_env(env_path: EnvPath = None) -> dict[str, str]:
    """Load Modal credentials from `.env` without requiring python-dotenv."""

    path = _find_env_file(env_path)
    if path is None:
        return {}

    loaded: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        if key not in MODAL_ENV_KEYS:
            continue
        if key not in os.environ:
            os.environ[key] = value
        loaded[key] = os.environ[key]
    return loaded


def _find_env_file(env_path: EnvPath) -> Optional[Path]:
    if env_path is not None:
        path = Path(env_path).expanduser()
        return path if path.exists() else None

    current = Path.cwd().resolve()
    for directory in (current, *current.parents):
        candidate = directory / ".env"
        if candidate.exists():
            return candidate
    return None


def _parse_env_line(line: str) -> Optional[tuple[str, str]]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()
    if "=" not in stripped:
        return None

    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def _resolve_grammar(grammar: str) -> str:
    return get_grammar(grammar) if grammar in GRAMMARS else grammar


def _default_grammar_name(grammar: str, grammar_name: Optional[str]) -> Optional[str]:
    if grammar_name is not None:
        return grammar_name
    return grammar if grammar in GRAMMARS else None


def _result_payload(result: GenerationResult, tokens: list[str]) -> dict[str, Any]:
    return {
        "text": result.text,
        "is_complete": result.is_complete,
        "tokens_generated": result.tokens_generated,
        "stopped_reason": result.stopped_reason,
        "diagnostics": result.diagnostics,
        "tokens": tokens,
    }


def _payload_result(payload: dict[str, Any]) -> GenerationResult:
    return GenerationResult(
        text=str(payload.get("text", "")),
        is_complete=bool(payload.get("is_complete", False)),
        tokens_generated=int(payload.get("tokens_generated", 0)),
        stopped_reason=str(payload.get("stopped_reason", "unknown")),
        diagnostics=dict(payload.get("diagnostics", {}) or {}),
    )


_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def _load_remote_model(
    model_name: str,
    grammar: str,
    model_kwargs: Optional[dict[str, Any]] = None,
):
    import torch

    from .models import get_model_class

    grammar_spec = _resolve_grammar(grammar)
    cache_key = (model_name, grammar_spec)
    if cache_key not in _MODEL_CACHE:
        kwargs = dict(model_kwargs or {})
        if "device" not in kwargs and "device_map" not in kwargs:
            kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL_CACHE[cache_key] = get_model_class(model_name).from_pretrained(
            model_name,
            grammar=grammar_spec,
            **kwargs,
        )
    return _MODEL_CACHE[cache_key]


def _generate_payload(
    kind: str,
    *,
    model_name: str,
    grammar: str,
    prompt: str,
    initial: str = "",
    max_tokens: int = 50,
    grammar_name: Optional[str] = None,
    model_kwargs: Optional[dict[str, Any]] = None,
    seed: Optional[int] = None,
    **generation_kwargs: Any,
) -> dict[str, Any]:
    model = _load_remote_model(model_name, grammar, model_kwargs)
    common = {
        "prompt": prompt,
        "initial": initial,
        "max_tokens": max_tokens,
        "grammar_name": _default_grammar_name(grammar, grammar_name),
        "seed": seed,
    }

    if kind == "constrained":
        result = model.generate_constrained(**common, **generation_kwargs)
    elif kind == "unconstrained":
        result = model.generate_unconstrained(**common, **generation_kwargs)
    else:
        raise ValueError(f"Unknown generation kind: {kind}")

    return _result_payload(result, [])


def _generate_mixed_payload(
    *,
    model_name: str,
    grammar: str,
    prompt: str,
    initial: str = "",
    max_tokens: int = 50,
    grammar_name: Optional[str] = None,
    think_budget: int = 128,
    model_kwargs: Optional[dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    from .environment import ReasoningEnvironment
    from .llm import set_generation_seed

    set_generation_seed(seed)
    model = _load_remote_model(model_name, grammar, model_kwargs)
    resolved_grammar_name = _default_grammar_name(grammar, grammar_name) or "grammar"
    env = ReasoningEnvironment(
        model,
        resolved_grammar_name,
        think_budget=think_budget,
        formal_budget=max_tokens,
        stop_on_complete=True,
    )
    env_result = env.generate(
        prompt,
        initial=initial,
        think_temperature=0.8,
    )
    final = env_result.final_output
    result = GenerationResult(
        text=final.content if final is not None else "",
        is_complete=env_result.is_complete,
        tokens_generated=env_result.total_tokens,
        stopped_reason=env_result.stopped_reason,
    )
    payload = _result_payload(result, [])
    payload.update(
        {
            "thoughts": env_result.all_thoughts,
            "think_tokens": env_result.think_tokens,
            "formal_tokens": env_result.formal_tokens,
            "reasoning_blocks": [str(block) for block in env_result.blocks],
        }
    )
    return payload


def _build_image():
    if _modal is None:
        return None
    image = _modal.Image.debian_slim(python_version="3.12").pip_install(
        "aufbau-rs>=0.1.2",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "accelerate",
        "safetensors",
    )
    if hasattr(image, "add_local_python_source"):
        try:
            image = image.add_local_python_source("p7", "proposition7")
        except Exception:
            pass
    return image


if _modal is not None:
    app = _modal.App(DEFAULT_APP_NAME)
    image = _build_image()

    @app.function(image=image, gpu=DEFAULT_GPU, timeout=60 * 60)
    def generate_constrained(
        model_name: str,
        grammar: str,
        prompt: str,
        initial: str = "",
        max_tokens: int = 50,
        grammar_name: Optional[str] = None,
        stop_on_complete: bool = True,
        model_kwargs: Optional[dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> dict[str, Any]:
        return _generate_payload(
            "constrained",
            model_name=model_name,
            grammar=grammar,
            prompt=prompt,
            initial=initial,
            max_tokens=max_tokens,
            grammar_name=grammar_name,
            model_kwargs=model_kwargs,
            seed=seed,
            stop_on_complete=stop_on_complete,
        )

    @app.function(image=image, gpu=DEFAULT_GPU, timeout=60 * 60)
    def generate_unconstrained(
        model_name: str,
        grammar: str,
        prompt: str,
        initial: str = "",
        max_tokens: int = 50,
        top_k: Optional[int] = None,
        temperature: float = 1.0,
        grammar_name: Optional[str] = None,
        model_kwargs: Optional[dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> dict[str, Any]:
        return _generate_payload(
            "unconstrained",
            model_name=model_name,
            grammar=grammar,
            prompt=prompt,
            initial=initial,
            max_tokens=max_tokens,
            grammar_name=grammar_name,
            model_kwargs=model_kwargs,
            seed=seed,
            top_k=top_k,
            temperature=temperature,
        )

    @app.function(image=image, gpu=DEFAULT_GPU, timeout=60 * 60)
    def generate_mixed(
        model_name: str,
        grammar: str,
        prompt: str,
        initial: str = "",
        max_tokens: int = 50,
        grammar_name: Optional[str] = None,
        think_budget: int = 128,
        model_kwargs: Optional[dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> dict[str, Any]:
        return _generate_mixed_payload(
            model_name=model_name,
            grammar=grammar,
            prompt=prompt,
            initial=initial,
            max_tokens=max_tokens,
            grammar_name=grammar_name,
            think_budget=think_budget,
            model_kwargs=model_kwargs,
            seed=seed,
        )
else:
    app = None

    def generate_constrained(*args: Any, **kwargs: Any) -> dict[str, Any]:
        _require_modal()
        raise AssertionError("unreachable")

    def generate_unconstrained(*args: Any, **kwargs: Any) -> dict[str, Any]:
        _require_modal()
        raise AssertionError("unreachable")

    def generate_mixed(*args: Any, **kwargs: Any) -> dict[str, Any]:
        _require_modal()
        raise AssertionError("unreachable")


def _require_modal():
    if _modal is None:
        raise ImportError("Modal support requires `pip install modal`.")
    return _modal


class ModalDeployment:
    """Client for a deployed Modal-backed proposition7 model."""

    def __init__(
        self,
        model_name: str,
        grammar: str = "stlc",
        *,
        env_path: EnvPath = None,
        token_id: Optional[str] = None,
        token_secret: Optional[str] = None,
        modal_token: Optional[str] = None,
        app_name: str = DEFAULT_APP_NAME,
        gpu: str = DEFAULT_GPU,
        model_kwargs: Optional[dict[str, Any]] = None,
    ):
        load_modal_env(env_path)
        token_id, token_secret = self._resolve_credentials(
            token_id=token_id,
            token_secret=token_secret,
            modal_token=modal_token,
        )

        self.model_name = model_name
        self.grammar = grammar
        self.token_id = token_id
        self.token_secret = token_secret
        self.app_name = app_name
        self.gpu = gpu
        self.model_kwargs = dict(model_kwargs or {})

        if self.token_id:
            os.environ["MODAL_TOKEN_ID"] = self.token_id
        if self.token_secret:
            os.environ["MODAL_TOKEN_SECRET"] = self.token_secret

    @staticmethod
    def _resolve_credentials(
        *,
        token_id: Optional[str],
        token_secret: Optional[str],
        modal_token: Optional[str],
    ) -> tuple[Optional[str], Optional[str]]:
        if modal_token and not (token_id and token_secret):
            if ":" in modal_token:
                token_id, token_secret = modal_token.split(":", 1)
            elif token_secret is None:
                token_secret = modal_token
        return (
            token_id or os.environ.get("MODAL_TOKEN_ID"),
            token_secret or os.environ.get("MODAL_TOKEN_SECRET"),
        )

    def deploy(self) -> "ModalDeployment":
        _require_modal()
        if app is None:
            raise RuntimeError("Modal app is unavailable; install modal and retry.")

        try:
            from modal.runner import deploy_app  # type: ignore[import-not-found]
        except ImportError:
            deploy_app = None

        if deploy_app is not None:
            deploy_app(app, name=self.app_name)
            return self

        if hasattr(app, "deploy"):
            app.deploy()
            return self
        raise RuntimeError("Deploy with `modal deploy -m p7.modal` for this Modal SDK.")

    def _function(self, name: str):
        modal = _require_modal()
        function_cls = getattr(modal, "Function")
        if hasattr(function_cls, "from_name"):
            fn = function_cls.from_name(self.app_name, name)
        elif hasattr(function_cls, "lookup"):
            fn = function_cls.lookup(self.app_name, name)
        else:
            raise RuntimeError("This Modal SDK cannot look up deployed functions.")
        if self.gpu and hasattr(fn, "with_options"):
            try:
                fn = fn.with_options(gpu=self.gpu)
            except TypeError:
                pass
        return fn

    def _remote_call(self, function_name: str, **kwargs: Any) -> dict[str, Any]:
        return self._function(function_name).remote(**kwargs)

    def _base_kwargs(self, prompt: str, initial: str, max_tokens: int) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "grammar": self.grammar,
            "prompt": prompt,
            "initial": initial,
            "max_tokens": max_tokens,
            "model_kwargs": self.model_kwargs,
        }

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
        payload = self._remote_call(
            "generate_constrained",
            **self._base_kwargs(prompt, initial, max_tokens),
            grammar_name=grammar_name,
            stop_on_complete=stop_on_complete,
            seed=seed,
        )
        return _payload_result(payload)

    def generate_unconstrained(
        self,
        prompt: str,
        *,
        initial: str = "",
        max_tokens: int = 50,
        top_k: Optional[int] = None,
        temperature: float = 1.0,
        grammar_name: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        payload = self._remote_call(
            "generate_unconstrained",
            **self._base_kwargs(prompt, initial, max_tokens),
            top_k=top_k,
            temperature=temperature,
            grammar_name=grammar_name,
            seed=seed,
        )
        return _payload_result(payload)

    def generate_mixed(
        self,
        prompt: str,
        *,
        initial: str = "",
        max_tokens: int = 50,
        grammar_name: Optional[str] = None,
        think_budget: int = 128,
        seed: Optional[int] = None,
    ) -> tuple[GenerationResult, dict[str, Any]]:
        payload = self._remote_call(
            "generate_mixed",
            **self._base_kwargs(prompt, initial, max_tokens),
            grammar_name=grammar_name,
            think_budget=think_budget,
            seed=seed,
        )
        return _payload_result(payload), {
            "thoughts": str(payload.get("thoughts", "")),
            "think_tokens": int(payload.get("think_tokens", 0) or 0),
            "formal_tokens": int(payload.get("formal_tokens", 0) or 0),
            "reasoning_blocks": list(payload.get("reasoning_blocks", [])),
        }

    def __call__(self, prompt: str, *, constrained: bool = True, **kwargs: Any):
        if constrained:
            return self.generate_constrained(prompt, **kwargs)
        return self.generate_unconstrained(prompt, **kwargs)


def generate_remote(
    prompt: str,
    *,
    model: str = "gpt2",
    grammar: str = "stlc",
    initial: str = "",
    max_tokens: int = 50,
    reason: bool = False,
    gpu: str = DEFAULT_GPU,
    **kwargs: Any,
):
    if reason:
        raise NotImplementedError("Remote reasoned generation is not wired yet.")
    deployment = ModalDeployment(
        model_name=model,
        grammar=grammar,
        env_path=kwargs.pop("env_path", None),
        token_id=kwargs.pop("token_id", None),
        token_secret=kwargs.pop("token_secret", None),
        modal_token=kwargs.pop("modal_token", None),
        app_name=kwargs.pop("app_name", DEFAULT_APP_NAME),
        gpu=gpu,
        model_kwargs=kwargs.pop("model_kwargs", None),
    )
    result = deployment.generate_constrained(
        prompt=prompt,
        initial=initial,
        max_tokens=max_tokens,
        stop_on_complete=kwargs.pop("stop_on_complete", True),
        seed=kwargs.pop("seed", None),
    )
    from .api import Result

    return Result(
        text=result.text,
        complete=result.is_complete,
        tokens=result.tokens_generated,
        reason=result.stopped_reason,
    )


__all__ = [
    "DEFAULT_APP_NAME",
    "DEFAULT_GPU",
    "ModalDeployment",
    "app",
    "generate_constrained",
    "generate_remote",
    "generate_mixed",
    "generate_unconstrained",
    "load_modal_env",
]
