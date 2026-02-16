from __future__ import annotations

import traceback
import logging
import time
import uuid
import threading
import queue
from typing import Any, Dict, List, Optional

from p7.models import get_model_class

from services.grammar import check_partial_completable
from services.streaming import sse
from services.models import mask_repeated_whitespace


_models: Dict[str, Any] = {}
logger = logging.getLogger("p7-backend")
_active_generations: Dict[str, Dict[str, Any]] = {}
_active_lock = threading.Lock()


def _register_generation(mode: str, model_name: str) -> str:
    gen_id = str(uuid.uuid4())
    with _active_lock:
        _active_generations[gen_id] = {
            "mode": mode,
            "model": model_name,
            "started_at": time.time(),
        }
    return gen_id


def _finish_generation(gen_id: str, reason: str) -> None:
    with _active_lock:
        entry = _active_generations.pop(gen_id, None)
    if entry:
        duration = time.time() - entry["started_at"]
        logger.info("generation done id=%s mode=%s model=%s reason=%s duration=%.2fs",
                    gen_id, entry["mode"], entry["model"], reason, duration)


def get_active_generations() -> Dict[str, Dict[str, Any]]:
    with _active_lock:
        snapshot = dict(_active_generations)
    return {
        gen_id: {
            **entry,
            "age_seconds": int(time.time() - entry["started_at"]),
        }
        for gen_id, entry in snapshot.items()
    }


def get_model_and_tokenizer(model_name: str):
    cache_key = f"{model_name}_model"
    tokenizer_key = f"{model_name}_tokenizer"
    if cache_key in _models:
        return _models[tokenizer_key], _models[cache_key]

    model_cls = get_model_class(model_name)
    tokenizer, model = model_cls.load_model_and_tokenizer(model_name)
    _models[tokenizer_key] = tokenizer
    _models[cache_key] = model
    return tokenizer, model


def build_vocab(tokenizer) -> List[str]:
    vocab = []
    vocab_len = len(tokenizer)
    for i in range(vocab_len):
        try:
            token = tokenizer.decode([i])
            vocab.append(token)
        except Exception:
            vocab.append("")
    return vocab


def get_wrapped_model(model_name: str, grammar: str):
    import torch

    tokenizer, model = get_model_and_tokenizer(model_name)
    model_cls = get_model_class(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model_cls(model, tokenizer, grammar=grammar, device=device, model_name=model_name)


def _validate_initial(spec: str, initial: str) -> Optional[str]:
    if not initial:
        return None
    ok, reason = check_partial_completable(spec, initial)
    if not ok:
        return f"Invalid continuation: {reason}"
    return None


def generate_constrained_stream(
    spec: str,
    prompt: str,
    initial: str,
    model_name: str,
    max_tokens: int = 50,
    grammar_tokens: Optional[int] = None,
    stop_on_complete: bool = False,
    mask_whitespace: bool = True,
):
    import torch

    try:
        logger.info(
            "constrained stream model=%s max_tokens=%s grammar_tokens=%s stop_on_complete=%s",
            model_name,
            max_tokens,
            grammar_tokens,
            stop_on_complete,
        )
        error = _validate_initial(spec, initial)
        if error:
            yield sse({'type': 'error', 'message': error})
            return

        gen_id = _register_generation("constrained_stream", model_name)
        wrapped = get_wrapped_model(model_name, spec)
        wrapped.model.eval()
        logger.info("Constrained stream model=%s", model_name)
        yield sse({'type': 'status', 'message': 'Starting constrained generation...'})

        constrained_text = initial
        grammar_budget = grammar_tokens if grammar_tokens is not None else max_tokens

        def logit_filter(logits: List[float], current_text: str) -> List[float]:
            if not mask_whitespace:
                return logits
            return mask_repeated_whitespace(logits, wrapped.vocab, current_text)

        step = 0
        try:
            gen = wrapped.iter_constrained(
                prompt=prompt,
                initial=initial,
                max_tokens=grammar_budget,
                stop_on_complete=stop_on_complete,
                grammar_name=None,
                logit_filter=logit_filter,
            )

            while True:
                try:
                    token = next(gen)
                except StopIteration as e:
                    result = e.value
                    stopped_reason = result.stopped_reason
                    is_complete = result.is_complete
                    break

                constrained_text += token
                step += 1
                yield sse({
                    'type': 'token',
                    'step': step,
                    'text': token,
                    'full_text': constrained_text,
                })

        except TypeError as e:
            yield sse({'type': 'error', 'message': f'Type error: {e}'})
            _finish_generation(gen_id, "type_error")
            return
        except Exception as e:
            logger.exception("Constrained stream failed")
            yield sse({'type': 'error', 'message': f'constrained_failed: {e}'})
            _finish_generation(gen_id, "model_error")
            return

        done_data = {'type': 'done', 'reason': stopped_reason, 'is_complete': is_complete}
        yield sse(done_data)
        _finish_generation(gen_id, stopped_reason)

    except GeneratorExit:
        logger.info("constrained stream client disconnected")
        return
    except ImportError as e:
        yield sse({'type': 'error', 'message': f'Missing dependency: {e}'})
    except Exception as e:
        yield sse({'type': 'error', 'message': str(e), 'traceback': traceback.format_exc()})


def generate_unconstrained_stream(
    prompt: str,
    model_name: str,
    max_tokens: int = 50,
    top_k: int = 50,
    temperature: float = 1.0,
):
    import torch

    try:
        logger.info(
            "unconstrained stream model=%s max_tokens=%s top_k=%s temperature=%s",
            model_name,
            max_tokens,
            top_k,
            temperature,
        )
        gen_id = _register_generation("unconstrained_stream", model_name)
        wrapped = get_wrapped_model(model_name, grammar="")
        wrapped.model.eval()
        logger.info("Unconstrained stream model=%s", model_name)

        yield sse({'type': 'status', 'message': 'Starting unconstrained generation...'})

        generated_text = ""
        token_queue: queue.Queue[str] = queue.Queue()
        result_holder: Dict[str, Any] = {}
        done_event = threading.Event()

        def on_token(token: str, step: int) -> None:
            token_queue.put(token)

        def run_generation() -> None:
            try:
                result_holder["result"] = wrapped.generate_unconstrained(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    top_k=top_k,
                    temperature=temperature,
                    on_token=on_token,
                    grammar_name=None,
                )
            except Exception as e:
                result_holder["error"] = e
            finally:
                done_event.set()

        thread = threading.Thread(target=run_generation, daemon=True)
        thread.start()

        step = 0
        while not done_event.is_set() or not token_queue.empty():
            try:
                token = token_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            generated_text += token
            step += 1
            yield sse({
                'type': 'token',
                'step': step,
                'text': token,
                'full_text': generated_text,
            })

        if "error" in result_holder:
            err = result_holder["error"]
            logger.exception("Unconstrained stream failed")
            yield sse({'type': 'error', 'message': f'unconstrained_failed: {err}'})
            _finish_generation(gen_id, "model_error")
            return

        result = result_holder.get("result")
        stopped_reason = result.stopped_reason if result else "max_tokens"
        done_data = {'type': 'done', 'reason': stopped_reason, 'full_text': generated_text}
        yield sse(done_data)
        _finish_generation(gen_id, stopped_reason)

    except GeneratorExit:
        logger.info("unconstrained stream client disconnected")
        return
    except ImportError as e:
        yield sse({'type': 'error', 'message': f'Missing dependency: {e}'})
    except Exception as e:
        yield sse({'type': 'error', 'message': str(e), 'traceback': traceback.format_exc()})
