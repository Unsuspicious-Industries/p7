"""LLM integration for typed constrained generation."""

from __future__ import annotations

from typing import List, Optional, Callable, Dict, Any, Iterator, Tuple

from .inference import GenerationResult, _select_and_feed_token
from .sampler import TypedSampler
from .sampler import _dbg, _debug_enabled
from .p7 import Grammar


def _short_repr(value: Any, limit: int = 200) -> str:
    rendered = repr(value)
    if len(rendered) <= limit:
        return rendered
    return rendered[: max(limit - 3, 0)] + "..."


def _preview_list(values: List[Any], limit: int = 10) -> List[Any]:
    if len(values) <= limit:
        return list(values)
    return list(values[:limit]) + [f"...(+{len(values) - limit} more)"]


class ConstrainedModel:
    """HuggingFace model wrapped with typed constrained decoding."""
    
    def __init__(
        self,
        model,
        tokenizer,
        grammar: str,
        device: str = "cpu",
        model_name: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.grammar = grammar
        self.model_name = model_name
        self._grammar_obj = Grammar(grammar) if grammar else None
        # Resolve actual device from model when possible
        try:
            param_device = str(next(self.model.parameters()).device)
        except Exception:
            param_device = device

        if hasattr(self.model, "hf_device_map") and self.model.hf_device_map:
            self.device = param_device
        else:
            self.device = device
            if param_device != device:
                try:
                    self.model.to(self.device)
                except Exception:
                    self.device = param_device
        self.vocab = [tokenizer.decode([i]) for i in range(len(tokenizer))]
        self._input_ids = None
        _dbg(
            "ConstrainedModel init: model_name={!r} device={} param_device={} vocab_size={} grammar_loaded={}",
            self.model_name,
            self.device,
            param_device,
            len(self.vocab),
            bool(self._grammar_obj),
        )

    def _input_length(self) -> int:
        if self._input_ids is None:
            return 0
        try:
            return int(self._input_ids.shape[1])
        except Exception:
            return 0

    def _debug_top_logits(self, logits: List[float], limit: int = 5) -> None:
        if not _debug_enabled() or not logits:
            return

        top_indices = sorted(range(len(logits)), key=lambda idx: logits[idx], reverse=True)[:limit]
        preview = [
            {
                "id": idx,
                "token": _short_repr(self.vocab[idx], 60),
                "logit": round(float(logits[idx]), 6),
            }
            for idx in top_indices
        ]
        _dbg("top_logits(limit={}): {}", limit, preview)

    def _debug_result(self, label: str, result: GenerationResult) -> None:
        _dbg(
            "{} result: text={} is_complete={} tokens_generated={} stopped_reason={!r}",
            label,
            _short_repr(result.text, 240),
            result.is_complete,
            result.tokens_generated,
            result.stopped_reason,
        )

    def get_grammar_obj(self) -> Grammar:
        return self._grammar_obj

    @classmethod
    def tokenizer_kwargs(cls) -> Dict[str, Any]:
        return {}

    @classmethod
    def model_kwargs(cls) -> Dict[str, Any]:
        return {}

    @classmethod
    def load_model_and_tokenizer(
        cls,
        model_name: str,
        **model_kwargs,
    ) -> Tuple[Any, Any]:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "transformers and torch required. "
                "pip install transformers torch"
            )

        load_kwargs = cls.tokenizer_kwargs()
        merged_model_kwargs = {
            **cls.model_kwargs(),
            **model_kwargs,
        }

        tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)

        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            model_size_gb = cls.estimate_model_size_gb(model_name)

            if model_size_gb > gpu_memory_gb * 0.7:
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                        **merged_model_kwargs,
                    )
                    return tokenizer, cls._align_model_tokenizer(tokenizer, model)
                except Exception:
                    try:
                        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            quantization_config=quantization_config,
                            device_map="auto",
                            low_cpu_mem_usage=True,
                            **merged_model_kwargs,
                        )
                        return tokenizer, cls._align_model_tokenizer(tokenizer, model)
                    except Exception:
                        pass

            if model_size_gb < gpu_memory_gb * 0.8:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    **merged_model_kwargs,
                )
                return tokenizer, cls._align_model_tokenizer(tokenizer, model)

            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                **merged_model_kwargs,
            )
            return tokenizer, cls._align_model_tokenizer(tokenizer, model)

        if "device_map" not in merged_model_kwargs:
            merged_model_kwargs["device_map"] = "cpu"
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            **merged_model_kwargs,
        )
        return tokenizer, cls._align_model_tokenizer(tokenizer, model)

    @classmethod
    def estimate_model_size_gb(cls, model_name: str) -> float:
        lower = model_name.lower()
        if "13b" in lower:
            return 26
        if "7b" in lower or "6.9b" in lower:
            return 14
        if "3b" in lower or "2.8b" in lower:
            return 6
        return 7

    @staticmethod
    def _align_model_tokenizer(tokenizer, model):
        try:
            model_vocab = model.get_input_embeddings().num_embeddings
            tokenizer_vocab = len(tokenizer)
            if tokenizer_vocab > model_vocab:
                model.resize_token_embeddings(tokenizer_vocab)
        except Exception:
            pass
        return model

    def format_prompt(self, prompt_text: str) -> str:
        return prompt_text

    def allow_system_prompt(self) -> bool:
        return True

    def think_open(self) -> str:
        return "<think>"

    def think_close(self) -> str:
        return "</think>"

    def start_tokens(self, mode: str, grammar_name: Optional[str] = None) -> List[str]:
        return []

    def start_tokens_unconstrained(self, grammar_name: Optional[str] = None) -> List[str]:
        add_bos = getattr(self.tokenizer, "add_bos_token", False)
        bos_token = getattr(self.tokenizer, "bos_token", None)
        if not add_bos and bos_token:
            return [bos_token]
        return []

    def start_tokens_constrained(self, grammar_name: Optional[str] = None) -> List[str]:
        add_bos = getattr(self.tokenizer, "add_bos_token", False)
        bos_token = getattr(self.tokenizer, "bos_token", None)
        if not add_bos and bos_token:
            return [bos_token]
        return []

    def _tokenizer_stop_tokens(self) -> List[str]:
        tokens: List[str] = []
        for attr in ["eos_token", "sep_token", "pad_token", "bos_token"]:
            token = getattr(self.tokenizer, attr, None)
            if token:
                tokens.append(token)
        additional = getattr(self.tokenizer, "additional_special_tokens", None)
        if additional:
            tokens.extend(list(additional))
        special_map = getattr(self.tokenizer, "special_tokens_map", None)
        if special_map:
            for value in special_map.values():
                if isinstance(value, list):
                    tokens.extend([v for v in value if v])
                elif value:
                    tokens.append(value)
        return self._dedupe_tokens(tokens)

    @staticmethod
    def _dedupe_tokens(tokens: List[str]) -> List[str]:
        seen = set()
        ordered = []
        for token in tokens:
            if not token or token in seen:
                continue
            seen.add(token)
            ordered.append(token)
        return ordered

    def stop_tokens_unconstrained(self, grammar_name: Optional[str] = None) -> List[str]:
        grammar_open = f"<{grammar_name}>" if grammar_name else ""
        grammar_close = f"</{grammar_name}>" if grammar_name else ""
        base = self._tokenizer_stop_tokens()
        extra = [
            self.think_close(),
            grammar_open,
            grammar_close,
            "<|end|>",
            "<|endoftext|>",
            "<|eot_id|>",
        ]
        return self._dedupe_tokens(base + extra)

    def stop_tokens_constrained(self, grammar_name: Optional[str] = None) -> List[str]:
        grammar_close = f"</{grammar_name}>" if grammar_name else ""
        base = self._tokenizer_stop_tokens()
        extra = [
            grammar_close,
            "<|end|>",
            "<|endoftext|>",
            "<|eot_id|>",
        ]
        return self._dedupe_tokens(base + extra)
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        grammar: str,
        device: str = "cpu",
        **model_kwargs,
    ) -> "ConstrainedModel":
        try:
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch required. "
                "pip install transformers torch"
            )

        tokenizer, model = cls.load_model_and_tokenizer(model_name, **model_kwargs)
        
        # Only move to device if not using device_map
        if "device_map" not in model_kwargs:
            model.to(device)
        else:
            # When using device_map, get actual device from model
            device = str(next(model.parameters()).device)
        
        model.eval()
        
        return cls(model, tokenizer, grammar, device, model_name=model_name)
    
    def _get_logits(self) -> List[float]:
        import torch
        
        with torch.no_grad():
            outputs = self.model(self._input_ids.to(self.device), use_cache=False)
            # Convert to float32 before tolist() for float16 models
            logits = outputs.logits[0, -1, :].float().cpu().tolist()
        return logits
    
    def _update_input_ids(self, token: str):
        import torch
        
        token_ids = self.tokenizer.encode(token, add_special_tokens=False)
        if token_ids:
            old_length = self._input_length()
            new_ids = torch.tensor([token_ids]).to(self.device)
            if self._input_ids is None:
                self._input_ids = new_ids
            else:
                self._input_ids = torch.cat([self._input_ids, new_ids], dim=1)
            _dbg(
                "_update_input_ids: token={!r} token_ids={} old_len={} new_len={}",
                token,
                token_ids,
                old_length,
                self._input_length(),
            )
        else:
            _dbg("_update_input_ids: token={!r} produced no token ids", token)
    
    def iter_constrained(
        self,
        initial: str = "",
        prompt: str = "",
        max_tokens: int = 50,
        greedy_k: int = 1,
        pre_top_k: Optional[int] = 100,
        use_feed_only: bool = False,
        stop_on_complete: bool = False,
        grammar_name: Optional[str] = None,
        logit_filter: Optional[Callable[[List[float], str], List[float]]] = None,
    ) -> Iterator[str]:
        start_tokens = "".join(self.start_tokens_constrained(grammar_name=grammar_name))
        full_prompt = self.format_prompt(prompt + start_tokens) + initial
        _dbg(
            "iter_constrained start: model_name={!r} device={} grammar_name={!r} max_tokens={} greedy_k={} pre_top_k={} use_feed_only={} stop_on_complete={} start_tokens={} prompt={} initial={}",
            self.model_name,
            self.device,
            grammar_name,
            max_tokens,
            greedy_k,
            pre_top_k,
            use_feed_only,
            stop_on_complete,
            _short_repr(start_tokens, 120),
            _short_repr(prompt, 180),
            _short_repr(initial, 180),
        )
        self._input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)
        _dbg(
            "iter_constrained prompt encoded: full_prompt={} input_len={}",
            _short_repr(full_prompt, 260),
            self._input_length(),
        )

        def logit_fn():
            current_text = sampler.current_text()
            logits = self._get_logits()
            _dbg(
                "iter_constrained logit_fn: current_text={} input_len={}",
                _short_repr(current_text, 180),
                self._input_length(),
            )
            self._debug_top_logits(logits, limit=5)
            if logit_filter:
                filtered = logit_filter(logits, current_text)
                if _debug_enabled():
                    changed = sum(1 for before, after in zip(logits, filtered) if before != after)
                    _dbg("iter_constrained logit_filter applied: changed_logits={}", changed)
                    self._debug_top_logits(filtered, limit=5)
                return filtered
            return logits

        sampler = TypedSampler(grammar=self.grammar, vocab=self.vocab, logit_fn=logit_fn)
        if _debug_enabled():
            try:
                completions = sampler.synthesizer.get_completions()
                _dbg(
                    "iter_constrained initial completions (n={}): {}",
                    len(completions),
                    _preview_list(completions, limit=20),
                )
            except Exception as e:
                _dbg("iter_constrained initial completions error: {}", e)

        if initial:
            try:
                sampler.set_input(initial)
            except TypeError as e:
                result = GenerationResult(
                    text=initial,
                    is_complete=False,
                    tokens_generated=0,
                    stopped_reason=f"type_error: {e}",
                )
                self._debug_result("iter_constrained(initial type error)", result)
                return result

        if stop_on_complete and sampler.is_complete():
            result = GenerationResult(
                text=sampler.current_text(),
                is_complete=True,
                tokens_generated=0,
                stopped_reason="complete",
            )
            self._debug_result("iter_constrained(initially complete)", result)
            return result

        tokens_generated = 0
        stopped_reason = "max_tokens"
        for step in range(max_tokens):
            _dbg(
                "iter_constrained step {}: current_text={} input_len={}",
                step,
                _short_repr(sampler.current_text(), 220),
                self._input_length(),
            )
            try:
                next_token = _select_and_feed_token(
                    sampler=sampler,
                    greedy_k=greedy_k,
                    pre_top_k=pre_top_k,
                    use_feed_only=use_feed_only,
                )
            except Exception as e:
                stopped_reason = f"type_error: {e}"
                _dbg("iter_constrained step {}: selection error={}", step, e)
                break

            if next_token is None:
                stopped_reason = "no_valid"
                _dbg(
                    "iter_constrained step {}: no valid token current_text={}",
                    step,
                    _short_repr(sampler.current_text(), 240),
                )
                break

            tokens_generated += 1
            _dbg("iter_constrained step {}: selected token={!r}", step, next_token)
            self._update_input_ids(next_token)
            _dbg(
                "iter_constrained step {}: emitted token={!r} tokens_generated={} current_text={} is_complete={}",
                step,
                next_token,
                tokens_generated,
                _short_repr(sampler.current_text(), 240),
                sampler.is_complete(),
            )
            yield next_token

            if stop_on_complete and sampler.is_complete():
                stopped_reason = "complete"
                _dbg("iter_constrained step {}: stopping because sampler is complete", step)
                break

        result = GenerationResult(
            text=sampler.current_text(),
            is_complete=sampler.is_complete(),
            tokens_generated=tokens_generated,
            stopped_reason=stopped_reason,
        )
        self._debug_result("iter_constrained", result)
        return result

    def generate(
        self,
        initial: str = "",
        prompt: str = "",
        max_tokens: int = 50,
        greedy_k: int = 1,
        pre_top_k: Optional[int] = 100,
        use_feed_only: bool = False,
        on_token: Optional[Callable[[str, int], None]] = None,
        grammar_name: Optional[str] = None,
        logit_filter: Optional[Callable[[List[float], str], List[float]]] = None,
    ) -> GenerationResult:
        tokens_generated = 0
        _dbg(
            "generate start: grammar_name={!r} max_tokens={} greedy_k={} pre_top_k={} use_feed_only={} prompt={} initial={}",
            grammar_name,
            max_tokens,
            greedy_k,
            pre_top_k,
            use_feed_only,
            _short_repr(prompt, 180),
            _short_repr(initial, 180),
        )
        gen = self.iter_constrained(
            prompt=prompt,
            initial=initial,
            max_tokens=max_tokens,
            greedy_k=greedy_k,
            pre_top_k=pre_top_k,
            use_feed_only=use_feed_only,
            stop_on_complete=False,
            grammar_name=grammar_name,
            logit_filter=logit_filter,
        )

        while True:
            try:
                token = next(gen)
            except StopIteration as e:
                result = e.value
                self._debug_result("generate", result)
                return result
            tokens_generated += 1
            _dbg("generate token {}: {!r}", tokens_generated - 1, token)
            if on_token:
                on_token(token, tokens_generated - 1)
    
    def until_complete(
        self,
        initial: str = "",
        prompt: str = "",
        max_tokens: int = 100,
        greedy_k: int = 1,
        pre_top_k: Optional[int] = 100,
        use_feed_only: bool = False,
        on_token: Optional[Callable[[str, int], None]] = None,
        grammar_name: Optional[str] = None,
        logit_filter: Optional[Callable[[List[float], str], List[float]]] = None,
    ) -> GenerationResult:
        tokens_generated = 0
        _dbg(
            "until_complete start: grammar_name={!r} max_tokens={} greedy_k={} pre_top_k={} use_feed_only={} prompt={} initial={}",
            grammar_name,
            max_tokens,
            greedy_k,
            pre_top_k,
            use_feed_only,
            _short_repr(prompt, 180),
            _short_repr(initial, 180),
        )
        gen = self.iter_constrained(
            prompt=prompt,
            initial=initial,
            max_tokens=max_tokens,
            greedy_k=greedy_k,
            pre_top_k=pre_top_k,
            use_feed_only=use_feed_only,
            stop_on_complete=True,
            grammar_name=grammar_name,
            logit_filter=logit_filter,
        )

        while True:
            try:
                token = next(gen)
            except StopIteration as e:
                result = e.value
                self._debug_result("until_complete", result)
                return result
            tokens_generated += 1
            _dbg("until_complete token {}: {!r}", tokens_generated - 1, token)
            if on_token:
                on_token(token, tokens_generated - 1)
    
    def generate_unconstrained(
        self,
        initial: str = "",
        prompt: str = "",
        max_tokens: int = 50,
        top_k: Optional[int] = None,
        temperature: float = 1.0,
        on_token: Optional[Callable[[str, int], None]] = None,
        stop_tokens: Optional[List[str]] = None,
        grammar_name: Optional[str] = None,
    ) -> GenerationResult:
        """Generate tokens without any grammar/type constraints (raw model output)."""
        tokens_generated = 0
        _dbg(
            "generate_unconstrained start: grammar_name={!r} max_tokens={} top_k={} temperature={} prompt={} initial={} stop_tokens={}",
            grammar_name,
            max_tokens,
            top_k,
            temperature,
            _short_repr(prompt, 180),
            _short_repr(initial, 180),
            _preview_list(stop_tokens or [], limit=20),
        )
        gen = self.iter_unconstrained(
            prompt=prompt,
            initial=initial,
            max_tokens=max_tokens,
            top_k=top_k,
            temperature=temperature,
            stop_tokens=stop_tokens,
            grammar_name=grammar_name,
        )

        while True:
            try:
                token = next(gen)
            except StopIteration as e:
                result = e.value
                self._debug_result("generate_unconstrained", result)
                return result
            tokens_generated += 1
            _dbg("generate_unconstrained token {}: {!r}", tokens_generated - 1, token)
            if on_token:
                on_token(token, tokens_generated - 1)

    def iter_unconstrained(
        self,
        initial: str = "",
        prompt: str = "",
        max_tokens: int = 50,
        top_k: Optional[int] = None,
        temperature: float = 1.0,
        stop_tokens: Optional[List[str]] = None,
        grammar_name: Optional[str] = None,
    ) -> Iterator[str]:
        import torch

        if stop_tokens is None:
            stop_tokens = self.stop_tokens_unconstrained(grammar_name=grammar_name)

        start_tokens = "".join(self.start_tokens_unconstrained(grammar_name=grammar_name))
        full_prompt = self.format_prompt(prompt + start_tokens) + initial
        _dbg(
            "iter_unconstrained start: model_name={!r} device={} grammar_name={!r} max_tokens={} top_k={} temperature={} start_tokens={} stop_tokens={} prompt={} initial={}",
            self.model_name,
            self.device,
            grammar_name,
            max_tokens,
            top_k,
            temperature,
            _short_repr(start_tokens, 120),
            _preview_list(stop_tokens, limit=20),
            _short_repr(prompt, 180),
            _short_repr(initial, 180),
        )
        self._input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)
        _dbg(
            "iter_unconstrained prompt encoded: full_prompt={} input_len={}",
            _short_repr(full_prompt, 260),
            self._input_length(),
        )
        generated_token_ids = []
        tokens_generated = 0
        stopped_reason = "max_tokens"

        for step in range(max_tokens):
            _dbg(
                "iter_unconstrained step {}: input_len={} tokens_generated={} partial_text={}",
                step,
                self._input_length(),
                tokens_generated,
                _short_repr(initial + self.tokenizer.decode(generated_token_ids), 220),
            )
            logits = self._get_logits()
            self._debug_top_logits(logits, limit=5)
            logits_t = torch.tensor(logits, dtype=torch.float32)
            if temperature != 1.0:
                logits_t = logits_t / max(temperature, 1e-6)

            sampling_mode = "full_vocab"
            if top_k and top_k > 0 and top_k < logits_t.shape[0]:
                values, indices = torch.topk(logits_t, top_k)
                probs = torch.softmax(values, dim=-1)
                choice = torch.multinomial(probs, num_samples=1).item()
                top_idx = indices[choice].item()
                sampling_mode = f"top_k={top_k}"
            else:
                probs = torch.softmax(logits_t, dim=-1)
                top_idx = torch.multinomial(probs, num_samples=1).item()
            token_str = self.tokenizer.decode([top_idx])
            _dbg(
                "iter_unconstrained step {}: sampled mode={} token_id={} token={}",
                step,
                sampling_mode,
                top_idx,
                _short_repr(token_str, 80),
            )

            matched_stop = None
            if token_str in stop_tokens:
                matched_stop = token_str
            else:
                for stop in stop_tokens:
                    if len(stop) > 1 and token_str.endswith(stop):
                        matched_stop = stop
                        break

            if matched_stop is not None:
                stopped_reason = "stop_token"
                _dbg(
                    "iter_unconstrained step {}: stop token matched token={} stop={}",
                    step,
                    _short_repr(token_str, 80),
                    _short_repr(matched_stop, 80),
                )
                break

            generated_token_ids.append(top_idx)
            tokens_generated += 1

            new_ids = torch.tensor([[top_idx]]).to(self.device)
            self._input_ids = torch.cat([self._input_ids, new_ids], dim=1)

            current_text = self.tokenizer.decode(generated_token_ids)
            if tokens_generated == 1:
                token_display = current_text
            else:
                prev_text = self.tokenizer.decode(generated_token_ids[:-1])
                token_display = current_text[len(prev_text):]
            _dbg(
                "iter_unconstrained step {}: emitted token={!r} tokens_generated={} current_text={}",
                step,
                token_display,
                tokens_generated,
                _short_repr(initial + current_text, 240),
            )
            yield token_display

        generated_text = initial + self.tokenizer.decode(generated_token_ids)

        result = GenerationResult(
            text=generated_text,
            is_complete=False,
            tokens_generated=tokens_generated,
            stopped_reason=stopped_reason,
        )
        self._debug_result("iter_unconstrained", result)
        return result
