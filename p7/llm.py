"""LLM intergration for typed constrained generation."""

from __future__ import annotations

from typing import List, Optional, Callable, Dict, Any, Iterator, Tuple

from .inference import GenerationResult
from .sampler import TypedSampler
from .sampler import _dbg


class ConstrainedModel:
    """HuggingFace model wrapped with typed constrainted decoding."""
    
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

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",
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
            new_ids = torch.tensor([token_ids]).to(self.device)
            if self._input_ids is None:
                self._input_ids = new_ids
            else:
                self._input_ids = torch.cat([self._input_ids, new_ids], dim=1)
    
    def iter_constrained(
        self,
        initial: str = "",
        prompt: str = "",
        max_tokens: int = 50,
        greedy_k: int = 1,
        pre_top_k: Optional[int] = 100,
        stop_on_complete: bool = False,
        grammar_name: Optional[str] = None,
        logit_filter: Optional[Callable[[List[float], str], List[float]]] = None,
    ) -> Iterator[str]:
        start_tokens = "".join(self.start_tokens_constrained(grammar_name=grammar_name))
        full_prompt = self.format_prompt(prompt + start_tokens) + initial
        self._input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)

        def logit_fn():
            logits = self._get_logits()
            if logit_filter:
                return logit_filter(logits, sampler.current_text())
            return logits

        sampler = TypedSampler(grammar=self.grammar, vocab=self.vocab, logit_fn=logit_fn)

        if initial:
            try:
                sampler.feed(initial)
            except TypeError as e:
                return GenerationResult(
                    text=initial,
                    is_complete=False,
                    tokens_generated=0,
                    stopped_reason=f"type_error: {e}",
                )

        if stop_on_complete and sampler.is_complete():
            return GenerationResult(
                text=sampler.current_text(),
                is_complete=True,
                tokens_generated=0,
                stopped_reason="complete",
            )

        tokens_generated = 0
        stopped_reason = "max_tokens"
        for step in range(max_tokens):
            next_token = sampler.infer_greedy(k=greedy_k, pre_top_k=pre_top_k)
            if next_token is None:
                stopped_reason = "no_valid"
                break

            try:
                sampler.feed(next_token)
            except TypeError as e:
                stopped_reason = f"type_error: {e}"
                break

            tokens_generated += 1
            self._update_input_ids(next_token)
            yield next_token

            if stop_on_complete and sampler.is_complete():
                stopped_reason = "complete"
                break

        return GenerationResult(
            text=sampler.current_text(),
            is_complete=sampler.is_complete(),
            tokens_generated=tokens_generated,
            stopped_reason=stopped_reason,
        )

    def generate(
        self,
        initial: str = "",
        prompt: str = "",
        max_tokens: int = 50,
        greedy_k: int = 1,
        pre_top_k: Optional[int] = 100,
        on_token: Optional[Callable[[str, int], None]] = None,
        grammar_name: Optional[str] = None,
        logit_filter: Optional[Callable[[List[float], str], List[float]]] = None,
    ) -> GenerationResult:
        tokens_generated = 0
        gen = self.iter_constrained(
            prompt=prompt,
            initial=initial,
            max_tokens=max_tokens,
            greedy_k=greedy_k,
            pre_top_k=pre_top_k,
            stop_on_complete=False,
            grammar_name=grammar_name,
            logit_filter=logit_filter,
        )

        while True:
            try:
                token = next(gen)
            except StopIteration as e:
                return e.value
            tokens_generated += 1
            if on_token:
                on_token(token, tokens_generated - 1)
    
    def until_complete(
        self,
        initial: str = "",
        prompt: str = "",
        max_tokens: int = 100,
        greedy_k: int = 1,
        pre_top_k: Optional[int] = 100,
        on_token: Optional[Callable[[str, int], None]] = None,
        grammar_name: Optional[str] = None,
        logit_filter: Optional[Callable[[List[float], str], List[float]]] = None,
    ) -> GenerationResult:
        tokens_generated = 0
        gen = self.iter_constrained(
            prompt=prompt,
            initial=initial,
            max_tokens=max_tokens,
            greedy_k=greedy_k,
            pre_top_k=pre_top_k,
            stop_on_complete=True,
            grammar_name=grammar_name,
            logit_filter=logit_filter,
        )

        while True:
            try:
                token = next(gen)
            except StopIteration as e:
                return e.value
            tokens_generated += 1
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
                return e.value
            tokens_generated += 1
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
        self._input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)
        generated_token_ids = []
        tokens_generated = 0
        stopped_reason = "max_tokens"

        for step in range(max_tokens):
            logits = self._get_logits()
            _dbg("generate_unconstrained step {}: logits={}", step, logits[:10])
            logits_t = torch.tensor(logits, dtype=torch.float32)
            if temperature != 1.0:
                logits_t = logits_t / max(temperature, 1e-6)

            if top_k and top_k > 0 and top_k < logits_t.shape[0]:
                values, indices = torch.topk(logits_t, top_k)
                probs = torch.softmax(values, dim=-1)
                choice = torch.multinomial(probs, num_samples=1).item()
                top_idx = indices[choice].item()
            else:
                probs = torch.softmax(logits_t, dim=-1)
                top_idx = torch.multinomial(probs, num_samples=1).item()
            token_str = self.tokenizer.decode([top_idx])

            if token_str in stop_tokens or any(
                (len(stop) > 1 and token_str.endswith(stop)) for stop in stop_tokens
            ):
                stopped_reason = "stop_token"
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
            yield token_display

        generated_text = initial + self.tokenizer.decode(generated_token_ids)

        return GenerationResult(
            text=generated_text,
            is_complete=False,
            tokens_generated=tokens_generated,
            stopped_reason=stopped_reason,
        )
