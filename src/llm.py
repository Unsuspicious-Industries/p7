from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
import torch

from .inference import GenerationResult


def _dedupe(tokens: List[str]) -> List[str]:
    seen = set()
    return [
        token for token in tokens if token and not (token in seen or seen.add(token))
    ]


def _dedupe_ids(token_ids: List[int]) -> List[int]:
    seen = set()
    return [
        token_id
        for token_id in token_ids
        if token_id >= 0 and not (token_id in seen or seen.add(token_id))
    ]


def _coerce_token_ids(value: Any) -> List[int]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        token_ids: List[int] = []
        for item in value:
            token_ids.extend(_coerce_token_ids(item))
        return token_ids
    try:
        return [int(value)]
    except (TypeError, ValueError):
        return []


def set_generation_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ConstrainedModel:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        grammar: str,
        device: str = "cpu",
        model_name: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.grammar = grammar
        self.device = device
        self.model_name = model_name
        self._input_ids = None
        self._past_key_values = None
        self._pending_input_ids = None

    @staticmethod
    def _dedupe_tokens(tokens: List[str]) -> List[str]:
        return _dedupe(tokens)

    @classmethod
    def tokenizer_kwargs(cls) -> Dict[str, Any]:
        return {}

    @classmethod
    def model_kwargs(cls) -> Dict[str, Any]:
        return {}

    @classmethod
    def load_model_and_tokenizer(cls, model_name: str, **model_kwargs):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "transformers and torch required. pip install transformers torch"
            ) from error

        if str(model_kwargs.get("torch_dtype", "")).lower() == "none":
            model_kwargs = dict(model_kwargs)
            model_kwargs.pop("torch_dtype", None)

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, **dict(cls.tokenizer_kwargs()),
            trust_remote_code=True
        )
        merged_model_kwargs = {**dict(cls.model_kwargs()), **model_kwargs}
        model = AutoModelForCausalLM.from_pretrained(model_name, **merged_model_kwargs)
        return tokenizer, model

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        grammar: str,
        device: str = "cpu",
        **model_kwargs,
    ) -> "ConstrainedModel":
        tokenizer, model = cls.load_model_and_tokenizer(model_name, **model_kwargs)

        if "device_map" in model_kwargs:
            try:
                device = str(next(model.parameters()).device)
            except Exception:
                pass
        else:
            if device == "cpu" and torch.cuda.is_available():
                device = "cuda"
            model.to(device)

        model.eval()
        return cls(model, tokenizer, grammar, device=device, model_name=model_name)

    def format_prompt(self, prompt_text: str) -> str:
        return prompt_text

    def allow_system_prompt(self) -> bool:
        return True

    def think_open(self) -> str:
        return "<think>"

    def think_close(self) -> str:
        return "</think>"

    def start_tokens_unconstrained(
        self, grammar_name: Optional[str] = None
    ) -> List[str]:
        return []

    def start_tokens_constrained(self, grammar_name: Optional[str] = None) -> List[str]:
        return []

    def get_grammar_obj(self) -> str:
        return self.grammar

    def _tokenizer_stop_tokens(self) -> List[str]:
        tokens: List[str] = []
        for attr in ["eos_token", "sep_token", "pad_token", "bos_token"]:
            token = getattr(self.tokenizer, attr, None)
            if token:
                tokens.append(token)
        additional = getattr(self.tokenizer, "additional_special_tokens", None)
        if additional:
            tokens.extend(list(additional))
        return _dedupe(tokens)

    def _tokenizer_stop_token_ids(self) -> List[int]:
        token_ids: List[int] = []
        for attr in ["eos_token_id", "sep_token_id", "pad_token_id", "bos_token_id"]:
            token_ids.extend(_coerce_token_ids(getattr(self.tokenizer, attr, None)))
        additional = getattr(self.tokenizer, "additional_special_tokens_ids", None)
        token_ids.extend(_coerce_token_ids(additional))
        return _dedupe_ids(token_ids)

    def _generation_stop_token_ids(self) -> List[int]:
        token_ids: List[int] = []
        for source in [
            getattr(self.model, "generation_config", None),
            getattr(self.model, "config", None),
        ]:
            if source is None:
                continue
            for attr in [
                "eos_token_id",
                "eos_token_ids",
                "eog_token_id",
                "eog_token_ids",
                "stop_token_id",
                "stop_token_ids",
            ]:
                token_ids.extend(_coerce_token_ids(getattr(source, attr, None)))
        return _dedupe_ids(token_ids)

    def _token_ids_for_stop_tokens(self, stop_tokens: List[str]) -> List[int]:
        token_ids: List[int] = []
        convert = getattr(self.tokenizer, "convert_tokens_to_ids", None)
        unknown_id = getattr(self.tokenizer, "unk_token_id", None)
        unknown_token = getattr(self.tokenizer, "unk_token", None)
        for token in stop_tokens:
            if convert is not None:
                for token_id in _coerce_token_ids(convert(token)):
                    if (
                        unknown_id is not None
                        and token_id == unknown_id
                        and token != unknown_token
                    ):
                        continue
                    token_ids.append(token_id)
            try:
                encoded = self.tokenizer.encode(token, add_special_tokens=False)
            except Exception:
                encoded = []
            if len(encoded) == 1:
                token_ids.extend(_coerce_token_ids(encoded[0]))
        return _dedupe_ids(token_ids)

    def _stop_token_ids(self, stop_tokens: List[str]) -> List[int]:
        return _dedupe_ids(
            self._tokenizer_stop_token_ids()
            + self._generation_stop_token_ids()
            + self._token_ids_for_stop_tokens(stop_tokens)
        )

    def stop_tokens_unconstrained(
        self, grammar_name: Optional[str] = None
    ) -> List[str]:
        tokens = self._tokenizer_stop_tokens() + [self.think_close()]
        return _dedupe(tokens)

    def stop_tokens_constrained(self, grammar_name: Optional[str] = None) -> List[str]:
        tokens = self._tokenizer_stop_tokens()
        # Also include generation stop tokens (eos from generation config)
        stop_ids = self._generation_stop_token_ids()
        convert = getattr(self.tokenizer, "convert_ids_to_tokens", None)
        if convert:
            for token_id in stop_ids:
                try:
                    token = convert(token_id)
                    if token and isinstance(token, str):
                        tokens.append(token)
                except Exception:
                    pass
        # Add common generation ending tags
        extra = ["<role>", "</s>", "<|end|>", "<|eot_id|>"]
        tokens.extend(extra)
        return _dedupe(tokens)

    def _set_prompt(self, prompt: str, initial: str, start_tokens: List[str]) -> None:
        encoded = self.tokenizer(
            self.format_prompt(prompt + "".join(start_tokens)) + initial,
            return_tensors="pt",
        )
        self._input_ids = encoded.input_ids.to(self.device)
        self._past_key_values = None
        self._pending_input_ids = self._input_ids

    def _append_token(self, token: str) -> None:
        token_ids = self.tokenizer.encode(token, add_special_tokens=False)
        if not token_ids:
            return
        new_ids = torch.tensor([token_ids], device=self.device)
        self._input_ids = (
            new_ids
            if self._input_ids is None
            else torch.cat([self._input_ids, new_ids], dim=1)
        )
        self._pending_input_ids = (
            new_ids
            if self._pending_input_ids is None
            else torch.cat([self._pending_input_ids, new_ids], dim=1)
        )

    def _append_token_id(self, token_id: int) -> None:
        new_ids = torch.tensor([[token_id]], device=self.device)
        self._input_ids = (
            new_ids
            if self._input_ids is None
            else torch.cat([self._input_ids, new_ids], dim=1)
        )
        self._pending_input_ids = (
            new_ids
            if self._pending_input_ids is None
            else torch.cat([self._pending_input_ids, new_ids], dim=1)
        )

    def _get_logits(self) -> List[float]:
        logits = self._get_logits_tensor()
        return logits.float().cpu().tolist()

    def _get_logits_tensor(self):
        input_ids = self._pending_input_ids
        if input_ids is None:
            input_ids = self._input_ids[:, -1:]
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                past_key_values=self._past_key_values,
                use_cache=True,
            )
        self._past_key_values = getattr(outputs, "past_key_values", None)
        self._pending_input_ids = None
        return outputs.logits[0, -1, :]

    def _decode_token_id(self, token_id: int) -> str:
        try:
            return self.tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return self.tokenizer.decode([token_id])

    def _token_texts(self, token_id: int) -> List[str]:
        texts = [self._decode_token_id(token_id)]
        convert = getattr(self.tokenizer, "convert_ids_to_tokens", None)
        if convert is not None:
            token = convert(token_id)
            if token is not None:
                texts.append(str(token))
        return _dedupe(texts)

    def _stop_token_label(self, token_id: int, token: Optional[str]) -> str:
        if token:
            return token
        texts = self._token_texts(token_id)
        if texts:
            return texts[0]
        return f"id:{token_id}"

    def _constrained_sample_with_masking(
        self,
        logits: torch.Tensor,
        synthesizer: Any,
        stop_tokens: set[str],
        stop_token_ids: set[int],
    ) -> tuple[Optional[int], Optional[str], bool]:
        temperature = 1.0
        max_retries = 512

        finite = torch.isfinite(logits)
        valid_mask = finite.clone()

        stop_token_ids = set(stop_token_ids)

        for _ in range(max_retries):
            if not valid_mask.any():
                return None, None, False

            valid_logits = logits.clone()
            valid_logits = valid_logits.masked_fill(~valid_mask, float("-inf"))

            probs = torch.softmax(valid_logits / max(temperature, 1e-6), dim=-1)

            if not torch.isfinite(probs).any():
                return None, None, False

            token_id = torch.multinomial(probs, num_samples=1).item()

            if token_id in stop_token_ids:
                return token_id, self._stop_token_label(token_id, None), True

            token = self._decode_token_id(token_id)
            #print(token)
            if not token:
                valid_mask[token_id] = False
                continue

            if token in stop_tokens or any(
                text in stop_tokens for text in self._token_texts(token_id)
            ):
                return token_id, token, True
            if token.strip() == "":
                valid_mask[token_id] = False
                continue

            try:
                synthesizer.try_feed(token)
                return token_id, token, False
            except RuntimeError:
                valid_mask[token_id] = False
                continue

        return None, None, False

    def generate_constrained(
        self,
        prompt: str = "",
        initial: str = "",
        max_tokens: int = 50,
        grammar_name: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        import aufbau

        set_generation_seed(seed)
        self._set_prompt(prompt, initial, self.start_tokens_constrained(grammar_name))
        stop_tokens = set(self.stop_tokens_constrained(grammar_name))
        stop_token_ids = set(self._stop_token_ids(list(stop_tokens)))
        synthesizer = aufbau.Synthesizer(self.grammar, "")

        if initial:
            try:
                synthesizer.set_input(initial)
                synthesizer.parse()
            except Exception as error:
                return GenerationResult(initial, False, 0, f"type_error: {error}")

        tokens_generated = 0
        stopped_reason = "max_tokens"

        for _ in range(max_tokens):
            try:
                logits = self._get_logits_tensor()
                token_id, token, is_stop = self._constrained_sample_with_masking(
                    logits,
                    synthesizer,
                    stop_tokens,
                    stop_token_ids,
                )
            except Exception as error:
                stopped_reason = f"type_error: {error}"
                break

            if token is None:
                stopped_reason = "complete" if synthesizer.is_complete() else "no_valid"
                break

            if is_stop:
                stopped_reason = "complete" if synthesizer.is_complete() else f"stop_token:{token}"
                break

            try:
                synthesizer.feed(token)
                fed = True
            except RuntimeError:
                fed = False
            if not fed:
                stopped_reason = "no_valid"
                break

            tokens_generated += 1
            self._append_token_id(token_id)

        return GenerationResult(
            text=synthesizer.input(),
            is_complete=synthesizer.is_complete(),
            tokens_generated=tokens_generated,
            stopped_reason=stopped_reason,
        )

    def _sample_unconstrained(self, top_k: Optional[int], temperature: float) -> int:
        logits = self._get_logits_tensor().float()
        logits = logits / max(temperature, 1e-6)
        if top_k is not None and 0 < top_k < logits.shape[0]:
            values, indices = torch.topk(logits, top_k)
            probs = torch.softmax(values, dim=-1)
            return indices[torch.multinomial(probs, num_samples=1).item()].item()
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    def generate_unconstrained(
        self,
        prompt: str = "",
        initial: str = "",
        max_tokens: int = 50,
        top_k: Optional[int] = None,
        temperature: float = 1.0,
        stop_tokens: Optional[List[str]] = None,
        grammar_name: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        set_generation_seed(seed)
        stop_tokens = stop_tokens or self.stop_tokens_unconstrained(grammar_name)
        stop_token_ids = set(self._stop_token_ids(stop_tokens))
        self._set_prompt(prompt, initial, self.start_tokens_unconstrained(grammar_name))

        generated_ids: List[int] = []
        tokens_generated = 0
        stopped_reason = "max_tokens"

        for _ in range(max_tokens):
            sampled = self._sample_unconstrained(top_k, temperature)
            if sampled in stop_token_ids:
                stopped_reason = f"stop_token:{self._stop_token_label(sampled, None)}"
                break
            next_ids = generated_ids + [sampled]
            next_text = self.tokenizer.decode(next_ids)
            stop_token = next(
                (stop for stop in stop_tokens if stop and next_text.endswith(stop)),
                None,
            )
            if stop_token:
                stopped_reason = f"stop_token:{stop_token}"
                break

            prev_text = self.tokenizer.decode(generated_ids) if generated_ids else ""
            token = next_text[len(prev_text) :]
            generated_ids = next_ids
            tokens_generated += 1
            self._append_token(token)

        return GenerationResult(
            text=initial + self.tokenizer.decode(generated_ids),
            is_complete=False,
            tokens_generated=tokens_generated,
            stopped_reason=stopped_reason,
        )