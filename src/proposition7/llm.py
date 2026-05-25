from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, Dict, List, Optional

try:
    import torch
except ImportError:  # pragma: no cover - exercised in packaging/import smoke tests
    torch = None  # type: ignore[assignment]

from .inference import GenerationResult


def _require_torch():
    if torch is None:
        raise ImportError(
            "Local generation requires torch. Install with `pip install -e '.[transformers]'`."
        )
    return torch


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

    torch = _require_torch()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _masked_entropy_bits(masked_logits: torch.Tensor) -> float:
    """Shannon entropy of the post-mask distribution, in bits.

    H = -∑ p_i · log₂(p_i)   where p_i = softmax(masked_logits)[i]

    masked_logits has -inf at grammar-invalid positions; softmax maps those
    to p=0.  We select only finite entries before computing the log to avoid
    log₂(0) = -inf.  Returns 0.0 for a fully masked or single-token
    distribution.
    """
    torch = _require_torch()
    finite_mask = torch.isfinite(masked_logits)
    if not finite_mask.any():
        return 0.0
    probs = torch.softmax(masked_logits.float(), dim=-1)
    valid_probs = probs[finite_mask]
    # Clamp avoids log₂(0) from numerical underflow near 0.
    log_probs = torch.log2(valid_probs.clamp(min=1e-40))
    return -(valid_probs * log_probs).sum().item()


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
            model_name, **dict(cls.tokenizer_kwargs()), trust_remote_code=True
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
        torch = _require_torch()
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

    def start_tokens_unconstrained(self) -> List[str]:
        return []

    def start_tokens_constrained(self) -> List[str]:
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



    def stop_tokens(self) -> List[str]:
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
        # Add think close token (same as unconstrained)
        tokens.append(self.think_close())
        # Add common generation ending tags
        extra = ["<role>", "</s>", "<|end|>", "<|eot_id|>"]
        tokens.extend(extra)
        return _dedupe(tokens)

    def stop_tokens_unconstrained(self) -> List[str]:
        return self.stop_tokens()

    def stop_tokens_constrained(self) -> List[str]:
        return self.stop_tokens()

    def _set_prompt(
        self,
        prompt: str,
        initial: str,
        start_tokens: Sequence[str] = (),
    ) -> None:
        encoded = self.tokenizer(
            self.format_prompt(prompt + "".join(start_tokens)) + initial,
            return_tensors="pt",
        )
        self._input_ids = encoded.input_ids.to(self.device)
        self._past_key_values = None
        self._pending_input_ids = self._input_ids

    def _append_token(self, token: str) -> None:
        torch = _require_torch()
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
        torch = _require_torch()
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
        torch = _require_torch()
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
        accumulated_input: str,
        stop_tokens: set[str],
        stop_token_ids: set[int],
        greedy=False,
        max_retries=2048,
        temperature=0.0,
    ) -> tuple[Optional[int], Optional[str], bool, str, float, float, int]:
        """Sample the next token under grammar constraints.

        Returns (token_id, token, is_stop, new_accumulated, pre_entropy, post_entropy, retries).

        new_accumulated: accumulated_input with the accepted token_content appended.
            token_content is the canonical form: the raw decoded string if it is a
            valid grammar prefix, or the lstrip()-ed version as a fallback (see
            SPACING WORKAROUND below).  Equals accumulated_input unchanged when
            token is None or is_stop is True.

        pre_entropy: Shannon entropy H (bits) of the model's raw distribution,
            before any grammar masking.
            H = -∑ p_i · log₂(p_i)  over all tokens with finite logits.
            This is computed once at the start, before the retry loop, so it
            is independent of how many candidates were tried.

        post_entropy: Shannon entropy H (bits) over valid_logits at acceptance
            time — the distribution after grammar masking and retry exclusions.
            H = -∑ p_i · log₂(p_i)  over grammar-valid tokens only.
            For zero-retry steps this equals the true grammar-valid entropy.
            For steps with retries, rejected tokens are absent (masked to -inf),
            so this is a lower bound on the true grammar-valid entropy.
            Temperature is NOT applied; H is always at T=1.

        retries: grammar-rejected candidates before the accepted token was found.
            High values indicate grammatically sparse positions in the lattice.
        """
        torch = _require_torch()
        # JANK: aufbau uses regex-based tokenization on the raw character string.
        # try_feed() treats its argument as a whitespace-delimited unit, so feeding
        # an LM sub-word token like ' 3' after '4' would produce '4 3' (two grammar
        # tokens) instead of '43' (one integer terminal). We therefore use parse()
        # on the full accumulated string instead of try_feed().
        #
        # SPACING WORKAROUND: LM tokenizers prefix most tokens with a leading space
        # (e.g. ' x', ' +', ' 3'). aufbau's regex terminals require that the full
        # character sequence match — e.g. '4 3' fails because the integer regex
        # sees two separate tokens. Strategy: try the raw token first (preserves
        # keyword/identifier boundaries like 'let' + ' x' → 'let x'), then fall back
        # to the lstripped version (handles digit continuation '4' + ' 3' → '43').
        import aufbau

        finite = torch.isfinite(logits)
        valid_mask = finite.clone()

        # Pre-mask entropy: model's raw uncertainty before grammar filtering.
        # Computed once here so it is independent of the retry loop.
        pre_entropy = _masked_entropy_bits(logits)

        stop_token_ids = set(stop_token_ids)
        # Single reusable synthesizer 
        test_synth = aufbau.Synthesizer(self.grammar, "")
        retries = 0

        for _ in range(max_retries):
            if not valid_mask.any():
                return None, None, False, accumulated_input, pre_entropy, 0.0, retries

            # Zero-out invalid logits with -inf so softmax assigns them p=0.
            valid_logits = logits.masked_fill(~valid_mask, float("-inf"))

            if greedy or temperature == 0.0:
                token_id = torch.argmax(valid_logits).item()
            else:
                # Scale by temperature, then re-normalise.
                # p_i = exp(l_i / T) / Z  (valid tokens only)
                probs = torch.softmax(valid_logits.float() / max(temperature, 1e-6), dim=-1)
                if not torch.isfinite(probs).any():
                    return None, None, False, accumulated_input, pre_entropy, 0.0, retries
                token_id = torch.multinomial(probs, num_samples=1).item()

            token = self._decode_token_id(token_id)
            if not token:
                valid_mask[token_id] = False
                retries += 1
                continue

            if token.strip() == "":
                valid_mask[token_id] = False
                retries += 1
                continue

            # Stop tokens (EOS, </s>, <|im_end|>, etc.) must be checked BEFORE
            # grammar validation: they won't form a valid grammar prefix, so they
            # would be masked otherwise and the model could never emit its natural
            # stop signal.
            if (
                token_id in stop_token_ids
                or token in stop_tokens
                or any(text in stop_tokens for text in self._token_texts(token_id))
            ):
                post_entropy = _masked_entropy_bits(valid_logits)
                return token_id, token, True, accumulated_input, pre_entropy, post_entropy, retries

            # Try token as-is first, then lstripped as fallback.
            # as-is preserves keyword/id boundaries ("let x" not "letx").
            # stripped fixes digit continuation ("43" not "4 3").
            token_content = None
            for candidate in (token, token.lstrip()):
                if not candidate:
                    continue
                test_synth.set_input(accumulated_input + candidate)
                try:
                    test_synth.parse()
                    token_content = candidate
                    break
                except RuntimeError:
                    pass

            if token_content is None:
                # Neither form is a valid grammar prefix; exclude from future samples.
                valid_mask[token_id] = False
                retries += 1
                continue

            # Valid grammar token accepted, we compute post-mask entropy at this
            # point (grammar-valid tokens only, minus retry-excluded tokens).
            post_entropy = _masked_entropy_bits(valid_logits)
            return token_id, token, False, accumulated_input + token_content, pre_entropy, post_entropy, retries

        # Exhausted all retries, return no valid token found.
        return None, None, False, accumulated_input, pre_entropy, 0.0, retries

    def generate_constrained(
        self,
        prompt: str = "",
        initial: str = "",
        max_tokens: int = 50,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        logit_filter: Optional[Callable[[torch.Tensor, str], Any]] = None,
    ) -> GenerationResult:
        import aufbau

        torch = _require_torch()
        set_generation_seed(seed)
        self._set_prompt(prompt, initial, self.start_tokens_constrained())
        stop_tokens = set(self.stop_tokens_constrained())
        stop_token_ids = set(self._stop_token_ids(list(stop_tokens)))
        synthesizer = aufbau.Synthesizer(self.grammar, "")

        if initial:
            try:
                synthesizer.set_input(initial)
                synthesizer.parse()
            except Exception as error:
                return GenerationResult(initial, False, 0, f"type_error: {error}")

        # accumulated_input: the canonical grammar-prefix string built during decoding.
        # Starts as `initial`; grows by token_content at each accepted step.
        # token_content may be the lstripped form of the decoded token (SPACING
        # WORKAROUND in _constrained_sample_with_masking), so accumulated_input
        # can diverge from tokenizer.decode(all_generated_ids).
        # It is what aufbau validates and is returned as result.text.
        accumulated_input = initial
        tokens_generated = 0
        stopped_reason = "max_tokens"
        step_token_ids: list[int] = []
        step_pre_entropies: list[float] = []
        step_entropies: list[float] = []
        step_retries: list[int] = []

        for i in range(max_tokens):
            print(f"Step {i}")
            try:
                logits = self._get_logits_tensor()
                if logit_filter is not None:
                    filtered = logit_filter(logits, accumulated_input)
                    if not isinstance(filtered, torch.Tensor):
                        filtered = torch.tensor(filtered, dtype=logits.dtype, device=logits.device)
                    logits = filtered.to(device=logits.device, dtype=logits.dtype)
                token_id, token, is_stop, accumulated_input, pre_entropy, post_entropy, retries = (
                    self._constrained_sample_with_masking(
                        logits,
                        accumulated_input,
                        stop_tokens,
                        stop_token_ids,
                        temperature=temperature,
                    )
                )
            except Exception as error:
                stopped_reason = f"type_error: {error}"
                break

            if token is None:
                # Grammar can't have "no valid" positions except when complete
                # If grammar is complete, report "complete" - not max_tokens
                if synthesizer.is_complete():
                    stopped_reason = "complete"
                else:
                    stopped_reason = "no_valid"
                break

            if is_stop:
                # When stop token hit, always report it - don't check is_complete
                stopped_reason = f"stop_token:{token}"
                break

            # Use set_input + parse instead of deprecated feed() to keep synthesizer
            # in sync with accumulated_input (our parse() based validation state)
            try:
                synthesizer.set_input(accumulated_input)
                synthesizer.parse()
                fed = True
            except RuntimeError:
                fed = False
            if not fed:
                # Check if accumulated_input is complete using fresh synthesizer state
                check_synth = aufbau.Synthesizer(self.grammar, "")
                check_synth.set_input(accumulated_input)
                try:
                    check_synth.parse()
                    is_complete = check_synth.is_complete()
                except:
                    is_complete = False
                if is_complete:
                    stopped_reason = "complete"
                else:
                    stopped_reason = "no_valid"
                break

            step_token_ids.append(token_id)
            step_pre_entropies.append(pre_entropy)
            step_entropies.append(post_entropy)
            step_retries.append(retries)
            tokens_generated += 1
            self._append_token_id(token_id)

        # Final is_complete check on accumulated_input
        final_synth = aufbau.Synthesizer(self.grammar, "")
        final_synth.set_input(accumulated_input)
        try:
            final_synth.parse()
            final_is_complete = final_synth.is_complete()
        except:
            final_is_complete = False

        return GenerationResult(
            text=accumulated_input,
            is_complete=final_is_complete,
            tokens_generated=tokens_generated,
            stopped_reason=stopped_reason,
            step_token_ids=step_token_ids,
            step_pre_entropies=step_pre_entropies,
            step_entropies=step_entropies,
            step_retries=step_retries,
        )

    def _sample_unconstrained(self, temperature: float, top_k: int = 0) -> int:
        torch = _require_torch()
        logits = self._get_logits_tensor().float()
        if top_k > 0 and top_k < logits.numel():
            top_values, _ = torch.topk(logits, top_k)
            logits = logits.masked_fill(logits < top_values[-1], float("-inf"))
        if temperature == 0.0:
            return torch.argmax(logits).item()
        logits = logits / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    def generate_unconstrained(
        self,
        prompt: str = "",
        initial: str = "",
        max_tokens: int = 50,
        temperature: float = 0.0,
        stop_tokens: Optional[List[str]] = None,
        seed: Optional[int] = None,
        top_k: int = 0,
    ) -> GenerationResult:
        set_generation_seed(seed)
        stop_tokens = stop_tokens or self.stop_tokens_unconstrained()
        stop_token_ids = set(self._stop_token_ids(list(stop_tokens)))
        self._set_prompt(prompt, initial, self.start_tokens_unconstrained())

        generated_ids: List[int] = []
        tokens_generated = 0
        stopped_reason = "max_tokens"

        for _ in range(max_tokens):
            sampled = self._sample_unconstrained(temperature, top_k=top_k)
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
