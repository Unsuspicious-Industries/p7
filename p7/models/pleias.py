"""
PleIAs model adapters.

Covers the SYNTH reasoning series:
  - PleIAs/Baguettotron  (321M, multilingual, vocab=65536)
  - PleIAs/Monad         (56.7M, English-only, vocab=8192)

Both models share the same ChatML-style prompt format and native
<think> / </think> reasoning tokens. The key differences from other
adapters are:

  1. Custom tokenizer (slow, no trust_remote_code needed)
  2. <|im_start|> / <|im_end> chat delimiters (note: Baguettotron uses
     `<|im_end>` without a closing `>` as a single token — the chat
     template handles this automatically via apply_chat_template)
  3. EOS is <|end_of_text|> (token id 2) on both models
  4. The model seeds its own <think> block: the assistant turn begins
     with "<think>\n" so we match DeepSeek's start-token pattern
  5. No trust_remote_code required
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .chat import ChatConstrainedModel


class PleiasConstrainedModel(ChatConstrainedModel):
    """
    Adapter for PleIAs SYNTH-series reasoning models.

    Handles Baguettotron (321M) and Monad (56.7M).  Both use:
    - ChatML prompt format (<|im_start|> / <|im_end>)
    - Native <think> / </think> reasoning tokens
    - <|end_of_text|> as the primary EOS token
    - bfloat16 weights, LlamaForCausalLM architecture

    The assistant turn is expected to begin with "<think>\\n", so
    `start_tokens_unconstrained()` seeds that automatically — consistent
    with how DeepseekConstrainedModel works.
    """

    # ------------------------------------------------------------------ #
    # Class-level overrides                                               #
    # ------------------------------------------------------------------ #

    @classmethod
    def tokenizer_kwargs(cls) -> Dict[str, Any]:
        # PleIAs tokenizers are slow (PreTrainedTokenizer, not Fast).
        # use_fast=False avoids the "fast tokenizer not available" warning.
        return {"use_fast": False}

    @classmethod
    def model_kwargs(cls) -> Dict[str, Any]:
        return {}

    # ------------------------------------------------------------------ #
    # Reasoning token customisation                                       #
    # ------------------------------------------------------------------ #

    def think_open(self) -> str:
        # The model is trained to begin its assistant turn with "<think>\n"
        return "<think>\n"

    def think_close(self) -> str:
        return "</think>"

    # ------------------------------------------------------------------ #
    # Start tokens                                                        #
    # ------------------------------------------------------------------ #

    def start_tokens_unconstrained(self, grammar_name: Optional[str] = None) -> List[str]:
        # Seed the <think> block so the model reasons before producing output.
        return [self.think_open()]

    def start_tokens_constrained(self, grammar_name: Optional[str] = None) -> List[str]:
        # In constrained mode we jump straight to grammar output — no think preamble.
        return []

    # ------------------------------------------------------------------ #
    # Stop tokens                                                         #
    # ------------------------------------------------------------------ #

    def stop_tokens_unconstrained(self, grammar_name: Optional[str] = None) -> List[str]:
        extra = [
            "<|end_of_text|>",
            "<|begin_of_text|>",
            "<|im_end>",    # Baguettotron's im_end (no trailing >)
            "<|im_end|>",   # Monad's im_end variant
            "<|im_start|>",
        ]
        return self._dedupe_tokens(super().stop_tokens_unconstrained(grammar_name) + extra)

    def stop_tokens_constrained(self, grammar_name: Optional[str] = None) -> List[str]:
        extra = [
            "<|end_of_text|>",
            "<|begin_of_text|>",
            "<|im_end>",
            "<|im_end|>",
            "<|im_start|>",
        ]
        return self._dedupe_tokens(super().stop_tokens_constrained(grammar_name) + extra)

    # ------------------------------------------------------------------ #
    # Prompt formatting                                                   #
    # ------------------------------------------------------------------ #

    def format_prompt(self, prompt_text: str) -> str:
        """
        Apply the ChatML template.

        Falls back to a hand-crafted template if the tokenizer's built-in
        template is unavailable (e.g. older HF versions or offline cache).
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt_text}]
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        # Manual ChatML fallback (matches Baguettotron/Monad training format)
        return (
            f"<|im_start|>user\n"
            f"{prompt_text}<|im_end>\n"
            f"<|im_start|>assistant\n"
        )
