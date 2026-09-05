"""The model-facing boundary of the decode loop.

proposition7 constrains generation; it does not run models. Everything the
masked loop needs from a language model is below, and nothing else is allowed
to reach it: no tensors, no framework types, no weights, no devices.

A `Runtime` is a *sequence in progress*. `reset` starts one, `extend` accepts a
token into it, and `logits` reports the model's distribution over the next
token given everything accepted so far. Implementations are free to keep a KV
cache, the loop only ever appends, never rewinds, so a cache is always valid.

Rejection happens above this line. When the mask rejects a candidate the loop
simply does not `extend`, and asks for no new logits; a runtime therefore never
learns that a rejection occurred and needs no rollback.

Two implementations exist:

    proposition7.llm.ConstrainedModel   transformers + torch, for local use
    wirt.runtime.TinygradRuntime        tinygrad + GGUF, for serving

Both satisfy the conformance suite in `tests/runtime_contract.py`, which is the
real definition of this protocol; the docstrings below are its prose.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    import numpy as np


@runtime_checkable
class Runtime(Protocol):
    """One model, one sequence at a time."""

    @property
    def vocab_size(self) -> int:
        """Length of every array `logits` returns. Constant for the runtime's life."""

    def encode_prompt(
        self,
        model_context: Sequence[tuple[str, str]],
        *,
        initial: str = "",
        assistant_prefix: str | None = None,
    ) -> list[int]:
        """Turn ordered (role, content) pairs into prompt tokens.

        Chat templating lives here because it is a property of the tokenizer,
        not of the constraint. `initial` is text the caller has already fixed,
        it is appended after the generation prompt and is subject to the mask
        like anything else.

        `assistant_prefix` is the opposite: it is written into the token context
        after the generation prompt and is *not* shown to the grammar. It exists
        for the framing a chat template puts between the assistant header and
        the model's first real token, for a reasoning model, an opened or an
        already-closed `<think>` block. That framing is part of the prompt, not
        part of the answer, so constraining it would be a category error.
        `None` means "whatever this runtime defaults to".

        Raises `ValueError` if the model has no template for these roles.
        """

    def reset(self, prompt_ids: Sequence[int]) -> None:
        """Begin a new sequence at `prompt_ids`, discarding any previous one.

        After this call `logits()` reports the distribution over the token that
        follows the prompt.
        """

    def extend(self, token_id: int) -> None:
        """Accept `token_id` as the next token of the sequence.

        After this call `logits()` reports the distribution over the token that
        follows it. Only ever called with an id the loop actually accepted.
        """

    def logits(self) -> "np.ndarray":
        """Next-token logits: a finite-or-`-inf` float32 array of `vocab_size`.

        Raw scores, not probabilities: no temperature, no top-k, no penalty.
        Sampling is the loop's business, because the mask has to be applied to
        the distribution before anything is drawn from it. `-inf` marks a token
        the runtime itself excludes; NaN is never permitted.
        """

    def token_text(self, token_id: int) -> str:
        """The token's surface text, with its own spacing preserved.

        The loop offers this string to the grammar in several spellings (raw,
        lstripped, space-joined), so a leading space must not be stripped here.
        Undecodable byte fragments return `""`, which the loop skips.
        """

    def is_stop(self, token_id: int) -> bool:
        """True for EOS, EOT, and any end-of-turn token of this model family.

        Checked *before* the grammar mask: a stop token is not a grammar token,
        so masking would forbid the model from ever ending its turn.
        """
