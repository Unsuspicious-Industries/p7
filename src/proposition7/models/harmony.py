"""Pure helpers for gpt-oss harmony-format completions."""

_FINAL_MARKER = "<|channel|>final<|message|>"
_END_MARKERS = ("<|return|>", "<|end|>", "<|call|>")


def extract_harmony_final(text: str) -> str:
    """Return final-channel content, or unchanged text without one."""
    marker_at = text.rfind(_FINAL_MARKER)
    if marker_at == -1:
        return text
    content = text[marker_at + len(_FINAL_MARKER) :]
    cut = min(
        (at for at in (content.find(marker) for marker in _END_MARKERS) if at != -1),
        default=len(content),
    )
    return content[:cut]
