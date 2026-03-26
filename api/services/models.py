from __future__ import annotations

from typing import Dict, List



def mask_repeated_whitespace(logits: List[float], vocab: List[str], current_text: str) -> List[float]:
    """Heuristic: reduce whitespace spam in constrained generation."""
    masked = list(logits)
    ends_with_ws = bool(current_text) and current_text[-1].isspace()

    for i, token in enumerate(vocab):
        if not token:
            continue
        if ends_with_ws:
            # If we already end in whitespace, block any whitespace-leading token
            if token.strip() == "" or token[0].isspace():
                masked[i] = -1e9

    return masked


def get_device_info() -> Dict[str, str]:
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "device": "cuda",
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": str(torch.cuda.device_count()),
            }
        return {"device": "cpu", "gpu_name": "", "gpu_count": "0"}
    except Exception:
        return {"device": "cpu", "gpu_name": "", "gpu_count": "0"}
