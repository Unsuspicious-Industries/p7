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


def sample_unconstrained_token(
    logits: List[float],
    tokenizer,
    top_k: int = 50,
    temperature: float = 1.0,
) -> str:
    """Sample a token from logits for more natural unconstrained text."""
    import torch

    logits_t = torch.tensor(logits, dtype=torch.float32)
    if temperature != 1.0:
        logits_t = logits_t / max(temperature, 1e-6)

    vocab_size = logits_t.shape[0]
    if top_k and top_k < vocab_size:
        values, indices = torch.topk(logits_t, top_k)
        probs = torch.softmax(values, dim=-1)
        choice = torch.multinomial(probs, num_samples=1).item()
        token_id = indices[choice].item()
    else:
        probs = torch.softmax(logits_t, dim=-1)
        token_id = torch.multinomial(probs, num_samples=1).item()

    return tokenizer.decode([token_id])




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
