import argparse
import os
from typing import Any


def coerce_token_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        ids: list[int] = []
        for item in value:
            ids.extend(coerce_token_ids(item))
        return ids
    try:
        token_id = int(value)
    except (TypeError, ValueError):
        return []
    return [token_id] if token_id >= 0 else []


def dedupe(ids: list[int]) -> list[int]:
    seen = set()
    return [
        token_id for token_id in ids if not (token_id in seen or seen.add(token_id))
    ]


def token_label(tokenizer: Any, token_id: int) -> str:
    try:
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        decoded = tokenizer.decode([token_id])
    if decoded:
        return decoded
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if convert is not None:
        token = convert(token_id)
        if token is not None:
            return str(token)
    return f"id:{token_id}"


def force_token(token_id: int):
    def logit_filter(logits: list[float], current_text: str) -> list[float]:
        del current_text
        filtered = [float("-inf")] * len(logits)
        filtered[token_id] = 0.0
        return filtered

    return logit_filter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify real model EOS/EOG token IDs stop constrained generation."
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    if not args.allow_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import proposition7

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "none": None,
    }[args.torch_dtype]
    local_files_only = not args.allow_download

    print(f"Loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=local_files_only,
    )
    model_kwargs: dict[str, Any] = {"local_files_only": local_files_only}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if not args.device_map:
        model.to(args.device)
    model.eval()

    device = str(next(model.parameters()).device)
    cm = proposition7.ConstrainedModel(
        model,
        tokenizer,
        proposition7.get_grammar("toy"),
        device=device,
        model_name=args.model,
    )

    tokenizer_eos_ids = coerce_token_ids(getattr(tokenizer, "eos_token_id", None))
    generation_eos_ids = dedupe(
        coerce_token_ids(
            getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        )
        + coerce_token_ids(
            getattr(getattr(model, "config", None), "eos_token_id", None)
        )
    )
    eos_ids = dedupe(tokenizer_eos_ids + generation_eos_ids)
    if not eos_ids:
        raise AssertionError("model/tokenizer exposes no EOS token IDs")
    eos_id = eos_ids[0]

    eog_ids = dedupe(
        coerce_token_ids(
            getattr(getattr(model, "generation_config", None), "eog_token_id", None)
        )
        + coerce_token_ids(
            getattr(getattr(model, "generation_config", None), "eog_token_ids", None)
        )
        + coerce_token_ids(
            getattr(getattr(model, "config", None), "eog_token_id", None)
        )
        + coerce_token_ids(
            getattr(getattr(model, "config", None), "eog_token_ids", None)
        )
        + [token_id for token_id in generation_eos_ids if token_id != eos_id]
    )
    role_ids = [
        token_id
        for token_id in coerce_token_ids(
            getattr(tokenizer, "additional_special_tokens_ids", None)
        )
        if token_id != eos_id and token_id not in eog_ids
    ]

    stop_ids = set(cm._stop_token_ids(cm.stop_tokens_constrained("toy")))
    print("Constrained stop IDs:", sorted(stop_ids))
    print(
        "EOS IDs:",
        [(token_id, token_label(tokenizer, token_id)) for token_id in eos_ids],
    )
    print(
        "EOG IDs:",
        [(token_id, token_label(tokenizer, token_id)) for token_id in eog_ids],
    )
    print(
        "Role/special IDs:",
        [(token_id, token_label(tokenizer, token_id)) for token_id in role_ids],
    )

    if not eog_ids:
        raise AssertionError("model/tokenizer exposes no separate EOG token IDs")
    eog_id = eog_ids[0]
    assert eos_id in stop_ids
    assert eog_id in stop_ids
    role_id = role_ids[0] if role_ids else None
    if role_id is not None:
        assert role_id in stop_ids

    def run_case(name: str, token_id: int, initial: str, complete: bool) -> None:
        label = token_label(tokenizer, token_id)
        result = cm.generate_constrained(
            prompt="Stop-token smoke test. Output only a valid toy expression.",
            initial=initial,
            max_tokens=3,
            logit_filter=force_token(token_id),
        )
        print(
            f"{name}: forced {token_id} {label!r}, text={result.text!r}, "
            f"complete={result.is_complete}, tokens={result.tokens_generated}, "
            f"reason={result.stopped_reason!r}"
        )
        assert result.text == initial
        assert result.is_complete is complete
        assert result.tokens_generated == 0
        assert result.stopped_reason.startswith("stop_token:")

    run_case("eos_after_complete", eos_id, "beep:Fizz", True)
    run_case("eog_after_complete", eog_id, "beep:Fizz", True)
    if role_id is not None:
        run_case("role_before_complete", role_id, "", False)
    else:
        run_case("eog_before_complete", eog_id, "", False)
    print("Stop-token checks passed")


if __name__ == "__main__":
    raise SystemExit(main())
