import os
import signal
import sys
import time


MODEL = "Qwen/Qwen3.5-9B"


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import proposition7

    print(f"Python {sys.version.split()[0]}")
    print(f"PyTorch {torch.__version__}")
    print(f"CUDA available {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU {torch.cuda.get_device_name(0)}")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    print("Tokenizer OK")

    print("\nLoading model...")
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        local_files_only=True,
        torch_dtype=dtype,
        device_map=device_map,
    )
    print("Model OK")

    print("\n=== Test 1: Forward with use_cache=True ===")
    inputs = tokenizer("Hello", return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
    t0 = time.time()
    with torch.no_grad():
        out1 = model(**inputs, use_cache=True)
    print(f"Took {time.time() - t0:.3f}s")
    print(f"Has past_key_values: {hasattr(out1, 'past_key_values')}")

    print("\n=== Test 2: Incremental forward with past ===")
    next_ids = torch.tensor([[100]], device=inputs["input_ids"].device)
    t1 = time.time()
    with torch.no_grad():
        out2 = model(next_ids, past_key_values=out1.past_key_values, use_cache=True)
    print(f"Took {time.time() - t1:.3f}s")
    print(f"Logits shape: {out2.logits.shape}")

    print("\n=== Test 3: ConstrainedModel generate_constrained ===")
    cm = proposition7.ConstrainedModel(
        model,
        tokenizer,
        grammar="start ::= 'a' 'b'",
        device=str(inputs["input_ids"].device),
        model_name=MODEL,
    )
    print("ConstrainedModel OK")

    def alarm_handler(signum, frame):
        del signum, frame
        raise TimeoutError("generation timed out")

    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(30)
    try:
        result = cm.generate_constrained(
            prompt="Complete: ",
            initial="",
            max_tokens=3,
        )
        signal.alarm(0)
        print(f"Result text: {result.text!r}")
        print(f"Tokens generated: {result.tokens_generated}")
        print(f"Stop reason: {result.stopped_reason}")
    except Exception as error:
        signal.alarm(0)
        print(f"ERROR: {error}")
        raise

    print("\n=== Done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
