import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import proposition7
from benchmarks.api import load_tasks, run_interaction


def first_task(language: str):
    for task in load_tasks([language]):
        if task.initial:
            return task
    raise RuntimeError(f"No task with initial prefix found for {language}")


def main() -> int:
    model_name = os.environ.get("P7_DEBUG_MODEL", "Qwen/Qwen3.5-9B")
    device = os.environ.get("P7_DEBUG_DEVICE", "cuda")
    print(f"Python {sys.version.split()[0]}")
    print(f"Model {model_name}")
    print(f"Device {device}")

    fun_task = first_task("fun")
    stlc_task = first_task("stlc")
    print(f"Fun task {fun_task.task_id} initial={fun_task.initial!r}")
    print(f"STLC task {stlc_task.task_id} initial={stlc_task.initial!r}")

    model = proposition7.get_model_class(model_name).from_pretrained(
        model_name,
        grammar=proposition7.get_grammar(fun_task.grammar),
        device=device,
        torch_dtype="float16" if device == "cuda" else "float32",
        device_map=device,
    )

    records = []
    for task in [fun_task, stlc_task]:
        model.grammar = proposition7.get_grammar(task.grammar)
        record = run_interaction(
            model,
            task,
            "constrained_direct",
            seed=7,
        )
        records.append((task, record))
        print(
            f"{task.grammar} {task.task_id}: error={record['error']} stop={record['stop_reason']} "
            f"tokens={record['tokens']} output={record['output']!r}"
        )
        assert record["output"].startswith(task.initial), (
            task.task_id,
            task.initial,
            record["output"],
        )

    print("All benchmark smoke assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
