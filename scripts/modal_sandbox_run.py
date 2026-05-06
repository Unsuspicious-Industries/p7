#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import threading


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the benchmark artifact container on Modal using a Sandbox."
    )
    parser.add_argument(
        "--config",
        default="benchmarks/configs/modal_qwen_smoke.toml",
        help="Benchmark config path inside the repository.",
    )
    parser.add_argument(
        "--gpu",
        default="A10G",
        help="Modal GPU type, e.g. A10G.",
    )
    parser.add_argument(
        "--app-name",
        default="p7-benchmark-sandbox",
        help="Modal app name used to own the sandbox.",
    )
    parser.add_argument(
        "--run-dir",
        default="/workspace/benchmarks/out/modal-qwen-smoke",
        help="Deterministic run directory inside the sandbox.",
    )
    parser.add_argument(
        "--local-out-dir",
        default="dist/modal-qwen-smoke",
        help="Local directory where raw.jsonl and results.json are copied back.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=6 * 60 * 60,
        help="Sandbox timeout in seconds.",
    )
    parser.add_argument(
        "runner_args",
        nargs=argparse.REMAINDER,
        help="Additional args forwarded to benchmarks/run.py after '--'.",
    )
    return parser.parse_args()


def main() -> None:
    try:
        import modal
    except ImportError as error:  # pragma: no cover - runtime environment dependent
        raise SystemExit(
            "Modal is not installed. Run `pip install -e \".[modal]\"`."
        ) from error

    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    local_out_dir = (root / args.local_out_dir).resolve()
    local_out_dir.mkdir(parents=True, exist_ok=True)

    config_path = args.config
    runner_args = list(args.runner_args)
    if runner_args and runner_args[0] == "--":
        runner_args = runner_args[1:]

    image = modal.Image.from_dockerfile(
        str(root / "artifact" / "Dockerfile"),
        context_dir=str(root),
    ).entrypoint([])

    command = [
        "python",
        "benchmarks/run.py",
        "--config",
        config_path,
        "--run-dir",
        args.run_dir,
        *runner_args,
    ]

    app = modal.App.lookup(args.app_name, create_if_missing=True)

    with modal.enable_output():
        sandbox = modal.Sandbox.create(
            "sleep",
            str(args.timeout),
            app=app,
            image=image,
            gpu=args.gpu,
            timeout=args.timeout,
            cpu=8,
            memory=32768,
        )

    try:
        process = sandbox.exec(*command)

        def _drain(stream, writer):
            for line in stream:
                print(line, end="", file=writer)

        stdout_thread = threading.Thread(
            target=_drain, args=(process.stdout, sys.stdout), daemon=True
        )
        stderr_thread = threading.Thread(
            target=_drain, args=(process.stderr, sys.stderr), daemon=True
        )
        stdout_thread.start()
        stderr_thread.start()

        returncode = process.wait()
        stdout_thread.join()
        stderr_thread.join()
        if returncode != 0:
            raise SystemExit(f"Benchmark command failed with exit code {returncode}")

        remote_run_dir = args.run_dir.rstrip("/")
        sandbox.filesystem.copy_to_local(
            f"{remote_run_dir}/raw.jsonl", str(local_out_dir / "raw.jsonl")
        )
        sandbox.filesystem.copy_to_local(
            f"{remote_run_dir}/results.json", str(local_out_dir / "results.json")
        )
        print(f"Copied raw.jsonl to {local_out_dir / 'raw.jsonl'}")
        print(f"Copied results.json to {local_out_dir / 'results.json'}")
    finally:
        sandbox.terminate()
        sandbox.detach()


if __name__ == "__main__":
    main()
