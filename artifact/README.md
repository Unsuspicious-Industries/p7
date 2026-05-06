# Artifact Container

Build a loadable Docker image tarball:

```bash
./scripts/build_artifact_image.sh
```

Load and run the smoke suite:

```bash
docker load -i dist/p7-benchmark-artifact.tar
docker run --rm -v "$PWD/artifact-output:/workspace/benchmarks/out" p7-benchmark-artifact:latest smoke
```

Run the full paper config on a GPU host:

```bash
docker run --rm --gpus all \
  -v "$PWD/artifact-output:/workspace/benchmarks/out" \
  -v "$PWD/.env:/workspace/.env:ro" \
  p7-benchmark-artifact:latest paper --resume
```

The container writes `raw.jsonl` and `results.json` under the mounted output directory.

## Modal A10G

Modal supports custom containers via `modal.Image.from_dockerfile(...)` and runtime
container execution via `modal.Sandbox.create(...)`.

Run the small Qwen validation suite on an A10G:

```bash
pip install -e ".[modal]"
modal setup
python scripts/modal_sandbox_run.py --config benchmarks/configs/modal_qwen_smoke.toml
```

The launcher copies `raw.jsonl` and `results.json` back to `dist/modal-qwen-smoke/`.
