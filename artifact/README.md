# Artifact Docker

The reviewer artifact is Docker-first. The image contains the full repository
snapshot needed to run the main benchmark config and any benchmark config present
under `benchmarks/configs/` at build time. It installs the Python dependencies
during the image build, but it does not include API keys, Hugging Face/model caches,
model weights, benchmark outputs, or `backup/`.

## Build Outputs

Build everything needed for review:

```bash
make artifact
```

This writes:

```text
dist/proposition7-benchmark-artifact.tar
dist/proposition7-review-bundle.tar.gz
dist/proposition7-review-manifest.txt
dist/proposition7-review.git-status.txt
dist/proposition7-review.source-diff.patch
```

The Docker archive is the runnable environment. The source bundle is included so
reviewers can inspect the exact tree, copy/edit configs, and mount an edited tree
over `/workspace` if they want additional experiments.

## Quick Checks

Load the image:

```bash
docker load -i dist/proposition7-benchmark-artifact.tar
```

Dry-run the main benchmark config config without GPU or API access:

```bash
docker run --rm proposition7-benchmark-artifact:latest \
  python benchmarks/run.py --config benchmarks/configs/main.toml --dry-run
```

List included configs:

```bash
docker run --rm proposition7-benchmark-artifact:latest \
  python -c 'from pathlib import Path; print("\n".join(str(p) for p in sorted(Path("benchmarks/configs").glob("*.toml"))))'
```

## Full Benchmark Run

The main benchmark config includes local GPU rows and OpenRouter rows. Create a
local `.env` file before the full run:

```bash
printf 'OPENROUTER_API_KEY=...\n' > .env
mkdir -p artifact-output hf-cache
```

Run the benchmark:

```bash
docker run --rm --gpus all \
  --env-file .env \
  -v "$PWD/artifact-output:/workspace/benchmarks/out" \
  -v "$PWD/hf-cache:/cache/huggingface" \
  proposition7-benchmark-artifact:latest \
  python benchmarks/run.py --config benchmarks/configs/main.toml --resume
```

Expected outputs:

```text
artifact-output/lmpl-main/config.toml
artifact-output/lmpl-main/raw.jsonl
artifact-output/lmpl-main/results.json
```

Models are downloaded at run time into the mounted `hf-cache/` directory. Keeping
that cache outside the image prevents the Docker archive from containing model
weights while still allowing resumed/repeated runs to reuse downloads.

## Other Experiments

Run any included config by changing `--config`:

```bash
docker run --rm --gpus all \
  --env-file .env \
  -v "$PWD/artifact-output:/workspace/benchmarks/out" \
  -v "$PWD/hf-cache:/cache/huggingface" \
  proposition7-benchmark-artifact:latest \
  python benchmarks/run.py --config benchmarks/configs/paper.toml --resume
```

For local-only configs, omit `--env-file .env`. For OpenRouter-only configs,
omit `--gpus all`.

To run an edited source tree or edited configs, unpack the source bundle and
mount it over `/workspace`:

```bash
tar -xzf dist/proposition7-review-bundle.tar.gz -C /tmp
docker run --rm --gpus all \
  --env-file .env \
  -v "/tmp/p7:/workspace:ro" \
  -v "$PWD/artifact-output:/workspace/benchmarks/out" \
  -v "$PWD/hf-cache:/cache/huggingface" \
  proposition7-benchmark-artifact:latest \
  python benchmarks/run.py --config benchmarks/configs/main.toml --resume
```

## Bundle Hygiene

`.dockerignore` and the source-bundle step exclude local state that should not be
submitted or baked into the image: `.env`, virtualenvs, `dist/`, benchmark output
directories, `backup/`, Hugging Face/model caches, and common model-weight file
extensions such as `.safetensors`, `.bin`, `.pt`, `.pth`, `.ckpt`, `.gguf`, and
`.onnx`.
