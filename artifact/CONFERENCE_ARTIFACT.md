# Conference Artifact Deployment

This artifact is intended to support artifact-evaluation review for
`sas26-paper45.pdf`. The primary reviewer path is a job-specific Docker image
that runs the paper reproduction config by default.

## What To Upload

Build the reviewer image on a machine with Docker:

```bash
make job-image JOB_CONFIG=benchmarks/configs/sas26_reproduction.toml
```

Upload these files from `dist/`:

```text
sas26_reproduction.docker.tar
sas26_reproduction.dry-run.txt
sas26_reproduction.git-status.txt
sas26_reproduction.manifest.txt
sas26_reproduction.source-diff.patch
```

The Docker archive contains the source tree, the benchmark tasks, the exact
reproduction config, and dependencies resolved from `uv.lock`. It does not
contain API keys or model caches.

## Reviewer Commands

Smoke-test the baked job without a GPU or OpenRouter key:

```bash
docker load -i sas26_reproduction.docker.tar
docker run --rm proposition7-benchmark-sas26_reproduction:latest job --dry-run
docker run --rm proposition7-benchmark-sas26_reproduction:latest test
```

Run the full reproduction on a GPU host:

```bash
docker run --rm --gpus all \
  -v "$PWD/artifact-output:/workspace/benchmarks/out" \
  -v "$PWD/.env:/workspace/.env:ro" \
  proposition7-benchmark-sas26_reproduction:latest
```

The mounted `.env` must define `OPENROUTER_API_KEY` for the OpenRouter rows in
Figure 7. Local constrained/unconstrained rows use Hugging Face model downloads
and require sufficient GPU memory for the selected model.

## Expected Outputs

The run writes:

```text
artifact-output/sas26-reproduction/config.toml
artifact-output/sas26-reproduction/raw.jsonl
artifact-output/sas26-reproduction/results.json
```

`benchmarks/configs/sas26_reproduction.toml` contains the extracted target
values from Figures 3, 4, and 7. The PDF mode mapping is:

```text
Raw / Unconstr.       -> unconstrained
Direct / Constrained -> constrained_direct
Mixed / Constrained  -> constrained_mixed
```

`unconstrained_cleaned`, `outlines`, and `outlines_mixed` are intentionally not
part of this reproduction config. `openai/gpt-5.4-mini` is evaluated as raw
`unconstrained` only.

## Reproducibility Metadata

The manifest records:

- Docker image name
- Docker archive path
- Baked benchmark config path
- Config SHA-256
- `uv.lock` SHA-256
- Dockerfile SHA-256
- Git commit
- Git status and source diff files
- Base image
- Baked dry-run log path

This is meant to satisfy the usual artifact-evaluation expectations: code,
workflow, exact inputs, expected outputs, environment metadata, and a simple
reviewer command path.
