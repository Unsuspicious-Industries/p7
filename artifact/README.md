# Artifact Container

Build artifacts through the unified Makefile:

```bash
make artifact
```

For conference artifact evaluation, prefer a job-specific image. It bakes the
exact benchmark config path into the image metadata and makes the selected job
the default container command. Dependencies are installed from `uv.lock` during
the image build:

```bash
make job-image JOB_CONFIG=benchmarks/configs/sas26_reproduction.toml
```

This writes a Docker archive, manifest, and dry-run log under `dist/`, for
example:

```text
dist/sas26_reproduction.docker.tar
dist/sas26_reproduction.dry-run.txt
dist/sas26_reproduction.git-status.txt
dist/sas26_reproduction.manifest.txt
dist/sas26_reproduction.source-diff.patch
```

Reviewer execution is then just:

```bash
docker load -i dist/sas26_reproduction.docker.tar
docker run --rm --gpus all \
  -v "$PWD/artifact-output:/workspace/benchmarks/out" \
  -v "$PWD/.env:/workspace/.env:ro" \
  proposition7-benchmark-sas26_reproduction:latest
```

The image does not contain secrets. The mounted `.env` must define
`OPENROUTER_API_KEY` for OpenRouter rows. Reviewers can validate the baked job
without a GPU or API key using:

```bash
docker run --rm proposition7-benchmark-sas26_reproduction:latest job --dry-run
```

The manifest records the image name, baked config path, config SHA-256,
`uv.lock` SHA-256, Dockerfile SHA-256, git commit, base image, dry-run log path,
and paths to the captured git status and source diff.

Force SIF export and fail if Apptainer/Singularity is missing:

```bash
make sif
```

Run the full paper config on a GPU host:

```bash
make paper
```

The container writes `raw.jsonl` and `results.json` under the mounted output directory.

Run the preferred deployment split at the same time in separate containers. Use
Vast/GPU for constrained local generation:

```bash
docker run --rm --gpus all \
  -v "$PWD/artifact-output:/workspace/benchmarks/out" \
  proposition7-benchmark-artifact:latest deploy --resume
```

### Vast.ai deployment (artifact image)

Deploy on Vast.ai using the pre-built artifact image:

```bash
# Build the artifact image (if not already built)
make artifact

# Deploy to Vast.ai (uses artifact image by default)
./scripts/run_on_vast.sh benchmarks/configs/deploy_constrained.toml

# Deploy with a different config
./scripts/run_on_vast.sh benchmarks/configs/vast_preferred.toml

# Dry run to see the plan
./scripts/run_on_vast.sh benchmarks/configs/deploy_constrained.toml --dry-run

# Disable artifact mode (clone repo and install on the instance)
./scripts/run_on_vast.sh benchmarks/configs/deploy_constrained.toml --no-artifact
```

The Vast.ai workflow loads the artifact Docker image on the instance and runs
the same container command as a local Docker run, keeping behavior aligned.

Run closed-model unconstrained baselines through OpenRouter at the same time:

```bash
docker run --rm \
  -v "$PWD/artifact-output:/workspace/benchmarks/out" \
  -v "$PWD/.env:/workspace/.env:ro" \
  proposition7-benchmark-artifact:latest openrouter --resume
```

The mounted `.env` must define `OPENROUTER_API_KEY`.

The two commands write separate run directories, so OpenRouter API work can run
while the GPU is busy with constrained decoding. Inside one container, the
runner only uses concurrency for multiple runs of one model at a time.

## Apptainer / Singularity

When `apptainer` or `singularity` is installed on the build host, or when `nix` is available to bootstrap `apptainer`, the artifact workflow also writes:

```text
dist/proposition7-benchmark-artifact.sif
```

Run the paper suite on a GPU host:

```bash
make sif-paper
```

The SIF is built from the Docker archive output using `apptainer build ... docker-archive:<tar>` so the Docker and Singularity artifacts stay aligned.

If the server does not already provide `apptainer` or `singularity`, both scripts fall back to:

```bash
nix shell nixpkgs#apptainer
```

so long as `nix` is installed.

The wrapper uses server-friendly defaults for Singularity-style environments:

- `--cleanenv`
- `--containall`
- `--writable-tmpfs`
- `--nv`

This keeps the root filesystem ephemeral while persisting benchmark outputs through the bound `artifact-output/` directory.

## Modal A10G

Modal supports custom containers via `modal.Image.from_dockerfile(...)` and runtime
container execution via `modal.Sandbox.create(...)`.

Run the full small-Qwen benchmark suite on an A10G:

```bash
pip install -e ".[modal]"
modal setup
make modal-full
```

The launcher copies `raw.jsonl` and `results.json` back to `dist/modal-qwen-full/`.
