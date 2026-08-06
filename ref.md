# Progress on aufbau-rs + p7 pruning study

## What we did

### aufbau-rs Python Package
- **Upgraded PyO3 0.23 → 0.29** in `../aufbau/Cargo.toml` for Python 3.13 compatibility
- **Bumped version 0.2.0 → 0.2.1**
- **Added CI skip for `build.rs` lc.sh** (`../aufbau/build.rs`) when `CI`, `MATURIN_PYPI_TOKEN`, or `PYO3_PYPI_TOKEN` env vars are set
- **Set up GitHub Actions CI** (`../aufbau/.github/workflows/publish-python.yml`) to build wheels for Python 3.9–3.13 on both x86_64 and aarch64

### CI Learnings (in `ref.md` and applied to workflow)
- `actions/download-artifact@v4` corrupts wheel files — upload directly from each build job
- `maturin-action@v1` ignores `python-version` input — use `python -m maturin build` directly
- `--compatibility manylinux2014` fails on Ubuntu 24.04 — use `--compatibility pypi` or `--compatibility manylinux_2_34`
- `maturin build --interpreter cp3.12` fails — dots must be stripped (`cp312`)
- twine 6.x rejects `license-file` — use `pipx run twine` (latest)

### Published to PyPI
- 11 wheels for 0.2.1 (cp39–cp313 × x86_64 + aarch64)
- x86_64 wheels work fine (verified: `from_grammar` works)

## BLOCKING ISSUE: aarch64 wheels missing methods

The aarch64 `.so` file (`aufbau.cpython-313-aarch64-linux-gnu.so`) is **missing** these PySynthesizer methods:
- `from_grammar` (staticmethod)
- `mask` (instance, &mut self)
- `status` (instance, &mut self)
- `root_type` (instance, &mut self)
- `in_scope` (instance, &self)

Only present methods: `__new__`, `ast`, `feed`, `input`, `parse`, `grammar`, `get_rule`, `try_feed`, `clear_ctx`, `set_input`, `add_to_ctx`, `is_complete`, `ast_str`

## what i think is happening
rust 1.96.0 on the arm runner is too new and pyo3 macros generate silently wrong code. the existing methods are not exactly simple — `add_to_ctx` uses `Type::parse` which is complex — so it's not a "complex types" issue. the pattern of the missing methods is: they all (except `from_grammar`) take `&mut self` and return non-`PyResult`, but so do existing methods like `is_complete`. there's no obvious arch-specific cfg or feature flag difference. the x86_64 runner uses a different rust version.

## what needs to be done

### immediate: get working aarch64 aufbau-rs wheel

**option A: build aufbau on pgx1 directly** (tried, couldn't ssh)
- ssh into pgx1, clone aufbau, `maturin develop` or `pip install .`
- then run pruning study

**option B: fix CI and publish 0.2.2**
- need: add `--interpreter` flag to `maturin build` in CI
- need: remove `target-version = "3.11"` from `pyproject.toml` or keep and use `abi3` correctly
- need: disable rust cache for aarch64 builds (or per-python-version cache keys)
- then trigger workflow_dispatch, download wheel, verify

**option C: install aufbau from source on pgx1 manually**
- user does it themselves: `pip install git+https://github.com/Unsuspicious-Industries/aufbau`

### after wheel works
1. On pgx1: `pip install --no-cache-dir --force-reinstall aufbau-rs`
2. Run: `python benchmarks/run.py --config pruning-study/config.toml --resume`
3. Check results in `pruning-study/raw.jsonl`
