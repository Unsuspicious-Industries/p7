"""Tests for benchmark infrastructure that do not require real models."""

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import proposition7
import pytest

import benchmarks.run as bench_run

from benchmarks.api import (
    BenchmarkTask,
    _load_task_file,
    build_prompt,
    build_token_log,
    check_parse,
    classify,
    extract_program_output,
    FatalBenchmarkInvariantError,
    grammar_name,
    load_tasks,
    stable_hash,
)
from benchmarks.agg import dedupe_rows, delta_rows
from benchmarks.oracles import check_resolution, stlc_type_of
from benchmarks.providers import OpenRouterModel
from benchmarks.run import (
    Job,
    group_jobs_by_model,
    parse_model_concurrency,
    run_model_concurrency,
    normalize_modes,
    split_model_jobs,
    read_existing_record_keys,
)
from benchmarks.utils import (
    model_param_billions,
    auto_model_concurrency,
    hf_model_cache_name,
    clean_hf_model_cache,
    gpu_vram_gib,
)


def make_task(**overrides):
    resolution = overrides.pop("resolution", {"mode": "exact"})
    data = {
        "task_id": "toy_trace",
        "grammar": "toy",
        "category": "toy:trace",
        "prompt": "Generate one Fizz.",
        "initial": "",
        "expected": "beep:Fizz",
        "max_tokens": 8,
        "resolution": resolution,
    }
    data.update(overrides)
    payload = {
        k: data[k]
        for k in [
            "task_id",
            "grammar",
            "category",
            "prompt",
            "initial",
            "expected",
            "max_tokens",
            "resolution",
        ]
    }
    return BenchmarkTask(
        **data,
        task_hash=stable_hash(payload),
        resolution_hash=stable_hash(resolution),
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_build_prompt_uses_pure_task_text_and_grammar_context():
    prompt = build_prompt(
        "stlc",
        "Complete the term.",
        mode="constrained_direct",
        initial="λx:Int.",
    )

    assert "Task: Complete the term." in prompt
    assert "Write the completed program directly" in prompt
    assert "write the program continuation immediately" not in prompt
    assert "Decoder prefix already present:\nλx:Int." in prompt
    assert "Language summary:" in prompt
    assert "λx:T.body" in prompt
    assert "Expression ::= AtomicExpression" not in prompt


def test_mixed_prompt_does_not_put_initial_before_thinking():
    prompt = build_prompt(
        "fun",
        "Write a square function.",
        mode="constrained_mixed",
        initial="let square: Int -> Int =",
    )

    assert "Workflow: think briefly" in prompt
    assert "write the final program" in prompt.lower()
    assert "let square: Int -> Int =" not in prompt


def test_unconstrained_thinking_prompt_does_not_put_initial_before_thinking():
    prompt = build_prompt(
        "fun",
        "Write a square function.",
        mode="unconstrained_thinking",
        initial="let square: Int -> Int =",
    )

    assert "Workflow: think briefly" in prompt
    assert "write the final program" in prompt.lower()
    assert "let square: Int -> Int =" not in prompt


def test_grammar_summaries_are_compact_for_benchmark_prompts():
    for name in proposition7.list_grammars():
        summary = proposition7.get_grammar_summary(name)
        assert summary.strip(), name
        assert len(summary.split()) <= 256, name


# ---------------------------------------------------------------------------
# Task loading and resolution
# ---------------------------------------------------------------------------


def test_toml_tasks_have_quality_language_mix():
    rows = load_tasks(["all"])
    counts = Counter(row.language for row in rows)

    assert len(rows) >= 75
    assert counts["stlc"] >= 20
    assert counts["fun"] >= 20
    assert counts["imp"] >= 20


def test_toml_tasks_have_resolution_for_every_task():
    rows = load_tasks(["all"])

    assert rows
    assert all(row.resolution.get("mode") for row in rows)
    assert all(
        row.resolution.get("mode") == "equivalence"
        for row in rows
        if row.language == "stlc"
    )
    assert all(
        row.resolution.get("mode") in {"equivalence", "samples"}
        for row in rows
        if row.language == "fun"
    )
    assert all(
        row.resolution.get("mode") in {"env", "samples"}
        for row in rows
        if row.language == "imp"
    )


def test_fun_resolution_mode_equivalence():
    task = make_task(grammar="fun", expected="1", resolution={"mode": "equivalence"})

    assert check_resolution(task, "1").ok
    assert check_resolution(task, "0 + 1").ok
    assert not check_resolution(task, "2").ok

    task_samples = make_task(
        grammar="fun",
        expected="let f: Int -> Int = (x: Int) => x; 0",
        resolution={"mode": "samples", "samples": [{"x": 5, "v": 5}]},
    )
    assert check_resolution(task_samples, "let f: Int -> Int = (x: Int) => x; 0").ok
    assert not check_resolution(
        task_samples, "let f: Int -> Int = (x: Int) => x + 1; 0"
    ).ok


def test_toml_expected_outputs_parse_and_pass_resolution():
    for row in load_tasks(["all"]):
        parse_ok, complete, error = check_parse(
            proposition7.get_grammar(grammar_name(row.grammar)), row.expected
        )

        assert parse_ok, (row.task_id, error)
        assert complete, row.task_id
        assert check_resolution(row, row.expected).ok, row.task_id


def test_unconstrained_output_extraction_strips_markdown_and_prose():
    spec = proposition7.get_grammar("stlc")
    output, extracted = extract_program_output(
        spec,
        "stlc",
        "Here is the term:\n```stlc\nλx:Int.x\n```\nThat is the answer.",
    )

    assert output == "λx:Int.x"
    assert extracted is True


def test_unconstrained_output_extraction_preserves_initial_prefix():
    spec = proposition7.get_grammar("stlc")
    initial = "λf:(Int->Int)."
    output, extracted = extract_program_output(
        spec,
        "stlc",
        initial + "The program is λx:Int.(f x)",
        initial=initial,
    )

    assert output == "λf:(Int->Int).λx:Int.(f x)"
    assert extracted is True
    assert check_parse(spec, output)[:2] == (True, True)


def test_stlc_resolution_types_are_inferred_correctly():
    task = next(
        row for row in load_tasks(["all"]) if row.task_id == "stlc_apply_twice_int"
    )

    assert task.resolution["type"] == "(Int -> Int) -> Int -> Int"
    assert stlc_type_of(task.expected) == task.resolution["type"]


def test_fun_samples_hidden_cases_catch_sample_overfitting():
    task = make_task(
        grammar="fun",
        initial="let double: Int -> Int =",
        expected="let noise: Int = 99; let double: Int -> Int = (x: Int) => x + x; 0",
        resolution={
            "mode": "samples",
            "fn": "double",
            "hidden_samples": 24,
            "samples": [
                {"x": 7, "v": 14},
                {"x": 0, "v": 0},
                {"x": -3, "v": -6},
            ],
        },
    )

    cheating = "let double: Int -> Int = (x: Int) => x * (x + 3) * (x - 7) + x + x; 0"
    honest = "let helper: Int = 1; let double: Int -> Int = (x: Int) => x + x; 0"

    cheat_result = check_resolution(task, cheating)
    honest_result = check_resolution(task, honest)

    assert not cheat_result.ok
    assert cheat_result.reason.startswith("hidden_sample_fail:")
    assert honest_result.ok


def test_fun_equivalence_on_function_values_uses_alpha_equivalence():
    task = make_task(
        grammar="fun",
        initial="let compose: (Int -> Int) -> (Int -> Int) -> Int -> Int =",
        expected=(
            "let compose: (Int -> Int) -> (Int -> Int) -> Int -> Int = "
            "(f: Int -> Int) => (g: Int -> Int) => (x: Int) => f(g(x)); compose"
        ),
        resolution={
            "mode": "equivalence",
            "fn": "compose",
            "structure": {"let_bindings": 1, "applications": 2, "lambdas": 3},
        },
    )

    renamed = (
        "let compose: (Int -> Int) -> (Int -> Int) -> Int -> Int = "
        "(u: Int -> Int) => (v: Int -> Int) => (n: Int) => u(v(n)); compose"
    )
    wrong = (
        "let compose: (Int -> Int) -> (Int -> Int) -> Int -> Int = "
        "(f: Int -> Int) => (g: Int -> Int) => (x: Int) => g(f(x)); compose"
    )

    assert check_resolution(task, renamed).ok
    assert not check_resolution(task, wrong).ok


def test_imp_samples_hidden_cases_catch_constant_cheat():
    task = make_task(
        grammar="imp",
        initial="{ let a: Int =",
        expected="{ let a: Int = 19; let b: Int = 23; let total: Int = a + b; }",
        resolution={
            "mode": "samples",
            "input_vars": ["a", "b"],
            "vars": ["total"],
            "hidden_samples": 20,
            "samples": [
                {"a": 19, "b": 23, "total": 42},
                {"a": 1, "b": 2, "total": 3},
                {"a": -5, "b": 8, "total": 3},
            ],
        },
    )

    cheating = "{ let a: Int = 19; let b: Int = 23; let total: Int = 42; }"
    honest = "{ let a: Int = 19; let b: Int = 23; let total: Int = a + b; }"

    assert not check_resolution(task, cheating).ok
    assert check_resolution(task, honest).ok


def test_task_loader_rejects_invalid_fun_samples_schema(tmp_path):
    bad_task = tmp_path / "bad_fun.toml"
    bad_task.write_text(
        """
id = "bad_fun"
grammar = "fun"
category = "fun:test"
max_tokens = 32

[prompt]
text = "bad"

[initial]
text = "let f: Int -> Int ="

[expected]
text = "let f: Int -> Int = (x: Int) => x; 0"

[resolution]
mode = "samples"
samples = [{x = 1}]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid samples spec"):
        _load_task_file(bad_task)


def test_task_loader_rejects_invalid_imp_samples_schema(tmp_path):
    bad_task = tmp_path / "bad_imp.toml"
    bad_task.write_text(
        """
id = "bad_imp"
grammar = "imp"
category = "imp:test"
max_tokens = 32

[prompt]
text = "bad"

[initial]
text = "{ let x: Int ="

[expected]
text = "{ let x: Int = 1; let y: Int = x + 1; }"

[resolution]
mode = "samples"
input_vars = ["x"]
vars = ["y"]
samples = [{x = 1}]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid imp samples spec"):
        _load_task_file(bad_task)


def test_fun_tasks_no_longer_use_zero_sentinel_tails():
    for row in load_tasks(["fun"]):
        assert "; 0" not in row.expected, row.task_id


# ---------------------------------------------------------------------------
# Token log and classification
# ---------------------------------------------------------------------------


def test_build_token_log_records_cumulative_prefixes():
    task = make_task()
    record = {
        "output": "beep:Fizz",
        "error": "ok",
        "passed": True,
        "parse_ok": True,
        "parse_complete": True,
        "parse_error": "",
        "stop_reason": "complete",
        "seed": 7,
        "tokens": 2,
        "seconds": 0.01,
    }

    log = build_token_log(
        task,
        "constrained_direct",
        "prompt",
        proposition7.get_grammar("toy"),
        [{"step": 0, "token": "beep"}, {"step": 1, "token": ":Fizz"}],
        record,
    )

    assert log["tokens_generated"] == 2
    assert [row["text_after"] for row in log["tokens"]] == ["beep", "beep:Fizz"]
    assert log["tokens"][-1]["parse_complete"] is True


def test_classify_returns_non_completable_for_dead_prefixes():
    assert (
        classify(
            False,
            False,
            False,
            "Parse error: no parse found at input length 7",
            None,
        )
        == "non_completable"
    )


# ---------------------------------------------------------------------------
# Resume and job keys
# ---------------------------------------------------------------------------


def test_resume_keys_include_backend_task_and_resolution_hashes(tmp_path):
    raw = tmp_path / "raw.jsonl"
    rows = [
        {
            "backend": "local",
            "model": "gpt2",
            "task_id": "t1",
            "task_hash": "h1",
            "resolution_hash": "r1",
            "mode": "constrained_direct",
            "try": 0,
        },
        {
            "backend": "openrouter",
            "model": "gpt2",
            "task_id": "t1",
            "task_hash": "h1",
            "resolution_hash": "r1",
            "mode": "constrained_direct",
            "try": 0,
        },
    ]
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    task = SimpleNamespace(task_id="t1", task_hash="h1", resolution_hash="r1")
    changed_task = SimpleNamespace(task_id="t1", task_hash="h2", resolution_hash="r1")

    assert Job("gpt2", "toy", task, "constrained_direct", 0).key(
        "local"
    ) in read_existing_record_keys(raw, "local")
    assert Job("gpt2", "toy", task, "constrained_direct", 0).key(
        "local"
    ) not in read_existing_record_keys(raw, "openrouter")
    assert Job("gpt2", "toy", changed_task, "constrained_direct", 0).key(
        "local"
    ) not in read_existing_record_keys(raw, "local")


# ---------------------------------------------------------------------------
# Model concurrency and sizing
# ---------------------------------------------------------------------------


def test_model_concurrency_setting_parses_manual_and_auto_values():
    assert parse_model_concurrency("auto") == "auto"
    assert parse_model_concurrency("4") == 4
    with pytest.raises(SystemExit):
        parse_model_concurrency("0")
    with pytest.raises(SystemExit):
        parse_model_concurrency("many")


def test_model_size_parsing_supports_common_hf_names():
    assert model_param_billions("gpt2") == pytest.approx(0.124)
    assert model_param_billions("EleutherAI/pythia-410m") == pytest.approx(0.410)
    assert model_param_billions("Qwen/Qwen3.5-0.8B") == pytest.approx(0.8)
    assert model_param_billions("google/gemma-4-E4B-it") == pytest.approx(4.0)


def test_auto_model_concurrency_scales_with_model_size(monkeypatch):
    args = SimpleNamespace(device="cuda", torch_dtype="auto")
    monkeypatch.setattr("benchmarks.utils.gpu_vram_gib", lambda _args: 24.0)

    assert auto_model_concurrency(args, "gpt2", 20) == 20
    assert auto_model_concurrency(args, "google/gemma-4-E4B-it", 10) == 1
    assert auto_model_concurrency(args, "Qwen/Qwen3.5-9B", 10) == 1


def test_split_model_jobs_distributes_work_without_empty_chunks():
    jobs = list(range(5))

    assert split_model_jobs(jobs, 2) == [[0, 2, 4], [1, 3]]
    assert split_model_jobs(jobs, 10) == [[0], [1], [2], [3], [4]]


def test_parallel_jobs_are_grouped_by_model_before_chunking():
    task = SimpleNamespace(
        task_id="t",
        task_hash="h",
        resolution_hash="r",
        grammar="toy",
        language="toy",
    )
    jobs = [
        Job("m1", "toy", task, "unconstrained", 0),
        Job("m2", "toy", task, "unconstrained", 0),
        Job("m1", "toy", task, "constrained_direct", 0),
        Job("m2", "toy", task, "constrained_direct", 0),
    ]

    groups = group_jobs_by_model(jobs)

    assert [model for model, _ in groups] == ["m1", "m2"]
    assert [[job.model_name for job in group] for _, group in groups] == [
        ["m1", "m1"],
        ["m2", "m2"],
    ]


# ---------------------------------------------------------------------------
# Mode validation
# ---------------------------------------------------------------------------


def test_normalize_modes_accepts_raw_and_cleaned_unconstrained_modes():
    local = normalize_modes(
        [
            "constrained_direct",
            "outlines",
            "outlines_mixed",
            "unconstrained",
            "unconstrained_cleaned",
            "unconstrained_thinking",
        ],
        backend="local",
    )
    remote = normalize_modes(
        ["unconstrained", "unconstrained_cleaned", "unconstrained_thinking"],
        backend="openrouter",
    )

    assert local == [
        "constrained_direct",
        "outlines",
        "outlines_mixed",
        "unconstrained",
        "unconstrained_cleaned",
        "unconstrained_thinking",
    ]
    assert remote == [
        "unconstrained",
        "unconstrained_cleaned",
        "unconstrained_thinking",
    ]


def test_normalize_modes_rejects_removed_aliases():
    with pytest.raises(SystemExit):
        normalize_modes(["constrained"], backend="local")
    with pytest.raises(SystemExit):
        normalize_modes(["unconstrained_raw"], backend="local")
    with pytest.raises(SystemExit):
        normalize_modes(["constrained_direct"], backend="openrouter")
    with pytest.raises(SystemExit):
        normalize_modes(["outlines_mixed"], backend="openrouter")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregation_dedupes_by_hash_aware_benchmark_job_key():
    rows = [
        {
            "backend": "local",
            "model": "gpt2",
            "task_id": "t1",
            "task_hash": "h1",
            "resolution_hash": "r1",
            "mode": "constrained_direct",
            "try": 0,
            "output": "old",
        },
        {
            "backend": "local",
            "model": "gpt2",
            "task_id": "t1",
            "task_hash": "h1",
            "resolution_hash": "r1",
            "mode": "constrained_direct",
            "try": 0,
            "output": "new",
        },
        {
            "backend": "local",
            "model": "gpt2",
            "task_id": "t1",
            "task_hash": "h2",
            "resolution_hash": "r1",
            "mode": "constrained_direct",
            "try": 0,
            "output": "changed",
        },
    ]

    deduped = dedupe_rows(rows)

    assert len(deduped) == 2
    assert deduped[0]["output"] == "new"
    assert deduped[1]["output"] == "changed"


def test_delta_rows_prefers_raw_unconstrained_when_available():
    summary = [
        {
            "backend": "local",
            "model": "m",
            "language": "toy",
            "mode": "constrained_direct",
            "exact_rate": 80.0,
            "pass_rate": 70.0,
            "parse_error_rate": 0.0,
            "non_completable_rate": 0.0,
            "avg_tokens": 10.0,
        },
        {
            "backend": "local",
            "model": "m",
            "language": "toy",
            "mode": "unconstrained_cleaned",
            "exact_rate": 60.0,
            "pass_rate": 55.0,
            "parse_error_rate": 20.0,
            "non_completable_rate": 5.0,
            "avg_tokens": 15.0,
        },
        {
            "backend": "local",
            "model": "m",
            "language": "toy",
            "mode": "unconstrained",
            "exact_rate": 40.0,
            "pass_rate": 35.0,
            "parse_error_rate": 45.0,
            "non_completable_rate": 10.0,
            "avg_tokens": 20.0,
        },
    ]

    [row] = delta_rows(summary, ("backend", "model", "language"))

    assert row["unconstrained_mode"] == "unconstrained"
    assert row["exact_delta"] == 40.0
    assert row["parse_error_delta"] == 45.0


def test_aggregator_handles_missing_input_and_tracks_timeout_rates(tmp_path):
    from benchmarks.agg import load_rows, summarize

    missing = tmp_path / "missing.jsonl"
    assert load_rows(missing) == []

    rows = [
        {
            "backend": "local",
            "model": "gpt2",
            "mode": "constrained_direct",
            "language": "fun",
            "error": "timeout",
            "exact": False,
            "tokens": 0,
            "seconds": 10.0,
        },
        {
            "backend": "local",
            "model": "gpt2",
            "mode": "constrained_direct",
            "language": "fun",
            "error": "ok",
            "exact": True,
            "tokens": 5,
            "seconds": 1.0,
        },
        {
            "backend": "local",
            "model": "gpt2",
            "mode": "constrained_direct",
            "language": "fun",
            "error": "model_error",
            "exact": False,
            "tokens": 0,
            "seconds": 2.0,
        },
    ]

    summary = summarize(rows, ("backend", "model", "mode", "language"))
    assert len(summary) == 1
    row = summary[0]
    assert row["timeout_rate"] == round(100.0 / 3.0, 2)
    assert row["other_error_rate"] == round(100.0 / 3.0, 2)


# ---------------------------------------------------------------------------
# Benchmark config parsing
# ---------------------------------------------------------------------------


def test_benchmark_config_parses_toml_matrix(tmp_path):
    config_path = tmp_path / "benchmark.toml"
    config_path.write_text(
        """
schema_version = 1

[run]
name = "example"
output_root = "benchmarks/out"

[tasks]
selectors = ["stlc"]
max_tasks = 1

[execution]
tries = 1
model_concurrency = "auto"
low_space = true

[local]
device = "cpu"
torch_dtype = "none"
device_map = ""

[local.model_kwargs]

[[matrix]]
name = "local-example"
backend = "local"
models = ["gpt2"]
modes = ["unconstrained"]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = bench_run.load_benchmark_config(config_path)

    assert config.run_name == "example"
    assert config.model_concurrency == "auto"
    assert config.low_space is True
    assert config.matrices[0].name == "local-example"
    assert config.matrices[0].models == ["gpt2"]


def test_resolve_run_paths_avoids_overwriting_existing_run_dir(tmp_path):
    config = SimpleNamespace(output_root=tmp_path, run_name="artifact")

    first = bench_run.resolve_run_paths(config, resume=False, explicit_run_dir="")
    first.run_dir.mkdir(parents=True)
    second = bench_run.resolve_run_paths(config, resume=False, explicit_run_dir="")
    resumed = bench_run.resolve_run_paths(config, resume=True, explicit_run_dir="")

    assert first.run_dir == tmp_path / "artifact"
    assert second.run_dir != first.run_dir
    assert resumed.run_dir == first.run_dir


def test_benchmark_config_rejects_legacy_parallel_tasks_key(tmp_path):
    config_path = tmp_path / "benchmark.toml"
    config_path.write_text(
        """
schema_version = 1

[run]
name = "example"

[tasks]
selectors = ["stlc"]

[execution]
parallel_tasks = "auto"

[local]
device = "cpu"
torch_dtype = "none"
device_map = ""

[local.model_kwargs]

[[matrix]]
name = "local-example"
backend = "local"
models = ["gpt2"]
modes = ["unconstrained"]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="execution\\.parallel_tasks was renamed"):
        bench_run.load_benchmark_config(config_path)


def test_run_script_dry_run_accepts_config(tmp_path):
    import subprocess
    import sys

    config_path = tmp_path / "benchmark.toml"
    config_path.write_text(
        """
schema_version = 1

[run]
name = "example"
output_root = "benchmarks/out"

[tasks]
selectors = ["stlc"]
max_tasks = 1

[execution]
tries = 1
model_concurrency = "auto"

[local]
device = "cpu"
torch_dtype = "none"
device_map = ""

[local.model_kwargs]

[[matrix]]
name = "local-example"
backend = "local"
models = ["gpt2"]
modes = ["unconstrained"]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/run.py",
            "--config",
            str(config_path),
            "--dry-run",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "[plan]" in result.stdout
    assert "pending_jobs=1" in result.stdout


# ---------------------------------------------------------------------------
# Config validation against paper
# ---------------------------------------------------------------------------


def test_sas26_reproduction_config_matches_pdf_mode_split():
    config = bench_run.load_benchmark_config(
        Path("benchmarks/configs/sas26_reproduction.toml")
    )
    matrices = {matrix.name: matrix for matrix in config.matrices}

    assert config.run_name == "sas26-reproduction"
    assert matrices["fig3-core-local"].modes == [
        "constrained_direct",
        "constrained_mixed",
        "unconstrained",
    ]
    assert matrices["fig7-frontier-constrained"].modes == [
        "constrained_mixed"
    ]
    assert matrices["fig7-openrouter-raw"].modes == ["unconstrained"]
    assert "openai/gpt-5.4-mini" in matrices["fig7-openrouter-raw"].models
    assert all(
        "unconstrained_cleaned" not in matrix.modes
        and "outlines" not in matrix.modes
        and "outlines_mixed" not in matrix.modes
        for matrix in config.matrices
    )


# ---------------------------------------------------------------------------
# OpenRouter provider
# ---------------------------------------------------------------------------


def test_openrouter_unconstrained_payload_preserves_prefix_contract(monkeypatch):
    model = OpenRouterModel("closed/model", api_key="test-key", env_path=None)
    payloads = []

    def fake_chat(payload):
        payloads.append(payload)
        return "Fizz", 3, "stop"

    monkeypatch.setattr(model, "_chat", fake_chat)

    result = model.generate_unconstrained(
        "Prompt text",
        initial="beep:",
        max_tokens=12,
        temperature=0.8,
        seed=11,
    )

    assert result.text == "beep:Fizz"
    assert result.tokens_generated == 3
    assert result.stopped_reason == "stop"
    assert payloads == [
        {
            "model": "closed/model",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Prompt text\n\n"
                        "Continue this exact prefix. Return only the completed program text, including the prefix.\n"
                        "Prefix:\nbeep:"
                    ),
                }
            ],
            "max_tokens": 12,
            "temperature": 0.8,
            "seed": 11,
        }
    ]


def test_make_model_uses_openrouter_adapter_without_local_model_load(
    tmp_path, monkeypatch
):
    env_path = tmp_path / ".env"
    env_path.write_text("OPENROUTER_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def fail_local_load(*args, **kwargs):
        del args, kwargs
        raise AssertionError("OpenRouter mode should not load a local HF model")

    monkeypatch.setattr(bench_run.proposition7, "get_model_class", fail_local_load)
    args = SimpleNamespace(
        backend="openrouter",
        device="cpu",
        torch_dtype="none",
        device_map="",
        model_kwargs={},
        model_kwargs_json="",
        openrouter_env=str(env_path),
    )

    model = bench_run.make_model(args, "openai/example", "toy", "unconstrained")

    assert isinstance(model, OpenRouterModel)
    assert model.model_name == "openai/example"
    assert model.api_key == "from-file"


def test_make_model_uses_outlines_wrapper_for_outlines_modes(monkeypatch):
    calls = []

    class FakeOutlinesModel:
        def __init__(self, model_name, **kwargs):
            calls.append((model_name, kwargs))

    monkeypatch.setattr(bench_run, "OutlinesSyntaxModel", FakeOutlinesModel)
    args = SimpleNamespace(
        backend="local",
        device="cpu",
        torch_dtype="none",
        device_map="",
        model_kwargs={"local_files_only": True},
        model_kwargs_json="",
    )

    bench_run.make_model(args, "gpt2", "toy", "outlines")
    bench_run.make_model(args, "gpt2", "toy", "outlines_mixed")

    assert len(calls) == 2
    assert calls[0][0] == "gpt2"
    assert calls[0][1]["grammar_name"] == "toy"
    assert calls[0][1]["device"] == "cpu"
    assert calls[0][1]["local_files_only"] is True
