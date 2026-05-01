from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import p7
import pytest
import benchmarks.run as bench_run
from benchmarks.models import LOCAL_ASCENDING_MODELS

from benchmarks.api import (
    BenchmarkTask,
    _load_task_file,
    build_prompt,
    build_token_log,
    check_parse,
    extract_program_output,
    grammar_name,
    load_tasks,
    stable_hash,
)
from benchmarks.agg import dedupe_rows, delta_rows
from benchmarks.oracles import check_resolution, stlc_type_of
from benchmarks.providers import OUTLINES_LARK
from benchmarks.run import (
    Job,
    auto_parallel_tasks,
    clean_hf_model_cache,
    group_jobs_by_model,
    hf_model_cache_name,
    model_param_billions,
    parse_parallel_tasks,
    run_parallel_by_model,
    selected_modes,
    split_jobs,
    read_existing_record_keys,
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


def test_build_prompt_uses_pure_task_text_and_grammar_context():
    prompt = build_prompt(
        "stlc",
        "Complete the term.",
        mode="constrained_direct",
        initial="λx:Int.",
    )

    assert "Task: Complete the term." in prompt
    assert "Direct constrained generation" in prompt
    assert "write the program continuation immediately" in prompt
    assert "the next generated text should be the value" in prompt
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
    assert "formal block" in prompt
    assert "let square: Int -> Int =" not in prompt


def test_grammar_summaries_are_compact_for_benchmark_prompts():
    for name in p7.list_grammars():
        summary = p7.get_grammar_summary(name)
        assert summary.strip(), name
        assert len(summary.split()) <= 256, name


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


def test_fun_multiply_sum_structure_assertion():
    task = next(row for row in load_tasks(["all"]) if row.task_id == "fun_multiply_sum")

    assert check_resolution(task, task.expected).ok
    assert not check_resolution(task, "39").ok
    assert not check_resolution(task, "let x: Int = 6; let y: Int = 7; x + y * 3").ok


def test_toml_expected_outputs_parse_and_pass_resolution():
    for row in load_tasks(["all"]):
        parse_ok, complete, error = check_parse(
            p7.get_grammar(grammar_name(row.grammar)), row.expected
        )

        assert parse_ok, (row.task_id, error)
        assert complete, row.task_id
        assert check_resolution(row, row.expected).ok, row.task_id


def test_unconstrained_output_extraction_strips_markdown_and_prose():
    spec = p7.get_grammar("stlc")
    output, extracted = extract_program_output(
        spec,
        "stlc",
        "Here is the term:\n```stlc\nλx:Int.x\n```\nThat is the answer.",
    )

    assert output == "λx:Int.x"
    assert extracted is True


def test_unconstrained_output_extraction_preserves_initial_prefix():
    spec = p7.get_grammar("stlc")
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
        p7.get_grammar("toy"),
        [{"step": 0, "token": "beep"}, {"step": 1, "token": ":Fizz"}],
        record,
    )

    assert log["tokens_generated"] == 2
    assert [row["text_after"] for row in log["tokens"]] == ["beep", "beep:Fizz"]
    assert log["tokens"][-1]["parse_complete"] is True


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


def test_low_space_cache_cleanup_keeps_requested_hf_model(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    keep = hub / hf_model_cache_name("org/keep")
    drop = hub / hf_model_cache_name("org/drop")
    keep.mkdir(parents=True)
    drop.mkdir(parents=True)
    (keep / "config.json").write_text("{}", encoding="utf-8")
    (drop / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))

    clean_hf_model_cache("org/keep")

    assert keep.exists()
    assert not drop.exists()


def test_release_cached_models_clears_cache_without_forcing_gc(monkeypatch):
    cache = {("local", "m"): object()}
    empty_cache_calls = []

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def empty_cache():
            empty_cache_calls.append(True)

    monkeypatch.setitem(__import__("sys").modules, "torch", SimpleNamespace(cuda=FakeCuda))

    bench_run.release_cached_models(cache)

    assert cache == {}
    assert empty_cache_calls == [True]


def test_parallel_task_setting_parses_manual_and_auto_values():
    assert parse_parallel_tasks("auto") == "auto"
    assert parse_parallel_tasks("4") == 4
    with pytest.raises(SystemExit):
        parse_parallel_tasks("0")
    with pytest.raises(SystemExit):
        parse_parallel_tasks("many")


def test_model_size_parsing_supports_common_hf_names():
    assert model_param_billions("gpt2") == pytest.approx(0.124)
    assert model_param_billions("EleutherAI/pythia-410m") == pytest.approx(0.410)
    assert model_param_billions("Qwen/Qwen3.5-0.8B") == pytest.approx(0.8)
    assert model_param_billions("google/gemma-4-E4B-it") == pytest.approx(4.0)


def test_auto_parallel_tasks_scales_with_model_size(monkeypatch):
    args = SimpleNamespace(device="cuda", torch_dtype="auto")
    monkeypatch.setattr(bench_run, "gpu_vram_gib", lambda _args: 24.0)

    assert auto_parallel_tasks(args, "gpt2", 20) == 20
    assert auto_parallel_tasks(args, "google/gemma-4-E4B-it", 10) == 1
    assert auto_parallel_tasks(args, "Qwen/Qwen3.5-9B", 10) == 1


def test_default_model_matrix_excludes_gated_llama_and_keeps_current_open_models(monkeypatch):
    args = SimpleNamespace(device="cuda", torch_dtype="auto")
    monkeypatch.setattr(bench_run, "gpu_vram_gib", lambda _args: 48.0)

    assert LOCAL_ASCENDING_MODELS == [
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.5-9B-Base",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-4B-Base",
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-1.5B",
        "Qwen/Qwen3.5-0.8B",
        "Ministral-3-8B-Instruct-2512-GGUF",
        "microsoft/Phi-4-mini-instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    ]
    assert all("llama" not in model.lower() for model in LOCAL_ASCENDING_MODELS)
    assert auto_parallel_tasks(args, "Qwen/Qwen3.5-9B", 32) >= 2


def test_selected_modes_accepts_unconstrained_raw_for_local_and_openrouter():
    local = selected_modes(
        SimpleNamespace(
            modes="constrained,unconstrained_raw,unconstrained",
            backend="local",
        )
    )
    remote = selected_modes(
        SimpleNamespace(
            modes="constrained_direct,unconstrained_raw,unconstrained",
            backend="openrouter",
        )
    )

    assert local == ["constrained_direct", "unconstrained_raw", "unconstrained"]
    assert remote == ["unconstrained_raw", "unconstrained"]


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


def test_parallel_runner_does_not_mix_models_in_one_group(tmp_path, monkeypatch):
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
    calls = []

    def fake_worker(worker_id, args, chunk, *rest):
        del worker_id, args, rest
        calls.append([job.model_name for job in chunk])
        return []

    monkeypatch.setattr(bench_run, "run_worker_chunk", fake_worker)
    args = SimpleNamespace(
        parallel_tasks=2,
        timeout=0,
        low_space=False,
        backend="local",
        _test_inline_workers=True,
    )

    run_parallel_by_model(args, jobs, tmp_path / "raw.jsonl", None)

    first_m2 = next(i for i, chunk in enumerate(calls) if chunk[0] == "m2")
    assert all(set(chunk) == {"m1"} for chunk in calls[:first_m2])
    assert all(set(chunk) == {"m2"} for chunk in calls[first_m2:])


def test_parallel_low_space_cleans_once_per_model(tmp_path, monkeypatch):
    task = SimpleNamespace(
        task_id="t",
        task_hash="h",
        resolution_hash="r",
        grammar="toy",
        language="toy",
    )
    jobs = [
        Job("m1", "toy", task, "unconstrained", 0),
        Job("m1", "toy", task, "constrained_direct", 0),
        Job("m2", "toy", task, "unconstrained", 0),
    ]
    cleaned = []

    def fake_worker(*args, **kwargs):
        del args, kwargs
        return []

    monkeypatch.setattr(bench_run, "run_worker_chunk", fake_worker)
    monkeypatch.setattr(bench_run, "clean_hf_model_cache", lambda name: cleaned.append(name))
    args = SimpleNamespace(
        parallel_tasks=2,
        timeout=0,
        low_space=True,
        backend="local",
        _test_inline_workers=True,
    )

    run_parallel_by_model(args, jobs, tmp_path / "raw.jsonl", None)

    assert cleaned == ["m1", "m2"]


def test_parallel_runner_allows_timeout_with_process_workers(tmp_path, monkeypatch):
    task = SimpleNamespace(
        task_id="t",
        task_hash="h",
        resolution_hash="r",
        grammar="toy",
        language="toy",
    )
    args = SimpleNamespace(
        parallel_tasks=2,
        timeout=1,
        low_space=False,
        backend="local",
        _test_inline_workers=True,
    )
    calls = []

    def fake_worker(worker_id, args, chunk, trace_enabled):
        del worker_id, args, trace_enabled
        calls.append([job.task_id for job in [job.task for job in chunk]])
        return []

    monkeypatch.setattr(bench_run, "run_worker_chunk", fake_worker)
    run_parallel_by_model(
        args,
        [Job("m1", "toy", task, "unconstrained", 0)],
        tmp_path / "raw.jsonl",
        None,
    )

    assert calls == [["t"]]


def test_process_worker_enforces_per_job_timeout(monkeypatch):
    import time

    task = make_task(grammar="toy")
    args = SimpleNamespace(
        backend="local",
        device="cpu",
        model_kwargs_json="",
        torch_dtype="none",
        device_map="",
        seed=7,
        think_budget=1,
        timeout=1,
    )

    monkeypatch.setattr(bench_run, "make_model", lambda *args, **kwargs: object())

    def slow_interaction(*args, **kwargs):
        del args, kwargs
        time.sleep(2)
        raise AssertionError("timeout did not interrupt job")

    monkeypatch.setattr(bench_run, "run_interaction", slow_interaction)

    outputs = bench_run.run_worker_chunk(
        0,
        args,
        [Job("gpt2", "toy", task, "unconstrained", 0)],
        False,
    )

    assert len(outputs) == 1
    record, traces = outputs[0]
    assert traces == []
    assert record["error"] == "timeout"
    assert record["stop_reason"] == "timeout"


def test_split_jobs_distributes_work_without_empty_chunks():
    jobs = list(range(5))

    assert split_jobs(jobs, 2) == [[0, 2, 4], [1, 3]]
    assert split_jobs(jobs, 10) == [[0], [1], [2], [3], [4]]


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


def test_delta_rows_prefers_unconstrained_raw_when_available():
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
        },
        {
            "backend": "local",
            "model": "m",
            "language": "toy",
            "mode": "unconstrained",
            "exact_rate": 60.0,
            "pass_rate": 55.0,
            "parse_error_rate": 20.0,
            "non_completable_rate": 5.0,
        },
        {
            "backend": "local",
            "model": "m",
            "language": "toy",
            "mode": "unconstrained_raw",
            "exact_rate": 40.0,
            "pass_rate": 35.0,
            "parse_error_rate": 45.0,
            "non_completable_rate": 10.0,
        },
    ]

    [row] = delta_rows(summary, ("backend", "model", "language"))

    assert row["unconstrained_mode"] == "unconstrained_raw"
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


def test_models_script_dry_run_builds_commands(tmp_path):
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/models.py",
            "--dry-run",
            "--without-closed",
            "--low-space",
            "--parallel-tasks",
            "auto",
            "--max-tasks",
            "1",
            "--tries",
            "1",
            "--out-dir",
            str(tmp_path / "out"),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "benchmarks/run.py" in result.stdout
    assert "--low-space" in result.stdout
    assert "--parallel-tasks auto" in result.stdout
    assert "benchmarks/agg.py" in result.stdout


def test_run_script_dry_run_accepts_parallel_auto(tmp_path):
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/run.py",
            "--dry",
            "--tasks",
            "stlc",
            "--models",
            "gpt2",
            "--modes",
            "unconstrained",
            "--max-tasks",
            "1",
            "--parallel-tasks",
            "auto",
            "--out",
            str(tmp_path / "raw.jsonl"),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "parallel_tasks=auto" in result.stdout


def test_outlines_has_syntax_adapters_for_every_builtin_grammar():
    assert set(p7.list_grammars()) <= set(OUTLINES_LARK)


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


def test_fun_samples_accepts_args_form_and_odd_expected_structure():
    task = make_task(
        grammar="fun",
        initial="let f: Int -> Int =",
        expected=(
            "let unused: Int = 123; let alias: Int -> Int = (q: Int) => q + 1;"
            " let f: Int -> Int = (x: Int) => alias(x) + 1; 0"
        ),
        resolution={
            "mode": "samples",
            "fn": "f",
            "hidden_samples": 12,
            "samples": [
                {"args": [0], "v": 2},
                {"args": [10], "v": 12},
                {"args": [-3], "v": -1},
            ],
        },
    )

    assert check_resolution(task, "let f: Int -> Int = (x: Int) => x + 2; 0").ok
    assert not check_resolution(task, "let f: Int -> Int = (x: Int) => x + 1; 0").ok


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


def test_fun_equivalence_on_function_values_uses_alpha_equivalence_not_any_closure():
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


def test_fun_tasks_no_longer_use_zero_sentinel_tails():
    for row in load_tasks(["fun"]):
        assert "; 0" not in row.expected, row.task_id


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
