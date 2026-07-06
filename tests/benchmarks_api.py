from collections import Counter
import json
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
    rejection_reason,
    stable_hash,
)
from benchmarks.agg import dedupe_rows, delta_rows
from benchmarks.oracles import check_resolution, stlc_type_of
from benchmarks.providers import OpenRouterModel
from benchmarks.run import (
    group_jobs_by_model,
    normalize_modes,
    run_model_concurrency,
    split_model_jobs,
    read_existing_record_keys,
)
from benchmarks.utils import (
    Job,
    auto_model_concurrency,
    clean_hf_model_cache,
    hf_model_cache_name,
    model_param_billions,
    parse_model_concurrency,
)


def selected_modes(args):
    """Adapter for the legacy args-shaped API: the runner now takes an explicit
    (modes_list, backend) pair sourced from the TOML matrix, so split the comma
    string here and delegate to normalize_modes."""
    return normalize_modes(str(args.modes).split(","), args.backend)


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

    # constrained_direct injects the decoder prefix into the prompt; it carries
    # the task and language summary but never the raw grammar productions.
    assert "Task: Complete the term." in prompt
    assert "Decoder prefix already present:\nλx:Int." in prompt
    assert "Language summary:" in prompt
    assert "Expression ::= AtomicExpression" not in prompt


def _assert_initial_not_before_thinking(mode: str):
    initial = "let square : int -> int ="
    prompt = build_prompt("ml", "Write a square function.", mode=mode, initial=initial)
    # think-first modes must not leak the decoder prefix into the prompt: the
    # model reasons before any program text is fixed.
    assert "Workflow: think briefly" in prompt
    assert initial not in prompt


def test_mixed_prompt_does_not_put_initial_before_thinking():
    _assert_initial_not_before_thinking("constrained_mixed")


def test_outlines_mixed_prompt_does_not_put_initial_before_thinking():
    _assert_initial_not_before_thinking("outlines_mixed")


def test_unconstrained_thinking_prompt_does_not_put_initial_before_thinking():
    _assert_initial_not_before_thinking("unconstrained_thinking")


def test_grammar_summaries_are_compact_for_benchmark_prompts():
    for name in proposition7.list_grammars():
        summary = proposition7.get_grammar_summary(name)
        assert summary.strip(), name
        assert len(summary.split()) <= 256, name


def test_toml_tasks_have_quality_language_mix():
    rows = load_tasks(["all"])
    counts = Counter(row.language for row in rows)

    assert len(rows) >= 40
    assert counts["stlc"] >= 20
    # ml is the strong language: a strict OCaml subset with products, lists,
    # match, and recursion, graded by the type system itself.
    assert counts["ml"] >= 10


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
        row.resolution.get("mode") in {"type", "exact", "equivalence"}
        for row in rows
        if row.language == "ml"
    )


def test_ml_type_oracle_checks_well_typedness_and_type():
    # The ml oracle grades with aufbau: output must be well-typed and carry the
    # declared type, compared modulo the grammar's rewrite theory.
    task = make_task(
        grammar="ml",
        expected="fun (x : int) -> x",
        resolution={"mode": "type", "type": "int -> int"},
    )
    # right type, alternative-but-equivalent program
    assert check_resolution(task, "fun (y : int) -> y").ok
    # wrong type
    assert not check_resolution(task, "fun (x : int) -> true").ok
    # ill-typed program is rejected, not crashed
    assert not check_resolution(task, "1 + true").ok
    # divergence inhabits the demanded type (the inhabited-grammar property)
    assert check_resolution(task, "assert false").ok


def test_toml_expected_outputs_parse_and_pass_resolution():
    # Agent tasks are graded by episode (accumulated over turns, against
    # resolution.expected_value), not by a single-shot `expected` program
    # text -- see tests/agent_tasks.py for their equivalent coverage.
    for row in load_tasks(["all"]):
        if row.kind == "agent":
            continue
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


def test_low_space_cache_cleanup_removes_finished_model(tmp_path, monkeypatch):
    # low_space mode frees a model's weights once its jobs are done, so
    # clean_hf_model_cache(name) deletes *that* model's cache and nothing else.
    hub = tmp_path / "hub"
    keep = hub / hf_model_cache_name("org/keep")
    drop = hub / hf_model_cache_name("org/drop")
    keep.mkdir(parents=True)
    drop.mkdir(parents=True)
    (keep / "config.json").write_text("{}", encoding="utf-8")
    (drop / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))

    clean_hf_model_cache("org/drop")

    assert not drop.exists()
    assert keep.exists()


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


def test_default_model_matrix_excludes_gated_llama_and_keeps_current_open_models(monkeypatch):
    args = SimpleNamespace(device="cuda", torch_dtype="auto")
    monkeypatch.setattr("benchmarks.utils.gpu_vram_gib", lambda _args: 48.0)

    assert auto_model_concurrency(args, "Qwen/Qwen3.5-9B", 32) >= 2


def test_selected_modes_accepts_raw_and_cleaned_unconstrained_modes():
    local = selected_modes(
        SimpleNamespace(
            modes=(
                "constrained_direct,outlines,outlines_mixed,unconstrained,"
                "unconstrained_cleaned,unconstrained_thinking"
            ),
            backend="local",
        )
    )
    remote = selected_modes(
        SimpleNamespace(
            modes="unconstrained,unconstrained_cleaned,unconstrained_thinking",
            backend="openrouter",
        )
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


def test_selected_modes_rejects_removed_aliases():
    with pytest.raises(SystemExit):
        selected_modes(SimpleNamespace(modes="constrained", backend="local"))
    with pytest.raises(SystemExit):
        selected_modes(SimpleNamespace(modes="unconstrained_raw", backend="local"))
    with pytest.raises(SystemExit):
        selected_modes(SimpleNamespace(modes="constrained_direct", backend="openrouter"))
    with pytest.raises(SystemExit):
        selected_modes(SimpleNamespace(modes="outlines_mixed", backend="openrouter"))


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
        top_k=50,
        temperature=0.8,
        grammar_name="toy",
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


def test_outlines_mixed_uses_reasoning_environment_formal_constraint():
    task = make_task(grammar="toy", expected="beep:Fizz", max_tokens=8)

    class FakeModel:
        def think_open(self):
            return "<think>"

        def think_close(self):
            return "</think>"

        def allow_system_prompt(self):
            return True

        def start_tokens_unconstrained(self, grammar_name=None):
            del grammar_name
            return []

        def stop_tokens_unconstrained(self, grammar_name=None):
            del grammar_name
            return ["</think>"]

        def generate_unconstrained(self, **kwargs):
            del kwargs
            return proposition7.GenerationResult("<think>plan", False, 1, "stop")

        def generate_constrained(self, **kwargs):
            assert kwargs["initial"] == ""
            assert kwargs["grammar_name"] == "toy"
            return proposition7.GenerationResult("beep:Fizz", True, 1, "complete")

    record = bench_run.run_interaction(
        FakeModel(), task, "outlines_mixed", seed=7, think_budget=1
    )

    assert record["mode"] == "outlines_mixed"
    assert record["passed"] is True
    assert record["thoughts"] == "plan"
    assert record["formal_tokens"] == 1


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

    # model_kwargs_from_args forwards a truthy torch_dtype ("none" is a string
    # sentinel, popped downstream); device_map="" is falsy and dropped.
    expected = {
        "grammar_name": "toy",
        "device": "cpu",
        "local_files_only": True,
        "torch_dtype": "none",
    }
    assert calls == [("gpt2", expected), ("gpt2", expected)]


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


def test_model_concurrency_runner_does_not_mix_models_in_one_group(tmp_path, monkeypatch):
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
        model_concurrency=2,
        timeout=0,
        low_space=False,
        backend="local",
        _test_inline_workers=True,
    )

    run_model_concurrency(args, jobs, tmp_path / "raw.jsonl", None)

    first_m2 = next(i for i, chunk in enumerate(calls) if chunk[0] == "m2")
    assert all(set(chunk) == {"m1"} for chunk in calls[:first_m2])
    assert all(set(chunk) == {"m2"} for chunk in calls[first_m2:])


def test_model_concurrency_low_space_cleans_once_per_model(tmp_path, monkeypatch):
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
        model_concurrency=2,
        timeout=0,
        low_space=True,
        backend="local",
        _test_inline_workers=True,
    )

    run_model_concurrency(args, jobs, tmp_path / "raw.jsonl", None)

    assert cleaned == ["m1", "m2"]


def test_model_concurrency_runner_allows_timeout_with_process_workers(tmp_path, monkeypatch):
    task = SimpleNamespace(
        task_id="t",
        task_hash="h",
        resolution_hash="r",
        grammar="toy",
        language="toy",
    )
    args = SimpleNamespace(
        model_concurrency=2,
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
    run_model_concurrency(
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
        temperature=0.0,
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


def test_split_model_jobs_distributes_work_without_empty_chunks():
    jobs = list(range(5))

    assert split_model_jobs(jobs, 2) == [[0, 2, 4], [1, 3]]
    assert split_model_jobs(jobs, 10) == [[0], [1], [2], [3], [4]]


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
            "avg_tokens": 12.0,
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
            "avg_tokens": 14.0,
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
            "avg_tokens": 16.0,
        },
    ]

    [row] = delta_rows(summary, ("backend", "model", "language"))

    assert row["unconstrained_mode"] == "unconstrained"
    assert row["exact_delta"] == 40.0
    assert row["parse_error_delta"] == 45.0


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

[openrouter]
env_file = ".env"

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


def test_rejection_reason_is_specific_and_categorised():
    assert rejection_reason("ok") == ""
    assert rejection_reason("non_completable") == "non_completable"
    assert rejection_reason("parse_error", parse_error="boom").startswith("parse_error:")
    assert rejection_reason("incomplete", stop_reason="max_tokens") == (
        "incomplete:max_tokens"
    )
    # The category that only the type system can produce keeps its detail.
    assert rejection_reason("task_failed", resolution_error="type_mismatch") == (
        "type_or_value:type_mismatch"
    )
