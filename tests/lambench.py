import os
from pathlib import Path

import pytest
import p7

from benchmarks.lambench import build_prompt, extract_submission, parse_task


def assert_lamb_complete(text: str) -> None:
    synthesizer = p7.Synthesizer(p7.get_grammar("lamb"), "")
    synthesizer.set_input(text)
    synthesizer.parse()
    assert synthesizer.is_complete()


def assert_lamb_rejects(text: str) -> None:
    synthesizer = p7.Synthesizer(p7.get_grammar("lamb"), "")
    synthesizer.set_input(text)
    try:
        synthesizer.parse()
    except RuntimeError as error:
        assert "no parse found" in str(error)
    else:
        raise AssertionError(f"Lamb program unexpectedly parsed: {text}")


def test_lambench_task_parser_and_prompt(tmp_path: Path):
    task_path = tmp_path / "cnat_add.tsk"
    task_path.write_text(
        "Add two Church nats.\n---\n@main(λf.λx.x, λf.λx.x)\n= λa.λb.b\n",
        encoding="utf-8",
    )

    task = parse_task(task_path)
    prompt = build_prompt(task)

    assert task.task_id == "cnat_add"
    assert task.tests[0].expr == "@main(λf.λx.x, λf.λx.x)"
    assert "The program must define @main" in prompt


def test_extract_submission_removes_markdown_wrapper():
    text = "Here is the answer:\n```lam\n@main = λx.x\n```"

    assert extract_submission(text) == "@main = λx.x"


def test_lambench_lamb_scope_accepts_previous_global_refs():
    assert_lamb_complete("@id = λx.x @main = @id")
    assert_lamb_complete("@id = λx.x @main = @id(@id)")
    assert_lamb_complete("@id = λx.x @const = λy.@id @main = @const(@id)")


def test_lambench_lamb_scope_rejects_missing_forward_and_recursive_refs():
    assert_lamb_rejects("@main = @missing")
    assert_lamb_rejects("@main = @id @id = λx.x")
    assert_lamb_rejects("@loop = @loop")


def test_lambench_lamb_scope_keeps_lambda_shadowing_parseable():
    assert_lamb_complete("@main = λx.λx.x")


def test_lambench_scoped_reference_solutions_parse_when_available():
    root = Path(
        os.environ.get(
            "LAMBENCH_DIR",
            "/tmp/lambench-main-extract/lambench-main",
        )
    )
    lam_dir = root / "lam"
    if not lam_dir.exists():
        pytest.skip("set LAMBENCH_DIR or download Lambench to run this test")

    scoped_examples = [
        lam_dir / "cnat_add.lam",
        lam_dir / "clst_map.lam",
        lam_dir / "cbin_mul.lam",
    ]
    existing = [path for path in scoped_examples if path.exists()]
    if not existing:
        pytest.skip("sample scoped-compatible Lambench references are unavailable")

    for path in existing:
        assert_lamb_complete(path.read_text(encoding="utf-8").strip())
