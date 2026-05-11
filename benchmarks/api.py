from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    import tomllib
except ImportError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

import proposition7
from proposition7 import build_task_prompt
from benchmarks.oracles import (
    check_resolution,
    validate_fun_samples_schema,
    validate_imp_samples_schema,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

DEFAULT_MODELS = [
    "Qwen/Qwen3.5-0.5B",
    "Qwen/Qwen3.5-1.5B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
]


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    grammar: str
    category: str
    prompt: str
    initial: str
    expected: str
    max_tokens: int
    resolution: dict[str, Any]
    task_hash: str
    resolution_hash: str

    @property
    def language(self) -> str:
        return self.grammar


@dataclass(frozen=True)
class BenchmarkInteraction:
    task: BenchmarkTask
    grammar_name: str
    mode: str
    prompt: str
    initial: str
    max_tokens: int
    expected: str


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _load_task_file(path: Path) -> BenchmarkTask:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    try:
        task_id = str(data["id"])
        grammar = str(data["grammar"])
        category = str(data["category"])
        max_tokens = int(data["max_tokens"])
        prompt = str(data["prompt"]["text"])
        initial = str(data["initial"]["text"])
        expected = str(data["expected"]["text"])
        resolution = dict(data["resolution"])
    except KeyError as error:
        raise ValueError(f"{path}: missing required TOML field {error}") from error

    _validate_task_resolution(path, task_id, grammar, resolution)

    canonical = {
        "id": task_id,
        "grammar": grammar,
        "category": category,
        "max_tokens": max_tokens,
        "prompt": prompt,
        "initial": initial,
        "expected": expected,
        "resolution": resolution,
    }
    return BenchmarkTask(
        task_id=task_id,
        grammar=grammar,
        category=category,
        prompt=prompt,
        initial=initial,
        expected=expected,
        max_tokens=max_tokens,
        resolution=resolution,
        task_hash=stable_hash(canonical),
        resolution_hash=stable_hash(resolution),
    )


def _validate_task_resolution(
    path: Path, task_id: str, grammar: str, resolution: dict[str, Any]
) -> None:
    mode = resolution.get("mode")
    if not isinstance(mode, str) or not mode.strip():
        raise ValueError(
            f"{path}: {task_id}: resolution.mode must be a non-empty string"
        )

    if grammar == "fun":
        allowed = {"exact", "equivalence", "value", "samples"}
        if mode not in allowed:
            raise ValueError(
                f"{path}: {task_id}: unsupported fun resolution mode {mode!r}"
            )
        if mode == "samples":
            try:
                validate_fun_samples_schema(resolution.get("samples"))
            except ValueError as error:
                raise ValueError(
                    f"{path}: {task_id}: invalid samples spec: {error}"
                ) from error
            if "fn" in resolution and not str(resolution.get("fn") or "").strip():
                raise ValueError(
                    f"{path}: {task_id}: resolution.fn must be a non-empty string"
                )
            if "hidden_samples" in resolution:
                hidden = resolution["hidden_samples"]
                try:
                    hidden_int = int(hidden)
                except Exception as error:
                    raise ValueError(
                        f"{path}: {task_id}: resolution.hidden_samples must be an int"
                    ) from error
                if hidden_int < 0:
                    raise ValueError(
                        f"{path}: {task_id}: resolution.hidden_samples must be >= 0"
                    )
        _validate_fun_structure(path, task_id, resolution.get("structure"))
        return

    if grammar == "stlc":
        if mode not in {"exact", "equivalence"}:
            raise ValueError(
                f"{path}: {task_id}: unsupported stlc resolution mode {mode!r}"
            )
        if "normalization" in resolution and not isinstance(
            resolution["normalization"], list
        ):
            raise ValueError(
                f"{path}: {task_id}: resolution.normalization must be a list"
            )
        if "type" in resolution and not str(resolution["type"]).strip():
            raise ValueError(f"{path}: {task_id}: resolution.type must be non-empty")
        return

    if grammar == "imp":
        if mode not in {"exact", "env", "samples"}:
            raise ValueError(
                f"{path}: {task_id}: unsupported imp resolution mode {mode!r}"
            )
        if mode == "env":
            if "env" in resolution and not isinstance(resolution["env"], dict):
                raise ValueError(f"{path}: {task_id}: resolution.env must be an object")
            if "vars" in resolution:
                vars_value = resolution["vars"]
                if not isinstance(vars_value, list) or not all(
                    isinstance(item, str) and item for item in vars_value
                ):
                    raise ValueError(
                        f"{path}: {task_id}: resolution.vars must be a list of names"
                    )
        if mode == "samples":
            try:
                validate_imp_samples_schema(
                    resolution.get("samples"),
                    resolution.get("input_vars"),
                    resolution.get("vars"),
                )
            except ValueError as error:
                raise ValueError(
                    f"{path}: {task_id}: invalid imp samples spec: {error}"
                ) from error
            if "hidden_samples" in resolution:
                try:
                    hidden_int = int(resolution["hidden_samples"])
                except Exception as error:
                    raise ValueError(
                        f"{path}: {task_id}: resolution.hidden_samples must be an int"
                    ) from error
                if hidden_int < 0:
                    raise ValueError(
                        f"{path}: {task_id}: resolution.hidden_samples must be >= 0"
                    )
            if "input_domains" in resolution:
                domains = resolution["input_domains"]
                if not isinstance(domains, dict):
                    raise ValueError(
                        f"{path}: {task_id}: resolution.input_domains must be an object"
                    )
                for name, domain in domains.items():
                    if not isinstance(name, str) or not name:
                        raise ValueError(
                            f"{path}: {task_id}: resolution.input_domains keys must be variable names"
                        )
                    if not isinstance(domain, dict):
                        raise ValueError(
                            f"{path}: {task_id}: resolution.input_domains.{name} must be an object"
                        )
                    if "min" in domain:
                        int(domain["min"])
                    if "max" in domain:
                        int(domain["max"])
        return

    if mode not in {"exact", "equivalence"}:
        raise ValueError(
            f"{path}: {task_id}: unsupported {grammar} resolution mode {mode!r}"
        )


def _validate_fun_structure(path: Path, task_id: str, structure: Any) -> None:
    if structure is None:
        return
    if not isinstance(structure, dict):
        raise ValueError(f"{path}: {task_id}: resolution.structure must be an object")
    for key in ["let_bindings", "applications", "lambdas"]:
        if key not in structure:
            continue
        try:
            value = int(structure[key])
        except Exception as error:
            raise ValueError(
                f"{path}: {task_id}: resolution.structure.{key} must be an int"
            ) from error
        if value < 0:
            raise ValueError(
                f"{path}: {task_id}: resolution.structure.{key} must be >= 0"
            )
    if "operations" in structure:
        operations = structure["operations"]
        if not isinstance(operations, list) or not all(
            isinstance(op, str) and op for op in operations
        ):
            raise ValueError(
                f"{path}: {task_id}: resolution.structure.operations must be a list of operators"
            )


def load_all_tasks() -> list[BenchmarkTask]:
    tasks = [_load_task_file(path) for path in sorted(DATA_DIR.glob("*.toml"))]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for task in tasks:
        if task.task_id in seen:
            duplicates.add(task.task_id)
        seen.add(task.task_id)
    if duplicates:
        raise SystemExit(f"Duplicate task ids: {', '.join(sorted(duplicates))}")
    return tasks


def load_tasks(names: Iterable[str]) -> list[BenchmarkTask]:
    selectors = {name.strip() for name in names if name and name.strip()}
    tasks = load_all_tasks()
    if not selectors or selectors & {"all", "core"}:
        return tasks

    selected = [
        task
        for task in tasks
        if task.task_id in selectors
        or task.grammar in selectors
        or task.category in selectors
    ]
    if selected:
        return selected
    raise SystemExit(f"Unknown task selectors: {', '.join(sorted(selectors))}")


def grammar_name(task_grammar: str) -> str:
    if task_grammar in proposition7.list_grammars():
        return task_grammar
    if task_grammar == "stlc_union" and "stlc" in proposition7.list_grammars():
        return "stlc"
    return task_grammar


# build_prompt is kept as an alias so existing callers keep working.
# The canonical implementation lives in proposition7.build_task_prompt.
def build_prompt(
    gname: str,
    instruction: str,
    *,
    mode: str = "constrained_direct",
    initial: str = "",
) -> str:
    return build_task_prompt(gname, instruction, mode=mode, initial=initial)


def build_interaction(task: BenchmarkTask, mode: str) -> BenchmarkInteraction:
    gname = grammar_name(task.grammar)
    return BenchmarkInteraction(
        task=task,
        grammar_name=gname,
        mode=mode,
        prompt=build_prompt(
            gname,
            task.prompt,
            mode=mode,
            initial=task.initial,
        ),
        initial=task.initial,
        max_tokens=task.max_tokens,
        expected=task.expected,
    )


def normalize(text: str) -> str:
    return " ".join(text.split())


def check_parse(spec: str, text: str) -> tuple[bool, bool, str]:
    try:
        synthesizer = proposition7.Synthesizer(spec, "")
        synthesizer.set_input(text)
        synthesizer.parse()
        return True, bool(synthesizer.is_complete()), ""
    except Exception as error:
        return False, False, str(error)


def _code_start_pattern(gname: str) -> re.Pattern[str]:
    patterns = {
        "imp": r"\{",
        "lamb": r"@",
        "toy": r"\b(?:beep|boop|blorp)\b|\(",
        "fun": r"\blet\b|\btrue\b|\bfalse\b|\d|\(",
        "stlc": r"λ|\(",
        "typescript": r"\b(?:function|let|const|return|if)\b",
    }
    return re.compile(patterns.get(gname, r"\S"))


def _candidate_program_texts(gname: str, text: str) -> list[str]:
    seen: set[str] = set()
    candidates: list[str] = []

    def add(candidate: str) -> None:
        stripped = candidate.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            candidates.append(stripped)

    add(text)

    for match in re.finditer(r"```[^\n`]*\n(.*?)```", text, flags=re.DOTALL):
        add(match.group(1))
    for match in re.finditer(r"```(.*?)```", text, flags=re.DOTALL):
        block = match.group(1)
        if "\n" in block:
            first, rest = block.split("\n", 1)
            if re.fullmatch(r"[A-Za-z0-9_+.-]*", first.strip()):
                block = rest
        add(block)

    lines = text.splitlines()
    for start, line in enumerate(lines):
        if not _code_start_pattern(gname).search(line.lstrip()):
            continue
        for end in range(len(lines), start, -1):
            add("\n".join(lines[start:end]))

    for match in list(_code_start_pattern(gname).finditer(text))[:32]:
        tail = text[match.start() :].strip()
        add(tail)
        words = tail.split()
        for end in range(min(len(words), 256), 0, -1):
            add(" ".join(words[:end]))

    return candidates


def extract_program_output(
    spec: str,
    gname: str,
    text: str,
    initial: str = "",
) -> tuple[str, bool]:
    candidates = _candidate_program_texts(gname, text)
    if initial and text.startswith(initial):
        suffix = text[len(initial) :]
        candidates.extend(
            initial + candidate for candidate in _candidate_program_texts(gname, suffix)
        )

    for candidate in candidates:
        parse_ok, complete, _ = check_parse(spec, candidate)
        if parse_ok and complete:
            return candidate, normalize(candidate) != normalize(text)
    return text, False


class FatalBenchmarkInvariantError(RuntimeError):
    pass


def classify(
    exact: bool,
    parse_ok: bool,
    complete: bool,
    parse_error: str = "",
    semantic_ok: Optional[bool] = None,
) -> str:
    if not parse_ok:
        if "no parse found" in parse_error.lower():
            return "non_completable"
        return "parse_error"
    if not complete:
        return "incomplete"
    if semantic_ok is not None:
        return "ok" if semantic_ok else "task_failed"
    if exact:
        return "ok"
    return "task_failed"


def result_diagnostics(result: Any) -> dict[str, Any]:
    diagnostics = getattr(result, "diagnostics", {}) or {}
    return dict(diagnostics) if isinstance(diagnostics, dict) else {}


def run_interaction(
    model: Any,
    task: BenchmarkTask,
    mode: str,
    *,
    seed: Optional[int] = None,
    think_budget: int = 128,
    temperature: float = 0.0,
    trace: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    interaction = build_interaction(task, mode)
    token_trace: list[dict[str, Any]] = []
    think_trace: list[dict[str, Any]] = []
    mode_switches: list[dict[str, Any]] = []
    spec = proposition7.get_grammar(interaction.grammar_name)

    started_at = time.time()
    mixed_extra: dict[str, Any] = {}
    try:
        if mode in {"constrained_mixed", "outlines_mixed"}:
            result, mixed_extra = run_mixed_generation(
                model,
                interaction,
                think_budget=think_budget,
                temperature=temperature,
                seed=seed,
            )
        elif mode == "unconstrained_thinking":
            result, mixed_extra = run_unconstrained_thinking_generation(
                model,
                interaction,
                think_budget=think_budget,
                temperature=temperature,
                seed=seed,
            )
        elif mode in {"constrained_direct", "outlines"}:
            result = model.generate_constrained(
                prompt=interaction.prompt,
                initial=interaction.initial,
                max_tokens=interaction.max_tokens,
                grammar_name=interaction.grammar_name,
                seed=seed,
                temperature=temperature,
            )
        elif mode in {"unconstrained", "unconstrained_cleaned"}:
            result = model.generate_unconstrained(
                prompt=interaction.prompt,
                initial=interaction.initial,
                max_tokens=interaction.max_tokens,
                temperature=temperature,
                grammar_name=interaction.grammar_name,
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown benchmark mode: {mode}")
    except Exception as error:
        seconds = time.time() - started_at
        output = f"ERROR: {error}"
        record = {
            "task_id": task.task_id,
            "language": task.language,
            "grammar": task.grammar,
            "category": task.category,
            "task_hash": task.task_hash,
            "resolution_hash": task.resolution_hash,
            "resolution_mode": str(task.resolution.get("mode", "exact")),
            "mode": mode,
            "expected": interaction.expected,
            "output": output,
            "exact": False,
            "parse_ok": False,
            "parse_complete": False,
            "error": "parse_error",
            "parse_error": str(error),
            "stop_reason": "model_error",
            "tokens": 0,
            "step_token_ids": [],
            "step_pre_entropies": [],
            "step_entropies": [],
            "step_retries": [],
            "diagnostics": {},
            "thoughts": "",
            "think_tokens": 0,
            "formal_tokens": 0,
            "reasoning_blocks": [],
            "seed": seed,
            "passed": False,
            "seconds": round(seconds, 4),
        }
        if trace is not None:
            trace.append(
                build_token_log(
                    task,
                    mode,
                    interaction.prompt,
                    spec,
                    token_trace,
                    record,
                    think_trace=think_trace,
                    mode_switches=mode_switches,
                )
            )
        return record

    seconds = time.time() - started_at
    raw_output = result.text
    output = raw_output
    output_extracted = False
    if mode == "unconstrained_cleaned":
        output, output_extracted = extract_program_output(
            spec,
            interaction.grammar_name,
            raw_output,
            interaction.initial,
        )
    parse_ok, parse_complete, parse_error = check_parse(spec, output)
    exact = normalize(output) == normalize(interaction.expected)
    resolution_result = None
    if parse_ok and parse_complete:
        resolution_result = check_resolution(task, output)
    semantic_ok = resolution_result.ok if resolution_result is not None else None
    error = classify(exact, parse_ok, parse_complete, parse_error, semantic_ok)
    if error == "non_completable" and mode in {
        "constrained_direct",
        "constrained_mixed",
        "outlines",
        "outlines_mixed",
    }:
        raise FatalBenchmarkInvariantError(
            "Constrained decoding produced a non-completable output: "
            f"mode={mode} task_id={task.task_id} model_output={output!r}"
        )
    record = {
        "task_id": task.task_id,
        "language": task.language,
        "grammar": task.grammar,
        "category": task.category,
        "task_hash": task.task_hash,
        "resolution_hash": task.resolution_hash,
        "resolution_mode": str(task.resolution.get("mode", "exact")),
        "mode": mode,
        "expected": interaction.expected,
        "output": output,
        "output_extracted": output_extracted,
        "exact": exact,
        "parse_ok": parse_ok,
        "parse_complete": parse_complete,
        "semantic_ok": bool(semantic_ok) if semantic_ok is not None else False,
        "resolution_error": resolution_result.reason
        if resolution_result is not None
        else "",
        "resolution_observed": resolution_result.observed
        if resolution_result is not None
        else None,
        "resolution_expected": resolution_result.expected
        if resolution_result is not None
        else None,
        "error": error,
        "parse_error": parse_error,
        "stop_reason": result.stopped_reason,
        "tokens": result.tokens_generated,
        # Per-step arrays (parallel, length == tokens): constrained modes only.
        # step_pre_entropies: H before grammar masking (model's raw distribution).
        # step_entropies: H after grammar masking (grammar-valid tokens only).
        # Difference (pre − post) measures how much the grammar mask shapes each step.
        "step_token_ids": result.step_token_ids,
        "step_pre_entropies": result.step_pre_entropies,
        "step_entropies": result.step_entropies,
        "step_retries": result.step_retries,
        "diagnostics": result_diagnostics(result),
        "thoughts": mixed_extra.get("thoughts", ""),
        "think_tokens": mixed_extra.get("think_tokens", 0),
        "formal_tokens": mixed_extra.get(
            "formal_tokens",
            result.tokens_generated
            if mode
            in {
                "constrained_direct",
                "constrained_mixed",
                "unconstrained_thinking",
                "outlines",
                "outlines_mixed",
            }
            else 0,
        ),
        "reasoning_blocks": mixed_extra.get("reasoning_blocks", []),
        "seed": seed,
        "passed": error == "ok",
        "seconds": round(seconds, 4),
    }
    if output_extracted:
        record["raw_output"] = raw_output
    if trace is not None:
        trace.append(
            build_token_log(
                task,
                mode,
                interaction.prompt,
                spec,
                token_trace,
                record,
                think_trace=think_trace,
                mode_switches=mode_switches,
            )
        )
    return record


def run_mixed_generation(
    model: Any,
    interaction: BenchmarkInteraction,
    *,
    think_budget: int,
    temperature: float,
    seed: Optional[int],
) -> tuple[Any, dict[str, Any]]:

    env = proposition7.ReasoningEnvironment(
        model,
        interaction.grammar_name,
        think_budget=think_budget,
        formal_budget=interaction.max_tokens,
        temperature=temperature,
    )
    env_result = env.generate(
        interaction.prompt,
        initial=interaction.initial,
    )
    final = env_result.final_output
    result = proposition7.GenerationResult(
        text=final.content if final is not None else "",
        is_complete=env_result.is_complete,
        tokens_generated=env_result.total_tokens,
        stopped_reason=env_result.stopped_reason,
        diagnostics={},
    )
    return result, {
        "thoughts": env_result.all_thoughts,
        "think_tokens": env_result.think_tokens,
        "formal_tokens": env_result.formal_tokens,
        "reasoning_blocks": [str(block) for block in env_result.blocks],
    }


def run_unconstrained_thinking_generation(
    model: Any,
    interaction: BenchmarkInteraction,
    *,
    think_budget: int,
    temperature: float,
    seed: Optional[int],
) -> tuple[Any, dict[str, Any]]:
    think_open: str = model.think_open() if hasattr(model, "think_open") else "<think>"
    think_close: str = model.think_close() if hasattr(model, "think_close") else "</think>"

    # Use think_open as the think-phase initial so _set_prompt can strip the
    # pre-closed think block injected by enable_thinking=False and open a clean
    # think block.  For models where start_tokens_unconstrained already includes
    # think_open (i.e. the template adds it automatically), use "" to avoid
    # injecting a duplicate tag.
    think_initial = think_open
    start_fn = getattr(model, "start_tokens_unconstrained", None)
    if callable(start_fn):
        try:
            toks = start_fn(interaction.grammar_name)
            if any(str(t).rstrip() == think_open.rstrip() for t in toks):
                think_initial = ""
        except Exception:
            pass

    think_prompt = (
        interaction.prompt
        + "\n\nThink briefly before writing the final program. Do not write the final program in this phase."
    )
    think_result = model.generate_unconstrained(
        prompt=think_prompt,
        initial=think_initial,
        max_tokens=think_budget,
        temperature=temperature,
        grammar_name=interaction.grammar_name,
        seed=seed,
    )
    thoughts = think_result.text.strip()
    # Strip the injected think_open tag and any trailing think_close that
    # wasn't consumed as a stop token.
    if thoughts.startswith(think_open):
        thoughts = thoughts[len(think_open):].lstrip("\n")
    if thoughts.endswith(think_close):
        thoughts = thoughts[: -len(think_close)].rstrip()

    formal_prompt = (
        build_prompt(
            interaction.grammar_name,
            interaction.task.prompt,
            mode="unconstrained_thinking",
            initial=interaction.initial,
        )
        + "\n\nPrior reasoning:\n"
        + thoughts
        + "\n\nNow write only the final program text."
    )
    formal_result = model.generate_unconstrained(
        prompt=formal_prompt,
        initial=interaction.initial,
        max_tokens=interaction.max_tokens,
        temperature=temperature,
        grammar_name=interaction.grammar_name,
        seed=seed,
    )
    result = proposition7.GenerationResult(
        text=formal_result.text,
        is_complete=formal_result.is_complete,
        tokens_generated=think_result.tokens_generated + formal_result.tokens_generated,
        stopped_reason=formal_result.stopped_reason,
        diagnostics={},
    )
    return result, {
        "thoughts": thoughts,
        "think_tokens": think_result.tokens_generated,
        "formal_tokens": formal_result.tokens_generated,
        "reasoning_blocks": [thoughts] if thoughts else [],
    }


def build_token_log(
    task: BenchmarkTask,
    mode: str,
    prompt: str,
    spec: str,
    token_trace: list[dict[str, Any]],
    record: dict[str, Any],
    *,
    think_trace: Optional[list[dict[str, Any]]] = None,
    mode_switches: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    tokens: list[dict[str, Any]] = []
    current_text = task.initial
    for row in token_trace:
        before = current_text
        current_text += row["token"]
        parse_ok, parse_complete, parse_error = check_parse(spec, current_text)
        tokens.append(
            {
                "step": row["step"],
                "token": row["token"],
                "char_start": len(before),
                "char_end": len(current_text),
                "text_after": current_text,
                "parse_ok": parse_ok,
                "parse_complete": parse_complete,
                "parse_error": parse_error,
            }
        )

    return {
        "task_id": task.task_id,
        "language": task.language,
        "grammar": task.grammar,
        "category": task.category,
        "task_hash": task.task_hash,
        "resolution_hash": task.resolution_hash,
        "resolution_mode": str(task.resolution.get("mode", "exact")),
        "mode": mode,
        "initial": task.initial,
        "expected": task.expected,
        "prompt": prompt,
        "output": record["output"],
        "error": record["error"],
        "passed": record["passed"],
        "parse_ok": record["parse_ok"],
        "parse_complete": record["parse_complete"],
        "parse_error": record["parse_error"],
        "stop_reason": record["stop_reason"],
        "seed": record.get("seed"),
        "tokens_generated": record["tokens"],
        "seconds": record["seconds"],
        "diagnostics": record.get("diagnostics", {}),
        "thoughts": record.get("thoughts", ""),
        "think_tokens": record.get("think_tokens", 0),
        "formal_tokens": record.get("formal_tokens", 0),
        "reasoning_blocks": record.get("reasoning_blocks", []),
        "think_trace": think_trace or [],
        "mode_switches": mode_switches or [],
        "tokens": tokens,
    }


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["model"], record["mode"], record["language"])].append(record)

    summary: list[dict[str, Any]] = []
    for (model_name, mode, language), group in sorted(grouped.items()):
        attempts = len(group)
        counts = Counter(record["error"] for record in group)
        exact = sum(1 for record in group if record["exact"])
        avg_tokens = sum(float(record.get("tokens", 0)) for record in group) / max(
            attempts, 1
        )
        avg_seconds = sum(float(record.get("seconds", 0.0)) for record in group) / max(
            attempts, 1
        )
        summary.append(
            {
                "model": model_name,
                "mode": mode,
                "language": language,
                "attempts": attempts,
                "exact_rate": round(100.0 * exact / max(attempts, 1), 2),
                "ok_rate": round(100.0 * counts.get("ok", 0) / max(attempts, 1), 2),
                "parse_error_rate": round(
                    100.0 * counts.get("parse_error", 0) / max(attempts, 1), 2
                ),
                "incomplete_rate": round(
                    100.0 * counts.get("incomplete", 0) / max(attempts, 1), 2
                ),
                "task_failed_rate": round(
                    100.0 * counts.get("task_failed", 0) / max(attempts, 1), 2
                ),
                "avg_tokens": round(avg_tokens, 2),
                "avg_seconds": round(avg_seconds, 2),
            }
        )
    return summary


def summarize_overall(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["model"], record["mode"])].append(record)

    summary: list[dict[str, Any]] = []
    for (model_name, mode), group in sorted(grouped.items()):
        attempts = len(group)
        exact = sum(1 for record in group if record["exact"])
        parse_ok = sum(1 for record in group if record["parse_ok"])
        parse_complete = sum(1 for record in group if record["parse_complete"])
        counts = Counter(record["error"] for record in group)
        summary.append(
            {
                "model": model_name,
                "mode": mode,
                "attempts": attempts,
                "exact_rate": round(100.0 * exact / max(attempts, 1), 2),
                "parse_ok_rate": round(100.0 * parse_ok / max(attempts, 1), 2),
                "parse_complete_rate": round(
                    100.0 * parse_complete / max(attempts, 1), 2
                ),
                "error_breakdown": dict(sorted(counts.items())),
            }
        )
    return summary


def failure_samples(
    records: list[dict[str, Any]], limit: int = 5
) -> list[dict[str, Any]]:
    failures = [record for record in records if record["error"] != "ok"]
    samples = []
    for record in failures[:limit]:
        samples.append(
            {
                "task_id": record["task_id"],
                "mode": record["mode"],
                "language": record["language"],
                "category": record["category"],
                "resolution_mode": record.get("resolution_mode", "exact"),
                "error": record["error"],
                "parse_error": record.get("parse_error", ""),
                "stop_reason": record.get("stop_reason", ""),
                "output_preview": record["output"][:200],
            }
        )
    return samples


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")
