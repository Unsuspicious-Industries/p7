"""Two-stage reasoning environment: think once, then write formal output."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Union

from grammars import get_grammar_info

# Mode strings used across the benchmark and environment.
BENCHMARK_MODES = frozenset({
    "constrained_direct",
    "constrained_mixed",
    "outlines",
    "outlines_mixed",
    "syntactic_only",
    "unconstrained",
    "unconstrained_cleaned",
    "unconstrained_thinking",
})


class Mode(Enum):
    """Current generation mode."""

    THINK = "think"
    FORMAL = "formal"


@dataclass
class ThinkBlock:
    """A block of unconstrained reasoning."""

    content: str
    tokens: int = 0

    def __str__(self) -> str:
        return f"<think>{self.content}</think>"


@dataclass
class FormalBlock:
    """A block of grammar-constrained formal output."""

    content: str
    grammar_name: str
    is_complete: bool
    tokens: int = 0

    def __str__(self) -> str:
        return f"<formal>{self.content}</formal>"


GrammarBlock = FormalBlock


@dataclass
class EnvironmentResult:
    """Result from environment generation."""

    blocks: List[Union[ThinkBlock, FormalBlock]] = field(default_factory=list)
    total_tokens: int = 0
    stopped_reason: str = "max_tokens"
    grammar_name: str = ""

    @property
    def think_blocks(self) -> List[ThinkBlock]:
        return [b for b in self.blocks if isinstance(b, ThinkBlock)]

    @property
    def formal_blocks(self) -> List[FormalBlock]:
        return [b for b in self.blocks if isinstance(b, FormalBlock)]

    @property
    def grammar_blocks(self) -> List[FormalBlock]:
        return self.formal_blocks

    @property
    def final_output(self) -> Optional[FormalBlock]:
        """Get the formal output block."""
        blocks = self.formal_blocks
        return blocks[-1] if blocks else None

    @property
    def all_thoughts(self) -> str:
        """Concatenate all thinking."""
        return "\n".join(b.content for b in self.think_blocks)

    @property
    def think_tokens(self) -> int:
        return sum(block.tokens for block in self.think_blocks)

    @property
    def formal_tokens(self) -> int:
        return sum(block.tokens for block in self.formal_blocks)

    @property
    def is_complete(self) -> bool:
        """Check if we have a complete grammar output."""
        final = self.final_output
        return final is not None and final.is_complete

    def __str__(self) -> str:
        return "".join(str(b) for b in self.blocks)


def build_system_prompt(
    grammar_name: str,
    task_description: Optional[str] = None,
    include_examples: bool = True,
    think_open: str = "<think>",
    think_close: str = "</think>",
) -> str:
    """
    Procedurally generate a compact system prompt for the given grammar.

    Args:
        grammar_name: Name of the grammar (e.g., "stlc", "ml", "c")
        task_description: Optional task-specific description
        include_examples: Whether to include syntax examples
        think_open: Model-specific opening tag for reasoning blocks
        think_close: Model-specific closing tag for reasoning blocks

    Returns:
        System prompt string
    """
    info = get_grammar_info(grammar_name)
    summary = str(info.get("summary") or info.get("description") or grammar_name)

    # Strip trailing whitespace from tags so they render cleanly in the prompt
    # (some models use "<think>\n" as their open token).
    think_open_display = think_open.rstrip()
    think_close_display = think_close.rstrip()

    lines = [
        f"You produce well-typed {info['short']}.",
        "",
        "Use exactly two blocks:",
        f"- {think_open_display}...{think_close_display}: brief free-form reasoning.",
        "- <formal>...</formal>: final grammar-constrained output only.",
        "",
        "The formal block must contain only program text: no prose, markdown, labels, or repeated task text.",
        "If a formal prefix is provided, it appears immediately after <formal>; continue from that prefix.",
        "",
        f"Language summary:\n{summary}",
    ]

    if info["syntax_hints"]:
        lines.extend(["", "Syntax:"])
        for hint in info["syntax_hints"]:
            lines.append(f"  - {hint}")

    if include_examples and info["examples"]:
        lines.extend(["", "Examples:"])
        for name, code in info["examples"]:
            lines.append(f"  {name}: {code}")

    if task_description:
        lines.extend(["", f"Task: {task_description}"])

    return "\n".join(lines)


# Per-grammar generation rules surfaced in the task-level prompt.
# These capture grammar-specific pitfalls that are not obvious from the summary.
_GRAMMAR_RULES: dict = {
    "stlc": (
        "STLC rule: produce one lambda term. If the prefix ends after a lambda dot, "
        "continue with that body immediately. For multi-argument functions add lambdas "
        "before the body. Double application must be nested: `(f (f x))` not `(f x)`."
    ),
    "ml": (
        "ML rule: produce one expression, a strict OCaml subset. Use `fun (x : t) -> "
        "body` for lambdas, `f(arg)` for application, and `let rec` for recursive "
        "functions over lists. `assert false` is the universal inhabitant when a branch "
        "is unreachable."
    ),
    "c": (
        "C rule: produce one or more function definitions of real, compilable C, any "
        "number of parameters. Declare variables before use. Use & and * for pointers "
        "and explicit casts `(T) e` rather than relying on implicit conversions. "
        "Arithmetic is + - / only, never * (reserved for pointer syntax). A function "
        "may call any function defined earlier in the same program, but never itself."
    ),
    "tool": (
        "Tool rule: produce a sequence of `let name = tool(arg);` bindings ending in "
        "`return value;`. Each tool takes exactly one argument of its declared type "
        "(a string/int literal or an already-bound variable) — the tools are "
        "search(string)->docs, summarize(docs)->string, count(docs)->int, "
        "format(int)->string."
    ),
    "tool_sexpr": (
        "Tool rule: the same fixed tools as `tool` (search/summarize/count/format), "
        "written as S-expressions: `(let name (tool arg))` for each binding, ending "
        "in `(return value)`."
    ),
}


def build_task_prompt(
    grammar_name: str,
    instruction: str,
    *,
    mode: str = "constrained_direct",
    initial: str = "",
) -> str:
    """Build the per-task user prompt for the given generation mode.

    This is the canonical prompt used by the benchmark across all modes and by
    ReasoningEnvironment when no explicit system_prompt is provided.  Keep the
    grammar summary and language rules here so there is exactly one place to edit
    them for reproducible eval runs.

    Args:
        grammar_name: Grammar identifier (``"stlc"``, ``"ml"``, ``"c"``, …).
        instruction: Task-specific instruction text from the benchmark task file.
        mode: One of the BENCHMARK_MODES strings.
        initial: Decoder-injected prefix (used only in constrained_direct / outlines).

    Returns:
        Prompt string ready to pass to the model.
    """
    info = get_grammar_info(grammar_name)
    summary = str(info.get("summary") or info.get("description") or grammar_name)
    rule = _GRAMMAR_RULES.get(grammar_name, "")

    lines = [
        f"Language: {grammar_name}",
        f"Language summary:\n{summary}",
        f"Task: {instruction}",
    ]
    if rule:
        lines.append(rule)

    if mode in {"constrained_mixed", "outlines_mixed", "unconstrained_thinking"}:
        lines.extend([
            "Workflow: think briefly, then write the final program text directly.",
            "Output only program text: no explanation, markdown, labels, or lead-in words.",
            "Stop as soon as the complete program satisfies the task.",
        ])
    elif mode in {"constrained_direct", "outlines", "syntactic_only"}:
        lines.extend([
            "Write the completed program directly.",
            "Output only program text: no explanation, markdown, labels, or lead-in words.",
            "If a prefix is already in the decoder, continue exactly after it and do not repeat it.",
            "Stop as soon as the complete program satisfies the task.",
        ])
        if initial:
            lines.extend([
                "Decoder prefix already present:",
                initial,
                "Begin your generated continuation immediately after this prefix.",
            ])

    return "\n".join(lines)


class ReasoningEnvironment:
    """
    Environment that performs exactly one think block and one formal block.

    Usage:
        from proposition7 import ConstrainedModel, GRAMMARS

        model = ConstrainedModel.from_pretrained("...", grammar=get_grammar("stlc"))
        env = ReasoningEnvironment(model, grammar_name="stlc")

        result = env.generate(
            prompt="Create a function that takes an Int and returns it",
            initial="λx:",
        )

        print(result.all_thoughts)  # CoT reasoning
        print(result.final_output)  # Well-typed formal output
    """

    def __init__(
        self,
        model,  # ConstrainedModel
        grammar_name: str,
        think_budget: int = 200,
        formal_budget: int = 100,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ):
        """
        Initialize the reasoning environment.

        Args:
            model: A ConstrainedModel with grammar loaded
            grammar_name: Name of the grammar
            think_budget: Max tokens per think block
            formal_budget: Max tokens per formal block
            system_prompt: Custom system prompt (auto-generated if None)
            temperature: Temperature for both think and formal generation
        """
        self.model = model
        self.grammar_name = grammar_name
        self.think_budget = think_budget
        self.formal_budget = formal_budget
        self.temperature = temperature
        self.system_prompt = system_prompt

        self.THINK_OPEN = self.model.think_open()
        self.THINK_CLOSE = self.model.think_close()

        # Stop tokens for think mode
        self._think_stop = _dedupe(
            self.model.stop_tokens_unconstrained(grammar_name) + [self.THINK_CLOSE]
        )

    def _unconstrained_start_includes_think(self) -> bool:
        start_tokens = getattr(self.model, "start_tokens_unconstrained", None)
        if not callable(start_tokens):
            return False
        try:
            tokens = start_tokens(self.grammar_name)
        except TypeError:
            tokens = start_tokens()
        think_open = self.THINK_OPEN.rstrip()
        return any(str(token).rstrip() == think_open for token in tokens)

    def _generate_think(
        self,
        prompt: str,
        initial: str = "",
    ) -> Tuple[str, int, str]:
        """
        Generate unconstrained thinking until </think> or <formal>.

        Returns: (content, tokens_generated, stopped_reason)
        """
        result = self.model.generate_unconstrained(
            prompt=prompt,
            initial=initial,
            max_tokens=self.think_budget,
            temperature=self.temperature,
            stop_tokens=self._think_stop,
            grammar_name=self.grammar_name,
        )

        content = result.text

        if content.startswith(self.THINK_OPEN):
            content = content[len(self.THINK_OPEN) :]

        for tag in [self.THINK_CLOSE]:
            if tag in content:
                idx = content.find(tag)
                content = content[:idx]
                break

        return content, result.tokens_generated, result.stopped_reason

    def _generate_formal(
        self,
        prompt: str,
        initial: str = "",
    ) -> Tuple[str, bool, int, str]:
        """
        Generate grammar-constrained output using Synthesizer for tracking.

        Returns: (content, is_complete, tokens_generated, stopped_reason)
        """
        result = self.model.generate_constrained(
            prompt=prompt,
            initial=initial,
            max_tokens=self.formal_budget,
            grammar_name=self.grammar_name,
            temperature=self.temperature,
        )

        return (
            result.text,
            result.is_complete,
            result.tokens_generated,
            result.stopped_reason,
        )

    def generate(
        self,
        prompt: str,
        initial: str = "",
    ) -> EnvironmentResult:
        """
        Generate one think block followed by one formal block.

        The `initial` prefix is inserted at the beginning of the formal block,
        never before the thinking block.

        Args:
            prompt: Initial prompt/question
            initial: Initial formal text (partial expression/program)

        Returns:
            EnvironmentResult with all blocks and metadata
        """
        result = EnvironmentResult(grammar_name=self.grammar_name)

        if self.system_prompt:
            full_prompt = self.system_prompt + "\n\n" + prompt
        else:
            full_prompt = prompt

        # Open the think block in the prompt unless the model's chat template
        # already starts inside one (start_tokens_unconstrained yields <think>).
        think_prompt = full_prompt
        if not self._unconstrained_start_includes_think():
            think_prompt = f"{full_prompt}\n{self.THINK_OPEN}"

        thought, think_tokens, _ = self._generate_think(prompt=think_prompt, initial="")
        result.blocks.append(ThinkBlock(content=thought, tokens=think_tokens))
        result.total_tokens += think_tokens

        # Hand the formal pass the closed think block followed by the opening
        # <formal> tag — the two-block protocol build_system_prompt describes.
        formal_prompt = (
            f"{full_prompt}"
            f"\n{self.THINK_OPEN}{thought}{self.THINK_CLOSE}"
            f"\n<formal>"
        )

        try:
            content, is_complete, formal_tokens, formal_reason = self._generate_formal(
                prompt=formal_prompt,
                initial=initial,
            )
        except Exception as e:
            result.stopped_reason = f"error: {e}"
            return result

        result.blocks.append(
            FormalBlock(
                content=content,
                grammar_name=self.grammar_name,
                is_complete=is_complete,
                tokens=formal_tokens,
            )
        )
        result.total_tokens += formal_tokens
        result.stopped_reason = "complete" if is_complete else formal_reason
        return result


def _dedupe(tokens: List[str]) -> List[str]:
    seen = set()
    return [
        token for token in tokens if token and not (token in seen or seen.add(token))
    ]
