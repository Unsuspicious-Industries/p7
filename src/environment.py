"""Two-stage reasoning environment: think once, then write formal output."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Union

from .grammars import get_grammar_info

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
        grammar_name: Name of the grammar (e.g., "stlc", "fun", "imp")
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


class ReasoningEnvironment:
    """
    Environment that performs exactly one think block and one formal block.
    
    Usage:
        from p7 import ConstrainedModel, GRAMMARS
        
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
        stop_on_complete: bool = True,
    ):
        """
        Initialize the reasoning environment.
        
        Args:
            model: A ConstrainedModel with grammar loaded
            grammar_name: Name of the grammar
            think_budget: Max tokens per think block
            formal_budget: Max tokens per formal block
            system_prompt: Custom system prompt (auto-generated if None)
        """
        self.model = model
        self.grammar_name = grammar_name
        self.think_budget = think_budget
        self.formal_budget = formal_budget
        self.stop_on_complete = stop_on_complete

        self.THINK_OPEN = self.model.think_open()
        self.THINK_CLOSE = self.model.think_close()
        self.formal_open = "<formal>"
        self.formal_close = "</formal>"

        # System prompt — generated after think tokens are resolved so the
        # prompt text references the model's actual reasoning tags.
        if self.model.allow_system_prompt():
            self.system_prompt = system_prompt or build_system_prompt(
                grammar_name,
                think_open=self.THINK_OPEN,
                think_close=self.THINK_CLOSE,
            )
        else:
            self.system_prompt = system_prompt or ""

        # Stop tokens for think mode
        self._think_stop = _dedupe(
            self.model.stop_tokens_unconstrained(grammar_name)
            + [self.THINK_CLOSE, self.formal_open, self.formal_close]
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
        temperature: float = 1.0,
    ) -> Tuple[str, int, str]:
        """
        Generate unconstrained thinking until </think> or <formal>.
        
        Returns: (content, tokens_generated, stopped_reason)
        """
        result = self.model.generate_unconstrained(
            prompt=prompt,
            max_tokens=self.think_budget,
            temperature=temperature,
            stop_tokens=self._think_stop,
            grammar_name=self.grammar_name,
        )
        
        content = result.text

        if content.startswith(self.THINK_OPEN):
            content = content[len(self.THINK_OPEN) :]

        for tag in [self.THINK_CLOSE, self.formal_open, self.formal_close]:
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
        think_temperature: float = 1.0,
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

        think_prompt = full_prompt
        if not self._unconstrained_start_includes_think():
            think_prompt += f"\n{self.THINK_OPEN}"

        thought, think_tokens, _ = self._generate_think(
            prompt=think_prompt,
            temperature=think_temperature,
        )
        result.blocks.append(ThinkBlock(content=thought, tokens=think_tokens))
        result.total_tokens += think_tokens

        formal_prompt = (
            f"{full_prompt}\n{self.THINK_OPEN}{thought}{self.THINK_CLOSE}\n"
            "Now write only the final program in the formal block. "
            "Continue exactly from any prefix that follows <formal>; do not repeat it.\n"
            f"{self.formal_open}"
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
    return [token for token in tokens if token and not (token in seen or seen.add(token))]
