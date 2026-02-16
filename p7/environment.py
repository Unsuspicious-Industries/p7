"""
Environment for structured reasoning with LLMs.

Allows LLMs to switch between:
- <think>...</think>: Unconstrained chain-of-thought reasoning
- <{grammar}>...</{grammar}>: Grammar-constrained guaranteed output

This enables CoT reasoning while ensuring the final output is well-typed.
Grammar-independent: tags are derived from grammar name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Tuple, Dict, Any

from .grammars import GRAMMARS, get_grammar, get_grammar_info


class Mode(Enum):
    """Current generation mode."""
    THINK = "think"
    GRAMMAR = "grammar"  # Grammar-constrained (tag varies by grammar)


@dataclass
class ThinkBlock:
    """A block of unconstrained reasoning."""
    content: str
    
    def __str__(self) -> str:
        return f"<think>{self.content}</think>"


@dataclass 
class GrammarBlock:
    """A block of grammar-constrained output."""
    content: str
    grammar_name: str
    is_complete: bool
    
    def __str__(self) -> str:
        return f"<{self.grammar_name}>{self.content}</{self.grammar_name}>"


@dataclass
class EnvironmentResult:
    """Result from environment generation."""
    blocks: List[ThinkBlock | GrammarBlock] = field(default_factory=list)
    total_tokens: int = 0
    stopped_reason: str = "max_tokens"
    grammar_name: str = ""
    
    @property
    def think_blocks(self) -> List[ThinkBlock]:
        return [b for b in self.blocks if isinstance(b, ThinkBlock)]
    
    @property
    def grammar_blocks(self) -> List[GrammarBlock]:
        return [b for b in self.blocks if isinstance(b, GrammarBlock)]
    
    @property
    def final_output(self) -> Optional[GrammarBlock]:
        """Get the last grammar block (the final output)."""
        blocks = self.grammar_blocks
        return blocks[-1] if blocks else None
    
    @property
    def all_thoughts(self) -> str:
        """Concatenate all thinking."""
        return "\n".join(b.content for b in self.think_blocks)
    
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
) -> str:
    """
    Procedurally generate a system prompt for the given grammar.
    
    Args:
        grammar_name: Name of the grammar (e.g., "stlc", "fun", "imp")
        task_description: Optional task-specific description
        include_examples: Whether to include syntax examples
        
    Returns:
        System prompt string
    """
    info = get_grammar_info(grammar_name)
    
    lines = [
        f"You are a reasoning assistant that produces well-typed {info['short']}.",
        "",
        "You can use two modes:",
        "- <think>...</think>: Free-form reasoning. Think step by step.",
        f"- <{grammar_name}>...</{grammar_name}>: Produce the final well-typed output. This is grammar-constrained.",
        "",
        "Process:",
        "1. Use <think> to reason about the problem",
        f"2. When ready, use <{grammar_name}> to produce typed output",
        "3. The output must be syntactically and type-correct",
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
    Environment that allows LLMs to reason with CoT then produce typed output.
    
    Grammar-independent: uses grammar name for tags (e.g., <stlc>, <fun>).
    
    Usage:
        from p7 import ConstrainedModel, GRAMMARS
        
        model = ConstrainedModel.from_pretrained("...", grammar=get_grammar("stlc"))
        env = ReasoningEnvironment(model, grammar_name="stlc")
        
        result = env.generate(
            prompt="Create a function that takes an Int and returns it",
            initial="λx:",
        )
        
        print(result.all_thoughts)  # CoT reasoning
        print(result.final_output)  # Well-typed output
    """
    
    def __init__(
        self,
        model,  # ConstrainedModel
        grammar_name: str,
        think_budget: int = 200,
        formal_budget: int = 100,
        system_prompt: Optional[str] = None,
        stop_on_complete: bool = False,
    ):
        """
        Initialize the reasoning environment.
        
        Args:
            model: A ConstrainedModel with grammar loaded
            grammar_name: Name of the grammar (for tags and prompts)
            think_budget: Max tokens per think block
            formal_budget: Max tokens per formal block
            system_prompt: Custom system prompt (auto-generated if None)
        """
        self.model = model
        self.grammar_name = grammar_name
        self.think_budget = think_budget
        self.formal_budget = formal_budget
        self.stop_on_complete = stop_on_complete

        # System prompt (procedurally generated or custom)
        if self.model.allow_system_prompt():
            self.system_prompt = system_prompt or build_system_prompt(grammar_name)
        else:
            self.system_prompt = system_prompt or ""

        self.THINK_OPEN = self.model.think_open()
        self.THINK_CLOSE = self.model.think_close()
        self.grammar_open = f"<{grammar_name}>"
        self.grammar_close = f"</{grammar_name}>"

        # Stop tokens for think mode
        self._think_stop = self.model.stop_tokens_unconstrained(grammar_name)
    
    def _generate_think(
        self,
        prompt: str,
        on_token: Optional[Callable[[str, int], None]] = None,
        top_k: Optional[int] = None,
        temperature: float = 1.0,
    ) -> Tuple[str, str, int]:
        """
        Generate unconstrained thinking until </think> or <grammar>.
        
        Returns: (content, stop_tag, tokens_generated)
        """
        result = self.model.generate_unconstrained(
            prompt=prompt,
            max_tokens=self.think_budget,
            top_k=top_k,
            temperature=temperature,
            on_token=on_token,
            stop_tokens=self._think_stop,
            grammar_name=self.grammar_name,
        )
        
        content = result.text
        stop_tag = ""
        
        # Check what stopped us - check all possible tags
        for tag in [self.THINK_CLOSE, self.grammar_open, self.grammar_close]:
            if tag in content:
                idx = content.find(tag)
                stop_tag = tag
                content = content[:idx]
                break
        
        return content, stop_tag, result.tokens_generated
    
    def _generate_formal(
        self,
        prompt: str,
        initial: str = "",
        on_token: Optional[Callable[[str, int], None]] = None,
        stop_on_complete: Optional[bool] = None,
        logit_filter: Optional[Callable[[List[float], str], List[float]]] = None,
    ) -> Tuple[str, bool, int]:
        """
        Generate grammar-constrained output.
        
        Returns: (content, is_complete, tokens_generated)
        """
        should_stop_on_complete = self.stop_on_complete if stop_on_complete is None else stop_on_complete
        if should_stop_on_complete:
            result = self.model.until_complete(
                prompt=prompt,
                initial=initial,
                max_tokens=self.formal_budget,
                on_token=on_token,
                grammar_name=self.grammar_name,
                logit_filter=logit_filter,
            )
        else:
            result = self.model.generate(
                prompt=prompt,
                initial=initial,
                max_tokens=self.formal_budget,
                on_token=on_token,
                grammar_name=self.grammar_name,
                logit_filter=logit_filter,
            )
        
        return result.text, result.is_complete, result.tokens_generated
    
    def generate(
        self,
        prompt: str,
        initial: str = "",
        max_blocks: int = 10,
        start_thinking: bool = True,
        on_think_token: Optional[Callable[[str, int], None]] = None,
        on_formal_token: Optional[Callable[[str, int], None]] = None,
        on_mode_switch: Optional[Callable[[Mode, str], None]] = None,
        stop_on_complete: Optional[bool] = None,
        think_top_k: Optional[int] = None,
        think_temperature: float = 1.0,
        formal_logit_filter: Optional[Callable[[List[float], str], List[float]]] = None,
    ) -> EnvironmentResult:
        """
        Generate with alternating think/grammar blocks.
        
        Args:
            prompt: Initial prompt/question
            initial: Initial text for grammar blocks (partial expression)
            max_blocks: Maximum number of blocks to generate
            start_thinking: Whether to start with <think> mode
            on_think_token: Callback for each token in think mode
            on_formal_token: Callback for each token in grammar mode
            on_mode_switch: Callback when switching modes (mode, tag)
            
        Returns:
            EnvironmentResult with all blocks and metadata
        """
        result = EnvironmentResult(grammar_name=self.grammar_name)
        
        # Build full prompt with system
        if self.system_prompt:
            full_prompt = self.system_prompt + "\n\n" + prompt
        else:
            full_prompt = prompt
        
        current_mode = Mode.THINK if start_thinking else Mode.GRAMMAR
        accumulated = full_prompt
        
        if start_thinking:
            accumulated += f"\n{self.THINK_OPEN}"
        else:
            accumulated += f"\n{self.grammar_open}"
        
        for block_idx in range(max_blocks):
            if on_mode_switch:
                tag = self.THINK_OPEN if current_mode == Mode.THINK else self.grammar_open
                on_mode_switch(current_mode, tag)
            
            if current_mode == Mode.THINK:
                content, stop_tag, tokens = self._generate_think(
                    prompt=accumulated,
                    on_token=on_think_token,
                    top_k=think_top_k,
                    temperature=think_temperature,
                )
                
                result.blocks.append(ThinkBlock(content=content))
                result.total_tokens += tokens
                accumulated += content
                
                if stop_tag == self.THINK_CLOSE:
                    accumulated += self.THINK_CLOSE
                elif stop_tag == self.grammar_open:
                    accumulated += self.grammar_open
                else:
                    # No clear transition, close think and start grammar
                    accumulated += self.THINK_CLOSE
                    accumulated += f"\n{self.grammar_open}"
                
                current_mode = Mode.GRAMMAR
                if stop_tag != self.grammar_open:
                    accumulated += f"\n{self.grammar_open}"
            
            else:  # GRAMMAR mode
                # Only use initial on first grammar block
                use_initial = initial if not result.grammar_blocks else ""
                
                try:
                    content, is_complete, tokens = self._generate_formal(
                        prompt=accumulated,
                        initial=use_initial,
                        on_token=on_formal_token,
                        stop_on_complete=stop_on_complete,
                        logit_filter=formal_logit_filter,
                    )
                except Exception as e:
                    # Grammar generation failed - stop here
                    result.stopped_reason = f"error: {e}"
                    break
                
                result.blocks.append(GrammarBlock(
                    content=content,
                    grammar_name=self.grammar_name,
                    is_complete=is_complete,
                ))
                result.total_tokens += tokens
                accumulated += content + self.grammar_close
                
                if is_complete:
                    result.stopped_reason = "complete"
                    break
                else:
                    # Not complete, need more thinking
                    current_mode = Mode.THINK
                    accumulated += f"\n{self.THINK_OPEN}"
        
        if result.stopped_reason != "complete":
            result.stopped_reason = "max_blocks"
        
        return result
