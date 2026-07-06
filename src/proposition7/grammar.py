from __future__ import annotations

import aufbau


class Grammar:
    def __init__(self, spec: str):
        self.spec = spec
        # Compile once; tokenize() reuses the synthesizer via set_input.
        self._synth = aufbau.Synthesizer(spec, "")

    def tokenize(self, text: str) -> list[str]:
        """Whitespace-split ``text`` into grammar tokens, checking that the
        running prefix stays parseable. ``[]`` if any prefix is rejected.

        Tokens are re-joined with single spaces (the grammar's token
        separator); the synthesizer is checked with ``status`` rather than
        advanced with ``feed``, which appends raw characters.
        """
        tokens: list[str] = []
        accumulated = ""
        for part in text.split():
            accumulated = f"{accumulated} {part}" if accumulated else part
            self._synth.set_input(accumulated)
            if self._synth.status() == "dead":
                return []
            tokens.append(part)
        return tokens
