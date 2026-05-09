from __future__ import annotations

import aufbau


class Grammar:
    def __init__(self, spec: str):
        self.spec = spec

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        synthesizer = aufbau.Synthesizer(self.spec, "")
        for part in text.split():
            try:
                synthesizer.feed(part)
            except RuntimeError:
                return []
            tokens.append(part)
        return tokens
