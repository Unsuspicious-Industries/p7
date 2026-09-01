"""How a sampled token is spelled for the grammar.

Separate from `llm` on purpose: this is a decision about tokens and text,
it needs no tensor library, and it is the part of the constrained decoder
most worth testing directly.
"""


def candidate_spellings(token: str, prefix: str) -> list[str]:
    """The spellings of `token` to offer the grammar, in priority order.

    raw          - preserves the model's own spacing ('let' + ' x' -> 'let x')
    lstripped    - continues the token in progress ('4' + ' 3' -> '43')
    space-joined - bare tokenizers with no leading space, where a grammar token
                   boundary is still needed ('x' then 'y' -> 'x y', not 'xy')

    The first admissible one wins, so a fallback never overrides a spelling the
    model actually emitted.

    The lstripped spelling is the dangerous one. It says "glue this to what came
    before", which is right inside a numeric literal and wrong across a boundary
    the model itself asked for. Offered unconditionally it let ' long' be
    re-spelled 'long' and welded to the preceding identifier, so 'long long rev'
    decoded as 'longlongrev' and 'size_t i' as 'size_ti'. Worse, an identifier is
    an absorbing state in these grammars -- every alphabetic suffix extends it,
    so the mask never rejects and the decode runs to max_tokens emitting
    'chingingingingching...'. That is what made the mixed arm fail 8 times in 11.

    So fusion is withheld when it would weld a word onto a word: the accepted
    prefix ends in a word character and the stripped token starts a new one.
    Digits and operators, the cases the fallback exists for, are untouched. When
    every spelling is refused the caller rejects the token and the model samples
    another, which is the honest outcome -- silently re-spelling its output is
    how the lexical structure got lost in the first place.
    """
    candidates: list[str] = [token] if token else []
    stripped = token.lstrip()
    if stripped and stripped not in candidates:
        welds_words = (
            prefix
            and (prefix[-1].isalnum() or prefix[-1] == "_")
            and (stripped[0].isalpha() or stripped[0] == "_")
        )
        if not welds_words:
            candidates.append(stripped)
    if stripped and prefix and not prefix[-1].isspace() and not token[:1].isspace():
        joined = " " + stripped
        if joined not in candidates:
            candidates.append(joined)
    return candidates
