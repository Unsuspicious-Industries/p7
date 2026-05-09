
def test_tokenization():
    from proposition7.grammar import Grammar

    grammar = Grammar("start ::= 'x' 'y' | 'z'")
    assert grammar.tokenize("x y") == ["x", "y"]
    assert grammar.tokenize("z") == ["z"]
    assert grammar.tokenize("a b") == []