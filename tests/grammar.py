
def test_tokenization():
    from p7.grammar import Grammar

    grammar = Grammar("start ::= 'x' 'y' | 'z'")
    assert grammar.tokenize("x y") == ["x", "y"]
    assert grammar.tokenize("z") == ["z"]
    assert grammar.tokenize("a b") == []