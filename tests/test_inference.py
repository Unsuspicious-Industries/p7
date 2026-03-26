import p7.inference as inference_module


class FakeTypedSampler:
    def __init__(self, grammar, vocab, logit_fn):
        self.grammar = grammar
        self.vocab = vocab
        self.logit_fn = logit_fn
        self.text = ""
        self.set_input_calls = []
        self.feed_calls = []

    def set_input(self, text):
        self.set_input_calls.append(text)
        self.text = text

    def feed(self, token):
        self.feed_calls.append(token)
        if token == "bad":
            raise RuntimeError(f"Parse failed for input='{self.text}' token='{token}'")
        self.text += token

    def infer_greedy(self, k=1, pre_top_k=None):
        if self.text == "seed":
            return "bad"
        return None

    def infer_text(self, k=10, pre_top_k=None):
        if self.text == "seed":
            return ["bad", "good"][:k]
        return []

    def current_text(self):
        return self.text

    def is_complete(self):
        return self.text.endswith("good")


def test_generate_falls_back_after_parse_failed_candidate(monkeypatch):
    monkeypatch.setattr(inference_module, "TypedSampler", FakeTypedSampler)

    streamed = []
    result, sampler = inference_module.generate(
        grammar="fake",
        vocab=["bad", "good"],
        logit_fn=lambda: [1.0, 0.0],
        initial="seed",
        max_tokens=1,
        on_token=lambda token, step: streamed.append((token, step)),
    )

    assert getattr(sampler, "set_input_calls") == ["seed"]
    assert getattr(sampler, "feed_calls") == ["bad", "good"]
    assert result.text == "seedgood"
    assert result.is_complete is True
    assert result.tokens_generated == 1
    assert result.stopped_reason == "max_tokens"
    assert streamed == [("good", 0)]


class FeedOnlySampler:
    def __init__(self, grammar, vocab, logit_fn):
        self.grammar = grammar
        self.vocab = vocab
        self.logit_fn = logit_fn
        self.text = ""
        self.feed_calls = []

    def set_input(self, text):
        self.text = text

    def feed(self, token):
        self.feed_calls.append(token)
        if token == "bad":
            raise RuntimeError("Parse failed for token='bad'")
        self.text += token

    def current_text(self):
        return self.text

    def is_complete(self):
        return self.text.endswith("ok")


def test_select_and_feed_tries_full_vocab_after_pre_top_k(monkeypatch):
    monkeypatch.setattr(inference_module, "TypedSampler", FeedOnlySampler)

    result, sampler = inference_module.generate(
        grammar="fake",
        vocab=["bad", "ok", "zzz"],
        logit_fn=lambda: [10.0, 9.0, 1.0],
        initial="",
        max_tokens=1,
        pre_top_k=1,
        use_feed_only=True,
    )

    assert getattr(sampler, "feed_calls") == ["bad", "ok"]
    assert result.text == "ok"
    assert result.is_complete is True


class NonParseErrorSampler:
    def __init__(self, grammar, vocab, logit_fn):
        self.vocab = vocab
        self.logit_fn = logit_fn
        self.text = ""

    def set_input(self, text):
        self.text = text

    def feed(self, token):
        raise RuntimeError("database is down")

    def current_text(self):
        return self.text

    def is_complete(self):
        return False


def test_non_parse_runtime_error_propagates_as_type_error(monkeypatch):
    monkeypatch.setattr(inference_module, "TypedSampler", NonParseErrorSampler)

    result, _ = inference_module.generate(
        grammar="fake",
        vocab=["x"],
        logit_fn=lambda: [1.0],
        max_tokens=1,
        use_feed_only=True,
    )

    assert result.stopped_reason.startswith("type_error:")
    assert "database is down" in result.stopped_reason


class ContextVarSampler:
    def __init__(self, grammar, vocab, logit_fn):
        self.vocab = vocab
        self.logit_fn = logit_fn
        self.text = ""
        self.feed_calls = []

    def set_input(self, text):
        self.text = text

    def feed(self, token):
        self.feed_calls.append(token)
        if token != "x":
            raise RuntimeError("Parse failed for wrong token")
        self.text += token

    def infer_text(self, k=10, pre_top_k=None):
        return []

    def current_text(self):
        return self.text

    def is_complete(self):
        return self.text.endswith("x")


def test_feed_only_path_accepts_context_like_token_without_completion_set(monkeypatch):
    monkeypatch.setattr(inference_module, "TypedSampler", ContextVarSampler)

    result, sampler = inference_module.generate(
        grammar="fake",
        vocab=["x", "y"],
        logit_fn=lambda: [5.0, 1.0],
        initial="lambda x.",
        max_tokens=1,
        use_feed_only=True,
    )

    assert getattr(sampler, "feed_calls") == ["x"]
    assert result.text == "lambda x.x"
    assert result.is_complete is True
