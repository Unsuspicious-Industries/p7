from types import SimpleNamespace

import torch

import proposition7


class FakeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 3
    sep_token = None
    sep_token_id = None
    pad_token = None
    pad_token_id = None
    bos_token = None
    bos_token_id = None
    unk_token = "<unk>"
    unk_token_id = 99
    additional_special_tokens = ["<role>"]
    additional_special_tokens_ids = [5]

    id_to_token = {
        0: "bad",
        1: "x",
        2: "y",
        3: "<eos>",
        4: "<eog>",
        5: "<role>",
    }
    token_to_id = {token: token_id for token_id, token in id_to_token.items()}

    def __call__(self, text, return_tensors=None):
        del text, return_tensors
        return SimpleNamespace(input_ids=torch.tensor([[0]], dtype=torch.long))

    def decode(self, ids, **kwargs):
        del kwargs
        return "".join(self.id_to_token[int(token_id)] for token_id in ids)

    def encode(self, token, add_special_tokens=False):
        del add_special_tokens
        return [self.token_to_id[token]] if token in self.token_to_id else []

    def convert_tokens_to_ids(self, token):
        return self.token_to_id.get(token, self.unk_token_id)

    def convert_ids_to_tokens(self, token_id):
        return self.id_to_token.get(int(token_id))


class FakeModel:
    def __init__(self, token_ids):
        self.token_ids = list(token_ids)
        self.calls = 0
        self.generation_config = SimpleNamespace(eos_token_id=3, eog_token_id=4)
        self.config = SimpleNamespace(eos_token_id=3)

    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        del past_key_values, use_cache
        token_id = self.token_ids[min(self.calls, len(self.token_ids) - 1)]
        self.calls += 1
        logits = torch.full(
            (1, 1, 6),
            -1000.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits[0, 0, token_id] = 1000.0
        return SimpleNamespace(logits=logits, past_key_values=None)


def make_model(token_ids, grammar="start ::= 'x' 'y'"):
    return proposition7.ConstrainedModel(
        FakeModel(token_ids),
        FakeTokenizer(),
        grammar,
        device="cpu",
    )


def test_constrained_generation_uses_torch_sampling_path():
    result = make_model([1, 2]).generate_constrained(max_tokens=3)

    assert result.text == "x y"
    assert result.is_complete is True
    assert result.tokens_generated == 2


def test_constrained_generation_stops_on_eos_token_id():
    result = make_model([3], grammar="start ::= 'x'").generate_constrained(
        initial="x",
        max_tokens=3,
    )

    assert result.text == "x"
    assert result.is_complete is True
    assert result.tokens_generated == 0
    assert result.stopped_reason == "complete"


def test_constrained_generation_stops_on_generation_config_eog_token_id():
    result = make_model([4], grammar="start ::= 'x'").generate_constrained(
        initial="x",
        max_tokens=3,
    )

    assert result.text == "x"
    assert result.is_complete is True
    assert result.tokens_generated == 0
    assert result.stopped_reason == "complete"


def test_constrained_generation_stops_on_role_token_before_parse_complete():
    result = make_model([5], grammar="start ::= 'x'").generate_constrained(
        max_tokens=3,
    )

    assert result.text == ""
    assert result.is_complete is False
    assert result.tokens_generated == 0
    assert result.stopped_reason == "stop_token:<role>"


def test_unconstrained_generation_stops_on_eos_token_id():
    result = make_model([3]).generate_unconstrained(max_tokens=3)

    assert result.text == ""
    assert result.tokens_generated == 0
    assert result.stopped_reason == "stop_token:<eos>"
