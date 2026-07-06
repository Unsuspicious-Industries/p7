from types import SimpleNamespace

import torch

import proposition7


class FakeTokenizer:
    """Each id maps to a whole grammar token (word/punctuation), mirroring
    how a real subword tokenizer would hand generate_constrained() one
    already-spelled string per step -- not single characters."""

    eos_token = "<eos>"
    eos_token_id = 6
    sep_token = None
    sep_token_id = None
    pad_token = None
    pad_token_id = None
    bos_token = None
    bos_token_id = None
    unk_token = "<unk>"
    unk_token_id = 99
    additional_special_tokens: list[str] = []
    additional_special_tokens_ids: list[int] = []

    id_to_token = {
        0: "beep",
        1: ":",
        2: "Fizz",
        3: "+",
        4: "boop",
        5: "Buzz",
        6: "<eos>",
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
    """Always proposes the next token in a fixed, hand-picked sequence:
    beep : Fizz + boop : Buzz -- a Concat of two TypedValues with
    *different* declared types (Fizz vs Buzz). Syntactically a complete
    Concat expression; semantically ill-typed (toy.auf's `cat` rule
    requires both sides to share one type)."""

    def __init__(self, token_ids):
        self.token_ids = list(token_ids)
        self.calls = 0
        self.generation_config = SimpleNamespace(eos_token_id=6, eog_token_id=None)
        self.config = SimpleNamespace(eos_token_id=6)

    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        del past_key_values, use_cache
        token_id = self.token_ids[min(self.calls, len(self.token_ids) - 1)]
        self.calls += 1
        logits = torch.full((1, 1, 7), -1000.0, dtype=torch.float32, device=input_ids.device)
        logits[0, 0, token_id] = 1000.0
        return SimpleNamespace(logits=logits, past_key_values=None)


_MISMATCHED_CONCAT = [0, 1, 2, 3, 4, 1, 5]  # beep : Fizz + boop : Buzz


def make_model(token_ids):
    return proposition7.ConstrainedModel(
        FakeModel(token_ids),
        FakeTokenizer(),
        proposition7.get_grammar("toy"),
        device="cpu",
    )


def test_semantic_mode_rejects_a_type_mismatched_concat():
    result = make_model(_MISMATCHED_CONCAT).generate_constrained(
        grammar_name="toy", max_tokens=len(_MISMATCHED_CONCAT) + 1
    )

    assert result.is_complete is False
    assert result.text != "beep:Fizz+boop:Buzz"


def test_syntax_only_mode_accepts_the_same_type_mismatched_concat():
    # Exactly the target length: toy.auf's Concat is right-recursive, so a
    # complete Concat is still a *live* prefix of a longer one -- one extra
    # step would let the mask legally extend past it with a trailing '+'.
    result = make_model(_MISMATCHED_CONCAT).generate_constrained(
        grammar_name="toy", max_tokens=len(_MISMATCHED_CONCAT), syntax_only=True
    )

    assert result.text == "beep:Fizz+boop:Buzz"
    assert result.is_complete is True
