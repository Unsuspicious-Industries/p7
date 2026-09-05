"""Harmony final-channel extraction (pure function -- no model weights)."""

from proposition7.models.harmony import extract_harmony_final


def test_analysis_then_final_keeps_only_final_content():
    text = (
        "<|channel|>analysis<|message|>the user wants a map over a list, "
        "I should use List.map<|end|><|start|>assistant<|channel|>final"
        "<|message|>let f = List.map (fun x -> x + 1)<|return|>"
    )
    assert extract_harmony_final(text) == "let f = List.map (fun x -> x + 1)"


def test_last_final_marker_wins():
    text = (
        "<|channel|>final<|message|>draft answer<|end|>"
        "<|start|>assistant<|channel|>final<|message|>real answer<|return|>"
    )
    assert extract_harmony_final(text) == "real answer"


def test_cut_at_first_end_marker_of_any_kind():
    assert extract_harmony_final("<|channel|>final<|message|>x + 1<|call|>tail") == "x + 1"
    assert extract_harmony_final("<|channel|>final<|message|>x + 1<|end|>tail") == "x + 1"
    assert extract_harmony_final("<|channel|>final<|message|>x + 1<|return|>tail") == "x + 1"


def test_unterminated_final_channel_keeps_the_tail():
    # Generation may hit max_tokens before emitting an end marker.
    text = "<|channel|>analysis<|message|>hmm<|end|><|channel|>final<|message|>int add("
    assert extract_harmony_final(text) == "int add("


def test_text_without_harmony_markup_is_unchanged():
    assert extract_harmony_final("let x = 3;") == "let x = 3;"
    assert extract_harmony_final("") == ""


def test_analysis_only_output_is_unchanged():
    # No final channel at all: nothing safe to extract, leave it for the
    # parser to reject honestly.
    text = "<|channel|>analysis<|message|>I am not sure what to do here"
    assert extract_harmony_final(text) == text
