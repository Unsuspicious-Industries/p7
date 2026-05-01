# Benchmark Report

Input rows: 3206

## Key metric

- `exact_rate`: exact text match with expected answer
- `parse_error_rate`: output not parseable by target grammar
- `non_completable_rate`: output is a dead grammar prefix; this should be 0 for constrained decoding
- `incomplete_rate`: parseable but not complete
- `semantic_mismatch_rate`: parseable/complete but wrong answer
- `timeout_rate`: job hit the configured timeout
- `other_error_rate`: uncategorized runtime/model errors

## Constrained vs unconstrained delta

Uses `unconstrained_raw` as the baseline when present.

- local Qwen/Qwen3.5-0.8B fun vs unconstrained_raw: exact delta 0.0 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-0.8B imp vs unconstrained_raw: exact delta -10.71 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-0.8B stlc vs unconstrained_raw: exact delta -3.03 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-2B fun vs unconstrained_raw: exact delta 0.0 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-2B imp vs unconstrained_raw: exact delta -7.14 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-2B stlc vs unconstrained_raw: exact delta -6.06 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-4B fun vs unconstrained_raw: exact delta 0.0 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-4B imp vs unconstrained_raw: exact delta -10.71 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-4B stlc vs unconstrained_raw: exact delta 0.0 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-4B-Base fun vs unconstrained_raw: exact delta 0.0 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-4B-Base imp vs unconstrained_raw: exact delta -14.29 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-4B-Base stlc vs unconstrained_raw: exact delta -9.09 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-9B fun vs unconstrained_raw: exact delta -18.18 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-9B imp vs unconstrained_raw: exact delta -14.29 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-9B stlc vs unconstrained_raw: exact delta -30.3 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-9B-Base fun vs unconstrained_raw: exact delta -9.09 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-9B-Base imp vs unconstrained_raw: exact delta -10.71 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-9B-Base stlc vs unconstrained_raw: exact delta -18.18 pts, parse-error reduction 0.0 pts
- local deepseek-ai/DeepSeek-R1-Distill-Qwen-7B fun vs unconstrained_raw: exact delta 0.0 pts, parse-error reduction 0.0 pts
- local deepseek-ai/DeepSeek-R1-Distill-Qwen-7B imp vs unconstrained_raw: exact delta 0.0 pts, parse-error reduction 0.0 pts
- local deepseek-ai/DeepSeek-R1-Distill-Qwen-7B stlc vs unconstrained_raw: exact delta 0.0 pts, parse-error reduction 0.0 pts
- local microsoft/Phi-4-mini-instruct fun vs unconstrained_raw: exact delta -3.03 pts, parse-error reduction 0.0 pts
- local microsoft/Phi-4-mini-instruct imp vs unconstrained_raw: exact delta -3.57 pts, parse-error reduction 0.0 pts
- local microsoft/Phi-4-mini-instruct stlc vs unconstrained_raw: exact delta 0.0 pts, parse-error reduction 0.0 pts
