# Benchmark Report

Input rows: 3384

## Key metric

- `exact_rate`: exact text match with expected answer
- `parse_error_rate`: output not parseable by target grammar
- `non_completable_rate`: output is a dead grammar prefix; this should be 0 for constrained decoding
- `incomplete_rate`: parseable but not complete
- `semantic_mismatch_rate`: parseable/complete but wrong answer
- `timeout_rate`: job hit the configured timeout
- `other_error_rate`: uncategorized runtime/model errors

## Constrained vs unconstrained delta

- local Ministral-3-8B-Instruct-2512-GGUF fun: exact delta 0.0 pts, parse-error reduction 0.0 pts
- local Ministral-3-8B-Instruct-2512-GGUF imp: exact delta 0.0 pts, parse-error reduction 0.0 pts
- local Ministral-3-8B-Instruct-2512-GGUF stlc: exact delta 0.0 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-0.8B fun: exact delta 0.0 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-0.8B imp: exact delta 0.0 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-0.8B stlc: exact delta -3.03 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-1.5B fun: exact delta 0.0 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-1.5B imp: exact delta 0.0 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-1.5B stlc: exact delta 0.0 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-2B fun: exact delta -3.03 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-2B imp: exact delta -3.57 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-2B stlc: exact delta -6.06 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-4B fun: exact delta 0.0 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-4B imp: exact delta 0.0 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-4B stlc: exact delta -18.18 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-4B-Base fun: exact delta 0.0 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-4B-Base imp: exact delta 0.0 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-4B-Base stlc: exact delta -30.3 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-9B fun: exact delta 0.0 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-9B imp: exact delta 0.0 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-9B stlc: exact delta -36.36 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-9B-Base fun: exact delta 0.0 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-9B-Base imp: exact delta -7.14 pts, parse-error reduction 0.0 pts
- local Qwen/Qwen3.5-9B-Base stlc: exact delta -39.39 pts, parse-error reduction 0.0 pts
- local deepseek-ai/DeepSeek-R1-Distill-Qwen-7B fun: exact delta 0.0 pts, parse-error reduction 0.0 pts
- local deepseek-ai/DeepSeek-R1-Distill-Qwen-7B imp: exact delta 0.0 pts, parse-error reduction 0.0 pts
- local deepseek-ai/DeepSeek-R1-Distill-Qwen-7B stlc: exact delta 0.0 pts, parse-error reduction 0.0 pts
- local microsoft/Phi-4-mini-instruct fun: exact delta -15.15 pts, parse-error reduction 0.0 pts
- local microsoft/Phi-4-mini-instruct imp: exact delta -10.71 pts, parse-error reduction 0.0 pts
- local microsoft/Phi-4-mini-instruct stlc: exact delta -15.15 pts, parse-error reduction 0.0 pts
