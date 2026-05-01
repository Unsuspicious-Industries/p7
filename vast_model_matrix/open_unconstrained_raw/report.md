# Benchmark Report

Input rows: 940

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

