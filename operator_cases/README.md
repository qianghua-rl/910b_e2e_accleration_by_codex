# Operator Optimization Cases

This directory is the few-shot library for Phase 3 operator work. Cases are
evidence-backed records, not code templates to copy blindly. A candidate must
match the operator contract, device, dtype, layout, execution mode, and memory
budget before a case is used as a reference.

Each JSON case records:

- the complete input/output contract and relevant shapes;
- the mechanism being tested and the expected cost model;
- correctness and performance evidence paths;
- accepted, rejected, or blocked status and the exact gate that decided it.

Use the selector from the repository root:

```bash
python scripts/operator_case.py list
python scripts/operator_case.py search --operator gdn_prefill --mode prefill --device Ascend910B3
python scripts/operator_case.py verify
```

The optimization loop is deliberately staged:

1. retrieve similar accepted and rejected cases;
2. write the candidate contract and Amdahl budget;
3. compare one reference implementation and one candidate on identical tensors;
4. check output, state, next-token logits, memory, and kernel/wrapper time;
5. run the E100 batch16 first-turn replay;
6. expand to 32 and 128 tasks only after the token and performance gates pass.

Rejected cases are retained because they teach the model when a locally fast
kernel is invalid, too large, or slower in the real workload.
