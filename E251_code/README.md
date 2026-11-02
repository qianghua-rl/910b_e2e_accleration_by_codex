# E251 Code Review Bundle

This folder collects the source path and selected evidence for the E250/E251 HF workflow, plus the matched vLLM-Ascend comparison. It is intended for code review and mentor discussion. Source files are copied from the repository's `scripts/` directory; they are not a separate implementation.

## What E250 Changes

The ordinary HF two-turn tool workflow rebuilds and tokenizes the full conversation for the second generation. E250 keeps per-row model state at first-turn EOS, verifies that the second-turn prompt has the cached prefix, and appends only the newly added assistant/tool suffix. Its GDN continuation handles unequal suffix lengths with history-aware chunk64 recurrence and whole-suffix causal convolution. If any prefix in a B16 group does not match, the whole group is recomputed from the full second-turn prompt.

## Source Map

| File | Role |
|---|---|
| `scripts/e251_full_workflow.py` | Resident full128 workflow, per-row EOS cache capture, prefix validation, B16 reuse or whole-batch fresh fallback |
| `scripts/e250_chunk_history.py` | E250 T>1 history-aware chunk64 recurrent suffix path |
| `scripts/e248_vector_history_conv.py` | Whole-suffix causal convolution and per-row historical-window state commit |
| `scripts/e176_workflow_bench.py` | First-turn generation, tool execution, second-turn assembly and timing boundaries |
| `scripts/e176_history_runtime.py` | Batched cache continuation and decode loop |
| `scripts/e168_batched_history.py` | Per-row cache pack/unpack and history runtime helpers |
| `scripts/e168_ragged_history_gate.py` | Model loading and fused norm/recurrent setup used by the workflow |
| `scripts/e166_batch16_row_cache_workflow_gate.py` | Frozen B16 task order and row-cache helper; imports the legacy hybrid-cache module |
| `scripts/nonaligned_hybrid_cache.py` | Import dependency of the E166 helper; not the E250 execution path |
| `scripts/recurrent_single_step.py` | Existing single-token GDN recurrent primitive used for T=1 and reference updates |
| `scripts/agent_protocol.py` | Prompt, tool-call parser, Python tool execution and answer helpers |
| `scripts/scoring_v2.py`, `scripts/scoring_v3.py` | Scoring and v3 time-answer normalization |
| `scripts/audit_e251.py` | Reopens raw batches and checks trajectory, tools, scorer, memory and fallback routes |
| `scripts/e251_rerun_abba.py`, `scripts/audit_e251_ab.py` | ABBA orchestration and independent aggregate decision for the 2026-11-02 rerun |
| `scripts/e251_full_ab.py` | Original E251 ABBA driver retained as historical protocol reference |
| `scripts/vllm_agent_bench.py` | vLLM-Ascend Agent runner; supports explicit task order for a matched comparison |
| `scripts/audit_e252.py`, `scripts/audit_vllm_e251_fair.py` | General vLLM quality/context audit and direct comparison against the E251 HF candidate |

The code and review material are separated from model weights and task datasets. Runtime inputs remain at the repository root: `model/qwen3.5-27b/`, `data/gsm8k_128.jsonl`, and the frozen E100 reference files under `results/`.

## Evidence

- `docs/mentor_progress_e250_vs_vllm_20261102.md`: progress report and operator-attempt review.
- `docs/e251_contract.md`: frozen E251 workload and acceptance requirements.
- `evidence/decision.json`, `evidence/independent_audit.json`: 2026-11-02 E180/E250/E250/E180 rerun summary and aggregate audit.
- `evidence/hf_abba/`: per-trial `gate.json` and raw-workflow audit summaries for all four trials.
- `evidence/vllm_fair_e251_order_20261102.jsonl`: raw single-run vLLM output using E251 task order and B16/tool-workers=1.
- `evidence/vllm_fair_e251_order_20261102.audit.json`: direct comparison against the E251 candidate workload. It confirms input alignment but rejects quality/work completion because task304 made no tool call.

The complete batch JSON/transcript files remain in `results/e251/abba_rerun_20261102/`; source hashes and trial commands are recorded in its `decision.json`. The package copies make review convenient but do not replace those original artifacts. The source hashes in the HF decision match the bundled copies of `e251_full_workflow.py`, `e251_full_ab.py`, `audit_e251.py`, `audit_e251_ab.py`, `e250_chunk_history.py`, `e248_vector_history_conv.py`, `e176_workflow_bench.py`, `e176_history_runtime.py`, and `docs/e251_contract.md`.

## Review Questions

1. Is the cache-consumed prefix boundary correct for all EOS rows, including the sampled EOS token and left-padded attention KV?
2. Are the GDN convolution and recurrent state commits correct for every valid suffix length, inactive row and padding position?
3. Does chunk64 plus whole-suffix convolution remove enough second-turn work to justify its complexity, and are there simpler alternatives with a larger full-workflow budget?
4. Which part of the system is the largest remaining optimization opportunity: first-turn prefill/decode, second-turn continuation, or runtime scheduling?
5. What is the likely cause of vLLM task304 failing to emit the tool call under the matched task order, and how should the runtime comparison handle this quality difference?

## Running Context

The scripts expect to run from the repository root and use repository-relative paths. The HF environment is `ascend-e2e`; the vLLM environment is `vllm-ascend-py310`. Do not mix their Python package stacks. The recorded E251 rerun used physical NPU0 with `scripts/env.sh` and the `ascend-e2e` Python 3.12 interpreter.

For a new single HF workflow trial, first satisfy the frozen first16, 32-task and full128 correctness prerequisites expected by `e251_full_workflow.py`, choose a new empty output directory, then run from the repository root:

```bash
source scripts/env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
python -u scripts/e251_full_workflow.py --output-dir results/<new-trial-dir> --performance-run
```

Add `--baseline` to select the unmodified history runtime for a matched control. The packaged ABBA driver targets an existing timestamped output root and deliberately refuses to overwrite it; point its `ROOT` at a new directory before starting a new four-trial sequence. Full128 trials take several minutes each and require an otherwise available NPU0.
