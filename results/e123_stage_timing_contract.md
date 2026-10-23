# E123 stage timing contract (before measurement)

Diagnostic only: preserve installed HF `_sample` statements, original order and calls; insert host perf_counter and current-stream NPU event boundaries. No model/protocol/cache/scheduler optimization. Single NPU, BF16 E100 batch16 concise greedy, fused norms and single-step recurrent, fresh cache each generate. Actual E100 first-batch prompts and first32 tokens; no teacher forcing.

Inputs: left-padded int64 input_ids and attention_mask [16,L] from original tokenizer, original strides/layout unchanged. Forward logits BF16 [16,1,248320] (actual vocabulary follows model); sampling creates original FP32 logits copy. KV/recurrent states remain private to original HF cache; no state mutation beyond original model. No packed sequence interface or cu_seqlens introduced. Output int64 [16,L+32]. Timer observes shape/layout and never converts tensors within stages.

Stages: scheduler = HF loop continuation check (including Python bool synchronization) plus decode input preparation; model_forward = original decode model call; sampling = logits copy, processors and greedy argmax; postprocess = model kwargs update plus EOS masking, token concatenation and stopping criteria. Prefill reported separately, including HF preparation. Agent task admission, tool execution and text decoding are outside this diagnostic and must not be inferred from HF scheduler timing.

Host wall is submission/inclusive synchronization time. NPU event boundaries are stream elapsed intervals, including host-induced idle/backlog; neither pure busy time nor cross-clock overlap proof. Events resolve after one generate-end synchronization; no stage synchronization. Terminal condition has step=32; prefill step=-1. Stages cover condition/preparation/compute/token update, with Python timer/context overhead outside/in boundaries; sums need not equal generate wall. Compare alternating uninstrumented controls to expose perturbation.

Budget: unknown until timing. A removable stage would need consistently >2% of representative wall time, across repeated controls, before considering implementation. Even measured stage fraction is only an upper bound, not proven overlap availability (token/cache dependencies). No performance acceptance in this diagnostic. Extra memory O(steps*stage segments) events and host records, no extra logits/state tensors; measure allocated peak. Arithmetic/numerical risk zero by design; exact all-row tokens against both uninstrumented and saved E100 required. Stop on any token drift/OOM/device failure; do not expand to32 tasks/128.

References: operator_cases/005 graph host-copy failure and 006 admission OOM preclude blind capture/cache migration; E118 stable token checks but timing drift and E119 bool sync motivate original-loop boundaries. No failed optimization reopened.

Command:
`source scripts/env.sh`
`python scripts/time_generation_stages.py --run results/phase2_concise_sdpa_batch16_final12_length128.jsonl --single-step-recurrent --fused-rmsnorm --fused-gated-rmsnorm > results/e123_generation_stages.log 2>&1`

Reference arithmetic/HF/current NPU comparisons are the same unchanged HF/NPU model against uninstrumented runs and saved E100 token IDs. No new numerical operator requires an independent FP32 kernel reference. This is instrumentation validation, not candidate logits/state equivalence acceptance.
