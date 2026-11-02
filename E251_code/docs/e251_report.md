# E251 E250 history runtime: full128 incremental acceleration passed

Candidate = HF history-aware chunk64 recurrent suffix append (E250) plus
whole-suffix causal convolution with per-row historical-window commit (E248),
on E180's resident NPU EOS snapshot workflow. E180 remains selectable as the
matched baseline in the same runner. Original E100 task order, B16 membership,
BF16 SDPA/greedy concise protocol, cap512/12, actual Python tools, EOS/state
management and task99/785 whole-batch fresh fallback are unchanged. No default,
dependency or site-package changes were made.

State/runtime admission: independent per-row FP32 recurrence oracle passed CPU
and NPU cases for actual unequal suffixes, zero/mixed rows, all-active, decode
and resumed rows. Inactive final state is exact, padded outputs zero, input
state immutable; actual BF16 state relative error0.001659 vs FP32 oracle and
0.004331 vs per-token E176 cache commits are reported as numerical diagnostics.
Whole-suffix convolution outputs and final state are exact vs serial E176
reference and independent historical-window construction. Local probes only
screened the mechanism; they are not used as the performance claim.

Original32-task resident smoke passed all raw E100 task/round checks: v3=29/32
with unchanged failures1161/423/952,6564 tokens,32 tools, allocated peak
59,228,705,280B and reserved60,683,190,272B. Independent audit25 checks passed.

Frozen-source full128 correctness gate passed independent raw audit. All128
two-round token/tool/prediction trajectories match E100 exactly; v3=122/128
with unchanged failed IDs 1161,423,952,1035,552,1048,15961 tokens,128 tool
calls. Batches0/1/4-7 reused snapshots; batch2 used fresh fallback for task99
and batch3 for task785. Allocated peak59,972,823,040B, reserved
61,863,886,848B. All runner/candidate/auditor hashes matched current files.

## Full128 matched ABBA

Physical device0, separate process/model per trial, original128 task order,
resident workflow, tools included, model load reported separately. Raw trajectory
and scoring audits plus prompt-length reconstruction ran after each timer.

| Order | Path | Workflow s | Load s | Tool s | Allocated peak | Reserved peak | v3 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | E180 baseline | 261.257287 | 10.42 | 4.683145 | 59,972,823,040 B | 61,863,886,848 B | 122/128 |
| 2 | E250 candidate | 249.031911 | 10.31 | 4.841262 | 59,972,823,040 B | 61,863,886,848 B | 122/128 |
| 3 | E250 candidate | 247.372380 | 10.30 | 4.677513 | 59,972,823,040 B | 61,863,886,848 B | 122/128 |
| 4 | E180 baseline | 264.412632 | 10.34 | 4.688187 | 59,972,823,040 B | 61,863,886,848 B | 122/128 |

The independently recomputed matched means are262.834959s baseline and
248.202145s candidate, a 5.567301% incremental gain. Paired gains are
4.679439% and 6.444568%, both positive. Baseline range1.207754%, candidate
range0.670864%; mean gain exceeds both. Every trial independently matches all
15961 E100 tokens,128 tool outputs and final predictions, scorer-v3 failure
set, and allocated memory is below62,094,008,832B. Reserved memory is recorded
and identical. Candidate mean is also6.349460% below historical E100
265.030128s; this is supplementary to matched E180 comparison.

Independent audit results/e251/abba_verified/independent_audit.json verifies
all four raw trial audits, artifact/timer/command/source hashes, baseline vs
candidate selection, exact workload flag and its own recomputation of the
performance decision (all checks true). Trial prompt lengths are separately
reconstructed for all128 tasks after each timer; ranges are347-465 first turn
and425-936 second turn, identical across paths because trajectories are exact.
The complete decision is in results/e251/abba_verified/decision.json.

## Decision

E250 passes the full128 incremental performance, strict trajectory, scorer-v3,
tool, memory and stability gates over the already-validated E180 path. It is a
verified opt-in acceleration candidate. Default integration/deployment was not
requested and was not performed. E246 remains rejected; its vLLM prompt-ordering
candidate has no performance or valid memory acceptance. The E233-E245 scheduler
trace remains diagnostic and does not establish the earlier row0 divergence's
root cause.
