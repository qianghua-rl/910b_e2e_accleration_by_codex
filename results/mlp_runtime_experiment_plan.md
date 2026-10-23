# E117 MLP / runtime discriminating experiments

Goal: reduce E100 unprofiled fixed128 makespan with quality intact. This is diagnostic evidence, not a new accepted optimization. No model implementation changed.

## Experiments executed

Real checkpoint layers8–11 gate/up/down BF16 weights, synthetic activations, M=1/16/64. Five measurement windows after warmup, eight calls/window; host wall, NPU event window, submission duration. Serial synchronizes after each call; queued synchronizes at window end; rotating queues four different layer weights. The event window includes submission gaps. Environments and raw samples: `mlp_runtime_diagnostic.json`. No activation-distribution/quality claim.

| M=16 | Serial host ms | Queued same weight ms | Queued rotating weights ms |
|---|---:|---:|---:|
| gate | 0.1432 | 0.0806 | 0.1724 |
| up | 0.1434 | 0.0804 | 0.1736 |
| down | 0.1664 | 0.1039 | 0.1676 |
| full MLP incl SiLU/multiply | 0.5483 | 0.4910 | 0.4927 |

Full queued MLP M1/M16/M64 =0.4729/0.4910/0.5347ms. M16 full queued event window0.4763ms vs host0.4910ms; submission0.1281ms. Submission overlaps execution; do not sum/subtract these to invent a pure launch cost.

Interpretation:
- Weak scaling of latency with FLOPs supports a fixed/data-access/low-M-efficiency cost; it does not prove HBM saturation. Do not prioritize more arithmetic throughput based solely on MLP's large profile fraction.
- Repeated single GEMM has strong weight-reuse effects (one gate weight178MB, about170MiB; full three-weight MLP about510MiB). Full MLP does not reproduce its apparent hot-weight benefit. Cache hypothesis is plausible, not hardware-counter proof: rotating changes weight values too and is not controlled cache eviction.
- Serial-versus-queued saves about0.057ms/full MLP, but E100 has no per-layer sync. It is an artificial stress contrast, not a deployable win. Do not simply multiply by64layers/1258steps.
- E089 weight-only INT8 already had no latency gain. These results do not justify repeating that implementation. E087 packing also failed full workload; a proposed packing variant must identify a new mechanism before coding.

Runtime check reuses E113 unlabelled trace to limit extra perturbation. Of positive device gaps (EVENT entries excluded), 208.005ms/4265 gaps occur when the next kernel's host flow start is after prior device end; 8.622ms/3847 gaps when it starts earlier. This is host timing association, **not removable host time**: dispatch origin is not command-ready time, and dependencies/profiler scheduling remain. E115 labels enlarged the window, so use unlabelled trace here. Evidence: `e100_runtime_gap_diagnostic.json`.

## Next experiments, before implementation

1. **Runtime host sensitivity, highest diagnostic priority.** Fixed E100 first-batch prompts, original model and same teacher-forced decode tokens through HF's original generation preparation. Compare threads=8 versus1/4 in interleaved order, plus a repeat8 control; same cache positions/context lengths, warmup and paired windows. Record unprofiled decode wall and device events, exact logits, tokens and cache shapes. If host wall improves consistently while device workload is unchanged, localize the host-heavy sections; if not, do not infer graph opportunity from the gap count. Earlier E008/phase1 thread work used different batch/config; do not present a retry as a new optimization without explaining this new E100 diagnostic scope. Pure trace/threads alone cannot prove graph capture works.
2. **MLP actual-path residency control.** Capture actual E100 MLP input at one early and late layer, verify shapes/stride. Interleave complete gate→up→activation→down measurements with disjoint-weight execution; compare full wrapper and device kernels, not sum isolated hot GEMMs. Inspect transpose/format/layout copy kernels. Only pursue a new layout/prepack mechanism if measurable format overhead or residency dependence exists and the layout can persist across decode without recurring conversion. No quantization or packing implementation until that condition is met.
3. **Only after host sensitivity supports it, graph eligibility audit.** Capture-free inspection of ops/copies and cache mutation at the exact selected segment, then a tiny capability probe with stable tensor lifetimes. Do not repeat E098/E103 unchanged. If synchronous H2D copy or dynamic cache storage remains, no full-model graph implementation. Graph presence/backend name does not imply eligibility.

Decision now: no new optimizer justified yet. Runtime host sensitivity is the next small discriminator; MLP cache/data-path audit remains a second diagnostic, not a chosen quantization project. Stop if no stable net savings; use 4.214ms/whole decode step as a conditional 2% E100 screening target. Any implementation must pass real local output/state, model logits/full greedy tool-round trajectories, then matched fixed-task quality/makespan; only winners expand to128.
