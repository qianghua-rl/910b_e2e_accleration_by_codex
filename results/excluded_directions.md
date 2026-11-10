# Excluded / stopped directions from the old workspace (reference only)

Source: `/remote-home/S45160/work/910b_e2e_accleration_by_codex/OPTIMIZATION_LOG.md` (E001–E279).
Purpose: avoid re-running directions already rejected on the same software stack.
Re-open only if the stack, the workload, or the implementation premise changed (see
`NEXT_E2E_OPTIMIZATION_PLAN.md`, "Reopen Triggers").

## Relevant to host-side launch overhead (decode, batch1 / batch16)

| Direction | Log entry | Outcome |
|---|---|---|
| Fused RMSNorm, full path (decode + prefill) | E034 | rejected: 128-task quality guard failed (121/128) |
| Fused RMSNorm, prefill-only | E035, E058 | rejected |
| Explicit per-token decode loop | E044 | correct but slower |
| Finished-row / threshold live-row compaction | E062, E064, E092 | rejected |
| Slot admission / continuous batching | E096, E158, E164 | OOM or not bit-compatible; stopped |
| Torch / ACL graph capture, npugraph_ex | E010, E098, E103 | capture unavailable on this stack |
| AOT eager recurrent compile | E081 | rejected |
| Fixed-shape graph + StaticCache replay | (E251 graph routes) | no accepted path |
| GDN projection packing (z/b/a, four-way) | E071, E076, E086, E086b, E087 | rejected (perf or memory) |
| GDN grouped projection | E076 | slower (~24%) |
| ND layout replacement | E144 | ~37% slower |
| Recurrent state in-place / addcmul / matmul reduce | E070, E072, E078 | 1–3% local, not end-to-end |
| Recurrent state two-kernel fusion | E116 | wrapper slower and not exact |
| vLLM-Ascend GDN kernel as HF adapter | E051, E052 | single-layer exact; full-model adapter ~10.5% slower (per-call state layout transposes) |
| vLLM-Ascend GDN kernel, real batch16 replay | E104, E107 | correctness not passed |
| MegaGDN PTO prefill | E108, E109 | first-token trajectory changed; rejected |
| Causal conv single step | E114, E144 | rejected |
| Fused tail (gate/norm/recurrent/state commit) | E279 | rejected |

## Relevant to MLP / projection GEMMs

| Direction | Log entry | Outcome |
|---|---|---|
| FP8 / INT8 linear | E017 | blocked by stack |
| INT4 / MLP quantization | E089 | latency candidate rejected |
| BF16 NZ MLP layout | E120, E122, E124, E126–E128 | rejected; attribution showed no realizable gain |
| gate/up call-site experiment | E130 | diagnostic only, not integrated |
| FP16 compute form | E060 | rejected |
| Fused SwiGLU | E059 | rejected |

## Scheduling / protocol (not host-kernel)

| Direction | Log entry | Outcome |
|---|---|---|
| Shorter decode cap | E019 | rejected |
| Batch size sweeps | E021, E057 | OOM / boundary |
| Length-aware / decode-length ordering | E022, E028, E101, E102 | no gain beyond noise |
| Concise prompt / shortest-code prompt | E011, E048 | protocol change, invalid under current rules |

## Still open (not in the excluded list)

- Remaining GDN torch-fallback elementwise chain (gating `g`/`beta`, l2norm of q/k, dtype casts) after E069. Only partially explored (E082 low-benefit variants).
- Un-labelled decoder-layer RMSNorm and residual adds (the ~1540 kernels/step in `other`). Covered only by the full fused RMSNorm attempt (E034), which failed quality.
- Host-side scalar synchronisations in SDPA and generation loop (E118 found ~1248 `local_scalar` calls; no removal candidate recorded as completed).

## Rejected in fresh workspace (batch16, HF optimised, first16)

| Direction | Result | Outcome |
|---|---|---|
| Decode causal conv as shifted multiply-add (replaces Conv2D in `torch_causal_conv1d_update`) | first16: A 124.85 s vs B 127.39 s (+2%), both 16/16; 10/16 final texts identical | rejected: slower, not numerically identical; small-op overhead cancels the Conv2D saving. Fused single-kernel conv (Triton/AscendC) not tried. |
| `--sync-free` (device-tensor constants, no host syncs in RMSNorm / l2norm / recurrent step) | ABBA on first16 (A2 115.05 s, A3 116.01 s vs C3 121.80 s, C4 118.85 s): mean C 120.3 s vs mean A 115.5 s (+4%). Earlier single-pair C2 116.48 s vs A 124.85 s was noise. | rejected: no gain; do not run full128. Unit-level diff is 0.0, so the host-sync count is not the limiting factor. |
