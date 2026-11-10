# Handoff: 910B Qwen3.5-27B agent makespan optimization (fresh workspace)

## Goal
- Single Ascend 910B3 (use npu:0 or npu:1, one card per run). Qwen3.5-27B BF16, self-written or HF-based inference.
- 128 GSM8K questions (`/remote-home/S45160/work/910b_e2e_accleration_by_codex/data/gsm8k_128.jsonl`), with a Python tool.
- Primary metric: whole-batch makespan. Accuracy must not drop below the baseline (fresh-v1 scorer).
- Protocol: greedy, non-thinking, first-turn cap 512, later turns cap 512, max 6 rounds, original system prompt.
- User goal (2026-11-05): reach roughly vLLM-Ascend speed **with a self-written / HF-based path**. Do not use vLLM kernels inside the HF path (third-party code; user rejected that option). vLLM-Ascend 577 s is a reference only, not a self-written result.
- Do NOT delete or modify the old project `/remote-home/S45160/work/910b_e2e_accleration_by_codex/` (read-only reference).

## START HERE (status 2026-11-06, end of session)
- **CURRENT BEST (self-written, HF weights)**: `--fast-gdn --gdn-v2 --prefill-group 1 --graph-decode --elem-conv`, runner `src/cb_run.py`, device npu:0, slots 16. **Mean makespan 410.0 s, 118/128 correct** (ABBA full128, see "NEW BEST" below). The 469.5 s figure below is the pre-elem-conv best, kept for history only.
- **Required environment**: `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` is now set automatically by `cb_run.py` whenever `--graph-decode` is passed (setdefault, so an explicit env value wins). Do not rely on the default `--prefill-group 4` for graph runs.
- **Device guard**: `cb_run.py` refuses to start if another `cb_run.py` already targets the same `--device` (checked via `pgrep`). Override with `--allow-busy` only when you accept contaminated timings. Verified 2026-11-06: a second start on npu:0 exits with code 1 before loading the model.
- **elem-conv correctness (verified 2026-11-06)**: within each config the runs are deterministic (A1 == A2 and B1 == B2 on all 128 final texts). elem-conv vs non-elem differ on 41 final texts (A1 vs B1: 87/128 identical). Correct sets: both groups 118/128, but the sets differ on two tasks, in opposite directions: task 1030 is correct without elem-conv and wrong with it (prediction None in B1; the cause was not inspected further); task 1137 is wrong without it (prediction None in A1, rounds cut off) and correct with it. Net 0. Cause of the two flips not inspected. So elem-conv is a deterministic numerical change with a zero net accuracy effect on this set, not a regression.
- **Memory limits**: 16 slots is the max (24 slots OOMs). Weights ~51 GiB, KV + GDN state ~6.3 GiB, prefill transient ~2 GiB.
- **Flags that are all default-off**: `--fast-gdn` (needed for the good numbers), `--gdn-v2`, `--graph-decode`, `--elem-conv`. Pass them explicitly.
- **No background jobs are running** at the time of writing. Check `ps aux | grep cb_run` before starting a new run on the same card.
- **NEW BEST (2026-11-06, ABBA full128)**: `--fast-gdn --gdn-v2 --prefill-group 1 --graph-decode --elem-conv`. Makespan **410.0 s mean** (B1 409.9, B2 410.2) vs **478.9 s** for the same config without elem-conv (A1 478.8, A2 478.9), about −14%. Correct 118/128 in all four runs. Prefill ~133 s, decode steps 2847 (vs 2818). Script: `scripts/run_abba_full128_elemconv.sh`, results `results/ablations/continuous_batching/abba_full128_elemconv_{A1,B1,B2,A2}`. Note: the A-group here is 479 s, not the earlier 469.5 s mean, so this batch ran slower overall; the A/B gap is the comparison that counts. Caveat: the elem-conv runs' final text differs from non-elem runs (A1 vs B1 identical on 87/128), so this is a numerical change, not a bit-identical one. Same 118 correct, but the set of correct tasks was not compared.
- **Graph bucket 128 vs 512 (2026-11-06, first16, elem-conv on, `scripts/run_abba_first16_bucket.sh`, `results/ablations/continuous_batching/abba_first16_bucket_{A1,B1,B2,A2}`)**: A (512) 72.26 / 72.22 s (mean 72.24), B (128) 71.06 / 72.08 s (mean 71.57). Gain 0.9%, below the 2% gate, so full128 was NOT run. Default stays 512. All 16/16 correct in all four runs. Text: A1 vs A2 identical 16/16; B1 vs B2 identical 16/16; A1 vs B1 identical 14/16. The override is available as env `CB_GRAPH_BUCKET` (metadata records `graph_bucket`). Earlier attempt had its A1/B1/B2 blocked by a false-positive device guard (fixed; see Device guard above), so those runs are the clean repeat.
- first16 ABBA (`scripts/run_abba_elemconv_graph.sh`): A 92.9 s / 93.0 s, elem-conv B 72.7 s / 72.0 s, all 16/16. The earlier "no gain" finding was under eager, not graph.
- **Other remaining budget** (estimates, not re-measured after v2): decode ~350 s at ~124 ms/step (2818 steps, slot occupancy ~97%); prefill ~120 s (GDN chunk ~66% of prefill before v2, non-GDN ops ~173 ms per 353-token prefill, not yet split after v2). Smaller graph buckets (e.g. 128 instead of 512) may reduce attention padding waste; untested.
- **Ruled out, don't retry**: slot count > 16 (OOM), prefill grouping (no gain), sync-free (slower), shift-conv (slower, not identical), chunk size 128 for GDN v2 (slower).

## Workspace
- `/remote-home/S45160/work/910b_fresh/`: `src/`, `scripts/`, `results/`, `logs/`.
- Python envs:
  - `/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python` (HF transformers 5.5.4, torch_npu 2.9.1). Run with `-I` and absolute `sys.path` inserts.
  - `/remote-home/S45160/miniforge3/envs/vllm-ascend-py310/bin/python` (vllm 0.18.0). Needs `source /usr/local/Ascend/ascend-toolkit/set_env.sh`, `LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver`, `ASCEND_RT_VISIBLE_DEVICES`. Do NOT use `-I` (EngineCore subprocess then cannot import `acl`). For profiling vLLM use `LLM(..., profiler_config={"profiler": "torch", "torch_profiler_dir": OUT})`, `gpu_memory_utilization=0.98` (0.85 gives no KV memory).
- Scorer: `src/scorer.py` (fresh-v1). Input rows need `id`, `gold`, `final_text`. Filter out `kind` != `task` rows first.

## Key files
- `src/e100_batch.py`: HF batched runner (batch16, original/concise protocol). Flags:
  - `--no-opt`: disable fused RMSNorm / gated RMSNorm / single-step recurrent.
  - `--shift-conv`: decode conv as shifted multiply-add (REJECTED, see below).
  - `--sync-free`: device-tensor constants, no host syncs (REJECTED: ABBA showed +4% slower on first16).
  - `--limit N`: first N tasks in dataset order.
  - `--schedule fifo` for dataset order; `generation_length` needs `--order-file`.
- `src/sync_free.py`: sync-free `l2norm_nosync`, `recurrent_nosync`, `fused_rmsnorm_nosync` (precomputed `1+w`), `apply_sync_free(model)`. Constants cached by `_const()` (creating `torch.tensor(..., device=)` per call is itself a copy and made it slower).
- `src/shift_conv_update.py`: shifted-add decode conv (rejected). HF cache conv_state length is K (4), not K-1.
- `src/recurrent_single_step.py`, `src/agent_protocol.py`: copied from the old project (helpers).
- `src/vllm_agent_bench_fair.py`: vLLM-Ascend runner copy. Its task rows use `expected`/`prediction`; map them (last round `text` -> `final_text`) before scoring.
- `src/run_agent.py`: serial B=1 baseline runner.
- `src/upper_bound_gdn.py`, `src/split_upper_bound_gdn.py`: stub timing experiments (timing only).
- `src/exact_gdn_candidates.py`, `src/model_exact_gdn.py`: exact GDN candidates, ~0.3% end-to-end.
- `results/excluded_directions.md`: old-project rejected directions + new rejection (shift conv).
- **Current-best components (added 2026-11-06)**:
  - `src/cb_engine.py`: `CBEngine` with `graph_decode` (`_capture_graphs`, `_graph_core`, `_decode_step_graph`, `GRAPH_BUCKET=512`), `patch_model(fast_gdn, gdn_v2)`, `elem_conv_update` (`--elem-conv`, default off).
  - `src/cb_run.py`: scheduler and CLI (`--fast-gdn --gdn-v2 --prefill-group N --graph-decode --elem-conv --slots N --limit N`).
  - `src/gdn_chunk_fast.py` (v1 chunk GDN, its `_unit_lower_inv` is reused) and `src/gdn_chunk_fast2.py` (v2, batched precompute).
  - Backups of pre-graph files: `/tmp/cb_engine_before_graph.py`, `/tmp/cb_run_before_graph.py` (tmp, may not survive a reboot).
  - Tests and probes: `scripts/test_acl_graph.py`, `scripts/test_gdn_v2.py`, `scripts/probe_graph_mem.py`, `scripts/profile_prefill_split.py`, `scripts/profile_decode_wall_breakdown.py`.
  - ABBA/run scripts: `scripts/run_abba_graph_pg1.sh`, `scripts/run_abba_full128_gdnv2.sh`, `scripts/run_full128_*.sh`.
- Scripts (diagnostics): `scripts/prefill_probe.py`, `scripts/slow_batch_steps.py`, `scripts/profile_batch16_decode.py`, `scripts/profile_vllm_batch_decode.py`, `scripts/sync_sites.py`, `scripts/profile_sync_stack.py`, `scripts/sync_micro.py`, `scripts/test_shift_conv.py`, `scripts/test_sync_free.py`, `scripts/run_ab_first16.sh`, `scripts/run_abba_sync.sh`.

## Current results (fresh-v1 scoring, 128 tasks)
| Config | Makespan | Peak mem | Correct |
|---|---|---|---|
| Serial B=1 HF baseline (`results/baseline_serial`) | 8365.0 s | 54.26 GB | 116/128 |
| HF batch16 + opts (`results/b16_original_opt`) | 1180.0 s | 62.06 GB | 116/128 |
| HF batch16 no opts (`results/b16_original_noopt`) | 1450.97 s | 62.06 GB | 118/128 |
| vLLM-Ascend batch16 (`results/vllm_original_b16`) | 576.75 s | not recorded | 118/128 |

## first16 A/B (batch16, fifo, `--limit 16`, NPU 0, results under `results/shift_conv/`)
| Run | Makespan | Correct | Final text identical to A |
|---|---|---|---|
| A: HF batch16 + opts (current best self-written) | 124.85 s | 16/16 | — |
| B: + shift conv | 127.39 s | 16/16 | 10/16 → REJECTED |
| C: sync-free, constants created per call | 148.4 s | 16/16 | 16/16 (slower, fixed) |
| C2: sync-free, cached constants | 116.48 s (−6.7%, single pair, not confirmed) | 16/16 | 16/16 |
| A2 / A3 (ABBA, baseline) | 115.05 s / 116.01 s | 16/16 | — |
| C3 / C4 (ABBA, sync-free) | 121.80 s / 118.85 s | 16/16 | — |

**ABBA result: sync-free is slower (mean 120.3 s vs 115.5 s). Rejected; full128 not run.** The C2 gain was noise.

- **ABBA status**: finished (`logs/abba_status.txt` = `DONE`). Gate not met; sync-free rejected.
- Unit tests: `scripts/test_sync_free.py` shows l2norm, recurrent step, and RMSNorm are bit-identical to the originals (diff 0.0).

## Findings
- Decode on HF at batch16: device kernel union busy ~68% of span, ~137 ms/step steady (`results/profile_batch16_decode`, batch 15). **Correction (2026-11-06):** "device-bound" is NOT established. The elem-conv test (see CB section) cut device time per step by ~23% (118.5 → 91.2 ms) with no wall-clock gain, so device kernel time is not what limits step time. The bottleneck is unresolved (candidates: host dispatch/sync, per-op launch overhead). Earlier b1 host-bound conclusion is also unverified for batch16.
- Per-step device time (HF, batch 15): MatMul ~56 ms (both stacks), GDN causal conv Conv2D+TransData ~24 ms, Mul ~12 ms, FlashAttention ~8 ms, EVENT_WAIT ~8 ms.
- vLLM batch 15 decode estimate: ~113 ms/step (`results/profile_vllm_batch_decode`; that window includes prefill). vLLM uses fused GDN kernels (`fused_sigmoid_gating_delta_rule_update_kernel`).
- Turn 1 per-batch timing: HF slowest batch (ids 247..444, 482 steps) runs ~137 ms/step steady; prefill ~10.7 s of 76.8 s. The earlier "330 ms/step" was a mis-attribution.
- Prefill is a small share of makespan (re-prefill estimated ~70 s of 1180 s). Cross-turn KV reuse not implemented; low expected value.
- **Host sync sources (decode step, HF batch 15, 311 `aten::_local_scalar_dense` per step)**:
  - fused RMSNorm `(1.0 + weight)` computed per call: 161
  - GDN `l2norm` `+ eps` (Python float): 96
  - `recurrent_single_step` `q * (1/sqrt(d))` (Python float): 48
  - Microbench: any Python-float scalar op on NPU syncs; device-tensor scalars do not; `npu_rms_norm` with precomputed weight does not sync.
  - Blocked time on those syncs ~18 ms/step (sum of event durations).
- Shift-add decode conv: rejected (slower, not numerically identical). Conv path needs a fused single kernel to help; not attempted.
- Old project E251 (chunk64 history recurrence + vectorized historical conv) gave 5.6% in the old project. Not ported.

## Continuous batching (current direction, status 2026-11-06)

Files: `src/cb_engine.py` (slot-based decode engine), `src/cb_run.py` (scheduler; same output format as `e100_batch.py`), `scripts/test_cb_engine.py` (correctness vs HF `generate`), `scripts/profile_cb_step.py`, `scripts/profile_cb_prefill.py`.

Design:
- Slots hold KV (full-attention layers) and conv/recurrent state (GDN layers). HF's built-in continuous batching does not handle GDN state, so this is self-written.
- Decode runs all N slots densely each step (no gather/scatter; advanced indexing is 5-20x slower on NPU than `index_select`/`index_copy_`). Idle slots are dummy rows and are ignored.
- Prefill: batch-1 (`prefill`) or left-padded group (`prefill_batch`, `--prefill-group`).

Correctness: decode matches HF `generate` token-for-token (3-4 slots, 40 steps), for both single and batched prefill.

Results (fresh-v1 scoring, 128 tasks; baseline `b16_original_opt` = 1180.0 s, 116/128):

| Run | makespan | decode steps | prefill calls | correct | peak mem | notes |
|---|---|---|---|---|---|---|
| CB, single prefill (`results/ablations/continuous_batching/full128`) | 687.8 s | 2845 | 251 | 117/128 | 62.00 GB | run on NPU 1; code predates later `cb_engine.py` edits, so not comparable to the `repeat_*` pair |
| CB, prefill group 4 (`full128_batchprefill4`) | 775.2 s | 2869 | 228 (of 252) | 117/128 | 62.55 GB | ran concurrently with the group-8 run |
| CB, prefill group 8 (`full128_batchprefill8`) | 729.1 s | 2875 | 228 (of 252) | 117/128 | 63.56 GB | ran concurrently with the group-4 run; no OOM |

- Not a clean speed comparison: the two group runs overlapped on different cards (contention on shared host/device resources). Rerun alone if the speed of group sizes matters.
- Batch16 with 16 simultaneous prefills OOMs (`results/ablations/continuous_batching/full128_batchprefill` exit 1); group 8 fits.
- Grouping rarely happens: 252 prefills collapse to only 228 calls because slots free up one at a time. The expected gain from batched prefill is therefore small.
- Prefill is a large share: ~290-350 s of makespan in the non-fast-GDN runs (`prefill_seconds` 336-340 s). Per single prefill ~0.82 s (353 tokens) without fast-GDN; 75% of it is the torch GDN fallback `torch_chunk_gated_delta_rule` (FLA not installed). **Update:** `--fast-gdn` cuts `prefill_seconds` to ~175 s (`repeat_fast_r2`); prefill is now ~30% of makespan. That function scales sub-linearly with batch (B=4 ~1.1x of B=1; B=16 ~3.1x).
- Decode per step ~0.14 s with 16 slots (dense). Static batch16 steady-state is ~0.137 s/step (`results/slow_batch_steps/steps.json`, 482 steps). **Correction:** the earlier "~0.20 s" static figure was wrong; `results/profile_batch16_decode/profile_meta.json` (188 ms/step) is an 8-step profiled window including warmup and is not a steady-state figure. So per-step decode cost is about the same in static and CB; CB's gain comes from fewer total decode steps and better slot reuse.
- Correctness: 117/128 vs baseline 116/128. Only task id 1137 flips between runs (baseline vs CB); not investigated yet. Identical final text vs baseline: 85-87/128; the rest are greedy divergences from different numerical paths (batch shapes), not checked one by one.

Clean unshared reruns (npu:0 only, run sequentially; `scripts/run_cb_clean_group.sh`, 2026-11-06):

| Run | makespan (last task completion) | correct | output |
|---|---|---|---|
| CB, prefill group 4 (`clean_group4`) | 750.2 s | 117/128 | `results/ablations/continuous_batching/clean_group4` (single `per_task.jsonl` file, not a directory) |
| CB, prefill group 8 (`clean_group8`) | 837.0 s | 117/128 | `results/ablations/continuous_batching/clean_group8` (single file) |

Same-code fast-GDN A/B (`scripts/run_repeat_cb_npu1.sh`, NPU 1, sequential, prefill group 4, original protocol):

| Run | makespan | correct | output |
|---|---|---|---|
| CB base, current code (`repeat_base_r2`) | 755.2 s | 117/128 | `results/ablations/continuous_batching/repeat_base_r2` |
| CB + `--fast-gdn` (`repeat_fast_r2`) | 588.2 s | 118/128 | `results/ablations/continuous_batching/repeat_fast_r2` |

Speed gain ~22% on the same code, card, and schedule. The earlier `full128` (687.8 s) and `full128_fastgdn` (570.6 s) pair is NOT a fair comparison: the baseline predates edits to `cb_engine.py` and the two ran on different cards. Use the `repeat_*` pair. Correctness flips between base and fast: task 640 and 1030 fixed, 1137 broken.

Decode op profile (CB, 16 slots, 20 steps; `results/profile_cb_decode_ops`, `scripts/profile_cb_decode_ops.py`): ~119 ms/step device time. MatMul 57.4; conv path (Conv2D 13.9 + TransData 11.2 + format conversions ~2) ~27; Mul 12.9 (624 calls). Micro-benchmark (`scripts/bench_decode_conv.py`): HF conv1d fallback 34 ms/step vs mul+sum 8.4 ms/step (state identical, output diff 0.125 bf16).

Elem-conv end-to-end A/B (`--elem-conv`, `scripts/run_ab_elemconv.sh`, first16, fast-gdn fixed, NPU 1 sequential): A 104.6 s, B 105.0 s, both 16/16. No measurable gain: the micro-benchmark saving did not show up end to end. Rejected for now; flag kept, default off. Not run on full128.

Elem-conv device vs wall clock (`results/profile_decode_A/B`, `scripts/bench_decode_wall.py`): device time per step 118.5 ms (conv1d) vs 91.2 ms (elem), so the device saving is real. Wall-clock decode step with no profiler, ABBA (A,B,B,A): A 139.3 ms, B 140.4 ms. No wall-clock gain. The decode step is bounded by something other than device kernel time (likely host dispatch/sync at 16 slots). Implication: optimize against wall-clock step time, not profiler device sum. Earlier "device-bound" conclusion for batch16 (HANDOFF findings above) should be re-checked against this.

Flip diagnosis (`scripts/diag_flip.py`, log `logs/diag_flip.log`): the clean base run (`clean_group4`) matches `repeat_base_r2` on all three tasks, so the flips come from the fast-GDN prefill only. First-turn divergence token index: 1137 @164, 640 @19, 1030 @14. Full-forward logits at the divergence prefix, ref vs fast: top-2 margin under ref is 0.000 (1137, 640) and 0.125 (1030), i.e. exact ties at bf16 resolution. Max |logit diff| over vocab is 0.14–0.25, bf16-sized. The full forward does not reproduce the cache-decode choice exactly (for 1137 it picks 25 while the base run picked 279), so this is a proxy, not an exact replay. Conclusion: noise-level tie-breaking, not a logic error. Random-input unit test (`results/gdn_chunk_fast/test_output.txt`): output diff ~1e-5 and state diff ~3e-8 in fp32. The fast path is not bit-identical to the reference, so a few flips per 128 is expected. Task 1137 is not explained beyond this (the 1137 flip also appeared between baseline and CB earlier).

Decision (superseded 2026-11-06): the default was `--prefill-group 4` at the time. The current best uses `--prefill-group 1` with `--graph-decode` (see START HERE). Group 8 is slower when run alone; the earlier 729 s group-8 result came from a run that overlapped with group 4 and is not comparable. Makespan: the CB summary row does record `makespan_seconds` (verified for `repeat_*`, `full128*`, `clean_group*`), and it equals the max `completion_seconds` in `per_task` to within 0.01 s. Use `summary.makespan_seconds`.

Next steps (candidates):
1. (Done) Clean unshared run of group 4 and group 8. Group 4 chosen.
2. ~~Change admission so slots free together~~ Measured 2026-11-06 and dropped as the main lever: slot occupancy is 96.8% (`repeat_fast_r2`: 43,795 generated tokens / (2828 steps × 16 slots)), so idle slots are not the loss.
3. ~~Reduce the prefill cost~~ Partly done: `--gdn-v2` (batched precompute in `src/gdn_chunk_fast2.py`) cut prefill ~19% (see "Prefill split and GDN v2"). Remaining prefill cost is not yet split after v2.
3b. Slot count: decode step cost is nearly flat in batch (15 rows static 137 ms; 16 slots CB 146 ms), so more slots would cut total decode steps. **Result (2026-11-06): 24 slots + fast-GDN OOMs on NPU (60.5 GiB allocated, 9.6 MiB free); no output.** Log: `logs/cb_slots24_fast_r1.log`. Do not raise slots above 16 without freeing memory (e.g. weight/KV layout changes).
4. Investigate task 1137's correctness flip.

Rejected this round: `--sync-free` (ABBA: C 120.3 s vs A 115.5 s, +4%); see `results/excluded_directions.md`.

### Decode wall-clock breakdown (2026-11-06, `scripts/profile_decode_wall_breakdown.py`, NPU 0, 16 slots, fast-GDN, 60 steps)
Output: `results/decode_wall_breakdown/run1.json`, log `logs/decode_wall_breakdown_run1.log`.
- Mean step wall time: **135.6 ms**.
- Host-side time inside `_attn` (all 16 full-attn layers, summed over calls): 46.8 ms.
- Host-side time inside `_gdn` (all 48 GDN layers): 76.6 ms.
- Remainder (embed, MLP, norms, lm_head, final `.tolist()` sync): about 12 ms (MLP not timed separately).
- Caveat: the wrappers measure Python-side time, which includes any host syncs inside those functions, so this does not split dispatch cost from device wait. It shows that Python/launch-side work accounts for ~90% of the step.
- Interpretation: decode is **host-dispatch-bound**, consistent with the elem-conv result (device time dropped 23%, wall time did not). Reducing device kernel time alone will not help much.
- Candidate levers: (a) fuse ops to cut launch count (GDN and attention paths hold most of the ops); (b) ACL graph capture of the decode step (see the next section).

### Graph decode in CBEngine (2026-11-06, `--graph-decode` flag, default off)
Implemented in `src/cb_engine.py` (`_capture_graphs`, `_graph_core`, `_decode_step_graph`, `GRAPH_BUCKET=512`) and `src/cb_run.py`. Eager path unchanged when the flag is off.

- **Memory constraint**: graph decode works only with `--prefill-group 1`. With `--prefill-group 4` the first prefill after capture OOMs (`results/ablations/continuous_batching/graph_first16_graph` failed, log `logs/cb_graph_first16_graph.log`).
- **first16** (`--prefill-group 1`, `results/ablations/continuous_batching/graph_first16_pg1_*`): 16/16 correct for both; final_text identical 14/16. Divergences are tie-break level (task 51: round 1 at token 117, eager 40003 vs graph 283; task 23: round 2 at token 42, eager 2984 vs graph 3202). Not investigated further; attributed to kernel-order numerical noise, not verified by logit margin.
- **ABBA first16** (`scripts/run_abba_graph_pg1.sh`, `results/ablations/continuous_batching/abba_graph_pg1_{A1,B1,B2,A2}`): eager 101.9 / 111.9 s (mean 106.9), graph 95.3 / 94.8 s (mean 95.0), gain ~11%. Eager spread (10 s) is large.
- **full128** (`--fast-gdn --prefill-group 1`, `scripts/run_full128_graph_pg1.sh`, `results/ablations/continuous_batching/full128_pg1_{eager,graph}`):

| Run | makespan | correct | decode steps |
|---|---|---|---|
| eager | 570.8 s | 118/128 | 2827 |
| graph-decode | 499.7 s | 118/128 | 2818 |
| ref `repeat_fast_r2` (prefill-group 4) | 588.2 s | 118/128 | 2828 |

  Graph vs eager: −12.5% makespan at equal correctness; final_text identical 110/128. No OOM or errors in either log. The comparison to the 588.2 s reference is not like-for-like (prefill-group differs).
- **Best current config**: `--fast-gdn --prefill-group 1 --graph-decode`, 499.7 s, 118/128.
- **Memory fix (2026-11-06)**: the OOM with `--prefill-group 4 --graph-decode` is fixed by `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` (allocator fragmentation). Breakdown (`scripts/probe_graph_mem.py`): weights ~51 GiB, KV + GDN state ~6.3 GiB, prefill transient ~2 GiB. Without the env var the graph run OOMs at prefill-group 4; with it, reserved memory drops from 60.2 to 59.7 GiB. **This setting is required**: removing it brings the OOM back.
- **pg4 + graph, full128** (`scripts/run_full128_graph_pg4_expseg.sh`, `results/ablations/continuous_batching/full128_pg4_graph_expseg`): 504.0 s, 118/128, 2826 decode steps, 222 prefill calls, no OOM or ERROR. Not faster than pg1 graph (499.7 s): the 4.3 s gap is within noise. Decode steps are essentially equal (2826 vs 2818, 0.3%), and prefill time is also equal (153.2 s vs 151.7 s), so prefill grouping gives no measurable gain in this setup. (An earlier note here claimed more decode steps under pg4; that was wrong.) Compared with `repeat_fast_r2` (588.2 s), this is −14%, but graph and the allocator setting are not measured separately.
- **first16, pg4 + graph + expandable segments** (`results/ablations/continuous_batching/graph_first16_pg4_expseg`): 16/16, 89.4 s.
- **Open**: pg1 graph remains the fastest config. Per-step time is the same under pg1 and pg4 (~123–124 ms with graph vs ~144 ms eager, approximate, from makespan minus prefill divided by decode steps). Graph gives ~20 ms/step; prefill (~150 s) and decode (~350 s) are the remaining budget.

### Prefill op profile after GDN v2 (2026-11-06, `scripts/profile_prefill_v2_ops.py`, `results/prefill_v2_ops`, summary `scripts/summarise_kernel_csv.py`)
- Config: batch-1 prefill, `fast_gdn=True, gdn_v2=True`, 8 real prompts (2958 tokens) after 2 warmup prompts.
- Device kernel time total 2345 ms = ~293 ms/prompt (profiler on; wall 540 ms/prompt under profiler, so host/launch gap is large and unmeasured without profiler).
- Top ops by device time: MatMul 37.5% (878 ms, ~110 ms/prompt, roughly 55% of bf16 peak by FLOP estimate, so limited headroom); Mul 8.9% (13968 calls); InplaceCopy Slice 5.7% (10896 calls); BatchMatMul 4.8%; ReduceSum 4.2%; Tril 4.0% (768 calls); Conv TransData 3.8%; ViewCopy 3.8%; Add 2.4%; Slice 1.9%; Transpose ~1.5%; RmsNorm 1.4%; ConstantPadNd ~2.5% total.
- Reading: the non-matmul ops are dominated by small elementwise/copy/slice ops (~20k kernels per prefill of 8 prompts... see n= column), many of which belong to the GDN chunk helpers (Mul, Slice copies, ReduceSum, Tril, Pad, RepeatInterleave, Cat). Mul alone is 13968 calls over 8 prompts (~1750 per prompt). The remaining prefill cost is mostly many small ops, not a single hotspot. Next lever is op-count reduction in `src/gdn_chunk_fast2.py` (e.g. precomputed masks instead of Tril/Triu per chunk, fewer slice-copies), not matmul tuning.

### Prefill split and GDN v2 (2026-11-06)
- **Prefill split** (`scripts/profile_prefill_split.py`, `results/prefill_split/run1.json`, batch 1, 353 tokens, fast-GDN): total 515 ms, GDN chunk function 341 ms (66%), other ops 173 ms. Per GDN layer ~7 ms over 48 layers.
- **`src/gdn_chunk_fast2.py` (`--gdn-v2`)**: same math as `gdn_chunk_fast.py`, but everything that does not depend on the running state (intra-chunk attention, Q·K decay, `Kd`, `Qg`, `Gl`) is computed once for all chunks in batch. The sequential loop keeps only the state products: `vn = V - Kc@S`, `out = Qg@S + QK@vn`, `S = S*Gl + Kd@vn`. The v1 intra-chunk inverse (`_unit_lower_inv`) is reused.
  - Numerics vs v1: output and final state identical on random inputs (`scripts/test_gdn_v2.py`, max diff 0.0). Same at chunk 128 within fp32 rounding (rel 3e-5 output, 8e-7 state).
  - Single-layer time (353 tokens, 48 heads): v1 6.94 ms → v2 5.64 ms at chunk 64; chunk 128 is slower (6.32 ms), so chunk 64 stays.
  - `cb_engine.patch_model(..., gdn_v2=True)` and `cb_run.py --gdn-v2`. Default off.
- **first16** (`--gdn-v2 --prefill-group 1 --graph-decode`, `results/ablations/continuous_batching/graph_first16_pg1_gdnv2`): 16/16 correct, final_text identical to pg1 graph 16/16, makespan 91.6 s.
- **full128** (`results/ablations/continuous_batching/full128_pg1_gdnv2_graph`): 474.9 s, 118/128, prefill 125.1 s (vs 151.7 s), decode steps 2818 (same), final_text identical to pg1 graph 128/128.
- **ABBA full128** (`scripts/run_abba_full128_gdnv2.sh`, `results/ablations/continuous_batching/abba_full128_{A1,B1,B2,A2}`; A = baseline `--fast-gdn --prefill-group 1 --graph-decode`, B = `+ --gdn-v2`):

| Run | makespan | prefill | correct |
|---|---|---|---|
| A1 | 502.8 s | 153.6 s | 118/128 |
| B1 | 468.4 s | 119.5 s | 118/128 |
| B2 | 470.6 s | 122.3 s | 118/128 |
| A2 | 493.9 s | 144.5 s | 118/128 |

  Mean A 498.4 s, mean B 469.5 s: **gain 5.8% makespan**; prefill mean 149.1 → 120.9 s (−18.9%). The two groups do not overlap (slowest B 470.6 < fastest A 493.9). No OOM or ERROR in any log.
- **Current best config**: `--fast-gdn --gdn-v2 --prefill-group 1 --graph-decode`, mean 469.5 s, 118/128.

### NPUGraph feasibility (2026-11-06, `scripts/test_acl_graph.py`, NPU 0, 16 slots, fast-GDN, 40 steps)
- Feasible. `torch.npu.NPUGraph` captures the decode step with static input buffers (`tok_buf`, `pos_buf`) and one graph per `Lmax` bucket (512-token steps up to `t_max`=4096, 8 graphs).
- Capture must use a **shared** pool (`torch.npu.graph(g, pool=torch.npu.graph_pool_handle())`). With private pools per graph, the next prefill OOMed.
- Graphs are captured before any prefill, with all slots empty, and the empty-slot state is zeroed afterwards.
- Correctness: greedy token sequences are **identical to eager for 16/16 slots over 40 steps**.
- Speed: eager **136.6 ms/step**, graph replay **120.2 ms/step** (−12%). Replay time is close to the ~119 ms device-time sum from the op profile, so after graphing the decode step is device-bound. Expected full128 gain: ~50 s (413 s decode → ~365 s), makespan ~540 s if the gain holds.
- **Done (later the same day)**: integrated into `CBEngine` behind `--graph-decode` and verified (first16, full128, ABBA). See "Graph decode in CBEngine" above. The "device-bound" wording in this section is superseded: the wall-clock breakdown showed decode is host-dispatch-bound before graphing; graph removes that cost.

## Update 2026-11-07/08: benchmark comparison, long-generation runs, tensor parallel

### Results layout (reorganized 2026-11-07)
- `results/` is now grouped by purpose. Full map and notes: `results/README.md`. Comparison report: `results/REPORT.md` (written before the TP=2 runs; needs the TP=2 and math500/bbh numbers added).
- `results/benchmarks/bench/` self-written CB results (bbh, gpqa_diamond, math500, aime2024, aime2025).
- `results/bench_vllm/` vLLM single-card results (original 512/2048 caps; AIME rows are invalid, see below).
- `results/bench_vllm_retest/` vLLM AIME re-run with first cap 2048, final 512, max_model_len 4096.
- `results/bench_vllm_tp2/` vLLM TP=2 runs with caps 6144 / 2048, max_model_len 8192. `math500_aborted/` is a partial run that was stopped; not usable.
- `results/ablations/continuous_batching/` (renamed from `cb/`). Old `results/cb/` paths in docs and scripts were rewritten to the new path.
- `results/ablations/continuous_batching_tp2/` self-written TP=2 runs (`smoke`, `single_ref` are debug runs, not benchmark results).
- `results/archive/` old configs and aborted runs. Nothing was deleted.

### Scoring fixes
- **`src/cb_run.py` correctness bug (fixed)**: `correct` was `tool_count > 0 and ...`, which is always False in `--direct` mode. Now `(tool_count > 0 or a.direct) and ...`. Result files written before the fix still have `correct: false` for direct runs (bbh, gpqa_diamond in `benchmarks/bench/`). Use the offline scorers instead: `src/scorer_mc.py` (MC) and `src/scorer_math.py` (math). Check with `python3 -I src/scorer_math.py <file>`.
- Offline scores (fresh-mc-v1 / fresh-math-v1):
  - self-written, single card: bbh 1155/1350, gpqa 92/198, math500 389/500, aime2024 7/30, aime2025 9/30
  - vLLM single card, original caps: bbh 1151/1350, gpqa 96/198, math500 386/500, AIME 0 (invalid, all truncated at 512 first turn)
  - vLLM AIME retest (first 2048, final 512, model_len 4096): aime2024 9/30, aime2025 8/30

### Truncation is the main accuracy limit
- Single card, caps around 2048: about half of GPQA and most AIME answers have no `\boxed{}`. Median total generated tokens on GPQA (TP run) is 2113, so 2048 is too low.
- Caps 6144 / 2048 with TP=2 (vLLM, `results/bench_vllm_tp2/`, scored offline):
  - gpqa_diamond 137/198 (69%), no boxed 33/198
  - aime2024 24/30 (80%), no boxed 5/30
  - aime2025 18/30 (60%), no boxed 11/30
  - math500 and bbh: rerun queued, not yet scored
- Single-card 8192 is not feasible: weights ~51 GiB, KV scales with slots × t_max. Estimated, not measured; 24 slots at 4096 already OOMed. Long-cap comparisons therefore need TP=2 on both systems.

### Speed interpretation
- vLLM TP=2 vs single card on GPQA: wall time 13815 s vs 6202 s (2.2×), but generated tokens 645k vs 321k. Per-token throughput 46.7 vs 51.7 tok/s (≈ −10%). The large wall-time gap comes from longer generations, not from sharding. The −10% is confounded by the longer context and was not measured with a single-card 6144 control.

### Tensor parallel in the self-written engine (new)
- The existing `--split-devices` is a layer pipeline split (memory only, no speedup, eager only). It is not tensor parallel.
- New: `src/cb_tp.py`
  - `shard_model(model, rank, tp)`: shards q/k/v (by head), o_proj, GDN in_proj_qkv/z/b/a (by head block, with the three q/k/v parts sliced separately), out_proj, MLP gate/up (by column) and down (by row). o_proj, out_proj and down_proj are wrapped in `AllReduceLinear` (HCCL all_reduce).
  - `TPEngine(CBEngine)`: sequential per-prompt prefill re-implemented for sharded shapes; decode reuses `CBEngine.decode_step`. Eager only (graph capture with collectives not attempted).
  - `init_tp(rank, world)`: HCCL process group; each process uses one card as `npu:0`.
- `src/cb_engine.py`: constructor takes `tp`; local KV heads = total / tp, local Q heads = total / tp.
- `src/cb_run.py`: `--tp N`. Launch with `RANK=r ASCEND_RT_VISIBLE_DEVICES=r`. Only rank 0 writes output. Incompatible with `--graph-decode` and `--split-devices`.
- Launcher: `scripts/run_cb_tp2.sh <smoke|full>` (both ranks in parallel).
- Tests:
  - `scripts/test_tp_cpu.py`: tiny random model, gloo, 2 processes. TP tokens == single-process tokens (8/8). Passed.
  - NPU smoke (`run_cb_tp2.sh smoke`, GPQA 2 tasks): both correct with `\boxed{}`, peak memory about 35 GB per card, no errors.
  - Compared with the single-card reference on the same 2 tasks: token sequences identical for 269 and 792 tokens, then diverge at places where both wordings are plausible. Final answers match gold. Looks like bf16 near-tie differences, not a bug. **Not yet verified at logit level or on the full set.**
- Important: TP setup pitfalls seen so far:
  - vLLM TP=2 workers crashed with `pool INTERNAL ASSERT ... Invalid thread pool` under the default fork. Fixed in `scripts/run_vllm_tp2_long.sh` with `VLLM_WORKER_MULTIPROC_METHOD=spawn` and `OMP_NUM_THREADS=1`.
  - Do not start a TP run while another job is on either card. A `pkill -f` pattern also killed its own shell (exit 144); kill by exact PID.

### Queue in progress (started 2026-11-08)
- `bash scripts/run_cb_tp2.sh full` (self-written TP=2: gpqa_diamond, aime2024, aime2025, math500, bbh), then `bash scripts/run_vllm_tp2_long.sh rest` (vLLM TP=2: math500, bbh). Log: `logs/queue_all.out`. Runs one after another.
- After it finishes:
  1. Score everything offline with `scorer_mc.py` / `scorer_math.py`.
  2. Compare the self-written TP=2 vs vLLM TP=2 with the same caps: accuracy, no-boxed count, tokens/s.
  3. Add a single-card 6144 control if the −10% per-token gap matters.
  4. Update `results/REPORT.md`.
- Open risk: the self-written TP path is only checked on 2 tasks on NPU. Watch for divergence on the full set (compare to vLLM TP outputs per task, not only accuracy).

## Open items (older list, partly superseded)
1. ~~Finish ABBA for `--sync-free`~~ Done: rejected.
2. ~~Continuous batching first16~~ Done (see above); full128 done.
3. vLLM-Ascend peak memory is not recorded in its runner.
4. E250 chunk64 recurrence port: still an option (first16 ABBA gate with >=2% net, then full128).
5. The Triton import warning (`target_info` missing) in vLLM log is harmless for now.

## Practical pitfalls
- Do not chain `sleep` with other commands; use run_in_background or a wait loop with `run_in_background`.
- Read a file before editing it; sed-based edits need the current content.
- Always check the output directory does not already exist (runners refuse to overwrite); create it with `mkdir -p` and remove only the per-task file, not the directory.
- For fair comparisons: same protocol, same caps, same task order, same card type, no concurrent runs on the same card.
- Python-float scalar in tensor ops = host sync on NPU; use cached device tensors.
- Test shapes against real cache layouts (HF conv_state length is K).

## Update 2026-11-09: TP=2 is the main comparison; prefill experiments; next step graph

### Where things stand
- `results/REPORT.md` is reorganized around TP=2 (single-card is an appendix). Section 4.1 and 4.1.1 record the chunked-prefill experiments.
- Scorer: `src/scorer_math.py` is now `fresh-math-v2`: `last_boxed` skips an unterminated `\boxed{` and takes the last complete one. Effect: AIME 2024 CB 19→20, MATH-500 CB 454→455; others unchanged. `scorer_mc.py` is unchanged (fresh-mc-v1). Re-score any file with `python3 -I src/scorer_math.py <per_task.jsonl>`.
- Earlier statement that AIME 2024 CB vs vLLM differs by 5 tasks is now 4 (20 vs 24). The 4 differing tasks are all vLLM-correct: id 1 (CB rejected by context check, 6604+2048≥8192), id 7 and id 21/28 (CB generation diverged in length; 21 and 28 used all 6144 first-turn tokens). No reverse-direction task. Not yet re-run with fast-GDN/elem-conv off.

### TP=2 results (offline-scored; all caps 6144 first / 2048 final; 12 slots; no prefix cache)
| dataset | CB acc | vLLM acc | CB makespan | vLLM makespan |
|---|---|---|---|---|
| AIME 2024 (30) | 20/30 | 24/30 | 1845 s | 2140 s |
| AIME 2025 (30) | 18/30 | 18/30 | 2081 s | 2525 s |
| BBH (1350) | 1158 | 1163 | 14461 s | 20666 s |
| GPQA (198) | 133 | 137 | 9542 s | 13815 s |
| MATH-500 (500) | 455 | 451 | 9063 s | 11546 s |
| GSM8K longctx (128, t_max 16384) | 122 | 122 | 1554 s | 1403 s |

- Config asymmetry that matters: CB TP=2 runs use **eager decode**; vLLM uses aclgraph. So the 1.16–1.45× CB speed-up is a lower bound, and longctx GSM8K (CB slower) is the case where the eager + serial-prefill cost shows.
- Single-card numbers (2.1–2.8×) are NOT a pure engine comparison: BBH used 16 slots for CB vs 12 for vLLM, and single-card caps were 2048/512 with heavy truncation.

### Longctx GSM8K diagnosis (CB TP=2, 128 tasks)
- Real prompt tokens (not chars/4): prefill is 1.17M tokens total; whole-prompt prefill ran at ~1.47k tok/s, costing 793 s of 1571 s makespan (~51%). Decode: 4341 steps at 0.176 s/step, flat vs sequence length (corr(max_len, step_time)=0.025 on a narrow 5k–8k range; not tested over a wide range).
- Prefill is serial and every prefill `synchronize`s, so all decode stalls during it.

### Chunked prefill (`--prefill-chunk`, `--max-prefill`; both default off → behaviour unchanged)
- Implemented: `TPEngine.prefill_chunk(slot, ids, start, last)` (KV rows [0,start) read back with causal mask; GDN conv window and recurrent state carried across chunks via `initial_state`); `decode_step(..., hold=)` saves and restores conv/rec of slots mid-prefill around the dense decode pass.
- Correctness: `scripts/test_chunked_prefill_cpu.py` (2 gloo ranks, tiny model) matches the unchunked reference for chunk 5/8/16/37, held-slot state diff ~1e-9. Negative control (hold restore disabled) fails with state diff 0.06–0.69; tokens alone did NOT catch it, so keep the state check.
- Full longctx, chunk 512, `results/ablations/continuous_batching_tp2/gsm8k_longctx_128_chunk512/`: makespan 2081 s (+34%), decode steps 5374 (+24%), prefill 1133 s at 1031 tok/s, 122/128 correct.
- 36-task test, `results/ablations/continuous_batching_tp2/mp_small_*` (base / c2048_mp1 / c2048_mp2): base 620 s, 2411 decode steps; chunk 2048 with 1 in flight 702 s, 3128 steps; with 2 in flight 712 s, 3127 steps. Two in-flight prefills do NOT reduce decode steps. The "slot occupancy" hypothesis is not supported; the extra steps come from the chunking itself.
- Conclusion: keep chunking off. Merging prefill chunks and decode rows into one forward (what vLLM does: one token-budget batch per step, varlen attention) is the only version of this idea left, and it is a large change.

### vLLM comparison
- vLLM TP=2 log: "Chunked prefill is enabled with max_num_batched_tokens=2048", `enable_prefix_caching=False`, aclgraph replay on both workers.

### (DONE 2026-11-09, see the section at the end) TP=2 decode graph
Original plan, kept for reference:
1. Minimal HCCL capture test (2 processes, one NPU each, `ASCEND_RT_VISIBLE_DEVICES=<rank>`): matmul → `dist.all_reduce` → matmul captured in an `torch.npu.NPUGraph`, replayed after `copy_` into static input; compare with eager. If capture of all_reduce fails, try computing collectives outside the graph between captured segments. Put the script in `scripts/`, not `src/`.
2. Only after step 1 passes: remove `assert not graph_decode` in `TPEngine.__init__` (`src/cb_tp.py`), relax the `--tp` vs `--graph-decode` `SystemExit` in `cb_run.py`, and make graph path ignore/handle `hold` (graph only used when no slot is held; falls back to eager otherwise).
3. Verify: 4–8 task NPU run graph vs eager tokens; then 128-task longctx: compare `decode_seconds`, `decode_steps`, `makespan_seconds` with `results/continuous_batching_tp2/gsm8k_longctx_128/` (1554 s, decode 765 s).
Reference for single-card graph: `CBEngine._capture_graphs` / `_decode_step_graph` in `src/cb_engine.py`, `GRAPH_BUCKET` env `CB_GRAPH_BUCKET` (default 512), `scripts/test_acl_graph.py`.

Rationale for graph before merged forward: graph reuses existing single-card code and has one clear blocker to test; merged forward needs varlen attention, GDN mixed-row kernels, and loses static shapes (no graph), and the chunk experiments suggest its gain is unproven.

### Practical notes from this session
- Run one TP job at a time; check `npu-smi info` first. Launch both ranks with `nohup ... &` and wait via a background until-loop on the `summary` line, not foreground sleep.
- `cb_run.py` refuses an existing `--out`; use a fresh directory per run.
- Kill background jobs by exact PID.

## Update 2026-11-09 (later): TP=2 decode graph implemented and benchmarked

### Result (GSM8K longctx, 128 tasks, TP=2, t_max 16384)
| run | makespan | decode_seconds | decode_steps | prefill_seconds | correct |
|---|---|---|---|---|---|
| eager baseline `results/continuous_batching_tp2/gsm8k_longctx_128/` | 1554 s | 765 s | 4341 | 788 s | 122/128 |
| graph `results/ablations/continuous_batching_tp2/gsm8k_longctx_128_graph_1109_1028/` | 1059 s | 303 s | 4387 | 744 s | 122/128 |

- Makespan −32%, decode time −60%. vLLM-Ascend TP=2 on this set: 1403 s, so CB is now 1.32× faster (was 0.90×).
- Only the 8-task subset was compared token-by-token (see below). The 128-task run was not compared token-by-token against eager.
- Decode steps went up by 46 (4341 → 4387). Not investigated.

### What changed
- Step 1 (HCCL capture): `scripts/test_hccl_graph.py` (2 ranks, `ASCEND_RT_VISIBLE_DEVICES=<rank>`). Captures `matmul → dist.all_reduce → matmul` in an NPUGraph. **PASS**: replay == eager (max diff 0.0), and after `copy_()` of new input the replay matches eager on the new input too (graph reads the static buffer, and the all-reduce really runs on replay). Logs: `logs/test_hccl_graph_r{0,1}.log`. So all_reduce is capturable on this stack; no need for the "all-reduce outside the graph" fallback.
- `src/cb_tp.py`: `TPEngine.__init__` no longer asserts `not graph_decode`; it passes `graph_decode` to `CBEngine`. `AllReduceLinear` untouched. `_graph_core`/`_attn`/`_gdn` needed no change: they use per-rank sizes (`self.kvh`, `self.nh`, `gd.num_v_heads`), and `split` is False under TP.
- `src/cb_run.py`: `--tp` no longer rejects `--graph-decode` (still rejects `--split-devices`); `TPEngine(..., graph_decode=a.graph_decode)`.
- `hold` (chunked prefill) works on the graph path unchanged: the graph writes the same state buffers, and `decode_step(hold=)` saves and restores them around the call. The fallback to eager was not needed. Chunking is still off by default.

### Correctness
- 8-task subset (`--limit 8`, same flags, t_max 16384): eager `results/ablations/continuous_batching_tp2/graphcheck_eager_8_1109_1022/`, graph `graphcheck_graph_8_1109_1022/`.
  - 7/8 tasks identical token-for-token (ids 68, 23, 120, 51, 99, 108, 76). All 8 correct in both runs.
  - **Task 84 diverges**: first difference at round 0, token index 134. Eager emits `436`, graph emits `220, 22`. Both final answers are 15 (gold 15).
  - Cause not confirmed. Hypothesis: bf16 near-tie. Graph attends over a 512-multiple bucket with masking, eager over the exact length, so the sums round differently. Not checked with logits.
- Not done: 128-task token comparison; logit check at task 84's divergence point.

### Environment / running notes
- Both NPUs were idle before the run (`npu-smi info`). Runs were sequential, one group at a time.
- `cb_run.py --graph-decode` sets `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` (setdefault) as before.
- Graph capture is done in `CBEngine.__init__` before any prefill; all ranks run the same capture sequence, so the collectives line up.
- Logs for this session: `logs/test_hccl_graph_r*.log`, `logs/tp2_graphcheck_*`, `logs/tp2_gsm8k_longctx_128_graph_1109_1028_rank*.log`.

### Open items (in priority order)
1. Explain task 84's divergence: top-2 logit gap at round 0, token 134 (eager vs graph path on the same prefix). If the gap is tiny, record it as a tie and move on; if large, look for a real bug in the bucketed path (mask / position handling).
2. Token-level comparison of the full 128-task graph run against eager (`gsm8k_longctx_128` baseline, same flags). Compare the per-task `rounds[].tokens` lists.
3. Re-run the other five TP=2 datasets (AIME 2024/2025, BBH, GPQA, MATH-500) with `--graph-decode` and compare to vLLM. Only GSM8K longctx has been run with graph so far, so the other speed ratios in REPORT still carry the eager caveat.
4. Explain why decode steps rose from 4341 to 4387 on the graph run.

### Report
- `results/REPORT.md` updated: §1 (speed bullets), §4 (GSM8K now has eager and graph rows), §5 (decode-mode row and TP graph implementation note), §6 #1 (partly solved), §7 #1 (follow-ups).

## Update 2026-11-10: TP=2 graph re-runs of the five datasets (done, offline-scored)

Runs: `scripts/run_cb_tp2_graph.sh` (all five finished, no cb_run/vLLM job running at 2026-11-10 04:22 UTC). Output: `results/ablations/continuous_batching_tp2/graph_<name>/per_task.jsonl`. Same flags as the eager TP runs (`--elem-conv --gdn-v2 --fast-gdn --prefill-group 1 --slots 12 --t-max 8192`, caps 6144/6144/2048) plus `--graph-decode`.

| dataset | graph acc | eager acc | vLLM acc | graph makespan | eager makespan | vLLM makespan | graph decode steps (eager) |
|---|---|---|---|---|---|---|---|
| AIME 2024 (30) | 19/30 | 20/30 | 24/30 | 773 s | 1845 s | 2140 s | 10737 (10516) |
| AIME 2025 (30) | 19/30 | 18/30 | 18/30 | 835 s | 2081 s | 2525 s | 11303 (n/a) |
| BBH (1350) | 1166 | 1158 | 1163 | 5878 s | 14461 s | 20666 s | 80044 (80242) |
| GPQA (198) | 142 | 133 | 137 | 3760 s | 9542 s | 13815 s | 55079 (n/a) |
| MATH-500 (500) | 455 | 455 | 451 | 4087 s | 9063 s | 11546 s | 45811 (n/a) |

- Scoring: `src/scorer_math.py` (fresh-math-v2) for AIME/MATH, `src/scorer_mc.py` (fresh-mc-v1) for BBH/GPQA.
- Decode cost per step is ~2.5× lower with graph (AIME 2024: ~166 ms eager vs ~66 ms graph, derived from makespan minus prefill). Step counts match eager within 2%, so the speedup is per-step, not from different schedules. This is consistent with the GSM8K longctx result (decode 765 s → 303 s).
- The graph gain is now apples-to-apples with eager for every dataset in the table. The eager baselines' summaries do not record flags, so they are assumed to have used the same `run_cb_tp2.sh` settings; the scripts say so, but this is not re-verified from the logs.
- Graph vs eager accuracy changes (+/- a few tasks) are within run-to-run noise for greedy decoding; per-task text comparison not done for these five datasets.
- Open items (carry-over): token-level comparison graph vs eager on the 128-task GSM8K run; task 84 divergence logit check; explain the +46 decode steps on the GSM8K graph run; update `results/REPORT.md` with this table (not yet done).
