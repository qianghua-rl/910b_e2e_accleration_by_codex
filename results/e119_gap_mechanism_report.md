# E119 specific gap mechanisms

No graph or optimizer was enabled in E100. Existing four fixed-replay E118 traces were analyzed with exact async_npu/HostToDevice flow endpoints. CPU intervals are disjointed by shortest enclosing recorded main-thread event. This is temporal attribution, not proof of critical-path savings.

## Timestamp partition, four decode steps

| Condition | Idle gaps ms | Before next host flow ms | Host flow→runtime marker ms | Marker→device ms |
|---|---:|---:|---:|---:|
| 8a | 218.681 | 187.170 | 29.127 | 2.385 |
| 1 | 237.488 | 183.801 | 51.521 | 2.166 |
| 4 | 202.286 | 174.610 | 25.392 | 2.284 |
| 8b | 247.238 | 202.147 | 42.776 | 2.315 |

Excluding EVENT pseudo entries causes small differences from profiler step Free totals. HostToDevice marker is a runtime correlation point, not proven hardware acceptance. Slightly inverted clock ordering is clipped, not interpreted as negative latency.

## Located mechanisms and evidence strength

1. **Host producer preparation/dispatch chain.** In 8a,187.17ms idle occurs before the next host flow source. Recorded disjoint host activities during total idle include empty_tensor24.33ms, npu_rms_norm23.81ms, aten matmul8.33ms, copy8.17ms, views/as_strided7.08ms and cat6.71ms; unrecorded host68.56ms remains. These are host execution stages, not kernel stalls. Allocation events do not prove physical HBM allocation; custom-op spans include validation/dispatch. Unrecorded time includes Python, scheduler, profiler and omitted calls: do not label it pure Python overhead.
2. **Variable producer→runtime submission delay.** Timestamp-clipped interval25.4–51.5ms across conditions. It can include queueing/runtime preparation. The largest8a1.508ms gap overlaps a1.989ms Enqueue@Convolution, but only one enqueue exceeds100us out of14,884. Therefore persistent queue backpressure is NOT established; do not tune queue settings from one outlier.
3. **Specific synchronous scalar read barriers.** Four-step trace has1248 local_scalar operations but only8 is_nonzero calls. Python interception on real E100 prompts identifies one SDPA `_ignore_causal_mask_sdpa` padding_mask.all() boolean and one HF generation `_has_unfinished_sequences` this_peer_finished boolean per decode. The other scalar operations cannot all be classified as NPU sync. Device bool conversion requires a Python-visible result; confirmed call stacks are in e119_scalar_stacks.json. Four-step is_nonzero inclusive host sum2.343ms; it is not all idle or all removable. Per-generation special-token/right-padding checks occur at startup, not each layer.
4. **After-runtime-marker delay is small in this partition**, about2.2–2.4ms idle across four steps. This argues against attributing the entire200+ms idle to already-submitted device dependency waits in these traces, but does not prove driver latency or dependencies are absent.

## Discriminating intervention

For this one fixed left-padded replay, attention-mask zeros persist as tokens append. Replaced only SDPA mask-skip eligibility with known False; retained actual causal mask and all math/cache operations. Five runs reference/cached_mask/reference/cached_mask/reference, fresh cache, same forced12tokens, 11decode forwards. Logits and tokens exact in every run. Candidate host spans1591.223ms and1484.863ms versus adjacent reference averages1578.297ms and1595.606ms: one regresses0.82%, one improves6.94%; no stable deployable gain. This intervention is diagnostic-only and not valid for arbitrary unpadded inputs; no model patch enabled. Timing noise prevents quantitative causal savings, even though the read location is proven.

## Decision

Do not blindly capture graph. First address/measure a specific stable producer segment and its allocations/layout/argument work, with cache-lifetime and wrapper budget; or remove a confirmed host read using general invariants and verify graph eligibility separately. Existing E098/E103 failure remains relevant, but this experiment does not prove those exact failures originate in these generation/mask booleans. The identified two per-step booleans do not explain most gaps and should not become a high-benefit optimization claim. No pure queue-backpressure, HBM-bound or full graph benefit is established.

This diagnosis locates concrete recurring host stages and two source-level sync mechanisms, while leaving68.56ms unrecorded host activity unresolved. It completes the specific gap-mechanism investigation at that evidence scope, not full causal accounting of every gap or the E100 optimization project.
