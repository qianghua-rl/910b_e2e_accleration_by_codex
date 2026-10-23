# E120 BF16 complete MLP buffers/layout diagnostic

Inspiration: DeepInfer explicit BF16 GEMM output buffers and layout contract; no DeepInfer code installed and no low precision or packing added. E100 default untouched.

## Inputs and contract

Actual E100 first16 prompts under original HF BF16 SDPA/fused norms/single-step recurrent. Capture layers8 and55 on first3 decode calls: six real BF16 [16,1,5120] activations, stride[5120,5120,1]. Capture not a timing run. Weights loaded from same checkpoint using weight index, not synthetic weights. Artifact `results/e100_mlp_activations.pt` and script `scripts/capture_e100_mlp.py`.

Original complete MLP: gate Linear → BF16 SiLU → up Linear → BF16 multiply → down Linear. Reuse variant retains operation order and rounding boundaries, uses mm(out=), silu.out and mul(out=), persistent four intermediate tensors plus output. No quantization/fusion/reassociation. Final output may alias persistent buffer; separate owned_clone variant includes clone for retained output. Persistence costs2,392,064bytes per layer. No concurrent streams/model patch tested.

Six outputs are bit-exact, finite, same output stride. Alias/lifetime conditions are explicit; this is not a full-model logits acceptance.

## Measurements

Per condition4warmup calls,9windows×8calls, interleaved baseline/reused/baseline/owned_clone/baseline, serial-sync and queued modes. Three captured activations rotate within layer. NPU events include host gaps; no profiler in timed windows. Profiling done separately4calls/variant. Earlier independent gate/up ordering trial retained in initial JSON; final run restores original gate→SiLU→up order.

| Layer | Mode | Adjacent baseline host ms | Candidate host ms |
|---|---|---:|---:|
| 8 | serial/reused | 0.544783 | 0.551312 |
| 8 | serial/reused_owned_clone | 0.546852 | 0.560516 |
| 8 | queued/reused | 0.486613 | 0.490391 |
| 8 | queued/reused_owned_clone | 0.486395 | 0.492598 |
| 55 | serial/reused | 0.544329 | 0.550976 |
| 55 | serial/reused_owned_clone | 0.542761 | 0.559960 |
| 55 | queued/reused | 0.486460 | 0.492686 |
| 55 | queued/reused_owned_clone | 0.486311 | 0.493105 |

Allocation events (aten::empty/empty_tensor/empty_strided) across4calls: baseline20, reuse0, clone4. Each variant still12GEMM kernels plus4SiLU/4multiply. No contiguous/format_cast operators in six final profiles. Original path already has a compatible contiguous layout; transpose views are not materialized format copies. Kernel summed durations remain about1.87–1.90ms/fourcalls, not lower. F.linear lowers aclnnMatmul while mm(out=) lowers aclnnMm, so this tests available output-buffer API as a whole; it is not a pure allocator-internals causal isolation.

## Disposition

Reuse eliminates recorded tensor-allocation events but does not lower complete-wrapper cost in either early/late layer. Even optimistic shared-output mode has no measured gain, and owning-output variant also fails. Reject this implementation at local performance gate: do not spend on model/128tasks. This does not prove all buffer management useless; allocation may already be cached/overlapped and this API still has dispatch/view costs, but their exact contribution was not independently measured. No claim of bandwidth saturation or universal regression.

DeepInfer's performance cannot be attributed to buffers alone: specialized GEMM, runtime and hardware differ. A new attempt needs a different mechanism and evidence (e.g. actual recurring conversion or a faster compatible GEMM), not repeating preallocation or prior INT8/projection-packing failures. Target remains E100 makespan/quality, not minimizing allocation-event count.

Evidence: `results/mlp_buffer_probe.json`, `results/mlp_buffer_probe_analysis.json`, final ordered trace directories/log, `scripts/mlp_buffer_probe.py`, `scripts/analyze_mlp_buffer_probe.py`. No full model performance/quality result claimed.
