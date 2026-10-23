# E115 E100 early-decode device attribution

Configuration: HF E100 BF16/SDPA batch16, fused norms, single-step recurrent; **4 early decode steps** after prefill/warmup. This is early-decode device attribution, not steady-state decode and not an end-to-end latency breakdown. Source: `results/profile_e100_decode_attributed/`; machine-readable output: `results/e100_decode_attribution.json`. No optimization was enabled.

| Category | Device attribution / 4 steps | Current conclusion |
|---|---:|---|
| MLP Matmul | 157.407 ms / 768 kernels | Largest identified compute hotspot |
| GDN convolution family | 116.823 ms / 1,344 kernels | 192 Conv2D kernels (55.776 ms) plus 1,152 adjacent TransData/MemSet kernels (61.048 ms); exact module ownership is not confirmed |
| Attention / GDN projection | 71.687 ms / 1,216 kernels | Projection-related hotspot |
| Recurrent Mul | 43.200 ms / 1,728 kernels | State operation |
| Recurrent ReduceSum | 14.776 ms / 768 kernels | State operation |
| Recurrent copy | 9.113 ms / 1,344 kernels | State/layout movement; not all copies are proven cache writes |
| LM head | 8.448 ms / 4 kernels | Output projection |

These category values are profiler device durations from the same four early decode steps. Categories can overlap in execution or attribution scope and must not be added into an endpoint latency decomposition.

MLP/projection attribution uses labelled Linear forwards and exact profiler `torch_to_npu` flow endpoints, not temporal adjacency. Other unlabelled matmul kernels remain in `other` in JSON. Recurrent source uses elementwise products and reductions, not matmul; its listed components total 76.751 ms including Add 5.984 ms. Counts cover lowered kernels, not necessarily Python calls.

## GDN convolution follow-up

`kernel_details.csv` contains 192 `aclnnConvolution_Conv2dWithFlag_Conv2D` kernels totaling 55.776 ms. Each convolution call is surrounded by a repeated family of layout operations: 384 `aclnnConvolution_TransData_TransData` kernels (7.947 ms), 192 `...TransDataToFzWithoutGroup_MemSet` (2.193 ms), 192 `...TransDataToFzWithoutGroup_TransData` (5.103 ms), 192 `...TransDataFzToDst_MemSet` (1.065 ms), and 192 `...TransDataFzToDst_TransData` (44.739 ms). The family totals 116.823 ms; TransData/MemSet alone totals 61.048 ms.

The trace shows these operations adjacent to each convolution, consistent with input/output layout conversion around the convolution. The device names do not contain a GDN layer or module identifier, so the mapping is currently only to the convolution call family, with module ownership unconfirmed. This evidence separates convolution execution from conversion overhead but does not show that any conversion is redundant or safely removable. No 2% end-to-end savings ceiling is established; do not start a convolution implementation experiment on this evidence alone.

State: FP32 `[16,48,128,128]` = 48 MiB per GDN layer. Source-level eager accounting estimates 12 state-sized reads/writes for decay, memory product/reduction, outer update/add and output product/reduction (vectors/casts omitted): 576 MiB per layer, 27 GiB per 48-layer decode step. This is logical tensor traffic, not measured HBM bytes or achieved bandwidth.

Launch/host: recurrent CPU marked inclusive duration 201.993 ms over four steps; ACLNN dispatch intervals sum about 29.86 ms. These are nested and must not be added together or to device time. Async flow delay includes queued work and device backlog, not pure launch latency. Device union 495.062 ms within a 952.157 ms window; original E113 unlabelled profile union 493.910 ms within a 710.295 ms window. Neither idle gap is a removable-work estimate.

Conclusion: the report identifies mixed costs, with MLP matmul largest, GDN convolution-family work second, and recurrent state operations separately measurable. The convolution follow-up finds substantial adjacent layout work, but neither its redundancy nor module ownership is proven. Continue only if a future read-only mapping or controlled A/B can establish an avoidable conversion/recomputation and a credible >=2% end-to-end ceiling; otherwise stop this branch. No E100 speedup is claimed.
