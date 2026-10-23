# E129 MLP Matmul Performance Upper Bound

Scope: E125 ND fixed replay, four decode steps, MLP-labelled Matmul kernels. FLOPs use `2*M*N*K`; BF16 element size is 2 bytes.

|M/N/K|calls|kernel ms|FLOPs/call (G)|effective TFLOP/s|AI FLOP/B|implied GB/s|memory roof TFLOP/s|efficiency|ideal memory ms|excess ms|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|16/5120/17408|256|40.591|2.852|17.99|15.94|1128.8|20.03|89.8%|36.46|4.13|
|16/17408/5120|512|116.242|2.852|12.56|15.94|788.3|20.03|62.7%|72.92|43.32|

## Same-shape Operation Breakdown
|M/N/K|operation|calls|kernel ms|median us/call|
|---|---|---:|---:|---:|
|16/5120/17408|down_proj|256|40.591|157.96|
|16/17408/5120|gate_proj|256|76.549|303.14|
|16/17408/5120|up_proj|256|39.693|155.16|

## Interpretation
- The same arithmetic shape probe measured 31.34 TFLOP/s, with a contiguous [K,N] right operand. The model Linear weight is stored [N,K] and consumed through a transpose view, so this reference is favorable and not layout-matched.
- The local BF16 M=512 GEMM measured 238.91 TFLOP/s and 512 MiB copy measured 1,256.66 GB/s. These provide empirical compute and bandwidth ceilings, not silicon specifications.
- Official/theoretical 910B3 BF16 peak remains unavailable in the local properties export; cube-core count alone is insufficient to derive it.
- The traffic model assumes one activation read, one full weight read and one output write per call. It is an estimated streaming floor, not measured HBM traffic; L2 residency may lower physical reads.
- Across all projections, 156.83 ms of Matmul kernels compares with a 109.38 ms one-pass streaming floor. The 47.45 ms gap is an opportunity budget, not proven implementation waste.
- Candidate screen: `M=16 N=17408 K=5120` reaches 12.56 TFLOP/s, 40.1% of the measured reference; inspect this shape first, subject to exact wrapper timing.
- Same-shape gate/up medians are 303.14/155.16 us (ratio 1.95x), replicated layer-by-layer. This is an operation/position-correlated difference, not proof that call position or cache behavior causes it.
- Next measurable hypothesis: the 1.95x gap may follow gate's MLP call position or its specific weight. A fixed-activation gate/up order crossover should distinguish those mechanisms; matching gate to up's current median bounds recoverable time at 37.88 ms over these four steps.
