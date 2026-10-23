# E125 NZ gain boundary report

E125-A reproduced the E121 local result with the original captured E100 activations, BF16 weights, `F.linear -> SiLU -> mul -> F.linear` order, four warmups, eight calls per synchronized window, and 20 alternating paired windows. All layer 8/55 outputs were bit-exact and finite.

| layer | boundary | ND host ms | NZ host ms | paired delta |
|---:|---|---:|---:|---:|
| 8 | gate matmul | 0.0978 | 0.0891 | -11.18% |
| 8 | up matmul | 0.0915 | 0.0895 | -5.58% |
| 8 | down matmul | 0.1099 | 0.0918 | -16.15% |
| 8 | complete MLP | 0.4873 | 0.3999 | -17.88% |
| 55 | gate matmul | 0.1029 | 0.0847 | -19.37% |
| 55 | up matmul | 0.0906 | 0.0844 | -8.27% |
| 55 | down matmul | 0.0944 | 0.0866 | -8.80% |
| 55 | complete MLP | 0.4908 | 0.4136 | -15.70% |

The E121 result is therefore reproducible at the captured-wrapper boundary; E124's near-zero complete-wrapper result was a different measurement path/temporal state and must not be used to claim that the local gain disappears.

For E125-C, matched fixed-replay decode profiles used the same first 16 E100 prompts, 12 forced tokens, SDPA, fused norms, single-step recurrent, and identical profiler schedule. Both replay outputs were exact. ND profile wall was 20.606 s and NZ 19.262 s, but profiler wall is instrumented and not an endpoint result. The device trace is more informative: ND had 17,416 kernels and 494,439.6 us total listed duration; NZ also had 17,416 kernels but 510,079.5 us. NZ replaced 768 `aclnnMatmul` launches with 768 `aclnnMatmulWeightNz` launches, while 1,220 regular matmuls remained. NZ weight kernels summed 172,189.9 us; regular matmuls summed 80,591.4 us. ND regular matmuls summed 237,218.4 us. NZ also introduced 45,099.2 us `trans_TransData_5` versus ND's 44,748.7 us `aclnnConvolution_TransDataFzToDst_TransData`, and its total listed device duration was 15,639.9 us higher. This profile does not prove causality for the unprofiled endpoint, but it disproves the assumption that all MLP matmuls switch to NZ or that local kernel savings directly equal model savings.

The first expanded boundary with a clear mismatch is between isolated wrapper timing and full decode execution: the wrapper saves about 16--18%, while the full model trace contains mixed regular/NZ matmuls and a higher total listed device duration. No further implementation is justified until the remaining regular matmuls and layout/transdata ownership are mapped to exact model modules. NZ remains rejected for deployment because E124-A full generation was consistently slower (`+3.718%`, `+7.324%`) despite exact trajectories.

Artifacts: `scripts/e125_nz_boundary_probe.py`, `results/e125_nz_boundaries.jsonl`, `results/e125_profile_nd/`, `results/e125_profile_nz/`, `results/e125_profile_nd.log`, and `results/e125_profile_nz.log`.
