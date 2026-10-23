# E121 bounded BF16 GEMM capability audit

Scope: installed910B3/CANN8.5.1/torch2.9.1+cpu/torch_npu2.9.1. No dependency upgrades, custom GEMM implementation or model patch. Current BF16 only; no quantization, projection packing or multidevice communication. Local real E100 layers8/55 inputs reused from capture, three decode samples each.

| Path | Evidence | Disposition |
|---|---|---|
| F.linear vs npu_linear | E083 same real shapes exact but no gain | Do not repeat |
| mm(out=) output reuse | E120 no full-wrapper gain | Do not repeat |
| JIT off | is_jit_compile_false() already True | Not a new optimization |
| physical K×N contiguous transpose | Both layers execute; ~0.463–0.465ms vs~0.477ms; output differs up to0.00390625 late layer | Lower priority, not equivalent |
| persistent FRACTAL_NZ weights | Actual format29; complete MLP~0.397–0.398ms vs~0.477ms; all6sample outputs exact | New eligible candidate for model gate |
| fused_matmul | Installed schema/header exists; op proto lists add/mul/GELU, no SiLU; BF16 note says Ascend910_95 | Not validated on910B3; no guessed fusion mode test |
| transpose_batchmatmul | Installed docs specify A2/A3 BF16,3D transpose/batch semantics | No matching unavoidable materialized transpose in current denseMLP; no reason to benchmark alias alone |
| grouped matmul | Prior real projection grouping slower/numerically changed | Not a new route merely renaming grouping |
| all_reduce/all_to_all GEMM | Multi-device communication path | Outside single-NPU scope |
| Ascend950 DeepInfer/DeepGEMM path | Requires different hardware/software and model | Not direct capability evidence |

## NZ proof

Baseline full operation order retained: gate Linear→SiLU→up Linear→multiply→down Linear, BF16 throughout. Three weights converted once via npu_format_cast(w,29), explicit allow_internal_format=True. Device format query confirms29. Same original ND activations. Layer8 baseline0.47731ms/NZ0.39812ms. Layer55 interleaved baseline0.47727/NZ0.39693/baseline0.47720/NZ0.39665ms. Measured queued full wrapper includes Python F.linear/activation/multiply/output allocation, warmup then7windows×8calls; event-kernel/whole-model results not inferred from this timing.

Layer55 profile four MLP calls:12aclnnMatmulWeightNz kernels summing1527.76us,4SiLU12.44us,4Mul13.14us. No format_cast/contiguous ops during execution. This is a distinct actual CANN GEMM path, not only fewer allocation events. All three outputs each layer exact/finite; input/output model logits not yet validated.

Setup includes initial format kernel compilation: layer8~15.41s, layer55~15.15s; transpose setup~155ms initial. Do not call these steady-state conversion costs or extrapolate all-layer startup linearly. Probe holds ND+NZ copies, unlike eventual in-place model conversion; no production memory/peak result. E100 model peak is already close to memory budget.

## Decision and gates

Prioritize ONE next candidate: persistent BF16 MLP NZ weight format, retaining original forward and all other model math. Preparation outside timed generation, track setup separately and total request makespan boundary. First confirm memory-safe conversion/lifetimes on full model, prefill/decode compatibility, exact logits and complete greedy tool-round trajectory on matched16tasks. Then repeated unprofiled A/B incl peak; only winner expands128/audit/scorer. Local0.080ms×64MLP≈5.1ms/decode step is merely a conditional screen (not a measured model saving), near the4.214ms2%target; residency, prefills and host changes can erase it. Explicitly do not adopt from local16.8% alone.

Sources: installed torch_npu schemas/docs/native YAML; `/usr/local/Ascend/cann-8.5.1/opp/built-in/op_proto/inc/ops_proto_nn.h` FusedMatMul product/type/fusion comment; CANN include/aclnnop headers. Headers/schema presence alone do not prove hardware support. Artifacts: bf16_gemm_capability_initial.json, bf16_gemm_capability_layer55.json, profile_bf16_nz_layer55/, scripts/bf16_gemm_capability_probe.py. No defaults changed.
