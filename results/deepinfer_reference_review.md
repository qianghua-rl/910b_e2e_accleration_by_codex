# DeepInfer reference review

Reviewed official main at commit `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9` (read-only clone `/tmp/deepinfer-review`). Source: https://github.com/deepseek-ai/DeepInfer . No installation or runtime/model changes.

## Compatibility

README supports DeepSeek-V4.1-Flash, NVIDIA SM100 and Ascend950. Ascend prerequisites CANN9.2/Python3.12; requirements-ascend pins torch2.13.0+cpu and torch-npu2.13.0rc1. Current project910B3/CANN8.5.1/Qwen3.5 differs. Hardware support does not imply950 kernels or full model framework runs on910B3. README performance charts are measured on4/64SM100 GPUs, not950 or our single NPU.

## Mechanisms verified in source

- worker/cuda_graph_runner.py preallocates device input metadata and model output at max batch, narrows buffers by capture batch bucket, reuses graph memory pool, copies metadata and replays. NPU uses transfer_to_npu mapping and handles lack of keep_graph/instantiate. Borrow stable buffers/lifetimes and explicit metadata boundary, not blind whole-HF graph capture.
- worker/worker.py execute_model_step forwards current step, postprocesses previous step, samples current step, and defers current decode postprocessing with an event/WorkerStepState when zero-overhead scheduling enabled. This is actual dependency-aware overlap, not simply removing a synchronization. Finished/EOS, tool boundary and cache ownership must remain correct in any transfer.
- model_executor/layers/linear.py BF16 path accepts out_acc/output buffers, uses32-byte output alignment and flattening, calls deep_gemm.bf16_gemm_nt. FP8 path has casting granularity32 and specialized scale/layout. V4 MLP code is MoE grouped GEMM/shared experts/EP; not our dense Qwen MLP replacement. Explicit output/layout contracts can be borrowed independently of lower precision.
- tilelang_ops/asc/prepare_v4_decode_metadata.py batches metadata preparation on device with persistent/vector-core loops. It is V4 sparse/SWA/block-table specific, not Qwen GDN metadata code.
- platform.py NPU sets TASK_QUEUE_ENABLE=0, expandable segments, internal formats and jit_compile=False. These are framework-specific choices, not evidence that toggling them accelerates HF. E119 did not prove persistent queue backpressure.

## Prioritized transfer experiments, no optimizer implementation yet

1. Dense MLP buffer/format boundary diagnostic: real E100 activations, original BF16 GEMM/rounding, compare complete MLP with allocator-created vs reused output/workspace buffers. Count empty_tensor and layout copies with short trace; check exact logits/state and synchronized plus queued wall/events. Require a meaningful full-wrapper net gain and unchanged ownership; avoid E087 projection packing/E089 quantization retries. DeepInfer's out-buffer/alignment API is concrete design evidence; profitability on910B3 remains unknown.
2. Host metadata/device-resident termination audit: locate only repeated metadata work or scalar dependence already observed inE119, establish generic invariants before caching. Do not hardcode padded-batch False as production logic. Static memory/metadata must be capture-compatible before considering graph; E098/E103 remain unresolved.
3. Dependency-aware asynchronous postprocess: only if measured host postprocessing time is sufficient; E118 showed little between-forward overhead, so lower priority. Need EOS/tool/cache-lifetime proof before any implementation.

Exclude from current project: DSpark/MTP/confidence verification (speculative methods prohibited), V4-specific SWA bounded replay, multi-device EP/EPLB/CP on single910B3, and unsupported950-native precision/kernel paths. Batch invariance/deterministic sampling README explicitly NVIDIA-only; not Ascend guarantee. Do not use release benchmark as our expected speedup.

Conclusion: valuable reference for runtime/buffer boundaries and an executable specialized GEMM interface. It motivates a concrete BF16 complete-MLP memory-boundary experiment, not installing DeepInfer, changing task-queue settings, or promising an end-to-end gain.
