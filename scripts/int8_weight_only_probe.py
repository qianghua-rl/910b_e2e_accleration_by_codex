"""Reproducible BF16-activation / symmetric INT8-weight NPU matmul probe."""
import json
import time
from pathlib import Path
from statistics import median

import torch
import torch_npu
from safetensors import safe_open


torch.manual_seed(42)
torch.npu.set_device(0)
torch.set_num_threads(8)

WEIGHT_KEY = "model.language_model.layers.8.mlp.gate_proj.weight"
WEIGHT_FILE = "model/qwen3.5-27b/model.safetensors-00002-of-00011.safetensors"
records = []


def median_call(fn, repeats=8):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    samples = []
    output = None
    for _ in range(repeats):
        started = time.perf_counter()
        output = fn()
        torch.npu.synchronize()
        samples.append(time.perf_counter() - started)
    return output, median(samples), samples


with torch.inference_mode():
    with safe_open(WEIGHT_FILE, framework="pt", device="cpu") as checkpoint:
        weight_cpu = checkpoint.get_tensor(WEIGHT_KEY)
    n, k = weight_cpu.shape
    weight = weight_cpu.to(device="npu", dtype=torch.bfloat16)
    weight_kn = weight.float().t().contiguous()
    scale = weight_kn.abs().amax(dim=0).clamp_min(1e-12) / 127.0
    qweight = (weight_kn / scale).round().clamp(-127, 127).to(torch.int8)
    scale_bf16 = scale.to(torch.bfloat16)
    for m in (1, 16):
        x = torch.randn((m, k), device="npu", dtype=torch.bfloat16)

        def reference():
            return torch.nn.functional.linear(x, weight)

        def quantized():
            return torch_npu.npu_weight_quant_batchmatmul(
                x, qweight, scale_bf16, None, antiquant_group_size=0
            )

        y_ref, ref_median, ref_samples = median_call(reference)
        y_quant, quant_median, quant_samples = median_call(quantized)
        error = y_quant.float() - y_ref.float()
        records.append({
            "m": m,
            "k": k,
            "n": n,
            "weight_key": WEIGHT_KEY,
            "activation_dtype": str(x.dtype),
            "weight_format": "symmetric_int8_per_output_channel",
            "scale_shape": list(scale_bf16.shape),
            "reference_median_s": ref_median,
            "reference_samples_s": ref_samples,
            "quantized_median_s": quant_median,
            "quantized_samples_s": quant_samples,
            "relative_rms": float(error.square().mean().sqrt() / y_ref.float().square().mean().sqrt()),
            "cosine": float(torch.nn.functional.cosine_similarity(
                y_quant.float().flatten(), y_ref.float().flatten(), dim=0
            )),
            "max_abs_error": float(error.abs().max()),
            "argmax_equal": bool(torch.equal(y_quant.argmax(-1), y_ref.argmax(-1))),
            "top1_match_fraction": float((y_quant.argmax(-1) == y_ref.argmax(-1)).float().mean()),
            "quantized_weight_bytes": qweight.numel() * qweight.element_size(),
            "bf16_weight_bytes": weight.numel() * weight.element_size(),
        })

out = {
    "device": torch.npu.get_device_name(0),
    "operator": "torch_npu.npu_weight_quant_batchmatmul",
    "quantization": "symmetric per-output-channel INT8, BF16 antiquant scales",
    "input_distribution": "fixed-seed standard-normal BF16 synthetic activations",
    "records": records,
}
Path("results/int8_weight_only_probe_real_weight.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2), flush=True)
