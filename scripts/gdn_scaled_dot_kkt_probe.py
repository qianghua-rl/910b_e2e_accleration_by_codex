"""Compare PTO scaled_dot_kkt with FP32 and standalone Ascend Triton references."""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu


CHUNK = 64
E100_SECONDS = 265.03012829087675
E100_GDN_PREFILL_CALLS = 16
GDN_LAYERS = 48


def python_cumsum(g):
    batch, length, heads = g.shape
    result = torch.empty_like(g, dtype=torch.float32)
    for row in range(batch):
        for start in range(0, length, CHUNK):
            stop = min(start + CHUNK, length)
            result[row, start:stop] = torch.cumsum(g[row, start:stop].float(), dim=0)
    return result


def python_kkt(key, beta, g_sum):
    batch, length, heads, _ = key.shape
    result = torch.zeros(batch, length, heads, CHUNK, dtype=torch.float32)
    for row in range(batch):
        for start in range(0, length, CHUNK):
            stop = min(start + CHUNK, length)
            valid = stop - start
            k = key[row, start:stop].float().permute(1, 0, 2)
            gram = torch.matmul(k, k.transpose(-1, -2))
            gates = g_sum[row, start:stop].float().transpose(0, 1)
            decay = torch.exp(gates[:, :, None] - gates[:, None, :])
            b = beta[row, start:stop].float().transpose(0, 1)
            a = gram * decay * b[:, :, None]
            a = torch.tril(a, diagonal=-1)
            result[row, start:stop, :, :valid] = a.permute(1, 0, 2)
    return result


def compare(actual, expected):
    delta = (actual.float() - expected.float()).abs()
    denom = expected.float().abs().clamp_min(1e-5)
    max_abs_error = float(delta.max().item())
    return {
        "equal": max_abs_error == 0.0,
        "max_abs_error": max_abs_error,
        "mean_abs_error": float(delta.mean().item()),
        "max_relative_error": float((delta / denom).max().item()),
        "finite": bool(torch.isfinite(actual).all()),
    }


def event_timing(fn, repeats):
    for _ in range(1):
        fn()
    torch.npu.synchronize()
    host_ms = []
    device_ms = []
    for _ in range(repeats):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        t0 = time.perf_counter()
        start.record()
        fn()
        end.record()
        torch.npu.synchronize()
        host_ms.append((time.perf_counter() - t0) * 1000)
        device_ms.append(start.elapsed_time(end))
    return {
        "host_median_ms": statistics.median(host_ms),
        "device_median_ms": statistics.median(device_ms),
        "host_samples_ms": host_ms,
        "device_samples_ms": device_ms,
    }


def peak_delta(fn):
    torch.npu.synchronize()
    base = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    value = fn()
    torch.npu.synchronize()
    peak = torch.npu.max_memory_allocated()
    del value
    return max(0, peak - base)


def reference_kkt_fn(k, beta, g_sum):
    # This standalone copy is based on FLA's Ascend implementation and avoids
    # importing vllm.triton_utils or the FLA package runtime.
    from gdn_scaled_dot_kkt_reference import chunk_scaled_dot_kkt_fwd

    return lambda: chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g_cumsum=g_sum,
        chunk_size=CHUNK,
        output_dtype=torch.float32,
    )


def pto_setup(k, beta, g_sum, cu):
    from megagdn_pto.kernel_libs import (
        chunk_gdn_causal_masks,
        load_scaled_dot_kkt,
        transpose_beta,
        transpose_gates,
    )
    from megagdn_pto.compile import BLOCK_DIM

    packed_k = k.reshape(1, -1, k.shape[2], k.shape[3]).to(torch.float16).contiguous()
    packed_beta = beta.reshape(1, -1, beta.shape[2]).to(torch.float16).contiguous()
    packed_g = g_sum.reshape(1, -1, g_sum.shape[2]).contiguous()
    cu32 = cu.to(torch.int32).contiguous()
    g_t = transpose_gates(packed_g)
    beta_t = transpose_beta(packed_beta)
    mask, _ = chunk_gdn_causal_masks("npu", k.device.index or 0, CHUNK)
    bd = BLOCK_DIM
    workspace = torch.zeros(bd * 2, CHUNK, CHUNK, device=k.device, dtype=torch.float16)
    out = torch.empty(
        1, packed_k.shape[1], packed_beta.shape[2], CHUNK,
        device=k.device, dtype=torch.float16,
    )
    lib = load_scaled_dot_kkt(k.shape[-1], CHUNK)
    stream = torch.npu.current_stream()._as_parameter_

    def direct_kernel():
        lib.call_kernel(
            bd, stream, packed_k.data_ptr(), beta_t.data_ptr(), g_t.data_ptr(),
            mask.data_ptr(), workspace.data_ptr(), out.data_ptr(), cu32.data_ptr(),
            cu.numel() - 1, packed_k.shape[1], packed_g.shape[1],
            packed_beta.shape[2], packed_k.shape[2],
        )
        return out.view(k.shape[0], k.shape[1], beta.shape[2], CHUNK)

    def wrapper():
        from megagdn_pto.kernel_libs import run_scaled_dot_kkt

        fresh_out = torch.empty_like(out)
        run_scaled_dot_kkt(
            packed_k, packed_beta, packed_g, mask, fresh_out,
            g_t=g_t, beta_t=beta_t, chunk_size=CHUNK, cu_seqlens=cu32,
            batch_size_override=cu.numel() - 1, key_heads=packed_k.shape[2],
        )
        return fresh_out.view(k.shape[0], k.shape[1], beta.shape[2], CHUNK)

    def prepare_inputs():
        packed_key = k.reshape(1, -1, k.shape[2], k.shape[3]).to(torch.float16).contiguous()
        packed_b = beta.reshape(1, -1, beta.shape[2]).to(torch.float16).contiguous()
        packed_gate = g_sum.reshape(1, -1, g_sum.shape[2]).contiguous()
        gate_t = transpose_gates(packed_gate)
        beta_t_prepared = transpose_beta(packed_b)
        return packed_key, packed_b, packed_gate, gate_t, beta_t_prepared

    def full_wrapper():
        from megagdn_pto.kernel_libs import run_scaled_dot_kkt

        packed_key, packed_b, packed_gate, gate_t, beta_t_prepared = prepare_inputs()
        full_out = torch.empty(
            1, packed_key.shape[1], packed_b.shape[2], CHUNK,
            device=k.device, dtype=torch.float16,
        )
        run_scaled_dot_kkt(
            packed_key, packed_b, packed_gate, mask, full_out,
            g_t=gate_t, beta_t=beta_t_prepared, chunk_size=CHUNK,
            cu_seqlens=cu32, batch_size_override=cu.numel() - 1,
            key_heads=packed_key.shape[2],
        )
        return full_out.view(k.shape[0], k.shape[1], beta.shape[2], CHUNK)

    return direct_kernel, wrapper, prepare_inputs, full_wrapper, out


def main():
    # The FP32 reference consists of many small batched CPU matmuls. Avoid
    # oversubscribing large host thread pools for these small tiles.
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact", "--input", dest="artifact",
        default="results/gdn_cumsum_e100_batch16.pt",
        help="capture artifact containing g, key, and beta tensors",
    )
    parser.add_argument("--output", default="results/gdn_scaled_dot_kkt_probe.json")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--skip-ceiling", action="store_true")
    parser.add_argument("--ceiling-batch", type=int, default=None)
    args = parser.parse_args()

    record = torch.load(args.artifact, map_location="cpu", weights_only=False)
    if "key" not in record or "beta" not in record:
        raise SystemExit("capture artifact lacks key/beta; rerun gdn_cumsum_capture.py")
    key_cpu = record["key"].contiguous()
    beta_cpu = record["beta"].contiguous()
    g_cpu = record["g"].float().contiguous()
    py_g_sum = python_cumsum(g_cpu)
    print("prepared FP32 gate cumsum", flush=True)
    py_kkt = python_kkt(key_cpu, beta_cpu, py_g_sum)
    print("prepared FP32 KKT reference", flush=True)

    key = key_cpu.to("npu")
    beta = beta_cpu.to("npu")
    g_sum = py_g_sum.to("npu")
    batch, length, heads, dim = key.shape
    cu = torch.arange(batch + 1, dtype=torch.int32, device="npu") * length
    reference_fn = reference_kkt_fn(key, beta, g_sum)
    print("launching independent Ascend reference", flush=True)
    reference = reference_fn()
    torch.npu.synchronize()
    print("independent Ascend reference completed", flush=True)
    candidate_direct, candidate_wrapper, pto_prepare, pto_full_wrapper, _ = pto_setup(
        key, beta, g_sum, cu
    )
    candidate = candidate_wrapper()
    torch.npu.synchronize()
    print("PTO KKT completed", flush=True)

    print("comparing reference against FP32", flush=True)
    python_compare = compare(reference, py_kkt.to("npu"))
    print("comparing PTO against reference", flush=True)
    candidate_compare = compare(candidate, reference)
    print("correctness comparisons completed", flush=True)
    python_tolerance = 2e-3
    current_python_ok = python_compare["max_abs_error"] <= python_tolerance
    candidate_stage_ok = candidate_compare["max_abs_error"] <= python_tolerance
    if not current_python_ok or not candidate_stage_ok or not candidate_compare["finite"]:
        result = {
            "operator": "gdn_prefill.scaled_dot_kkt",
            "status": "rejected_correctness",
            "python_vs_reference": python_compare,
            "candidate_vs_reference": candidate_compare,
            "tolerance": python_tolerance,
            "timing": "not run because correctness gate failed",
            "e100_token_ids": record["token_ids"],
        }
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        raise SystemExit("scaled_dot_kkt correctness gate failed")

    print("timing independent Ascend reference", flush=True)
    current_input_timing = event_timing(reference_fn, args.repeats)
    print("timing PTO direct kernel", flush=True)
    pto_kernel_timing = event_timing(candidate_direct, args.repeats)
    print("timing PTO preparation", flush=True)
    pto_prepare_timing = event_timing(pto_prepare, args.repeats)
    print("timing full PTO wrapper", flush=True)
    pto_wrapper_timing = event_timing(pto_full_wrapper, args.repeats)
    print("measuring PTO peak allocation", flush=True)
    pto_peak = peak_delta(pto_full_wrapper)

    # Generous ceiling uses max_model_len for every sequence and ignores all
    # conversion and downstream costs. It is a screening upper bound only.
    max_t = 4096
    ceiling_batch = args.ceiling_batch or batch
    if args.skip_ceiling:
        long_reference_ms = None
        ceiling_seconds = None
        ceiling_fraction = None
    else:
        long_key = key[:ceiling_batch, :1].expand(ceiling_batch, max_t, heads, dim).contiguous()
        long_beta = beta[:ceiling_batch, :1].expand(ceiling_batch, max_t, heads).contiguous()
        long_g = g_sum[:ceiling_batch, :1].expand(ceiling_batch, max_t, heads).contiguous()
        print(f"timing 4096-token reference for ceiling batch={ceiling_batch}", flush=True)
        long_reference_fn = reference_kkt_fn(long_key, long_beta, long_g)
        long_reference_ms = event_timing(long_reference_fn, max(1, args.repeats // 2))["device_median_ms"]
        print("4096-token ceiling timing completed", flush=True)
        ceiling_seconds = long_reference_ms * GDN_LAYERS * E100_GDN_PREFILL_CALLS / 1000
        ceiling_fraction = ceiling_seconds / E100_SECONDS

    result = {
        "operator": "gdn_prefill.scaled_dot_kkt",
        "status": "stage_candidate_passed_ceiling_pending_model_gate" if ceiling_fraction is not None and ceiling_fraction > 0.02 else ("rejected_low_ceiling" if ceiling_fraction is not None else "correctness_passed_ceiling_skipped"),
        "contract": {
            "key": {"shape": list(key_cpu.shape), "stride": list(key_cpu.stride()), "dtype": str(key_cpu.dtype)},
            "beta": {"shape": list(beta_cpu.shape), "stride": list(beta_cpu.stride()), "dtype": str(beta_cpu.dtype)},
            "g": {"shape": list(g_cpu.shape), "stride": list(g_cpu.stride()), "dtype": str(g_cpu.dtype)},
            "output": {"shape": list(reference.shape), "dtype": str(reference.dtype), "layout": "[B,T,H,C=64]"},
            "state": "stateless stage",
            "cu_seqlens": f"PTO packed rows [0,{length},...,{batch * length}], int32; reset chunk index at each sequence",
            "workspace": "PTO [2*BLOCK_DIM,64,64] FP16 plus A output FP16",
        },
        "python_fp32_reference": python_compare,
        "pto_vs_reference": candidate_compare,
        "numerical_tolerance": python_tolerance,
        "independent_ascend_triton_reference": {
            "implementation": "scripts/gdn_scaled_dot_kkt_reference.py (standalone FLA Ascend Triton port)",
            "gate_argument": "g_cumsum",
            "timing": current_input_timing,
        },
        "pto_candidate": {
            "kernel_launch_timing": pto_kernel_timing,
            "full_wrapper_timing": pto_wrapper_timing,
            "peak_delta_bytes": pto_peak,
            "conversion": "BF16 key/beta to FP16; g_sum stays FP32; pack dense rows and transpose gate metadata",
        },
        "cost_ceiling": {
            "shape": [ceiling_batch, max_t, heads, dim],
            "reference_device_ms_per_prefill_call": long_reference_ms,
            "reference": "independent FLA Ascend Triton backend, not the installed vLLM kernel",
            "assumption": "all 48 GDN layers across 16 prefill calls run at max_model_len and the full KKT stage cost is removable",
            "seconds": ceiling_seconds,
            "fraction_of_E100": ceiling_fraction,
            "passes_2_percent": None if ceiling_fraction is None else ceiling_fraction > 0.02,
        },
        "e100_reference": {
            "task_ids": record["task_ids"],
            "token_ids": record["token_ids"],
            "prompt_lengths": record["prompt_lengths"],
        },
        "model_logits_and_token_replay": "not run by stage probe",
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
