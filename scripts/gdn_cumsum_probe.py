"""Compare one GDN chunk_cumsum stage on captured E100 gate inputs."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch_npu

sys.path.insert(0, "/tmp/megagdn-pto")
from megagdn_pto.kernel_libs import run_chunk_cumsum
from vllm.model_executor.layers.fla.ops.cumsum import chunk_local_cumsum


CHUNK = 64


def python_reference(g):
    batch, length, heads = g.shape
    result = torch.empty_like(g, dtype=torch.float32)
    for row in range(batch):
        for start in range(0, length, CHUNK):
            stop = min(start + CHUNK, length)
            result[row, start:stop] = torch.cumsum(g[row, start:stop].float(), dim=0)
    return result


def hf_reference(g):
    batch, length, _ = g.shape
    pad = (-length) % CHUNK
    padded = torch.nn.functional.pad(g.float(), (0, 0, 0, pad))
    return padded.view(batch, -1, CHUNK, g.shape[-1]).cumsum(dim=2).view(
        batch, -1, g.shape[-1]
    )[:, :length]


def timing(fn, repeats):
    for _ in range(4):
        fn()
    torch.npu.synchronize()
    host_ms = []
    device_ms = []
    for _ in range(repeats):
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        started = time.perf_counter()
        start_event.record()
        fn()
        end_event.record()
        torch.npu.synchronize()
        host_ms.append((time.perf_counter() - started) * 1000)
        device_ms.append(start_event.elapsed_time(end_event))
    return {
        "host_median_ms": statistics.median(host_ms),
        "host_samples_ms": host_ms,
        "device_median_ms": statistics.median(device_ms),
        "device_samples_ms": device_ms,
    }


def peak_delta(fn):
    torch.npu.synchronize()
    baseline = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    value = fn()
    torch.npu.synchronize()
    peak = torch.npu.max_memory_allocated()
    del value
    return max(0, peak - baseline)


def compare(actual, expected):
    delta = (actual.float() - expected.float()).abs()
    return {
        "equal": bool(torch.equal(actual, expected)),
        "max_abs_error": float(delta.max().item()),
        "mean_abs_error": float(delta.mean().item()),
        "finite": bool(torch.isfinite(actual).all()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default="results/gdn_cumsum_e100_batch16.pt")
    parser.add_argument("--output", default="results/gdn_cumsum_probe.json")
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()

    record = torch.load(args.artifact, map_location="cpu", weights_only=False)
    g_cpu = record["g"].float().contiguous()
    batch, length, heads = g_cpu.shape
    py_ref = python_reference(g_cpu)
    hf_ref = hf_reference(g_cpu)
    python_hf = compare(py_ref, hf_ref)

    g = g_cpu.to("npu")
    expected = hf_ref.to("npu")
    cu = torch.arange(batch + 1, dtype=torch.int32, device="npu") * length
    packed = g.reshape(1, batch * length, heads).contiguous()
    packed_out = torch.empty_like(packed, dtype=torch.float32)

    def current():
        return chunk_local_cumsum(g, chunk_size=CHUNK, output_dtype=torch.float32)

    def pto_kernel():
        run_chunk_cumsum(
            packed, packed_out, chunk_size=CHUNK, cu_seqlens=cu,
            batch_size_override=batch,
        )
        return packed_out

    def pto_wrapper():
        output = torch.empty((1, batch * length, heads),
                             device="npu", dtype=torch.float32)
        run_chunk_cumsum(
            packed, output, chunk_size=CHUNK, cu_seqlens=cu,
            batch_size_override=batch,
        )
        return output.view(batch, length, heads)

    current_out = current()
    torch.npu.synchronize()
    pto_out = pto_kernel().view(batch, length, heads)
    torch.npu.synchronize()
    current_check = compare(current_out, expected)
    pto_check = compare(pto_out, expected)

    if not python_hf["equal"] or not current_check["equal"] or not pto_check["equal"]:
        result = {
            "operator": "gdn_prefill.chunk_cumsum",
            "status": "rejected_correctness",
            "chunk_size": CHUNK,
            "python_vs_hf_reference": python_hf,
            "current_vllm_npu": current_check,
            "pto_single_stage_candidate": pto_check,
            "timing": "not run because a stage output differed",
            "e100_capture": {
                "task_ids": record["task_ids"],
                "token_ids": record["token_ids"],
                "prompt_lengths": record["prompt_lengths"],
                "padded_prompt_length": record["padded_prompt_length"],
            },
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        raise SystemExit("cumsum stage output mismatch; stopped before timing")

    # An intentionally loose E100 ceiling: all 16 prefill calls, all 48 GDN
    # layers, and every sequence at the 4096-token model limit.
    long_g = g[:, :1].expand(batch, 4096, heads).contiguous()
    long_cu = torch.arange(batch + 1, dtype=torch.int32, device="npu") * 4096
    long_packed = long_g.reshape(1, batch * 4096, heads).contiguous()
    long_out = torch.empty_like(long_packed, dtype=torch.float32)

    def current_long():
        return chunk_local_cumsum(long_g, chunk_size=CHUNK, output_dtype=torch.float32)

    def pto_long():
        run_chunk_cumsum(
            long_packed, long_out, chunk_size=CHUNK, cu_seqlens=long_cu,
            batch_size_override=batch,
        )
        return long_out

    ceiling_wall_ms = timing(current_long, args.repeats)["host_median_ms"] * 48 * 16
    model_limit_upper_bound_seconds = ceiling_wall_ms / 1000
    e100_fraction_upper_bound = model_limit_upper_bound_seconds / 265.03012829087675

    result = {
        "operator": "gdn_prefill.chunk_cumsum",
        "chunk_size": CHUNK,
        "contract": {
            "input": {"shape": list(g_cpu.shape), "stride": list(g_cpu.stride()),
                      "dtype": str(record["g"].dtype), "layout": "[B,T,H]"},
            "output": {"shape": list(g_cpu.shape), "dtype": "torch.float32",
                       "layout": "[B,T,H]"},
            "state": "no recurrent state input or mutation",
            "cu_seqlens": "None for dense HF/vLLM reference; PTO packs B rows and uses offsets [0,T,...,B*T]",
        },
        "python_vs_hf_reference": python_hf,
        "current_vllm_npu": {
            "correctness": current_check,
            "stage": timing(current, args.repeats),
            "peak_delta_bytes": peak_delta(current),
        },
        "pto_single_stage_candidate": {
            "correctness": pto_check,
            "kernel_with_launch_wrapper": timing(pto_kernel, args.repeats),
            "full_wrapper": timing(pto_wrapper, args.repeats),
            "peak_delta_bytes": peak_delta(pto_wrapper),
            "extra_workspace_bytes": 0,
            "sequence_pack_copy_bytes": 0,
            "cu_seqlens_bytes": int(cu.numel() * cu.element_size()),
        },
        "max_cost_ceiling": {
            "assumption": "baseline cumsum cost removed for all 48 GDN layers in all 16 E100 prefill calls, with every sequence at max_model_len=4096",
            "baseline_cumsum_wall_seconds": model_limit_upper_bound_seconds,
            "fraction_of_E100_makespan": e100_fraction_upper_bound,
            "passes_2_percent_ceiling_gate": e100_fraction_upper_bound >= 0.02,
        },
        "e100_capture": {
            "task_ids": record["task_ids"],
            "token_ids": record["token_ids"],
            "prompt_lengths": record["prompt_lengths"],
            "padded_prompt_length": record["padded_prompt_length"],
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
if __name__ == "__main__":
    main()
