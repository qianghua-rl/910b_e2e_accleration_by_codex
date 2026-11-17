"""Why is fused router slower? Time HF Gemma4TextRouter.forward vs gemma_cb.router_fused
on the same hidden states (N=16 rows, real layer weights), plus dtype/shape info.

Usage: probe_router_fused.py <out_dir> --device npu:1 [--iters 200]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch
import torch_npu  # noqa: F401
from transformers import AutoModelForCausalLM

from gemma_cb import router_fused
from gemma_fast_norm import patch_gemma_norms
from gemma_run import MODEL


def timeit(fn, x, iters, dev):
    for _ in range(10):
        fn(x)
    torch.npu.synchronize(dev)
    t = time.perf_counter()
    for _ in range(iters):
        fn(x)
    torch.npu.synchronize(dev)
    return 1000 * (time.perf_counter() - t) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--device", default="npu:1")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--rows", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dev = a.device
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map=dev).eval()
    patch_gemma_norms(model)
    layer = next(l for l in model.model.language_model.layers if getattr(l, "enable_moe_block", False))
    r = layer.router
    r._fold = (r.scale * r.scalar_root_size).detach()
    x = torch.randn(a.rows, model.config.get_text_config().hidden_size,
                    dtype=torch.bfloat16, device=dev)

    with torch.no_grad():
        ref = r(x)
        fused = router_fused(r, x)
        idx_eq = bool((ref[2] == fused[2]).all().item())
        w_diff = (ref[1].float() - fused[1].float()).abs().max().item()

        t_hf = timeit(lambda z: r(z), x, a.iters, dev)
        t_fu = timeit(lambda z: router_fused(r, z), x, a.iters, dev)

        # Ops in the fused version, one call each, to see dtypes and any host syncs.
        h = r.norm(x)
        info = {
            "fold_dtype": str(r._fold.dtype), "fold_shape": list(r._fold.shape),
            "norm_out_dtype": str(h.dtype),
            "scale_dtype": str(r.scale.dtype), "per_expert_scale_dtype": str(r.per_expert_scale.dtype),
        }
        # Does the fused path force a host sync? Count .item() style calls via a sync-probe.
        torch.npu.synchronize(dev)
        t = time.perf_counter()
        _ = r.norm(x) * r._fold
        torch.npu.synchronize(dev)
        info["norm_times_fold_ms"] = round(1000 * (time.perf_counter() - t), 3)

    res = {"rows": a.rows, "iters": a.iters, "hf_router_ms": round(t_hf, 4),
           "fused_router_ms": round(t_fu, 4), "idx_equal": idx_eq,
           "weights_max_absdiff": w_diff, **info}
    with open(os.path.join(a.out_dir, "router_probe.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("ROUTER_PROBE", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
