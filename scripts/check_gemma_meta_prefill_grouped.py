"""Metadata check: does --prefill-grouped change the tensors seen by the decode path?

Run once without and once with --prefill-grouped (separate processes), then diff the JSON.
Records dtype / device / shape / stride / contiguity for, per MoE layer:
  - decode: inputs to experts_grouped (xin, idx, w) and its output (index_add_ result),
    and the layer input x, on the first decode step
  - prefill: inputs to the HF experts module (captured by a forward pre-hook, which fires even
    when ex.forward is replaced)

Usage: check_gemma_meta_prefill_grouped.py <out_json> --device npu:1 [--prefill-grouped]
"""
import argparse
import json
import sys

sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")

import torch
import torch_npu  # noqa: F401
from transformers import AutoModelForCausalLM

import gemma_cb
from gemma_run import MODEL
from gemma_cb import GemmaCB
from gemma_fast_norm import patch_gemma_norms

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--device", default="npu:1")
ap.add_argument("--prefill-grouped", action="store_true")
a = ap.parse_args()


def meta(t):
    return {"dtype": str(t.dtype), "device": str(t.device), "shape": list(t.shape),
            "stride": list(t.stride()), "contiguous": bool(t.is_contiguous())}


model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=a.device).eval()
patch_gemma_norms(model, noscale=True)
layers = model.model.language_model.layers
ex_to_li = {id(l.experts): li for li, l in enumerate(layers) if getattr(l, "enable_moe_block", False)}

prefill_meta, decode_meta = {}, {}


def pre_hook(mod, args):
    li = ex_to_li.get(id(mod))
    if li is not None and li not in prefill_meta:
        prefill_meta[li] = {"hidden_states": meta(args[0]), "top_k_index": meta(args[1]),
                            "top_k_weights": meta(args[2])}


for l in layers:
    if getattr(l, "enable_moe_block", False):
        l.experts.register_forward_pre_hook(pre_hook)

eng = GemmaCB(model, None, 16, 2048, a.device, grouped=True, fused_rope=True,
              prefill_grouped=a.prefill_grouped)

# decode-side wrapper: record the first decode step only (flag set below)
state = {"record": False}
orig_eg = gemma_cb.experts_grouped


def wrapped(ex, x, idx, w, geglu=False):
    out = orig_eg(ex, x, idx, w, geglu=geglu)
    if state["record"]:
        li = ex_to_li[id(ex)]
        decode_meta.setdefault(li, {"xin": meta(x), "idx": meta(idx), "w": meta(w),
                                    "out": meta(out)})
    return out


gemma_cb.experts_grouped = wrapped

torch.manual_seed(0)
for s in range(16):
    ids = torch.randint(0, 200000, (1, 140 + s), device=a.device)
    eng.prefill_into_slot(s, ids)
tokens = torch.randint(0, 200000, (16,), device=a.device)
pos = torch.tensor([140 + s for s in range(16)], device=a.device)
state["record"] = True
logits = eng.decode_step(tokens, pos)
state["record"] = False
torch.npu.synchronize()

out = {"prefill_grouped": a.prefill_grouped, "prefill_experts_input": prefill_meta,
       "decode_experts_io": decode_meta, "logits": meta(logits)}
with open(a.out, "w") as f:
    json.dump(out, f, indent=1)
print("DONE", a.out, "layers_prefill", len(prefill_meta), "layers_decode", len(decode_meta))
