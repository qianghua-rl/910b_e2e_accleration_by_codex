"""Split upper-bound: which GDN decode component carries the time?

Timing only (stub outputs are wrong by design). Each mode removes ONE component's
decode (T=1) work, keeps everything else, and is compared with the unmodified model:
  ref : unmodified
  P   : linear projections removed (in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj)
  C   : decode causal conv update removed (conv_state not updated)
  N   : gated RMSNorm removed (returns its input)
Prefill (T>1) is never stubbed. Interleaved, alternating order, same process.
Usage: split_upper_bound_gdn.py [--device npu:0] [--steps 64] [--rounds 2]
"""
import argparse
import json
import sys

import torch
import torch_npu  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
from model_exact_gdn import MODEL, DATA, TOOLS, run  # noqa: E402

PROJ = ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]


class ZeroLinearT1(torch.nn.Module):
    """Returns zeros of the real output width for T=1; original Linear otherwise."""
    def __init__(self, orig):
        super().__init__()
        self.orig = orig
        self.out_features = orig.out_features

    def forward(self, x):
        if x.shape[1] != 1:
            return self.orig(x)
        return torch.zeros(*x.shape[:-1], self.out_features, dtype=x.dtype, device=x.device)


HEADS_V = 48  # linear_num_value_heads


class IdentityNormT1(torch.nn.Module):
    """Gated norm stub. The caller reshapes to [B*T*H, V]; decode (T=1, B=1) has exactly
    H=48 rows, prefill has T*48 rows, so decode is detected by row count."""
    def __init__(self, orig):
        super().__init__()
        self.orig = orig

    def forward(self, hidden_states, gate=None):
        if hidden_states.shape[0] == HEADS_V:
            return hidden_states
        return self.orig(hidden_states, gate)


def get_gdn(model):
    base = model.model.language_model if hasattr(model.model, "language_model") else model.model
    return [l for l in base.layers if getattr(l, "linear_attn", None) is not None]


def install_originals(model):
    for layer in get_gdn(model):
        la = layer.linear_attn
        if "_orig_state" in la.__dict__:
            continue
        la._orig_state = {
            "proj": {n: getattr(la, n) for n in PROJ},
            "conv": la.causal_conv1d_update,
            "norm": la.norm,
        }


def set_mode(model, mode):
    for layer in get_gdn(model):
        la = layer.linear_attn
        orig = la._orig_state
        for n in PROJ:
            setattr(la, n, ZeroLinearT1(orig["proj"][n]) if mode == "P" else orig["proj"][n])
        la.causal_conv1d_update = _noop_conv(orig["conv"]) if mode == "C" else orig["conv"]
        la.norm = IdentityNormT1(orig["norm"]) if mode == "N" else orig["norm"]


def _noop_conv(orig):
    def fn(hidden_states, conv_state, weight, bias=None, activation=None):
        # decode T=1: return the activated input shape without touching conv_state
        if hidden_states.shape[-1] != 1:
            return orig(hidden_states, conv_state, weight, bias, activation)
        return hidden_states.clone()
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()

    q = [json.loads(l) for l in open(DATA)][0]
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map=a.device).eval()
    install_originals(model)
    msgs = [{"role": "user", "content": q["question"] +
             "\nSolve with Python, then give the final answer as \\boxed{N}."}]
    prompt = tok.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True,
                                     tokenize=False, enable_thinking=False)
    ids = tok(prompt, return_tensors="pt").input_ids.to(a.device)

    modes = ["ref", "P", "C", "N"]
    res = {m: [] for m in modes}
    for r in range(a.rounds):
        seq = modes if r % 2 == 0 else modes[::-1]
        for m in seq:
            set_mode(model, m)
            _, _, ms = run(model, tok, ids, a.steps, a.device)
            res[m].append(ms)
            print(f"round {r} mode {m}: {ms:.2f} ms/step", flush=True)

    ref = sum(res["ref"]) / len(res["ref"])
    summary = {m: {"runs": v, "mean_ms": sum(v) / len(v),
                   "vs_ref_pct": 100 * (sum(v) / len(v) / ref - 1)} for m, v in res.items()}
    print("SUMMARY", json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
