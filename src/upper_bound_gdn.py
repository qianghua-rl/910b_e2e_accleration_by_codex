"""Upper-bound experiment: how much decode time can GDN host/device work account for?

Timing only. Outputs under the stubs are WRONG by design and are never used for accuracy.
Modes (interleaved ABBA, repeated, same process, one model load):
  ref : unmodified model
  A   : GDN recurrent call (T=1) replaced by a shape-correct no-op (state unchanged)
  B   : entire GDN decode forward (T=1) replaced by zeros (projections, conv, recurrence,
        gated norm and output projection all removed)
Result is the decode ms/step per mode. If A is within noise of ref, the recurrent path is
not a lever. B bounds the whole GDN decode cost.
Usage: upper_bound_gdn.py [--device npu:0] [--steps 64] [--rounds 2]
"""
import argparse
import json
import sys

import torch
import torch_npu  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
from model_exact_gdn import MODEL, DATA, TOOLS, run  # noqa: E402


def layers_with_gdn(model):
    base = model.model.language_model if hasattr(model.model, "language_model") else model.model
    return [l.linear_attn for l in base.layers if getattr(l, "linear_attn", None) is not None], base


def set_mode(model, mode, hidden):
    gdn, _ = layers_with_gdn(model)
    for la in gdn:
        if "forward_orig" not in la.__dict__:
            la.forward_orig = la.forward
            la.recurrent_orig = la.recurrent_gated_delta_rule

        if mode == "ref":
            la.forward = la.forward_orig
            la.recurrent_gated_delta_rule = la.recurrent_orig
        elif mode == "A":
            la.forward = la.forward_orig

            def noop_recurrent(query, key, value, g, beta, initial_state,
                               output_final_state, use_qk_l2norm_in_kernel=True, _la=la):
                if query.shape[1] != 1:
                    return _la.recurrent_orig(query, key, value, g=g, beta=beta,
                                              initial_state=initial_state,
                                              output_final_state=output_final_state,
                                              use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
                out = torch.zeros(query.shape[0], 1, query.shape[2], value.shape[-1],
                                  dtype=value.dtype, device=value.device)
                return out, (initial_state if output_final_state else None)
            la.recurrent_gated_delta_rule = noop_recurrent
        elif mode == "B":
            def zero_forward(hidden_states, cache_params=None, attention_mask=None, _h=hidden, _la=la):
                if hidden_states.shape[1] != 1:
                    return _la.forward_orig(hidden_states, cache_params=cache_params,
                                            attention_mask=attention_mask)
                return torch.zeros(hidden_states.shape[0], 1, _h, dtype=hidden_states.dtype,
                                   device=hidden_states.device)
            la.forward = zero_forward


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
    hidden = model.config.text_config.hidden_size if hasattr(model.config, "text_config") \
        else model.config.hidden_size
    msgs = [{"role": "user", "content": q["question"] +
             "\nSolve with Python, then give the final answer as \\boxed{N}."}]
    prompt = tok.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True,
                                     tokenize=False, enable_thinking=False)
    ids = tok(prompt, return_tensors="pt").input_ids.to(a.device)

    results = {"ref": [], "A": [], "B": []}
    order = ["ref", "A", "B"]
    for r in range(a.rounds):
        seq = order if r % 2 == 0 else order[::-1]   # alternate order to cancel drift
        for mode in seq:
            set_mode(model, mode, hidden)
            _, _, ms = run(model, tok, ids, a.steps, a.device)
            results[mode].append(ms)
            print(f"round {r} mode {mode}: {ms:.2f} ms/step", flush=True)

    summary = {m: {"runs": v, "mean_ms": sum(v) / len(v)} for m, v in results.items()}
    for m in ["A", "B"]:
        summary[m]["vs_ref_pct"] = 100 * (summary[m]["mean_ms"] / summary["ref"]["mean_ms"] - 1)
    print("SUMMARY", json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
