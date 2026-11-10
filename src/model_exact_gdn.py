"""Full-model check of GDN candidates A+B (decode only, T=1) against the unpatched model.

One model load. For each mode (reference, candidate) we run prefill + N greedy decode
steps on the same question with a fresh cache, recording logits and token ids per step.
Acceptance: all logits bit-identical and identical greedy tokens for N steps.
Timing: unprofiled decode wall per step, reported for information.
Usage: model_exact_gdn.py [--device npu:0] [--steps 64] [--question 0]
"""
import argparse
import json
import sys
import time

import torch
import torch_npu  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
from exact_gdn_candidates import candidate_recurrent  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import (  # noqa: E402
    torch_recurrent_gated_delta_rule as reference_recurrent,
)

MODEL = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/model/qwen3.5-27b"
DATA = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/data/gsm8k_128.jsonl"
TOOLS = [{"type": "function", "function": {
    "name": "python", "description": "Run Python code; print the result.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]


def make_patched(orig):
    """Route T=1 calls to the candidate; everything else to the original function."""
    def fn(query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=True):
        if query.shape[1] != 1 or initial_state is None:
            return orig(query, key, value, g=g, beta=beta, initial_state=initial_state,
                        output_final_state=output_final_state,
                        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
        out, state = candidate_recurrent(query, key, value, g, beta, initial_state,
                                         use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
        return out, (state if output_final_state else None)
    return fn


def set_mode(model, patched):
    n = 0
    for layer in model.model.language_model.layers if hasattr(model.model, "language_model") \
            else model.model.layers:
        la = getattr(layer, "linear_attn", None)
        if la is None:
            continue
        la.recurrent_gated_delta_rule = make_patched(reference_recurrent) if patched else reference_recurrent
        n += 1
    return n


def run(model, tok, ids, steps, device):
    logits_all, toks = [], []
    with torch.no_grad():
        out = model(ids, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1]
        for s in range(steps):
            nxt = logits.argmax(-1, keepdim=True)
            toks.append(int(nxt))
            logits_all.append(logits.float().cpu().clone())
            if s == steps - 1:
                break
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1]
        torch.npu.synchronize(device)
        # timing pass: fresh cache, unprofiled decode wall per step
        out = model(ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        torch.npu.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(steps - 1):
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        torch.npu.synchronize(device)
        per_step_ms = (time.perf_counter() - t0) / (steps - 1) * 1000
    return logits_all, toks, per_step_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--question", type=int, default=0)
    ap.add_argument("--candidate-first", action="store_true")
    a = ap.parse_args()

    q = [json.loads(l) for l in open(DATA)][a.question]
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map=a.device).eval()
    msgs = [{"role": "user", "content": q["question"] +
             "\nSolve with Python, then give the final answer as \\boxed{N}."}]
    prompt = tok.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True,
                                     tokenize=False, enable_thinking=False)
    ids = tok(prompt, return_tensors="pt").input_ids.to(a.device)
    print("prompt tokens", ids.shape[1], flush=True)

    n = set_mode(model, patched=False)
    print("GDN layers patched-capable:", n, flush=True)
    if a.candidate_first:
        set_mode(model, patched=True)
        cand_logits, cand_toks, cand_ms = run(model, tok, ids, a.steps, a.device)
        print(f"candidate: decode {cand_ms:.2f} ms/step", flush=True)
        set_mode(model, patched=False)
        ref_logits, ref_toks, ref_ms = run(model, tok, ids, a.steps, a.device)
        print(f"reference: decode {ref_ms:.2f} ms/step", flush=True)
    else:
        ref_logits, ref_toks, ref_ms = run(model, tok, ids, a.steps, a.device)
        print(f"reference: decode {ref_ms:.2f} ms/step", flush=True)
        set_mode(model, patched=True)
        cand_logits, cand_toks, cand_ms = run(model, tok, ids, a.steps, a.device)
        print(f"candidate: decode {cand_ms:.2f} ms/step", flush=True)

    tokens_equal = ref_toks == cand_toks
    first_diff = next((i for i, (x, y) in enumerate(zip(ref_logits, cand_logits))
                       if not torch.equal(x, y)), None)
    max_diff = max((x - y).abs().max().item() for x, y in zip(ref_logits, cand_logits))
    print(f"tokens_equal={tokens_equal}  logits_first_diff_step={first_diff}  "
          f"max_logit_diff={max_diff:.3e}", flush=True)
    print("ref tokens :", ref_toks[:16], flush=True)
    print("cand tokens:", cand_toks[:16], flush=True)
    print(f"speedup per decode step: {ref_ms / cand_ms:.3f}x "
          f"({ref_ms:.2f} -> {cand_ms:.2f} ms)", flush=True)
    sys.exit(0 if (tokens_equal and first_diff is None) else 1)


if __name__ == "__main__":
    main()
