"""Diagnose greedy divergence between torch GDN prefill (ref) and fast GDN prefill.

For each flipped task: take the base run's generated prefix up to the first token where the
fast run differs (index d). Run a full forward over prompt + prefix with each GDN chunk
implementation and compare last-position logits. If the top-2 margin under the ref path is
smaller than the ref-vs-fast logit difference, the flip is within numerical noise.
Run with ASCEND_RT_VISIBLE_DEVICES=1 and --device npu:0 (one card, nothing else on it).
"""
import json
import sys

import torch
import torch_npu  # noqa: F401
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
from agent_protocol import SYSTEM, TOOLS  # noqa: E402
from cb_engine import patch_model  # noqa: E402
from gdn_chunk_fast import fast_chunk_gated_delta_rule  # noqa: E402

MODEL = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/model/qwen3.5-27b"
DATA = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/data/gsm8k_128.jsonl"
BASE = "/remote-home/S45160/work/910b_fresh/results/ablations/continuous_batching/repeat_base_r2/per_task.jsonl"
FAST = "/remote-home/S45160/work/910b_fresh/results/ablations/continuous_batching/repeat_fast_r2/per_task.jsonl"
TASKS = [1137, 640, 1030]
DEV = "npu:0"


def load_rows(path):
    return {x["id"]: x for x in map(json.loads, open(path)) if x.get("kind") == "task"}


def main():
    base, fast = load_rows(BASE), load_rows(FAST)
    tasks = {}
    for line in open(DATA):
        t = json.loads(line)
        tasks[t["id"]] = t

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": DEV}).eval()
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    patch_model(model, fast_gdn=False)  # RMSNorm/recurrent patches as in the runs; chunk = torch fallback

    gdn = [m for m in model.modules() if hasattr(m, "chunk_gated_delta_rule")]
    ref_fn = gdn[0].chunk_gated_delta_rule
    print(f"GDN modules: {len(gdn)}", flush=True)

    def set_chunk(fn):
        for m in gdn:
            m.chunk_gated_delta_rule = fn

    def last_logits(ids):
        with torch.inference_mode():
            out = model(input_ids=ids, use_cache=False, logits_to_keep=1)
        return out.logits[0, -1].float()

    for i in TASKS:
        b_tokens = base[i]["rounds"][0]["tokens"]
        f_tokens = fast[i]["rounds"][0]["tokens"]
        d = next((k for k, (x, y) in enumerate(zip(b_tokens, f_tokens)) if x != y), None)
        if d is None:
            print(f"task {i}: no first-turn divergence", flush=True)
            continue
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": tasks[i]["question"]}]
        prompt = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        p_ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids
        prefix = torch.tensor([b_tokens[:d]], dtype=p_ids.dtype)
        ids = torch.cat([p_ids, prefix], dim=1).to(DEV)

        set_chunk(ref_fn)
        lr = last_logits(ids)
        set_chunk(fast_chunk_gated_delta_rule)
        lf = last_logits(ids)
        set_chunk(ref_fn)

        top_r = torch.topk(lr, 2)
        top_f = torch.topk(lf, 2)
        margin_r = (top_r.values[0] - top_r.values[1]).item()
        margin_f = (top_f.values[0] - top_f.values[1]).item()
        logit_diff = (lr - lf).abs().max().item()
        at_d_diff = (lr[b_tokens[d]] - lf[b_tokens[d]]).item()
        print(f"task {i} divergence idx {d}: base tok {b_tokens[d]} fast tok {f_tokens[d]}", flush=True)
        print(f"  ref argmax {top_r.indices[0].item()} (top2 {top_r.indices[1].item()}), margin {margin_r:.4f}", flush=True)
        print(f"  fast argmax {top_f.indices[0].item()} (top2 {top_f.indices[1].item()}), margin {margin_f:.4f}", flush=True)
        print(f"  max |logit_ref - logit_fast| over vocab {logit_diff:.4f}; at base tok {at_d_diff:+.4f}", flush=True)
        print(f"  verdict: {'NOISE-LEVEL (ref margin <= logit diff)' if margin_r <= logit_diff else 'margin exceeds logit diff: investigate'}", flush=True)


if __name__ == "__main__":
    main()
