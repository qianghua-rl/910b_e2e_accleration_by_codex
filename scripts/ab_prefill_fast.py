"""End-to-end A/B of batch-1 prefill: fallback chunk GDN vs src/gdn_chunk_fast.py.
Checks last-token logits agreement and times prefill (min of 3). Run on one NPU only."""
import json
import sys
import time

import torch
import torch_npu  # noqa: F401
from transformers import AutoTokenizer, DynamicCache, Qwen3_5ForConditionalGeneration

sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
from agent_protocol import SYSTEM, TOOLS  # noqa: E402
from cb_engine import patch_model  # noqa: E402
from gdn_chunk_fast import fast_chunk_gated_delta_rule  # noqa: E402

MODEL = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/model/qwen3.5-27b"
DATA = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/data/gsm8k_128.jsonl"
DEV = "npu:0"

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": DEV}).eval()
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
patch_model(model)
cfg = model.config.text_config
rows = [json.loads(l) for l in open(DATA)]
rows = [r for r in rows if r.get("kind", "task") == "task"][:4]
prompts = []
for t in rows:
    p = tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": t["question"]}],
                                tools=TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompts.append(tok(p, add_special_tokens=False, return_tensors="pt").input_ids.to(DEV))
print("prompt tokens", [x.shape[1] for x in prompts], flush=True)

lm = model.model.language_model
gdn_layers = [l.linear_attn for i, l in enumerate(lm.layers) if cfg.layer_types[i] == "linear_attention"]
ref_fn = gdn_layers[0].chunk_gated_delta_rule  # fallback bound at load time


def set_mode(fast):
    for gd in gdn_layers:
        gd.chunk_gated_delta_rule = fast_chunk_gated_delta_rule if fast else ref_fn


def run(ids):
    cache = DynamicCache(config=cfg)
    torch.npu.synchronize(DEV)
    t0 = time.perf_counter()
    out = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    torch.npu.synchronize(DEV)
    return out.logits[:, -1].float(), time.perf_counter() - t0


with torch.inference_mode():
    results = {}
    for mode, fast in [("ref", False), ("fast", True)]:
        set_mode(fast)
        run(prompts[0])  # warm-up
        logits, times = [], []
        for ids in prompts:
            lg, _ = run(ids)
            logits.append(lg)
            times.append(min(run(ids)[1] for _ in range(3)))
        results[mode] = (logits, times)
        print(f"{mode}: per-prompt min prefill (s) = {[round(x, 3) for x in times]} "
              f"sum={sum(times):.3f}", flush=True)

    for i, (a, b) in enumerate(zip(results["ref"][0], results["fast"][0])):
        top_ref, top_fast = a.argmax(-1).item(), b.argmax(-1).item()
        print(f"prompt {i}: logits maxdiff={(a - b).abs().max().item():.3e} "
              f"ref_top={top_ref} fast_top={top_fast} same={top_ref == top_fast}", flush=True)
