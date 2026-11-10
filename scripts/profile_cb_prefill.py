"""Time one batch-1 prefill (as used by CBEngine.prefill) and break it down by module type."""
import json
import sys
import time

import torch
import torch_npu  # noqa: F401
from transformers import AutoTokenizer, DynamicCache, Qwen3_5ForConditionalGeneration

sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
from agent_protocol import SYSTEM, TOOLS  # noqa: E402
from cb_engine import patch_model  # noqa: E402
from transformers.models.qwen3_5 import modeling_qwen3_5 as M  # noqa: E402

MODEL = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/model/qwen3.5-27b"
DATA = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/data/gsm8k_128.jsonl"
DEV = "npu:0"

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": DEV}).eval()
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
patch_model(model)
cfg = model.config.text_config
t = [json.loads(l) for l in open(DATA)][0]
p = tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": t["question"]}],
                            tools=TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok(p, add_special_tokens=False, return_tensors="pt").input_ids.to(DEV)
print("prompt tokens", ids.shape[1], flush=True)

acc = {}


def timed(name, fn):
    def w(*a, **k):
        torch.npu.synchronize(DEV)
        t0 = time.perf_counter()
        r = fn(*a, **k)
        torch.npu.synchronize(DEV)
        acc[name] = acc.get(name, 0.0) + time.perf_counter() - t0
        return r
    return w


def run_prefill():
    cache = DynamicCache(config=cfg)
    torch.npu.synchronize(DEV)
    t0 = time.perf_counter()
    model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    torch.npu.synchronize(DEV)
    return time.perf_counter() - t0


with torch.inference_mode():
    for _ in range(2):
        run_prefill()
    times = [run_prefill() for _ in range(3)]
    print("prefill wall (no instrumentation):", [round(x, 3) for x in times], flush=True)

    lm = model.model.language_model
    for i, layer in enumerate(lm.layers):
        if cfg.layer_types[i] == "linear_attention":
            gd = layer.linear_attn
            gd.chunk_gated_delta_rule = timed("gdn_chunk", gd.chunk_gated_delta_rule)
            gd.causal_conv1d_fn = None
            orig_conv = gd.causal_conv1d_update
            gd.conv1d.forward = timed("gdn_conv", gd.conv1d.forward)
            gd.norm.forward = timed("gdn_norm", gd.norm.forward)
        else:
            layer.self_attn.forward = timed("attn", layer.self_attn.forward)
        layer.mlp.forward = timed("mlp", layer.mlp.forward)
    acc.clear()
    w = run_prefill()
    print(f"instrumented prefill: {w:.3f}s", flush=True)
    for k, v in sorted(acc.items(), key=lambda x: -x[1]):
        print(f"{k:12s} {v:7.3f} s ({v/w*100:4.1f}%)", flush=True)
    print(f"rest         {w-sum(acc.values()):7.3f} s", flush=True)
