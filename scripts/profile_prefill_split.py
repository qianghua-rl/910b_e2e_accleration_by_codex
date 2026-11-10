"""Split one batch-1 prefill (fast-GDN) into GDN-chunk time vs everything else. Wraps the chunk
function and the GDN layer's forward; reports mean over reps after warmup. Writes JSON to argv[1]."""
import json, sys, time
import torch, torch_npu  # noqa: F401
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, DynamicCache
sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
from agent_protocol import SYSTEM, TOOLS  # noqa: E402
import cb_engine
from cb_engine import patch_model  # noqa: E402
import gdn_chunk_fast
from transformers.models.qwen3_5 import modeling_qwen3_5 as M

MODEL = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/model/qwen3.5-27b"
DATA = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/data/gsm8k_128.jsonl"
DEV = "npu:0"; REPS = 5; OUT = sys.argv[1]
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": DEV}).eval()
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
patch_model(model, fast_gdn=True)
cfg = model.config.text_config
t = json.loads(open(DATA).readline())
p = tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": t["question"]}],
                            tools=TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok(p, add_special_tokens=False, return_tensors="pt").input_ids.to(DEV)
acc = {"chunk_gdn": 0.0}
orig = None
for layer in model.model.language_model.layers:
    if hasattr(layer, "linear_attn"):
        orig = layer.linear_attn.chunk_gated_delta_rule
        break
def timed(*a, **k):
    torch.npu.synchronize(DEV); t0 = time.perf_counter()
    r = orig(*a, **k)
    torch.npu.synchronize(DEV); acc["chunk_gdn"] += time.perf_counter() - t0
    return r
rows = []
with torch.inference_mode():
    for layer in model.model.language_model.layers:
        if hasattr(layer, "linear_attn"):
            layer.linear_attn.chunk_gated_delta_rule = timed
    for rep in range(REPS + 1):
        acc["chunk_gdn"] = 0.0
        torch.npu.synchronize(DEV); t0 = time.perf_counter()
        out = model(input_ids=ids, past_key_values=DynamicCache(config=cfg), use_cache=True, logits_to_keep=1)
        torch.npu.synchronize(DEV); tot = time.perf_counter() - t0
        if rep > 0:
            rows.append({"total_ms": tot * 1000, "chunk_gdn_ms": acc["chunk_gdn"] * 1000})
mean = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
mean["other_ms"] = mean["total_ms"] - mean["chunk_gdn_ms"]
mean["tokens"] = ids.shape[1]
print(json.dumps(mean, indent=1), flush=True)
json.dump({"mean": mean, "rows": rows}, open(OUT, "w"), indent=1)
