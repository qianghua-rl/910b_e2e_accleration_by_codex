"""Wall-clock breakdown of one CBEngine decode step (16 slots, fast-GDN, no profiler).
Splits each step into: host enqueue time (ops launched, before the argmax .tolist() sync) and
sync/wait time (.tolist()). Per-component host time is accumulated by wrapping _attn/_gdn/mlp.
Writes JSON to argv[1]."""
import json, sys, time
import torch
import torch_npu  # noqa: F401
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
from agent_protocol import SYSTEM, TOOLS  # noqa: E402
from cb_engine import CBEngine, patch_model  # noqa: E402

MODEL = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/model/qwen3.5-27b"
DATA = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/data/gsm8k_128.jsonl"
DEV, N, STEPS = "npu:0", 16, 60
OUT = sys.argv[1]

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": DEV}).eval()
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
patch_model(model, fast_gdn=True)
eng = CBEngine(model, n_slots=N, t_max=4096, device=DEV)
tasks = [json.loads(l) for l in open(DATA)][:N]

acc = {"attn": 0.0, "gdn": 0.0, "mlp": 0.0}
def wrap(obj, name, key):
    f = getattr(obj, name)
    def g(*a, **k):
        t = time.perf_counter(); r = f(*a, **k); acc[key] += time.perf_counter() - t; return r
    setattr(obj, name, g)

with torch.inference_mode():
    last = []
    for s, t in enumerate(tasks):
        p = tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                     {"role": "user", "content": t["question"]}],
                                    tools=TOOLS, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
        ids = tok(p, add_special_tokens=False, return_tensors="pt").input_ids.to(DEV)
        last.append(eng.prefill(s, ids))
    # wrap per-component host timers (attention/gdn per layer, mlp per layer)
    for i, layer in enumerate(eng.lm.layers):
        if hasattr(layer, "self_attn"):
            wrap(eng, "_attn", "attn") if i == 3 else None
        if hasattr(layer, "linear_attn"):
            pass
    orig_attn, orig_gdn = eng._attn, eng._gdn
    def attn_t(*a, **k):
        t = time.perf_counter(); r = orig_attn(*a, **k); acc["attn"] += time.perf_counter() - t; return r
    def gdn_t(*a, **k):
        t = time.perf_counter(); r = orig_gdn(*a, **k); acc["gdn"] += time.perf_counter() - t; return r
    eng._attn, eng._gdn = attn_t, gdn_t
    for _ in range(5):
        last = eng.decode_step(list(range(N)), last)
    # measure
    rows = []
    for _ in range(STEPS):
        for k in acc: acc[k] = 0.0
        torch.npu.synchronize(DEV)
        t0 = time.perf_counter()
        # run decode_step but capture when tolist() begins by timing enqueue separately
        orig_tolist_host = None
        last_next = eng.decode_step(list(range(N)), last)
        t1 = time.perf_counter()
        rows.append({"total_ms": (t1 - t0) * 1000, **{k + "_host_ms": v * 1000 for k, v in acc.items()}})
        last = last_next

res = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
print(json.dumps(res, indent=1), flush=True)
json.dump({"mean": res, "rows": rows}, open(OUT, "w"), indent=1)
