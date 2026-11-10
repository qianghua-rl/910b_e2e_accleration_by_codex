# Per-task comparison of vLLM vs CB (graph) AIME runs. Usage: python3 -I scripts/diff_aime_vllm_vs_cb.py <vllm.jsonl> <cb.jsonl>
import json, sys
sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
from scorer_math import score_one

def load(p):
    rows = {}
    for l in open(p):
        r = json.loads(l)
        if r.get("kind") == "task":
            rows[r["id"]] = r
    return rows

def info(r):
    if r is None:
        return None
    text = r.get("final_text") or r.get("text") or ""
    pred = r.get("prediction") if "prediction" in r else r.get("pred")
    gold = str(r.get("gold", r.get("expected")))
    ok = score_one(text, gold)["correct"]
    toks = sum(len(x.get("tokens", [])) for x in r.get("rounds", []))
    rounds = len(r.get("rounds", []))
    return dict(ok=bool(ok), pred=pred, gold=gold, toks=toks, rounds=rounds, boxed="\\boxed" in text, text=text)

v = load(sys.argv[1]); c = load(sys.argv[2])
ids = sorted(set(v) | set(c))
n_v = n_c = 0
print("id gold | vLLM ok rounds toks boxed pred | CB ok rounds toks boxed pred")
for i in ids:
    a = info(v.get(i)); b = info(c.get(i))
    n_v += bool(a and a["ok"]); n_c += bool(b and b["ok"])
    if a and b and a["ok"] != b["ok"]:
        print(f"{i:>3} {a['gold']:>5} | {a['ok']!s:5} {a['rounds']} {a['toks']:>5} {a['boxed']!s:5} {a['pred']} | {b['ok']!s:5} {b['rounds']} {b['toks']:>5} {b['boxed']!s:5} {b['pred']}  <-- DIFF")
print(f"correct: vLLM {n_v}, CB {n_c}")
