"""Alignment test for src/gemma_cb.py against HF generate (greedy).
N=1: one prompt in one slot; must match HF token-for-token (logic check).
N=4: four prompts decoded together; reports per-slot match (batch numerics may diverge).
Usage: test_gemma_cb.py [--device npu:1] [--steps 40]
"""
import argparse, json, sys, torch, torch_npu  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
import agent_protocol as P
from gemma_cb import GemmaCB

MODEL = "/remote-home/S45160/work/910b_fresh/gemma-4-26B-A4B-it"
DATA = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/data/gsm8k_128.jsonl"
ap = argparse.ArgumentParser()
ap.add_argument("--device", default="npu:1")
ap.add_argument("--steps", type=int, default=40)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=a.device).eval()
qs = [json.loads(l) for l in open(DATA)][:4]
ids_list = []
for q in qs:
    msgs = [{"role": "system", "content": P.SYSTEM}, {"role": "user", "content": q["question"]}]
    p = tok.apply_chat_template(msgs, tools=P.TOOLS, add_generation_prompt=True,
                                tokenize=False, enable_thinking=False)
    ids_list.append(tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to(a.device))

# HF reference: greedy, same number of new tokens
ref = []
with torch.no_grad():
    for ids in ids_list:
        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=a.steps,
                             min_new_tokens=a.steps, do_sample=False, pad_token_id=tok.pad_token_id)
        ref.append(out[0, ids.shape[1]:].tolist())
print("ref ready", flush=True)

def cb_run(slot_ids):
    eng = GemmaCB(model, tok, n_slots=len(slot_ids), t_max=2048, device=a.device)
    first, lens, gen = [], [], [[] for _ in slot_ids]
    for s, ids in enumerate(slot_ids):
        lg = eng.prefill_into_slot(s, ids)
        t = int(lg.argmax().item()); first.append(t); gen[s].append(t); lens.append(int(ids.shape[1]))
    nxt = first[:]
    for _ in range(a.steps - 1):
        logits = eng.decode_step(torch.tensor(nxt, device=a.device),
                                 torch.tensor(lens, device=a.device))
        am = logits.argmax(-1).tolist()
        for s in range(len(slot_ids)):
            gen[s].append(am[s]); nxt[s] = am[s]; lens[s] += 1
    return gen

def report(tag, gen, idx):
    for s, i in enumerate(idx):
        r = ref[i]; g = gen[s]
        n = min(len(r), len(g))
        first_div = next((k for k in range(n) if r[k] != g[k]), None)
        print(f"{tag} prompt{i}: match {sum(r[k]==g[k] for k in range(n))}/{n}, first_div={first_div}", flush=True)

# N=1 for each prompt separately
for i, ids in enumerate(ids_list):
    report("N=1", cb_run([ids]), [i])
# N=4 together
report("N=4", cb_run(ids_list), list(range(4)))
