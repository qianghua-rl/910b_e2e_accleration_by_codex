# Compare round-0 prompt token counts: CB (cb_run template call) vs vLLM recorded prompt_tokens.
# Usage: python3 -I scripts/compare_prompt_tokens.py <vllm per_task.jsonl> <cb per_task.jsonl> <data.jsonl> id [id ...]
import json, sys
sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
from transformers import AutoTokenizer
from agent_protocol import SYSTEM, TOOLS
MODEL = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/model/qwen3.5-27b"
tok = AutoTokenizer.from_pretrained(MODEL)

def load(p):
    d = {}
    for l in open(p):
        r = json.loads(l)
        if r.get("kind") == "task":
            d[r["id"]] = r
    return d

vllm = load(sys.argv[1]); cb_rows = load(sys.argv[2])
data = {}
for l in open(sys.argv[3]):
    r = json.loads(l); data[r["id"]] = r
for i in map(int, sys.argv[4:]):
    q = data[i]["question"]
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
    p = tok.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True, enable_thinking=False, tokenize=False)
    n = len(tok.encode(p, add_special_tokens=False))
    ids_cb = tok(p, add_special_tokens=False).input_ids
    v = vllm.get(i, {}).get("rounds", [{}])[0].get("prompt_tokens")
    print(f"id {i}: CB-template tokens {n} | vLLM prompt_tokens {v} | match {n == v}")
