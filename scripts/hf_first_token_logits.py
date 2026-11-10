# HF logits at the first generated position for given AIME ids (turn 0, all optimizations off).
# Usage: python3 -I scripts/hf_first_token_logits.py <data.jsonl> <device> id [id ...]
import json, sys
sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
import torch, torch_npu  # noqa: F401
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from agent_protocol import SYSTEM, TOOLS
MODEL = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/model/qwen3.5-27b"
data_path, device, want = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]]
tasks = {json.loads(l)["id"]: json.loads(l) for l in open(data_path)}
torch.set_num_threads(8)
tok = AutoTokenizer.from_pretrained(MODEL)
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": device}).eval()
for i in want:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": tasks[i]["question"]}]
    p = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(p, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        logits = model(ids).logits[0, -1].float()
    top = torch.topk(logits, 5)
    print(f"id {i}: prompt {ids.shape[1]} tokens")
    for v, t in zip(top.values.tolist(), top.indices.tolist()):
        print(f"   {t:>7} {tok.decode([t])!r:>16} logit {v:.3f}")
    for t in (1206, 248058):
        print(f"   token {t} {tok.decode([t])!r}: logit {logits[t].item():.3f}, rank {(logits > logits[t]).sum().item()}")
