"""Measure prefill-only time for a real turn-1 batch (first 16 tasks of the HF opt run).
Prefill is approximated by generate(max_new_tokens=1). Output: JSON on stdout."""
import json, sys, time
import torch, torch_npu  # noqa: F401
sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
from agent_protocol import SYSTEM, TOOLS  # noqa: E402
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration  # noqa: E402

MODEL = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/model/qwen3.5-27b"
DATA = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/data/gsm8k_128.jsonl"
RES = "/remote-home/S45160/work/910b_fresh/results/b16_original_opt/per_task.jsonl"
DEV = "npu:0"

questions = {str(json.loads(l)["id"]): json.loads(l)["question"] for l in open(DATA)}
rows = [json.loads(l) for l in open(RES)]
tasks = [r for r in rows if r.get("kind") == "task" and len(r["rounds"]) >= 2][:16]

tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
def render(msgs):
    return tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)

prompt_turn0, prompt_turn1 = [], []
for t in tasks:
    r0 = t["rounds"][0]
    base = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": questions[str(t["id"])]}]
    prompt_turn0.append(render(base))
    msgs = base + [{"role": "assistant", "content": r0["text"]},
                   {"role": "tool", "name": "python", "content": r0["tool_output"]}]
    prompt_turn1.append(render(msgs))

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": DEV}).eval()

def prefill_seconds(prompts):
    inputs = tok(prompts, add_special_tokens=False, padding=True, return_tensors="pt").to(DEV)
    times = []
    with torch.inference_mode():
        for _ in range(2):
            torch.npu.synchronize(DEV)
            t = time.perf_counter()
            model.generate(**inputs, max_new_tokens=1, do_sample=False, use_cache=True)
            torch.npu.synchronize(DEV)
            times.append(time.perf_counter() - t)
    return {"padded_len": inputs["input_ids"].shape[1], "tokens_sum": int(inputs["attention_mask"].sum()),
            "times_s": times}

out = {"turn0_prefill": prefill_seconds(prompt_turn0), "turn1_prefill": prefill_seconds(prompt_turn1),
       "n": len(tasks)}
print("PROBE " + json.dumps(out))
