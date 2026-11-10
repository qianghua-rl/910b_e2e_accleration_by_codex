"""Per-step decode timing for the slowest turn-1 batch (ids 247..444 from HF opt run).
Same optimisations as src/e100_batch.py (fused RMSNorm, gated RMSNorm, single-step recurrent).
Records wall time of every decode step via a LogitsProcessor (synchronised)."""
import json, sys, time
import torch, torch_npu  # noqa: F401
sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
from agent_protocol import SYSTEM, TOOLS  # noqa: E402
from recurrent_single_step import recurrent_single_step  # noqa: E402
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, LogitsProcessor, LogitsProcessorList  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet, Qwen3_5RMSNorm, Qwen3_5RMSNormGated  # noqa: E402

MODEL = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/model/qwen3.5-27b"
DATA = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/data/gsm8k_128.jsonl"
RES = "/remote-home/S45160/work/910b_fresh/results/b16_original_opt/per_task.jsonl"
OUT = "/remote-home/S45160/work/910b_fresh/results/slow_batch_steps/steps.json"
DEV = "npu:0"
BATCH_IDS = {247, 249, 259, 265, 272, 294, 319, 350, 375, 399, 411, 420, 423, 428, 444}
CAP = 512

questions = {str(json.loads(l)["id"]): json.loads(l)["question"] for l in open(DATA)}
rows = [json.loads(l) for l in open(RES)]
tasks = [r for r in rows if r.get("kind") == "task" and r["id"] in BATCH_IDS and len(r["rounds"]) >= 2]
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
prompts = []
for t in tasks:
    r0 = t["rounds"][0]
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": questions[str(t["id"])]},
            {"role": "assistant", "content": r0["text"]}, {"role": "tool", "name": "python", "content": r0["tool_output"]}]
    prompts.append(tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False))

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": DEV}).eval()

def fused_rmsnorm(self, x):
    return torch_npu.npu_rms_norm(x, (1.0 + self.weight).to(x.dtype), self.eps)[0]
def fused_gated_rmsnorm(self, hidden_states, gate=None):
    normalized = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
    return torch_npu.npu_swiglu(torch.cat((gate, normalized), dim=-1))
for m in model.modules():
    if isinstance(m, Qwen3_5RMSNorm):
        m.forward = fused_rmsnorm.__get__(m, Qwen3_5RMSNorm)
    elif isinstance(m, Qwen3_5RMSNormGated):
        m.forward = fused_gated_rmsnorm.__get__(m, Qwen3_5RMSNormGated)
    elif isinstance(m, Qwen3_5GatedDeltaNet):
        m.recurrent_gated_delta_rule = recurrent_single_step

class Timer(LogitsProcessor):
    def __init__(self): self.stamps = []
    def __call__(self, input_ids, scores):
        torch.npu.synchronize(DEV)
        self.stamps.append(time.perf_counter())
        return scores

inputs = tok(prompts, add_special_tokens=False, padding=True, return_tensors="pt").to(DEV)
timer = Timer()
with torch.inference_mode():
    t0 = time.perf_counter()
    gen = model.generate(**inputs, max_new_tokens=CAP, do_sample=False, use_cache=True,
                         logits_processor=LogitsProcessorList([timer]))
    torch.npu.synchronize(DEV)
    total = time.perf_counter() - t0
steps = [(b - a) for a, b in zip(timer.stamps[:-1], timer.stamps[1:])]
prefill_plus_first = timer.stamps[0] - t0
out = {"n_rows": len(tasks), "input_len": int(inputs["input_ids"].shape[1]),
       "generated_steps": gen.shape[1] - inputs["input_ids"].shape[1], "total_s": total,
       "first_logits_after_s": prefill_plus_first, "step_s": steps}
with open(OUT, "w") as f:
    json.dump(out, f)
bucket = 50
for i in range(0, len(steps), bucket):
    seg = steps[i:i + bucket]
    print(f"steps {i:4d}-{i+len(seg)-1:4d}: mean {sum(seg)/len(seg)*1000:7.1f} ms")
print("SUMMARY " + json.dumps({k: v for k, v in out.items() if k != "step_s"}))
