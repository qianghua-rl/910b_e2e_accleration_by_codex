"""Profile batch16 decode steps (same batch as scripts/slow_batch_steps.py), optimisations on.
Warm up, then profile N decode steps. Reports wall time, device-busy union time, and idle gaps."""
import json, os, sys, time
import torch, torch_npu  # noqa: F401
import torch_npu.profiler as tnp
sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
from agent_protocol import SYSTEM, TOOLS  # noqa: E402
from recurrent_single_step import recurrent_single_step  # noqa: E402
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet, Qwen3_5RMSNorm, Qwen3_5RMSNormGated  # noqa: E402

MODEL = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/model/qwen3.5-27b"
DATA = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/data/gsm8k_128.jsonl"
RES = "/remote-home/S45160/work/910b_fresh/results/b16_original_opt/per_task.jsonl"
OUT = sys.argv[1]
DEV = "npu:0"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
BATCH_IDS = {247, 249, 259, 265, 272, 294, 319, 350, 375, 399, 411, 420, 423, 428, 444}

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

inputs = tok(prompts, add_special_tokens=False, padding=True, return_tensors="pt").to(DEV)
os.makedirs(OUT, exist_ok=True)
with torch.inference_mode():
    # Prefill + warmup decode with the real cache layout, then profile STEPS decode steps.
    out = model(**inputs, use_cache=True, logits_to_keep=1)
    past = out.past_key_values
    mask = inputs["attention_mask"]
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    for _ in range(2):
        mask = torch.cat([mask, mask.new_ones(mask.shape[0], 1)], dim=1)
        out = model(input_ids=nxt, attention_mask=mask, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    torch.npu.synchronize(DEV)
    prof = tnp.profile(
        activities=[tnp.ProfilerActivity.CPU, tnp.ProfilerActivity.NPU],
        on_trace_ready=tnp.tensorboard_trace_handler(OUT),
        record_shapes=False, profile_memory=False, with_stack=True)
    prof.start()
    t1 = time.perf_counter()
    for _ in range(STEPS):
        mask = torch.cat([mask, mask.new_ones(mask.shape[0], 1)], dim=1)
        out = model(input_ids=nxt, attention_mask=mask, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    torch.npu.synchronize(DEV)
    wall = time.perf_counter() - t1
    prof.stop()

meta = {"steps": STEPS, "batch": inputs["input_ids"].shape[0], "wall_s": wall,
        "wall_ms_per_step": wall * 1000 / STEPS}
json.dump(meta, open(os.path.join(OUT, "profile_meta.json"), "w"), indent=2)
print("PROFILE_META " + json.dumps(meta), flush=True)
