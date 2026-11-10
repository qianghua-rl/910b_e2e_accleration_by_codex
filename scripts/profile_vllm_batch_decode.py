"""vLLM-Ascend decode profile on the same slow batch (15 turn-1 prompts from HF opt run).
Measures prefill-only (1 token) and prefill+64 tokens wall time, and profiles the 64-token run."""
import json, os, sys, time
sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
from agent_protocol import SYSTEM, TOOLS  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

MODEL = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/model/qwen3.5-27b"
DATA = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/data/gsm8k_128.jsonl"
RES = "/remote-home/S45160/work/910b_fresh/results/b16_original_opt/per_task.jsonl"
OUT = sys.argv[1]
DECODE = int(sys.argv[2]) if len(sys.argv) > 2 else 64
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

os.makedirs(OUT, exist_ok=True)
os.environ["VLLM_TORCH_PROFILER_DIR"] = OUT
llm = LLM(model=MODEL, tensor_parallel_size=1, dtype="bfloat16", max_model_len=2048,
          gpu_memory_utilization=0.98, enable_prefix_caching=False,
          profiler_config={"profiler": "torch", "torch_profiler_dir": OUT})

def timed(max_tokens, profile=False):
    sp = SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True)
    if profile: llm.start_profile()
    t = time.perf_counter()
    llm.generate(prompts, sp, use_tqdm=False)
    dt = time.perf_counter() - t
    if profile: llm.stop_profile()
    return dt

timed(1)                       # warmup (graph/kernel compile)
prefill_s = timed(1)
full_s = timed(DECODE + 1, profile=True)
meta = {"batch": len(prompts), "decode_steps": DECODE, "prefill_s": prefill_s, "prefill_plus_decode_s": full_s,
        "decode_ms_per_step_est": (full_s - prefill_s) * 1000 / DECODE}
json.dump(meta, open(os.path.join(OUT, "profile_meta.json"), "w"), indent=2)
print("PROFILE_META " + json.dumps(meta), flush=True)
