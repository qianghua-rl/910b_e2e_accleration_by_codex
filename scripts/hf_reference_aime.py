# HF single-sequence reference for AIME-style agent runs, all optimizations OFF.
# Mirrors cb_run.py protocol: same system prompt, tools, caps (first 6144, final 2048), tool output clip,
# max 6 turns, EOS handling. Each task is generated alone (batch 1, no padding, no slots).
# Output schema matches cb_run per_task.jsonl so scripts/diff_aime_vllm_vs_cb.py can read it.
# Usage: python3 -I scripts/hf_reference_aime.py <out.jsonl> <data.jsonl> <device> id [id ...]
import json, sys, time
sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
import torch, torch_npu  # noqa: F401
from transformers import AutoTokenizer
from transformers import Qwen3_5ForConditionalGeneration
from agent_protocol import SYSTEM, TOOLS, parse_call, execute, answer, correct, clip_tool_output

MODEL = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/model/qwen3.5-27b"
FIRST, FINAL, CLIP, MAX_TURNS = 6144, 2048, 2000, 6

out_path, data_path, device = sys.argv[1], sys.argv[2], sys.argv[3]
want = [int(x) for x in sys.argv[4:]]
tasks = {}
for l in open(data_path):
    r = json.loads(l); tasks[r["id"]] = r

torch.set_num_threads(8); torch.manual_seed(42)
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": device}).eval()
eos = model.generation_config.eos_token_id
eos = set(eos if isinstance(eos, list) else [eos])

def gen(prompt_ids, cap):
    with torch.inference_mode():
        out = model.generate(prompt_ids, attention_mask=torch.ones_like(prompt_ids), max_new_tokens=cap,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    new = out[0, prompt_ids.shape[1]:].tolist()
    ids = []
    for v in new:
        ids.append(v)
        if v in eos:
            break
    return ids

with open(out_path, "w") as f:
    f.write(json.dumps({"kind": "metadata", "framework": "hf-single-reference", "protocol": "original",
                        "first_max_tokens": FIRST, "final_max_tokens": FINAL, "tool_output_max_chars": CLIP,
                        "max_turns": MAX_TURNS, "optimizations": False}) + "\n")
    for i in want:
        t = tasks[i]
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": t["question"]}]
        rounds, tool_count, t0 = [], 0, time.perf_counter()
        turn = 0
        while True:
            prompt = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
            ids_t = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            cap = FIRST if turn == 0 else FINAL
            ids = gen(ids_t, cap)
            text = tok.decode(ids, skip_special_tokens=True)
            code = parse_call(text)
            rec = dict(turn=turn, tokens=ids, text=text, prompt_tokens=int(ids_t.shape[1]))
            msgs.append({"role": "assistant", "content": text})
            if code is not None:
                result = clip_tool_output(execute(code), CLIP)
                tool_count += 1
                rec.update(code=code, tool_output=result)
                msgs.append({"role": "tool", "name": "python", "content": result})
            rounds.append(rec)
            if code is not None and turn < MAX_TURNS - 1:
                turn += 1
                continue
            break
        pred = answer(text) if code is None else None
        f.write(json.dumps({"kind": "task", "id": i, "gold": t["answer"], "final_text": text,
                            "prediction": pred, "tool_count": tool_count,
                            "correct": correct(pred, t["answer"]) if pred else False,
                            "completion_seconds": time.perf_counter() - t0, "rounds": rounds}) + "\n")
        f.flush()
        print("done", i, "rounds", len(rounds), "pred", pred, flush=True)
print("finished", flush=True)
