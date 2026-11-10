"""Single-question smoke test: load Qwen3.5-27B on one NPU, run one greedy agent episode."""
import json
import sys
import time

import torch
import torch_npu  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/model/qwen3.5-27b"
DATA = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/data/gsm8k_128.jsonl"
DEVICE = "npu:1"


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    q = [json.loads(l) for l in open(DATA)][idx]
    print("question:", q["question"], "| gold:", q["answer"], flush=True)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEVICE
    ).eval()
    print(f"load: {time.time() - t0:.1f}s", flush=True)

    msgs = [{"role": "user", "content": q["question"] + "\nSolve with Python, then give the final answer as \\boxed{N}."}]
    tools = [{"type": "function", "function": {
        "name": "python", "description": "Run Python code; print the result.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]
    prompt = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True,
                                     tokenize=False, enable_thinking=False)
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    t1 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=512, do_sample=False)
    torch.npu.synchronize()
    dt = time.time() - t1
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
    n = out.shape[1] - ids.shape[1]
    print(f"generate: {dt:.2f}s, {n} tokens, {n / dt:.2f} tok/s", flush=True)
    print("OUTPUT:\n" + text, flush=True)
    print(f"peak HBM allocated: {torch.npu.max_memory_allocated() / 2**30:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
