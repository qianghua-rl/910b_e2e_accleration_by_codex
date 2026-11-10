"""Module-labelled decode profile at batch=1 (same config as the serial baseline).

Runs one question through prefill, warms up, then profiles N decode steps.
Each decoder layer's linear_attn (GDN), self_attn (full attention) and mlp are wrapped
with record_function labels so device time can be attributed by module type.
Usage: profile_decode.py <out_dir> [--device npu:1] [--steps 4] [--question 0]
"""
import argparse
import collections
import json
import os
import time

import torch
import torch.profiler as tp
import torch_npu  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/model/qwen3.5-27b"
DATA = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/data/gsm8k_128.jsonl"
LABELS = ("linear_attn", "self_attn", "mlp")


def attach_labels(model):
    """Wrap each labelled submodule's forward in a record_function range."""
    stacks = collections.defaultdict(list)
    for name, mod in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf not in LABELS:
            continue
        label = f"{leaf}:{name}"

        def pre(m, args, _label=label):
            rf = tp.record_function(_label)
            rf.__enter__()
            stacks[_label].append(rf)

        def post(m, args, out, _label=label):
            stacks[_label].pop().__exit__(None, None, None)

        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--device", default="npu:1")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--question", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    q = [json.loads(l) for l in open(DATA)][a.question]
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map=a.device).eval()
    tools = [{"type": "function", "function": {
        "name": "python", "description": "Run Python code; print the result.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]
    msgs = [{"role": "user", "content": q["question"] +
             "\nSolve with Python, then give the final answer as \\boxed{N}."}]
    prompt = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True,
                                     tokenize=False, enable_thinking=False)
    ids = tok(prompt, return_tensors="pt").input_ids.to(a.device)

    attach_labels(model)
    with torch.no_grad():
        t0 = time.time()
        out = model(ids, use_cache=True)
        torch.npu.synchronize(a.device)
        print(f"prefill {ids.shape[1]} tokens: {time.time() - t0:.2f}s", flush=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        # warmup decode step (not profiled)
        out = model(nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        torch.npu.synchronize(a.device)

        import torch_npu.profiler as tnp
        prof = tnp.profile(
            activities=[tnp.ProfilerActivity.CPU, tnp.ProfilerActivity.NPU],
            on_trace_ready=tnp.tensorboard_trace_handler(a.out_dir),
            record_shapes=False, profile_memory=False, with_stack=False)
        prof.start()
        t1 = time.time()
        for _ in range(a.steps):
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        torch.npu.synchronize(a.device)
        wall = time.time() - t1
        prof.stop()

    # torch_npu writes an ascend_pt directory; locate its Chrome-format trace.
    trace = None
    for root, _, files in os.walk(a.out_dir):
        if "trace_view.json" in files:
            trace = os.path.join(root, "trace_view.json")
    summary = {"device": a.device, "steps": a.steps, "decode_wall_s": wall,
               "decode_ms_per_step_profiled": wall * 1000 / a.steps,
               "trace": trace}
    json.dump(summary, open(os.path.join(a.out_dir, "profile_meta.json"), "w"), indent=2)
    print("PROFILE_META", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
