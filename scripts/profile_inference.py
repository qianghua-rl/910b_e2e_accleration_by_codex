"""Separate diagnostic run: full model prefill and cached decode, not e2e timing."""
import json
import time
import torch
import torch_npu
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from agent_protocol import SYSTEM, TOOLS

torch.set_num_threads(8)
model = Qwen3_5ForConditionalGeneration.from_pretrained('model/qwen3.5-27b',
    dtype=torch.bfloat16, attn_implementation='eager', device_map={'':'npu:0'}).eval()
tok = AutoTokenizer.from_pretrained('model/qwen3.5-27b')
question = json.loads(open('data/gsm8k_128.jsonl').readline())['question']
inputs = tok.apply_chat_template([{'role':'system','content':SYSTEM},
    {'role':'user','content':question}], tools=TOOLS, add_generation_prompt=True,
    enable_thinking=False, return_dict=True, return_tensors='pt').to('npu')
with torch.inference_mode():
    out = model(**inputs, use_cache=True, logits_to_keep=1)
    for _ in range(4):
        out = model(input_ids=out.logits.argmax(-1), past_key_values=out.past_key_values,
                    use_cache=True, logits_to_keep=1)
    del out
    torch.npu.synchronize()
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=True, profile_memory=True, with_stack=False,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler('results/profile_baseline')) as prof:
        with torch.profiler.record_function('prefill'):
            out = model(**inputs, use_cache=True, logits_to_keep=1)
        for _ in range(8):
            with torch.profiler.record_function('decode'):
                out = model(input_ids=out.logits.argmax(-1), past_key_values=out.past_key_values,
                            use_cache=True, logits_to_keep=1)
        torch.npu.synchronize()
