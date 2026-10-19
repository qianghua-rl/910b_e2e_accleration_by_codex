"""Full checkpoint BF16 parameter exploration before optimization phases."""
import json
import time
import traceback
import torch
import torch_npu
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

torch.set_num_threads(8)
torch.manual_seed(42)
path = 'model/qwen3.5-27b'
def emit(row):
    print(json.dumps(row), flush=True)
    with open('results/model_parameters.jsonl', 'a') as f:
        f.write(json.dumps(row)+'\n')

t = time.perf_counter()
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    path, dtype=torch.bfloat16, attn_implementation='eager', device_map={'': 'npu:0'}).eval()
torch.npu.synchronize()
emit(dict(kind='load', seconds=time.perf_counter()-t,
          allocated=torch.npu.memory_allocated(), peak=torch.npu.max_memory_allocated()))
tokenizer = AutoTokenizer.from_pretrained(path)
question = json.loads(open('data/gsm8k_128.jsonl').readline())['question']
ids = tokenizer.apply_chat_template([{'role':'user','content':question}], tokenize=True,
                                   add_generation_prompt=True, enable_thinking=False,
                                   return_dict=False)
with torch.inference_mode():
    for batch in [1, 2, 4, 8]:
        for length in [128, 512, 2048]:
            row = dict(kind='shape', batch=batch, length=length)
            try:
                torch.npu.empty_cache()
                torch.npu.reset_peak_memory_stats()
                inp = torch.tensor((ids*((length+len(ids)-1)//len(ids)))[:length], device='npu').repeat(batch,1)
                torch.npu.synchronize()
                t = time.perf_counter()
                out = model(input_ids=inp, use_cache=True, logits_to_keep=1)
                torch.npu.synchronize()
                row['prefill_seconds'] = time.perf_counter()-t
                row['finite'] = bool(out.logits.isfinite().all().item())
                samples=[]
                for _ in range(8):
                    nxt = out.logits.argmax(-1)
                    torch.npu.synchronize()
                    t = time.perf_counter()
                    out = model(input_ids=nxt, past_key_values=out.past_key_values, use_cache=True, logits_to_keep=1)
                    torch.npu.synchronize()
                    samples.append(time.perf_counter()-t)
                row.update(decode_seconds=samples, allocated=torch.npu.memory_allocated(), peak=torch.npu.max_memory_allocated())
                del inp, out, nxt
            except Exception:
                row['error'] = traceback.format_exc()
                emit(row)
                raise
            emit(row)
