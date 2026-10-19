"""S1 prototype: greedy token loop using model.forward and a reusable cache."""
import json, time
import torch, torch_npu
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from agent_protocol import CONCISE_SYSTEM, TOOLS, parse_call, execute, answer, correct

torch.set_num_threads(8); torch.manual_seed(42)
path = 'model/qwen3.5-27b'
tok = AutoTokenizer.from_pretrained(path)
model = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'':'npu:0'}).eval()
task = json.loads(open('data/gsm8k_128.jsonl').readline())
messages = [{'role':'system','content':CONCISE_SYSTEM},{'role':'user','content':task['question']}]
rounds=[]; tool_count=0; prediction=None; start=time.perf_counter()
with torch.inference_mode():
    for turn in range(6):
        prompt = tok.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True, enable_thinking=False, tokenize=False)
        ids = tok(prompt, return_tensors='pt', add_special_tokens=False).input_ids.to('npu')
        generated=[]; past=None; input_ids=ids
        for _ in range(512):
            out=model(input_ids=input_ids, past_key_values=past, use_cache=True, logits_to_keep=1)
            past=out.past_key_values; nxt=out.logits[:,-1,:].argmax(-1,keepdim=True)
            generated.append(int(nxt.item())); input_ids=nxt
            if int(nxt.item()) in set(tok.eos_token_id if isinstance(tok.eos_token_id,list) else [tok.eos_token_id]): break
        torch.npu.synchronize(); text=tok.decode(generated,skip_special_tokens=True)
        rec=dict(turn=turn,tokens=generated,text=text); code=parse_call(text)
        messages.append({'role':'assistant','content':text})
        if code is None:
            prediction=answer(text); rounds.append(rec); break
        val=execute(code); tool_count+=1; rec.update(code=code,tool_output=val)
        messages.append({'role':'tool','name':'python','content':val}); rounds.append(rec)
row=dict(id=task['id'],rounds=rounds,prediction=prediction,expected=task['answer'],tool_count=tool_count,correct=tool_count>0 and correct(prediction,task['answer']),seconds=time.perf_counter()-start)
print(json.dumps(row,ensure_ascii=False),flush=True)
open('results/single_step_agent_smoke.json','w').write(json.dumps(row)+'\n')
