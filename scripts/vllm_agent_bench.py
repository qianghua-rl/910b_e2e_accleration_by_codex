"""README-compatible batch agent benchmark using vLLM-Ascend."""
import argparse, hashlib, json, time
from pathlib import Path
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from agent_protocol import CONCISE_SYSTEM, MINIMAL_SYSTEM, TOOLS, parse_call, execute, answer, correct

p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--limit',type=int,default=128); p.add_argument('--batch-size',type=int,default=16); p.add_argument('--max-model-len',type=int,default=2048); p.add_argument('--gpu-memory-utilization',type=float,default=.98); p.add_argument('--protocol',choices=['concise','minimal'],default='concise')
a=p.parse_args(); out=Path(a.output)
if out.exists(): raise SystemExit('output exists')
system_prompt=CONCISE_SYSTEM if a.protocol=='concise' else MINIMAL_SYSTEM
model_path='model/qwen3.5-27b'; data_bytes=Path('data/gsm8k_128.jsonl').read_bytes(); tasks=[json.loads(x) for x in data_bytes.splitlines()][:a.limit]
t=time.perf_counter(); tok=AutoTokenizer.from_pretrained(model_path,padding_side='left'); llm=LLM(model=model_path,tensor_parallel_size=1,dtype='bfloat16',max_model_len=a.max_model_len,gpu_memory_utilization=a.gpu_memory_utilization); load=time.perf_counter()-t
sampling=SamplingParams(temperature=0,max_tokens=512,detokenize=True)
def emit(x):
 with out.open('a') as f: f.write(json.dumps(x,ensure_ascii=False)+'\n')
 print(json.dumps(x,ensure_ascii=False),flush=True)
emit(dict(kind='metadata',framework='vllm-ascend',vllm_version=__import__('vllm').__version__,tasks=len(tasks),batch_size=a.batch_size,protocol=a.protocol,system_prompt=system_prompt,data_sha256=hashlib.sha256(data_bytes).hexdigest(),max_new_tokens=512,max_rounds=6,do_sample=False,enable_thinking=False,load_seconds=load))
start=time.perf_counter()
with_tools=dict(tools=TOOLS, add_generation_prompt=True, enable_thinking=False, tokenize=False)
with __import__('contextlib').nullcontext():
 for base in range(0,len(tasks),a.batch_size):
  active=[dict(task=x,messages=[{'role':'system','content':system_prompt},{'role':'user','content':x['question']}],rounds=[],tool_count=0,started=time.perf_counter()) for x in tasks[base:base+a.batch_size]]
  for turn in range(6):
   prompts=[tok.apply_chat_template(s['messages'],**with_tools) for s in active]
   results=llm.generate(prompts,sampling,use_tqdm=False)
   survivors=[]
   for s,r in zip(active,results):
    text=r.outputs[0].text; ids=r.outputs[0].token_ids; code=parse_call(text); rec=dict(turn=turn,tokens=ids,text=text)
    s['messages'].append({'role':'assistant','content':text})
    if code is not None:
     ttool=time.perf_counter(); val=execute(code); s['tool_count']+=1; rec.update(code=code,tool_output=val,tool_seconds=time.perf_counter()-ttool); s['messages'].append({'role':'tool','name':'python','content':val}); survivors.append(s)
    else: s['prediction']=answer(text)
    s['rounds'].append(rec)
    if code is None or turn==5:
     emit(dict(kind='task',id=s['task']['id'],rounds=s['rounds'],prediction=s.get('prediction'),expected=s['task']['answer'],tool_count=s['tool_count'],correct=s['tool_count']>0 and correct(s.get('prediction'),s['task']['answer']),service_seconds=time.perf_counter()-s['started'],completion_seconds=time.perf_counter()-start))
   active=survivors
   if not active: break
emit(dict(kind='summary',makespan_seconds=time.perf_counter()-start))
