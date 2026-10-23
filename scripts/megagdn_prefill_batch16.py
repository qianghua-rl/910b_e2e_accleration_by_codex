"""Run the E100 first batch and record one greedy token per request."""
import argparse, json, os, sys, time
from pathlib import Path
TASK_IDS = [640,1137,214,1030,950,428,866,175,1130,445,1161,611,108,1288,554,249]
def main():
    p=argparse.ArgumentParser(); p.add_argument('--case',choices=['triton','pto_kkt_only','pto_staged','pto_mega'],required=True); p.add_argument('--output',required=True); p.add_argument('--model',default='model/qwen3.5-27b'); a=p.parse_args()
    os.environ['ASCEND_RT_VISIBLE_DEVICES']='0'; os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD','spawn'); patch='/tmp/megagdn-pto/vllm_patch'
    os.environ['VLLM_PTO_KKT_ONLY']='1' if a.case=='pto_kkt_only' else '0'
    if a.case=='pto_kkt_only': patch=str(Path(__file__).resolve().parent/'pto_kkt_patch')
    if a.case.startswith('pto_'):
        os.environ['VLLM_PTO_PATCH_DIR']=patch
        os.environ['VLLM_PTO_MEGAKERNEL']='1' if a.case=='pto_mega' else '0'
        os.environ['VLLM_PTO_CHUNK_SIZE']='128'
    import vllm_ascend.utils as vua
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    vua.adapt_patch(is_global_patch=False)
    if a.case == 'pto_kkt_only':
        os.environ['VLLM_PTO_KKT_ONLY']='1'
        sys.path.insert(0,patch); from apply import apply_kkt_only_patch; apply_kkt_only_patch()
    elif a.case.startswith('pto_'):
        sys.path.insert(0,patch); from apply import apply_pto_patch; apply_pto_patch()
    tasks={int(x['id']):x for x in (json.loads(line) for line in Path('data/gsm8k_128.jsonl').read_text().splitlines())}
    tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True)
    system=('Solve the math problem using the python tool. First respond with only a python tool call containing a short program that prints the answer. Do not explain before the tool call. After seeing the tool output, reply with only the numeric answer in \\boxed{N}.')
    tools=[{'type':'function','function':{'name':'python','description':'Execute Python code and return printed output.','parameters':{'type':'object','properties':{'code':{'type':'string'}},'required':['code']}}}]
    prompts=[tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':tasks[i]['question']}],tools=tools,tokenize=False,add_generation_prompt=True,enable_thinking=False) for i in TASK_IDS]
    llm=LLM(model=a.model,trust_remote_code=True,dtype='bfloat16',max_model_len=4096,tensor_parallel_size=1,enforce_eager=True,max_num_batched_tokens=8192,max_num_seqs=16,disable_log_stats=False,enable_prefix_caching=False)
    sp=SamplingParams(temperature=0.0,max_tokens=1,min_tokens=1,ignore_eos=False,seed=42); llm.generate(prompts,sp,use_tqdm=False); started=time.perf_counter(); outputs=llm.generate(prompts,sp,use_tqdm=False); wall=time.perf_counter()-started
    rows=[]
    for i,out in zip(TASK_IDS,outputs): rows.append({'id':i,'prompt_tokens':len(out.prompt_token_ids),'tokens':list(out.outputs[0].token_ids),'text':out.outputs[0].text})
    result={'case':a.case,'task_ids':TASK_IDS,'batch_size':16,'protocol':'concise','wall_seconds':wall,'rows':rows,'logits_finite':'not measured','tokens_are_expected_e100':all(x['tokens']==[248058] for x in rows)}
    Path(a.output).write_text(json.dumps(result)); print(json.dumps({'case':a.case,'wall_seconds':wall,'prompt_tokens':sum(x['prompt_tokens'] for x in rows),'generated_tokens':sum(len(x['tokens']) for x in rows),'tokens_are_expected_e100':result['tokens_are_expected_e100']}),flush=True)
if __name__=='__main__': main()
