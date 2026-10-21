"""Capture a real Qwen3.5 B=16 recurrent decode step."""
import argparse,json
from pathlib import Path
import torch,torch_npu
from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
import transformers.models.qwen3_5.modeling_qwen3_5 as qm
p=argparse.ArgumentParser(); p.add_argument('--output',default='results/gdn_real_batch_artifact.pt'); p.add_argument('--batch-size',type=int,default=16); a=p.parse_args()
snap={}; hf=qm.torch_recurrent_gated_delta_rule
def cap(q,k,v,g,beta,initial_state,output_final_state,use_qk_l2norm_in_kernel=False):
    if initial_state is not None and not snap: snap.update({n:x.detach().cpu() for n,x in zip(('q','k','v','g','beta','state'),(q,k,v,g,beta,initial_state))})
    return hf(q,k,v,g,beta,initial_state,output_final_state,use_qk_l2norm_in_kernel)
qm.fused_recurrent_gated_delta_rule=cap; path='model/qwen3.5-27b'; tok=AutoTokenizer.from_pretrained(path)
question=json.loads(Path('data/gsm8k_128.jsonl').read_text().splitlines()[0])['question']
from agent_protocol import CONCISE_SYSTEM,TOOLS
prompt=tok.apply_chat_template([{'role':'system','content':CONCISE_SYSTEM},{'role':'user','content':question}],tools=TOOLS,add_generation_prompt=True,enable_thinking=False,tokenize=False)
model=Qwen3_5ForConditionalGeneration.from_pretrained(path,dtype=torch.bfloat16,attn_implementation='sdpa').to('npu').eval(); inputs=tok([prompt]*a.batch_size,return_tensors='pt',padding=True,add_special_tokens=False).to('npu')
with torch.inference_mode(): generated=model.generate(**inputs,max_new_tokens=2,do_sample=False,use_cache=True,return_dict_in_generate=True,output_scores=True); torch.npu.synchronize()
if not snap: raise RuntimeError('no recurrent decode tensor captured')
q,k,v,g,beta,state=(snap[n].to('npu') for n in ('q','k','v','g','beta','state'))
with torch.inference_mode(): out,final=hf(q,k,v,g,beta,state.clone(),True,True); torch.npu.synchronize()
torch.save({'q':q.cpu(),'k':k.cpu(),'v':v.cpu(),'g':g.cpu(),'beta':beta.cpu(),'state':state.cpu(),'hf_output':out.cpu(),'hf_state':final.cpu(),'reference_logits':generated.scores[-1].cpu() if generated.scores else None,'generated_ids':generated.sequences.cpu()},a.output)
print(json.dumps({'artifact':a.output,'shapes':{n:list(snap[n].shape) for n in snap}}))
