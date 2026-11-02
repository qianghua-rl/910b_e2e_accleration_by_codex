"""E172 arithmetic without diagnostic captures/events; isolated experiment only."""
import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
from e168_batched_history import BatchedHistory
from recurrent_single_step import recurrent_single_step


class HistoryRuntime(BatchedHistory):
    def conv(self,x,state,weight,bias=None,activation=None):
        outputs=[]
        for t in range(x.shape[-1]):
            active=(self.lengths>t).view(-1,1,1)
            working=state.clone()
            out=hf.torch_causal_conv1d_update(x[...,t:t+1],working,weight,bias,activation)
            state.copy_(torch.where(active,working,state))
            outputs.append(torch.where(active,out,0))
        return torch.cat(outputs,dim=-1)

    def recurrent(self,query,key,value,g,beta,initial_state,output_final_state,
                  use_qk_l2norm_in_kernel=False):
        state=initial_state
        outputs=[]
        for t in range(query.shape[1]):
            out,next_state=recurrent_single_step(query[:,t:t+1],key[:,t:t+1],value[:,t:t+1],
                g[:,t:t+1],beta[:,t:t+1],state,True,use_qk_l2norm_in_kernel)
            active=self.lengths>t
            state=torch.where(active[:,None,None,None],next_state.to(initial_state.dtype),state)
            outputs.append(torch.where(active[:,None,None,None],out,0))
        return torch.cat(outputs,dim=1),state if output_final_state else None


def continue_batched(model,runtime,cache,ids,mask,positions,suffixes,eos,max_new_tokens=12):
    """Append once, then greedily decode full B; EOS samples are not consumed."""
    runtime.lengths=ids.new_tensor(list(map(len,suffixes)))
    starts=positions[:,:1]+ids.new_tensor([[len(suffix)] for suffix in suffixes])
    out=model(input_ids=ids,attention_mask=mask,position_ids=positions,past_key_values=cache,
        use_cache=True,return_dict=True,logits_to_keep=0)
    cache=out.past_key_values
    next_ids=torch.stack([out.logits[row,len(suffix)-1].argmax(-1) for row,suffix in enumerate(suffixes)])
    del out
    calls=1;active=[True]*len(suffixes);generated=[[] for _ in suffixes]
    for step in range(max_new_tokens):
        for row,token in enumerate(next_ids.tolist()):
            if active[row]:
                generated[row].append(token)
                if token in eos:active[row]=False
        if not any(active) or step==max_new_tokens-1:break
        live=ids.new_tensor(active,dtype=torch.long)
        mask=torch.cat((mask,live[:,None]),dim=-1)
        positions=starts+step
        runtime.lengths=live
        out=model(input_ids=next_ids[:,None],attention_mask=mask,position_ids=positions,
            past_key_values=cache,use_cache=True,return_dict=True,logits_to_keep=1)
        cache=out.past_key_values;next_ids=out.logits[:,-1].argmax(-1)
        calls+=1
        del out
    return generated,cache,mask,positions,calls
