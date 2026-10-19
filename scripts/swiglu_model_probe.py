"""Model-level Ascend SwiGLU probe on a tiny real Qwen3.5."""
import copy, json, time, torch, torch_npu
from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP
cfg=json.load(open('model/qwen3.5-27b/config.json'))['text_config']
cfg.update(num_hidden_layers=4, layer_types=cfg['layer_types'][:4], vocab_size=1024)
c=Qwen3_5TextConfig(**cfg); c._attn_implementation='sdpa'
torch.manual_seed(42); torch.set_num_threads(8)
ref=Qwen3_5ForCausalLM(c).to(dtype=torch.bfloat16,device='npu').eval(); fused=copy.deepcopy(ref)
def fwd(self,x):
    gate=self.gate_proj(x); up=self.up_proj(x)
    return self.down_proj(torch_npu.npu_swiglu(torch.cat((gate,up),dim=-1)))
n=0
for m in fused.modules():
    if isinstance(m,Qwen3_5MLP): m.forward=fwd.__get__(m,Qwen3_5MLP); n+=1
ids=torch.randint(0,1024,(1,32),device='npu')
with torch.inference_mode():
    t=time.perf_counter(); a=ref(ids,use_cache=True,logits_to_keep=1); torch.npu.synchronize(); ta=time.perf_counter()-t
    t=time.perf_counter(); b=fused(ids,use_cache=True,logits_to_keep=1); torch.npu.synchronize(); tb=time.perf_counter()-t
d=(a.logits.float()-b.logits.float()).abs()
r=dict(mlp=n,ref_seconds=ta,fused_seconds=tb,speedup=ta/tb,max_logit_error=float(d.max().cpu()),mean_logit_error=float(d.mean().cpu()),argmax_equal=bool(a.logits.argmax(-1).eq(b.logits.argmax(-1)).all().item()),finite=bool(b.logits.isfinite().all().item()))
print(json.dumps(r),flush=True); open('results/swiglu_model_probe.json','w').write(json.dumps(r)+'\n')
