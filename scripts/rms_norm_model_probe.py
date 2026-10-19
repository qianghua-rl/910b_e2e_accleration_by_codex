"""Model-level fused RMSNorm numerical probe on a tiny real Qwen3.5."""
import copy, json, time
import torch, torch_npu
from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

cfg = json.load(open('model/qwen3.5-27b/config.json'))['text_config']
cfg.update(num_hidden_layers=4, layer_types=cfg['layer_types'][:4], vocab_size=1024)
config = Qwen3_5TextConfig(**cfg); config._attn_implementation='sdpa'
torch.manual_seed(42); torch.set_num_threads(8)
model = Qwen3_5ForCausalLM(config).to(dtype=torch.bfloat16, device='npu').eval()
fused = copy.deepcopy(model)
def fused_forward(self, x):
    return torch_npu.npu_rms_norm(x, (1.0 + self.weight).to(x.dtype), self.eps)[0]
count=0
for m in fused.modules():
    if isinstance(m, Qwen3_5RMSNorm):
        m.forward = fused_forward.__get__(m, Qwen3_5RMSNorm); count += 1
ids=torch.randint(0,1024,(1,32),device='npu')
with torch.inference_mode():
    t=time.perf_counter(); a=model(ids,use_cache=True,logits_to_keep=1); torch.npu.synchronize(); ta=time.perf_counter()-t
    t=time.perf_counter(); b=fused(ids,use_cache=True,logits_to_keep=1); torch.npu.synchronize(); tb=time.perf_counter()-t
da=(a.logits.float()-b.logits.float()).abs()
row=dict(norms=count,ref_seconds=ta,fused_seconds=tb,speedup=ta/tb,max_logit_error=float(da.max().cpu()),mean_logit_error=float(da.mean().cpu()),argmax_equal=bool(a.logits.argmax(-1).eq(b.logits.argmax(-1)).all().item()),finite=bool(b.logits.isfinite().all().item()))
print(json.dumps(row),flush=True); open('results/rms_norm_model_probe.json','w').write(json.dumps(row)+'\n')
