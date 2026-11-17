"""Check npu_moe_gating_top_k_softmax against HF softmax+topk on a real Gemma router (layer 5, npu:1).
Reports: index agreement, weight max diff, and dispatched op count for the softmax+topk segment only."""
import sys, collections, torch, torch_npu  # noqa: F401
from torch.utils._python_dispatch import TorchDispatchMode
sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
from transformers import AutoModelForCausalLM
from gemma_run import MODEL
dev = "npu:1"
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=dev).eval()
lay = [l for l in m.model.language_model.layers if getattr(l, "enable_moe_block", False)][5]
r = lay.router
g = torch.Generator(device="cpu").manual_seed(2)
x = torch.randn(16, m.config.get_text_config().hidden_size, generator=g).to(torch.bfloat16).to(dev)

class C(TorchDispatchMode):
    def __init__(self): super().__init__(); self.c = collections.Counter()
    def __torch_dispatch__(self, f, t, a=(), k=None):
        self.c[str(f)] += 1; return f(*a, **(k or {}))

K = m.config.get_text_config().top_k_experts
with torch.inference_mode():
    h = r.norm(x) * r.scale * r.scalar_root_size
    logits = r.proj(h)
    def hf_seg(l):
        p = torch.softmax(l, dim=-1)
        w, i = torch.topk(p, k=K, dim=-1)
        return w, i
    w_ref, i_ref = hf_seg(logits)
    with C() as c1:
        hf_seg(logits)
    w_f, i_f, _ = torch_npu.npu_moe_gating_top_k_softmax(logits.contiguous(), None, k=K)
    with C() as c2:
        torch_npu.npu_moe_gating_top_k_softmax(logits.contiguous(), None, k=K)
    # normalized weights (what the router actually uses)
    wn_ref = w_ref / w_ref.sum(-1, keepdim=True)
    wn_f = w_f.float() / w_f.float().sum(-1, keepdim=True)
    same_set = (torch.sort(i_ref, -1)[0] == torch.sort(i_f.long(), -1)[0]).all(-1).float().mean().item()
print("HF softmax+topk ops:", sum(c1.c.values()), dict(c1.c))
print("fused gating ops:   ", sum(c2.c.values()), dict(c2.c))
print("expert set agreement per row:", same_set)
print("normalized weight max diff:", (wn_f - wn_ref.float()).abs().max().item())
