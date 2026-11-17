"""Compare one MoE expert block: experts_grouped(geglu=False) vs geglu=True (with half-swapped gate_up)."""
import sys, torch, torch_npu  # noqa: F401
sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
from transformers import AutoModelForCausalLM
from gemma_run import MODEL
from gemma_cb import experts_grouped
dev = "npu:1"
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=dev).eval()
lay = [l for l in m.model.language_model.layers if getattr(l, "enable_moe_block", False)][0]
ex = lay.experts
E, H = ex.num_experts, ex.gate_up_proj.shape[2]
g = torch.Generator(device="cpu").manual_seed(1)
x = torch.randn(64, H, generator=g).to(torch.bfloat16).to(dev)
idx = torch.stack([torch.randperm(E, generator=g)[:8] for _ in range(64)]).to(dev)
w = torch.softmax(torch.randn(64, 8, generator=g), -1).to(torch.bfloat16).to(dev)
with torch.inference_mode():
    ref = experts_grouped(ex, x, idx, w, geglu=False)
    # HF reference (per-expert loop) for ground truth
    hf = ex(x, idx, w)
    W0 = ex.gate_up_proj.data.clone()
    half = W0.shape[1] // 2
    ex.gate_up_proj.data = torch.cat([W0[:, half:], W0[:, :half]], 1).contiguous()
    got = experts_grouped(ex, x, idx, w, geglu=True)
    ex.gate_up_proj.data = W0
print("grouped vs HF          max diff", (ref.float() - hf.float()).abs().max().item(), "ref max", ref.float().abs().max().item())
print("geglu(swapped) vs grp  max diff", (got.float() - ref.float()).abs().max().item())
