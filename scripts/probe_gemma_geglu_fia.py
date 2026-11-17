"""Probe two fusions in one Gemma CB decode step on the same random state.
  geglu: MoE experts gelu(gate)*up via torch_npu.npu_geglu (weights half-swapped at construction).
  fia:   attention QK/scale/mask/softmax/PV via torch_npu.npu_fused_infer_attention_score.
Each mode runs in its own process (geglu changes weights at construction time).
Modes: base | geglu | fia | both. Prints op count, step time, logits diff vs the base reference
(base reference is saved to /tmp/geglu_fia_ref.pt by --mode base and loaded by the others),
and per-layer attention output diff for layers 0 (sliding) and 5 (full) when fia is on."""
import argparse, sys, time, os
sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
import torch
import torch_npu  # noqa: F401
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import AutoModelForCausalLM, AutoTokenizer
from gemma_run import MODEL
from gemma_cb import GemmaCB
from gemma_fast_norm import patch_gemma_norms

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["base", "geglu", "fia", "both"])
ap.add_argument("--device", default="npu:1")
ap.add_argument("--pos", type=int, default=200)
ap.add_argument("--tact", type=int, default=256)
ap.add_argument("--iters", type=int, default=20)
a = ap.parse_args()
REF = "/tmp/geglu_fia_ref.pt"
dev, N = a.device, 16
geglu = a.mode in ("geglu", "both")
fia = a.mode in ("fia", "both")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=dev).eval()
patch_gemma_norms(model, noscale=True)
eng = GemmaCB(model, tok, N, 2048, dev, grouped=True, fused_rope=True, geglu=geglu, fia=fia)

g = torch.Generator(device="cpu").manual_seed(0)
for li in range(len(eng.K)):
    eng.K[li].copy_(torch.randn(eng.K[li].shape, generator=g).mul_(0.5).to(torch.bfloat16))
    eng.V[li].copy_(torch.randn(eng.V[li].shape, generator=g).mul_(0.5).to(torch.bfloat16))
toks = torch.randint(0, 200000, (N,), generator=g).to(dev)
pos = torch.full((N,), a.pos, dtype=torch.long, device=dev)


class Counter(TorchDispatchMode):
    def __init__(self):
        super().__init__(); self.n = 0
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.n += 1
        return func(*args, **(kwargs or {}))


with torch.inference_mode():
    # per-layer attention check (fia vs base _attn on identical input)
    if fia:
        x = eng.lm.embed_tokens(toks.view(N, 1))
        rope = {lt: eng.lm.rotary_emb(x, pos.view(N, 1), lt) for lt in set(eng.layer_types)}
        masks = eng._masks(pos, a.tact)
        for li in (0, 5):
            layer = eng.lm.layers[li]
            lt = eng.layer_types[li]
            cos, sin = rope[lt]
            h_in = layer.input_layernorm(x)
            eng.fia = False
            ref = eng._attn(li, layer, h_in, pos, cos, sin, masks[lt], eng.ar)
            eng.fia = True
            got = eng._attn(li, layer, h_in, pos, cos, sin, masks[lt], eng.ar)
            print(f"layer {li} ({lt}) attn out: max|ref|={ref.float().abs().max().item():.4g} "
                  f"max diff={(got.float() - ref.float()).abs().max().item():.4g}", flush=True)
    out = eng.decode_step(toks, pos, eng.ar, tact=a.tact)
    torch.npu.synchronize(dev)
    c = Counter()
    with c:
        eng.decode_step(toks, pos, eng.ar, tact=a.tact)
    torch.npu.synchronize(dev)
    t0 = time.time()
    for _ in range(a.iters):
        eng.decode_step(toks, pos, eng.ar, tact=a.tact)
    torch.npu.synchronize(dev)
    ms = (time.time() - t0) / a.iters * 1000
logits = out.float().cpu()
print(f"mode={a.mode} ops={c.n} step_ms={ms:.1f} logits abs max={logits.abs().max().item():.4g}", flush=True)
if a.mode == "base":
    torch.save(logits, REF)
elif os.path.exists(REF):
    ref = torch.load(REF)
    d = (logits - ref).abs().max().item()
    agree = (logits.argmax(-1) == ref.argmax(-1)).float().mean().item()
    print(f"vs base: max_logit_diff={d:.4g} argmax_agree={agree:.3f}", flush=True)
