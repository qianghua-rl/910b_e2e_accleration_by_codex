"""Feasibility of NPUGraph capture for CBEngine decode (16 slots, fast-GDN).
Graphs are captured per Lmax bucket BEFORE any prefill, with all slots empty (state gets garbage,
prefill overwrites it). Then the same 16 tasks are decoded eagerly and via graph replay, and the
greedy token sequences are compared. Prints per-step wall time for both paths."""
import json, sys, time, math
import torch
import torch_npu  # noqa: F401
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
from agent_protocol import SYSTEM, TOOLS  # noqa: E402
from cb_engine import CBEngine, patch_model  # noqa: E402

MODEL = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/model/qwen3.5-27b"
DATA = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/data/gsm8k_128.jsonl"
DEV, N, T, STEPS, BUCKET = "npu:0", 16, 4096, 40, 512
dev = DEV

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": DEV}).eval()
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
patch_model(model, fast_gdn=True)
eng = CBEngine(model, n_slots=N, t_max=T, device=DEV)

# static inputs
tok_buf = torch.zeros(N, dtype=torch.long, device=dev)
pos_buf = torch.zeros(N, dtype=torch.long, device=dev)
idx = eng._all_idx

def core(Lmax):
    """Same math as CBEngine.decode_step, but inputs come from static buffers and nothing is
    created from host data, so it can be captured. Returns argmax tokens [N] (device tensor)."""
    pos = pos_buf
    mask = (torch.arange(Lmax, device=dev)[None, :] <= pos[:, None])[:, None, None, :]
    x = eng.lm.embed_tokens(tok_buf[:, None])
    pos3 = pos.view(1, N, 1).expand(3, N, 1)
    cos, sin = eng.lm.rotary_emb(x, pos3)
    for i, layer in enumerate(eng.lm.layers):
        h = layer.input_layernorm(x)
        if eng.cfg.layer_types[i] == "full_attention":
            h = eng._attn(layer.self_attn, i, h, cos, sin, idx, pos, Lmax, mask)
        else:
            h = eng._gdn(layer.linear_attn, i, h, idx)
        x = x + h
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    x = eng.lm.norm(x)
    logits = eng.model.lm_head(x[:, 0])
    return logits.argmax(-1)

def bucket_of(L):
    return min(T, int(math.ceil(L / BUCKET)) * BUCKET)

graphs, outs = {}, {}
buckets = list(range(BUCKET, T + 1, BUCKET))
with torch.inference_mode():
    pos_buf.zero_(); tok_buf.zero_()
    core(BUCKET)  # eager warmup (creates lazy helpers); mutates empty slots only
    torch.npu.synchronize(dev)
    t0 = time.perf_counter()
    pool = torch.npu.graph_pool_handle()
    for b in buckets:
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g, pool=pool):
            o = core(b)
        graphs[b], outs[b] = g, o
    torch.npu.synchronize(dev)
    print(f"captured {len(graphs)} graphs in {time.perf_counter()-t0:.1f}s", flush=True)
    # reset empty-slot state written during capture
    for i in eng.kv_k: eng.kv_k[i].zero_(); eng.kv_v[i].zero_()
    for i in eng.conv: eng.conv[i].zero_()
    for i in eng.rec: eng.rec[i].zero_()
    eng.lens = [0] * N

    tasks = [json.loads(l) for l in open(DATA)][:N]
    def prefill_all():
        last = []
        for s, t in enumerate(tasks):
            p = tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                         {"role": "user", "content": t["question"]}],
                                        tools=TOOLS, tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)
            ids = tok(p, add_special_tokens=False, return_tensors="pt").input_ids.to(DEV)
            last.append(eng.prefill(s, ids))
        return last

    def run(mode):
        last = prefill_all()
        toks = [[t] for t in last]
        times = []
        for _ in range(STEPS):
            lens = eng.lens[:]
            if mode == "eager":
                ts = time.perf_counter()
                nxt = eng.decode_step(list(range(N)), [tk[-1] for tk in toks])
                times.append(time.perf_counter() - ts)
            else:
                b = bucket_of(max(lens) + 1)
                pos_buf.copy_(torch.tensor(lens, dtype=torch.long))
                tok_buf.copy_(torch.tensor([tk[-1] for tk in toks], dtype=torch.long))
                torch.npu.synchronize(dev)
                ts = time.perf_counter()
                graphs[b].replay()
                nxt = outs[b].tolist()
                times.append(time.perf_counter() - ts)
                for s in range(N): eng.lens[s] += 1
            for s in range(N): toks[s].append(nxt[s])
        return toks, times

    te, et = run("eager")
    tg, gt = run("graph")
    print(f"eager mean step: {sum(et)/len(et)*1000:.1f} ms", flush=True)
    same = sum(a == b for a, b in zip(te, tg))
    print(f"identical token sequences: {same}/{N}", flush=True)
    for s in range(N):
        if te[s] != tg[s]:
            k = next(i for i,(a,b) in enumerate(zip(te[s],tg[s])) if a!=b)
            print(f"  slot {s} first diff at step {k}", flush=True)
    print(f"graph replay mean step: {sum(gt)/len(gt)*1000:.1f} ms (steps={len(gt)})", flush=True)
