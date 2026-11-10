"""CPU correctness test for chunked prefill in src/cb_tp.py (TPEngine.prefill_chunk + hold-aware decode).

Two processes (gloo, tensor parallel), tiny random Qwen3.5, float32-free bf16 like the engine.
Request A is prefilled in chunks of size c while request B (already prefilled) keeps decoding in the
same engine with A held. Greedy tokens of A and B must equal the unchunked reference exactly.
Run:
  /remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python -I scripts/test_chunked_prefill_cpu.py
"""
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration  # noqa: E402

import cb_tp  # noqa: E402

PORT = "29573"
STEPS = 8
LA, LB = 37, 23
CHUNKS = [5, 8, 16, 37]


def make_model():
    torch.manual_seed(0)
    tc = dict(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
              num_attention_heads=4, num_key_value_heads=2, head_dim=16,
              linear_num_key_heads=4, linear_num_value_heads=8, linear_key_head_dim=16,
              linear_value_head_dim=16, linear_conv_kernel_dim=4,
              layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
              max_position_embeddings=512, rms_norm_eps=1e-6, attention_bias=False, hidden_act="silu")
    cfg = Qwen3_5Config(text_config=tc, vision_config=dict(depth=1, hidden_size=32, num_heads=2,
                                                          intermediate_size=64, out_hidden_size=64))
    return Qwen3_5ForConditionalGeneration(cfg).to(torch.bfloat16).eval()


def new_engine(world, rank):
    model = cb_tp.shard_model(make_model(), rank, world)
    return cb_tp.TPEngine(model, n_slots=2, t_max=128, device="cpu", tp=world)


def slot_state(eng, slot):
    return [t[slot].float().clone() for d in (eng.conv, eng.rec) for t in d.values()]


def reference(world, rank, ids):
    eng = new_engine(world, rank)
    toks = [eng.prefill(0, ids)]
    state = slot_state(eng, 0)
    for _ in range(STEPS - 1):
        toks.append(eng.decode_step([0], [toks[-1]])[0])
    return toks, state


def interleaved(world, rank, idsA, idsB, c):
    eng = new_engine(world, rank)
    tB = [eng.prefill(1, idsB)]                       # B is fully prefilled, then decodes during A's chunks
    firstA = None
    for p in range(0, LA, c):
        last = p + c >= LA
        f = eng.prefill_chunk(0, idsA[:, p:p + c], p, last)
        if last:
            firstA = f
        else:
            tB.append(eng.decode_step([1], [tB[-1]], hold=(0,))[0])
    state = slot_state(eng, 0)
    tA = [firstA]
    while len(tA) < STEPS or len(tB) < STEPS:
        out = eng.decode_step([0, 1], [tA[-1], tB[-1]])
        tA.append(out[0])
        tB.append(out[1])
    return tA[:STEPS], tB[:STEPS], state


def worker(rank, world):
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{PORT}", rank=rank, world_size=world)
    torch.manual_seed(1)
    idsA = torch.randint(0, 256, (1, LA))
    idsB = torch.randint(0, 256, (1, LB))
    refA, refS = reference(world, rank, idsA)
    refB, _ = reference(world, rank, idsB)
    ok = True
    for c in CHUNKS:
        tA, tB, st = interleaved(world, rank, idsA, idsB, c)
        diff = max((a - b).abs().max().item() for a, b in zip(st, refS))
        good = (tA == refA) and (tB == refB) and diff < 1e-2
        ok &= good
        if rank == 0:
            print(f"chunk={c:>3}: {'MATCH' if good else 'MISMATCH'}"
                  f"  A_ok={tA == refA} B_ok={tB == refB} state_maxdiff={diff:.2e}", flush=True)
            if not good:
                print("  refA", refA, "\n  gotA", tA, "\n  refB", refB, "\n  gotB", tB, flush=True)
    if rank == 0:
        print("ALL MATCH" if ok else "FAILURES", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
