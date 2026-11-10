"""CPU correctness test for src/cb_tp.py: tensor-parallel (2 processes, gloo) vs single-process CBEngine.

Tiny random Qwen3.5 (same seed in every process), float32. Checks that TP prefill + eager decode produce
exactly the same greedy tokens as the single-process engine. Run:
  /remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python -I scripts/test_tp_cpu.py
"""
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration  # noqa: E402

import cb_tp  # noqa: E402
from cb_engine import CBEngine  # noqa: E402

PORT = "29571"
STEPS = 8
PROMPT_LEN = 37


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
    return Qwen3_5ForConditionalGeneration(cfg).to(torch.bfloat16).eval()   # engine buffers are bf16


def run_tokens(engine, ids):
    first = engine.prefill(0, ids)
    toks = [first]
    for _ in range(STEPS - 1):
        toks.append(engine.decode_step([0], [toks[-1]])[0])
    return toks


def worker(rank, world):
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{PORT}", rank=rank, world_size=world)
    torch.manual_seed(1)
    ids = torch.randint(0, 256, (1, PROMPT_LEN))

    ref_model = make_model()
    ref = run_tokens(CBEngine(ref_model, n_slots=2, t_max=128, device="cpu"), ids)

    model = cb_tp.shard_model(make_model(), rank, world)
    eng = cb_tp.TPEngine(model, n_slots=2, t_max=128, device="cpu", tp=world)
    tp_toks = run_tokens(eng, ids)

    if rank == 0:
        print("reference tokens:", ref)
        print("TP tokens       :", tp_toks)
        print("MATCH" if ref == tp_toks else "MISMATCH", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
