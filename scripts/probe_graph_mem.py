"""Memory breakdown for graph decode: allocated/reserved after load, engine init, graph capture,
and after a prefill-group of 4. Args: graph (0/1), t_max."""
import json, sys, torch, torch_npu  # noqa: F401
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
from agent_protocol import SYSTEM, TOOLS  # noqa: E402
from cb_engine import CBEngine, patch_model  # noqa: E402
MODEL = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/model/qwen3.5-27b"
DATA = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/data/gsm8k_128.jsonl"
DEV = "npu:0"; GRAPH = bool(int(sys.argv[1])); T = int(sys.argv[2])
def mem(tag):
    a = torch.npu.memory_allocated(DEV) / 2**30; r = torch.npu.memory_reserved(DEV) / 2**30
    print(f"{tag:28s} alloc {a:6.2f} GiB  reserved {r:6.2f} GiB", flush=True)
mem("start")
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": DEV}).eval()
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
mem("after model load")
patch_model(model, fast_gdn=True)
eng = CBEngine(model, n_slots=16, t_max=T, device=DEV, graph_decode=GRAPH)
mem("after CBEngine init")
tasks = [json.loads(l) for l in open(DATA)][:4]
with torch.inference_mode():
    ids = []
    for t in tasks:
        p = tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                     {"role": "user", "content": t["question"]}],
                                    tools=TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids.append(tok(p, add_special_tokens=False, return_tensors="pt").input_ids.to(DEV))
    eng.prefill_batch([0, 1, 2, 3], ids)
    torch.npu.synchronize(DEV)
mem("after prefill group 4")
print("peak alloc", round(torch.npu.max_memory_allocated(DEV) / 2**30, 2), "GiB", flush=True)
