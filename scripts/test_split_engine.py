"""Correctness check for the two-card layer split (cb_run --split-devices).

Runs CBEngine greedy decode on a few fixed prompts and dumps the generated token ids to JSON.
Run once in single-card mode (--device npu:0) and once in split mode (--split-devices npu:0,npu:1),
then compare the two files with --compare. Both runs must be eager (no graph decode).

  python -I scripts/test_split_engine.py --out results/split_test/single.json --device npu:0
  python -I scripts/test_split_engine.py --out results/split_test/split.json --split-devices npu:0,npu:1
  python -I scripts/test_split_engine.py --compare results/split_test/single.json results/split_test/split.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")

MODEL = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/model/qwen3.5-27b"
DATA = "/remote-home/S45160/work/910b_fresh/data/benchmarks/math500.jsonl"


def compare(a_path, b_path):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    same = sum(x["tokens"] == y["tokens"] for x, y in zip(a["runs"], b["runs"]))
    first_div = []
    for x, y in zip(a["runs"], b["runs"]):
        d = next((i for i, (p, q) in enumerate(zip(x["tokens"], y["tokens"])) if p != q), None)
        first_div.append(d)
    print(f"identical token sequences: {same}/{len(a['runs'])}")
    print("first divergence index per prompt:", first_div)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--split-devices", default=None)
    ap.add_argument("--n-prompts", type=int, default=3)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--t-max", type=int, default=1024)
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        return compare(*a.compare)

    import torch
    import torch_npu  # noqa: F401
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    import cb_engine
    from cb_engine import CBEngine, patch_model, split_device_map

    dev = a.device
    if a.split_devices:
        d0, d1 = a.split_devices.split(",")
        dev = d0
        dmap = split_device_map(64, d0, d1)
    else:
        dmap = {"": a.device}
    t0 = time.perf_counter()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map=dmap).eval()
    tok = AutoTokenizer.from_pretrained(MODEL)
    patch_model(model, fast_gdn=True, gdn_v2=True)
    eng = CBEngine(model, n_slots=a.slots, t_max=a.t_max, device=dev, graph_decode=False)
    print(f"loaded in {time.perf_counter() - t0:.1f}s, split={eng.split}, ldev[0]={eng.ldev[0]}, "
          f"ldev[-1]={eng.ldev[-1]}, ddev={eng.ddev}")

    import json as _j
    rows = [_j.loads(l) for l in open(DATA)][: a.n_prompts]
    runs = []
    with torch.inference_mode():
        for slot, r in enumerate(rows):
            msgs = [{"role": "user", "content": r["question"]}]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
            ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
            first = eng.prefill(slot, ids)
            gen = [first]
            for _ in range(a.steps - 1):
                gen.append(eng.decode_step([slot], [gen[-1]])[0])
            runs.append({"id": r["id"], "tokens": gen})
            eng.free(slot)
            eng._reset_state()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"split": a.split_devices, "runs": runs}, open(a.out, "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
