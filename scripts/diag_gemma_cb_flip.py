"""At the first divergence of GemmaCB vs HF generate, compare logits.
Reference = HF full forward (no cache) on prompt + HF-generated prefix.
Reports: max |logit diff| between CB decode logits and HF full-forward logits at that step,
and HF top-2 margin. A near-tie (tiny margin) with bf16-sized diff means a numeric flip."""
import json, sys, torch, torch_npu  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
import agent_protocol as P
from gemma_cb import GemmaCB
MODEL = "/remote-home/S45160/work/910b_fresh/gemma-4-26B-A4B-it"
DATA = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/data/gsm8k_128.jsonl"
DEV = "npu:1"; STEPS = 40
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=DEV).eval()
qs = [json.loads(l) for l in open(DATA)][:4]
for i in [1, 3]:
    q = qs[i]
    msgs = [{"role": "system", "content": P.SYSTEM}, {"role": "user", "content": q["question"]}]
    p = tok.apply_chat_template(msgs, tools=P.TOOLS, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to(DEV)
    with torch.no_grad():
        ref = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=STEPS, min_new_tokens=STEPS, do_sample=False, pad_token_id=tok.pad_token_id)[0, ids.shape[1]:].tolist()
    eng = GemmaCB(model, tok, n_slots=1, t_max=2048, device=DEV)
    lg = eng.prefill_into_slot(0, ids); gen = [int(lg.argmax())]; L = ids.shape[1]
    cb_logits = {0: lg.float()}
    for k in range(1, STEPS):
        out = eng.decode_step(torch.tensor([gen[-1]], device=DEV), torch.tensor([L], device=DEV))
        cb_logits[k] = out[0].float(); gen.append(int(out[0].argmax())); L += 1
    d = next(k for k in range(STEPS) if ref[k] != gen[k])
    # HF full forward on prompt + ref[:d] to get the reference distribution at step d
    full = torch.cat([ids, torch.tensor([ref[:d]], device=DEV)], dim=1)
    with torch.no_grad():
        hf = model(input_ids=full).logits[0, -1].float()
    top2 = torch.topk(hf, 2)
    print(f"prompt{i}: first_div={d} ref_tok={ref[d]} cb_tok={gen[d]}", flush=True)
    print(f"  HF top2 ids={top2.indices.tolist()} vals={[round(v,3) for v in top2.values.tolist()]} margin={top2.values[0].item()-top2.values[1].item():.4f}", flush=True)
    print(f"  max|cb - hf_full| at step {d} = {(cb_logits[d]-hf).abs().max().item():.4f}  | argmax cb={int(cb_logits[d].argmax())} hf={int(hf.argmax())}", flush=True)
