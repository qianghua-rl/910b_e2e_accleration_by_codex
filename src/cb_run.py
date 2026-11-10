"""Continuous-batching agent runner (HF weights + CBEngine). Output format matches e100_batch.py.

Requests are (task, turn) units. A request is admitted into a free slot (batch-1 prefill of the full
conversation so far), decoded one token per engine step together with all other active slots, and
released as soon as it hits EOS or its cap. A tool call on a non-final turn re-queues the task for the
next turn (re-prefill, same semantics as the synchronous runner).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

# Graph decode needs expandable segments (allocator fragmentation otherwise OOMs at prefill-group > 1).
# Must be set before torch_npu initialises the allocator.
if "--graph-decode" in sys.argv:
    os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")

import torch
import torch_npu  # noqa: F401
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
from agent_protocol import SYSTEM, CONCISE_SYSTEM, TOOLS, parse_call, execute, answer, correct, clip_tool_output  # noqa: E402
from cb_engine import CBEngine, patch_model  # noqa: E402

MODEL = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/model/qwen3.5-27b"
DATA = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/data/gsm8k_128.jsonl"
DIRECT_SYSTEM = ("Answer the question. Reason step by step, then give the final answer in \\boxed{}. "
                 "For multiple-choice questions, put only the option letter in \\boxed{}, e.g. \\boxed{B}.")


class Req:
    __slots__ = ("task", "messages", "turn", "tool_count", "rounds", "slot", "gen", "cap", "last")

    def __init__(self, task, system_prompt):
        self.task = task
        self.messages = [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": task["question"]}]
        self.turn = 0
        self.tool_count = 0
        self.rounds = []
        self.slot = None
        self.gen = []
        self.cap = 0
        self.last = None


def other_runs_on_device(device):
    """Return pids of other cb_run.py processes whose --device matches (ignores the current process)."""
    try:
        listing = subprocess.run(["pgrep", "-af", "cb_run.py"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return []
    me = str(os.getpid())
    hits = []
    for line in listing.splitlines():
        pid, _, cmd = line.partition(" ")
        if pid == me or "pgrep" in cmd:
            continue
        parts = cmd.split()
        # only count real interpreter processes; shell wrappers may mention cb_run.py in their argv
        if not any(os.path.basename(p).startswith("python") and not p.startswith("-") for p in parts[:2]):
            continue
        if "src/cb_run.py" not in cmd and "cb_run.py" not in parts[:3]:
            continue
        dev = parts[parts.index("--device") + 1] if "--device" in parts else "npu:0"
        if dev == device:
            hits.append(pid)
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--allow-busy", action="store_true",
                    help="start even if another cb_run.py is already on this device (timings will be contaminated)")
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--t-max", type=int, default=4096)
    ap.add_argument("--protocol", choices=["original", "concise"], default="original")
    ap.add_argument("--final-max-new-tokens", type=int, default=512)
    ap.add_argument("--first-max-new-tokens", type=int, default=512, help="turn-0 generation cap (agent mode)")
    ap.add_argument("--tool-output-max-chars", type=int, default=0, help="keep last N chars of tool output (0=off)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--data", default=DATA, help="jsonl with id/question/answer (default: gsm8k_128)")
    ap.add_argument("--prefill-group", type=int, default=4, help="max prompts per batched prefill (memory)")
    ap.add_argument("--max-prefill", type=int, default=1,
                    help="TP chunked prefill: max prompts in flight at once")
    ap.add_argument("--prefill-chunk", type=int, default=0,
                    help="TP only: split each prompt prefill into chunks of this many tokens and run one chunk "
                         "per engine step, interleaved with decode of the other slots (0 = whole prompt at once)")
    ap.add_argument("--fast-gdn", action="store_true", help="self-written chunk GDN for prefill (gdn_chunk_fast.py)")
    ap.add_argument("--gdn-v2", action="store_true", help="prefill GDN via gdn_chunk_fast2 (batched precompute)")
    ap.add_argument("--elem-conv", action="store_true", help="decode conv as mul+sum over K (cb_engine.elem_conv_update)")
    ap.add_argument("--graph-decode", action="store_true",
                    help="replay captured NPUGraph per 512-token bucket for decode (cb_engine._capture_graphs)")
    ap.add_argument("--truncate-to-fit", action="store_true",
                    help="if prompt + cap >= t_max, cap generation at t_max - prompt - 1 instead of rejecting the task")
    ap.add_argument("--min-fit-tokens", type=int, default=256,
                    help="with --truncate-to-fit: reject if the truncated cap would be below this")
    ap.add_argument("--direct", action="store_true",
                    help="no tools: single-turn reasoning answer in \\boxed{} (GPQA/BBH); tools not passed to template")
    ap.add_argument("--direct-max-new-tokens", type=int, default=2048,
                    help="per-task generation cap in --direct mode")
    ap.add_argument("--split-devices", default=None,
                    help="two cards, e.g. npu:0,npu:1: layer-wise split of weights and KV/GDN state "
                         "(pipeline, one token at a time). Frees memory per card; no speedup. Eager decode only.")
    ap.add_argument("--tp", type=int, default=1,
                    help="tensor parallel over N cards. Launch N processes with RANK=0..N-1 and "
                         "ASCEND_RT_VISIBLE_DEVICES=<rank>; only rank 0 writes output. Eager decode only.")
    ap.add_argument("--tp-port", default="29561")
    a = ap.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    if a.tp > 1:
        if a.split_devices:
            raise SystemExit("--tp is incompatible with --split-devices")
        a.device = "npu:0"            # each TP process sees exactly one card
    if a.split_devices:
        if a.graph_decode:
            raise SystemExit("--split-devices is incompatible with --graph-decode")
        d0, d1 = a.split_devices.split(",")
        a.device = d0

    busy = [] if a.tp > 1 else other_runs_on_device(a.device)   # TP ranks share a name by design
    if a.split_devices:
        busy = busy + other_runs_on_device(d1)
    if busy and not a.allow_busy:
        raise SystemExit(f"another cb_run.py is already on {a.device} (pids {busy}); "
                         "wait for it or pass --allow-busy")
    out = Path(a.out)
    if out.exists():
        raise SystemExit("output exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    system_prompt = DIRECT_SYSTEM if a.direct else {"original": SYSTEM, "concise": CONCISE_SYSTEM}[a.protocol]
    torch.set_num_threads(8)
    torch.manual_seed(42)
    t0 = time.perf_counter()
    if a.tp > 1:
        import cb_tp
        cb_tp.init_tp(rank, a.tp, a.tp_port)
        model = cb_tp.shard_model(
            Qwen3_5ForConditionalGeneration.from_pretrained(
                MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map=None).eval(),
            rank, a.tp).to(a.device)
    else:
        if a.split_devices:
            import cb_engine as _ce
            dmap = _ce.split_device_map(64, d0, d1)
        else:
            dmap = {"": a.device}
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map=dmap).eval()
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    patch_model(model, fast_gdn=a.fast_gdn, gdn_v2=a.gdn_v2)
    import cb_engine
    cb_engine.ELEM_CONV = a.elem_conv
    if a.tp > 1:
        import cb_tp
        eng = cb_tp.TPEngine(model, n_slots=a.slots, t_max=a.t_max, device=a.device, tp=a.tp,
                             graph_decode=a.graph_decode)
    else:
        eng = CBEngine(model, n_slots=a.slots, t_max=a.t_max, device=a.device, graph_decode=a.graph_decode)
    torch.npu.synchronize(a.device)
    load_s = time.perf_counter() - t0

    data = Path(a.data).read_bytes()
    tasks = [json.loads(l) for l in data.splitlines()]
    if a.limit is not None:
        tasks = tasks[:a.limit]
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])

    def emit(row):
        if rank != 0:                 # TP: non-zero ranks compute the same tokens but do not write
            return
        with out.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    pending = deque(Req(t, system_prompt) for t in tasks)
    slot_req = [None] * a.slots
    free = list(range(a.slots))
    n_done = 0
    n_steps = 0
    prefill_s = 0.0
    n_prefill = 0
    n_prefill_calls = 0
    # Per-event timing (diagnostics): one row per prefill call and per decode step.
    prefill_log = []   # [turn, prompt_tokens, seconds, active_decoding_before]
    decode_log = []    # [n_active, max_len_in_step, seconds]
    decode_s = 0.0
    pfs = []           # chunked prefills in progress: [req, slot, ids, tokens_done]

    def finish(r):
        """Called when r's generation for the current turn is complete. Returns True if task is done."""
        nonlocal n_done
        ids = []
        for v in r.gen:
            ids.append(v)
            if v in eos:
                break
        text = tok.decode(ids, skip_special_tokens=True)
        code = None if a.direct else parse_call(text)
        rec = dict(turn=r.turn, tokens=ids, text=text)
        r.messages.append({"role": "assistant", "content": text})
        final_text = text
        if code is not None:
            ts = time.perf_counter()
            result = clip_tool_output(execute(code), a.tool_output_max_chars)
            r.tool_count += 1
            rec.update(code=code, tool_output=result, tool_seconds=time.perf_counter() - ts)
            r.messages.append({"role": "tool", "name": "python", "content": result})
        r.rounds.append(rec)
        if code is not None and r.turn < 5:
            r.turn += 1
            r.gen = []
            pending.append(r)
            return False
        prediction = answer(text) if code is None else None
        t = r.task
        emit(dict(kind="task", id=t["id"], gold=t["answer"], final_text=final_text,
                  prediction=prediction, tool_count=r.tool_count,
                  correct=(r.tool_count > 0 or a.direct) and correct(prediction, t["answer"]),
                  completion_seconds=time.perf_counter() - start, rounds=r.rounds))
        n_done += 1
        return True

    def done_gen(r):
        return r.gen[-1] in eos or len(r.gen) >= r.cap

    emit(dict(kind="metadata", load_seconds=load_s, protocol=a.protocol, system_prompt=system_prompt,
              schedule="fifo-continuous", batch_size=a.slots, slots=a.slots, t_max=a.t_max,
              final_max_new_tokens=a.final_max_new_tokens, first_max_new_tokens=a.first_max_new_tokens, tool_output_max_chars=a.tool_output_max_chars, optimizations=True, engine="cb_engine",
              prefill_group=a.prefill_group, prefill_chunk=a.prefill_chunk, max_prefill=a.max_prefill, fast_gdn=a.fast_gdn, elem_conv=a.elem_conv,
              graph_decode=a.graph_decode, gdn_v2=a.gdn_v2,
              graph_bucket=getattr(cb_engine, "GRAPH_BUCKET", None),
              data_sha256=__import__("hashlib").sha256(data).hexdigest()))

    torch.npu.reset_peak_memory_stats(a.device)
    start = time.perf_counter()
    def fit_cap(prompt_len, cap):
        """Return the generation cap to use, or None to reject the request."""
        if prompt_len + cap < a.t_max:
            return cap
        if a.truncate_to_fit:
            room = a.t_max - prompt_len - 1
            if room >= a.min_fit_tokens:
                return room
        return None

    with torch.inference_mode():
        while pending or pfs or any(r is not None for r in slot_req):
            if a.prefill_chunk > 0:
                # chunked admission: up to --max-prefill prompts in flight; one chunk per engine step,
                # round-robin over the in-flight prompts; the other slots keep decoding
                while len(pfs) < a.max_prefill and pending and free:
                    r = pending.popleft()
                    s = free.pop()
                    prompt = tok.apply_chat_template(r.messages, tools=None if a.direct else TOOLS, tokenize=False,
                                                     add_generation_prompt=True, enable_thinking=False)
                    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(a.device)
                    r.cap = (a.direct_max_new_tokens if a.direct
                             else a.first_max_new_tokens if r.turn == 0 else a.final_max_new_tokens)
                    fc = fit_cap(ids.shape[1], r.cap)
                    if fc is None:
                        t = r.task
                        emit(dict(kind="task", id=t["id"], task=t.get("task"), gold=t["answer"], final_text="",
                                  prediction=None, rounds=r.rounds, tool_count=r.tool_count, correct=False,
                                  error=f"prompt {ids.shape[1]} + gen {r.cap} >= t_max {a.t_max}",
                                  completion_seconds=time.perf_counter() - start))
                        n_done += 1
                        free.append(s)
                    else:
                        r.cap = fc
                        pfs.append([r, s, ids, 0])   # request, slot, prompt ids, tokens prefilled so far
                if pfs:
                    e = pfs.pop(0)
                    r, s, ids, p = e
                    C = min(a.prefill_chunk, ids.shape[1] - p)
                    last = p + C == ids.shape[1]
                    n_active_before = sum(1 for x in slot_req if x is not None)
                    tp = time.perf_counter()
                    first = eng.prefill_chunk(s, ids[:, p:p + C], p, last)
                    torch.npu.synchronize(a.device)
                    dt = time.perf_counter() - tp
                    prefill_s += dt
                    n_prefill_calls += 1
                    prefill_log.append([r.turn, C, round(dt, 4), n_active_before])
                    if last:
                        n_prefill += 1
                        r.slot = s
                        r.gen = [first]
                        r.last = first
                        slot_req[s] = r
                        if done_gen(r):
                            slot_req[s] = None
                            eng.free(s)
                            free.append(s)
                            finish(r)
                    else:
                        e[3] = p + C
                        pfs.append(e)               # back of the queue (round-robin)
            # admission: all free slots take pending requests in ONE batched prefill
            elif free and pending:
                group = []
                while free and pending:
                    r = pending.popleft()
                    s = free.pop()
                    prompt = tok.apply_chat_template(r.messages, tools=None if a.direct else TOOLS, tokenize=False,
                                                     add_generation_prompt=True, enable_thinking=False)
                    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(a.device)
                    r.cap = (a.direct_max_new_tokens if a.direct
                             else a.first_max_new_tokens if r.turn == 0 else a.final_max_new_tokens)
                    fc = fit_cap(ids.shape[1], r.cap)
                    if fc is None:
                        # Record as failed (same as vllm_async_bench) instead of aborting the run.
                        t = r.task
                        emit(dict(kind="task", id=t["id"], task=t.get("task"), gold=t["answer"], final_text="",
                                  prediction=None, rounds=r.rounds, tool_count=r.tool_count, correct=False,
                                  error=f"prompt {ids.shape[1]} + gen {r.cap} >= t_max {a.t_max}",
                                  completion_seconds=time.perf_counter() - start))
                        n_done += 1
                        free.append(s)
                        continue
                    r.cap = fc
                    group.append((r, s, ids))
                firsts = []
                n_active_before = sum(1 for x in slot_req if x is not None)
                for g0 in range(0, len(group), a.prefill_group):
                    sub = group[g0:g0 + a.prefill_group]
                    tp = time.perf_counter()
                    firsts += eng.prefill_batch([s for _, s, _ in sub], [ids for _, _, ids in sub])
                    torch.npu.synchronize(a.device)
                    dt = time.perf_counter() - tp
                    prefill_s += dt
                    n_prefill_calls += 1
                    prefill_log.append([sub[0][0].turn, sum(ids.shape[1] for _, _, ids in sub),
                                        round(dt, 4), n_active_before])
                n_prefill += len(group)
                for (r, s, _), first in zip(group, firsts):
                    r.slot = s
                    r.gen = [first]
                    r.last = first
                    slot_req[s] = r
                    if done_gen(r):
                        slot_req[s] = None
                        eng.free(s)
                        free.append(s)
                        finish(r)
            active = [s for s in range(a.slots) if slot_req[s] is not None]
            if not active:
                continue
            max_len = max(eng.lens) + 1
            td = time.perf_counter()
            hold = tuple(e[1] for e in pfs)               # slots mid-prefill must keep their state
            if hold:
                nxt = eng.decode_step(active, [slot_req[s].last for s in active], hold=hold)
            else:
                nxt = eng.decode_step(active, [slot_req[s].last for s in active])
            dt_step = time.perf_counter() - td
            decode_s += dt_step
            decode_log.append([len(active), max_len, round(dt_step, 4)])
            n_steps += 1
            for s, t in zip(active, nxt):
                r = slot_req[s]
                r.gen.append(t)
                r.last = t
                if done_gen(r):
                    slot_req[s] = None
                    eng.free(s)
                    free.append(s)
                    finish(r)
            if n_steps % 200 == 0:
                print(f"step {n_steps} done {n_done}/{len(tasks)} active {len(active)} "
                      f"pending {len(pending)} t={time.perf_counter()-start:.1f}s", flush=True)
    torch.npu.synchronize(a.device)
    emit(dict(kind="summary", makespan_seconds=time.perf_counter() - start, decode_steps=n_steps,
              prefill_count=n_prefill, prefill_calls=n_prefill_calls, prefill_seconds=prefill_s,
              decode_seconds=decode_s, prefill_log=prefill_log, decode_log=decode_log,
              peak_bytes=int(torch.npu.max_memory_allocated(a.device))))


if __name__ == "__main__":
    main()
