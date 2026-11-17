"""Slot-based continuous-batching decode for Gemma-4-26B-A4B-it (HF weights, self-written attention).

Design (see experiments/dependency_graph.md, Gemma section):
  - Every HF submodule except attention is reused as-is (norms, MLP, router, experts,
    layer_scalar, embedding, final norm, lm_head), so the numerics of those parts are HF's.
  - Attention is rewritten to read/write per-slot KV buffers [N, kv_heads, T, head_dim].
    Sliding layers use the same buffer length T with a window mask (window 1024), so no ring
    buffer is needed as long as T <= the sliding window or the mask is applied (it is).
  - Prefill of a turn runs through HF (batch 1, DynamicCache); its KV is copied into the slot.
  - decode_step runs all N slots densely; idle slots are dummy rows and their outputs ignored.
  - The protocol is the same as gemma_run.py (re-prefill the whole conversation every turn,
    greedy, 512 tokens per turn, up to 6 turns, same tool-call parsing and stop tokens).

Limitations (checked, not silently ignored):
  - prompt length must be <= SLIDING_WINDOW for the prefill copy (HF's sliding cache keeps only
    the last window); asserted.
  - prompt + 512 must fit in t_max; asserted.
"""
import time

import torch
import torch_npu  # noqa: F401
from transformers import DynamicCache
from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb


def _rope_fused(x, cos, sin, unsqueeze_dim=2):
    # torch_npu.npu_rotary_mul uses the rotate_half convention (checked in scripts/check_gemma_rope_fused2.py).
    # cos/sin come in as [N,1,hd] from the rotary module; unsqueeze to [N,1,1,hd] as unsqueeze_dim=2 does.
    return torch_npu.npu_rotary_mul(x, cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim))

MAX_NEW = 512
MAX_TURNS = 6


def experts_grouped(ex, x, idx, w, geglu=False):
    """Grouped-matmul MoE experts (replaces HF's per-expert Python loop).
    ex: HF Gemma4TextExperts; x [T,H]; idx [T,K] expert ids; w [T,K] routing weights.
    All (token, expert) assignments are sorted by expert, run through two
    npu_grouped_matmul calls, scaled by w, and scatter-added back to tokens.
    No host sync (no nonzero/item). Summation order differs from HF, so outputs differ slightly."""
    import torch_npu
    E = ex.num_experts
    T, K = idx.shape
    flat = idx.reshape(-1)
    order = torch.argsort(flat, stable=True)
    tok = order // K
    xs = x[tok]                                                    # [T*K, H]
    gl = torch.cumsum(torch.bincount(flat, minlength=E), 0)        # cumsum group_list (device)
    gu = torch_npu.npu_grouped_matmul([xs], [ex.gate_up_proj.transpose(1, 2)],
                                      group_list=gl, split_item=2, group_type=0)[0]
    if geglu:
        # npu_geglu: gelu(first half) * second half, tanh approximation (= gelu_pytorch_tanh)
        h = torch_npu.npu_geglu(gu, dim=-1, approximate=1)[0]
    else:
        g, u = gu.chunk(2, dim=-1)
        h = ex.act_fn(g) * u
    y = torch_npu.npu_grouped_matmul([h], [ex.down_proj.transpose(1, 2)],
                                     group_list=gl, split_item=2, group_type=0)[0]
    y = y * w.reshape(-1)[order][:, None].to(y.dtype)
    return torch.zeros_like(x).index_add_(0, tok, y.to(x.dtype))


def router_fused(r, x):
    """Same math as Gemma4TextRouter.forward, with the constant scale folded into r._fold.
    Only the rounding of (h*scale)*root vs h*(scale*root) differs."""
    h = r.norm(x) * r._fold
    probs = torch.softmax(r.proj(h), dim=-1)
    w, idx = torch.topk(probs, k=r.config.top_k_experts, dim=-1)
    w = w / w.sum(dim=-1, keepdim=True)
    w = w * r.per_expert_scale[idx]
    return probs, w, idx


class GemmaCB:
    def __init__(self, model, tok, n_slots, t_max, device, grouped=False, nocopy=False,
                 slice_kv=False, fast_router=False, fused_rope=False, fia=False, geglu=False,
                 prefill_grouped=False):
        self.grouped = grouped
        self.prefill_grouped = prefill_grouped
        self.prefill_s, self.prefill_n = 0.0, 0
        if prefill_grouped:
            # Prefill-only: the HF batch-1 prefill uses grouped experts too. Decode calls
            # experts_grouped directly when grouped=True, so replacing ex.forward affects prefill only.
            assert grouped, "--prefill-grouped requires --grouped-experts"
            for layer in model.model.language_model.layers:
                if getattr(layer, "enable_moe_block", False):
                    ex = layer.experts
                    ex.forward = (lambda ex: (lambda hs, ti, tw: experts_grouped(ex, hs, ti, tw)))(ex)
        self.fia = fia
        self.geglu = geglu
        if geglu:
            # npu_geglu activates the SECOND half (gelu(second) * first); HF uses gelu(first) * second.
            # Swap the two halves of each expert's gate_up_proj rows once (exact row permutation).
            # Construction-time only: after this, experts_grouped(geglu=False) would be wrong.
            for layer in model.model.language_model.layers:
                if getattr(layer, "enable_moe_block", False):
                    ex = layer.experts
                    W = ex.gate_up_proj.data
                    half = W.shape[1] // 2
                    ex.gate_up_proj.data = torch.cat([W[:, half:], W[:, :half]], dim=1).contiguous()
                    del W
        self.fused_rope = fused_rope
        self.nocopy = nocopy
        self.fast_router = fast_router
        if fast_router:
            for layer in model.model.language_model.layers:
                if getattr(layer, "enable_moe_block", False):
                    r = layer.router
                    # fold the two constant multiplies (scale, root size) into one vector
                    r._fold = (r.scale * r.scalar_root_size).detach()
        self.slice_kv = slice_kv
        self.model = model
        self.tok = tok
        self.cfg = model.config.get_text_config()
        self.lm = model.model.language_model
        self.N, self.T, self.dev = n_slots, t_max, device
        self.window = self.cfg.sliding_window
        self.nh = self.cfg.num_attention_heads
        self.layer_types = self.cfg.layer_types
        self.K, self.V = [], []
        for li, layer in enumerate(self.lm.layers):
            attn = layer.self_attn
            kvh = self.nh // attn.num_key_value_groups
            self.K.append(torch.zeros(n_slots, kvh, t_max, attn.head_dim,
                                      dtype=torch.bfloat16, device=device))
            self.V.append(torch.zeros(n_slots, kvh, t_max, attn.head_dim,
                                      dtype=torch.bfloat16, device=device))
        self.ar = torch.arange(n_slots, device=device)
        self.neg = torch.finfo(torch.bfloat16).min

    # ---------- prefill (HF) and copy into slot ----------
    def prefill_into_slot(self, slot, ids):
        L = ids.shape[1]
        assert L <= self.window, f"prompt {L} > sliding window {self.window}"
        assert L + MAX_NEW <= self.T, f"prompt {L} + {MAX_NEW} exceeds t_max {self.T}"
        cache = DynamicCache(config=self.model.config)
        torch.npu.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = self.model(input_ids=ids, past_key_values=cache, use_cache=True)
        for li in range(len(self.lm.layers)):
            lay = cache.layers[li]
            self.K[li][slot, :, :L] = lay.keys[0, :, :L]
            self.V[li][slot, :, :L] = lay.values[0, :, :L]
        torch.npu.synchronize()
        self.prefill_s += time.time() - t0
        self.prefill_n += 1
        return out.logits[0, -1]

    # ---------- decode ----------
    def _masks(self, pos, Tact):
        j = torch.arange(Tact, device=self.dev)[None, :]          # [1, Tact]
        p = pos[:, None]                                          # [N, 1]
        full = (j <= p)
        slide = full & (j > p - self.window)
        to_bias = lambda ok: torch.where(ok, 0.0, self.neg).to(torch.bfloat16)[:, None, None, :]
        return {"full_attention": to_bias(full), "sliding_attention": to_bias(slide)}

    def _attn(self, li, layer, h, pos, cos, sin, mask, sl):
        attn = layer.self_attn
        N, hd = h.shape[0], attn.head_dim
        kvh = self.nh // attn.num_key_value_groups
        q = attn.q_norm(attn.q_proj(h).view(N, 1, self.nh, hd))
        rope = _rope_fused if self.fused_rope else apply_rotary_pos_emb
        q = rope(q, cos, sin, unsqueeze_dim=2).transpose(1, 2)   # [N,nh,1,hd]
        kr = attn.k_proj(h).view(N, 1, kvh, hd)
        vr = kr if attn.use_alternative_attention else attn.v_proj(h).view(N, 1, kvh, hd)
        k = rope(attn.k_norm(kr), cos, sin, unsqueeze_dim=2)     # [N,1,kvh,hd]
        v = attn.v_norm(vr)
        self.K[li][sl, :, pos] = k[:, 0]
        self.V[li][sl, :, pos] = v[:, 0]
        Tact = mask.shape[-1]
        if self.slice_kv and sl is self.ar:
            # all slots active and in order: read a view instead of gathering a copy
            Kc = self.K[li][:, :, :Tact]
            Vc = self.V[li][:, :, :Tact]
        else:
            Kc = self.K[li][sl, :, :Tact]
            Vc = self.V[li][sl, :, :Tact]
        g = attn.num_key_value_groups
        if self.fia:
            # one fused kernel: QK^T * scale + mask, softmax, PV; GQA via num_key_value_heads.
            # additive bf16 mask -> boolean (True = masked out)
            o = torch_npu.npu_fused_infer_attention_score(
                q.contiguous(), Kc, Vc, atten_mask=(mask < 0), num_heads=self.nh,
                num_key_value_heads=kvh, input_layout="BNSD", scale=attn.scaling)[0]
            return attn.o_proj(o.transpose(1, 2).reshape(N, 1, self.nh * hd))
        if self.nocopy:
            # GQA without materializing repeated K/V: query heads h = kv*g + j share kv head kv.
            qg = q.reshape(N, kvh, g, hd)
            w = torch.matmul(qg, Kc.transpose(2, 3)) * attn.scaling + mask        # [N,kvh,g,Tact]
            w = torch.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
            o = torch.matmul(w, Vc).reshape(N, 1, self.nh * hd)
            return attn.o_proj(o)
        Kr = Kc[:, :, None].expand(N, kvh, g, Tact, hd).reshape(N, self.nh, Tact, hd)
        Vr = Vc[:, :, None].expand(N, kvh, g, Tact, hd).reshape(N, self.nh, Tact, hd)
        w = torch.matmul(q, Kr.transpose(2, 3)) * attn.scaling + mask            # HF eager order
        w = torch.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
        o = torch.matmul(w, Vr).transpose(1, 2).reshape(N, 1, self.nh * hd)
        return attn.o_proj(o)

    def _layer(self, li, layer, x, pos, cos, sin, mask, sl):
        res = x
        h = self._attn(li, layer, layer.input_layernorm(x), pos, cos, sin, mask, sl)
        x = res + layer.post_attention_layernorm(h)
        res = x
        h = layer.mlp(layer.pre_feedforward_layernorm(x))
        if layer.enable_moe_block:
            h1 = layer.post_feedforward_layernorm_1(h)
            flat = res.reshape(-1, res.shape[-1])
            if self.fast_router:
                _, w, idx = router_fused(layer.router, flat)
            else:
                _, w, idx = layer.router(flat)
            xin = layer.pre_feedforward_layernorm_2(flat)
            h2 = experts_grouped(layer.experts, xin, idx, w, geglu=self.geglu) if self.grouped \
                else layer.experts(xin, idx, w)
            h2 = layer.post_feedforward_layernorm_2(h2.reshape(res.shape))
            h = h1 + h2
        x = res + layer.post_feedforward_layernorm(h)
        return x * layer.layer_scalar

    @torch.no_grad()
    def decode_step(self, tokens, pos, sl=None, tact=None):
        """tokens, pos: int64 tensors [n] on device; sl: slot indices [n] (default: all N slots).
        tact: optional host-computed attention length (must equal the value derived from pos).
        Passing it avoids the device->host sync of pos.max().item().
        Returns logits [n, V] (softcapped). With sl given, only those slots are computed."""
        n = tokens.shape[0]
        if sl is None:
            sl = self.ar
        x = self.lm.embed_tokens(tokens.view(n, 1))
        pid = pos.view(n, 1)
        if tact is None:
            Tact = int(pos.max().item()) + 1
            Tact = min(self.T, ((Tact + 63) // 64) * 64)
        else:
            Tact = tact
        masks = self._masks(pos, Tact)
        rope = {lt: self.lm.rotary_emb(x, pid, lt) for lt in set(self.layer_types)}
        for li, layer in enumerate(self.lm.layers):
            lt = self.layer_types[li]
            cos, sin = rope[lt]
            x = self._layer(li, layer, x, pos, cos, sin, masks[lt], sl)
        x = self.lm.norm(x)
        logits = self.model.lm_head(x[:, 0])
        cap = self.cfg.final_logit_softcapping
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        return logits


def run_cb(model, tok, qs, eos_ids, n_slots, t_max, device, log=print, compact=False,
           margin_log=None, grouped=False, slice_kv=False, fast_router=False, host_tact=False,
           fused_rope=False, fia=False, geglu=False, nocopy=False, prefill_grouped=False):
    """compact=True: decode only the active slots each step (idle slots are not computed).
    margin_log: optional list; if given, appends (task_id, gen_index, token, top2_gap) for every
    generated token (diagnostic only, does not change decoding)."""
    """Continuous batching over tasks qs. Returns per-task rows in gemma_run.py's format."""
    from gemma_run import parse_gemma_call, strip_specials  # lazy: avoid import cycle
    import agent_protocol as P

    eng = GemmaCB(model, tok, n_slots, t_max, device, grouped=grouped, slice_kv=slice_kv, fused_rope=fused_rope,
                  fast_router=fast_router, fia=fia, geglu=geglu, nocopy=nocopy,
                  prefill_grouped=prefill_grouped)
    eos = set(eos_ids)
    queue = list(qs)
    slots = [None] * n_slots
    done = []
    t_all = time.time()

    def start_turn(s, task):
        p = tok.apply_chat_template(task["msgs"], tools=P.TOOLS, add_generation_prompt=True,
                                    tokenize=False, enable_thinking=False)
        ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        logits = eng.prefill_into_slot(s, ids)
        first = int(logits.argmax().item())
        slots[s] = {"task": task, "gen": [first], "len": int(ids.shape[1]), "next": first,
                    "prompt_tokens": int(ids.shape[1])}
        if first in eos or len(slots[s]["gen"]) >= MAX_NEW:
            finish_turn(s)

    def finish_turn(s):
        st = slots[s]
        task = st["task"]
        raw = tok.decode(st["gen"], skip_special_tokens=False)
        if "<|tool_call>" in raw and "<tool_call|>" not in raw:
            raw = raw + "<tool_call|>"
        text = strip_specials(raw)
        task["final_text"] = text
        rec = {"turn": len(task["turns"]), "prompt_tokens": st["prompt_tokens"],
               "gen_tokens": len(st["gen"])}
        task["n_tokens"] += len(st["gen"])
        code = parse_gemma_call(raw)
        if code is None:
            task["turns"].append(rec)
            return finish_task(s)
        out_text = P.execute(code)
        rec["tool_output"] = out_text
        task["turns"].append(rec)
        if len(task["turns"]) >= MAX_TURNS:
            return finish_task(s)
        task["msgs"].append({"role": "assistant", "content": "",
                             "tool_calls": [{"type": "function",
                                             "function": {"name": "python",
                                                          "arguments": {"code": code}}}]})
        task["msgs"].append({"role": "tool", "name": "python", "content": out_text})
        start_turn(s, task)

    def finish_task(s):
        task = slots[s]["task"]
        slots[s] = None
        pred = P.answer(task["final_text"])
        ok = P.correct(pred, task["q"]["answer"])
        done.append({"id": task["q"]["id"], "gold": task["q"]["answer"], "pred": pred,
                     "correct": ok, "turns": task["turns"], "n_tokens": task["n_tokens"],
                     "final_text": task["final_text"]})
        log(f"[{len(done)}/{len(qs)}] id={task['q']['id']} correct={ok} "
            f"turns={len(task['turns'])} tok={task['n_tokens']} elapsed={time.time() - t_all:.0f}s")
        refill()

    def new_task(q):
        return {"q": q, "msgs": [{"role": "system", "content": P.SYSTEM},
                                 {"role": "user", "content": q["question"]}],
                "turns": [], "n_tokens": 0, "final_text": ""}

    def refill():
        for s in range(n_slots):
            if slots[s] is None and queue and slots[s] is None:
                start_turn(s, new_task(queue.pop(0)))

    refill()
    while any(st is not None for st in slots):
        active = [s for s in range(n_slots) if slots[s] is not None]
        tokens = [slots[s]["next"] if slots[s] is not None else 0 for s in range(n_slots)]
        pos = [slots[s]["len"] if slots[s] is not None else 0 for s in range(n_slots)]
        if compact:
            full = len(active) == n_slots  # all rows active, in slot order
            sl_t = eng.ar if full else torch.tensor(active, device=device)
            tact = None
            if host_tact:
                # same value decode_step would derive from pos.max(), computed on the host (no sync)
                tact = min(eng.T, ((max(slots[s]["len"] for s in active) + 1 + 63) // 64) * 64)
            logits = eng.decode_step(torch.tensor([slots[s]["next"] for s in active], device=device),
                                     torch.tensor([slots[s]["len"] for s in active], device=device),
                                     sl_t, tact=tact)
            nxt_by_slot = dict(zip(active, logits.argmax(-1).tolist()))
        else:
            logits = eng.decode_step(torch.tensor(tokens, device=device),
                                     torch.tensor(pos, device=device))
            nxt = logits.argmax(-1).tolist()
            nxt_by_slot = {s: nxt[s] for s in active}
        if margin_log is not None:
            top2 = torch.topk(logits.float(), 2, dim=-1).values
            gaps = (top2[:, 0] - top2[:, 1]).tolist()
            for i, s in enumerate(active):
                r = i if compact else s
                margin_log.append((slots[s]["task"]["q"]["id"], len(slots[s]["task"]["turns"]),
                                   len(slots[s]["gen"]), nxt_by_slot[s], gaps[r]))
        for s in active:
            st = slots[s]
            t = nxt_by_slot[s]
            st["gen"].append(t)
            st["len"] += 1
            st["next"] = t
            if t in eos or len(st["gen"]) >= MAX_NEW:
                finish_turn(s)
    log(f"prefill_n={eng.prefill_n} prefill_s={eng.prefill_s:.1f} prefill_grouped={prefill_grouped}")
    return done, time.time() - t_all
