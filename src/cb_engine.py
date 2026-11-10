"""Slot-based continuous-batching decode engine for Qwen3.5 (HF weights, self-written decode loop).

Why not HF generate / HF continuous batching: generate() runs a static batch until the longest row
finishes (finished rows keep computing: ~53% of row-steps wasted in the batch16 run), and HF's paged
cache does not manage the GatedDeltaNet recurrent/conv states.

Design:
  - N slots. Full-attention layers keep a per-slot KV buffer [N, kvh, T, hd]; GDN layers keep per-slot
    conv state [N, C, K] and recurrent state [N, Hv, dk, dv] (fp32, as produced by the single-step rule).
  - prefill(slot, ids): batch-1 HF forward with a DynamicCache, then the cache is copied into the slot.
  - decode_step(slots, tokens): one token for the given active slots only. Gathers the active rows,
    attends over [0, max_len_active] with a per-row boolean mask, scatters the new KV/state back.
  - Module weights, norms, MLP, rotary and lm_head are HF's own modules (same math as the HF forward).
"""
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from transformers import DynamicCache
from transformers.models.qwen3_5 import modeling_qwen3_5 as M

from recurrent_single_step import recurrent_single_step

# Decode causal depthwise conv: False = HF torch conv1d fallback (baseline); True = mul + sum over the
# K-window (same math, bf16 rounding order differs). Set by cb_run.py --elem-conv.
ELEM_CONV = False

# Graph decode (cb_run.py --graph-decode): one captured NPUGraph per Lmax bucket of this many tokens.
GRAPH_BUCKET = int(__import__("os").environ.get("CB_GRAPH_BUCKET", "512"))


def split_device_map(n_layers, d0, d1):
    """HF device_map for a two-card layer split: embeddings and first half of the layers on d0,
    second half, final norm and lm_head on d1. Keys follow Qwen3_5ForConditionalGeneration."""
    m = {"model.visual": d0, "model.language_model.embed_tokens": d0,
         "model.language_model.rotary_emb": d0,
         "model.language_model.norm": d1, "lm_head": d1}
    half = n_layers // 2
    for i in range(n_layers):
        m[f"model.language_model.layers.{i}"] = d0 if i < half else d1
    return m


def elem_conv_update(x, state, weight, bias):
    """x [n,C,1], state [n,C,K] (updated in place), weight [C,K], bias [C]. Returns silu(conv) [n,C,1]."""
    hs = torch.cat([state, x], dim=-1).to(weight.dtype)           # [n,C,K+1]
    state.copy_(hs[:, :, -state.shape[-1]:])
    win = hs[:, :, -weight.shape[-1]:]                             # [n,C,K]: window ending at new token
    out = (win * weight).sum(-1, keepdim=True)
    if bias is not None:
        out = out + bias.view(1, -1, 1)
    return F.silu(out).to(x.dtype)


def patch_model(model, fast_gdn=False, gdn_v2=False):
    """Same fused/single-step patches as e100_batch.py (default options).
    fast_gdn: chunked GDN prefill via src/gdn_chunk_fast.py instead of the torch fallback.
    gdn_v2: use src/gdn_chunk_fast2.py (batched precompute, same math; implies fast_gdn)."""
    if gdn_v2:
        from gdn_chunk_fast2 import fast_chunk_gated_delta_rule_v2 as fast_chunk_gated_delta_rule
    elif fast_gdn:
        from gdn_chunk_fast import fast_chunk_gated_delta_rule
    def fused_rmsnorm(self, x):
        return torch_npu.npu_rms_norm(x, (1.0 + self.weight).to(x.dtype), self.eps)[0]

    def fused_gated_rmsnorm(self, hidden_states, gate=None):
        normalized = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
        return torch_npu.npu_swiglu(torch.cat((gate, normalized), dim=-1))

    for m in model.modules():
        if isinstance(m, M.Qwen3_5RMSNorm):
            m.forward = fused_rmsnorm.__get__(m, M.Qwen3_5RMSNorm)
        elif isinstance(m, M.Qwen3_5RMSNormGated):
            m.forward = fused_gated_rmsnorm.__get__(m, M.Qwen3_5RMSNormGated)
        elif isinstance(m, M.Qwen3_5GatedDeltaNet):
            m.recurrent_gated_delta_rule = recurrent_single_step
            if fast_gdn:
                m.chunk_gated_delta_rule = fast_chunk_gated_delta_rule


class CBEngine:
    def __init__(self, model, n_slots, t_max, device, graph_decode=False, tp=1):
        self.model = model
        self.lm = model.model.language_model
        cfg = model.config.text_config
        self.cfg = cfg
        self.dev = device
        self.N = n_slots
        self.T = t_max
        self.tp = tp                       # tensor-parallel size; heads are local per rank (cb_tp.py)
        self.kvh = cfg.num_key_value_heads // tp
        self.hd = cfg.head_dim
        self.nh = cfg.num_attention_heads // tp
        self.g = self.nh // self.kvh
        self.K = cfg.linear_conv_kernel_dim
        self.lens = [0] * n_slots          # host-side current length per slot
        self.kv_k, self.kv_v, self.conv, self.rec = {}, {}, {}, {}
        # Per-layer device = device holding that layer's weights. Single card: all equal `device`.
        # Two-card split (load_split_map): first half on one card, second half on the other.
        self.ldev = [self.lm.layers[i].input_layernorm.weight.device for i in range(len(self.lm.layers))]
        self.split = len(set(self.ldev)) > 1
        self.ddev = self.lm.norm.weight.device        # final norm / lm_head device
        bf = torch.bfloat16
        for i, t in enumerate(cfg.layer_types):
            d = self.ldev[i]
            if t == "full_attention":
                self.kv_k[i] = torch.zeros(n_slots, self.kvh, t_max, self.hd, dtype=bf, device=d)
                self.kv_v[i] = torch.zeros(n_slots, self.kvh, t_max, self.hd, dtype=bf, device=d)
            else:
                gd = self.lm.layers[i].linear_attn
                self.conv[i] = torch.zeros(n_slots, gd.conv_dim, self.K, dtype=bf, device=d)
                self.rec[i] = torch.zeros(n_slots, gd.num_v_heads, gd.head_k_dim, gd.head_v_dim,
                                          dtype=torch.float32, device=d)
        self.rec_dtype = torch.float32
        self._all_idx = torch.arange(n_slots, dtype=torch.long, device=device)
        self._head_off = {}
        self.graph_decode = graph_decode
        if graph_decode and self.split:
            raise ValueError("--graph-decode is not supported with --split-devices")
        if graph_decode:
            self._capture_graphs()

    # ------------------------------------------------------------------ graph decode
    def _capture_graphs(self):
        """Capture one NPUGraph per Lmax bucket (GRAPH_BUCKET tokens) with a shared memory pool.
        Must run before any prefill: the slots are empty, and all state written during warmup and
        capture is zeroed afterwards. Verified in scripts/test_acl_graph.py (token-identical to eager)."""
        dev = self.dev
        self._tok_buf = torch.zeros(self.N, dtype=torch.long, device=dev)
        self._pos_buf = torch.zeros(self.N, dtype=torch.long, device=dev)
        self._graphs, self._graph_out = {}, {}
        pool = torch.npu.graph_pool_handle()
        with torch.inference_mode():
            self._graph_core(GRAPH_BUCKET)          # eager warmup: creates lazy helpers outside capture
            torch.npu.synchronize(dev)
            for b in range(GRAPH_BUCKET, self.T + 1, GRAPH_BUCKET):
                g = torch.npu.NPUGraph()
                with torch.npu.graph(g, pool=pool):
                    o = self._graph_core(b)
                self._graphs[b], self._graph_out[b] = g, o
            torch.npu.synchronize(dev)
            self._reset_state()

    def _reset_state(self):
        for i in self.kv_k:
            self.kv_k[i].zero_()
            self.kv_v[i].zero_()
        for i in self.conv:
            self.conv[i].zero_()
        for i in self.rec:
            self.rec[i].zero_()
        self.lens = [0] * self.N

    def _graph_core(self, Lmax):
        """Decode math identical to decode_step, but inputs come from static buffers (no host-created
        tensors), so it can be captured. Returns argmax tokens [N] on device."""
        dev, N = self.dev, self.N
        pos, idx = self._pos_buf, self._all_idx
        mask = (torch.arange(Lmax, device=dev)[None, :] <= pos[:, None])[:, None, None, :]
        x = self.lm.embed_tokens(self._tok_buf[:, None])
        pos3 = pos.view(1, N, 1).expand(3, N, 1)
        cos, sin = self.lm.rotary_emb(x, pos3)
        for i, layer in enumerate(self.lm.layers):
            h = layer.input_layernorm(x)
            if self.cfg.layer_types[i] == "full_attention":
                h = self._attn(layer.self_attn, i, h, cos, sin, idx, pos, Lmax, mask)
            else:
                h = self._gdn(layer.linear_attn, i, h, idx)
            x = x + h
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        x = self.lm.norm(x)
        logits = self.model.lm_head(x[:, 0])
        return logits.argmax(-1)

    def _decode_step_graph(self, slots, tokens):
        tok_full = [0] * self.N
        for s, t in zip(slots, tokens):
            tok_full[s] = t
        L = max(self.lens) + 1
        assert L <= self.T, f"decode length {L} > T {self.T}"
        b = min(self.T, -(-L // GRAPH_BUCKET) * GRAPH_BUCKET)
        self._pos_buf.copy_(torch.tensor(self.lens, dtype=torch.long))
        self._tok_buf.copy_(torch.tensor(tok_full, dtype=torch.long))
        self._graphs[b].replay()
        nxt = self._graph_out[b].tolist()
        for s in slots:
            self.lens[s] += 1
        return [nxt[s] for s in slots]

    # ------------------------------------------------------------------ prefill
    @torch.inference_mode()
    def prefill(self, slot, ids):
        """ids: [1, L] on device. Returns the first generated token (greedy)."""
        L = ids.shape[1]
        assert L < self.T, f"prompt {L} >= T {self.T}"
        cache = DynamicCache(config=self.cfg)
        out = self.model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        first = int(out.logits[0, -1].argmax(-1).item())
        for i, t in enumerate(self.cfg.layer_types):
            c = cache.layers[i]
            if t == "full_attention":
                self.kv_k[i][slot, :, :L] = c.keys[0]
                self.kv_v[i][slot, :, :L] = c.values[0]
            else:
                self.conv[i][slot] = c.conv_states[0].to(self.conv[i].dtype)
                self.rec[i][slot] = c.recurrent_states[0].to(self.rec[i].dtype)
        self.lens[slot] = L
        return first

    @torch.inference_mode()
    def prefill_batch(self, slots, id_list):
        """Left-padded batched prefill of several requests in one forward (GDN torch fallback scales
        sub-linearly with batch). Exact: GDN state stays zero through left padding (k=v=0 there),
        conv zero-padding matches, RoPE uses per-row positions. Returns first token per row."""
        B = len(id_list)
        lens = [x.shape[1] for x in id_list]
        Lp = max(lens)
        assert Lp < self.T
        dev = self.dev
        ids = torch.zeros(B, Lp, dtype=torch.long, device=dev)
        mask = torch.zeros(B, Lp, dtype=torch.long, device=dev)
        for b, x in enumerate(id_list):
            ids[b, Lp - lens[b]:] = x[0]
            mask[b, Lp - lens[b]:] = 1
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        cache = DynamicCache(config=self.cfg)
        out = self.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                         past_key_values=cache, use_cache=True, logits_to_keep=1)
        firsts = out.logits[:, -1].argmax(-1).tolist()
        for i, t in enumerate(self.cfg.layer_types):
            c = cache.layers[i]
            for b, slot in enumerate(slots):
                L = lens[b]
                p = Lp - L
                if t == "full_attention":
                    self.kv_k[i][slot, :, :L] = c.keys[b, :, p:]
                    self.kv_v[i][slot, :, :L] = c.values[b, :, p:]
                else:
                    self.conv[i][slot] = c.conv_states[b].to(self.conv[i].dtype)
                    self.rec[i][slot] = c.recurrent_states[b].to(self.rec[i].dtype)
        for b, slot in enumerate(slots):
            self.lens[slot] = lens[b]
        return firsts

    # ------------------------------------------------------------------ decode
    @torch.inference_mode()
    def decode_step(self, slots, tokens):
        """Decodes ALL N slots in one dense pass (no gather/scatter: those were the NPU bottleneck).
        Idle slots compute on dummy input and their results are ignored; their lengths do not advance,
        and a slot's state is fully rewritten by prefill when it is reused.
        slots: active slot ids; tokens: last token per active slot. Returns next token per active slot."""
        if self.graph_decode:
            return self._decode_step_graph(slots, tokens)
        N = self.N
        tok_full = [0] * N
        for s, t in zip(slots, tokens):
            tok_full[s] = t
        Lmax = max(self.lens) + 1
        pos = torch.tensor(self.lens, dtype=torch.long, device=self.dev)
        tok = torch.tensor(tok_full, dtype=torch.long, device=self.dev)
        mask = (torch.arange(Lmax, device=self.dev)[None, :] <= pos[:, None])[:, None, None, :]

        x = self.lm.embed_tokens(tok[:, None])                       # [N,1,H]
        pos3 = pos.view(1, N, 1).expand(3, N, 1)
        cos, sin = self.lm.rotary_emb(x, pos3)
        for i, layer in enumerate(self.lm.layers):
            d = self.ldev[i]
            # Card boundary: move the row state to this layer's card (no-op when already there).
            x, cos, sin, mask = self._to(x, d), self._to(cos, d), self._to(sin, d), self._to(mask, d)
            pos, idx = self._to(pos, d), self._to(self._all_idx, d)
            h = layer.input_layernorm(x)
            if self.cfg.layer_types[i] == "full_attention":
                h = self._attn(layer.self_attn, i, h, cos, sin, idx, pos, Lmax, mask)
            else:
                h = self._gdn(layer.linear_attn, i, h, idx)
            x = x + h
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        x = self._to(x, self.ddev)
        x = self.lm.norm(x)
        logits = self.model.lm_head(x[:, 0])
        nxt = logits.argmax(-1).tolist()
        for s in slots:
            self.lens[s] += 1
        return [nxt[s] for s in slots]

    @staticmethod
    def _to(t, d):
        return t if t.device == d else t.to(d)

    def _kv_rows(self, idx, pos):
        # flat row index into kv.view(N*kvh*T, hd) for (slot, kv_head, position) of each active row
        d = idx.device
        if d not in self._head_off:
            self._head_off[d] = (torch.arange(self.kvh, device=d) * self.T)[None, :]
        return (idx[:, None] * (self.kvh * self.T) + self._head_off[d] + pos[:, None]).reshape(-1)

    def _attn(self, attn, i, h, cos, sin, idx, pos, Lmax, mask):
        n = h.shape[0]
        qg = attn.q_proj(h).view(n, 1, -1, self.hd * 2)
        q, gate = torch.chunk(qg, 2, dim=-1)
        gate = gate.reshape(n, 1, -1)
        q = attn.q_norm(q).transpose(1, 2)                           # [n,nh,1,hd]
        k = attn.k_norm(attn.k_proj(h).view(n, 1, self.kvh, self.hd)).transpose(1, 2)
        v = attn.v_proj(h).view(n, 1, self.kvh, self.hd).transpose(1, 2)
        q, k = M.apply_rotary_pos_emb(q, k, cos, sin)

        # NPU: advanced indexing is ~5-20x slower than index_copy_/index_select; use flat row indices.
        rows = self._kv_rows(idx, pos)
        self.kv_k[i].view(-1, self.hd).index_copy_(0, rows, k[:, :, 0, :].reshape(-1, self.hd))
        self.kv_v[i].view(-1, self.hd).index_copy_(0, rows, v[:, :, 0, :].reshape(-1, self.hd))
        Kg = self.kv_k[i][:, :, :Lmax]                               # [n,kvh,Lmax,hd] (view)
        Vg = self.kv_v[i][:, :, :Lmax]
        qk = q.reshape(n, self.kvh, self.g, self.hd)                 # kv-major head order (as repeat_kv)
        o = F.scaled_dot_product_attention(qk, Kg, Vg, attn_mask=mask.expand(n, 1, self.g, Lmax),
                                           scale=attn.scaling)
        o = o.reshape(n, 1, self.nh * self.hd) * torch.sigmoid(gate)
        return attn.o_proj(o)

    def _gdn(self, gd, i, h, idx):
        n = h.shape[0]
        mixed = gd.in_proj_qkv(h).transpose(1, 2)                    # [n,C,1]
        z = gd.in_proj_z(h).reshape(n, 1, -1, gd.head_v_dim)
        b = gd.in_proj_b(h)
        a = gd.in_proj_a(h)

        cs = self.conv[i]                                            # updated in place
        if ELEM_CONV:
            assert gd.activation == "silu"
            mixed = elem_conv_update(mixed, cs, gd.conv1d.weight.squeeze(1), gd.conv1d.bias)
        else:
            mixed = gd.causal_conv1d_update(mixed, cs, gd.conv1d.weight.squeeze(1), gd.conv1d.bias, gd.activation)
        mixed = mixed.transpose(1, 2)
        query, key, value = torch.split(mixed, [gd.key_dim, gd.key_dim, gd.value_dim], dim=-1)
        query = query.reshape(n, 1, -1, gd.head_k_dim)
        key = key.reshape(n, 1, -1, gd.head_k_dim)
        value = value.reshape(n, 1, -1, gd.head_v_dim)

        beta = b.sigmoid()
        g = -gd.A_log.float().exp() * F.softplus(a.float() + gd.dt_bias)
        if gd.num_v_heads // gd.num_k_heads > 1:
            r = gd.num_v_heads // gd.num_k_heads
            query = query.repeat_interleave(r, dim=2)
            key = key.repeat_interleave(r, dim=2)

        core, new_rs = gd.recurrent_gated_delta_rule(
            query, key, value, g=g, beta=beta, initial_state=self.rec[i],
            output_final_state=True, use_qk_l2norm_in_kernel=True)
        self.rec[i].copy_(new_rs.to(self.rec[i].dtype))

        core = core.reshape(-1, gd.head_v_dim)
        z = z.reshape(-1, gd.head_v_dim)
        core = gd.norm(core, z).reshape(n, 1, -1)
        return gd.out_proj(core)

    def free(self, slot):
        self.lens[slot] = 0
