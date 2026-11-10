"""Tensor parallel (TP) support for the self-written CB engine (Qwen3.5 hybrid: GatedDeltaNet + full attention).

Each rank is a separate process that sees one card as npu:0 (launch with ASCEND_RT_VISIBLE_DEVICES=<rank>).
Weights are sharded by head / channel block; the output projections of attention, GatedDeltaNet and MLP
are followed by an all-reduce (HCCL). Embeddings, norms and lm_head are replicated.

Per-rank shapes for Qwen3.5-27B with tp=2:
  full attention: 12 q heads (of 24), 2 kv heads (of 4), head_dim 256
  GatedDeltaNet:  8 k heads (of 16), 24 v heads (of 48), conv channels 1024+1024+3072
  MLP:            intermediate 8704 (of 17408)

Prefill is re-implemented here (one sequence per call) because HF's forward cannot see sharded shapes.
Decode reuses CBEngine.decode_step unchanged: the sharded modules expose their local sizes.
"""
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch_npu  # noqa: F401
from transformers.models.qwen3_5 import modeling_qwen3_5 as M

from cb_engine import CBEngine


class AllReduceLinear(nn.Module):
    """Row-parallel linear: local partial product, then sum over ranks (output identical on all ranks)."""

    def __init__(self, lin):
        super().__init__()
        self.lin = lin

    def forward(self, x):
        y = self.lin(x)
        dist.all_reduce(y)
        return y


def _set_weight(lin, w, out_dim=None):
    lin.weight = nn.Parameter(w.contiguous(), requires_grad=False)
    lin.out_features, lin.in_features = w.shape[0], w.shape[1]


def _rows(start, count):
    return torch.arange(start, start + count)


def shard_model(model, rank, tp):
    """In-place sharding of a freshly loaded (CPU) Qwen3_5ForConditionalGeneration."""
    cfg = model.config.text_config
    assert tp > 0 and rank < tp
    assert not getattr(cfg, "attention_bias", False), "attention_bias not supported in TP"
    hd = cfg.head_dim
    nh, kvh = cfg.num_attention_heads, cfg.num_key_value_heads
    assert nh % tp == 0 and kvh % tp == 0
    lm = model.model.language_model
    H = cfg.hidden_size
    for i, layer in enumerate(lm.layers):
        if cfg.layer_types[i] == "full_attention":
            at = layer.self_attn
            nh_l, kvh_l = nh // tp, kvh // tp
            # q_proj: per head [q(hd), gate(hd)] interleaved; take a contiguous block of heads.
            w = at.q_proj.weight.view(nh, 2 * hd, H)[rank * nh_l:(rank + 1) * nh_l].reshape(-1, H)
            _set_weight(at.q_proj, w)
            _set_weight(at.k_proj, at.k_proj.weight[rank * kvh_l * hd:(rank + 1) * kvh_l * hd])
            _set_weight(at.v_proj, at.v_proj.weight[rank * kvh_l * hd:(rank + 1) * kvh_l * hd])
            o = at.o_proj.weight.view(H, nh, hd)[:, rank * nh_l:(rank + 1) * nh_l].reshape(H, -1)
            _set_weight(at.o_proj, o)
            at.o_proj = AllReduceLinear(at.o_proj)
        else:
            gd = layer.linear_attn
            kh, vh = gd.num_k_heads, gd.num_v_heads
            dk, dv = gd.head_k_dim, gd.head_v_dim
            kh_l, vh_l = kh // tp, vh // tp
            key_dim, value_dim = gd.key_dim, gd.value_dim
            # in_proj_qkv rows: [q (kh*dk) | k (kh*dk) | v (vh*dv)]; keep the same head blocks in each part.
            idx = torch.cat([
                _rows(rank * kh_l * dk, kh_l * dk),
                _rows(key_dim + rank * kh_l * dk, kh_l * dk),
                _rows(2 * key_dim + rank * vh_l * dv, vh_l * dv),
            ])
            _set_weight(gd.in_proj_qkv, gd.in_proj_qkv.weight.index_select(0, idx))
            gd.conv1d.weight = nn.Parameter(gd.conv1d.weight.index_select(0, idx).contiguous(), requires_grad=False)
            gd.conv1d.out_channels = gd.conv1d.in_channels = idx.numel()
            gd.conv1d.groups = idx.numel()
            vrows = _rows(rank * vh_l * dv, vh_l * dv)
            _set_weight(gd.in_proj_z, gd.in_proj_z.weight.index_select(0, vrows))
            hrows = _rows(rank * vh_l, vh_l)
            _set_weight(gd.in_proj_b, gd.in_proj_b.weight.index_select(0, hrows))
            _set_weight(gd.in_proj_a, gd.in_proj_a.weight.index_select(0, hrows))
            gd.A_log = nn.Parameter(gd.A_log.data[hrows].contiguous(), requires_grad=False)
            gd.dt_bias = nn.Parameter(gd.dt_bias.data[hrows].contiguous(), requires_grad=False)
            _set_weight(gd.out_proj, gd.out_proj.weight[:, rank * vh_l * dv:(rank + 1) * vh_l * dv])
            gd.out_proj = AllReduceLinear(gd.out_proj)
            # local sizes used by the forward code
            gd.num_k_heads, gd.num_v_heads = kh_l, vh_l
            gd.key_dim, gd.value_dim = kh_l * dk, vh_l * dv
            gd.conv_dim = 2 * gd.key_dim + gd.value_dim
        mlp = layer.mlp
        inter = mlp.gate_proj.weight.shape[0]
        assert inter % tp == 0
        r0, r1 = rank * (inter // tp), (rank + 1) * (inter // tp)
        _set_weight(mlp.gate_proj, mlp.gate_proj.weight[r0:r1])
        _set_weight(mlp.up_proj, mlp.up_proj.weight[r0:r1])
        _set_weight(mlp.down_proj, mlp.down_proj.weight[:, r0:r1])
        mlp.down_proj = AllReduceLinear(mlp.down_proj)
    return model


def init_tp(rank, world, master_port="29561"):
    import os
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", master_port)
    torch_npu.npu.set_device(0)               # each process sees exactly one card
    dist.init_process_group("hccl", rank=rank, world_size=world)


class TPEngine(CBEngine):
    """CBEngine on sharded weights. Decode may be graph-captured: the HCCL all_reduce in AllReduceLinear
    is captured inside the NPUGraph (verified in scripts/test_hccl_graph.py). Graphs are captured in
    CBEngine.__init__, i.e. before any prefill, and every rank must run the same capture sequence."""

    def __init__(self, model, n_slots, t_max, device, tp, graph_decode=False):
        super().__init__(model, n_slots, t_max, device, graph_decode=graph_decode, tp=tp)

    @torch.inference_mode()
    def prefill(self, slot, ids):
        """Single sequence prefill with sharded weights. ids: [1, L] on device. Returns first token."""
        L = ids.shape[1]
        assert L < self.T, f"prompt {L} >= T {self.T}"
        m, cfg, hd, K = self.lm, self.cfg, self.hd, self.K
        x = m.embed_tokens(ids)                                         # [1,L,H]
        pos3 = torch.arange(L, device=self.dev).view(1, 1, L).expand(3, 1, L)
        cos, sin = m.rotary_emb(x, pos3)
        for i, layer in enumerate(m.layers):
            h = layer.input_layernorm(x)
            if cfg.layer_types[i] == "full_attention":
                at = layer.self_attn
                qg = at.q_proj(h).view(1, L, -1, hd * 2)
                q, gate = torch.chunk(qg, 2, dim=-1)
                gate = gate.reshape(1, L, -1)
                q = at.q_norm(q).transpose(1, 2)                         # [1,nh,L,hd]
                k = at.k_norm(at.k_proj(h).view(1, L, self.kvh, hd)).transpose(1, 2)
                v = at.v_proj(h).view(1, L, self.kvh, hd).transpose(1, 2)
                q, k = M.apply_rotary_pos_emb(q, k, cos, sin)
                self.kv_k[i][slot, :, :L] = k[0]
                self.kv_v[i][slot, :, :L] = v[0]
                kr = k.repeat_interleave(self.g, dim=1)                  # kv-major, as repeat_kv
                vr = v.repeat_interleave(self.g, dim=1)
                o = F.scaled_dot_product_attention(q, kr, vr, is_causal=True, scale=at.scaling)
                o = o.transpose(1, 2).reshape(1, L, -1) * torch.sigmoid(gate)
                h = at.o_proj(o)
            else:
                gd = layer.linear_attn
                mixed = gd.in_proj_qkv(h).transpose(1, 2)               # [1,C,L]
                z = gd.in_proj_z(h).reshape(1, L, -1, gd.head_v_dim)
                b = gd.in_proj_b(h)
                a = gd.in_proj_a(h)
                # conv state for decode = last K inputs (left zero-padded if L < K)
                if L >= K:
                    conv_state = mixed[:, :, -K:]
                else:
                    conv_state = F.pad(mixed, (K - L, 0))
                mixed = F.silu(F.conv1d(mixed, gd.conv1d.weight, gd.conv1d.bias,
                                        padding=K - 1, groups=gd.conv_dim)[:, :, :L])
                mixed = mixed.transpose(1, 2)
                query, key, value = torch.split(mixed, [gd.key_dim, gd.key_dim, gd.value_dim], dim=-1)
                query = query.reshape(1, L, -1, gd.head_k_dim)
                key = key.reshape(1, L, -1, gd.head_k_dim)
                value = value.reshape(1, L, -1, gd.head_v_dim)
                beta = b.sigmoid()
                g = -gd.A_log.float().exp() * F.softplus(a.float() + gd.dt_bias)
                r = gd.num_v_heads // gd.num_k_heads
                if r > 1:
                    query = query.repeat_interleave(r, dim=2)
                    key = key.repeat_interleave(r, dim=2)
                core, last = gd.chunk_gated_delta_rule(
                    query, key, value, g=g, beta=beta, initial_state=None,
                    output_final_state=True, use_qk_l2norm_in_kernel=True)
                self.conv[i][slot] = conv_state[0].to(self.conv[i].dtype)
                self.rec[i][slot] = last[0].to(self.rec[i].dtype)
                core = gd.norm(core.reshape(-1, gd.head_v_dim), z.reshape(-1, gd.head_v_dim))
                h = gd.out_proj(core.reshape(1, L, -1))
            x = x + h
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        x = m.norm(x)
        logits = self.model.lm_head(x[:, -1])
        self.lens[slot] = L
        return int(logits[0].argmax(-1).item())

    @torch.inference_mode()
    def prefill_batch(self, slots, id_list):
        # Sequential per-prompt prefill (cb_run is used with --prefill-group 1 for TP).
        return [self.prefill(s, ids) for s, ids in zip(slots, id_list)]

    @torch.inference_mode()
    def prefill_chunk(self, slot, ids, start, last):
        """Prefill tokens [start, start+C) of one prompt into `slot`. Earlier chunks of the same prompt
        must already be in the slot (KV rows [0, start), conv/recurrent state after `start` tokens).
        ids: [1, C] on device. Returns the first generated token when `last`, else None.
        Exact chunking: attention reads all cached KV with a causal mask; GDN carries conv window and
        recurrent state; the chunk kernel takes the previous state as initial_state."""
        C = ids.shape[1]
        S = start + C
        assert S < self.T, f"prompt {S} >= T {self.T}"
        m, cfg, hd, K = self.lm, self.cfg, self.hd, self.K
        x = m.embed_tokens(ids)                                          # [1,C,H]
        qpos = torch.arange(start, S, device=self.dev)
        pos3 = qpos.view(1, 1, C).expand(3, 1, C)
        cos, sin = m.rotary_emb(x, pos3)
        kpos = torch.arange(S, device=self.dev)
        causal = (kpos[None, :] <= qpos[:, None])[None, None]            # [1,1,C,S] bool
        for i, layer in enumerate(m.layers):
            h = layer.input_layernorm(x)
            if cfg.layer_types[i] == "full_attention":
                at = layer.self_attn
                qg = at.q_proj(h).view(1, C, -1, hd * 2)
                q, gate = torch.chunk(qg, 2, dim=-1)
                gate = gate.reshape(1, C, -1)
                q = at.q_norm(q).transpose(1, 2)                          # [1,nh,C,hd]
                k = at.k_norm(at.k_proj(h).view(1, C, self.kvh, hd)).transpose(1, 2)
                v = at.v_proj(h).view(1, C, self.kvh, hd).transpose(1, 2)
                q, k = M.apply_rotary_pos_emb(q, k, cos, sin)
                self.kv_k[i][slot, :, start:S] = k[0]
                self.kv_v[i][slot, :, start:S] = v[0]
                kr = self.kv_k[i][slot:slot + 1, :, :S].repeat_interleave(self.g, dim=1)   # [1,nh,S,hd]
                vr = self.kv_v[i][slot:slot + 1, :, :S].repeat_interleave(self.g, dim=1)
                o = F.scaled_dot_product_attention(q, kr, vr, attn_mask=causal, scale=at.scaling)
                o = o.transpose(1, 2).reshape(1, C, -1) * torch.sigmoid(gate)
                h = at.o_proj(o)
            else:
                gd = layer.linear_attn
                mixed = gd.in_proj_qkv(h).transpose(1, 2)                # [1,Cch,C]
                z = gd.in_proj_z(h).reshape(1, C, -1, gd.head_v_dim)
                b = gd.in_proj_b(h)
                a = gd.in_proj_a(h)
                # previous K inputs of this sequence (zeros at the start, matching the causal pad)
                if start == 0:
                    prev = torch.zeros(1, gd.conv_dim, K, dtype=mixed.dtype, device=mixed.device)
                else:
                    prev = self.conv[i][slot:slot + 1].to(mixed.dtype)
                full_in = torch.cat([prev, mixed], dim=-1)               # [1,Cch,K+C]
                # kernel K, no padding: K-1 history columns + C new -> C outputs
                mixed = F.silu(F.conv1d(full_in[:, :, 1:], gd.conv1d.weight, gd.conv1d.bias,
                                        groups=gd.conv_dim))             # [1,Cch,C]
                mixed = mixed.transpose(1, 2)
                query, key, value = torch.split(mixed, [gd.key_dim, gd.key_dim, gd.value_dim], dim=-1)
                query = query.reshape(1, C, -1, gd.head_k_dim)
                key = key.reshape(1, C, -1, gd.head_k_dim)
                value = value.reshape(1, C, -1, gd.head_v_dim)
                beta = b.sigmoid()
                g = -gd.A_log.float().exp() * F.softplus(a.float() + gd.dt_bias)
                r = gd.num_v_heads // gd.num_k_heads
                if r > 1:
                    query = query.repeat_interleave(r, dim=2)
                    key = key.repeat_interleave(r, dim=2)
                init = None if start == 0 else self.rec[i][slot:slot + 1]
                core, last_state = gd.chunk_gated_delta_rule(
                    query, key, value, g=g, beta=beta, initial_state=init,
                    output_final_state=True, use_qk_l2norm_in_kernel=True)
                self.conv[i][slot] = full_in[0, :, -K:].to(self.conv[i].dtype)
                self.rec[i][slot] = last_state[0].to(self.rec[i].dtype)
                core = gd.norm(core.reshape(-1, gd.head_v_dim), z.reshape(-1, gd.head_v_dim))
                h = gd.out_proj(core.reshape(1, C, -1))
            x = x + h
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        self.lens[slot] = S
        if not last:
            return None
        x = m.norm(x[:, -1:])
        logits = self.model.lm_head(x[:, 0])
        return int(logits[0].argmax(-1).item())

    @torch.inference_mode()
    def decode_step(self, slots, tokens, hold=()):
        """Eager decode over all slots. `hold`: slots mid-prefill. The dense decode pass also advances
        their conv/recurrent state with dummy input, so save and restore it around the step."""
        if not hold:
            return super().decode_step(slots, tokens)
        saved = [(i, s, self.conv[i][s].clone(), self.rec[i][s].clone())
                 for s in hold for i in self.conv]
        out = super().decode_step(slots, tokens)
        for i, s, c, r in saved:
            self.conv[i][s].copy_(c)
            self.rec[i][s].copy_(r)
        return out
