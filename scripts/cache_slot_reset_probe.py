"""Check Qwen3.5 hybrid-cache slot reset with masked historical KV rows."""
import json
import time
from pathlib import Path

import torch
import torch_npu
from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM


torch.set_num_threads(8)
torch.manual_seed(42)
cfg = json.loads(Path("model/qwen3.5-27b/config.json").read_text())["text_config"]
cfg.update(num_hidden_layers=4, layer_types=cfg["layer_types"][:4], vocab_size=1024)
config = Qwen3_5TextConfig(**cfg)
config._attn_implementation = "sdpa"
model = Qwen3_5ForCausalLM(config).to(dtype=torch.bfloat16, device="npu").eval()

old_prompt_len = 48
new_prompt_len = 11
old_ids = torch.randint(0, 1024, (2, old_prompt_len), device="npu")
new_ids = torch.randint(0, 1024, (1, new_prompt_len), device="npu")
dummy_ids = old_ids[1:2, -1:].expand(1, new_prompt_len)


def prefill(ids):
    return model(ids, use_cache=True, logits_to_keep=1, return_dict=True)


with torch.inference_mode():
    baseline = prefill(new_ids)
    baseline_logits = baseline.logits[:, -1, :].float()
    baseline_cache = baseline.past_key_values

    original = prefill(old_ids)
    cache = original.past_key_values
    before_bytes = sum(
        value.numel() * value.element_size()
        for layer in cache.layers
        for name in ("keys", "values", "conv_states", "recurrent_states")
        if (value := getattr(layer, name, None)) is not None
    )

    for layer in cache.layers:
        if getattr(layer, "is_conv_states_initialized", False):
            layer.conv_states[0].zero_()
        if getattr(layer, "is_recurrent_states_initialized", False):
            layer.recurrent_states[0].zero_()

    input_ids = torch.cat((new_ids, dummy_ids), dim=0)
    attention_mask = torch.zeros((2, old_prompt_len + new_prompt_len),
                                 dtype=torch.long, device="npu")
    attention_mask[0, old_prompt_len:] = 1
    attention_mask[1, :old_prompt_len] = 1
    attention_mask[1, old_prompt_len] = 1
    position_ids = (attention_mask.cumsum(-1) - 1).clamp_min(0)
    position_ids = position_ids[:, -new_prompt_len:]
    cache_position = torch.arange(old_prompt_len, old_prompt_len + new_prompt_len,
                                  device="npu")

    started = time.perf_counter()
    replaced = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache_position=cache_position,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
        return_dict=True,
    )
    torch.npu.synchronize()
    replacement_seconds = time.perf_counter() - started
    replacement_logits = replaced.logits[0:1, -1, :].float()
    error = (replacement_logits - baseline_logits).abs()
    after_bytes = sum(
        value.numel() * value.element_size()
        for layer in cache.layers
        for name in ("keys", "values", "conv_states", "recurrent_states")
        if (value := getattr(layer, name, None)) is not None
    )

    valid_lengths = attention_mask.sum(-1)
    max_valid = int(valid_lengths.max().item())
    for layer in cache.layers:
        if not getattr(layer, "is_initialized", False):
            continue
        for name in ("keys", "values"):
            tensor = getattr(layer, name)
            compacted = torch.zeros(
                (*tensor.shape[:-2], max_valid, tensor.shape[-1]),
                dtype=tensor.dtype, device=tensor.device,
            )
            for row_index in range(tensor.shape[0]):
                positions = attention_mask[row_index].bool().nonzero(as_tuple=False).flatten()
                count = positions.numel()
                compacted[row_index, ..., :count, :] = tensor[row_index].index_select(
                    -2, positions.to(tensor.device)
                )
            setattr(layer, name, compacted)

    rebased_mask = torch.arange(max_valid, device="npu")[None, :] < valid_lengths[:, None]
    rebased_mask = rebased_mask.long()
    rebase_bytes = sum(
        value.numel() * value.element_size()
        for layer in cache.layers
        for name in ("keys", "values", "conv_states", "recurrent_states")
        if (value := getattr(layer, name, None)) is not None
    )

    next_token = baseline_logits.argmax(-1, keepdim=True)
    baseline_decode = model(
        input_ids=next_token,
        attention_mask=torch.ones((1, new_prompt_len + 1), dtype=torch.long, device="npu"),
        position_ids=torch.tensor([[new_prompt_len]], device="npu"),
        cache_position=torch.tensor([new_prompt_len], device="npu"),
        past_key_values=baseline_cache,
        use_cache=True,
        logits_to_keep=1,
        return_dict=True,
    )
    candidate_ids = torch.cat((next_token, dummy_ids[:, :1]), dim=0)
    candidate_mask = torch.cat((rebased_mask, torch.ones((2, 1), dtype=torch.long, device="npu")), dim=-1)
    candidate_positions = (candidate_mask.cumsum(-1) - 1).clamp_min(0)[:, -1:]
    candidate_decode = model(
        input_ids=candidate_ids,
        attention_mask=candidate_mask,
        position_ids=candidate_positions,
        cache_position=torch.tensor([max_valid], device="npu"),
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
        return_dict=True,
    )
    torch.npu.synchronize()
    baseline_next = baseline_decode.logits[:, -1, :].float()
    candidate_next = candidate_decode.logits[0:1, -1, :].float()
    next_error = (candidate_next - baseline_next).abs()

    row = {
        "layers": len(cache.layers),
        "layer_types": cfg["layer_types"],
        "old_prompt_len": old_prompt_len,
        "new_prompt_len": new_prompt_len,
        "replaced_slot_index": 0,
        "cache_bytes_before": before_bytes,
        "cache_bytes_after": after_bytes,
        "cache_bytes_after_rebase": rebase_bytes,
        "cache_bytes_added": after_bytes - before_bytes,
        "cache_sequence_length_before": old_prompt_len,
        "cache_sequence_length_after_replacement": old_prompt_len + new_prompt_len,
        "cache_sequence_length_after_rebase": max_valid,
        "replacement_seconds": replacement_seconds,
        "logits_max_abs_error": float(error.max().cpu()),
        "logits_mean_abs_error": float(error.mean().cpu()),
        "argmax_equal": bool(torch.equal(replacement_logits.argmax(-1), baseline_logits.argmax(-1))),
        "next_logits_max_abs_error": float(next_error.max().cpu()),
        "next_logits_mean_abs_error": float(next_error.mean().cpu()),
        "next_argmax_equal": bool(torch.equal(candidate_next.argmax(-1), baseline_next.argmax(-1))),
        "finite": bool(torch.isfinite(replacement_logits).all()),
        "peak_bytes": int(torch.npu.max_memory_allocated()),
    }

Path("results/cache_slot_reset_probe.json").write_text(json.dumps(row, indent=2) + "\n")
print(json.dumps(row), flush=True)
