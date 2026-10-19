"""Validate hybrid-cache slot reset and KV rebasing on the full Qwen3.5 model."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch_npu
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5GatedDeltaNet,
    Qwen3_5RMSNorm,
    Qwen3_5RMSNormGated,
)

from agent_protocol import CONCISE_SYSTEM, TOOLS
from recurrent_single_step import recurrent_single_step


parser = argparse.ArgumentParser()
parser.add_argument("--order-file", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
output_path = Path(args.output)
if output_path.exists():
    raise SystemExit(f"Output already exists: {output_path}")

torch.set_num_threads(8)
torch.manual_seed(42)
model_path = "model/qwen3.5-27b"
task_data = {
    str(row["id"]): row
    for row in (json.loads(line) for line in Path("data/gsm8k_128.jsonl").read_text().splitlines())
}
estimates = {}
for line in Path(args.order_file).read_text().splitlines():
    row = json.loads(line)
    if row.get("kind") == "task" and row.get("rounds"):
        estimates[str(row["id"])] = len(row["rounds"][0].get("tokens", []))
ordered_ids = sorted(estimates, key=lambda task_id: estimates[task_id], reverse=True)
if len(ordered_ids) < 17:
    raise SystemExit("Order file must contain at least 17 task records")
old_ids, new_id = ordered_ids[:16], ordered_ids[16]

tok = AutoTokenizer.from_pretrained(model_path, padding_side="left")

def render(task_id):
    return tok.apply_chat_template(
        [{"role": "system", "content": CONCISE_SYSTEM},
         {"role": "user", "content": task_data[task_id]["question"]}],
        tools=TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


old_inputs = tok([render(task_id) for task_id in old_ids], add_special_tokens=False,
                 padding=True, return_tensors="pt").to("npu")
new_inputs = tok([render(new_id)], add_special_tokens=False,
                 padding=True, return_tensors="pt").to("npu")
new_prompt_len = int(new_inputs["attention_mask"].sum(-1)[0].item())
new_ids_tensor = new_inputs["input_ids"][:, -new_prompt_len:]
old_width = old_inputs["input_ids"].shape[1]

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    model_path, dtype=torch.bfloat16, attn_implementation="sdpa",
    device_map={"": "npu:0"},
).eval()

def fused_rmsnorm(self, x):
    return torch_npu.npu_rms_norm(x, (1.0 + self.weight).to(x.dtype), self.eps)[0]

def fused_gated_rmsnorm(self, hidden_states, gate=None):
    normalized = torch_npu.npu_rms_norm(
        hidden_states, self.weight, self.variance_epsilon
    )[0]
    return torch_npu.npu_swiglu(torch.cat((gate, normalized), dim=-1))

for module in model.modules():
    if isinstance(module, Qwen3_5RMSNorm):
        module.forward = fused_rmsnorm.__get__(module, Qwen3_5RMSNorm)
    elif isinstance(module, Qwen3_5RMSNormGated):
        module.forward = fused_gated_rmsnorm.__get__(module, Qwen3_5RMSNormGated)
    elif isinstance(module, Qwen3_5GatedDeltaNet):
        module.recurrent_gated_delta_rule = recurrent_single_step

with torch.inference_mode():
    torch.npu.synchronize()
    baseline_started = time.perf_counter()
    baseline = model(**new_inputs, use_cache=True, logits_to_keep=1, return_dict=True)
    torch.npu.synchronize()
    baseline_prefill_seconds = time.perf_counter() - baseline_started
    baseline_logits = baseline.logits[:, -1, :].float()
    baseline_cache = baseline.past_key_values

    initial = model(**old_inputs, use_cache=True, logits_to_keep=1, return_dict=True)
    cache = initial.past_key_values
    old_next_tokens = initial.logits[:, -1, :].argmax(-1, keepdim=True)
    initial_mask = old_inputs["attention_mask"]
    before_seq_len = cache.get_seq_length()
    before_cache_bytes = sum(
        tensor.numel() * tensor.element_size()
        for layer in cache.layers
        for name in ("keys", "values", "conv_states", "recurrent_states")
        if (tensor := getattr(layer, name, None)) is not None
    )

    live_state_backups = []
    for layer in cache.layers:
        if getattr(layer, "is_conv_states_initialized", False):
            live_state_backups.append((layer, "conv_states", layer.conv_states[1:].clone()))
            layer.conv_states[0].zero_()
        if getattr(layer, "is_recurrent_states_initialized", False):
            live_state_backups.append((layer, "recurrent_states", layer.recurrent_states[1:].clone()))
            layer.recurrent_states[0].zero_()

    batch_size = old_inputs["input_ids"].shape[0]
    replacement_ids = old_inputs["input_ids"][:, -1:].expand(
        batch_size, new_prompt_len
    ).clone()
    replacement_ids[0] = new_ids_tensor[0]
    replacement_mask = torch.zeros(
        (batch_size, old_width + new_prompt_len), dtype=torch.long, device="npu"
    )
    replacement_mask[0, old_width:] = 1
    replacement_mask[1:, :old_width] = initial_mask[1:]
    replacement_positions = (replacement_mask.cumsum(-1) - 1).clamp_min(0)
    replacement_positions = replacement_positions[:, -new_prompt_len:]
    replacement_cache_position = torch.arange(
        old_width, old_width + new_prompt_len, device="npu"
    )
    replacement_started = time.perf_counter()
    replaced = model(
        input_ids=replacement_ids,
        attention_mask=replacement_mask,
        position_ids=replacement_positions,
        cache_position=replacement_cache_position,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
        return_dict=True,
    )
    torch.npu.synchronize()
    replacement_prefill_seconds = time.perf_counter() - replacement_started
    replaced_logits = replaced.logits[0:1, -1, :].float()
    for layer, name, saved_rows in live_state_backups:
        getattr(layer, name)[1:].copy_(saved_rows)

    valid_lengths = replacement_mask.sum(-1)
    compacted_width = int(valid_lengths.max().item())
    rebase_started = time.perf_counter()
    for layer in cache.layers:
        if not getattr(layer, "is_initialized", False):
            continue
        for name in ("keys", "values"):
            tensor = getattr(layer, name)
            compacted = torch.zeros(
                (*tensor.shape[:-2], compacted_width, tensor.shape[-1]),
                dtype=tensor.dtype, device=tensor.device,
            )
            for row_index in range(batch_size):
                positions = replacement_mask[row_index].bool().nonzero(as_tuple=False).flatten()
                count = positions.numel()
                compacted[row_index, ..., :count, :] = tensor[row_index].index_select(
                    -2, positions.to(tensor.device)
                )
            setattr(layer, name, compacted)
    torch.npu.synchronize()
    rebase_seconds = time.perf_counter() - rebase_started

    rebased_mask = (torch.arange(compacted_width, device="npu")[None, :]
                    < valid_lengths[:, None]).long()
    rebased_cache_bytes = sum(
        tensor.numel() * tensor.element_size()
        for layer in cache.layers
        for name in ("keys", "values", "conv_states", "recurrent_states")
        if (tensor := getattr(layer, name, None)) is not None
    )

    next_token = baseline_logits.argmax(-1, keepdim=True)
    baseline_next = model(
        input_ids=next_token,
        attention_mask=torch.ones((1, new_prompt_len + 1), dtype=torch.long, device="npu"),
        position_ids=torch.tensor([[new_prompt_len]], device="npu"),
        cache_position=torch.tensor([new_prompt_len], device="npu"),
        past_key_values=baseline_cache,
        use_cache=True,
        logits_to_keep=1,
        return_dict=True,
    ).logits[:, -1, :].float()

    candidate_ids = torch.cat((next_token, old_next_tokens[1:]), dim=0)
    candidate_mask = torch.cat((rebased_mask, torch.ones((batch_size, 1), dtype=torch.long,
                                                          device="npu")), dim=-1)
    candidate_positions = (candidate_mask.cumsum(-1) - 1).clamp_min(0)[:, -1:]
    candidate_next = model(
        input_ids=candidate_ids,
        attention_mask=candidate_mask,
        position_ids=candidate_positions,
        cache_position=torch.tensor([compacted_width], device="npu"),
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
        return_dict=True,
    ).logits[0:1, -1, :].float()
    torch.npu.synchronize()

replacement_error = (replaced_logits - baseline_logits).abs()
next_error = (candidate_next - baseline_next).abs()
result = {
    "layers": len(cache.layers),
    "old_task_ids": old_ids,
    "new_task_id": new_id,
    "old_batch_width": old_width,
    "new_prompt_tokens": new_prompt_len,
    "cache_seq_len_before": before_seq_len,
    "cache_seq_len_after_append": old_width + new_prompt_len,
    "cache_seq_len_after_rebase": compacted_width,
    "cache_bytes_before": before_cache_bytes,
    "cache_bytes_after_rebase": rebased_cache_bytes,
    "replacement_logits_max_abs_error": float(replacement_error.max().cpu()),
    "replacement_argmax_equal": bool(torch.equal(
        replaced_logits.argmax(-1), baseline_logits.argmax(-1)
    )),
    "next_logits_max_abs_error": float(next_error.max().cpu()),
    "next_argmax_equal": bool(torch.equal(candidate_next.argmax(-1), baseline_next.argmax(-1))),
    "finite": bool(torch.isfinite(candidate_next).all()),
    "baseline_prefill_seconds": baseline_prefill_seconds,
    "replacement_prefill_seconds": replacement_prefill_seconds,
    "kv_rebase_seconds": rebase_seconds,
    "peak_bytes": int(torch.npu.max_memory_allocated()),
    "data_sha256": hashlib.sha256(Path("data/gsm8k_128.jsonl").read_bytes()).hexdigest(),
}
output_path.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
