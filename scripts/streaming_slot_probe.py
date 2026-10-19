"""Prototype fixed-slot admission for first-turn tool-call generation."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
import torch_npu
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet, Qwen3_5RMSNorm, Qwen3_5RMSNormGated
from agent_protocol import CONCISE_SYSTEM, TOOLS
from recurrent_single_step import recurrent_single_step
parser = argparse.ArgumentParser()
parser.add_argument("--order-file", required=True)
parser.add_argument("--reference-run", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--tasks", type=int, default=32)
parser.add_argument("--slots", type=int, default=16)
parser.add_argument("--refill-threshold", type=int, default=4)
args = parser.parse_args()
output_path = Path(args.output)
if output_path.exists():
    raise SystemExit(f"Output already exists: {output_path}")
if args.tasks <= args.slots or args.slots < 1 or args.refill_threshold < 1:
    raise SystemExit("Require tasks > slots >= refill_threshold >= 1")
torch.set_num_threads(8)
torch.manual_seed(42)
model_path = "model/qwen3.5-27b"
data_bytes = Path("data/gsm8k_128.jsonl").read_bytes()
task_data = {str(r["id"]): r for r in map(json.loads, data_bytes.splitlines())}
estimates = {}
for line in Path(args.order_file).read_text().splitlines():
    row = json.loads(line)
    if row.get("kind") == "task" and row.get("rounds"):
        estimates[str(row["id"])] = len(row["rounds"][0].get("tokens", []))
ordered_ids = sorted(estimates, key=lambda task_id: estimates[task_id], reverse=True)
task_ids = ordered_ids[:args.tasks]
if len(task_ids) != args.tasks:
    raise SystemExit("Order file has too few tasks")
reference = {}
reference_batches = {}
requested = set(task_ids)
for line in Path(args.reference_run).read_text().splitlines():
    row = json.loads(line)
    if row.get("kind") == "task" and str(row["id"]) in requested:
        task_id = str(row["id"])
        rr = row["rounds"][0]
        reference[task_id] = rr["tokens"]
        reference_batches[(rr["turn"], round(rr["batch_generation_seconds"], 6))] = rr["batch_generation_seconds"]
if reference.keys() != requested:
    raise SystemExit("Reference run is missing requested tasks")
reference_seconds = sum(reference_batches.values())
tok = AutoTokenizer.from_pretrained(model_path, padding_side="left")
prompts = {task_id: tok.apply_chat_template([{"role": "system", "content": CONCISE_SYSTEM}, {"role": "user", "content": task_data[task_id]["question"]}], tools=TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False) for task_id in task_ids}
model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": "npu:0"}).eval()
def fused_rmsnorm(self, x):
    return torch_npu.npu_rms_norm(x, (1.0 + self.weight).to(x.dtype), self.eps)[0]
def fused_gated_rmsnorm(self, hidden_states, gate=None):
    normalized = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
    return torch_npu.npu_swiglu(torch.cat((gate, normalized), dim=-1))
for module in model.modules():
    if isinstance(module, Qwen3_5RMSNorm):
        module.forward = fused_rmsnorm.__get__(module, Qwen3_5RMSNorm)
    elif isinstance(module, Qwen3_5RMSNormGated):
        module.forward = fused_gated_rmsnorm.__get__(module, Qwen3_5RMSNormGated)
    elif isinstance(module, Qwen3_5GatedDeltaNet):
        module.recurrent_gated_delta_rule = recurrent_single_step
eos_ids = model.generation_config.eos_token_id
eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
pad_id = tok.pad_token_id if tok.pad_token_id is not None else model.generation_config.eos_token_id
total_tasks = len(task_ids)
slot_ids = [None] * args.slots
slot_active = [False] * args.slots
generated = {task_id: [] for task_id in task_ids}
admission_groups = []
decode_steps = decode_rows = prefill_calls = rebase_calls = 0
def position_ids_from_mask(mask):
    text_positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 0)
    return text_positions.unsqueeze(0).expand(4, -1, -1)
def encode_prompts(task_group):
    batch = tok([prompts[task_id] for task_id in task_group], add_special_tokens=False, padding=True, return_tensors="pt").to("npu")
    return batch["input_ids"], batch["attention_mask"]

def refill_existing_cache(cache, cache_mask, task_group, rows, live_rows):
    prompt_ids, prompt_mask = encode_prompts(task_group)
    prompt_width = prompt_ids.shape[1]
    old_width = cache.get_seq_length()
    row_to_prompt = {row: index for index, row in enumerate(rows)}
    dummy = torch.tensor(pad_id, dtype=torch.long, device="npu")
    input_ids = dummy.expand(args.slots, prompt_width).clone()
    suffix_mask = torch.zeros((args.slots, prompt_width), dtype=torch.long, device="npu")
    combined_mask = torch.cat((cache_mask, suffix_mask), dim=-1)
    for row in range(args.slots):
        if row in row_to_prompt:
            source = row_to_prompt[row]
            input_ids[row] = prompt_ids[source]
            suffix_mask[row] = prompt_mask[source]
            combined_mask[row, old_width:] = prompt_mask[source]
        elif live_rows[row]:
            input_ids[row] = pending[row]

    state_backups = []
    for layer in cache.layers:
        for name, flag in (("conv_states", "is_conv_states_initialized"),
                           ("recurrent_states", "is_recurrent_states_initialized")):
            if getattr(layer, flag, False):
                tensor = getattr(layer, name)
                keep = [row for row in range(args.slots) if row not in row_to_prompt]
                state_backups.append((tensor, keep, tensor[keep].clone()))
        for row in rows:
            if getattr(layer, "is_conv_states_initialized", False):
                layer.conv_states[row].zero_()
            if getattr(layer, "is_recurrent_states_initialized", False):
                layer.recurrent_states[row].zero_()

    positions = position_ids_from_mask(combined_mask)[:, :, -prompt_width:]
    out = model(
        input_ids=input_ids,
        attention_mask=combined_mask,
        position_ids=positions,
        cache_position=torch.arange(old_width, old_width + prompt_width, device="npu"),
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
        return_dict=True,
    )
    for tensor, keep, backup in state_backups:
        tensor[keep].copy_(backup)

    valid_lengths = combined_mask.sum(-1)
    for layer in cache.layers:
        if not getattr(layer, "is_initialized", False):
            continue
        for name in ("keys", "values"):
            tensor = getattr(layer, name)
            for row in range(args.slots):
                indices = combined_mask[row].bool().nonzero(as_tuple=False).flatten()
                count = indices.numel()
                if count:
                    compact = tensor[row].index_select(-2, indices.to(tensor.device))
                    tensor[row, :, :count, :].copy_(compact)
            setattr(layer, name, tensor[..., :int(valid_lengths.max().item()), :])
    return out.logits[:, -1, :], combined_mask

torch.npu.synchronize()
timer_started = time.perf_counter()
initial_group = task_ids[:args.slots]
slot_ids = list(initial_group)
slot_active = [True] * args.slots
initial_ids, cache_mask = encode_prompts(initial_group)
initial_batch = tok([prompts[task_id] for task_id in initial_group], add_special_tokens=False, padding=True, return_tensors="pt").to("npu")
initial_out = model(**initial_batch, position_ids=position_ids_from_mask(initial_batch["attention_mask"]), use_cache=True, logits_to_keep=1, return_dict=True)
cache, cache_mask, logits = initial_out.past_key_values, initial_batch["attention_mask"], initial_out.logits[:, -1, :]
prefill_calls += 1
model.model.rope_deltas = torch.zeros((args.slots, 1), dtype=torch.long, device="npu")
pending = logits.argmax(-1).tolist()
queue_index = args.slots

with torch.inference_mode():
    while any(slot_active) or queue_index < total_tasks:
        free_rows = []
        for row in range(args.slots):
            if not slot_active[row]:
                free_rows.append(row)
                continue
            token_id = int(pending[row])
            task_id = slot_ids[row]
            generated[task_id].append(token_id)
            if token_id in eos_ids or len(generated[task_id]) >= 512:
                slot_active[row] = False
                free_rows.append(row)

        while (free_rows and queue_index < total_tasks
               and (len(free_rows) >= args.refill_threshold
                    or total_tasks - queue_index <= len(free_rows)
                    or not any(slot_active))):
            group_size = min(len(free_rows), total_tasks - queue_index)
            rows = free_rows[:group_size]
            del free_rows[:group_size]
            group = task_ids[queue_index:queue_index + group_size]
            queue_index += group_size
            new_logits, cache_mask = refill_existing_cache(cache, cache_mask, group, rows, slot_active)
            prefill_calls += 1
            rebase_calls += 1
            model.model.rope_deltas = torch.zeros((args.slots, 1), dtype=torch.long, device="npu")
            for row, task_id, next_id in zip(rows, group, new_logits.argmax(-1).tolist()):
                slot_ids[row] = task_id
                slot_active[row] = True
                pending[row] = int(next_id)
            admission_groups.append({"rows": rows, "task_ids": group})

        if not any(slot_active):
            continue

        step_ids = torch.tensor(
            [pending[row] if slot_active[row] else pad_id for row in range(args.slots)],
            dtype=torch.long, device="npu",
        ).unsqueeze(-1)
        active_mask = torch.tensor(slot_active, dtype=torch.long, device="npu").unsqueeze(-1)
        next_mask = torch.cat((cache_mask, active_mask), dim=-1)
        step_positions = position_ids_from_mask(next_mask)[:, :, -1:]
        out = model(
            input_ids=step_ids,
            attention_mask=next_mask,
            position_ids=step_positions,
            cache_position=torch.tensor([cache.get_seq_length()], device="npu"),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        cache, cache_mask = out.past_key_values, next_mask
        pending = out.logits[:, -1, :].argmax(-1).tolist()
        decode_steps += 1
        decode_rows += sum(slot_active)

torch.npu.synchronize()
candidate_seconds = time.perf_counter() - timer_started
records = []
for task_id in task_ids:
    ids, ref_ids = generated[task_id], reference[task_id]
    common = min(len(ids), len(ref_ids))
    first_diff = next((i for i in range(common) if ids[i] != ref_ids[i]), None)
    records.append({
        "id": task_id,
        "tokens": len(ids),
        "reference_tokens": len(ref_ids),
        "exact": ids == ref_ids,
        "first_difference": first_diff,
        "tool_call": "<tool_call>" in tok.decode(ids, skip_special_tokens=True),
    })

result = {
    "tasks": total_tasks,
    "slots": args.slots,
    "refill_threshold": args.refill_threshold,
    "task_ids": task_ids,
    "reference_batch_generation_seconds": reference_seconds,
    "streaming_generation_seconds": candidate_seconds,
    "speedup": reference_seconds / candidate_seconds,
    "exact_task_count": sum(row["exact"] for row in records),
    "all_tokens_exact": all(row["exact"] for row in records),
    "all_tool_calls_present": all(row["tool_call"] for row in records),
    "decode_steps": decode_steps,
    "active_row_steps": decode_rows,
    "prefill_calls": prefill_calls,
    "cache_rebase_calls": rebase_calls,
    "admission_groups": admission_groups,
    "peak_bytes": int(torch.npu.max_memory_allocated()),
    "data_sha256": hashlib.sha256(data_bytes).hexdigest(),
    "records": records,
}
output_path.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2), flush=True)
