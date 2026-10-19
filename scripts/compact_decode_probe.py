"""Compare HF greedy decoding with EOS-finished-row cache compaction."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from types import MethodType

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
parser.add_argument("--task-ids", required=True, help="Comma-separated fixed task IDs")
parser.add_argument("--order-file", required=True, help="Run used for first-turn length ordering")
parser.add_argument("--reference-run", required=True, help="E084-style JSONL used for token comparison")
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
requested = set(args.task_ids.split(","))
if not requested or requested - task_data.keys():
    raise SystemExit("Unknown or empty task ID list")

estimates = {}
for line in Path(args.order_file).read_text().splitlines():
    row = json.loads(line)
    if row.get("kind") == "task" and row.get("rounds"):
        estimates[str(row["id"])] = len(row["rounds"][0].get("tokens", []))
if requested - estimates.keys():
    raise SystemExit("Missing first-turn generation estimates")
ordered_ids = sorted(requested, key=lambda task_id: estimates[task_id], reverse=True)

reference = {}
for line in Path(args.reference_run).read_text().splitlines():
    row = json.loads(line)
    if row.get("kind") == "task" and str(row["id"]) in requested:
        reference[str(row["id"])] = row["rounds"][0]["tokens"]
if reference.keys() != requested:
    raise SystemExit("Reference run does not contain every requested task")

tok = AutoTokenizer.from_pretrained(model_path, padding_side="left")
prompts = [tok.apply_chat_template(
    [{"role": "system", "content": CONCISE_SYSTEM},
     {"role": "user", "content": task_data[task_id]["question"]}],
    tools=TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False,
) for task_id in ordered_ids]
inputs = tok(prompts, add_special_tokens=False, padding=True, return_tensors="pt").to("npu")
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


def compact_sample(self, input_ids, logits_processor, stopping_criteria,
                   generation_config, synced_gpus=False, streamer=None, **model_kwargs):
    if generation_config.do_sample or generation_config.num_beams != 1:
        raise ValueError("Probe supports greedy decoding only")
    pad_id = int(generation_config._pad_token_tensor.item())
    batch_size, prompt_width = input_ids.shape
    max_new_tokens = generation_config.max_length - prompt_width
    sequences = torch.full(
        (batch_size, generation_config.max_length), pad_id,
        dtype=input_ids.dtype, device=input_ids.device,
    )
    sequences[:, :prompt_width] = input_ids
    original_rows = torch.arange(batch_size, device=input_ids.device)
    active_ids = input_ids
    outputs = self._prefill(
        active_ids, generation_config, model_kwargs,
        is_first_iteration=not generation_config.is_assistant,
    )
    prefill_consumed = False
    generated_steps = 0
    while active_ids.shape[0] and generated_steps < max_new_tokens:
        if prefill_consumed:
            model_inputs = self.prepare_inputs_for_generation(
                active_ids, next_sequence_length=1 if model_kwargs["use_cache"] else None,
                **model_kwargs,
            )
            with self._optimize_model_for_decode():
                outputs = self(**model_inputs, return_dict=True)
        prefill_consumed = True
        model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs)
        logits = outputs.logits[:, -1, :].to(
            copy=True, dtype=torch.float32, device=active_ids.device
        )
        scores = logits_processor(active_ids, logits)
        next_tokens = torch.argmax(scores, dim=-1)
        active_ids = torch.cat((active_ids, next_tokens[:, None]), dim=-1)
        sequences[original_rows, prompt_width + generated_steps] = next_tokens
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        generated_steps += 1

        stopped = stopping_criteria(active_ids, None)
        if stopped.ndim > 1:
            stopped = stopped.squeeze(-1)
        keep = ~stopped.bool()
        if not keep.any():
            break
        if not keep.all():
            keep_indices = keep.nonzero(as_tuple=False).flatten()
            cache = model_kwargs.get("past_key_values")
            cache.batch_select_indices(keep_indices)
            old_batch = active_ids.shape[0]
            for key, value in list(model_kwargs.items()):
                if (key != "past_key_values" and isinstance(value, torch.Tensor)
                        and value.ndim >= 2 and value.shape[0] == old_batch):
                    model_kwargs[key] = value.index_select(0, keep_indices)
            active_ids = active_ids.index_select(0, keep_indices)
            original_rows = original_rows.index_select(0, keep_indices)
        del outputs

    if streamer is not None:
        streamer.end()
    return sequences[:, :prompt_width + generated_steps]


torch.npu.synchronize()
started = time.perf_counter()
reference_output = model.generate(
    **inputs, max_new_tokens=512, do_sample=False, use_cache=True,
)
torch.npu.synchronize()
reference_seconds = time.perf_counter() - started
del reference_output
torch.npu.empty_cache()

model._sample = MethodType(compact_sample, model)
torch.npu.synchronize()
started = time.perf_counter()
compact_output = model.generate(
    **inputs, max_new_tokens=512, do_sample=False, use_cache=True,
)
torch.npu.synchronize()
compact_seconds = time.perf_counter() - started
compact_rows = compact_output[:, inputs["input_ids"].shape[1]:].tolist()
eos_ids = model.generation_config.eos_token_id
eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])

records = []
for index, task_id in enumerate(ordered_ids):
    ids = compact_rows[index]
    stop = next((i + 1 for i, token in enumerate(ids) if token in eos_ids), len(ids))
    ids = ids[:stop]
    ref_ids = reference[task_id]
    common = min(len(ids), len(ref_ids))
    first_diff = next((i for i in range(common) if ids[i] != ref_ids[i]), None)
    records.append({
        "id": task_id,
        "reference_tokens": len(ref_ids),
        "compact_tokens": len(ids),
        "exact": ids == ref_ids,
        "first_difference": first_diff,
        "compact_token_ids": ids,
    })

result = {
    "tasks": len(ordered_ids),
    "task_ids": ordered_ids,
    "order_file": args.order_file,
    "reference_run": args.reference_run,
    "data_sha256": hashlib.sha256(Path("data/gsm8k_128.jsonl").read_bytes()).hexdigest(),
    "reference_seconds": reference_seconds,
    "compact_seconds": compact_seconds,
    "speedup": reference_seconds / compact_seconds,
    "all_tokens_exact": all(record["exact"] for record in records),
    "records": records,
}
output_path.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2), flush=True)
