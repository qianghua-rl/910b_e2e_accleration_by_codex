"""Capture the real E100 first-batch Qwen3.5 GDN prefill gate tensor."""
import argparse
import json
from pathlib import Path
import time

import torch
import torch_npu
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5RMSNorm,
    Qwen3_5RMSNormGated,
)

from agent_protocol import CONCISE_SYSTEM, TOOLS


TASK_IDS = [640, 1137, 214, 1030, 950, 428, 866, 175,
            1130, 445, 1161, 611, 108, 1288, 554, 249]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/gdn_cumsum_e100_batch16.pt")
    args = parser.parse_args()

    torch.set_num_threads(8)
    torch.manual_seed(42)
    data = [json.loads(line) for line in Path("data/gsm8k_128.jsonl").read_text().splitlines()]
    tasks = {int(row["id"]): row for row in data}
    tokenizer = AutoTokenizer.from_pretrained("model/qwen3.5-27b", padding_side="left")
    prompts = [tokenizer.apply_chat_template(
        [{"role": "system", "content": CONCISE_SYSTEM},
         {"role": "user", "content": tasks[task_id]["question"]}],
        tools=TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    ) for task_id in TASK_IDS]
    inputs = tokenizer(prompts, add_special_tokens=False, padding=True,
                       return_tensors="pt").to("npu")

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        "model/qwen3.5-27b", dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": "npu:0"},
    ).eval()

    def fused_rmsnorm(self, hidden_states):
        return torch_npu.npu_rms_norm(
            hidden_states, (1.0 + self.weight).to(hidden_states.dtype), self.eps,
        )[0]

    def fused_gated_rmsnorm(self, hidden_states, gate=None):
        normalized = torch_npu.npu_rms_norm(
            hidden_states, self.weight, self.variance_epsilon,
        )[0]
        return torch_npu.npu_swiglu(torch.cat((gate, normalized), dim=-1))

    for module in model.modules():
        if isinstance(module, Qwen3_5RMSNorm):
            module.forward = fused_rmsnorm.__get__(module, Qwen3_5RMSNorm)
        elif isinstance(module, Qwen3_5RMSNormGated):
            module.forward = fused_gated_rmsnorm.__get__(module, Qwen3_5RMSNormGated)

    captured = {}
    for module in model.modules():
        original = getattr(module, "chunk_gated_delta_rule", None)
        if not callable(original):
            continue

        def capture(query, key, value, g, beta, *positional,
                    _original=original, **keyword):
            if not captured and query.shape[1] > 1:
                captured["g"] = g.detach().float().cpu().contiguous()
                captured["shape"] = list(g.shape)
                captured["stride"] = list(g.stride())
                captured["dtype"] = str(g.dtype)
            return _original(query, key, value, g, beta, *positional, **keyword)

        module.chunk_gated_delta_rule = capture

    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=1, do_sample=False, use_cache=True,
        )
        torch.npu.synchronize()
    if "g" not in captured:
        raise RuntimeError("No multi-token GDN prefill call was captured")

    token_ids = generated[:, inputs["input_ids"].shape[1]:].squeeze(-1).cpu().tolist()
    record = {
        "g": captured["g"],
        "task_ids": TASK_IDS,
        "token_ids": token_ids,
        "shape": captured["shape"],
        "stride": captured["stride"],
        "dtype": captured["dtype"],
        "elapsed_seconds": time.perf_counter() - started,
        "prompt_lengths": inputs["attention_mask"].sum(-1).cpu().tolist(),
        "padded_prompt_length": int(inputs["input_ids"].shape[1]),
        "protocol": "concise",
        "attention": "sdpa",
        "fused_rmsnorm": True,
        "fused_gated_rmsnorm": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(record, output)
    print(json.dumps({key: value for key, value in record.items() if key != "g"}))


if __name__ == "__main__":
    main()
