"""Quality-track history append: ragged chunk recurrence, E248 causal conv."""
import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
import e176_workflow_bench as workflow
from e248_vector_history_conv import VectorHistory


class ChunkHistory(VectorHistory):
    def recurrent(self, query, key, value, g, beta, initial_state, output_final_state,
                  use_qk_l2norm_in_kernel=False):
        if query.shape[1] == 1:
            return super().recurrent(query, key, value, g, beta, initial_state,
                                     output_final_state, use_qk_l2norm_in_kernel)
        live = torch.arange(query.shape[1], device=query.device)[None, :] < self.lengths[:, None]
        # Zero beta and log-decay after each row's true end: padded steps apply
        # identity to its state. Zero queries enforce zero inactive output.
        q = torch.where(live[:, :, None, None], query, 0)
        k = torch.where(live[:, :, None, None], key, 0)
        v = torch.where(live[:, :, None, None], value, 0)
        decay = torch.where(live[:, :, None], g, 0)
        strength = torch.where(live[:, :, None], beta, 0)
        out, final = hf.torch_chunk_gated_delta_rule(q, k, v, decay, strength,
            chunk_size=64, initial_state=initial_state, output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
        out = torch.where(live[:, :, None, None], out, 0)
        # HF commits native cache dtype once at the layer boundary.
        return out, final.to(initial_state.dtype) if final is not None else None


if __name__ == '__main__':
    workflow.HistoryRuntime = ChunkHistory
    raise SystemExit(workflow.run())
