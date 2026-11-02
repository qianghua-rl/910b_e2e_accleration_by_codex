"""Opt-in whole-suffix causal conv with historical window and ragged commits."""
import torch
import torch.nn.functional as F
import e176_workflow_bench as workflow
from e176_history_runtime import HistoryRuntime


class VectorHistory(HistoryRuntime):
    def conv(self, x, state, weight, bias=None, activation=None):
        if x.shape[-1] == 1:
            return super().conv(x, state, weight, bias, activation)
        width = state.shape[-1]
        joined = torch.cat((state, x), dim=-1).to(weight.dtype)
        # HF's K-element saved window plus a new token produces two windows;
        # the first is already consumed. For T new tokens keep the last T.
        out = F.conv1d(joined, weight.unsqueeze(1), bias, groups=x.shape[1])
        out = F.silu(out[..., -x.shape[-1]:]).to(x.dtype)
        active = torch.arange(x.shape[-1], device=x.device)[None, :] < self.lengths[:, None]
        out = torch.where(active[:, None, :], out, 0)
        indices = self.lengths[:, None] + torch.arange(width, device=x.device)[None, :]
        committed = joined.gather(-1, indices[:, None, :].expand(-1, x.shape[1], -1))
        state.copy_(torch.where((self.lengths > 0)[:, None, None], committed, state))
        return out


if __name__ == '__main__':
    workflow.HistoryRuntime = VectorHistory
    raise SystemExit(workflow.run())
