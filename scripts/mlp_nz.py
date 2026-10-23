"""Opt-in persistent BF16 NZ MLP weights; sequential conversion limits peak."""
import time
import torch
import torch_npu

def convert_mlp_nz(model):
    torch.npu.config.allow_internal_format=True
    torch.npu.synchronize()
    start=time.perf_counter();before=torch.npu.memory_allocated();count=0
    previous_peak=torch.npu.max_memory_allocated()
    torch.npu.reset_peak_memory_stats()
    for name,module in model.named_modules():
        if '.mlp.' not in name or not name.endswith(('gate_proj','up_proj','down_proj')):
            continue
        weight=module.weight
        if weight.dtype != torch.bfloat16 or weight.device.type!='npu':
            raise ValueError(f'Unexpected weight: {name}')
        free,_=torch.npu.mem_get_info()
        if free < weight.numel()*weight.element_size()+256*1024**2:
            raise RuntimeError(f'Insufficient temporary NZ conversion space at {name}')
        converted=torch_npu.npu_format_cast(weight.detach(),29)
        torch.npu.synchronize()
        if torch_npu.get_npu_format(converted)!=29:
            raise RuntimeError(f'NZ format not retained: {name}')
        module.weight=torch.nn.Parameter(converted,requires_grad=False)
        del weight,converted
        count+=1
    torch.npu.synchronize();torch.npu.empty_cache()
    if count == 0:
        raise RuntimeError('No MLP weights matched; candidate was not activated')
    return dict(converted_weights=count,seconds=time.perf_counter()-start,
                allocated_before=before,allocated_after=torch.npu.memory_allocated(),
                preparation_peak_bytes=torch.npu.max_memory_allocated(),previous_peak_bytes=previous_peak)
