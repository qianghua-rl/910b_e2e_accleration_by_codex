"""Offline safetensors header audit and analytic KV/recurrent state estimates."""
import collections
import json
import struct
from pathlib import Path

root = Path('model/qwen3.5-27b')
c = json.loads((root/'config.json').read_text())['text_config']
groups = collections.defaultdict(lambda:dict(tensors=0, elements=0, stored_bytes=0))
for file in sorted(root.glob('*.safetensors')):
    with file.open('rb') as f:
        size = struct.unpack('<Q',f.read(8))[0]
        header = json.loads(f.read(size))
    for name, info in header.items():
        if name=='__metadata__':
            continue
        group = 'vision' if 'visual' in name else 'mtp' if name.startswith('mtp.') else 'text'
        n=1
        for d in info['shape']:
            n*=d
        groups[group]['tensors']+=1
        groups[group]['elements']+=n
        groups[group]['stored_bytes']+=info['data_offsets'][1]-info['data_offsets'][0]
linear_layers = c['layer_types'].count('linear_attention')
full_layers = c['layer_types'].count('full_attention')
state = linear_layers*c['linear_num_value_heads']*c['linear_key_head_dim']*c['linear_value_head_dim']*2
conv = linear_layers*(2*c['linear_num_key_heads']*c['linear_key_head_dim']+
                      c['linear_num_value_heads']*c['linear_value_head_dim'])*c['linear_conv_kernel_dim']*2
kv_per_token = full_layers*2*c['num_key_value_heads']*c['head_dim']*2
rows=[]
for batch in [1,2,4,8,16,32,128]:
    for length in [128,512,2048,8192,32768]:
        rows.append(dict(batch=batch,length=length,recurrent_bytes=batch*state,
                         conv_bytes=batch*conv,kv_bytes=batch*length*kv_per_token,
                         total_cache_bytes=batch*(state+conv+length*kv_per_token)))
result=dict(checkpoint_groups=groups,recurrent_bytes_per_request=state,
            conv_bytes_per_request=conv,kv_bytes_per_token_per_request=kv_per_token,
            estimates=rows,limitations='Cache only; excludes allocator, activations, workspace and model weights')
Path('results/model_memory.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='estimates'},indent=2))
