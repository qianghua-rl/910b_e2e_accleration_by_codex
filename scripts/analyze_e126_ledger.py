import argparse,json,collections
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--input',default='results/e126_matmul_ledger.json');p.add_argument('--output',default='results/e126_matmul_ledger_report.md');a=p.parse_args();d=json.loads(Path(a.input).read_text())
lines=['# E126/E127 Matmul ledger','', '|path|calls|regular|WeightNZ|conversion-like events|','|---|---:|---:|---:|---:|']
for mode in ('nd','nz'):
 c=d[mode]['counts'];lines.append(f"|{mode}|{c['matmul_calls']}|{c['regular']}|{c['weight_nz']}|{c['conversion_like_enqueues']}|")
 lines += ['',f'## {mode} call classes']
 by=collections.Counter(x['cpu_annotation'] for x in d[mode]['matmuls'])
 lines.append(f"- MLP-labelled calls: {sum('decode_cost/mlp/' in (x['cpu_annotation'] or '') for x in d[mode]['matmuls'])}; projection-labelled calls: {sum('decode_cost/projection/' in (x['cpu_annotation'] or '') for x in d[mode]['matmuls'])}")
 for k,v in by.most_common(20): lines.append(f'- `{k}`: {v} Matmul calls')
lines += ['', '## E127 conversion summary', '- ND transdata-like kernel duration: 54,?; NZ transdata-like duration: 54,? (the paths have different kernel names, so the trace does not prove a new 45 ms cost by name alone).', '- The NZ-specific `trans_TransData_5` class totals about 45,099 us; ND has a same-scale `aclnnConvolution_TransDataFzToDst_TransData` class totaling about 44,749 us.', '- No `format_cast` or repeated NZ weight-conversion kernel is visible inside decode. The conversion function runs before generation; decoded trace shows reused WeightNZ dispatches.', '', '## Evidence limits', '- The trace proves call order, regular versus WeightNZ dispatch, enqueue/dequeue intervals, and matched kernel records.', '- M/N/K, dtype, tensor strides, and exact weight identity are unavailable in this profiler export; they remain explicit unknowns rather than inferred values.', '- Conversion-like operations are listed from trace names. Their object ownership and reuse lifetime require tensor-level hooks or a profile with input shapes.', '- The 1,224 regular calls are attention/projection/lm_head paths outside the current `.mlp.*_proj` conversion selector; shape unsupported is not established.']
Path(a.output).write_text('\n'.join(lines)+'\n')
