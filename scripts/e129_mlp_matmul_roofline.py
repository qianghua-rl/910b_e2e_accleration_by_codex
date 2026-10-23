"""Aggregate real E125/E126 MLP Matmul calls and compute bounded roofline facts."""
import argparse, csv, json, math, glob, bisect
from collections import defaultdict
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument('--ledger',default='results/e126_matmul_ledger.csv'); p.add_argument('--trace-dir',default='results/e125_profile_nd'); p.add_argument('--hardware',default='results/hardware.stdout.log'); p.add_argument('--output',default='results/e129_mlp_roofline.json'); p.add_argument('--report',default='results/e129_mlp_roofline.md'); a=p.parse_args()
trace_path=glob.glob(str(Path(a.trace_dir)/'**/ASCEND_PROFILER_OUTPUT/trace_view.json'),recursive=True)[0]
trace=json.loads(Path(trace_path).read_text())
# Link each hardware kernel to its async_npu source flow, then use the exact
# enclosing model-module annotation. This avoids nearest-kernel timestamp gaps.
sources={e['id']:e for e in trace if e.get('cat')=='async_npu' and e.get('ph')=='s'}
hardware_kernels={(e['pid'],round(float(e['ts']),2)):e for e in trace if e.get('ph')=='X' and 'Matmul' in e.get('name','')}
annotations=[e for e in trace if e.get('cat')=='cpu_op' and e.get('ph')=='X' and e.get('name','').startswith('decode_cost/mlp/')]
by_thread=defaultdict(list)
for e in annotations: by_thread[(e['pid'],e['tid'])].append(e)
for v in by_thread.values():v.sort(key=lambda e:float(e['ts']))
starts={k:[float(e['ts']) for e in v] for k,v in by_thread.items()}
rows=[]
for flow_end in (e for e in trace if e.get('cat')=='async_npu' and e.get('ph')=='f'):
    src=sources.get(flow_end['id'])
    if not src: continue
    host_pid=src['pid']; host_tid=src['tid']; ts=float(src['ts'])
    evs=by_thread.get((host_pid,host_tid),[]); i=bisect.bisect_right(starts.get((host_pid,host_tid),[]),ts)-1
    if i<0: continue
    ann=evs[i]
    if ts>float(ann['ts'])+float(ann['dur']): continue
    module=ann['name'].split('/mlp/',1)[1]
    layer=int(module.split('.layers.')[1].split('.')[0])
    op=module.rsplit('.',1)[-1]
    if op not in ('gate_proj','up_proj','down_proj'): continue
    kernel=hardware_kernels.get((flow_end['pid'],round(float(flow_end['ts']),2)))
    if not kernel: continue
    if op=='down_proj': m,n,k=16,5120,17408
    else: m,n,k=16,17408,5120
    rows.append(dict(op=op,M=m,N=n,K=k,dtype='bfloat16',weight_layout='ND(original)',kernel_name=kernel['name'],kernel_duration_us=float(kernel['dur']),layer=layer))
groups=defaultdict(list)
for r in rows:
    key=(r['M'],r['N'],r['K'],r['dtype'],r['weight_layout'])
    groups[key].append(r)
# Shape-matched BF16 microbenchmark from hardware_probe: M16/K5120/N17408 = 31.3387 TF/s.
shape_ceiling={(16,17408,5120):31.338718192443352}
measured_hbm_gbs=1256.6648423332472
measured_compute_tflops=238.91379134274277
out=[]
for key,rs in sorted(groups.items(),key=lambda x:(x[0][0],x[0][1],x[0][2])):
    m,n,k,dtype,layout=key; count=len(rs); flops=2*m*n*k; total_flops=flops*count
    times=[float(r['kernel_duration_us']) for r in rs]
    # ND BF16 lower-bound bytes: activation + weight + output, each 2 bytes.
    bytes_per=(m*k+k*n+m*n)*2
    total_bytes=bytes_per*count
    ideal_memory_ms=total_bytes/(measured_hbm_gbs*1e9)*1000
    ideal_compute_ms=total_flops/(measured_compute_tflops*1e12)*1000
    seconds=sum(times)/1e6
    tflops=total_flops/seconds/1e12 if seconds else None
    gbps=total_bytes/seconds/1e9 if seconds else None
    ceilv=shape_ceiling.get((m,n,k))
    intensity=flops/bytes_per
    memory_roof_tflops=intensity*measured_hbm_gbs/1000
    ops=defaultdict(list)
    for r in rs: ops[r['op']].append(float(r['kernel_duration_us']))
    out.append(dict(M=m,N=n,K=k,dtype=dtype,layout=layout,calls=count,
                    op_calls={op:len(v) for op,v in ops.items()},
                    op_kernel_us={op:sum(v) for op,v in ops.items()},
                    op_kernel_median_us={op:sorted(v)[len(v)//2] for op,v in ops.items()},
                    op_ideal_memory_ms={op:len(v)*bytes_per/(measured_hbm_gbs*1e9)*1000 for op,v in ops.items()},
                    flops_per_call=flops,total_flops=total_flops,kernel_us_sum=sum(times),
                    kernel_us_median=sorted(times)[len(times)//2],kernel_us_min=min(times),kernel_us_max=max(times),
                    bytes_lower_bound_per_call=bytes_per,bytes_lower_bound_total=total_bytes,
                    ideal_memory_floor_ms=ideal_memory_ms,ideal_compute_floor_ms=ideal_compute_ms,
                    time_above_roof_ms=sum(times)/1000-max(ideal_memory_ms,ideal_compute_ms),
                    effective_tflops=tflops,lower_bound_gbps=gbps,
                    arithmetic_intensity_flop_per_byte=intensity,
                    measured_memory_roof_tflops=memory_roof_tflops,
                    memory_roof_fraction=tflops/memory_roof_tflops,
                    measured_compute_ceiling_tflops=measured_compute_tflops,
                    measured_shape_ceiling_tflops=ceilv,
                    ceiling_fraction=tflops/ceilv if ceilv else None,
                    gap_reason=('shape-matched ceiling available' if ceilv else 'no shape-matched BF16 ceiling in local probe')))
hardware={'device':'Ascend910B3','properties':'see results/hardware.stdout.log','bf16_theoretical_tflops':None,'bf16_ceiling_reason':'No authoritative BF16 peak field was available in local device/properties output; do not infer it from cube_core_num.'}
result=dict(scope='E125 fixed replay, four decode steps, MLP-labelled calls only',matched_calls=len(rows),groups=out,
            total_kernel_ms=sum(g['kernel_us_sum'] for g in out)/1000,
            ideal_memory_floor_ms=sum(g['ideal_memory_floor_ms'] for g in out),
            excess_over_empirical_roof_ms=sum(g['time_above_roof_ms'] for g in out),hardware=hardware,
            source={'ledger':a.ledger,'trace':trace_path,'hardware_probe':a.hardware},limitations=[
                'Kernel CSV duration is device kernel interval, not complete wrapper or end-to-end time.',
                'Lower-bound bytes count one BF16 read of activation and weight plus one BF16 output write; cache reuse, NZ physical traffic and intermediate tensors are not measured.',
                'The 31.34 TFLOP/s same-shape probe used a contiguous [K,N] right operand, while Linear stores [N,K] and supplies its transpose view; it is a favorable local reference, not strict layout-matched ceiling.',
                'The empirical roofline combines a large-buffer copy bandwidth and a separate M=512 BF16 GEMM throughput; it is not a vendor theoretical peak.',
                'No optimization was enabled; no next candidate was implemented.'])
Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
lines=['# E129 MLP Matmul Performance Upper Bound','', 'Scope: E125 ND fixed replay, four decode steps, MLP-labelled Matmul kernels. FLOPs use `2*M*N*K`; BF16 element size is 2 bytes.','', '|M/N/K|calls|kernel ms|FLOPs/call (G)|effective TFLOP/s|AI FLOP/B|implied GB/s|memory roof TFLOP/s|efficiency|ideal memory ms|excess ms|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for g in out:
    ceiling='unavailable' if g['measured_shape_ceiling_tflops'] is None else f"{g['measured_shape_ceiling_tflops']:.2f}"
    frac='n/a' if g['ceiling_fraction'] is None else f"{g['ceiling_fraction']:.3f}"
    lines.append(f"|{g['M']}/{g['N']}/{g['K']}|{g['calls']}|{g['kernel_us_sum']/1000:.3f}|{g['flops_per_call']/1e9:.3f}|{g['effective_tflops']:.2f}|{g['arithmetic_intensity_flop_per_byte']:.2f}|{g['lower_bound_gbps']:.1f}|{g['measured_memory_roof_tflops']:.2f}|{g['memory_roof_fraction']:.1%}|{g['ideal_memory_floor_ms']:.2f}|{g['time_above_roof_ms']:.2f}|")
lines += ['', '## Same-shape Operation Breakdown', '|M/N/K|operation|calls|kernel ms|median us/call|', '|---|---|---:|---:|---:|']
for g in out:
    for op,count in g['op_calls'].items():
        lines.append(f"|{g['M']}/{g['N']}/{g['K']}|{op}|{count}|{g['op_kernel_us'][op]/1000:.3f}|{g['op_kernel_median_us'][op]:.2f}|")
lines += ['', '## Interpretation', '- The same arithmetic shape probe measured 31.34 TFLOP/s, with a contiguous [K,N] right operand. The model Linear weight is stored [N,K] and consumed through a transpose view, so this reference is favorable and not layout-matched.', '- The local BF16 M=512 GEMM measured 238.91 TFLOP/s and 512 MiB copy measured 1,256.66 GB/s. These provide empirical compute and bandwidth ceilings, not silicon specifications.', '- Official/theoretical 910B3 BF16 peak remains unavailable in the local properties export; cube-core count alone is insufficient to derive it.', '- The traffic model assumes one activation read, one full weight read and one output write per call. It is an estimated streaming floor, not measured HBM traffic; L2 residency may lower physical reads.', f"- Across all projections, {result['total_kernel_ms']:.2f} ms of Matmul kernels compares with a {result['ideal_memory_floor_ms']:.2f} ms one-pass streaming floor. The {result['excess_over_empirical_roof_ms']:.2f} ms gap is an opportunity budget, not proven implementation waste."]
for g in out:
    if g['ceiling_fraction'] and g['ceiling_fraction']<0.8: lines.append(f"- Candidate screen: `M={g['M']} N={g['N']} K={g['K']}` reaches {g['effective_tflops']:.2f} TFLOP/s, {g['ceiling_fraction']:.1%} of the measured reference; inspect this shape first, subject to exact wrapper timing.")
    if g['M']==16 and g['N']==17408:
        gate=g['op_kernel_median_us'].get('gate_proj'); up=g['op_kernel_median_us'].get('up_proj')
        lines.append(f"- Same-shape gate/up medians are {gate:.2f}/{up:.2f} us (ratio {gate/up:.2f}x), replicated layer-by-layer. This is an operation/position-correlated difference, not proof that call position or cache behavior causes it.")
        saving=max(0,gate-up)*g['op_calls']['gate_proj']/1000
        lines.append(f"- Next measurable hypothesis: the 1.95x gap may follow gate's MLP call position or its specific weight. A fixed-activation gate/up order crossover should distinguish those mechanisms; matching gate to up's current median bounds recoverable time at {saving:.2f} ms over these four steps.")
Path(a.report).write_text('\n'.join(lines)+'\n')
print(json.dumps({'groups':len(out),'report':a.report},indent=2))
