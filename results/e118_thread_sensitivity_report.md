# E118 thread sensitivity: final diagnostic conclusion

Objective: fixed E100 prompts/decode trajectory, interleaved threads8/1/4 with8 controls; check logits, host wall and device work before deciding on runtime implementation.

## Evidence

Unprofiled replay (`e100_thread_sensitivity_analysis.json`):17 trials, fresh caches, identical12 forced tokens per each of16 requests; original logits collected before forcing. Every trial logits bit-exact/finite and argmax/replay exact. Four paired samples each of threads1/4 showed inconsistent sign; all adjacent8 controls drifted >2% (2.37–16.34%). No stable threading improvement. Forward NPU events include submission gaps; their similarity to host wall alone cannot establish compute-bound execution.

Supplemental fixed replay profiles, four identical early decode steps, sequential8→1→4→8, original E100 model configuration:

| Condition | Computing union ms | Stage ms | Free ms | Lowered kernel count |
|---|---:|---:|---:|---:|
| 8a | 494.270 | 712.701 | 218.431 | 17416 |
| 1 | 494.450 | 731.701 | 237.251 | 17416 |
| 4 | 494.896 | 696.962 | 202.066 | 17416 |
| 8b | 495.586 | 742.605 | 247.019 | 17416 |

Kernel counts/fixed tokens are identical across all profiles. Computing union range is only0.27%; free time range202–247ms. Repeated8 control stage differs by4.2% while compute differs0.27%; most of that control difference is free time. Thus in these sampled traces temporal variation is principally in execution gaps, not kernel work. This does NOT retroactively assign every unprofiled fluctuation to host CPU, nor prove all gaps removable. The profiles are separate runs and profiler perturbs submission. Thread4's single profile improvement is not a validated speedup.

CPU affinity is captured in profile metadata. CPU frequency/governor nodes are not exposed; telemetry failure retained in `e118_host_telemetry.json`. No claim about DVFS, contention or exact gap cause is justified. No pure launch latency, memory-bandwidth limit or device-only bottleneck was established.

## Decision

Keep E100 threads8 and original runtime. Do not implement graph/continuous scheduling or advertise threading savings from these results. The experiment separates stable kernel work from variable gap costs sufficiently to reject a FLOPs-only explanation of the observed timing drift, but does not isolate host-submit versus dependency/driver scheduling within gaps. Any further runtime work should identify one specific recurring gap with host/runtime flow evidence or establish a stable capture-eligible segment; it must differ from failed E098/E103 and have a complete-wrapper budget. The overall E100 optimization target is unchanged. This completes the requested threading diagnostic, not the broader optimization project.

Artifacts: `e100_thread_sensitivity_initial.json`, `e100_thread_sensitivity.json`, `e100_thread_sensitivity_analysis.json`, `e118_thread_kernel_cost.json`, four `profile_e118_threads_*` traces/logs. Analyzer: `scripts/analyze_thread_kernel_cost.py`. No default model changes, no full128 speedup/quality claim.
