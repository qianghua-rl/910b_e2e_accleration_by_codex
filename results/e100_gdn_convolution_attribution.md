# E100 GDN convolution-family read-only check

Source: `results/profile_e100_decode_attributed/06f2b62264eb_155092_20261022124514418_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv` and `trace_view.json`. Scope is the same four early decode steps; this is not steady-state or end-to-end latency attribution.

| Kernel family | Count | Device time |
|---|---:|---:|
| `aclnnConvolution_Conv2dWithFlag_Conv2D` | 192 | 55.776 ms |
| `aclnnConvolution_TransData_TransData` | 384 | 7.947 ms |
| `...TransDataToFzWithoutGroup_MemSet` | 192 | 2.193 ms |
| `...TransDataToFzWithoutGroup_TransData` | 192 | 5.103 ms |
| `...TransDataFzToDst_MemSet` | 192 | 1.065 ms |
| `...TransDataFzToDst_TransData` | 192 | 44.739 ms |
| **Convolution family total** | **1,344** | **116.823 ms** |

There are 1,152 TransData/MemSet kernels totaling 61.048 ms. Timeline inspection shows the conversion kernels occur in a repeated group immediately before/after each Conv2D. This supports a conversion-related cost hypothesis, but does not prove duplication: the trace has no semantic contract showing that a conversion can be omitted or reused. Device names also do not carry GDN layer/module identity, so ownership remains unconfirmed beyond the convolution call family.

Decision: no implementation or A/B is justified yet. A next experiment requires an exact layout contract and a complete wrapper/logit comparison, plus a credible >=2% end-to-end upper bound. Local conversion or Conv2D speed ratios must not substitute for that gate.
