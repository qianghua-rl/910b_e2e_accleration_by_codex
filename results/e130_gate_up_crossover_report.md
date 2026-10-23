# E130 gate/up crossover

E130-A used the captured E100 real BF16 activations from layers 8 and 55, shape `[16,1,5120]`, original ND weights, and the original gate -> SiLU -> up -> down MLP semantics. Gate and up weights were tested in both orders on the same activation. Each layer used 100 same-activation paired order crossovers, alternating order, continuous stream submission with one final synchronization, and per-projection NPU events. All outputs, strides and formats were checked.

| layer | gate first median us | gate second median us | up first median us | up second median us | gate/up wrapper order median ms | up/gate wrapper order median ms | paired delta ms (up/gate - gate/up) | 95% bootstrap CI | exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 8 | 203 | 161 | 205 | 187 | 0.6143 | 0.5874 | -0.0243 | [-0.0293, -0.0219] | yes |
| 55 | 201 | 157 | 200 | 175 | 0.5925 | 0.5777 | -0.0167 | [-0.0179, -0.0129] | yes |

The difference follows execution position: both weights are slower in the first projection slot, and both become faster in the second slot. This held across 100 same-input paired crossovers with bootstrap 95% CIs excluding zero. The wrapper crossover is stable in this continuous-submission experiment, but it is only a standalone two-projection MLP measurement. Both weights have stride `[5120,1]`, NPU format `2`, and data pointers aligned modulo 512.

An additional controlled reuse test read/wrote a 512 MiB scratch buffer and synchronized before each two-Matmul scenario, exceeding the reported 192 MiB L2. After this eviction, the first Matmul took `377–389 us` for either weight; the second different-weight Matmul took `182–188 us`; repeating the same weight cut the second call to `64–66 us` with exact duplicate output. This supports both a large first-slot/stream-start cost and an additional same-weight cache-reuse effect. It does not show that distinct gate/up weights share those cached lines.

The existing full-model trace adds context: gate projection is preceded by RMSNorm; up projection is preceded by SiLU. Host Linear spans are similar (layer8 `120/106 us`, layer55 `116/101 us`), both Matmuls use the same visible `MatMulV2` kernel name, and the trace shows no explicit host sync between the calls. All captured NPU async flows used stream 47, so there is no cross-stream overlap among these profiled calls. The 910B trace does not expose physical cache hit data or complete operand format metadata, leaving the full-model gate/up difference only partly explained.

The earlier `37.88 ms` estimate was invalid as a recoverable budget: it subtracted isolated gate and up medians without matching the actual order-dependent wrapper context. The E130 crossover difference is about `16.7–24.3 us` per two-projection MLP. A simple 64 layers x 4 steps extrapolation is only about `4.3–6.2 ms`, or `0.4–0.7%` of the corresponding four-step stage window, before checking full-model effects. In the actual E125 profile, layer8 gate/up MatMulV2 kernels were `300.2/158.3 us` and layer55 `277.6/154.6 us`. The full-model position effect is larger than the standalone order-crossing saving, so the crossover result cannot be used to claim those `37.88 ms` are recoverable.

No full Decode A/B was run: the visible NPU reported only about `44 GB` free, while the E100 model had `54.7 GB` allocated before generation and `62.1 GB` peak. The measured crossover ceiling is below the project's 2% candidate screen, so the additional model load is not justified by this evidence.

Decision: E130-B identifies a stable position-correlated effect, but no implementation factor has been isolated and no full Decode gain is established. Do not change MLP code or claim endpoint recovery. Stop this candidate under the 2% gain ceiling. A future candidate would need a new mechanism beyond simple call reordering, then pass fixed token replay and complete Decode timing.

Artifacts: `results/e130_gate_up_crossover.json`, `results/e130_same_weight_reuse.json`, `results/e130_weight_cache_control.json`; scripts: `scripts/e130_gate_up_crossover.py`, `scripts/e130_same_weight_reuse.py`, `scripts/e130_weight_cache_control.py`.
