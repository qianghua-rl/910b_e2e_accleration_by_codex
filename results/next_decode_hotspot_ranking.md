# Post-NZ Decode Hotspot Ranking

Reference: E100 BF16/SDPA batch16, four fixed replay decode steps, no optimization enabled. Device times are profiler attribution, not endpoint wall time; nested intervals are not summed across categories.

| rank | hotspot | device time / 4 steps | evidence | next hypothesis |
|---:|---|---:|---|---|
| 1 | MLP Matmul | 157.407 ms, 768 kernels | `results/e100_decode_attribution.json` | Any change must reduce complete model decode, since NZ local gain did not transfer |
| 2 | GDN convolution/transdata and recurrent path | convolution family 116.823 ms; recurrent Mul 43.200 ms + ReduceSum 14.776 ms + copies 9.113 ms | E115 attribution | Separate recurrent state passes or convolution path with exact state/logit replay |
| 3 | Attention/GDN projections | 71.687 ms, 1,216 Matmul kernels | E115 attribution and E126 ledger | Identify one projection shape with stable wrapper budget before changing layout |
| 4 | Other copies/layout/normalization | InplaceCopy 13.670 ms, RmsNorm 7.815 ms, Cat 6.233 ms, Silu 5.434 ms | E115 kernel attribution | Only pursue a stage with a measured complete-wrapper reduction |
| 5 | LM head | 8.448 ms, 4 Matmul kernels | E115 attribution | Low share; not first candidate |

Runtime gaps remain a separate hypothesis: E118 found computing union stable within 0.27% while free time varied 202–247 ms, and E119 located host preparation/dispatch and two scalar barriers without proving removable cost. Scheduler/sampling/postprocess were about 3% in E123. No graph or async change is currently justified.

The NZ route is closed as a verified negative: local MLP wrapper speedup is real, but complete generation was slower by 3.718% and 7.324% in E124. The next experiment must state one measurable mechanism and an endpoint gate before implementation.
