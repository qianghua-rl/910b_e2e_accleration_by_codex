# 可迁移的优化路径（从 910B 经验提炼）

本文档把 910B 上的优化实验整理成与硬件无关的原则和阶段表。只有 910B 有实测数据；其他卡的内容都是推断，需要在对应硬件上验证。

数据来源：`HANDOFF.md`（实验记录）、`results/excluded_directions.md`（否决方向）、`results/REPORT.md`；H100 对照来自 `/remote-home/S45160/work/e2e_acceleration_by_claude`（`README.md`、`OPTIMIZATION_LOG.md`）。

## 1. 910B 上实际走过的路径

基线 8365 s → 当前最优 410 s（约 20×），同一批 128 题，正确数 116 → 118。

| 阶段 | 改动 | makespan | 收益来源（实测） | 来源 |
|---|---|---|---|---|
| 0 | 串行 B=1 HF（128 题；127 题为 8267 s） | 8365 s | — | `results/gsm8k_128_baseline_hf` |
| 1 | batch16 + 融合 RMSNorm 等 | 1180 s | 并行化，约 7× | `results/b16_original_opt` |
| 2 | 自写连续批处理（slot 制，KV 与 GDN 状态按 slot 存放） | 755 s | 去掉 batch 内已完成行的空算，最大的一步 | `ablations/continuous_batching/repeat_base_r2` |
| 3 | GDN chunk prefill 替换 torch 回退实现 | 588 s | prefill 中约 66% 花在回退路径，同代码 A/B 约 −22% | `repeat_fast_r2` |
| 4 | decode 步捕获成 NPU Graph（共享内存池） | 500 s | decode 墙钟约 90% 是 host 派发开销 | pg1 graph 499.7 s |
| 5 | GDN v2：与状态无关的部分批量预计算 | 470 s | prefill −19%（149.1 → 120.9 s），两组不重叠 | `HANDOFF.md` 第 206–209 行 |
| 6 | elem-conv：深度卷积改 mul+sum | 410 s | 同配置 A/B 479 s → 410 s（约 −14%）；device 时间 −23% | `scripts/run_abba_full128_elemconv.sh` |

注意：阶段 6 的 device 时间下降 23%，但单独看墙钟收益不明显，只有在 graph 之后才体现出来。这正是“先判断瓶颈类型”的原因。

### 否决或证伪的方向（910B）

- 24 slot：OOM。
- sync-free（去掉 RMSNorm / l2norm / recurrent step 的 host 同步）：慢约 4%，单元级 diff 为 0，说明 host 同步次数不是限制因素。
- shift-conv（decode 卷积改 shifted multiply-add）：慢约 2%，且不逐位一致。
- chunk 128：未采用。
- graph bucket 128：收益低于 2% 门槛。
- 旧工作区（E001–E279）中的否决项，见 `excluded_directions.md`，包括融合 RMSNorm 全路径（质量守卫 121/128 未通过）、GDN 投影打包、MegaGDN PTO prefill 等。

## 2. 与硬件无关的原则

括号里是 910B 上的具体表现。

1. **测量方法**
   - 用墙钟 step 时间判断优化是否有效，而不只看 profiler 的 device 时间和。
   - 同卡 ABBA 交替测量，两组的范围要不重叠才算有效。
   - 逐题比较正确集合，不只比总数（elem-conv 的 final_text 与非 elem 不一致，正确数相同但正确集合未比较，这一点还没完成）。
   - 检查同配置是否确定性。
   - 噪声约 ±10 s，小于 2% 的收益不可判断。
2. **调度层收益通常最大**：连续批处理（slot 复用，decode 总步数下降）比 kernel 级优化大得多。910B 上约 1.6×，H100 上 7.8×。前提是模型状态（KV、recurrent / conv state）能按 slot 管理。H100 的经验还表明，架构级改动（引擎重写）之后 kernel 收益很快递减。
10. **成本驱动因素要先测清**：减少 token 数不一定减少时间，成本可能在调用次数（eager 调用、host 同步）。H100 上 prefix reuse 减少 89% 的 prefill token，墙钟收益为 0。
11. **关键路径决定 makespan**：批处理后，最长的 agent 链（串行轮数）决定总时间，单步的平均时间不再重要。H100 Phase 2 后期即是如此。
12. **理论下限（SOL）要算**：H100 用 "权重字节 / 带宽 + 每 token 状态字节 / 带宽" 估算下限，估得 7.1 s，实测 8.36 s，因此能判断还剩多少空间。框架的探针应当输出同样的 SOL 估计。
13. **功耗墙会改变结论**（H100 实测）：大 GEMM 下频率降到约 1.6 GHz，孤立 micro-benchmark 的结论不一定成立。
14. **能写 kernel 不等于比库快**（H100 实测）：精确累加的 FP8 GEMM 上，Triton 和 CUTLASS 都追不上 cuBLAS。
3. **先判断瓶颈类型，再选优化**：
   - host 派发受限（launch-bound）→ graph 捕获、减少算子数、融合；
   - device 受限 → kernel 优化、矩阵乘调优；
   - 用“device 时间和”与“墙钟”的差距判断。910B 上 decode 是 host 受限，所以 elem-conv 单独不起作用。
4. **去掉框架的回退路径**：用专用的线性注意力 / GDN kernel 替代 torch 参考实现。
5. **状态无关计算尽量批量化**（阶段 5）。
6. **显存决定并发上限**：权重、KV、状态的预算算清楚后，slot 数就固定了（910B 上为 16；24 会 OOM）。
7. **graph 捕获的通用条件**：静态 shape（按长度分 bucket）、静态输入缓冲、共享内存池、预先捕获后再 prefill。
8. **allocator 碎片**：需要可扩展段之类的配置，否则图模式下 prefill 可能 OOM。
9. **栈相关结论要重新判断**：旧工作区记录“Torch / ACL graph capture 在本栈不可用”（E010、E098、E103），而新工作区的 graph 已经可用。同一个方向在不同软件栈上结论可能相反，记录时必须带上栈的版本信息。

### 910B 专有内容（换卡后需要重新验证，不能直接搬）

- Python float 标量触发 host sync（`aten::_local_scalar_dense`）。
- 高级索引很慢。
- ACL graph 的 API。
- HCCL 通信。
- expandable_segments 的具体设置。

## 3. 映射到其他显卡

H100 仓库（`/remote-home/S45160/work/e2e_acceleration_by_claude`，Qwen3.5-27B，同一 127 题与同样的解码规则）已经给出 NVIDIA 侧的实测路径。

| 阶段 | 910B（实测） | H100（实测） | AMD |
|---|---|---|---|
| 起点 | 串行 B=1 HF | 串行 HF eager，1470.6 s | 未测 |
| 去掉 Python dispatch | decode 捕获 NPU Graph：588 → 500 s（约 −15%） | StaticCache + CUDA graph：1673.6 → 660.8 s（2.5×，含 task 423，与上一行基线不同） | 未测 |
| 连续批处理 + slot | 1180 → 755 s（约 1.6×） | 474.8 → 60.8 s（**7.8×**），是 H100 上最大的一步 | 未测 |
| 引擎重写（自写 forward + 调度器） | 同上（未拆分） | 18.19 → 10.95 s（1.7×；README 称 "约 2×"） | 未测 |
| GDN / 线性注意力 kernel | 自写 chunk prefill，同代码 A/B 约 −22% | Triton / 自写 chunk，融合递推与输出，持续有效 | 需确认 ROCm 上 Triton 支持 |
| 低精度 GEMM | 未做（缺端到端验证的路径） | FP8 per-tensor，28.6 s 阶段（约 1.25×） | 需确认 |
| 后期 kernel 改动 | elem-conv 约 −14%（需 graph 后才体现） | 几十个 kernel 合计约 7%，单项 <1% 时即到墙 | 同上 |
| host 同步检查 | sync-free 证伪（无收益） | 未作为瓶颈列出 | 同上 |
| prefix / 状态复用 | 未单独列为收益项 | prefill token −89%，**墙钟收益为 0**（成本在 eager 调用次数，不在 token 数） | 未测 |

要点：

- **两边都证明了 "slot 复用 + 连续批处理" 是最大收益的一步**，前提都是状态能按 slot 管理。这一条可以保留为跨平台结论。
- **H100 的后期收益递减很快**：引擎重写之后，几十个 kernel 只剩约 7%。910B 目前还在 kernel 和 graph 阶段，尚未到这个位置。
- **H100 上 decode 是大 batch 下的功耗墙问题**（FP8 大 GEMM 下约 630–690 W，频率降到约 1.6 GHz），不是 host 派发问题。这与 910B 的 host 受限结论相反，正说明瓶颈类型必须由探针测出，不能预设。
- **H100 的 Amdahl 教训**：优化前先确认成本驱动因素是 "次数" 还是 "数量"。prefix reuse 减少了 89% 的 prefill token，收益为 0。910B 的阶段表里也应当加上这一项检查。

AMD 栏仍然全部是推断。

## 4. 框架设计：探针 → 候选 → 门控 → 记录

“自进化”的合理实现是：不固定优化顺序，先用探针测出当前卡的瓶颈，再决定下一步做什么。

1. **显存探针**：测权重、KV、状态的占用，算出最大 slot 数。
2. **瓶颈探针**：比较 device 时间和墙钟 step 时间。差距大 → 优先 graph / 减算子；差距小 → 优先 kernel。
3. **图捕获探针**：能否捕获 decode 步，能否共享内存池（对应 `scripts/test_acl_graph.py`）。
4. **host 同步探针**：统计 decode 中的同步次数（`aten::_local_scalar_dense` 计数）。注意 910B 上这一项的收益已被证伪，探针只作为诊断信息，不直接触发优化。
5. **kernel 可用性探针**：专用 GDN / 线性注意力 kernel 是否可用，能否替代回退实现。
6. **门控**：每个候选都做 ABBA 验证，收益门槛 2%，并检查正确集合和确定性。

### 阶段注册表（核心数据结构，建议格式）

每个候选记录以下字段：

- `id`、`name`
- `preconditions`：前置条件（如“状态可按 slot 管理”）
- `probe`：对应的探针
- `expected_gain`：预期收益及依据
- `gate`：ABBA 门控条件
- `status`：accepted / rejected / unverified
- `stack`：软件栈版本（驱动、CANN / CUDA、框架）
- `evidence`：结果目录或日志路径

910B 上的 `HANDOFF.md` 实验记录和 `excluded_directions.md` 可以直接作为这套注册表的种子数据。否决项也要写进注册表，避免框架重复搜索已证伪的路径（sync-free、shift-conv、24 slot 等）。

## 5. 局限

- 只有一个模型（Qwen3.5-27B，混合架构）、一个负载（GSM8K 工具 agent，greedy）、一种卡。“适用于所有显卡”目前是假设，不是结论。
- 部分结论依赖模型结构：48 层 GDN + 16 层全注意力，所以 GDN kernel 占的比重大。纯 Transformer 模型的优化顺序会不同。
- 910B 的 vLLM 对照没有完全对等：vLLM 用 aclgraph，我们的连续批处理在 TP=2 下是 eager。对比时要说明。
- elem-conv 的数值结果与非 elem 不完全一致，需要补做正确集合的逐题比较。
- **H100 仓库中的 Ascend 报告（`results/H100_VS_ASCEND_COMPARISON.md`、`result_report.md`）不作为依据**，其中的数字（如 3466.2 s、656.4 s）已被用户确认为不可信，本文档不引用。
- 910B 的数字只以本仓库为准。串行基线来自 `results/gsm8k_128_baseline_hf`：128 题 8365.0 s（116/128）；去掉 task 423 后 127 题为 8267.1 s（116/127，accuracy 0.913）。task 423 在 910B 上答错，耗时 97.9 s。
- 当前最优 410 s（118/128）是 128 题口径。与 H100 的 127 题结果比较时，统一标注为 128 题，不再重算。

## 6. 建议的下一步

1. 把 `HANDOFF.md` 的实验整理成上面的阶段注册表。
2. 在第二种卡（最好是 NVIDIA）上跑同一套探针，看哪些结论能迁移、哪些失效。
3. 补做 elem-conv 与非 elem 的逐题正确集合比较。
