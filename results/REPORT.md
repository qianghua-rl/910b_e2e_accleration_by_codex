# 对比报告：自研 CB 引擎 vs vLLM-Ascend（Qwen3.5-27B，TP=2）

## 1. 结论

- **TP=2 是本报告的主要结果。** 单卡测评因反复 OOM 和输出截断无法给出可比结果，因此迁移到双卡 TP=2，两边统一 12 并发、首轮 6144、终轮 2048。
- **准确率**：在 BBH、MATH-500、GSM8K longctx 上两边基本一致（差距 < 1.5 个百分点）。GPQA 差 2 个百分点（CB 67.2% vs vLLM 69.2%）。AIME 2025 两边都是 60.0%。AIME 2024 上 CB 为 66.7%，vLLM 为 80.0%，差 4 题；差异题已逐题核对，主要来自生成长度分叉和上下文拒绝（见第 6 节）。30 题的标准误约 8–9 个百分点，这一差距仍不能定性。
- **速度**：TP=2 下 CB 在 AIME、BBH、GPQA、MATH-500 上快 1.16–1.45 倍；长上下文 GSM8K 上 CB 慢约 11%，是唯一反向的数据集。
- **TP=2 graph decode（2026-11-10 完成）**：开启 `--graph-decode` 后，CB 在五个数据集上的 makespan 均明显下降（每步解码约 2.5 倍加速），相对 vLLM 的速度比为 AIME 2024 2.8×、AIME 2025 3.0×、BBH 3.5×、GPQA 3.7×、MATH-500 2.8×（见第 4.2 节）。准确率与 eager 相比变动在 ±9 题以内（GPQA 142 对 133，AIME 2024 19 对 20）；与 vLLM 相比，BBH、GPQA、MATH-500、AIME 2025 持平或更高，AIME 2024 少 5 题（19 对 24）。注意：CB graph 与 eager 的逐题文本一致性尚未比对，因此准确率差异不能归因于 graph 本身。
- **速度比值的上限受限于配置**：早先的 1.16–1.45× 是 CB eager 对 vLLM aclgraph 的结果，已被 graph 结果取代，仅作为历史记录保留。
- **主要质量问题是输出截断**：MATH-500 和 GPQA 中约 18%（CB 91/500）和 18%（CB 35/198）的题没有 `\boxed{}`，直接计为错误。

## 2. 实验设置（TP=2）

| 项目 | 自研 CB（TP=2） | vLLM-Ascend（TP=2） |
|---|---|---|
| 模型 | Qwen3.5-27B | Qwen3.5-27B |
| 并发 | 12 slot | `max_num_seqs` 12 |
| 首轮 / 终轮输出上限 | 6144 / 2048 | 6144 / 2048 |
| 上下文上限 | t_max 8192（longctx 为 16384） | `max_model_len` 8192（longctx 为 16384） |
| Decode | **eager**（`graph_decode: False`） | aclgraph（piecewise） |
| Prefix caching | 无（每轮重新 prefill） | 无 |
| 评分 | 离线评分器（`scorer_mc` / `scorer_math`） | 离线评分器 |

数据：`results/continuous_batching_tp2/`（自研）、`results/bench_vllm_tp2/`（vLLM）。

## 3. 准确率（TP=2，离线评分器重算）

| 数据集 | 题数 | 自研 CB | vLLM | 无 `\boxed{}` CB / vLLM |
|---|---|---|---|---|
| AIME 2024 | 30 | 20 / 30 = **66.7%** | 24 / 30 = **80.0%** | 9 / 5 |
| AIME 2025 | 30 | 18 / 30 = **60.0%** | 18 / 30 = **60.0%** | 10 / 11 |
| BBH | 1350 | 1158 / 1350 = **85.8%** | 1163 / 1350 = **86.1%** | 12 / 14 |
| GPQA Diamond | 198 | 133 / 198 = **67.2%** | 137 / 198 = **69.2%** | 35 / 33 |
| MATH-500 | 500 | 455 / 500 = **91.0%** | 451 / 500 = **90.2%** | 91 / 90 |
| GSM8K longctx | 128 | 122 / 128 = **95.3%** | 122 / 128 = **95.3%** | 1 / 1 |

注：GSM8K longctx 使用 16384 上下文。其他数据集为 8192。无 `\boxed{}` 的题数按 `prediction` 字段为空计算。

## 4. 耗时（TP=2 makespan）

| 数据集 | 自研 CB | vLLM | vLLM / CB |
|---|---|---|---|
| AIME 2024 | 1845 s | 2140 s | 1.16× |
| AIME 2025 | 2081 s | 2525 s | 1.21× |
| BBH | 14461 s | 20666 s | 1.43× |
| GPQA Diamond | 9542 s | 13815 s | 1.45× |
| MATH-500 | 9063 s | 11546 s | 1.27× |
| GSM8K longctx（16384），CB eager（基线） | 1554 s | 1403 s | 0.90× |
| GSM8K longctx（16384），CB graph decode | 1059 s | 1403 s | 1.32× |

GSM8K longctx 的时间分解（CB eager 基线）：prefill 共 256 次调用，耗时 788 s（占 makespan 的 51%），decode 765 s，4341 步。这说明长上下文的主要开销在 prefill，且 prefill 是串行执行、会阻塞 decode。

GSM8K longctx 开启 TP=2 graph decode 后（`results/ablations/continuous_batching_tp2/gsm8k_longctx_128_graph_1109_1028/`）：prefill 共 256 次调用，耗时 744 s，decode 303 s，4387 步（步数略多，原因未查）。makespan 从 1554 s 降到 1059 s（−32%），decode 时间降低约 60%。准确率 122/128，与 eager 基线相同。注意：只在 8 题子集上做了 token 比较（见第 5 节），128 题的逐 token 一致性未验证。

### 4.2 TP=2 graph decode 五数据集重跑（2026-11-10）

脚本 `scripts/run_cb_tp2_graph.sh`，参数与 eager TP=2 运行相同，仅增加 `--graph-decode`。数据：`results/ablations/continuous_batching_tp2/graph_<name>/per_task.jsonl`。

| 数据集 | 题数 | CB graph 准确率 | CB eager | vLLM | CB graph makespan | CB eager makespan | vLLM makespan | vLLM / CB graph | CB graph 解码步数（eager） |
|---|---|---|---|---|---|---|---|---|---|
| AIME 2024 | 30 | 19（63.3%） | 20 | 24 | 773 s | 1845 s | 2140 s | 2.77× | 10737（10516） |
| AIME 2025 | 30 | 19（63.3%） | 18 | 18 | 835 s | 2081 s | 2525 s | 3.02× | 11303（—） |
| BBH | 1350 | 1166（86.4%） | 1158 | 1163 | 5878 s | 14461 s | 20666 s | 3.52× | 80044（80242） |
| GPQA Diamond | 198 | 142（71.7%） | 133 | 137 | 3760 s | 9542 s | 13815 s | 3.67× | 55079（—） |
| MATH-500 | 500 | 455（91.0%） | 455 | 451 | 4087 s | 9063 s | 11546 s | 2.83× | 45811（—） |

- 评分：AIME 与 MATH-500 用 `scorer_math.py`（fresh-math-v2），BBH 与 GPQA 用 `scorer_mc.py`（fresh-mc-v1）。
- 解码步数与 eager 基本一致（误差在 2% 以内），说明加速来自每步解码耗时下降，而非调度不同。以 AIME 2024 为例：eager 约 166 ms/步，graph 约 66 ms/步（由 makespan 减去 prefill 耗时再除以步数估算）。与 GSM8K longctx 的 decode 时间下降 60% 一致。
- 准确率的小幅变化（AIME 2024 少 1 题，GPQA 多 9 题，BBH 多 8 题）尚未逐题比对文本，因此不作结论。
- eager 基线的 summary 未记录参数，假定其使用 `run_cb_tp2.sh` 的同一套设置，此假定未从日志中核实。

### 4.1 分块预填充实验（TP=2 longctx，未采用）

动机：整段预填充会阻塞所有 decode。实现 `TPEngine.prefill_chunk`（KV 与 GDN state 跨块延续）后，每个 engine step 只处理一个 512 token 的 chunk，其余 slot 继续 decode。

| 指标 | 整段预填充（基线） | 分块 512 |
|---|---|---|
| makespan | 1554 s | **2081 s（慢 34%）** |
| 准确率 | 122 / 128 | 122 / 128 |
| prefill 耗时 / 吞吐 | 793 s / 1473 tok/s | 1133 s / 1031 tok/s（2415 个 chunk） |
| decode 步数 / 单步耗时 | 4341 / 0.176 s | 5374（+24%） / 0.174 s |

结论：
- 分块后单位 token 的 prefill 效率下降（每个 chunk 都要对全部已缓存 KV 做注意力，并重复展开 KV）。
- 每次只预填充一个请求，预填充期间该 slot 不参与 decode，batch 常不满，decode 步数增加。单步耗时没有变化，问题在 slot 占用。
- 正确性：CPU 张量并行测试（`scripts/test_chunked_prefill_cpu.py`，chunk 5/8/16/37）与参考 tokens 完全一致，held slot 状态误差约 1e-9；负对照（去掉 hold 恢复）状态误差 0.06–0.69，测试能检出。NPU 上 4 题 smoke 的预测一致，部分题目文本在数个 token 后分叉（fp 数值差异）。
- 结论：该实现不作为默认配置，`--prefill-chunk` 默认 0。要改善吞吐需要解决 slot 占用，见第 7 节。

#### 4.1.1 多请求并行预填充的小规模测试（36 题，TP=2）

目的：验证“同时预填充 2 个请求能否减少 decode 步数”。

| 配置 | makespan | decode 步数 | prefill 耗时 | decode 耗时 | 正确 |
|---|---|---|---|---|---|
| 基线（整段预填充） | 620 s | 2411 | 203 s | 413 s | 35 / 36 |
| chunk 2048，同时 1 个 | 702 s（+13%） | 3128 | 153 s | 545 s | 35 / 36 |
| chunk 2048，同时 2 个 | 712 s（+15%） | 3127 | 160 s | 548 s | 35 / 36 |

结论：
- 同时预填充 2 个请求（`--max-prefill 2`）没有减少 decode 步数（3127 对 3128），也没有缩短 makespan。之前“slot 占用是主要原因”的假设没有被这组数据支持。
- 分块版本的 decode 步数比基线多约 30%。每个 engine step 只推进一个 chunk，一个长 prompt 要占用多个 step 才能完成预填充，期间其 slot 不产生 token，这部分代价与并行数量无关。
- 分块的总 prefill 耗时（153–160 s）低于基线（203 s），但 decode 步数增加抵消了这一收益，净效果为变慢。
- 36 题样本较小，但 decode 步数差异（约 30%）远大于随机波动，与 128 题的结果（+24%）方向一致。

后续：分块路线在当前结构下没有收益。要继续，需要把 prefill chunk 与 decode 行合并到同一次 forward，否则不建议继续投入。

### 4.3 单步指标（表格口径：prefill tok/s、B1/B4 ms/step、decode tok/s）

测量方式：固定 batch（slot 数 = B，全部占满，不做调度），预热 5 步后计时 100 步，每步同步。prefill 为 batch-1，8 条 GSM8K prompt（平均 370 token），2 次预热后计时。配置统一为 fast-gdn + gdn-v2 + elem-conv + graph decode，BF16。decode tok/s = B ÷ 步耗时（batch 内每秒生成的 token 总数）。

| 配置 | 脚本 | Prefill tok/s | B1 ms/step | B1 decode tok/s | B4 ms/step | B4 decode tok/s |
|---|---|---|---|---|---|---|
| 单卡 910B3 | `scripts/bench_table_metrics.py` | 878 | 61.7 | 16.2 | 71.1 | 56.3 |
| TP=2（2×910B3） | `scripts/bench_table_metrics_tp2.py` | 867 | 43.3 | 23.1 | 50.1 | 79.9 |

原始数据：`results/bench_910b_table/{b1,b4,tp2_b1,tp2_b4}.json`。

- TP=2 把 B1 步耗时从 61.7 降到 43.3 ms（−30%），decode tok/s 从 16.2 升到 23.1（+43%）。B4 的 decode tok/s 从 56.3 升到 79.9（+42%）。
- prefill 在两种配置下都约 850–880 tok/s，TP=2 没有带来 prefill 加速。prefill 是 batch-1 且逐条串行，瓶颈不在 TP 通信上，这与 4.1 的结论一致。
- 与表格的对照仍需注意：表中 Qwen3.8-27B 为 4×910B3（TP=4），我们只能到 TP=2；模型版本也不同（3.5 对 3.8）。表中同类机型的 prefill 约 7000 tok/s，我们只有约 870，差距主要在 prefill 路径。

## 5. 配置公平性

| 项目 | 是否对等 | 说明 |
|---|---|---|
| 并发（12 slot） | 是 | 两边一致 |
| 输出上限（6144 / 2048） | 是 | 两边一致 |
| 上下文上限 | 是 | 8192 或 16384，两边一致 |
| Decode 方式 | 是（CB 六个数据集均有 graph 运行） | CB 的 TP=2 已支持 graph decode（`--graph-decode`），vLLM 为 aclgraph。第 3、4 节的 graph 行为主要比较对象；eager 行仅作历史记录 |
| Prefix caching | 是 | 两边都关闭 |
| 运行方式 | 未确认 | 两边是否同时占用 NPU 未查时间窗口，需要确认 |

**TP=2 graph decode 的实现与验证**：原先的断言已移除（`src/cb_tp.py` 的 `TPEngine.__init__`），`cb_run.py` 允许 `--tp` 与 `--graph-decode` 同时使用。HCCL all-reduce 可以被捕获进 NPUGraph，已由双卡最小测试验证（`scripts/test_hccl_graph.py`：replay 与 eager 完全一致，且换输入后结果随输入变化）。8 题子集（`--limit 8`，`results/ablations/continuous_batching_tp2/graphcheck_{eager,graph}_8_1109_1022/`）比较 token：7 题逐 token 一致；题 84 在第 0 轮第 134 个 token 处分叉（eager 为 `436`，graph 为 `220, 22`），最终答案均为 15（正确）。原因未确认，推测是 bf16 近似并列时的舍入差异（graph 在 512 的倍数桶上做带 mask 的注意力，eager 用实际长度），尚未用 logits 验证。

## 6. 发现的问题与状态

| # | 问题 | 影响 | 状态 |
|---|---|---|---|
| 1 | CB 的 TP=2 decode 为 eager | 速度比值偏低，TP=2 比较不完全对等 | 大部分解决。TP=2 graph decode 已在六个数据集上完成（第 4.2 节，GSM8K longctx 1554 s → 1059 s）。未解决：graph 与 eager 的逐题文本一致性（仅 8 题子集比对，题 84 有 1 处分叉，见第 5 节）；GSM8K graph 解码步数多出 46 步，原因未查 |
| 2 | AIME 2024 上 CB graph 比 vLLM 少 5 题（19 vs 24）；eager 为少 4 题（20 vs 24） | graph 与 vLLM 的差异题为 0、1、7、21、28；全部为 vLLM 对、CB 错，无反向题。配对符号检验 p ≈ 0.06，边界水平 | 已逐题核对（`scripts/diff_aime_vllm_vs_cb.py`）。见下方"问题 2 逐题分析" |
| 2a | 评分器 v1 取最后一个 `\boxed`，若输出末尾是残缺的 `\boxed`（重复循环被截断）则整题判错 | 题 16 的 `prediction` 为正确答案 721，但被判错 | 已在 v2 修复（跳过未闭合的 boxed），题 16 计为正确。v1 结果已不再使用 |
| 3 | MATH-500 与 GPQA 中无 `\boxed{}` 的题数较多（CB 91 / 35） | 准确率被截断拉低 | 未解决。可以提高终轮上限或改用更长的首轮上限 |
| 4 | longctx 8192 运行无效（114 题因上下文不足被拒绝） | 不能用于比较 | 已作废，数据保留在 `results/ablations/continuous_batching_tp2/gsm8k_longctx_128_max8192/` 和 `results/bench_vllm_tp2/gsm8k_longctx_128_max8192/` |
| 5 | `src/cb_run.py` 的 `correct` 字段在 direct 模式下错误（BBH/GPQA） | 旧结果文件中的 `correct` 字段全错 | 代码已修复，旧文件未重写。统计以离线评分器为准 |
| 6 | `bench_vllm_tp2/math500_aborted/` 为中断运行 | 不完整，不用于比较 | 仅作记录 |

### 问题 2 逐题分析（AIME 2024，graph 对 vLLM）

脚本：`scripts/diff_aime_vllm_vs_cb.py`。五道差异题的情况如下：

| 题 id | vLLM | CB graph | 原因 |
|---|---|---|---|
| 28 | 正确，2 轮 | 错误，第 2 轮被拒绝 | **实现差异**：CB 在 `prompt + 终轮上限 ≥ t_max` 时直接拒绝该题（`src/cb_run.py` 约第 301 行）。本题首轮 6022 token 后 prompt 为 6671，6671 + 2048 ≥ 8192，因此被拒。vLLM 的 `max_model_len` 同为 8192，但会把终轮 `max_tokens` 截到剩余空间，所以能完成。 |
| 1 | 正确，2 轮，3314 token | 错误，首轮 6144 token 截断，无 `\boxed{}` | 生成分叉：两边在第 26 个字符处即不同（"we will first…" 对 "we need to find…"）。 |
| 21 | 正确，2 轮，4463 token | 错误，首轮 6144 token 截断 | 生成分叉：第 617 个字符处开始不同，CB 版本陷入长推理。 |
| 0 | 正确，2 轮，1534 token | 错误，3 轮后终轮 2048 截断，无 `\boxed{}` | 生成分叉：第 992 个字符处，vLLM 用 sympy，CB 用 fractions。 |
| 7 | 正确，2 轮 | 错误，终轮 2048 截断在重复循环中 | 生成分叉：第 393 个字符处开始不同。 |

要点：
- 只有第 28 题是可以确定的实现差异，原因是 CB 的上下文检查比 vLLM 严格。修复方向是把终轮上限截到 `t_max − prompt`（或减去一个小余量），而不是拒绝。这一修改会改变 CB 的行为，尚未实施，也未重跑。
- 其余四题的分叉都发生在早期（字符 26 至 992），这与 greedy 解码在近似并列处的数值差异相符，但没有 logits 证据，不能确认是噪声。
- 被拒题也因运行而异：eager 拒绝了题 1 和题 5，graph 拒绝了题 5 和题 28。因此第 6 节此前"题 1 为上下文拒绝"的说法只适用于 eager 运行，已在本分析中更正。
- eager 与 graph 的 AIME 2024 准确率仅差 1 题（20 对 19），说明这一数据集上 CB 自身已有约 1 题的运行间波动。5 题的差距与 30 题的标准误（约 8–9 个百分点）相比仍不能定性。

### 问题 2 修复测试：终轮截断代替拒绝（`--truncate-to-fit`，2026-11-10）

`src/cb_run.py` 新增 `--truncate-to-fit`（默认关闭）：当 `prompt + 上限 ≥ t_max` 时，把生成上限截到 `t_max − prompt − 1`，剩余空间低于 `--min-fit-tokens`（默认 256）时仍拒绝。运行：`scripts/run_cb_tp2_graph_trunc_aime2024.sh`，结果 `results/ablations/continuous_batching_tp2/graph_trunc_aime2024/per_task.jsonl`。

| 运行 | 准确率 | makespan | 被拒题 | 与 vLLM 差异题 |
|---|---|---|---|---|
| CB graph（原行为） | 19/30 | 773 s | 5、28 | 0、1、7、21、28 |
| CB graph + truncate-to-fit | **20/30** | 819 s（+46 s，+6%） | 5 | 0、1、7、21 |
| vLLM | 24/30 | 2140 s | — | — |

- 题 28 由错变对（截断后生成 2 轮，给出 `\boxed{}`）。
- 题 5 仍被拒：prompt 8116，剩余空间 75 token，低于 256 的下限。eager 运行中题 5 的 prompt 为 8966，同样无法容纳。
- makespan 增加几乎全部来自题 28：它的完成时间（819 s）与整体 makespan 基本相同，即它是最后结束的任务。这是单次运行的结果，未重复测量，run-to-run 波动未知。
- 与 vLLM 的差距从 5 题缩小到 4 题（20 对 24），剩余 4 题（0、1、7、21）的原因仍是生成分叉，见上文逐题分析。

### 问题 2 补充：prompt token 一致，但生成 token 大面积分叉（2026-11-10）

脚本：`scripts/compare_prompt_tokens.py`。

- **输入 prompt 完全一致**：AIME 2024 的 30 题，CB 使用的 chat template（tools、system、enable_thinking=False）渲染后的 token 数与 vLLM 记录的首轮 `prompt_tokens` 全部相同（30/30）。
- **生成 token 大面积分叉**：首轮生成的 token 序列，30 题中只有 1 题完全一致。分叉位置从第 6 个 token（题 1）到第 2675 个（题 28）不等，中位数约 150 个 token。
- 这说明问题不只是"少数近似并列处的翻转"。在绝大多数题中，两边在开头几百个 token 内就已分叉，更像是系统性的数值或实现差异。之前"近似并列的噪声"的解释证据不足，需要修正。
- 可能原因（未验证）：CB 的 fast-GDN / gdn_v2 prefill、elem-conv、graph 桶 + mask，或 vLLM 自身的 fused kernel 与 bf16 累加顺序。
- 决定性的检查：用 HF 的单条 `generate`（不用 fast-GDN、elem-conv、graph，不分批）跑这几题，看它与 CB 还是与 vLLM 一致。CB 的 eager 解码此前与 HF `generate` 逐 token 一致（3–4 slot、40 步），因此若 HF 与 CB 一致，问题在 prefill 或 CB 的优化路径。

### 问题 2 HF 单条 generate 对照（6 题，全部优化关闭，2026-11-10）

脚本：`scripts/hf_reference_aime.py`（协议同 cb_run：相同 system、tools、上限 6144/2048、工具输出截断、最多 6 轮；每题单独 generate）。结果 `results/ablations/hf_reference/aime2024_ref6.jsonl`。首步 logits 探针：`scripts/hf_first_token_logits.py`。

| 题 | vLLM | CB graph+trunc | HF 参考 | 说明 |
|---|---|---|---|---|
| 0 | ✓ | ✗ | ✓ | HF 与 CB 在首轮第 555 个 token 分叉 |
| 1 | ✓ | ✗ | ✗ | HF 与 CB 在首轮第 28 个 token 之前一致，与 vLLM 在第 6 个 token 分叉 |
| 7 | ✓ | ✗ | ✗ | HF 与 CB 一致到第 453 个 token，与 vLLM 在第 128 个分叉 |
| 9 | ✓ | ✓ | ✓ | |
| 21 | ✓ | ✗ | ✓ | 首 token 为精确平手（见下） |
| 28 | ✓ | ✓ | ✓ | |

- 6 题准确率：vLLM 6/6，HF 4/6，CB 2/6。HF 也低于 vLLM，说明 vLLM 与 HF 之间本身就有路径差异，不只是 CB 的问题。6 题的样本太小，不能定论。
- **题 21 的首步是 bf16 精确平手**：HF 上 `To`（1206）与 `<tool_call>`（248058）的 logit 都是 24.000，argmax 取较小索引 `To`。vLLM 与 CB 都选了 `<tool_call>`，说明它们的数值路径让后者略高。这是数值噪声，不是逻辑错误。
- **题 1 的首步 margin 为 2.0**（`To` 25.125，`<tool_call>` 23.125），首 token 不分叉；它的分叉在第 6 个 token，原因未做 logits 检查。
- 结论：分叉的机制确实是 bf16 近似并列加上路径差异，HF、CB、vLLM 三者两两都会在某些题上分叉。CB 的准确率低是否系统性，需要更大的样本（全部 30 题的 HF 参考）才能判断。

## 7. 后续建议

1. **TP=2 graph decode 的后续**：（a）查明 8 题子集中题 84 的分叉原因（用 logits 检查该位置的 top-2 差值）；（b）对 128 题 GSM8K 和五个数据集做逐 token 比较（graph 对 eager），区分数值噪声与真实差异；（c）查明 GSM8K graph 比 eager 多出 46 个解码步的原因。（五个数据集的 graph 重跑已完成，见第 4.2 节。）
2. **AIME 2024 差距的后续验证**：（a）将 CB 的终轮上限截到 `t_max − prompt`，代替直接拒绝，然后重跑 AIME 2024，看题 28 是否恢复；（b）关闭 fast-GDN 和 elem-conv 重跑，或重复 2–3 次，看哪些题稳定分叉；（c）对题 1、21 的首个分叉点做 logits 检查。
3. **校验截断对准确率的影响**：对 MATH-500 和 GPQA，把终轮上限从 2048 提高到 4096，观察无 `\boxed{}` 题数的变化。
4. **统计不确定性**：AIME 每个数据集只有 30 题，应多次重复或扩大题集后再下结论。
5. **分块预填充的 slot 占用问题**（见 4.1）：尝试 chunk 2048 并允许 2 个请求同时预填充，小规模测试 decode 步数是否回到约 4341。若不能，考虑把 prefill chunk 与 decode 行合并到同一次 forward。

## 8. 附录：单卡结果（背景，不作为主要结论）

单卡实验最初用于测评，但因 KV 缓存随 slot × t_max 增长，单卡最多 16 slot，24 slot 已 OOM，并且在 2048 / 512 的上限下 GPQA 与 AIME 的截断严重，因此迁移到 TP=2。单卡结果仅供参考：

- 单卡 BBH 的并发不对等（CB 16 slot，vLLM 12），输出上限也可能不一致，**不能作为纯引擎对比**。
- 单卡 GPQA、MATH-500、AIME 的并发和上限基本对等，但截断严重，准确率偏低（如 GPQA CB 46.5%）。
- 单卡 makespan 比值（2.1–2.8×）高于 TP=2，主要来自配置差异和截断导致的生成长度差异。

数据：`results/continuous_batching/`、`results/bench_vllm/`。

## 9. 数据位置

- 自研 CB（TP=2，eager 基线）：`results/continuous_batching_tp2/{aime2024,aime2025,bbh,gpqa_diamond,math500,gsm8k_longctx_128}/per_task.jsonl`
- 自研 CB（TP=2，graph decode）：`results/ablations/continuous_batching_tp2/graph_{aime2024,aime2025,bbh,gpqa_diamond,math500}/per_task.jsonl` 与 `gsm8k_longctx_128_graph_1109_1028/`；启动脚本 `scripts/run_cb_tp2_graph.sh`
- vLLM（TP=2）：`results/bench_vllm_tp2/{aime2024,aime2025,bbh,gpqa_diamond,math500,gsm8k_longctx_128}/per_task.jsonl`
- 自研 CB（单卡）：`results/continuous_batching/{bbh,gpqa_diamond,math500,aime2024,aime2025}/per_task.jsonl`
- vLLM（单卡）：`results/bench_vllm/{bbh,gpqa_diamond,math500,aime2024,aime2025}/per_task.jsonl`（AIME 为重测结果）
- 作废的 8192 运行：`results/ablations/continuous_batching_tp2/gsm8k_longctx_128_max8192/`、`results/bench_vllm_tp2/gsm8k_longctx_128_max8192/`
- 评分器：`src/scorer_mc.py`（选择题，fresh-mc-v1）、`src/scorer_math.py`（数学题，fresh-math-v2；v2 起跳过未闭合的 `\boxed`）
- 重测脚本：`scripts/retest_vllm_aime.sh`
