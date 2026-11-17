# Gemma-4-26B-A4B-it 优化实验日志

用途：记录 Gemma 路径上每个实验、它的测量、中间判断和被修正的判断，作为之后构建优化依赖图（类似 `experiments/dependency_graph.md` 对 Qwen 路径的做法）的输入。记录原则：每条结论都标注证据强度；被推翻的判断保留并注明，不删除。

最后更新：2026-11-15（交接前整理：评分统一为 fresh-v1，补充 E-G17 与 Qwen 差距分析，更新下一步）

**给新窗口的一句话状态**：Gemma 串行基线已完成（fresh-v1 108/128，10588.6 s）；`--fast-norm` 在 full128 上 8710.8 s（fresh-v1 106/128）；`--cb` 16 slot 在 first16 上 770–794 s（fresh-v1 14/16），仍远慢于 Qwen 路径（100 s 以内）。**当前暂停，不启动 full128，等待用户决定是否先做 MoE experts 的只读拆分分析。**

**评分说明（重要）**：早期 E-G01、E-G07、E-G08、E-G16 的数字曾用 `agent_protocol.correct` 计算，它不解析 `20\%` 之类的答案，比 fresh-v1 严。**以下所有正确数均以 `src/scorer.py`（fresh-v1）为准**，旧数字保留在对应条目中并标注。

---

## 0. 环境与对象

| 项 | 内容 |
|---|---|
| 模型 | `/remote-home/S45160/work/910b_fresh/gemma-4-26B-A4B-it`，Gemma4ForConditionalGeneration，文本部分 30 层（25 sliding，5 full），MoE 128 experts top-8，hidden 2816，sliding window 1024，`attention_k_eq_v=True`，`final_logit_softcapping=30` |
| 硬件 | 910B3，npu:0 与 npu:1，每次一张卡，串行 |
| 协议 | 与 Qwen 路径同构的 GSM8K agent：128 题，greedy，非 thinking，512 token/轮，最多 6 轮，python 工具 10 s 超时，评分 fresh-v1（`src/scorer.py`，`agent_protocol.py`）。Gemma 使用 `agent_protocol.SYSTEM`，与 Qwen 的 `run_agent.py` 的 `USER_SUFFIX` 不同 |
| HF 环境 | `/remote-home/S45160/miniforge3/envs/ascend-e2e`（transformers 5.5.4，torch_npu 2.9.1，python 3.12） |
| 主要代码 | `src/gemma_run.py`（runner，`--fast-norm`、`--cb` 开关）、`src/gemma_fast_norm.py`、`src/gemma_cb.py`（CB 引擎）、`scripts/profile_gemma_b1_breakdown.py`、`scripts/calibrate_gemma_step.py`、`scripts/test_gemma_cb.py`、`scripts/diag_gemma_cb_flip.py`、`scripts/run_gemma_abba_norm.sh` |

---

## 1. 实验时间线与结果

### E-G01 串行基线（serial_baseline）

- 配置：B=1，HF `generate()`，无优化，128 题。
- 结果：**fresh-v1 108/128**（旧 `agent_protocol` 评分为 106/128，已弃用），**makespan 10588.6 s，生成 60,426 token，峰值 allocated 51.9 GB**。
- 错题（fresh-v1）：175, 214, 231, 304, 423, 445, 515, 550, 552, 590, 640, 857, 952, 1001, 1011, 1030, 1035, 1048, 1161, 1288 等 20 题，其中 pred 为空的占多数。
- 观察：
  - 61 题没有调用工具就直接作答，其中 41 题正确；67 题调用了工具，其中 65 题正确。
  - 17 题输出里没有 `\boxed{}`（pred 为空）。
  - 没有任务跑满 6 轮。
  - 错题列表：140, 175, 214, 231, 304, 423, 445, 458, 515, 550, 552, 590, 640, 857, 952, 1001, 1011, 1030, 1035, 1048, 1161, 1288。
- 注意：这是 Gemma 的第一个基线，协议与 Qwen 的 `run_agent.py` 不同，**不能直接与 README 中 Qwen 的 8365 s 比较**。

### E-G02 B=1 decode 分解（第一版，hook 在手写循环上）

- 脚本：`scripts/profile_gemma_b1_breakdown.py`，npu:1，60 步，140-token prompt。
- 结果（ms/step）：总 336，host 入队 335，设备等待 0.9，attention 64，MLP+experts 186。
- 当时的判断：decode 是 host 派发主导（99.7% 在入队）。
- **后被推翻**，见 E-G03。原因：这是手写循环，比 `generate()` 慢约 2 倍，不代表 baseline 的开销。

### E-G03 校准：手写循环 vs generate()（calibrate_gemma_step.py）

- 结果：
  - `generate()`：164–172 ms/token（3 条 prompt）。
  - 手写循环（无 hook，每步同步）：367–376 ms/step。
- 排查（均不是原因）：
  - attention mask 传入与否：333 vs 345 ms。
  - `DynamicCache(config=...)`：345 ms。
  - 去掉每步 `synchronize`：332 ms。
- 确认：同一条 prompt 上 `generate()` 为 128 token、约 183 ms/token，差距真实存在。
- **未找到手写循环慢的确切原因**，但之后不再用手写循环做测量。
- 结论：E-G02 的绝对数字作废，只保留"手写循环不能代表 generate()"这一条。

### E-G04 在 generate() 路径上加 hook 的 breakdown（层级）

- 结果：169 ms/token，其中 30 层合计 156 ms（92%），attention 51 ms，MLP+experts 28 ms。
- 钩子开销：加钩子前后 165–183 vs 169 ms，视为可忽略。
- 当时的判断：层内还有约 77 ms 未归因。

### E-G05 generate() 路径，子模块级 breakdown（probe_children）

每 token 时间，30 层，每个模块每 token 调用 30 次：

| 模块 | ms/token |
|---|---|
| self_attn | 50.1 |
| experts | 20.2 |
| router | 15.7 |
| mlp（稠密） | 6.9 |
| input_layernorm | 8.0 |
| post_attention_layernorm | 8.1 |
| pre_feedforward_layernorm | 7.5 |
| post_feedforward_layernorm | 7.5 |
| post_feedforward_layernorm_1 | 7.9 |
| post_feedforward_layernorm_2 | 8.0 |
| pre_feedforward_layernorm_2 | 7.8 |
| 合计 7 个 norm | ~54.7 |
| 层内未归因（残差、layer_scalar） | ~12 |
| 总计 | 172.8 |

- 观察：7 个 RMSNorm 合计 32%，每个约 0.26 ms，每 token 210 次调用。norm 计算量很小，耗时主要是算子调度。
- 判断（**待验证**）：`Gemma4RMSNorm` 是 Python 实现（float 转换、pow、mean、Python float 的 eps、pow(-0.5)、两次乘法、type_as），怀疑有 host 同步，类似 Qwen 上的 311 次/步 `_local_scalar_dense`。这没有被 profiler 直接证实。

### E-G06 RMSNorm 融合微基准与单条 prompt 检查

- `torch_npu.npu_rms_norm` 替换 `Gemma4RMSNorm.forward`（`src/gemma_fast_norm.py`）。
- 微基准（单次调用，D=2816，bf16）：原版 247.2 µs，融合 48.8 µs，输出 max abs diff 0.0。
- 单条 prompt（128 token，generate）：176.1 → 132.1 ms/token（约 −25%）。
- 同一条 prompt 的 logits：最大差 1.4766，argmax 相同；生成 token 在第 103 位分叉。
- 补丁范围：433 个带 weight 的 norm 被替换；`with_scale=False` 的 norm（v_norm、router 内部 norm）保持原样。
- 补丁的边界条件：若 dtype 或最后一维不匹配则回退原 forward，不静默改变形状。

### E-G07 first16 ABBA：`--fast-norm`（run_gemma_abba_norm.sh）

顺序 A1 → B1 → B2 → A2，npu:0，串行。

| 运行 | makespan (s) | 正确 |
|---|---|---|
| A1 原版 | 1365.2 | 12/16 |
| B1 fast-norm | 978.5 | 12/16 |
| B2 fast-norm | 1000.9 | 12/16 |
| A2 原版 | 1326.5 | 12/16 |

- A 均值 1345.9，B 均值 989.7，约 −26%。两组区间不重叠。
- 错题四次都是 140, 175, 214, 231，与全量基线一致。A1 vs B1 对错无翻转。
- 组内确定性：A1=A2 16/16 逐字一致，B1=B2 16/16 逐字一致。A vs B 只有 6/16 逐字一致。

### E-G08 full128 `--fast-norm`（full128_fastnorm）

- npu:0，同 runner，`--fast-norm`，433 个 norm 打补丁。
- 结果：**fresh-v1 106/128，makespan 8710.8 s，生成 61,760 token**（旧评分为 104/128，已弃用）。
- 与基线（E-G01，三小时前、不同批次）比较：makespan −17.7%（10588.6 → 8710.8），fresh-v1 正确 108 → 106。
- 逐题对比（fresh-v1）：丢失 428、950；无新增正确。
  - 428：输出在 512 token 上限被截断（代码块写到一半），没有 `\boxed{}`。
  - 950：同样没有 `\boxed{}`，单轮。
  - 两题都是输出格式/截断问题，不是数值算错。
- 逐字一致：49/128。
- **注意**：这是跨批次的比较，没有做同批次原版对照。用户决定不做同批次对照（"不用了，继续按照依赖图优化端到端 makespan"）。因此 −17.7% 的数字应视为"方向确认、幅度待同批次验证"，与 first16 ABBA 的 −26% 相互印证，但不是确定的幅度。

### E-G09 router 的 scalar 同步（微基准，未实施）

- `Gemma4TextRouter.forward` 中 `hidden_states * self.scale * self.scalar_root_size`，`scalar_root_size` 是 Python float。
- 微基准（单次调用）：router 整体 505 µs；norm 单独 185 µs；norm + 张量化 scale 328 µs。
- 估算收益：约 4 ms/token（占约 130 ms/token 的 ~3%）。
- 决定：**暂不实施**，等大改动后一起测。

### E-G10 显存探针（probe_mem.py）

- 实测：权重 48.07 GiB（加载后 `memory_allocated`），512-token prefill 峰值 0.62 GiB，加载后 reserved 48.85 GiB。
- 由 config 计算（未实测分配）：
  - sliding KV：每 token 200 KiB（25 层，K+V，8 kv heads × 256 dim，bf16）。窗口 1024 上限约 200 MiB/slot。
  - full KV：每 token 10 KiB（5 层，K=V 共享，2 global kv heads × 512 dim）。t_max=2048 约 20 MiB/slot。
- 基线最长上下文（prompt + 生成）：805 token，中位数 527，p99 783。0 题超过 2048。
- 结论：t_max=2048 足够；16 slot 起步，理论上可到约 40 slot，未测。
- 修正：探针第一次因在模型加载前调用 `reset_peak_memory_stats` 报错，已修。

### E-G11 CB 引擎（src/gemma_cb.py）的设计决定

- 复用 HF 模块：norm、MLP、router、experts、layer_scalar、embedding、final norm、lm_head。数值语义由 HF 保证。
- 自写注意力：每层 KV 为 `[N, kvh, T, hd]` 的 slot 缓冲，sliding 层用窗口掩码（窗口 1024，大于最长上下文，不需要环形缓冲）。
- prefill：每个 turn 走 HF batch-1 `DynamicCache`，结果拷入 slot。这和基线每轮重新 prefill 的行为一致。
- decode：N 个 slot 稠密计算，空闲 slot 为哑行。
- 断言：prompt ≤ sliding window；prompt + 512 ≤ t_max。
- 注意力的 softmax 在 fp32 中计算后转回 bf16，与 HF eager 顺序一致（scale → softcap（无）→ mask → softmax）。

### E-G12 CB 对齐测试（test_gemma_cb.py）

- 参照：HF `generate()`，贪心，40 token。
- 结果：

| 用例 | match | first_div |
|---|---|---|
| N=1 prompt0 | 40/40 | — |
| N=1 prompt1 | 27/40 | 15 |
| N=1 prompt2 | 40/40 | — |
| N=1 prompt3 | 28/40 | 28 |
| N=4 （同上四条） | 与 N=1 完全相同 | 15, 28 |

- 结论：批处理本身不引入偏差（N=1 与 N=4 结果一致）。分叉来自数值。

### E-G13 分叉诊断（diag_gemma_cb_flip.py）

| prompt | first_div | HF top-2 margin | max|cb − hf_full| | argmax 一致 |
|---|---|---|---|---|
| prompt1 | 15 | 0.25 | 0.625 | 是（均为 9279） |
| prompt3 | 28 | 0.125 | 0.625 | 是（均为 506） |

- 判断：两处都是 bf16 平局（logit 量级约 26，bf16 1–2 ulp），HF 完整前向自己也选中 CB 的 token。**判定为数值翻转，不是逻辑错误。**
- 与 E-G06 的 RMSNorm 翻转性质相同。

### E-G14 `--cb` 冒烟测试（8 题，4 slot，npu:1）

- 结果：**8/8 正确，makespan 828.4 s，3181 token（≈260 ms/token）**。
- 对比：B=1 `generate()` ≈175 ms/token。**CB 冒烟测试比 B=1 还慢**，与理论预期相反。这是一个未解决的问题。
- 逐任务 token：23:174, 51:591, 68:431, 76:341, 84:285, 99:722, 108:404, 120:233。

### E-G15 CB 性能分解（probe_cb_step.py / probe_cb_parts.py）

| 测量 | 数值 |
|---|---|
| prefill（每 prompt，HF batch-1 + 拷贝） | 956 ms |
| decode 整步，N=16 | 673 ms（第一次，含预热） |
| decode 整步，N=16 | 383 ms（parts 脚本，位置固定 200） |
| decode 整步，N=16 | 295 ms（同上，再测） |
| decode 整步，N=4 | 347 ms |
| 单层 sliding，N=16 | 9.7–12.3 ms |
| 单层 full，N=16 | 12.3 ms |
| attention（sliding，单层） | 2.3 ms |
| router（单层） | 0.6 ms |
| lm_head | 1.3 ms |
| embed | 0.1 ms |

- 观察：
  - 单步耗时几乎不随 slot 数变化（4 slot 347 ms，16 slot 295 ms）。与 Qwen 的"decode 步成本与 batch 宽度几乎无关"一致。
  - 单层约 10 ms，30 层约 300–370 ms，与整步吻合。
  - 单层里 attention 只占 2.3 ms，router 0.6 ms，其余约 7–9 ms 未拆分（MLP、experts、norm）。
- 理论推算：16 slot、295 ms/步，约 18 ms/token，远快于 B=1 的 175 ms/token。
- **与 E-G14 的矛盾**：冒烟测试中 4 slot 应为约 87 ms/token，实测 260 ms/token。差距约 170 ms/token，**还没有找到来源**。
- 已排除的候选（有证据）：
  - prefill 不是主因：8 题、11 个 turn，约 11 s，占 828 s 很小。
  - 单步 decode 成本本身：实测不随 slot 增长。
- 未排除的候选：每步的 `torch.tensor(...)` 主机到设备拷贝、`.tolist()` 同步、`Tact = int(pos.max().item())` 同步、调度循环的 Python 开销、掩码每步重建、slot 利用率（尾部只剩少数长任务时，每步仍为 N 步成本）。
- 注：8 题中 4 slot 很快降到只剩 1–2 个活跃，尾部的低利用率很可能是主因。这一点还未用计数器验证。

### E-G16 CB first16，16 slot（已完成，单次）

- 命令：`src/gemma_run.py results/gemma/cb_first16_s16 --device npu:1 --cb --slots 16 --limit 16`，日志 `logs/gemma_cb_first16_s16.log`。
- 结果：**13/16 正确，makespan 770.3 s，生成 7806 token**，峰值 allocated 59.7 GB，reserved 62.7 GB。
- 与 `--fast-norm` B=1 first16 比较（B1 为 978.5 s，B2 为 1000.9 s）：
  - makespan 约快 22%（770 vs 均值 990）。
  - 逐题：CB 丢失 0 题，新增正确 231（B1 错）。错题 140、175、214 与 B1 的 140、175、214 一致。
  - 逐字一致 8/16。
  - 生成 token 比 B1 多约 8%（7806 vs 7221），主要来自 231 的第三轮。
- 有效吞吐：770 s / 7806 token ≈ 99 ms/token，已远低于 B=1 的 ~130 ms/token（fast-norm），说明 E-G14 冒烟时的 260 ms/token 与 E-G15 的理论差距在 16 slot 下不再成立。**E-G15 的"矛盾"需要重新解释**：冒烟只用 4 slot 且 8 题尾部利用率低，不代表 CB 本身慢。
- 注意：140 的预测是 `20\%`，`agent_protocol.correct` 不解析它，判错；`scorer.py`（fresh-v1，取最后一个数字）会判对。**所有 Gemma 的 runner 结果都用 `agent_protocol.correct` 评分，与 fresh-v1 的 `scorer.py` 规则不完全一致**，需要用 `scorer.py` 对全部 Gemma 结果重新评分后再下结论。
- 限制：单次运行，不是 ABBA。
- **fresh-v1 重评**：13/16 应更正为 **14/16**（140 被 `\%` 误判，现判对）。错题 175、214（pred 为空，512 token 截断或未出 boxed）。

### E-G16b CB first16 重复运行（cb_first16_s16_rep，npu:1，14:55–15:09 UTC）

- 结果：**fresh-v1 14/16，makespan 794.3 s**，生成 7806 token（与第一次完全相同）。
- 与第一次比较：生成文本 16/16 逐字一致，正确性无翻转。makespan 770 vs 794（约 3% 波动）。
- 结论：CB first16 的正确性和 token 数完全可复现，速度的波动在几十秒量级。

### E-G17 与 Qwen 路径的差距（分析，未实验）

- 用户反馈：Qwen 路径在 CB 阶段 first16 已优化到 100 s 以内，Gemma CB 为 770–794 s，**不可接受**。决定：**暂停推进，不启动 full128**。
- 量化差距：
  - Gemma CB decode 整步（16 slot）约 295 ms；Qwen 约 120–140 ms。单步成本约 2×。
  - Gemma CB first16 约 99 ms/token（770 s / 7806 token），比 Qwen 慢约一个数量级（按用户反馈的 100 s 量级推算）。单步 2× 之外，还有约 5× 未解释的差距。
- 已知可疑来源（均未验证）：
  1. **MoE experts 是 HF 的 Python 循环**：`Gemma4TextExperts.forward` 对命中的每个 expert 做一次循环（`nonzero`、`where`、`index_add_`），每层每步循环最多 128 次，主机开销大。这是目前最可疑的单点。
  2. **单层 7–9 ms 未拆分**：MLP、experts、7 个 norm 的 device/host 时间未知。
  3. **prefill 每轮重做**：batch-1 HF prefill 约 956 ms/条，first16 约 20+ 轮，合计约 20 s，不是主因，但仍占约 3%。
  4. **无图捕获**：Qwen 的 F 节点收益约 12%，Gemma 未做，且 MoE 的形状是否静态未确认。
  5. **尾部利用率**：first16 在 16 slot 下仍会有轮次不齐，活跃 slot 数未记录。
- **注意**：前面"CB 比 B=1 快 20%"只是 Gemma 自己的对比，不代表达到 Qwen 水平。
- 建议的下一步（需用户确认）：只读的 profiler 拆分，聚焦 experts 循环的 device 与 host 时间；不改代码，不占用 NPU 超过一次短测。

---

## 2. 被修正或推翻的判断（按时间）

| 原判断 | 来源 | 修正 | 修正依据 |
|---|---|---|---|
| "decode 是 host 派发主导，99.7% 在入队" | E-G02 | 作废，手写循环不代表 generate() | E-G03 校准 |
| "层内 77 ms 未归因" | E-G04 | 大部分归因为 7 个 norm（约 55 ms）和 attention/MoE | E-G05 |
| "norm 慢是因为 Python float 同步" | E-G05 推断 | **未证实**，仍待 profiler 验证 | — |
| "router scale 同步是重要收益" | E-G09 | 收益约 3%，暂不做 | 微基准 |
| "CB 冒烟的慢是 prefill 造成" | E-G14/15 | 排除，prefill 占比很小 | 11 s / 828 s |
| "CB 每步成本随 batch 增长" | 未明确 | 不成立，几乎不增长 | E-G15 |
| "fast-norm full128 的 −17.7% 是确定收益" | E-G08 | 应视为方向确认，幅度需同批次验证；用户决定不做同批次对照 | 跨批次比较 |

---

## 3. 当前未解决的问题（按对 makespan 的潜在影响排序）

1. **CB 冒烟比 B=1 慢（260 vs 175 ms/token）**：已由 E-G16 部分解释（16 slot 下 CB 实测约 99 ms/token，快于 B=1）。冒烟慢的原因可能是 4 slot + 8 题的尾部利用率低，未直接验证。
2. **slot 尾部利用率**：8 题、4 slot 时尾部活跃 slot 很少，单步成本不变，利用率低。需要每步记录活跃 slot 数来验证。
3. **norm 的 host 同步假设**：`Gemma4RMSNorm` 的 Python float 是否真的触发同步，未用 profiler 证实。fast-norm 已经有效，但机制未知。
4. **MLP / experts 的 device 时间**：单层约 7–9 ms 未拆分，可能是下一个大头。
5. **同批次对照**：fast-norm full128 的 −17.7% 没有同批次原版验证。
6. **128 题的 CB 结果**：未运行。
7. **router scalar 同步**：约 4 ms/token，低优先级。
8. **full128 的两题截断（428、950）**：512 token 上限导致，与 fast-norm 无关，但需要确认原版同题是否也受限（基线 428 是否正确未单独检查）。

---

## 4. 与 Qwen 路径的对应关系（用于依赖图）

| Qwen 节点 | Gemma 对应 | 状态 |
|---|---|---|
| A 串行基线 | E-G01 | 完成，fresh-v1 108/128，10588.6 s |
| B 融合算子（RMSNorm 等） | E-G06/E-G07/E-G08 `--fast-norm` | 完成，first16 −26%，full128 −17.7%（跨批次），fresh-v1 106/128 |
| C 连续批处理 | E-G11–E-G17 `--cb` | 正确性通过（fresh-v1 first16 14/16，可复现）；first16 770–794 s，快于 fast-norm B=1 约 20%，但**远慢于 Qwen 的 100 s 以内**，暂停 |
| D fast-GDN | 不适用（无 GDN） | — |
| E 显存约束 | E-G10 | 完成，t_max=2048，16 slot 起步 |
| F graph decode | 未做 | 需先确认 MoE 与 attention 的形状是否静态化 |
| G GDN v2 | 不适用 | — |
| H elem-conv | 不适用（无 conv1d），MoE/attention 的对应优化待定 | — |
| 新增：router scalar 同步 | E-G09 | 收益小，暂缓 |

Qwen 路径的经验，在 Gemma 上**需要重新验证**的部分：
- "decode 是 host 派发主导" —— 未在 generate() 路径上证实（E-G03 推翻了基于手写循环的版本）。
- "图捕获收益约 12%" —— 不能假设，需要先确认 MoE 形状是否静态。
- "C 带来约 2 倍" —— Gemma 的单步成本不随 batch 增长（E-G15），理论上收益更大，但实测冒烟未体现。

---

## 5. 环境与依赖问题（vLLM 对照，已搁置）

- 目标：在新环境 `vllm-ascend-gemma` 中建立 vLLM 对照。
- 进展：
  - `vllm==0.30.0` 与 vllm-ascend 0.30.0 的 torch 版本冲突（0.30.0 要 torch 2.13.0，wheel 要 2.10.0）。
  - 最终安装 vllm 0.19.1，但 vllm-ascend 0.30.0 的启动补丁依赖 `vllm.parser.glm47_moe`，不存在，报 ImportError。
  - 换 vllm-ascend 0.23.0，补丁依赖 `_approximate_gcd`，也不存在。
  - `triton-ascend==3.6.0.post2` 在 pip 源中不存在；py310 环境中是 3.2.0。
- 结论：vllm 与 vllm-ascend 的版本组合无法确认兼容。用户决定先走 HF 路线，搁置 vLLM 对照。
- 在 HF 路线的 `ascend-e2e` 环境中，`torch_npu` 与 `transformers 5.5.4` 工作正常，不受影响。

---

## 6. 产出文件索引

| 文件 | 内容 |
|---|---|
| `results/gemma/serial_baseline/` | E-G01 |
| `results/gemma/profile_b1/breakdown.json` | E-G02（已作废） |
| `results/gemma/calibration/calib.json` | E-G03 |
| `results/gemma/abba_norm/{A1,B1,B2,A2}/` | E-G07 |
| `results/gemma/full128_fastnorm/` | E-G08 |
| `results/gemma/cb_test/` | E-G12（对齐测试输出目录，未写结果文件） |
| `results/gemma/cb_smoke/` | E-G14 |
| `results/gemma/cb_first16_s16/` | E-G16（完成，fresh-v1 14/16，770.3 s） |
| `results/gemma/cb_first16_s16_rep/` | E-G16b（完成，fresh-v1 14/16，794.3 s） |
| `logs/gemma_serial_baseline.log` | E-G01 |
| `logs/gemma_abba_norm_*.log`，`logs/gemma_abba_norm_driver.log` | E-G07 |
| `logs/gemma_full128_fastnorm.log` | E-G08 |
| `logs/gemma_cb_first16_s16.log` | E-G16 |

---

## 7. 当前状态与下一步（交接用）

**状态**
- 没有任何 Gemma 实验在后台运行（已确认）。
- 不推进 full128 CB。用户明确决定暂停，原因：CB first16 约 780 s，Qwen 路径 100 s 以内。
- 自动循环已停止（无定时唤醒）。

**已确认的事实**
- 正确性：fresh-v1 为准。基线 108/128；fast-norm 106/128（丢 428、950，均为截断）；CB first16 14/16，可复现。
- 速度：fast-norm 比基线快约 18%（跨批次）；CB first16 比 fast-norm B=1 快约 20%，但绝对速度仍差一个数量级。

**下一步（待用户确认，未授权执行）**
1. 只读的 MoE experts 拆分：profiler 或分段计时，看 `Gemma4TextExperts.forward` 的 device 与 host 时间。判断是否需要重写 experts（如按 expert 分组的批量 matmul，或静态化路由）。
2. 拆开单层剩余的 7–9 ms（MLP、experts、7 个 norm）。
3. 若 experts 确认是主因，再考虑 F 节点（图捕获）是否可行——需要先让 MoE 的形状静态化。
4. 修 `gemma_run.py` 中 CB 的 `Tact` 同步和每步 `.tolist()`，属于小改动，需先测再改。
5. full128 CB：等速度达到可接受水平后，再经用户确认启动。

**不要做的事**
- 不要再用 `agent_protocol.correct` 评分，只用 `src/scorer.py`。
- 不要把 CB 的 20% 对比当成"已经接近 Qwen"。

---

## 8. 追加记录：CB 分析与优化（E-G17 至 E-G22，2026-11-15 至 11-16）

**注意**：第 7 节的"下一步"是 11-15 之前写的，已部分过时。本节记录之后的实际进展。

### 8.1 诊断链（每步都用 wall 或同进程对照验证）

| ID | 假设 | 结论 |
|---|---|---|
| E-G17 | experts 的 Python 循环是 CB 慢的主因 | 未验证（仅分析） |
| E-G18 | prefill、工具执行或 host 是 makespan 主因 | 否决：decode 占 96.5%，prefill 3.4%，工具 0.05%，host 0.05%；每步固定成本约 750–830 ms，不随活跃 slot 数变化 |
| E-G19 | 每层命中约 80 个 expert | 否决：实测平均 22.5 个（含哑行） |
| E-G20 | B=1 时 CB 单步约等于 generate()，慢在 attn 或 norm | 否决：B=1 下 CB 步 351 ms，generate 160 ms；experts 占 55%（198 ms），attn 19%，其余 26% |
| E-G21 | 只计算活跃 slot（compact）缩短 makespan ≥2% | 接受（first16）：A 均值 793.3 s，B 均值 672.0 s，约 −15%；正确数 14→13（231 的平局翻转） |

### 8.2 优化：MoE experts 分组计算（E-G22，接受）

**方法**
- 旧路径：HF `Gemma4TextExperts.forward` 对命中的每个 expert 做一次 Python 循环（含 `nonzero` 同步）。
- 新路径（`src/gemma_cb.py::experts_grouped`，opt-in）：
  1. 把 T×K 个 (token, expert) 分配按 expert 排序（`argsort`，stable）；
  2. 用 `bincount` + `cumsum` 得到 group_list，留在 device 上；
  3. 两次 `torch_npu.npu_grouped_matmul`（gate_up、down），参数 `split_item=2, group_type=0`；
  4. 乘路由权重，`index_add_` 回 token。
- 启用：`src/gemma_run.py --cb --compact --grouped-experts`。默认路径不变。
- 踩过的坑：`split_item=0` 在 `group_type=0` 下报 161002（AclNN 不支持 x 不分离、y 分离），改为 `split_item=2` 后正常。

**数值检查**（`scripts/check_gemma_grouped_numerics.py`，合成权重，Gemma-4 expert 维度，top-8，T=1/4/16）
- 最大相对误差 8.4e-3，最大绝对误差 3.9e-3（约 1 个 bf16 ulp），所有输出有限。
- 与 HF 的差异来自求和顺序，属于预期。

**ABBA 结果**（`scripts/run_gemma_abba_grouped.sh`，first16，16 slot，npu:1，两组均 `--compact`）

| 运行 | makespan (s) | 正确数（agent_protocol） | token 数 |
|---|---|---|---|
| A1（HF 循环） | 705.1 | 12/16 | 7493 |
| B1（分组） | 377.1 | 12/16 | 7493 |
| B2（分组） | 358.7 | 12/16 | 7493 |
| A2（HF 循环） | 674.7 | 12/16 | 7493 |

- A 均值 689.9 s，B 均值 367.9 s，约 −47%，区间不重叠。
- 文本：A1=A2 16/16；B 与 A 的文本 15/16 一致。唯一差异是 231，在 char 1380 处 "interchangeably"→"synonymously"，与 E-G21 中翻转的 231 同题（平局）。
- 峰值显存：allocated 59.5 GB，reserved 61.4 GB（与 A 相同）。

**判定**：接受（first16）。满足"wall 收益 ≥2% 且正确数不降"。

### 8.3 未解决与注意事项

- **只验证了 first16（16 题）**，未验证 full128。47% 的收益远超估计，需要 full128 ABBA 确认后才能当作稳定收益。
- 正确数按 agent_protocol 评分，未用 `src/scorer.py` 重评。E-G16、E-G21 的 fresh-v1 为 14/16，不能与本节 12/16 直接比较。
- A 组绝对耗时在不同批次间漂移（本批 690 s，E-G21 批次 793 s），只比较同一 ABBA 内的配对。
- 231 的分叉未做 token 级对齐，只记录了文本分叉位置。
- 之前的 `split_item` 错误与本次修复不影响结论，但 `experts_grouped` 目前只在合成权重上做过数值检查，尚未在真实模型的单层上对照。

### 8.4 产出文件（追加）

| 文件 | 内容 |
|---|---|
| `src/gemma_cb.py` | `experts_grouped()`，`GemmaCB(grouped=)`，`run_cb(grouped=)` |
| `src/gemma_run.py` | `--grouped-experts` 参数，summary 中记录 `grouped_experts` |
| `scripts/check_gemma_grouped_numerics.py` | 分组与 HF experts 的数值对照 |
| `scripts/run_gemma_abba_grouped.sh` | 本次 ABBA 驱动脚本 |
| `results/gemma/abba_grouped/{A1,B1,B2,A2}/` | E-G22 结果 |
| `logs/gemma_abba_grouped_*.log`，`logs/gemma_abba_grouped_driver.log` | E-G22 日志 |
| `experiments/gemma_records.jsonl` | 追加 E-G22 记录 |

### 8.5 E-G23：叠加 fast-norm（first16，已接受）

- 配置：`--cb --slots 16 --compact --grouped-experts`，叠加 `--fast-norm`（433 个 norm 被替换）。
- ABBA（B1 → C1 → C2 → B2）：B 为 367.1 s 与 382.2 s（均值 374.7 s）；C 为 244.0 s 与 247.4 s（均值 245.7 s）。约 −34%，区间不重叠。
- 正确数：四次均为 12/16，无翻转。
- 注意：C 组生成 token 为 7219，B 组为 7493，生成路径改变（与 E-G06 一致，仅 5/16 文本相同）。因此按 token 归一化：B ≈ 20.0 tok/s，C ≈ 29.4 tok/s，约 +47%。
- 仍然只有 first16，未验证 full128，正确数为 agent_protocol 评分。
- 当前 first16 ≈ 246 s，距离 Qwen 对应的约 100 s（first16）还有明显差距。

### 8.6 E-G24：当前最佳配置的分段 profile（N=16，first16 前 16 题）

- 配置：`--cb --compact --grouped-experts --fast-norm`，N=16 全部活跃，32 步。
- 无 hook 单步：293.7 ms（约 54 token/s 的稳态上限，不含 prefill、尾部与调度）。
- 分段（带同步，上界）：attn 115 ms（38%），experts 74 ms（25%），router 44 ms（15%），dense mlp 21 ms（7%），其余 45 ms（15%）。
- 结论：attention 现在是最大项。router 每层约 1.5 ms，偏高，疑似小算子的 host 启动开销。
- 候选 1（已测）：GQA 不 expand K/V（`GemmaCB(nocopy=True)`）。与 expand 路径 logits 最大差 0.0、argmax 0 翻转（16 步）；单步 273.8 → 267.6 ms，约 −2.3%。收益很小，未达到 2% 门槛的稳定性要求，未做 ABBA，暂不接受为优化。
- 下一步候选：`_attn` 里 `self.K[li][sl, :, :Tact]` 的高级索引会复制整段 cache；当 sl 为全部 slot 时可改用切片视图；并检查 router 的小算子合并。

### 8.7 E-G25：slice_kv（K/V 读视图，替代高级索引 gather）

- 改动：`GemmaCB(slice_kv=True)`。仅当所有 slot 全部活跃且按顺序时（`sl is self.ar`），`_attn` 直接切片 `K[li][:, :, :Tact]`，不再复制；`run_cb` 的 compact 分支在全活跃时传 `eng.ar`。启用：`--slice-kv`。
- 等价性（`scripts/check_gemma_nocopy.py`，16 步，N=16）：logits 最大差 0.0，argmax 0 翻转。
- 单对计时：gather 310.9 ms，slice 302.0 ms（约 −2.9%）。单对噪声较大，未作为依据。
- ABBA（`scripts/run_gemma_abba_slice.sh`）：A = compact + grouped + fast-norm（E-G23 C 配置），B = A + slice-kv，first16，npu:1，顺序 A1 → B1 → B2 → A2。结果待填入。
- 结果（已完成）：A1 244.0 s、B1 265.4 s、B2 261.7 s、A2 252.9 s。A 均值 248.5 s，B 均值 263.5 s，B 慢约 6%。正确数 12/16 全部一致，文本 16/16 全部一致。
- **判定：否决**。等价性通过，但 device 上更慢。单对计时（302 ms vs 311 ms）方向相反，证明单对不可靠。默认关闭，`--slice-kv` 保留用于复现。
- 已写入 `experiments/gemma_records.jsonl`（E-G25，decision: rejected）。

### 8.8 E-G26：fused router（折叠 scale 常数）— 已否决

- 改动：`router_fused()` 把 `scale * scalar_root_size` 折叠成一个向量，其余（norm、proj、softmax、topk、归一化、per_expert_scale）同 HF。启用：`--fast-router`。
- ABBA（`scripts/run_gemma_abba_router.sh`，first16，npu:1，stack 配置）：A1 246.6 s、B1 368.5 s、B2 358.0 s、A2 246.9 s。A 均值 246.8 s，B 均值 363.3 s，B 慢约 47%，区间不重叠。
- 正确性：B 为 13/16（比 A 多 231），但文本仅 6/16 与 A 一致，token 数 7284 vs 7219，属于生成路径改变，不计为正确性收益。
- **判定：否决，默认关闭。** 未查清变慢原因（未做分段 profile，未检查是否引入额外同步或 dtype 转换）。已写入 `experiments/gemma_records.jsonl`（E-G26，decision: rejected）。

### 8.9 E-G27：fused router 变慢的原因（诊断，未找到）

- 单层微基准（16 行、真实层权重、200 次）：HF router 0.887 ms，fused 0.835 ms。fused 并不更慢。
- 两者 expert 选择在平局行上不完全一致，权重最大差 4.4e-3，fold 与 norm 均为 bf16，无额外 dtype 转换。
- 分段 profile（32 步）：fused 整步 327 ms，stack 基线 294 ms。但 attn 段（代码未改动）也高 22 ms，router 段只高 5.5 ms。整体偏慢约 10%，说明本次单次运行噪声较大。
- **结论**：router 计算不是 E-G26 那 47% 变慢的原因，原因未确定。已写入 `experiments/gemma_records.jsonl`（E-G27，decision: diagnostic）。
- 下一步若要查：按 task 拆 makespan（decode 步数、tool 时间、prefill 时间），而不是只看单步段。

---

## 9. 交接摘要（截至 2026-11-16 08:20，开新窗口用）

**当前最佳配置（first16，16 slot，npu:1）**：`--cb --slots 16 --compact --grouped-experts --fast-norm`。约 247 s（同批 ABBA A 组均值 246.8 s），正确数 12/16（agent_protocol 评分，未用 fresh-v1 重评）。

**目标**：Qwen 全量 128 题 makespan 400 s，换算为 Gemma first16 约 100 s。当前约 2.5 倍差距。全量 128 尚未用当前配置测过。

**本窗口的接受与否决**

| ID | 改动 | 结论 |
|---|---|---|
| E-G22 | grouped experts（`npu_grouped_matmul`） | 接受：−47%（first16） |
| E-G23 | 叠加 fast-norm | 接受：−34%（first16） |
| E-G24 | profile（attn 38%、experts 25%、router 15%）；nocopy 单对 −2.3% | 诊断；nocopy 未做 ABBA，未采纳 |
| E-G25 | slice_kv（K/V 切片视图） | 否决：慢 6%；默认关闭 |
| E-G26 | fused router（折叠 scale） | 否决：慢 47%，且改变生成路径（231 翻转） |
| E-G27 | 查 E-G26 的原因 | 未找到：router 微基准 fused 更快（0.835 vs 0.887 ms）；整步偏慢约 10% 但 attn 也变慢，噪声较大 |

**未解决的问题**
- E-G26 的 47% 变慢原因未知，router 计算不能解释。下一步：按 task 拆 makespan（decode 步数、tool 时间、prefill 时间）。
- 尚未测 full128 的当前配置。
- E-G22/E-G23 的收益只在 first16 上验证，需要 full128 ABBA 确认。
- 正确数未用 `src/scorer.py` 重评。

**代码位置**
- `src/gemma_cb.py`：`experts_grouped`、`router_fused`、`GemmaCB(grouped, nocopy, slice_kv, fast_router)`、`run_cb(...)`
- `src/gemma_run.py`：`--cb --compact --grouped-experts --fast-norm`，以及可选的 `--slice-kv`、`--fast-router`
- 驱动脚本：`scripts/run_gemma_abba_{grouped,stack,slice,router}.sh`
- 诊断脚本：`scripts/probe_gemma_stack_segments.py`、`scripts/probe_router_fused.py`、`scripts/check_gemma_nocopy.py`、`scripts/check_gemma_grouped_numerics.py`
- 记录：`experiments/gemma_records.jsonl`（E-G01 至 E-G27，28 行，已验证可解析）

**不要做的事**
- 不要只看单步时间。E-G26 说明单步段和 makespan 可能不一致。
- 不要用单对计时决定采纳（E-G25 的单对与 ABBA 方向相反）。
- 不要把 agent_protocol 的正确数当作最终结果。

---

## 10. 追加记录：算子合并与图路线（E-G28 至 E-G39，2026-11-16）

记录文件同步：`experiments/gemma_records.jsonl`（E-G28 至 E-G39）。

### 10.1 当前算子数（E-G33 至 E-G35 之后）

- 配置：`--cb --slots 16 --compact --grouped-experts --fast-norm --fast-norm-noscale --fused-rope`，N=16，Tact=256。
- 用 `scripts/count_gemma_step_ops.py` 统计：**2722 个/步，约 91 个/层**。
- 主要类别：`npu_rms_norm` 331、`mul` 246、`linear` 236、`unsqueeze` 222、`reshape` 210、`transpose` 152、`index` 150、`add` 120、`to.dtype` 105。
- 每层 MoE 路由有一串小算子（`argsort`、`bincount`、`cumsum`、`index_add_` 等，每层各 1 次）。

### 10.2 E-G36：fused FIA + fused geglu（first16）— 否决

- 改动：`GemmaCB(fia=, geglu=)`。FIA 用 `npu_fused_infer_attention_score` 替代 QK/mask/softmax/PV；geglu 用 `npu_geglu` 替代 MoE 的 `chunk + gelu + mul`，构造时交换 `gate_up_proj` 两半（`npu_geglu` 激活的是后半段）。
- 结果：A 组（基线）12/16，192 s；B 组（fia+geglu）1/16，127 s。B 组变快是因为输出错误，不算加速。
- 算子数 2722 → 2332（−14%）。
- 单层检查：MoE 块对 grouped 最大差 0.0625（bf16 ulp）；单算子 FIA 对 fp32 参考误差 0.003；整网随机状态探针 logit 差 44–55，argmax 一致率 0.125–0.25。

### 10.3 E-G37：隔离 FIA 与 geglu — 定位为 geglu

- 脚本：`scripts/run_gemma_isolate_fia_geglu.sh`（各自单独开启，基于 E-G35 配置，first16）。
- F（只开 fia）：12/16，197 s，文本与基线一致 10/16，无速度收益。
- G（只开 geglu）：3/16，149 s，文本与基线一致 0/16。
- **结论：破坏来自 geglu，不来自 fia。** geglu 的错误原因未查（按用户要求暂不深究）。两者都默认关闭。
- 过程中出现一次执行错误：一条命令中误加了 `(until false; do :; done)` 死循环，实验脚本未运行，已停掉该进程后重新启动。

### 10.4 E-G38：nocopy（GQA 不复制 K/V）— 否决

- 脚本：`scripts/run_gemma_abba_nocopy.sh`（ABBA，A = E-G35 B 配置，B = A + `--nocopy`）。
- A1 199.6 s / A2 199.4 s（12/16，文本一致 16/16）；B1 199.5 s / B2 194.2 s（12/16，文本一致 9/16，token 7160 vs 6759）。
- wall 无收益（约 −1.3%，在噪声内），生成路径改变。**否决，默认关闭。**
- E-G24 的单对计时（−2.3%）未被 ABBA 支持。

### 10.5 MoE router 小算子（诊断，未改动 src）

- HF router：`norm → mul(scale) → mul(root) → linear → softmax → topk → sum → div → index → mul`，每层约 10 个算子。
- 检查：`scripts/check_router_gating_fused.py`。`npu_moe_gating_top_k_softmax` 把 softmax + topk（2 个算子）合并为 1 个；专家集合 16/16 行一致，归一化权重最大差 0.00082。
- 整步只省 30 个算子（约 1.1%），且不包含 norm、scale 与 per_expert_scale。**判断：收益过小，不做。** 未写入记录。

### 10.6 NPUGraph 图路线（E-G30、E-G31、E-G39）— 停止

- E-G30（`results/gemma/graph_step/result_INVALID_replay_noop.json`）：回放 0.13 ms，32/32 步与 eager 不一致，哨兵测试确认 replay 未重写输出。最小 matmul 探针同样失败。
- E-G31：侧流预热 + `torch.npu.graph` 最大差 258；`npugraph_ex` 输出含 nan。
- 排查 transformers 版本：`vllm-ascend-py310` 中 transformers 4.57.6 无 `gemma4` 模块，测试未执行。
- **E-G39**：新建独立 venv `/remote-home/S45160/venvs/gemma-tf5`（`--system-site-packages`，继承 torch 2.9.0 / torch_npu 2.9.0，单独装 transformers 5.5.4、accelerate 1.15.0）。图回放仍失败（32/32 步不一致，argmax 错 512 次，0.06 ms 为无效回放）。
- **结论**：失败不是 transformers 版本造成的，在 torch_npu 2.9.0 与 2.9.1 上都可复现。更可能是 CANN 或 NPUGraph 运行时问题，尚未用其他 CANN 版本验证。两个环境都未修改。

### 10.7 本节新增产出文件

- 脚本：`scripts/probe_gemma_geglu_fia.py`、`scripts/check_geglu_fia_ops.py`、`scripts/check_geglu_expert_block.py`、`scripts/run_gemma_abba_fia_geglu.sh`、`scripts/run_gemma_isolate_fia_geglu.sh`、`scripts/run_gemma_abba_nocopy.sh`、`scripts/check_router_gating_fused.py`
- 结果：`results/gemma/abba_fia_geglu/`、`results/gemma/isolate_fia_geglu/`、`results/gemma/abba_nocopy/`、`results/gemma/graph_step_tf5venv/`
- 日志：`logs/gemma_abba_fia_geglu_*.log`、`logs/gemma_isolate_{F,G}.log`、`logs/gemma_abba_nocopy_*.log`、`logs/graph_step_*.log`
- 代码（均默认关闭）：`src/gemma_cb.py` 新增 `fia`、`geglu`、`nocopy` 参数；`src/gemma_run.py` 新增 `--fia`、`--geglu`、`--nocopy`。

### 10.8 当前结论与下一步

- **当前最佳配置不变**：`--cb --slots 16 --compact --grouped-experts --fast-norm --fast-norm-noscale --fused-rope`（E-G35，first16 约 190–245 s，正确数 12/16）。
- 本节所有新候选（fia、geglu、nocopy、router 融合、NPUGraph）均否决或停止，默认关闭。
- 仍未解决：geglu 为什么破坏输出；NPUGraph 回放为何静默失效（候选原因是 CANN/运行时）。
- 剩余优化方向集中在 PyTorch 层的小算子（`mul`、`to.dtype`、`index`、`unsqueeze`/`reshape`），需要逐个确认来源后再做。
- 全量 full128 仍未用当前配置测过，所有 ABBA 结论只限于 first16。

---

## 11. 追加记录：prefill 路线（E-G40 至 E-G46，2026-11-16）

记录文件同步：`experiments/gemma_records.jsonl`（E-G40 至 E-G46）。

### 11.1 结果概览

| ID | 内容 | 结论 |
|---|---|---|
| E-G40 | grouped experts 用于 HF batch-1 prefill（first16 ABBA） | wall −39%，区间不重叠；正确数不降（B 新增 231）；**候选，待 full128 ABBA 验证**（未接受） |
| E-G41 | 跨轮复用 KV（只 prefill 后缀） | 收益约 10–15%，未实现 |
| E-G42 | grouped prefill 的速度与数值（随机 token） | 速度 −30% 至 −55%；单层真实路由输入一致（≤0.001），整网 logits 在 T≥256 时 argmax 改变，原因未查 |
| E-G43 | 计数器 probe 的 A/B ABBA（B→A→B→A） | 单步差在两种顺序下复现（A 226–256 ms，B 131–132 ms）；解码侧差距主要来自单步成本 |
| E-G44 | decode 中 experts 张量元数据 | A、B 完全一致（0 处差异） |
| E-G45 | experts 段分段计时 | experts 段只占整步差的约 25% |
| E-G46 | 整步分段计时 | 差距分散在 attn、experts、router、dense mlp 与其余各段，无单一热点 |

### 11.2 评分与判定说明

- E-G40 的 summary.json 中 `correct`（A 12、B 13）是 agent_protocol 口径。fresh-v1 重评为 A 13/16、B 14/16。之后以 fresh-v1 为准。
- E-G40 的判定标准有两条，需明确区分：E-G35 要求"正确题集合不变"，E-G22、E-G23 要求"正确数不降"。B 不满足前者，满足后者。本次按后者记为候选，等待 full128 验证。
- 之前 "B 的正确率不能接受" 的说法是错误的。B 的正确数不降，主要问题是提速来源未知，以及文本变化较大（9/16）。

### 11.3 机制的状态（未解决，不阻塞主线）

- 解码路径的张量元数据在 A、B 之间一致（E-G44）。
- experts 段约解释 25% 的差距（E-G45）；差距分散在所有段（E-G46）。
- 候选假设：HF prefill 之后的显存分配器状态。未验证（可用 `torch.npu.empty_cache()` 或比较 reserved/allocated）。
- 决策：机制研究不作为接受或拒绝的前提。只有 full128 结果不稳定时再回头研究。

### 11.4 下一步

1. 在 full128 上做 A/B ABBA（A = 当前最佳配置，B = A + `--prefill-grouped`），以 fresh-v1 重评，比较 makespan、正确集合与文本变化。**需要用户确认后再启动**（占用较长 NPU 时间）。
2. 机制研究作为可选的短实验，不阻塞 1。

---

## 12. E-G47：prefill-grouped 的 full128 ABBA（2026-11-16，已接受）

- 脚本：`scripts/run_gemma_abba_prefill_full128.sh`，顺序 A1 → B1 → B2 → A2，npu:1。A = `--cb --slots 16 --compact --grouped-experts --fast-norm --fast-norm-noscale --fused-rope`，B = A + `--prefill-grouped`。
- wall：A1 1183.8 s，B1 657.0 s，B2 666.0 s，A2 1167.8 s。A 均值 1175.8 s，B 均值 661.5 s，**约 −44%，区间不重叠**。
- fresh-v1（`src/scorer.py`）：A1 109/128，B1 110/128，B2 110/128，A2 109/128。
  - B 新增 231、1011，丢失 950。
  - A 组两次的错题集合一致；B 组两次的错题集合一致。
- 文本：A1=A2 128/128，B1=B2 128/128，A 与 B 之间 55/128 一致（文本分叉较多，与 E-G40 的 first16 一致）。
- **判定：接受**（wall 收益远超 2% 门槛，正确数不降）。
- 注意：这是当前最佳配置的候选优化，只有这一组 ABBA。fresh-v1 之前的基线是 108/128（E-G01）；本批次的 A 组 109/128 不能直接与之比较，只比较同批次配对。
- 记录：`experiments/gemma_records.jsonl` 追加 E-G47。

### 下一步
1. 把 `--prefill-grouped` 加入当前最佳配置的默认参数（或在文档中注明必须开启），之后的实验都以此为 A 组。
2. attention 段的算子减少（K/V 写入、GQA expand）：仍需 first16 ABBA 验证，未开始。
