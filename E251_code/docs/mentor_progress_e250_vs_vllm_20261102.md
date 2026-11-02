# Qwen3.5-27B Ascend 910B3 推理优化进展

**汇报日期：2026-11-02**  
**主题：HF/E250 优化方法与 vLLM-Ascend 对照**

## 汇报摘要

本项目面向单卡 Ascend 910B3 上的 Qwen3.5-27B，优化固定 128 道 GSM8K 题的两轮 Agent 工作流。模型先生成 Python 工具调用，执行工具后再输出答案；主要指标是完整工作流的墙钟时间。

当前 HF/E250 的一轮代表成绩为 **279.848 秒**。

按 E251 的题目顺序、B16 首轮批次和串行 Python 工具执行配置，vLLM-Ascend 的单次 workflow 时间为 **280.878 秒**，比选取的 E250 单次 279.848 秒慢 **1.030 秒**；但这次 vLLM 在 task304 没有生成工具调用，只有 **127 次工具调用**，v3 得分 **121/128**，而 E250 完成 128 次工具调用、得分 **122/128**。因此，两个单次数字可以作为运行时间参照，不能据此断言两套方法在相同工作量和质量下谁更快。

## 工作负载与环境

| 项目 | 固定设置 |
|---|---|
| 硬件 | 单卡 Ascend 910B3，物理设备0 |
| 模型 | 本地 Qwen3.5-27B，BF16 |
| 数据集 | `data/gsm8k_128.jsonl`，128题，SHA256=`8ad965a8bea67b7a06454324427cebc1233a48624187dcb519eb0190b471fae4` |
| Agent 协议 | concise 原生 Qwen tool-call；首轮 Python 工具调用，工具返回后输出 `\boxed{N}`；关闭 thinking |
| 解码 | Greedy，首轮最多512个新 token，最终答案最多12个新 token |
| 批次与顺序 | B16，共8批；对照运行使用 E251 冻结的原始 E100 题目顺序 |
| 工具 | 每题真实执行 Python；本次 vLLM 对照设 `tool-workers=1`，与 HF runner 串行工具执行对齐 |
| 计时 | 模型就绪后的完整 resident workflow，包括推理、工具和状态管理；加载/引擎初始化时间另报 |

HF 使用独立 `ascend-e2e` 环境（Python 3.12、PyTorch 2.9.1、torch_npu 2.9.1、Transformers 5.5.4）。vLLM 使用独立 `vllm-ascend-py310` 环境（Python 3.10、PyTorch/torch_npu 2.9.0、vLLM/vLLM-Ascend 0.18.0）。两边环境不能混装，因此这是两套完整 runtime 的端到端比较，不是相同软件栈下单独比较某个 kernel。

## HF 方法：E250

原始 HF Agent 工作流按 B16 执行：第一轮把 system/user prompt 送入 `generate()`，解析并执行 Python 工具调用，再将 assistant/tool 消息追加到对话；第二轮重新模板化、tokenize 完整对话，再从新 prompt 调用 `generate()`。这样会重新处理第一轮已经处理过的 prompt 和生成前缀。

E250 将第二轮改成“保存第一轮每行状态，只追加新内容”，并为 Qwen3.5 GDN 的不等长 suffix 实现历史状态感知的批量更新：

1. **常驻模型并处理完整128题。** 任务分成8个 B16 批次，单次加载模型后驻留 NPU；第一轮生成、工具协议和贪心解码设置不变。
2. **在各行 EOS 边界保存续算状态。** 保存每行的 attention KV、GDN convolution state、GDN recurrent state，以及 mask/position 等元数据到 NPU。快照边界排除尚未被模型消费的 EOS sampled token。
3. **只计算第二轮新增 suffix。** 工具返回后仍重建完整第二轮 prompt，但逐行检查它是否以快照已消费 token 为前缀；确认后只 tokenize、打包新增加的 assistant/tool suffix，并以历史 KV 和 recurrent/conv state 作为初始状态续算。
4. **按每行有效长度处理 GDN suffix。** T>1 recurrent suffix 使用 chunk64 recurrence 和真实历史 initial state；inactive/padding 步不改变 state，输出为零。T>1 causal convolution 将历史窗口与完整新 suffix 一起计算，只保留 suffix 输出，并按各行有效长度提交新的历史窗口。T=1 decode 仍使用原单 token 路径。
5. **前缀不匹配就整批回退。** 如果任一行无法证明 prefix 匹配，就对整批16题重新执行完整第二轮 prompt，不混用不确定的旧状态。full128 中 batch2/task99、batch3/task785 触发回退，其余6批复用快照。

因此 E250 的工作量减少来自两处：跨工具轮复用 attention/GDN 历史状态，避免第二轮重算已处理前缀；对新增 suffix 使用 chunk recurrent 和整段 causal convolution，避免逐 token 重复处理历史窗口与状态提交。它不改模型权重、BF16精度、prompt、工具行为或生成上限。

实现入口为 [`e251_full_workflow.py`](../scripts/e251_full_workflow.py)（常驻两轮流程、EOS快照、prefix校验和回退）、[`e250_chunk_history.py`](../scripts/e250_chunk_history.py)（chunk recurrent）和 [`e248_vector_history_conv.py`](../scripts/e248_vector_history_conv.py)（整段suffix convolution）。这套方法未改默认 runner、依赖或 site-packages。

## 算子优化尝试与结果

项目并非没有在算子层面做改进。我们测过归一化/激活融合、GDN recurrent、causal convolution、MLP 权重布局和 decode graph。经验是：kernel 单独更快不代表实际 workflow 更快；算子还必须保持调用边界上的输入布局、BF16 舍入、逐行状态提交和完整 token/工具行为。

| 算子/路径 | 尝试内容 | 结果与未接纳原因 |
|---|---|---|
| RMSNorm / Gated RMSNorm / SwiGLU | 用 Ascend fused RMSNorm、Gated RMSNorm 和 SwiGLU 替换模型中的框架算子；部分路径做 prefill-only 和 decode-only 隔离 | 这是少数确实进入最终 HF 路径的底层优化。E084 的 Gated RMSNorm/SwiGLU 组合将 full128 从 E069 的 280.736 s 降到 271.120 s，v3 同为122/128，作为当时的有效优化保留。早期 E034/E058/E059 的不同混合路径则因质量或 workflow 回退淘汰；不能把所有 RMSNorm/SwiGLU 变体视为等价实现。见 [`OPTIMIZATION_LOG.md`](../../OPTIMIZATION_LOG.md)。 |
| GDN 单 token recurrent | 将 Qwen GDN 的 recurrent 更新换成 Ascend fused recurrent kernel；另测 baseline-prepared 单 kernel wrapper | E200 在 B16 改变6题首轮 token、2题工具文本，且 allocated memory 超限；E201 虽结构性 state 检查通过，workflow 仍有轨迹差异，完整 wrapper ABBA 候选慢3.45%。主要问题是融合算术/舍入与原实现不完全一致，且该实现没有更快。 |
| Causal convolution 与 layout 替换 | 测 Ascend custom conv、ND fused conv、不同 layout/状态提交，以及 ragged suffix kernel | 有些 custom conv 在调用处能执行，但 state ownership、padding 行和 caller cache 写回必须完全正确；E199 custom-conv-only 已导致5题轨迹变化。E144 的 ND fused conv 虽减少 trace 中的转换/kernel 数，完整 decode 却慢约37%。E265 直接 ragged Triton conv 的精确完整 wrapper 为 baseline 的0.559倍；后续 channel-tiled kernel 局部 wrapper 快6.28倍，但 E268 full first16 ABBA 反而慢13.65%。局部 kernel 收益没有转成完整批次收益。 |
| Prefill causal-conv Triton | 针对真实 B16、T>1 suffix 写 padded ragged 和 channel-tiled kernel | 对捕获的连续输入，早期局部数据看起来有收益；恢复生产输入的真实 stride 后，E275 C8/T64 完整 wrapper 从53.19 ms变为548.66 ms，慢10.30倍。E276 其他 channel-major tile exact，但比 baseline 慢7.53–8.54倍。访存布局和隐式转换成本决定了 kernel 在真实模型中的表现。 |
| GDN recurrent 历史续算 | 测 active-mask 复用、per-token gather/scatter、行排序/打包、chunk64 续算 | E203 的逐步压紧 exact 但慢10.95%；E205 的排序/打包局部有收益，first16 ABBA 最终仅快1.297%，小于候选波动3.523%。E260 单独换 recurrent wrapper 的收益0.813%，一组回退且波动更大。单独拆开看都没过门槛；chunk64 recurrent 与整段历史卷积组合后成为 E250，并通过 full128 exact/ABBA 验收。 |
| Decode GDN tail fusion | 融合 QK normalization、gate、recurrent update/output 和 BF16 state commit，减少中间 tensor 与 launch | E279 的完整 wrapper ABBA 每个 case 都慢约67–80 μs。E281/E293 改善 tile 和显式 BF16 rounding 后，局部 wrapper 快约311–326 μs/调用；E286 first16 曾快15.04%。但该实现使 task304 在512 token内无法 EOS；E299 full128 candidate 均值比 native baseline 慢3.121%，v3为121而不是122。局部 fusion 速度不能弥补数值偏差引起的长输出和回退成本，因此不接纳为 full128 优化。 |
| Graph / MLP layout 与量化 | 测 recurrent graph、临时 decode graph、MLP NZ/persistent layout、INT8 weight-only、gate/up 重排 | graph 有局部 replay 结果但完整 workflow 未稳定变快，部分路径超显存；NZ 全 generation 慢3.718–7.324%；INT8 单层 M=1/16 分别慢约8.4%/3.2%；gate/up 重排可测预算约为短 decode window 的0.4–0.7%，低于2%筛选线。收益不足或内存/正确性不满足，故停止这些具体实现。 |

### 为什么算子实验经常无法接纳

1. **BF16 的细小差异会沿生成链放大。** Q/K normalization、beta/gate 准备、recurrent reduction 的舍入顺序或 GEMM shape 改变，可能只产生很小的中间误差；但 greedy 解码遇到接近的 top-1 logits 时会选出另一个 token，后续状态和 prompt 随之改变。E299 的 task304 就是局部 tail 误差最终变成未 EOS、少工具调用和更长 workflow 的例子。
2. **实际调用边界比 kernel 名字复杂。** GDN 的输入常是非连续 view，B/C/T 维布局、每行有效长度和 caller-owned conv/recurrent state 都有语义。若只返回 output、临时 state 没有写回，或 padding 行被更新，即使 kernel 能运行也不是可替换实现。E184/E185 的 state commit 诊断展示了这一类问题。
3. **kernel 之外还有转换、准备、分配和同步成本。** 一个 fused kernel 可能减少算子数，却引入 transposition、contiguous copy、workspace 或更差的访存。E144 和 E275 说明“少 kernel/少 conversion”不自动等于更快；必须计完整 wrapper。
4. **优化点可能太小，或只覆盖少数调用。** 完整 workflow 里某个算子占比有限时，局部数倍加速也可能只省很少端到端时间。E205、E260/E261、gate/up 重排均显示净收益未超过波动或投入门槛。
5. **测量输入必须和生产一致。** 连续化的 capture、重复同一层、短 prompt、first16 smoke 都可能绕开真实 stride、长尾 task、后续状态和 batch 交互。E268 first16 的正确性通过仍没能预测 full128 tail candidate 的表现。
6. **运行时能力也会限制候选。** 有些 ragged causal-conv API 在 Python 层有 schema，但 CANN 缺少执行符号；当前 HF 环境的 Triton NPU driver 也不可用。另一个环境中的探针不能证明它能接入 `ascend-e2e` workflow。

因此目前保留的不是“所有局部指标最好”的 kernel，而是通过独立 state oracle、真实调用 wrapper、原始 full128 轨迹和 matched ABBA 的组合路径。局部测试用于排除错误和估算预算；最终接纳看完整任务是否更快且完成同等工作。

### 正确性与资源结果

- 独立 CPU/NPU per-row recurrence oracle 覆盖实际不等长 suffix、混合/全活动行、decode 和 resumed row；inactive state 保持不变，padding 输出为零，输入 state 不被修改。
- 整段 suffix convolution 的输出和最终 state 与串行 E176 参考及独立历史窗口构造 exact。
- full128 的 15,961 个生成 token、128 次工具输出和最终预测与冻结 E100 轨迹完全一致；scorer v3 为 **122/128**，失败集合仍为 `{1161, 423, 952, 1035, 552, 1048}`。
- 本次 full128 allocated peak 为 **59,972,823,040 B**，低于 **62,094,008,832 B** 上限；reserved peak 为 **61,863,886,848 B**。

## HF E251 重跑数据

2026-11-02 在物理设备0、冻结 `ascend-e2e` 环境中重新执行完整 E180/E250/E250/E180 ABBA。每轮使用新进程并重新加载模型。加载时间单列，不计入 workflow makespan。

| 顺序 | 路径 | workflow | 模型加载 | 工具调用 | v3 | allocated peak |
|---:|---|---:|---:|---:|---:|---:|
| 1 | E180 baseline | 289.411 s | 10.654 s | 128 | 122/128 | 59,972,823,040 B |
| 2 | E250 candidate | 282.915 s | 10.419 s | 128 | 122/128 | 59,972,823,040 B |
| 3 | E250 candidate | **279.848 s** | 10.353 s | 128 | 122/128 | 59,972,823,040 B |
| 4 | E180 baseline | 285.242 s | 10.457 s | 128 | 122/128 | 59,972,823,040 B |

ABBA 汇总：E180 均值 **287.327 s**，E250 均值 **281.382 s**，均值收益 **2.069%**；两组配对收益分别为 **2.244%** 和 **1.891%**，均为正。独立审计通过，候选收益超过两侧观测波动。原 E251 报告中的 E250 均值是 248.202 s；本次同配置重跑明显更慢，因此本报告以本次同轮 ABBA 作为当前环境参考，不把历史 248.202 s 与今天的单次 vLLM 运行拼成同轮比较。

## vLLM-Ascend 对照

为了对齐 E251，本次 vLLM 单次运行使用与 HF 相同的 128 题及原始题目顺序、B16 首轮批次、`max_model_len=2048`、BF16、concise greedy、512 token 上限和串行 Python 工具。显式关闭 prefix caching。运行环境为 `vllm-ascend-py310`，vLLM-Ascend 0.18.0。

| 路径 | workflow | 初始化/加载 | 题数 | token | 工具调用 | v3 |
|---|---:|---:|---:|---:|---:|---:|
| E250 candidate（ABBA均值） | **281.382 s** | 10.386 s | 128 | 15,961 | 128 | 122/128 |
| E250 candidate（代表单次） | **279.848 s** | 10.353 s | 128 | 15,961 | 128 | 122/128 |
| vLLM-Ascend（单次） | **280.878 s** | 131.539 s | 128 | 16,419 | 127 | 121/128 |

vLLM 的 `load_seconds` 包含 tokenizer 和 LLM engine 初始化、torch.compile、warmup 与 ACL graph capture；HF 的约10.4秒主要是模型/tokenizer 加载。两者加载字段不完全等价，应谨慎解释。简单相加后，vLLM 本次 cold total 约 **412.418 s**，HF E250 candidate 的 ABBA 均值加其加载均值约 **291.768 s**；初始化口径差异大，冷启动数字只作参考。

本次 vLLM 的请求输入顺序和首轮 B16 结构与 E251 对齐，128题全部有记录且没有 context violation 或工具异常。不过 task304 没有发出 Python 工具调用，导致该批次第二轮只有15个请求；vLLM 最终 **127 次工具调用**，少于 HF 的128次。v3 得分121/128，新失败为 task304；其余失败集合与 HF 相同。31题的 token 轨迹与 HF 不同，4题的工具输出不同。故 vLLM 时间虽比 E250 代表单次快 **1.030 s（0.37%）**、比 E250 ABBA 均值快 **0.504 s（0.18%）**，但少做了一个工具轮次且质量少1分，这不是同质量同工作量下的有效性能胜出。

本次单次 vLLM 运行没有持续采集 NPU HBM 峰值，因此不报告跨 runtime 显存比较。此前还有一次 vLLM 运行未显式匹配 E251 的题目顺序，该结果不用于本报告比较。

## 结果解读与下一步

- 已有可靠的 HF 增量优化：E250 相对同轮 E180 ABBA 平均快 **2.069%**，质量、轨迹和 allocated memory 护栏通过。
- 按用户指定的代表单次值，HF/E250 是 **279.848 s**，vLLM-Ascend 是 **280.878 s**。两者工作流墙钟接近，但单次比较不能替代完整的稳定性实验。
- 目前不能声称 vLLM 更快：该次 vLLM 少调用一次工具、v3 少1分。也不能声称 HF 在两套 runtime 的统计意义上更快：vLLM 只跑一次，没有 matched repeats，且 runtime 解码轨迹不同。
- 下一步先定位 vLLM task304 的工具调用缺失，再在不改任务、顺序、质量门槛和工具工作量的前提下重新评估。若要形成正式的 runtime 性能结论，至少需要重复 vLLM 运行并报告波动；当前结果保留为单次对照。

## 证据索引

- 原始 E250 full128 ABBA：[`decision.json`](../evidence/decision.json)
- E250 独立汇总与审计：[`independent_audit.json`](../evidence/independent_audit.json)
- E251 原始配置与验收门槛：[`results/e251_contract.md`](e251_contract.md)
- 历史 E251 报告（候选均值248.202 s）：[`results/e251_report.md`](e251_report.md)
- vLLM 原始输出：[`vllm_fair_e251_order_20261102.jsonl`](../evidence/vllm_fair_e251_order_20261102.jsonl)
- vLLM 对 E251 直接审计：[`vllm_fair_e251_order_20261102.audit.json`](../evidence/vllm_fair_e251_order_20261102.audit.json)
- vLLM runner：[`vllm_agent_bench.py`](../scripts/vllm_agent_bench.py)
