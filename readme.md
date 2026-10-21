## 1. 场景与度量

项目按单请求执行、多请求调度、底层算子优化逐步推进，最终目标是在质量约束下缩短固定 128 题离线 Agent 工作流的端到端 makespan。各阶段使用匹配的指标：Phase 1 看单请求 latency 和模型执行成本；Phase 2 看批处理吞吐与 makespan；Phase 3 验证通用算子/内存优化，并回到 Phase 2 工作负载确认端到端收益。中间阶段的进展不能替代最终工作负载验收。

**Agent**
- 求解 GSM8K 数学题：模型先调用 `python` 工具写程序算答案，看到输出后给出 `\boxed{N}`。
- 使用 Qwen 原生 tool-call 格式，非 thinking 模式，greedy 解码。
- 每轮最多生成 512 token，最多 6 轮。
- 128 道固定题目（`data/gsm8k_128.jsonl`），开始时全部可用，相当于离线批处理。

**模型：Qwen3.5-27B**
- 混合架构：48 层 Gated DeltaNet（线性注意力，带循环状态）+ 16 层 full attention（GQA）。
- 词表 248k；bf16 权重约 54 GB。

**硬件**
- 1 × 910B

**指标**
- 主指标：整批题目的 **makespan**（墙钟时间）。
- 护栏：accuracy 不能下降（基线 0.93–0.95）。
- 辅助观察：各部分时间占比、task 延迟分布、GPU busy/idle。
- 利用 NPU 的计算、存储与执行特性，最大化端到端推理性能。


## 2. 硬件摸底
通过可控实验，了解 NPU 的计算、内存、并行与通信特征辅助后续优化流程。

## 3. 参数探索
系统改变影响推理性能的参数，学习参数与延迟、吞吐、内存占用之间的关系。

## 4. 优化轨迹

三个阶段的约束逐步放宽：

| 阶段 | 允许做什么 | 阶段目标与度量 |
|---|---|---|
| Phase 1 | 不写 kernel；可以装库、改 launch 方式 | 优化并理解 batch=1 的 prefill/decode 执行路径；记录 latency、每 token 成本和正确性，为后续批处理提供稳定模型路径 |
| Phase 2 | 同上 | 面向多 Agent 并发和固定 128 题离线批处理，优化吞吐与 makespan；同时监测显存、尾部等待和质量 |
| Phase 3 | **可以写 kernel**，但只做通用优化（不针对 workload 特化）；**不做 MTP / speculative decoding**；面向 910B3 的原生算子与内存优化 | 从 Phase 1/2 的关键路径选择算子或内存候选；局部测量用于验证机制，最终仍以 Phase 2 端到端工作负载验收。逼近设备能力是研究方向，不单独作为项目完成标准 |

## 5. 方法论

1. **Profile**：三层工具。
   - L1：分段计时；
   - L2：`torch.profiler`；
   - L3：使用NPU硬件支持的性能分析工具获取时间线，并结合自定义脚本进行关键路径归因。
2. **Amdahl 选择**：对每个候选，估算它占的比例 `p` 和局部加速 `s`，预期整体加速为 `1/((1-p)+p/s)`；按"收益/工作量"排序。
3. **只做一个改动**，然后做正确性检查：
   - greedy 输出 bit 一致；
   - kernel 对照参考实现；
   - longctx accuracy。
4. **测量**：e2e bench。
5. 把预估、实测和偏差原因记进 `OPTIMIZATION_LOG.md`。

## 6. 通用优化思路
1. 先找真正的瓶颈，再按 Amdahl 定律估算收益。 日志里，短测高估了 cuDNN 冷启动的影响，性能分析器也把主机开销测高了。应在代表性工作负载上拆分预填充、解码、工具调用、调度和数据搬运，并用未插桩的 A/B 运行确认。
2. 减少每步固定开销。 基线每生成一个 token 要调度数千个小 kernel，CPU 发射开销很大。图执行、编译融合、批量提交等方案的共同目标，是把很多细碎工作变成更少的提交；非 CUDA 平台可寻找图捕获、静态执行、批处理或算子融合的等价能力。
3. 减少不必要的数据读写和复制。 日志中的原位 recurrent state 更新、避免 K/V 重复展开、融合中间结果、缓存复用，都在减少搬运。这个方向对所有加速器都有效，但需要核对内存布局、同步语义和目标设备的带宽层级。
4. 提高并行利用率，同时控制尾部等待。 连续批处理、动态调度、按批大小分桶和紧凑化 live rows，减少空转并改善长尾。适合检查的指标包括活跃行数、每步有效工作量、最长任务链和单任务延迟；吞吐提高通常会牺牲部分单任务延迟。
5. 优化数据精度与计算形态，但必须按设备验证。 FP8 在特定 GEMM 形状上显著加速，但不同量化粒度会改变算子选择、速度和批间数值一致性。日志里也有低精度矩阵求逆导致长上下文准确率从 0.945 崩到 0.417 的反例。对其他设备应先查其实际支持的数据格式、累加精度和高效矩阵形状，再做层级误差与端到端准确率测试。
6. 区分内存带宽、计算吞吐、启动延迟和功耗上限。 达到带宽或算力上限后，继续调 kernel 往往收益很小；功耗受限时，缩短一个低功耗阶段甚至可能让整体收益消失。优化目标应是模型运行时的端到端关键路径，而非孤立 kernel 的最好成绩。
7. 验证正确性和测量可信度。 日志强调了病态输入、长上下文测试、固定轨迹 replay、同设备交替 A/B，以及区分数值导致的轨迹变化与实际速度变化。这些方法与硬件无关，迁移时尤其重要。



你是本项目的算子优化工程师。请继续优化当前仓库中的 Qwen3.5-27B 单卡 Ascend910B3 推理路径。

必须遵守以下实验规则：

1. 先阅读：
   - OPTIMIZATION_LOG.md
   - readme.md
   - operator_cases/README.md
   - 与当前候选相关的 operator_cases/*.json
2. 当前参考是 E100：
   - batch16
   - concise protocol
   - BF16
   - greedy
   - 单 NPU
   - 128 题 makespan 265.030128 秒
   - scorer v3 为 122/128
3. 每次只修改一个变量，不同时改变模型、协议、调度和算子。
4. 新候选必须先写出：
   - 算子契约
   - 输入输出 shape、stride、dtype、layout
   - state 和 cu_seqlens 语义
   - 当前瓶颈和理论收益上限
   - 额外显存和数值风险
   - 验证命令
   - 停止条件
5. 按以下顺序执行：
   a. Python/FP32 reference 对照
   b. HF reference 对照
   c. 当前 NPU/vLLM 实现对照
   d. 单阶段候选实现
   e. kernel、wrapper、layout conversion、workspace、launch 分层计时
   f. 固定 prompt 的 logits 和 greedy top-1 对照
   g. E100 首批 16 题首轮 token 轨迹对照
   h. 只有通过后才做 32 题 smoke
   i. 只有 32 题通过后才做 128 题
6. 任意一个 task 的 token 轨迹改变、state 语义不一致、OOM、设备异常，立即淘汰该候选，不扩大测试。
7. 局部 kernel 加速不能作为最终结论。候选必须满足：
   - batch16 token 轨迹不变
   - 稳定收益至少 2%，并超过基线波动
   - peak memory 不超过 E100 预算
   - 128 题 scorer v3 至少 122/128
   - makespan 快于 E100
8. 必须使用 operator_cases/ 中的相似成功和失败案例作为参考，并说明哪些条件可以迁移、哪些不能迁移。
9. 不要重开日志中已经明确淘汰的方向，除非软件栈、硬件接口或实现前提已经发生变化。
10. 完成后更新：
    - 实现代码
    - 验证脚本
    - operator_cases/ 新案例
    - OPTIMIZATION_LOG.md
11. 最终报告必须包含：
    - 修改文件
    - 使用的参考案例
    - 实际命令
    - correctness 结果
    - token 轨迹结果
    - kernel/model/e2e 时间
    - peak memory
    - 是否接纳、淘汰或阻断
    - 下一步建议

先检查当前工作区和已有结果，再决定本轮最值得执行的一个候选。不要只给计划，必须完成能执行的验证。

针对 GDN prefill，可以追加这个候选限定：

本轮只研究 GDN prefill，不改 decode recurrent、不改 Agent scheduler、不改 prompt 协议。

优先顺序：
1. 确认 Qwen3.5 的 q/k/v/g/beta/state layout 和 HF reference 语义；
2. 逐个验证 chunk_cumsum、scaled_dot_kkt、solve_tril、wy_fast、chunk_h、chunk_o；
3. 先做单阶段替换，再考虑 MegaGDN fused kernel；
4. 任何阶段 output、final_state 或下一 token logits 不一致，都停止该阶段；
5. 只有 E100 batch16 首 token 全部一致，才允许继续性能测试。

针对其他算子，可以使用这个更短的单候选模板：

本轮只优化 [算子名称]。

参考案例：
- [case 文件名]

当前实现：
- 文件：
- 调用位置：
- 输入 shape：
- 输出 shape：

请完成：
1. 读取参考案例和已有 benchmark；
2. 写出当前算子契约和收益预算；
3. 实现一个可回退的候选；
4. 编写或复用 reference correctness 检查；
5. 比较 output、state、logits、top-1、kernel 时间、wrapper 时间和 peak memory；
6. 在固定 E100 batch16 首轮上比较 token IDs；
7. 根据门槛将候选标记为 accepted、rejected 或 blocked；
8. 更新 operator_cases/ 和 OPTIMIZATION_LOG.md。

停止条件：
- 任意 token 轨迹改变；
- 数值语义无法解释；
- OOM 或设备异常；
- batch16 稳定收益低于 2%。

请根据 Phase 1 成本归因和 operator_cases 检索结果选择候选。若没有足够证据证明某个算子具有至少 2% 的端到端收益上限，先补齐诊断，不要盲目实现 kernel。