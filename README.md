# 单卡 Ascend 910B3 上的 LLM Agent 端到端加速：Qwen3.5-27B，同一批题从 8365 s 到 410 s

这个仓库记录了一次 **profile 驱动的端到端优化**。场景是一个会调用 Python 工具的数学 agent，在单张 Ascend 910B3 上做完一批题目。

- 代码、实验日志、每一步的测量结果都在仓库里。
- 约束：**推理路径必须是自写 / HF 权重的路径**。不把 vLLM 的 kernel 接进 HF 路径（用户明确拒绝）。vLLM-Ascend 只作为外部参照。
- 学习重点：完整走一遍 **profile → 估算收益 → 实现 → 验证正确性 → 再 profile**，并记录每一次估算与实测的偏差。

## 先看懂这几个词

| 词 | 意思 |
|---|---|
| **makespan** | 从开始到**这一批题全部做完**的墙钟时间。所有对比都只看它，越小越好。 |
| **task** | 一道题。agent 要和模型来回好几轮：模型写代码 → 执行 Python → 模型看结果 → 给答案。 |
| **accuracy** | 答对的比例，用来保证"变快"没有以"变笨"为代价。 |
| **baseline** | 起点：HF transformers 逐题串行跑。 |
| **vLLM-Ascend** | 外部参照：用同一批题跑 vLLM-Ascend batch16。 |

**对比是不是在同一个东西上？** 是的。表中每个数字都是**同一批 128 道题、同一个模型、同一张 NPU、同样的解码规则**（greedy、非 thinking、首轮与后续轮每轮最多 512 token、最多 6 轮、原始 system prompt），只是推理系统不同。

## 一句话结果

**主测试（gsm8k_128，128 道题，fresh-v1 评分）**

| 系统 | makespan | 相对 baseline | 正确数 |
|---|---|---|---|
| baseline（HF，逐题串行，`results/baseline_serial`） | 8365.0 s | 1× | 116/128 |
| HF batch16 + 优化（`results/b16_original_opt`） | 1180.0 s | 7.1× | 116/128 |
| 连续批处理 CBEngine（`results/ablations/continuous_batching/repeat_base_r2`，prefill-group 4） | 755.2 s | 11.1× | 117/128 |
| + fast-GDN prefill（`results/ablations/continuous_batching/repeat_fast_r2`） | 588.2 s | 14.2× | 118/128 |
| + graph decode，prefill-group 1（`results/ablations/continuous_batching/full128_pg1_graph`） | 499.7 s | 16.7× | 118/128 |
| + GDN v2 批量预计算（ABBA 均值，`results/ablations/continuous_batching/abba_full128_{A1,B1,B2,A2}`） | 469.5 s | 17.8× | 118/128 |
| **+ elem-conv（ABBA 均值，`results/ablations/continuous_batching/abba_full128_elemconv_{A,B}*`）** | **410.0 s** | **20.4×** | **118/128** |
| 参照：vLLM-Ascend batch16（`results/vllm_original_b16`，仅参考） | 576.75 s | — | 118/128 |

**当前最优配置**：`--fast-gdn --gdn-v2 --prefill-group 1 --graph-decode --elem-conv`，NPU 0，16 slot。410.0 s 为 ABBA 四次运行的均值（B1 409.9 s，B2 410.2 s；对照组 A1 478.8 s，A2 478.9 s）。四次运行都是 118/128。

结论：**自写路径（HF 权重 + 自写 decode 循环 + 自写 GDN prefill）在同一批题上已快于 vLLM-Ascend 参照（410.0 s vs 576.75 s）。
---

## 目录

1. [场景与度量](#1-场景与度量)
2. [优化轨迹](#2-优化轨迹)
3. [方法论](#3-方法论)
4. [主要经验](#4-主要经验)
5. [还有什么空间](#5-还有什么空间)
6. [仓库结构](#6-仓库结构)
7. [复现](#7-复现)
8. [延伸阅读](#8-延伸阅读)

---

## 1. 场景与度量

**Agent**
- 求解 GSM8K 数学题：模型先调用 `python` 工具写程序算答案，看到输出后给出 `\boxed{N}`。
- Qwen 原生 tool-call 格式，非 thinking 模式，greedy 解码。
- 首轮与后续轮每轮最多生成 512 token，最多 6 轮。
- 128 道固定题目（`/remote-home/S45160/work/910b_e2e_accleration_by_codex/data/gsm8k_128.jsonl`），开始时全部可用，相当于离线批处理。

**模型：Qwen3.5-27B（BF16）**
- 混合架构：48 层 Gated DeltaNet（线性注意力，带 conv 状态与循环状态）+ 16 层 full attention（GQA）。
- 权重约 51 GiB。

**硬件**
- 1 × Ascend 910B3（64 GB HBM），每次只用一张卡（npu:0 或 npu:1），不与其他任务共卡。

**指标**
- 主指标：整批题目的 **makespan**（取 `summary.makespan_seconds`）。
- 护栏：正确数不能下降（基线 116–118/128）。
- 辅助观察：prefill 与 decode 时间占比、每步时间、slot 占用率。

---

## 2. 优化轨迹

按时间顺序，每一步都是"先 profile、再做一个改动、再 A/B 测量"。

| 阶段 | 改动 | 证据（profile 看到了什么） | makespan | 正确数 |
|---|---|---|---|---|
| 0 | 串行 B=1 HF 基线 | — | 8365.0 s | 116/128 |
| 1 | batch16 + 优化（fused RMSNorm、gated RMSNorm、single-step recurrent） | 逐题串行浪费大量算力 | 1180.0 s（无优化 1450.97 s，118/128） | 116/128 |
| 2 | 自写连续批处理 `CBEngine`（slot 制，KV 与 GDN 状态按 slot 存放） | 静态 batch 中已完成的行仍在算（约 53% 行·步浪费） | 755.2 s（prefill-group 4） | 117/128 |
| 3 | 自写 chunk GDN prefill（`gdn_chunk_fast.py`，`--fast-gdn`） | prefill 中 66% 时间在 torch 回退的 GDN chunk 函数上 | 588.2 s（同代码 A/B，−22%） | 118/128 |
| 4 | NPUGraph 捕获 decode 步（`--graph-decode`，共享内存池，512-token bucket） | decode 的 wall-clock 约 90% 花在 Python/launch 侧（device 时间只占一部分） | 570.8 → 499.7 s（pg1，−12.5%） | 118/128 |
| 5 | GDN v2：状态无关的部分批量预计算（`gdn_chunk_fast2.py`，`--gdn-v2`） | prefill 中 GDN chunk 占 66%；v2 把与状态无关的计算合并为一次批量 | ABBA：498.4 → 469.5 s（−5.8%），prefill 149 → 121 s | 118/128 |
| 6 | elem-conv：decode 的深度卷积改为 mul + sum（`--elem-conv`） | 卷积 device 时间高；elem-conv 把每步 device 时间约降 23% | ABBA：478.9 → 410.0 s（−14%） | 118/128 |

### 失败或负收益的尝试（每一项都有测量）

| 尝试 | 结果 | 处理 |
|---|---|---|
| 24 slot（超过 16） | NPU OOM（60.5 GiB 已分配，9.6 MiB 空闲） | 不再尝试，除非改变权重或 KV 布局 |
| prefill 分组（pg4、pg8）与 graph decode 结合 | 无收益（pg4 与 pg1 相差 4 s，在噪声内）；pg4 + graph 需要 expandable segments | 默认 pg1 |
| sync-free（去掉 Python float 标量引起的 host sync） | ABBA 慢约 4%（120.3 s vs 115.5 s） | 否决，保留文件但不启用 |
| shift-conv（decode 卷积的移位累加写法） | 变慢，且与基线不逐位一致（10/16 final text 相同） | 否决 |
| GDN v2 chunk 128 | 比 chunk 64 慢（6.32 ms vs 5.64 ms / 层） | 保留 chunk 64 |
| graph bucket 128（替代 512） | first16 ABBA 仅快 0.9%，低于 2% 门槛 | 不跑 full128，默认保持 512；可用 `CB_GRAPH_BUCKET` 覆盖 |

---

## 3. 方法论

### 3.1 循环

1. **Profile**：分段计时（`profile_decode_wall_breakdown.py`）+ `torch_npu.profiler`（`profile_cb_decode_ops.py`、`profile_prefill_v2_ops.py`），用 `summarise_kernel_csv.py` 按算子汇总设备时间。
2. **估算收益**：对每个候选估算占比与局部加速，预期整体收益。
3. **只做一个改动**，然后做正确性检查：
   - 单元测试对照（`scripts/test_gdn_v2.py`、`scripts/test_sync_free.py`、`scripts/test_cb_engine.py`）；
   - 同 protocol 的 first16 A/B，再推进到 full128。
4. **ABBA 测量**：同卡、顺序运行，按 A-B-B-A 或 A-B-A-B 排列，取均值。
5. 把预估、实测和偏差记入 HANDOFF。

### 3.2 ABBA：把系统速度和噪声分开

- 单次运行的 makespan 有几秒到十几秒的波动（同配置的 eager 运行，最小与最大相差 10 s）。
- 因此每个决定都用 ABBA（至少两对）来做，两组不重叠才算确定的收益。例如 GDN v2 的 A 组（493.9–502.8 s）与 B 组（468.4–470.6 s）不重叠。
- 每次都确认没有其他进程占用同一张卡。

### 3.3 正确性：逐题对比，而不只看总数

- 只看正确数会掩盖翻转：elem-conv 的正确数与基线同为 118/128，但在 1030 和 1137 两题上方向相反（净 0）。
- 因此每个改动都记录：正确题集合的差异、final text 的逐字一致数、以及同配置两次运行是否逐字一致（确定性）。
- 逐字不一致不一定是错误：bf16 下的数值改变会使 greedy 在近似平局处分叉。这些分叉需要用 logit 差来解释（已对 1137、640、1030 做过 margin 分析，margin 在 bf16 精度内）。

---

## 4. 主要经验

1. **先测"成本驱动因素"，而不是"工作量"。** prefill 的 token 数不是决定时间的因素，eager 调用的次数和 host 开销才是。
2. **decode 的瓶颈是 host dispatch，不是 device 计算。** 证据：elem-conv 把 device 时间降了 23%，但 eager 模式下 wall-clock 没有变化。把 decode 步捕获成 NPUGraph 后才看到收益。**优化要对准 wall-clock 步时间，而不是 profiler 的 device 时间之和。**
3. **Python float 标量会触发 host sync。** 在 NPU 上，`x * 0.5` 这类 Python float 运算会同步；device tensor 标量不会。RMSNorm 的 `(1+w)` 每次新算也会同步。缓存为 device tensor 后可消除。
4. **高级索引在 NPU 上很慢。** 用 `index_select` / `index_copy_` 替代，快 5–20 倍。
5. **graph 捕获需要共享内存池。** 每个 graph 一个私有池时，下一次 prefill 会 OOM。
6. **allocator 碎片会导致 OOM。** 图模式 + prefill 分组时，需要 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`。现在由 `cb_run.py` 在 `--graph-decode` 时自动设置。
7. **内存决定并发上限。** 权重 ~51 GiB，KV + GDN 状态 ~6.3 GiB，prefill 瞬时 ~2 GiB。16 slot 是上限。
8. **大收益来自架构，小改动收益递减。** 连续批处理（约 2×）和 graph decode（约 12%）是主要收益，之后的 GDN 与卷积改动各为 5–15%。
9. **测量方法本身就是工程。** 同一配置跑两次就可能差 10 s，先保证 ABBA 与同卡条件，再谈优化。
10. **一个运行脚本要防止"同卡双开"。** 本次就出现过：误报的占用检查导致 ABBA 中有三组失败，需要重跑。占用检查现在只匹配 python 解释器进程。

---

## 5. 还有什么空间

**当前时间分布（full128，elem-conv 开启，ABBA 均值，由 summary 推算）**
- makespan ≈ 410 s，其中 prefill ≈ 134 s（约 33%），decode ≈ 276 s。
- decode 每步约 97 ms（2847 步）；不带 elem-conv 时约 124 ms/步。

**prefill 算子 profile（v2 之后，8 题，2958 token）**
- 设备时间合计约 293 ms/题（profiler 开启）。
- MatMul 占 37.5%，按 FLOP 估算已接近 bf16 峰值的 55%，提升空间有限。
- 其余是大量小算子：Mul（约 1750 次/题）、切片拷贝、ReduceSum、Tril（每 chunk 一次）、Pad、RepeatInterleave，多数属于 GDN chunk 辅助计算。

**按预期收益排序的候选方向**

| 方向 | 预期 | 状态 |
|---|---|---|
| 跨轮次 KV / GDN 状态复用（每题的工具调用后重复 prefill，共 252 次 prefill 中约 124 次是重复） | prefill 134 s 的大部分 | 未做。需先验证前缀 token 完全一致（chat template 边界） |
| GDN v2 减少算子数（预计算 mask 代替每 chunk 的 Tril/Triu，减少切片拷贝与 Pad） | prefill 内部，待测 | 未做 |
| decode 的 persistent 或融合 kernel（MatMul 占约一半的 device 时间） | 不确定 | 未做 |
| 24+ slot（需改变权重或 KV 布局） | 减少总 decode 步数 | 未做 |

**未解决的问题**
- 1030、1137 两题的 elem-conv 翻转根因未深入检查。
- 无 profiler 条件下的 prefill 单题耗时未直接测量（profiler 下墙钟约 540 ms/题，设备时间约 293 ms/题，差距说明 host/launch 开销可能较大）。

---

## 6. 仓库结构

```
.
├── README.md                    本文件
├── HANDOFF.md                   交接与完整实验记录（英文/中文混合，最详细）
├── src/
│   ├── cb_run.py                连续批处理调度与 CLI（当前最优入口）
│   ├── cb_engine.py             CBEngine：slot 制 decode、prefill、graph 捕获、patch_model、elem_conv_update
│   ├── gdn_chunk_fast.py        v1 chunk GDN（prefill，--fast-gdn）
│   ├── gdn_chunk_fast2.py       v2 chunk GDN（批量预计算，--gdn-v2）
│   ├── recurrent_single_step.py 单步 recurrent 规则（decode）
│   ├── sync_free.py             sync-free 版本的 norm / l2norm / recurrent（已否决）
│   ├── shift_conv_update.py     移位累加卷积（已否决）
│   ├── e100_batch.py            HF batch16 runner（"batch16 + 优化"基线）
│   ├── run_agent.py             串行 B=1 基线 runner
│   ├── agent_protocol.py        system prompt、工具定义、答案解析
│   ├── scorer.py                fresh-v1 评分
│   ├── vllm_agent_bench_fair.py vLLM-Ascend 对照 runner
│   └── exact_gdn_candidates.py, model_exact_gdn.py  精确 GDN 候选（约 0.3% 端到端，未采用）
├── scripts/
│   ├── run_abba_full128_elemconv.sh   当前最优 ABBA
│   ├── run_abba_full128_gdnv2.sh      GDN v2 ABBA
│   ├── run_abba_first16_bucket.sh     graph bucket 512 vs 128 ABBA
│   ├── run_full128_graph_pg1.sh 等     各阶段的 full128 运行脚本
│   ├── test_gdn_v2.py, test_cb_engine.py, test_acl_graph.py, test_sync_free.py   正确性测试
│   ├── profile_cb_decode_ops.py       decode 算子 profile
│   ├── profile_prefill_v2_ops.py      prefill 算子 profile（当前最优配置）
│   ├── profile_decode_wall_breakdown.py  decode wall-clock 分解
│   ├── profile_prefill_split.py       prefill 的 GDN / 其他拆分（v1）
│   ├── summarise_kernel_csv.py        汇总 kernel_details.csv 的设备时间
│   ├── diag_flip.py                   翻转题的 logit margin 诊断
│   └── probe_graph_mem.py             显存分解
├── results/
│   ├── baseline_serial/, b16_original_opt/, b16_original_noopt/, vllm_original_b16/
│   ├── cb/                            连续批处理各运行（per_task.jsonl，summary 行含 makespan）
│   ├── prefill_split/, prefill_v2_ops/, profile_*/   profile 输出
│   └── excluded_directions.md         已否决方向的记录
└── logs/                              每次运行的 stdout/stderr
```

---

## 7. 复现

### 环境

| 组件 | 版本 / 位置 |
|---|---|
| 硬件 | Ascend 910B3，CANN 8.5.1 |
| Python（主） | `/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python`：Python 3.12，torch_npu 2.9.1，transformers 5.5.4 |
| Python（vLLM 对照） | `/remote-home/S45160/miniforge3/envs/vllm-ascend-py310/bin/python`：vllm 0.18.0 |
| 模型 | Qwen3.5-27B（本地路径见 `src/cb_run.py` 中的 `MODEL`） |

vLLM 环境需要：`source /usr/local/Ascend/ascend-toolkit/set_env.sh`，`LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver`，`ASCEND_RT_VISIBLE_DEVICES`。不要用 `-I`，否则 EngineCore 子进程无法导入 `acl`。

> 注意：`src/` 与 `scripts/` 中的绝对路径（`/remote-home/S45160/...`）是当前机器上的，换环境需要修改。`cb_run.py` 中的 `sys.path` 也是硬编码的。

### 命令

```bash
# 当前最优配置（npu:0，16 slot）
# PYTORCH_NPU_ALLOC_CONF 由 cb_run.py 在 --graph-decode 时自动设置
ASCEND_RT_VISIBLE_DEVICES=0 /remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python -I src/cb_run.py \
  --out results/ablations/continuous_batching/<name>/per_task.jsonl --device npu:0 \
  --fast-gdn --gdn-v2 --prefill-group 1 --graph-decode --elem-conv

# 可选：graph bucket（默认 512；128 仅快约 0.9%）
CB_GRAPH_BUCKET=128 ... (同上)

# ABBA（完整 128 题）
bash scripts/run_abba_full128_elemconv.sh

# 正确性测试（改动后先跑）
python -I scripts/test_gdn_v2.py
python -I scripts/test_cb_engine.py

# 汇总一个运行的 makespan 与正确数
grep '"kind": "summary"' results/ablations/continuous_batching/<name>/per_task.jsonl
```

运行规则（来自实践）：
- 同一张卡不要同时跑两个任务。`cb_run.py` 会检查并拒绝；确需覆盖时加 `--allow-busy`（计时会被污染）。
- 输出目录已存在时运行会拒绝。重跑前删除输出目录，不要只删单个文件。
- 同一配置要做 ABBA，不要只跑一次。

---

## 8. 延伸阅读

| 想看什么 | 去哪里 |
|---|---|
| 完整实验记录、每一步的证据、所有失败尝试 | [`HANDOFF.md`](HANDOFF.md) |
| 已否决方向 | [`results/excluded_directions.md`](results/excluded_directions.md) |
| 某个运行的原始数据 | `results/<name>/per_task.jsonl`（summary 行含 makespan、decode 步数、prefill 时间、峰值显存） |
| 当前最优配置的 ABBA 原始结果 | `results/ablations/continuous_batching/abba_full128_elemconv_{A1,B1,B2,A2}/per_task.jsonl` |
| prefill 算子 profile | `results/prefill_v2_ops/`（用 `scripts/summarise_kernel_csv.py` 汇总） |
| 与参考项目的对比 | 参考项目 `/remote-home/S45160/work/910b_e2e_accleration_by_codex/`（只读，未修改） |
