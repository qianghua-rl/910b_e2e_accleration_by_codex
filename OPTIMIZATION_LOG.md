# 单卡 Qwen3.5-27B / GSM8K 优化实验记录

## 当前优化路线（更新至 2026-10-23；E130 与归因证据矩阵后）

本节是后续工作的执行入口。下方历史实验按当时认识保留；历史“下一步”、初步归因和估算若与本节冲突，以本节及其引用的最终证据为准。探索允许边做边改，不要求事前预知所有接口、数值和编译问题；规则的作用是降低实验成本并防止把弱证据当成果。

### 目标与冻结参考

目标仍是固定 128 题、单 NPU、原生 Python tool call、non-thinking、greedy、最多 6 轮的未插桩 makespan；scorer v3 至少 `122/128`，同时报告失败 ID、生成 token 与显存峰值。禁止针对答案特化或使用 MTP/speculative。

- 自有最佳 E100：HF Qwen3.5-27B、BF16、SDPA、batch16、generation_length 排序、fused RMSNorm/Gated RMSNorm、single-step recurrent、首轮 cap512/final cap12；`265.030128 s / 122/128 / 15,961 tokens / peak 62,094,008,832 bytes`。证据：`results/phase2_concise_sdpa_batch16_final12_length128.jsonl` 及 v3 派生记录。
- E047 vLLM-Ascend 系统对照：`239.194 s / 122/128 / 16,629 tokens`，差距 25.836 s。不同 token 轨迹和 runtime，不能据此直接比较 kernel 效率。超过它需同任务完整验收，不靠微基准外推。
- cap12 是当前任务生成预算设置，不作为通用底层优化贡献。E100 默认路径不因新候选改变；用 opt-in 开关进行实验，获胜后才接纳。
- 冻结参考不等于禁止重测：候选比较、依赖变化或数据矛盾时，跑必要的同环境 matched baseline；不要只为了“刷新基线”重跑全量。

### 已获得的成本证据与当前结论

| 证据 | 可以支持的判断 | 不能支持的判断 |
|---|---|---|
| E113，E100 首 batch 的四步 decode：计算 union 493.910 ms，stage 710.295 ms；Matmul 237.046、Convolution 116.231、Mul 49.617、Copy 22.529、ReduceSum 14.147 ms | 优先检查重复矩阵乘、短卷积、recurrent 状态读写及其调用边界；E100 记录重建为 1,258 个 batch16 decode 步骤 | 四步早期 trace 不代表所有上下文；kernel 时间和可能重叠，free/idle 不都是可消除成本，不能直接乘成整批收益 |
| E114 FP32 reduction 短卷积：随机 exact，真实局部有差异，前16 token 相同，完整工具轮 task428 改变 | 随机测试和短 token gate 都不足；该算术版本拒绝 | 不能否定所有短卷积实现 |
| E114 原始 conv 窗口版：真实输出/cache、短程 logits exact，完整16题轨迹/质量一致；81.146→82.124 s，局部几乎不快 | 此实现无测得收益，不接纳、不扩展128题 | 单次慢1.2%不能证明普遍回退；也没有支持继续扩展的收益证据 |
| E112 staged GDN：16/16 首 token 变0；C64 chunk-H 编译不兼容 | **该 staged 补丁不可用**，不能用于 KKT 单变量验收 | 全0不能独立定位哪一阶段出错；C64编译失败不等于证明C128运行失败的精确根因 |
| E112 正确 worker 路由后的 KKT-only：16/16 首 token 相同；6.651→7.120 s | 只证明首 token 兼容，当前无提速证据；未接纳 | 未验证 logits/长轨迹；独立参考147 ms不是实际 vLLM基线；B1的4096-token筛选值21.44%不是E100可实现收益或已证上限 |
| E115/E100 early-decode attribution：4个早期 step；MLP Matmul 157.407 ms，GDN convolution family 116.823 ms，projection 71.687 ms，recurrent Mul/ReduceSum/copy 43.200/14.776/9.113 ms | 各项是设备侧 attribution；不能相加为 endpoint latency。卷积族含 Conv2D 55.776 ms 与邻近 TransData/MemSet 61.048 ms | 对应设备时间本身不等于可消除成本；GDN 转换归属和可避免比例尚未确认 |
| E116 recurrent state fusion 与 E114 causal-convolution update | 逻辑中间态或窗口工作可作为机制假设；已测实现分别 wrapper 更慢/不精确，以及局部预算约0.200 ms/48层步、16题 makespan 慢1.21% | 这些具体实现停止；不能据此推断所有 recurrent/conv 机制都不可能 |
| E120–E130 MLP/runtime：buffer reuse无完整 wrapper收益；persistent NZ full generation慢3.718–7.324%；E130 gate/up交叉上限外推4.3–6.2 ms/4步，约0.4–0.7% stage window | 可观察到局部 Matmul 优势和调用位置差异，但其完整路径收益没有转化；位置候选低于2%筛选 | 停止已测 buffer/NZ/简单重排路线；不把局部roofline差额或孤立kernel差当作可回收预算 |

详见 `results/e100_decode_profile_summary.json`、`results/causal_conv_window_matched_comparison.json`、`results/gdn_model_gate_kkt_only_comparison.json`。不重复采整模型长 trace，除非缺失证据会改变候选选择。

### 候选路线决策（三字段）

| 路线 | 已确认的成本 | 可能消除的成本 | 可信的端到端收益上限与决策 |
|---|---|---|---|
| GDN convolution/layout conversion | 116.823 ms/4个早期 decode step；Conv2D 55.776 ms，邻近 TransData/MemSet 61.048 ms。设备 kernel 名称没有 GDN module/layer 身份 | 只有被证明冗余或可复用的转换部分；当前没有这种证明 | **无可信收益上限，暂停。** 现有 trace 无法继续回答归属；不做算子改动或 A/B，等待模块级证据 |
| GDN recurrent state | Mul 43.200、ReduceSum 14.776、copy 9.113、Add 5.984 ms/4步；状态流量估算不是实测 HBM | 减少真实 wrapper 中的状态中间态/遍数 | **未建立端到端上限。** E116 两 kernel 实现 wrapper 慢且非 exact，当前实现停止；新机制需先有真实状态 wrapper 预算 |
| T=1 causal convolution update | E114 保守裁窗约0.200 ms/48层步局部机会，远低于 4.214 ms/步筛选参考；matched16 慢1.21% | 该实现减少的窗口/拼接工作 | **已低于门槛且模型级无收益，停止。** 不扩展128题 |
| MLP persistent NZ / buffer reuse | MLP matmul 约157 ms/4步；NZ局部 wrapper 曾有15–18%收益，但只切换768/1,992次Matmul；buffer reuse未降低完整wrapper成本 | NZ支持的MLP matmul计算；buffer分配事件（已测复用无效） | **无正向 endpoint 上限。** NZ matched full generation慢3.718–7.324%，buffer reuse无收益；两路线停止/封存 |
| MLP gate/up重排 | E130真实交叉局部差16.7–24.3 μs/两投影；粗略四步外推4.3–6.2 ms | 调用位置相关的部分差异 | **约0.4–0.7%四步stage window，低于2%。停止。** 不做 full Decode A/B |
| Runtime thread/graph/async | E118计算union仅变0.27%，空隙变化更大；E119定位部分host prep/scalar barrier | 仅有稳定、可归因且可捕获/移除的具体片段 | **无可信上限。暂停。** 不把空隙总量当可回收预算，不盲目重试graph |
| Attention/GDN projection packing/layout | 71.687 ms/4步；历史 grouped projection 较慢且非 exact | 单一真实shape的layout或launch成本 | **无可信收益上限。暂停。** 只有新机制能通过低成本完整wrapper证伪且预算达到门槛时再评估 |

详细证据、限制及历史引用见 `results/candidate_evidence_matrix.md`。GDN 转换分支当前明确暂停：转换累计时间较大，但模块归属、转换必要性与可消除份额均未证实；不为了维持项目进度而制造实验。若未来出现模块级 layout/ownership 证据，再评估是否能给出可信的 >=2% 端到端上限及低成本证伪办法。整条 PTO GDN pipeline 与 KKT-only 扩展验收也继续暂停。

### 实验如何推进：小实验获得信息，再按证据调整

1. **先检索，不照搬。** 用 `operator_cases/`、`scripts/operator_case.py` 和本日志找相近case，写清可迁移机制、不同环境/形状/精度及旧停止原因。三方Python/HF/vLLM对照只在语义和runtime确实可比时采用；不为凑齐reference引入另一条pipeline。
2. **写一个可修改的假设。** 简记要消除的工作、真实入口、可能收益、最大风险和最小区分实验；不要求编码前所有未知都解决。明确标注测得值、条件外推和未知量。
3. **一次隔离一个主要变化。** 先拆 kernel、layout/转换、wrapper/state、worker路由，再做模型接入。融合本身可以是候选，但先说明它改变哪些边界，不能冒充单阶段替换；不要求所有无关阶段都先重写。
4. **先验证可执行路径。** 实际解释器、torch/torch_npu、Triton driver、CANN与设备、动态库ABI、输出dtype均以运行结果为准。`py_compile`只证明语法，不执行torch导入或证明NPU支持；CPU-build torch配torch_npu不等于NPU不可用。HF E100用ascend-e2e，现有NPU Triton probe用vllm-ascend-py310；不要混装包。`scripts/env.sh`会将PATH指向ascend-e2e，切换环境须检查最终解释器。多进程补丁必须证实EngineCore实际执行了候选。
5. **局部正确性先用真实输入补强。** 检查output、更新后的cache、alias/原位语义、序列边界、自然exp/exp2、FP16/BF16/FP32舍入；随机样本只作快速筛查。遇到差异先查第一处分歧，允许在同一假设内修改实现，不必每次失败都放弃方向。
6. **模型gate逐步延长。** 固定prompt logits/top-1和多步greedy → 固定16题完整工具轮（长输出、已知失败题）→ 必要时32题 → 胜者128题。短程token相同不代表完整轨迹相同，token非空不证明logits finite。数值差异与质量变化分开记录。
7. **测完整调用成本。** kernel计时之外覆盖pack、cast、transpose、metadata、workspace、cache更新、launch和同步；正确性shadow/reference开销不计入性能。只比较相同task IDs、顺序、协议、生成cap、runtime和插桩条件；启动/冷编译/warmup分开。小差异先做必要重复或交替A/B，不能靠重复直到出现最好值。
8. **按证据决定投入。** 明确不可能有效的具体方法可提前排除；合理但未知的先做便宜实验。出现设备异常先定位设备状态，长时间无输出先加阶段标记，不反复杀掉同一运行并重新猜测原因。停止时留下可复现证据和允许重开的新条件。

### 收益预算与接纳口径

E100若仍有1,258个batch16 decode步骤，节省2% makespan需平均每步**净省4.214 ms**；若跨48层均摊约87.8 μs/层。这是决定是否值得继续投入的条件预算，不是每个局部算子必须独自达到的硬性正确性规则；调用数/轨迹变化时须重新计算。2%用作近期筛选目标，小而可靠、成本低的收益也可说明理由后评估，不能自动宣称成果。

- **等价加速候选**：local/cache、模型logits误差与greedy轨迹逐层检查。出现轨迹变化不再称等价加速；可先定位舍入/实现错误。若要改成允许数值/轨迹变化的候选，明确重设实验声明，再走独立质量验收，不能在失败后悄悄放宽门槛。
- **端到端接纳**：完整128题audit、scorer v3至少122/128，检查新增失败与原因；未插桩makespan可靠优于E100，报告token数和peak。只有16题通过时只声称16题结果；局部失败或无收益可提前淘汰，不必为了流程跑128题。
- **证据强度**：缺失符号、已确认ABI错误、计算上限不足可以提前排除对应方法；单次轻微慢、四步profile、近似公式和首token一致都只支持有限判断。精确根因未定位时写“观察/怀疑/待证”，不把症状当根因。

### 当前下一步与记录模板

当前没有候选同时具备可信的 >=2% 端到端收益上限和低成本证伪方法，因此暂不启动新优化实验。下一步是整理/接收能改变路线决策的新证据，优先级为：GDN 转换的模块级 ownership 与输入输出 layout；或其他路线中可量化且可隔离的真实 wrapper 机制。只有形成候选预算并有低成本证伪路径后，才进入单变量实验；否则保持 E100 默认实现和当前最佳结果不变。

每次实验记录：**假设与case依据 → 配置/代码与证据路径 → 最小实验 → 观察及第一处分歧 → 修改假设/实现 → 正确性覆盖范围 → matched性能与预算 → 接纳/拒绝/未完成及重开条件**。记录最终版本和历史artifact对应关系；文件名变更/结果覆盖必须避免让旧结果误指向新代码。默认实现始终保留已验收路径。

### 历史路线（仅追溯，不执行）

1. **实验目标漂移**：日志从 batch=1 解码参数、HF 固定批次，转向 vLLM serving 和全量请求池，但阶段标题及 E037 汇总没有同步更新。之后每个实验先标明 `single-request`、`offline-128` 或 `online-arrival`，只在相同场景内比较。
2. **诊断数据不够聚焦**：E047 已有 128 题 makespan、逐题输出和 summary，不需要为“刷新基线”重跑；离线汇总已显示工具时间很小、生成 token 集中于首轮，但 runner 未记录每轮 `llm.generate()` 耗时，仍难区分 prefill、decode 和轮间同步成本。
3. **候选质量门槛需要结合失败归因**：历史原始基线为 `118/128`，但当前 E100 已建立 `122/128` 参考；新候选必须达到当前护栏 `>=122/128`，同时逐项检查失败 ID 和失败类型。协议变更先用 hard-task 子集 smoke，最终仍由完整 128 题 scorer v3/audit 定案。不得针对数据集答案特化提示。
4. **Amdahl 证据不足**：E007 profiler 空隙占比不是可优化 host 工作占比；局部算子收益只作筛选信号。只有当诊断能改变候选选择时才采 profile，最终以端到端结果判断。
5. **日志可操作性下降**：历史章节有过期“下一步”和重复进度。后续更新本路线节的当前状态；实验记录配置、端到端收益、正确性、失败归因和证据路径，历史数据保留。

### 旧实验建议（已废弃，仅供历史追溯；不得覆盖上面的当前执行版）

1. **保留已验证起点，不刷新基线。** 以 E047 的 vLLM-Ascend 128-request 配置（239.194 s、v2 122/128）作为当前最优已知方案。E020 的 HF 结果保留作历史对照。只有环境变更、配置不明、设备异常或关键结果互相矛盾时，才复测对应结果；比较新候选时只跑必要的同条件对照。
2. **下一步补齐生成阶段时间，不再做离线汇总。** 修改 vLLM runner，逐轮记录 `llm.generate()` wall time、活跃请求数、prompt/generated token 数，并汇总工具执行时间。先用 16–32 题样本验证计时和审计字段，样本须包含长输出/失败题；该诊断运行不作为性能结论，也不与其它 NPU 任务并发。E047 已提供工具时间和 token 总量，不重复采集已有字段。
3. **据阶段时间选择一个优化假设。** 若首轮 prefill 占主导，评估 chunked prefill 或 prompt 路径；若 decode 占主导，评估 vLLM scheduler/KV 参数；若轮间存在明显空档，再考虑 agent 外层调度。当前数据只支持“生成比工具执行更值得优先看”，不支持提前断言上述哪条路径必胜。
4. **用小样本快速淘汰候选。** 对选中的单一改动，以固定的 16–32 题 hard-task 样本和 E047 同配置前缀做对照。记录 makespan、输出 token、v2 正确性/失败变化及显存；若没有明确端到端改善或引入质量/稳定性问题，立即停止该候选，不做全量。
5. **仅对 smoke 胜者跑 128 题。** 使用现有 v2 scoring 和 audit 验收，硬门槛 `>=118/128` 并检查新增失败 ID。报告相对 E047 的 makespan、token 数、精度及失败归因；冷启动单列。通过后更新当前最佳，未通过则保留 E047。
6. **kernel/cache 是证据驱动的后续方向。** 只有阶段计时指向 scheduler、hybrid cache 或算子瓶颈，且当前软件栈存在可执行路径，才打开相应工作。E010/E016/E017/E030/E052 在当前栈上的失败/淘汰结果不重复探测；依赖变化后先做最小 capability probe。

项目成功标准是相同 128 题、既定正确性护栏通过时 makespan 低于 E047。测量只为选择和验收优化服务，不把刷新基线、增加诊断数据或接近理论极限单独算作进展。

## 约束与验收

遵循 readme.md：固定 128 题、原生 python tool call、non-thinking、greedy、每轮 512 token、至多 6 轮；离线场景主指标为整批 makespan，accuracy 不低于实测基线。readme.md 的 Phase 3 允许面向 910B3 的通用原生算子和内存优化，不规定必须使用 FP8；禁止 MTP/speculative。FP8 per-tensor 曾是早期实验约束，其能力探针失败（E016），不再视作项目固定要求。每个候选记录单变量变化、正确性、实测及偏差；Amdahl 估算仅在 profile 证据足够时使用。未测项不能作为优化成果。

## 2026-10-14 环境建立 / E000

- 初始工作树仅 README、模型和数据，没有已有 benchmark、实验记录或正在运行的推理进程。
- CPU：aarch64，192 核，4 socket，8 NUMA；设备可见 2 张 Ascend910B3，各标称 65536 MiB HBM。实验限制单设备，不使用另一张卡进行模型切分。
- PyTorch 实测属性：20 cube cores、40 vector cores、192 MB L2、62420 MB 可用总显存。
- 已存在环境 `ascend-e2e`：torch 2.9.1+cpu、torch_npu 2.9.1、transformers 5.5.4；系统 toolkit 报告 8.0.RC3。版本组合兼容性必须逐项实测。
- 默认 python 无推理依赖；首次 npu-smi 因 libc_sec.so 缺失失败，source toolkit 后仍缺 driver 库。补齐 driver/lib64/{driver,common} 路径后 npu-smi 和 NPU tensor 加法成功。
- 可重现环境：`source scripts/env.sh`。固定一个逻辑设备；不启用 CPU offload 代替单卡推理性能。
- 本机是 NPU，README 的 CUDA nsys 命令不适用；L3 将检查 msprof / Ascend 设备 trace 能力，明确记录与原要求的差异。

## E001 硬件与算子形状探索（第一轮完成）

- 命令：`source scripts/env.sh; python scripts/hardware_probe.py > results/hardware.stdout.log 2>&1`。
- 控制变量：单卡、BF16、8 host threads、3 次预热；每项 5 组同步墙钟计时，报告中位数及全部样本。
- 扫描：HBM copy 1–512 MiB；实际 MLP up-projection K=5120/N=17408，M=1–512；pinned H2D 1–256 MiB。copy 按读+写总字节计带宽。小尺寸结果包含 dispatch 开销，不解释为物理带宽。
- 原始数据：`results/hardware.jsonl`，标准输出及失败信息均保留。
- 尚未建立模型基线；没有 accuracy 或加速结论。后续还需 KV/循环状态内存、推理参数、L1/L2/L3、三个优化阶段及全量验证。

### E001 结果与解释

- 进程正常退出（exit 0），20 条 JSON 记录（含设备信息）。256/512 MiB copy 的读写总带宽分别为 1.247/1.257 TB/s；64 MiB 达 4.353 TB/s，工作集可留在 192 MB L2，因此不能当作 HBM 带宽。
- MLP 形状 M=1/32/128/512：中位延迟 92.97/91.79/113.20/382.01 μs，吞吐 1.92/62.14/201.56/238.91 TFLOP/s。M=1–32 几乎恒定延迟，提示批处理可摊薄权重读取和 dispatch 成本。权重约 170 MiB，可能命中 L2；这不是全模型 decode 性能预测，需补充超过 L2 的轮换权重实验。
- pinned H2D 256 MiB 为 25.63 GB/s，远低于设备内存 copy；逐 token 搬权重不可能接近驻留推理的物理极限。
- 当前 peak 字段为进程累计峰值，包含此前 copy 分配，不能拿它比较 batch 内存；下一轮应对每个条件单独 reset peak，并报告增量。
- 下一步：补充缓存控制、device event 与 host 同步计时对照、线程数/stream 控制实验，然后进行模型兼容性及 KV/循环状态、序列长度/batch 参数探索。完成第 2/3 节前不宣称进入优化阶段。

## E002 缓存、线程与 stream 控制（完成）

- 命令：`source scripts/env.sh; python scripts/hardware_control.py > results/hardware_control.stdout.log 2>&1`；exit 0，48 个配置保存于 `results/hardware_control.jsonl`。
- 全因子：host threads=1/8/32，M=1/8/32/128，同一权重或轮换 4 份独立权重（合计约 680 MiB，大于 L2），stream=1/2。每项 8 次预热、5 组×40 次同步计时；另用设备 event 测量单 stream 40 次。固定 BF16、K/N、随机种子。
- 8 threads / 单 stream：M=1/8/32/128 的缓存重复延迟为 108.36/108.11/111.57/139.74 μs；轮换权重为 152.37/153.58/156.35/177.67 μs。单 stream event 对应轮换值 149.94/150.67/153.29/174.96 μs；host 计时多约 2–3 μs。脚本 stream context 调度额外开销意味着不能直接与 E001 做优化收益比较。
- M=1 轮换权重：1/8/32 threads 单 stream 为 153.08/152.37/152.56 μs；双 stream 为 155.81/155.31/154.94 μs。这个受权重带宽限制的形状增加线程/stream 未带来收益；后续优先探索 batch，而非假定多 stream 可加速。
- reset peak 已改为每配置独立执行。该实验不测通信集体操作，因为任务只允许一张卡；设备与 host 的通信由 E001 H2D 覆盖，后续增加 D2H/状态影响。

## E003 模型算子兼容性（完成）

- `scripts/model_compatibility.py` 使用实际 hidden/head/MLP 维度、3 层 Gated DeltaNet + 1 层 full attention，随机 BF16 权重和缩小词表；只做执行诊断，不能替代完整模型的性能或正确性基线。
- 命令：`source scripts/env.sh; python scripts/model_compatibility.py > results/model_compatibility.stdout.log 2>&1`。prefill 和有 cache decode 检查 finite；batch/length=(1,16)/(1,128)/(4,128)。
- 标准实现检测到 CUDA fast-path 库未安装，回退 PyTorch；需实测 NPU 回退及后续完整模型加载。
- exit 0；三个条件的 prefill/decode 全部 finite。首次 prefill 0.719 s 含初始化；其余 0.069/0.077 s。随机小模型不测 accuracy，也不作为性能优化基线。

## E004 完整模型参数扫描（完成，含 OOM 边界）

- 命令：`source scripts/env.sh; python scripts/model_parameters.py > results/model_parameters.stdout.log 2>&1`。
- 使用原始完整 checkpoint、BF16、eager attention、单 NPU 驻留；扫描 batch=1/2/4/8、输入长度=128/512/2048，记录 prefill、8 次 cache decode、显存。重复真实题目 prompt 构造长度只用于参数实验，不用于 accuracy。
- 每个形状 reset peak；首轮含冷启动，后续分析须明确其影响。尚未进行 Phase 1 优化。
- 首次尝试在加载前因缺少 accelerate 失败，无参数测量结果。失败日志保留为 `results/model_parameters.missing_accelerate.log`。安装 accelerate 1.15.0 成功（`results/install_accelerate.log`），随后原命令重跑；未改变模型、精度或扫描条件。
- 第二次成功加载完整权重：23.881 s，allocated=54,713,982,976 bytes；模板默认返回 BatchEncoding，构造长度时报 TypeError。日志 `results/model_parameters.batchencoding.log` 和 JSON 错误行保留。显式 `return_dict=False` 修正接口后第三次扫描已产生 finite 的完整模型 prefill/decode 数据。

## E005 端到端基线框架（准备，未开始计分）

- `scripts/agent_protocol.py` 固定 system prompt、原生 XML python tool 格式、10 s Python 超时、boxed 数字评分；成功必须至少执行一次工具。所有优化共享同一协议，不根据题目定制提示或程序。
- `scripts/agent_bench.py` 是 sequential/batch=1 reference runner：非 thinking、greedy、每轮 512 token、最多 6 轮；保存每轮 token ID、文本、工具代码/输出、分段计时、每题完成时间、整批 makespan 及数据 SHA256。模型加载时间单独记录；所有题目从批次开始即视为可用。
- 脚本已通过语法检查；需要真实模型 smoke、128 题准确率及 profiler 才能认定建立基线。准备代码不是 Phase 1 性能改动。
- E004 实际结果：batch=1/2/4 的长度 128/512/2048 均 finite；batch=8,length=2048 在约 56.94 GiB active allocation 时 OOM。batch=4,length=2048 峰值约 60.29 GB，逼近单卡容量。batch=1 length=128 的首条重复记录是接口修复前的错误行，后续成功记录优先；完整 JSON 保留全部历史。

### E005 实题 smoke（完成）及全量基线（运行中）

- `results/agent_smoke.jsonl`：题 id=23，2 轮、278 个生成 token，原生 python 调用打印 8，看到输出后生成 boxed 8；correct=true。makespan=47.267 s（加载 16.691 s 单独排除），generation=47.169 s，prepare=0.054 s，python=0.040 s。单题不是 128 题 accuracy 证据。
- 分析工具：`python scripts/summarize_bench.py results/agent_smoke.jsonl`。会明确区分 partial 和 complete run，报告阶段总时间和 completion/service 分布。
- 全量命令：`source scripts/env.sh; python scripts/agent_bench.py --output results/baseline_bf16_eager_128.jsonl > results/baseline_bf16_eager_128.stdout.log 2>&1`。单卡串行 reference；全部 128 题不抽样，greedy token ID 作为后续比较参考。耗时未完成前不外推 accuracy。
- 下一步：基线运行期间准备 L2/L3 profiling；不启动争用同卡的第二个性能实验。硬件/形状探索已完成首轮，仍需真实推理参数、状态内存及 longctx 扩展。
- 已准备 `scripts/profile_inference.py`：独立完整模型诊断，预热后采集 1 次 prefill、8 次 decode，CPU+NPU trace、shape 和 memory；使用 `torch.profiler.record_function` 标注阶段。仅语法检查通过，尚未实际采集，不与基线争用 NPU。
- `msprof --help` 成功，帮助输出保存在 `results/msprof_help.txt`，具备 Ascend L3 CLI 入口；实际 trace 能否采集仍待验证。查询 profiler 签名成功，尝试把 pybind ProfilerActivity 当 iterable 的诊断命令 TypeError，不影响推理实验；采集脚本显式指定 CPU/NPU 枚举。

## E006 离线状态内存与检查点审计（完成）

- 命令 `python scripts/model_memory.py`，只读取 safetensors header，不占用 NPU；输出 `results/model_memory.json`。文本参数 26,895,998,464 个、53,792,013,824 存储字节；MTP 849,398,784 字节，vision 921,460,192 字节。MTP 权重存在于原始检查点不等于启用 MTP；当前 benchmark 是标准 generate，无 speculative。
- 初版从 config/reference 计算时把 recurrent state 按 FP32 估成每请求 150,994,944 bytes（144 MiB）；该 dtype 假设后来被完整模型实测证伪，见 E050。按真实 DynamicCache 的 BF16 recurrent state，每请求 75,497,472 bytes（72 MiB）；BF16 conv state 3,932,160 bytes（3.75 MiB），16 层 full attention BF16 KV 每请求每 token 65,536 bytes（64 KiB）。128 个请求的 recurrent state 仍约 9 GiB，不含 attention KV、权重和 workspace。
- JSON 包含 batch=1/2/4/8/16/32/128 与 length=128/512/2048/8192/32768 的 cache 估算；不包含权重、激活、allocator 和 workspace，因此仅是容量下界，不能替代 E004 的实测峰值。
- 新增 `scripts/compare_runs.py`，严格比较相同数据/协议、全部 task ID、逐轮 token ID、accuracy 和 makespan；partial run 不会被判为通过。对已完成 smoke 自比较通过只是检查工具可运行，不是优化正确性证据。
- 新增 `scripts/analyze_trace.py`，按显式选定 device pid 合并算子区间，避免重叠时长重复计入 busy；报告 idle gaps 和 operator sums。依赖 critical path 不能从时间邻近推断，仍需 trace 的 correlation/flow 证据。

## Phase 2 候选准备（未执行，不计优化结果）

- 等待完整基线期间准备 `scripts/agent_batch_bench.py`：相同模型/协议/greedy 上限，left padding 的同步批量 generate，结束的请求移出下一轮，逐请求保存 token 和工具输出。未运行 NPU，未跨越 Phase 1 测量顺序。
- 这是基础 batching 候选，尚无 continuous admission 或工具执行与推理重叠；后续根据 profile 决定是否值得增加调度复杂性。prepare/generation 时间按有效 batch 均摊，并单独记录 batch_generation_seconds，避免把批次墙钟重复计入总时间。
- 必须先通过 batch=1 对 reference 的 token 一致性，再扫描 batch=2/4/8，并实测 128 题 accuracy、makespan、显存及长上下文。语法检查通过不代表这些门槛已通过。
- CPU tokenizer 实测：批量候选的 batch=1 路径与 reference 的第一题 prompt 完全相同（353 token）。这是输入一致性检查，尚无生成一致性证据。

## 基线期间的物理下界与后续调度

- 以 E001 大工作集约 1.25 TB/s 及文本 BF16 权重 53.792 GB 估计，假设每 token 流式读一次全部文本权重，理想带宽时间约 43 ms。它忽略 KV/state、dispatch、低效小算子、同步等，也未证明当前每步实际读流量，因此只作 roofline 参照，不能称已测物理极限。
- E004 batch=1 decode 约 0.17–0.20 s，高于该简化下界；下一步 profile 必须区分线性层、recurrent fallback、attention、host launch 和空隙，才能给出有证据的 p/s 及 Amdahl 排序。不得以一个大算子的微基准收益直接当全模型收益。
- `scripts/after_baseline.py` 已启动，等待当前基线 PID 719332（记录 /proc starttime 防止 PID 复用）。仅进程终止且 JSON 具有完整 128 个唯一 task 和 summary 时，运行独立 `profile_inference.py`；基线失败/不完整则退出，不重启、不把 partial 当成功。输出 `results/after_baseline.stdout.log`。此流程只排队 L2 采集，不会同时占用 NPU。
- 新增 `scripts/audit_run.py`：根据原数据重算 boxed 评分、检查唯一 task ID、轮次/512-token 上限及工具记录一致性。已完成的前 7 题及 smoke 无审计错误；partial 模式结果明确不代表全量完成。独立 smoke 与全量首题逐 token 相同。

## Phase 3 能力审查准备（未执行优化）

- 已安装 torch_npu 的 npu_quant_matmul 文档在 A2/A3 产品条目只列 int8/int4 输入，不能据 torch.float8 dtype 存在推断 910B3 原生 FP8 GEMM 可用。后续需要独立硬件实测，禁止用 INT8 偷换 README 的 FP8 per-tensor 约束。
- `scripts/fp8_reference.py` 明确 E4M3FN、每 tensor 单一 absmax scale、FP32 accumulation，供未来通用 kernel 数值对照；只是 CPU 数值参考，不是 Phase 3 加速方案，不改当前 BF16 基线，也不声称 quantization 后 greedy 会与 BF16 一致。FP8 的优化候选须与相同 FP8 reference 比较，并另外验证整体 accuracy 护栏。
- FP8 reference CPU smoke exit 0，零张量量化和有限值/shape 检查通过；日志 `results/fp8_reference.stdout.log`。尚未进行 NPU FP8 实测。

## 可追溯性快照

- `results/environment.freeze.txt` 保存当前环境依赖；`results/provenance_initial.json` 保存脚本、env、README、数据、config/generation_config/chat_template 的 SHA256 和时间。快照在基线启动后生成，未在计时期间顺序读取 54 GB checkpoint 做全量 hash，避免额外 I/O 干扰；这不是完整权重内容校验。
- `summarize_bench.py` 对 partial accuracy 显式标为 `completed_subset_only`，防止把前几题正确率误报为完整数据集质量。基线进程仍存活，继续等待；未重启或更改运行中的配置。

## Longctx 验证准备（尚未运行）

- `scripts/longctx_probe.py` 准备独立合成 retrieval 护栏：目标长度 2048/4096/8192，needle 放在 10%/50%/90%，greedy 生成最多 32 token；记录实际 prompt 长度、完整生成 token、准确性、显存和耗时。固定六位数字，无 GSM8K 标签参与。此任务检查长上下文检索能力，不替代固定 128 题数学准确率，也不是完整长上下文基准集。
- 长度超过 eager attention 单卡容量时需保存 OOM，不能缩短后宣称原长度通过。当前只完成语法检查；待基线/profile 后先建立 reference，再比较候选。

## E005 中途失败归因（全量仍运行）

- 完成 24 题时首次失败 id=304：第一轮生成 512 token，长篇分析被截断，尚未形成完整 python tool call，prediction=null、tool_count=0，按协议计错。不是评分解析丢失 boxed，也没有 Python 执行异常。
- 不针对失败题更改 prompt、不放宽 512 token/6 轮限制、不把不完整工具调用当执行成功。完整基线 accuracy 必须保留该失败；当前 23/24 只代表已完成子集，不是最终质量护栏结论。
- 完成第 30 题时出现第二个错误：生成 boxed -8，数据答案为 8；工具已执行一次，因此是模型答案符号错误而非工具未调用。该题总 service 约 159.27 s、两轮合计 920 token，解释了这段观察窗口无新完成记录的原因；进程没有卡死。保留原始结果，不改变评分规则。
- id=428 同样首轮满 512 token，截断在 Python 程序中间，无完整工具调用、无 boxed；按失败保留。完成 31 题时 28 正确，当前子集低于 README 提到的 0.93–0.95，不能提前声称已满足基线质量，必须等 128 题再判断。
- 完成 52 题时新增 id=552：首轮 512 token 截断、tool_count=0、prediction=null，数据答案 450；此时已完成子集 48/52。截断失败累计 3 题，另有 1 题符号错误，均保留在原始基线中。
- 完成 66 题时新增 id=640：首轮 512 token、无完整工具调用、prediction=null，数据答案 43500。累计 4 题截断失败及 1 题符号错误，子集 61/66；不改变基线条件。
- id=952：已执行 python、输出 boxed 1800，数据答案 360；模型将时间解释为 8 AM 到 11 PM 共 15 小时，属于题意解释错误。完成 91 题时累计 6 题错误（4 截断、2 答案错误），当前 85/91；评分及 prompt 不变。
- 评分器缺陷：id=1001 实际正确生成 `boxed{2:00 text{ PM}}`（原文含 LaTeX 反斜杠），v1 正则不支持嵌套花括号而误报 prediction=null。已实现 `scoring_v2.py` 的 balanced box 解析及整点 `:00 AM/PM` 数值归一化；不把非整点或负数符号错误归一成正确答案。正/反例检查通过。
- `rescore_run.py` 写独立派生文件，保留原始 prediction/correct，不改 token/text/timing 或运行中的基线；当前 partial 派生文件 `results/baseline_partial_scoring_v2_94.jsonl` 只改变 id=1001 为正确。完整结束后需重新生成全量 v2 派生文件，所有候选使用同一 v2 评分。v1 审计仍针对原始文件，不能直接用于 v2 派生文件。
- 审计工具现按 metadata.scorer 自动选择 v1/v2；94 题 v2 partial 及 95 题 v1 原始记录审计均通过。比较工具增加 scorer 版本一致性检查，避免跨评分版本误报 accuracy guardrail 通过。
- id=1030：首轮 512 token 截断、无完整工具调用、prediction=null，数据答案 24。完成 98 题时 v1 原始记录 90 正确（其中 id=1001 为已确认评分器误判，v2 应加回 1）；未改运行中的配置。
- id=1035：正常执行 python 后 boxed 40，标准答案 35，模型将瓶子位置按含两个端点计数，属于题意/端点解释错误，保留为失败。完成 99 题时原始 v1 90 正确，v2 应为 91（已确认 1 个解析误判修复）。
- id=1048：模型 boxed 9860.78，标准答案 7400；模型采用逐年复合折旧（20000×0.79³），标签采用固定原价的线性折旧（20000×(1−0.21×3)）。这是题意解释与固定标签不一致，不能通过评分归一化改判；保留为错。当前 102 题 v2 93 正确，即使余下全对最高也只有 119/128=0.9296875，无法证明达到 README 的 0.93–0.95。完整运行仍需保留作为现状基线；后续必须单独解决质量偏差，不能降低护栏。
- id=1161：正常工具执行后 boxed 140，标准答案 170；模型将第二次售出的 1/4 按原始土地面积计算。保留为建模/题意错误。完成 114 题时 v1 103 正确（v2 加回已确认的 id=1001）。

## Phase 1 launch 参数实验准备（未执行）

- `scripts/phase1_decode_probe.py` 预备扫描 threads=8/1/4/16/32/8（首尾重复用于观察漂移），相同完整模型、fresh cache、固定 teacher-forced token、每条件 4 次 warmup + 20 次同步 decode。保存 logits bit equality、argmax equality、max error 及全部时间样本；这不能替代真实 greedy e2e 验证。
- `agent_bench.py` 增加可选 `--threads`（默认仍为 8），仅供后续单变量测试；正在运行的基线已载入原脚本，不受此编辑影响。原脚本 hash 已保存在 provenance_initial，后续候选需记录新 hash。
- 只有 profile 显示 host launch 占比后才据此估算 Amdahl p；没有证据前不宣称线程参数收益。脚本目前只做语法检查，未占用 NPU。

## E005 全量基线完成

- 128 个唯一 task 和 summary 完整；原始 v1 及重评分 v2 均通过 `audit_run.py` 的全量审计。v2 文件：`results/baseline_bf16_eager_128.scoring_v2.jsonl`，只改变 id=1001 的评分。
- makespan=7836.171112 s（130.603 min），生成 44,082 token；generation=7829.832395 s，prepare=1.171820 s，python=4.873845 s。service median=55.997770 s；所有题目从 t=0 可用，completion p50=3956.189458 s，p95=7419.505894 s。
- v1 117/128；v2 **118/128=0.921875**。低于 README 0.93–0.95，不能将其当已达质量要求的最终方案。10 个真实失败：5 个 512-token 截断、5 个答案/题意错误；另 1 个 v1 解析误判已修复。
- profile 队列已确认原基线进程结束，启动独立完整模型 `profile_inference.py`（PID 748130），未与基线重叠。后续先获取可用 L2/L3 证据及关键路径占比，再选择 Phase 1 候选。
- 质量后续：单独探索通用简洁工具调用提示，保持模型/数据/non-thinking/greedy/512 token/6 轮不变，不对题目或标签特化。协议改变后必须重新建立全量质量/性能基线；不能拿不同 prompt 的 token 差异宣称 kernel 正确性或加速。

## E007 完整模型 L2 / Ascend 设备 trace（完成）

- profile exit 0；目录 `results/profile_baseline/1c91821700b6_748130_20261014153411849_ascend_pt`，同时导出 CPU、CANN、Ascend Hardware 事件，共 1,338,868 个事件。258 条 memory 记录不完整，显存分析仍以 E004 为准。
- 设备 pid=766085536（由 process_name=Ascend Hardware 元数据确认）。`analyze_trace.py` 首次遇到 CANN 的字符串时间戳报 TypeError，已统一转 float 后成功；原始 trace 未改变。
- 1 prefill + 8 decode 的设备窗口 3.786938 s，设备算子区间并集 1.141553 s（30.14%），idle 2.645384 s。该占比来自带 profiler 开销的诊断，不直接当未插桩 e2e busy。Matmul 累计 0.486134 s/4473 次，Mul 0.113563 s/13938 次，conv TransData 0.100764 s/432 次，Slice 0.070545 s/6642 次；结果 `results/profile_baseline_analysis.json`。
- Amdahl 初步候选：仅以此 trace 为近似，host/launch 空隙 p≈0.699，若局部缩短 2 倍，整体预测 1.537×；线程扫描工作量低，先执行。注意空隙不全是可消除 host 工作，且 profiler 会放大 launch 成本，实际偏差需测。Matmul p≈0.128，局部 2× 仅预测 1.069×，暂后排。
- NPU 已在 profiler 正常退出后启动 `phase1_decode_probe.py`，逐条件检查 bit equality 与时间；不写 kernel，未启用 speculative/MTP。依赖 critical path 关联与独立 L3 msprof 仍待补齐，当前只完成时间区间归因。

## E008 Phase 1 threads 扫描（完成，未选为收益）

- exit 0，6 个条件全部 logits bit equal、argmax equal、max error=0。threads=8/1/4/16/32/8 的 decode 中位数分别 182.784/180.779/183.042/183.307/184.664/183.787 ms。首尾 8 线程有约 0.55% 漂移，1 线程差异仅约 1.1%–1.7%，尚无足够证据宣称稳定加速；保持默认 8。
- 与 E007 假设局部空隙缩短 2× 的 1.537× 乐观预测差距很大：线程数没有消除 Python/算子 dispatch，也不影响 NPU 算子数，profiler 空隙不能等同可由线程调节的工作。此候选不值得直接花一轮全量 e2e 验证，优先考察减少算子 launch 的库/图执行路径。

## E009 通用 concise 协议探索（8 题 smoke 运行中）

- 新增独立 `CONCISE_SYSTEM`，要求先仅输出简短 python tool call，看到输出后仅给 boxed 数字；不提任何题目、答案或特定建模规则。原始协议默认不变，baseline 文件不变。
- 命令：`source scripts/env.sh; python scripts/agent_bench.py --protocol concise --limit 8 --output results/concise_protocol_smoke8.jsonl > results/concise_protocol_smoke8.stdout.log 2>&1`。这是质量/交互协议探索，8 题无法证明质量达标；若通过需重新全量建立 reference。
- runner metadata 记录 protocol/system_prompt/tools，比较器禁止跨 original/concise 声称 token 一致或公平性能收益。当前仍 BF16/eager/8 threads，其余上限全部固定。

## E010 图编译兼容性探针准备（未运行）

- `scripts/compile_probe.py` 使用实际单步 recurrent-rule 形状 B=1/H=48/K=V=128，固定随机输入对比 PyTorch reference 与现有 torchair reduce-overhead 编译；记录输出和 state 的 bit equality、最大误差、首次编译和稳态时间。只是通用算子组合编译，不写自定义 kernel。
- Amdahl 依据 E007 小算子次数与空隙：只有实测编译可减少 launch 并保持正确后，才从 trace 关联其 p；此前 0.699 是所有空隙的上界，不能直接作为这个函数的 p。先做兼容性/局部收益检查，失败则保留错误并更换库或 launch 路径。
- 当前只通过语法检查，等待 E009 smoke 结束后独占 NPU 执行。

## E010 图编译探针结果（失败）

- `compile_probe.py` reference 正常：单步 recurrent rule 稳态约 0.93 ms。torchair reduce-overhead 编译首次调用失败，底层 `aclmdlRICaptureBegin` 不可用（ERR00008 PTA resource not found）；未产生编译输出、不能宣称 launch 融合或加速。reference 与失败日志保留在 `results/compile_probe.json` / `.stdout.log`。
- 结论：当前环境的 graph capture 路径不可用，Phase 1 不采用 torchair；不得为了通过而切换不等价后端或忽略错误。后续可测试 `npugraph_ex`，但需独立 probe 和数值/性能门槛。

## Phase 1 candidate: concise protocol full run（运行中）

- smoke 8 题完整且 v1 评分 8/8，makespan=182.359 s，service median=22.156 s，955 generated tokens；原协议同规模参考约 47 s/题。smoke 只证明协议可执行，不能证明 128 题质量。
- 已启动全量命令：`source scripts/env.sh; python scripts/agent_bench.py --protocol concise --output results/phase1_concise_128.jsonl > results/phase1_concise_128.stdout.log 2>&1`。模型、数据、BF16/eager、8 threads、greedy、512 token、6 轮均固定；唯一变量是通用 system prompt。运行期间不启动其他 NPU 任务。
- 成功门槛：全量 v2 accuracy 不低于原始 baseline 118/128，greedy token equality 不适用（protocol intentionally changed），makespan 与 per-task 分布单独比较；并需长上下文 probe。若质量下降，候选淘汰，不能只接受速度。
- 当前进度（2026-10-15）：27/128 完成，27 正确；service median=19.848 s，generation 累计 610.819 s。仍是 partial evidence，未宣称候选通过。
- 继续进度：56/128 完成，54 正确（partial 0.9643），service median=20.742 s，generation=1361.366 s。出现 2 个失败，尚未归因；不能用 partial accuracy 作为最终护栏。
- 系统 overlay 临时层一度仅 260 KB 可用，导致诊断脚本无法创建 here-doc。使用 `find -delete` 清理可重建的 `/tmp` 工具缓存后释放约 1.2 GB；部分 root-owned 文件无法删除，仓库 `results/` 未删除，简洁基线进程 PID 776560 未中断。该磁盘事件及清理输出保留在会话记录中。
- 简洁协议当前 2 个失败已归因：id=423 首轮已调用工具但第二轮生成满 512 token、未 boxed（截断）；id=552 调用工具后 boxed 900，标签 450（模型计算错误）。两类错误仍会触发质量护栏，协议没有因失败题特化。
- 当前进度继续到 88/128，86 正确（partial 0.9773），service median=19.557 s，generation=2053.642 s；仍未达到全量验收条件。
- 当前进度继续到 107/128，101 正确（partial 0.9439），service median=20.307 s，generation=2546.527 s；仍需全量 v2 评分和 makespan。

## E011 Phase 1 concise 全量结果（候选通过初步门槛）

- 128/128 完整，原始 v1=121/128；v2=122/128=0.953125。全部审计通过。makespan=3055.056022 s（50.918 min），相对 BF16/original baseline 7836.171112 s（130.603 min）为 **2.565x**，减少 4781.115090 s（61.0%）。generation=3048.760902 s，service median=20.077 s，completion p50=1583.931 s、p95=2906.513 s，生成 16,264 token。
- v2 accuracy 122/128 高于 baseline 118/128，满足 accuracy guardrail；失败仍有 6 题，需列出归因并完成 longctx guardrail，不能把 0.953 解释成无错误。
- `compare_runs.py` 报 protocol_equal=false、逐题 token mismatch 是预期结果：候选唯一改变了通用 system prompt，不能要求与原协议 token bit equality。它仍确认 128 task 完整、v2 accuracy guardrail=true、speedup=2.565；该报告保留作为“禁止跨协议 bit equality”的证据。
- 这不是 kernel/FP8/图编译优化，而是 Phase 1 允许的 launch/prompt 工作量优化；没有 MTP/speculative、没有 workload 特化、没有 kernel 修改。需完成 longctx 后才可认定 Phase 1 candidate。
- 失败归因：id=423 二轮 512-token 截断；552 算错 900/450；952 算错 1800/360；1001 v1 解析误判但 v2 正确；1035 端点建模 40/35；1048 复合/线性折旧解释 9860.78/7400；1161 分数土地售卖建模 140/170。v2 实际失败 6 题（423、552、952、1035、1048、1161）。

## E012 concise long-context guardrail（失败/OOM）

- 命令：`source scripts/env.sh; python scripts/longctx_probe.py --output results/longctx_concise.jsonl > results/longctx_concise.stdout.log 2>&1`。
- probe 在首个超过 2k 的 eager full-attention 条件尝试分配约 5.92 GiB 时 OOM（约 54.68 GiB active、5.24 GiB free）；未产生可用 JSON 记录，完整 stderr 保留在 `results/longctx_concise.stdout.log`。
- 这不是 longctx 通过，也不是 concise 质量失败；它暴露了当前 eager attention 单卡容量边界。按验收要求，不能缩短长度或忽略 OOM 后宣称 longctx accuracy。Phase 1 的 128 题 accuracy/makespan 通过初步门槛，但 longctx guardrail 尚未满足，完整目标仍未完成。

## E013 SDPA long-context fallback（完成）

- `longctx_probe.py --attention sdpa --lengths 2048 4096 8192` 在同一 910B3、BF16、greedy、相同 synthetic retrieval 条件下 9/9 正确；2048/4096/8192 的 needle fraction=0.1/0.5/0.9 均返回 `739281`。
- SDPA 8192 实测约 8.94–9.02 s，peak=57,954,415,616 bytes，未 OOM；相对 eager 的 8192 OOM，这是通用 attention backend 参数优化，不改模型/数据或题目特化。
- 需要注意：原 Phase 1 e2e 基线和 concise 都用 `attn_implementation=eager`。将 SDPA 用于完整 128 题需重新建立同协议 baseline/candidate，否则不能把 SDPA longctx probe 的结果直接归因到 concise。当前先记录为参数探索与 longctx guardrail 证据。

## E014 concise + SDPA full e2e（运行中）

- 已启动固定 concise 协议、唯一变量 `attn_implementation=sdpa` 的 128 题实验：`results/phase1_concise_sdpa_128.jsonl`。相同模型、数据、greedy、6 轮/512 token、8 threads、工具协议；禁止与其它 NPU 实验并发。
- 当前 12/128，12 正确，service median=20.798 s，generation=262.749 s；partial 结果不作最终结论。完成后与 concise/eager 比较 makespan 和 v2 accuracy，SDPA 长上下文结果只作独立 guardrail。

### E014 全量结果（完成）

- 128/128 完成；原始 v1=121/128，v2=122/128=0.953125，重评分文件 `results/phase1_concise_sdpa_128.scoring_v2.jsonl` 全量审计通过。
- makespan=2890.922601 s（48.182 min），相比 concise/eager 的 3055.056022 s 加速 1.0568x；相比 initial original/eager baseline 7836.171112 s 总加速 2.710x。生成 16,203 token，service median=18.953 s，completion p50=1495.983 s、p95=2749.625 s。
- v2 质量为 122/128，SDPA 三种长上下文长度和三种 needle 位置均通过。失败保留为真实算术/截断错误，v1 parser 误判由 v2 修复。
- 当前通过的 Phase 1 组合：通用 concise tool-call protocol + torch SDPA + BF16 + greedy；无 MTP/speculative，无 kernel 特化。Phase 2 多 agent makespan 与 Phase 3 FP8/kernel 尚未实现或验证，不能声称 README 三阶段全部完成。
- 当前 26/128，26 正确，service median=18.953 s，generation=555.741 s；仍为 partial evidence。
- 当前 12/128，12 正确，service median=20.798 s，generation=262.749 s；partial 结果不作最终结论。完成后与 concise/eager 比较 makespan 和 v2 accuracy，SDPA 长上下文结果只作独立 guardrail。
- E015 Phase 2 batch=2,8-task smoke 已完成：8/8 正确，batch wall makespan=362.609 s；相同任务 sequential concise 单题 median 约 20 s，batching 没有优势。这个结果仅 smoke，且不是全量并发结果。已扩展 `agent_batch_bench.py` 支持 concise protocol，下一步应对同步批处理做更合理的持续 batch 调度再扫描；不得拿重复计入的 task service time当 makespan。

### E015 Phase 2 batch=8 full result（完成）

- 128/128 完成，v2=122/128=0.953125，审计通过；makespan=817.811420 s（13.630 min），相对 Phase 1 concise+SDPA sequential 2890.922601 s 为 **3.535x**，相对原始 baseline 为 **9.583x**。生成 16,252 token，batch wall p50=456.634 s、p95=817.809 s。
- 该结果是同步批处理的固定 batch=8、离线全量可用场景；工具执行仍在每轮处理，未使用 speculative/MTP，未写 kernel。批次中的所有请求从 t=0 可用，满足 README 离线批处理场景。
- v2 accuracy 仍达到 122/128；当前 Phase 2 候选通过主质量护栏。需注意 `service_seconds` 是每 task wall 分摊/记录，主 makespan 必须取 summary wall 时间，不能把每题 service 相加。
- SDPA e2e 当前 56/128，54 正确（partial 0.9643），service median=19.368 s，generation=1282.395 s；前两个失败与 eager concise 相同（423 截断、552 计算错误），尚无新失败类型。

## E016 Phase 3 FP8 capability probe（完成，当前环境不支持）

- `scripts/fp8_npu_probe.py` 在 Ascend910B3 上对实际 per-tensor E4M3FN quantization、`torch.mm` 和 `torch_npu.npu_quant_matmul` 做最小探针。
- NPU `torch.float8_e4m3fn` copy/GEMM 不支持：acl 类型 36（UNDEFINED），`aclnnMm` 仅支持 float/float16/bfloat16。`npu_quant_matmul` 报 `only aclnnQuantMatmulV5 support A8W4` / `ERR01002 OPS invalid type`。结果与完整错误保存在 `results/fp8_npu_probe.json` / `.stdout.log`。
- 因此无法在当前 CANN/torch_npu 组合中实施 README 要求的 FP8 per-tensor 线性层；不能以 int8/A8W4 替代 FP8，也不能标记 Phase 3 完成。Phase 1/2 BF16+SDPA 的结果仍有效，但完整三阶段目标保持未完成。

## Final evidence audit (2026-10-15)

- `audit_run.py` 全量通过：original/eager baseline、concise/eager Phase 1、concise/SDPA Phase 1、concise/SDPA/batch=8 Phase 2，均 128 unique tasks、summary、协议字段和工具记录完整。
- `longctx_sdpa.jsonl`：9/9 retrieval checks correct；eager 8192 OOM 仍保留为边界证据。
- `fp8_npu_probe.json`：FP8 torch GEMM 和 npu_quant_matmul 均明确失败，故 Phase 3 未达成。最终状态是 Phase 1 和 Phase 2 已实测完成，Phase 3 被硬件软件栈能力阻断；不得标记整个 README 目标 complete。
- 追加尝试 torchair `npugraph_ex`：仍在 CANN graph capture 阶段失败，底层找不到 `aclmdlRICaptureBegin`（ERR00008）。结果保留于 `results/compile_probe_npugraph_ex.stdout.log`；因此当前两种通用图执行模式均不可用。

## E017 INT8 fallback probe after relaxing FP8 constraint（失败）

- 用户放弃 FP8 固定精度约束后，尝试硬件支持方向的 INT8 per-tensor `npu_quant_matmul`，实际形状 M=1/8/32/128、K=5120、N=17408，BF16 reference 对照。
- 四个形状全部报 `ERR01002 OPS invalid type: only aclnnQuantMatmulV5 support A8W4, please upgrade CANN`，没有产生可测输出或性能数据。证据：`results/int8_npu_probe.json` 和 `.stdout.log`。
- 因当前 CANN 版本连 INT8 线性层 fallback 也不可用，Phase 3 仍无法落地；不能把未执行的 INT8 或 FP8 方案写入模型，也不能宣称 Phase 3 完成。若未来升级 CANN 到支持 A8W4，需要重新做 kernel 对照、全量 accuracy 和 e2e makespan。

## E018 INT4 weight matmul capability probe（可执行但精度不足）

- 用户放弃固定 FP8 精度约束后，进一步探测当前 CANN 可用的 `npu_weight_quant_batchmatmul`（INT4 packed weight）接口；这是与此前 `npu_quant_matmul` INT8/A8W4 失败路径不同的算子。
- 形状 M=1/8、K=5120、N=17408 均可执行。per-output-channel 对称量化约 0.47--0.53 ms，group=128 约 0.85--0.95 ms；group=128 相对 BF16 reference RMS 误差约 11.8%，cosine 约 0.993。即使算子可运行，尚无模型级校准、greedy 输出或 GSM8K accuracy 证据，不能替换全模型。
- 结果保存在 `results/int4_weight_probe.json` / `.stdout.log`。该路径暂不进入 Phase 1/2 e2e；需要模型级量化校准和完整精度门槛，或升级软件栈后再继续。

## E019 Phase 1 shorter decode cap probe（淘汰）

- 为验证 48 分钟结果中 512-token 上限是否造成可避免的尾部开销，给 `agent_bench.py` 增加可配置 `--max-new-tokens`，保持默认 512 不变；在 concise+SDPA 下运行 8-task、上限 128 的 smoke。
- 8 题中 2 题首轮正好耗尽 128 token，未形成完整 tool call，准确率为 6/8；makespan 157.95 s，未显示足够时延收益。该候选淘汰，不做 128 题全量运行。证据：`results/phase1_short128_smoke8.jsonl`。

## E020 Phase 2 batch=16 full result（当前最佳）

- 在 concise protocol + SDPA + BF16 下，将同步离线 batch 从 8 增至 16；模型、数据、greedy、6 轮/512 token 和工具执行规则不变。
- 16-task smoke 为 16/16 正确，峰值显存约 60.5 GB。128-task 全量结果 `results/phase2_concise_sdpa_batch16_128.scoring_v2.jsonl`，审计 128/128 通过，v2 accuracy=122/128=0.953125。
- makespan=586.012400 s（9.767 min），峰值显存=61.76 GB；相对 batch=8 的 817.811420 s 加速 1.396x，相对 Phase 1 sequential SDPA 的 2890.922601 s 加速 4.933x，相对原始 baseline 的 7836.171112 s 加速 13.372x。
- 该结果仍是同步固定 batch 的 Phase 2，不使用 MTP/speculative，不写 workload 特化 kernel；所有 128 题从 t=0 可用，符合离线批处理场景。

## E021 Phase 2 batch=20 probe（OOM 淘汰）

- batch=20 的 20-task smoke 通过 20/20，峰值显存约 61.95 GB；但这不足以代表全量运行，因为不同题目 prompt 长度和 recurrent 临时张量会改变峰值。
- 128-task 运行在完成 60 题后 OOM：已分配约 57.81 GiB、剩余约 14 MiB，尝试额外分配 452 MiB 失败，无 summary，不能作为有效全量结果。原始 partial 文件 `results/phase2_concise_sdpa_batch20_128.jsonl` 和 stderr 保留；batch=20 淘汰，最终采用 batch=16。

## E022 Agent static length-aware schedule probe（无收益，淘汰）

- 用户指出应优化 agent 系统架构。先评估当前架构：batch runner 每轮使用一次 HF `generate()`，每个 request 的 decode KV cache 随调用结束释放，工具执行后重新构造完整消息并重新 prefill；调用间存在 round barrier。当前无可用图捕获和 paged/continuous batching runtime，贸然做 CPU 侧交错调用不能实现真正异步设备执行，并可能重复 prefill。
- 为减少变长问题在 left-padding batch 中的 padding，给 `agent_batch_bench.py` 增加 `--schedule length`：按问题 tokenizer 长度排序后做固定 16-task batch。32-task smoke 准确性通过已完成任务，makespan=204.739 s、peak=60.50 GB；相对 FIFO 估计未显示收益，不继续全量。结果 `results/phase2_length_schedule_smoke32.jsonl`。
- 该静态排序调度候选淘汰。后续架构收益应来自可跨工具轮保留 KV cache 的连续批处理 runtime 或 chunked prefill/decode serving，而非任务重排；需先确认兼容 Qwen3.5 Gated DeltaNet 状态的 runtime 能力。

## E023 Agent 状态复用审计（进行中）

- 用户要求把优化重点从单纯 batch 调参转向 agent 架构。审计 `transformers` 5.5.4 的 Qwen3.5 实现确认：`past_key_values` 是包含 16 层 attention KV 与 48 层 Gated DeltaNet recurrent state 的 `Cache`，`forward()` 和 `prepare_inputs_for_generation()` 均支持跨调用传入；因此工具返回后的增量 prefill 在接口层面可行，不能继续假定只能完整重算。
- 新增 `scripts/cache_reuse_probe.py`：构造 system/user -> assistant tool call -> tool output 的真实消息片段，比较完整 prompt forward 与先缓存前缀、再仅 forward 追加 token 的 logits（max error、argmax、两种 prefill 时间），并记录峰值显存。该脚本不改变现有 benchmark 协议。
- 首次执行未进入模型 forward：设备无可见活动进程但 npu-smi 仍报告 NPU0/NPU1 分别占用 20,414/25,410 MiB，模型加载 allocator warmup 仅剩 43.99 GiB 可用而申请 50.96 GiB，触发 OOM。原始日志保留在 `results/cache_reuse_probe.stdout.log`；这属于当前驱动/设备残留状态，不能作为 cache 数值不一致证据。
- 下一步必须在设备 HBM 清空后执行该 probe；若 logits bit/argmax 一致，再实现保留每个 agent `Cache` 的增量 batch runner，并分别测量完整重算与增量路径的 prefill、makespan、显存。若 cache 对追加 tool token 不支持 batch 动态扩展，则记录限制并转向 CPU 工具执行与 NPU decode 重叠。

## E024 缩小真实结构 cache API 探针（完成）

- 因完整 checkpoint 仍受设备残留 HBM 限制，使用真实 Qwen3.5 结构的 4 层随机模型在同一 910B3 上执行 `scripts/cache_api_tiny_probe.py`；层类型、Gated DeltaNet recurrent state、attention cache 和 SDPA 路径均来自当前 Transformers 实现。
- 32 token 前缀 + 11 token 追加：完整 forward 0.7715 s；前缀 forward 0.0558 s、带 `past_key_values` 的追加 forward 0.0591 s，增量路径相对完整重算 **6.72x**。峰值显存 2.90 GiB（缩小模型）。
- 输出有限且 argmax 一致；BF16/SDPA 的 logits max abs error=0.1171875、mean=0.0280，因此不能宣称 bit equality，但该误差来自不同 shape/算子路径，必须在完整模型上用 greedy token 和 GSM8K accuracy 护栏验证。
- 这证明 agent 架构方向可执行：工具轮次可以保存 cache 并仅编码新增 assistant/tool token。尚未把它接入全量 runner，原因是完整 27B cache 的显存占用、batch 动态 cache 管理和端到端正确性仍未测量。

## E025 完整模型复测前置检查（未通过设备条件）

- 2026-10-15 后续检查仍发现 NPU0/NPU1 分别约 20.7/25.4 GiB HBM 被占用，npu-smi 同时显示无可见运行进程。两张卡均无法为 54 GiB BF16 checkpoint 加上 cache/workspace 提供完整容量；这不是 Python allocator 可通过 `empty_cache()` 修复的进程内碎片。
- 因此暂不把未运行的 cache runner 计为性能结果，也不将缩小模型的 6.72x 外推到 27B。需要设备管理员清理残留上下文或重置 NPU 后，才能完成完整模型 cache logits/greedy、8-task smoke、128-task accuracy 和 makespan 对照。

## E026 设备恢复权限检查（外部阻断）

- 当前用户属于 `davinci`/`sudo` 组，但环境没有 `sudo` 命令；`npu-smi set -h` 明确要求 root。无法在不破坏其他任务的情况下执行 NPU reset 或清理设备级残留 HBM。
- 因此后续完整模型 cache runner 需要管理员在设备空闲窗口执行 reset/释放上下文。代码、缩小结构探针和验证标准均已准备，当前不再重复提交必然 OOM 的 54 GiB checkpoint 加载。

## E027 完整 27B cache 复用探针（完成，暂不采用）

- 设备 HBM 清理后，`scripts/cache_reuse_probe.py` 成功加载完整 Qwen3.5-27B BF16（加载 17.91 s，峰值 54.92 GB），并执行完整 prompt 与 prefix-cache+追加 token 对照。
- 51 token 前缀、25 token 追加：完整 forward 1.3597 s；prefix 0.6387 s + cached append 0.6378 s，总计 1.2765 s，仅 **1.065x**，短 prompt 下没有明显收益，不能把缩小模型的 6.72x 外推到完整模型。
- 完整路径与 cache 路径 logits 均有限且 argmax 一致，但 max abs error=7.09375、mean=0.6854；因此不满足 bit equality，也未达到可直接替换 agent runner 的正确性门槛。误差可能来自不同序列形状/SDPA 与 recurrent cache 路径，必须以逐 token greedy 和 GSM8K accuracy 进一步确认。
- 结论：完整模型 cache API 可运行，但当前简单“工具轮追加 token”方案收益不足且数值差异较大，暂不接入 Phase 2；后续应优先研究固定形状 cache/continuous batching，或只在追加上下文远大于前缀时采用。

## E028 decode-length 分组调度探针（淘汰）

- 为直接针对同步 batch 的 round barrier，给 `agent_batch_bench.py` 增加 `--schedule decode_length --order-file`：使用已完成 batch=16 运行的每题实际生成 token 数作为估计，将相近 decode 工作量的请求放入同一批次。
- concise + SDPA + BF16 + batch=16、16 题 smoke：准确率 16/16，makespan=75.724 s；同条件 FIFO 16 题 smoke=71.311 s，反而慢 **6.2%**。两次生成 token 分别 2037/2045，差异不足以解释收益，排序开销和批次组成漂移占主导。
- 该静态历史长度排序不进入全量；要消除 barrier，必须实现运行时 continuous admission/slot replacement，而不是重新排列离线任务。代码选项保留用于后续复现实验。

## E029 Ascend recurrent fused-op 能力发现（待实测）

- 当前 `torch_npu 2.9.1` 暴露了 `npu_recurrent_gated_delta_rule` 和 functional 版本，且 meta 注册要求 value 为 3D、state 为 4D；这正对应 Qwen3.5 Gated DeltaNet 的 recurrent decode 路径。
- Transformers 当前实际使用 `fla` 的 Python fallback（日志明确提示 Triton 不支持当前平台），因此 910B 原生 recurrent fused-op 是比静态任务排序更有潜力的优化点。
- 尚未直接替换模型调用：需要严格对齐 query/key/value/state 的布局、`g`/`beta` 参数和 state 更新语义，先在 4 层真实结构模型做 reference 数值对照，再进行单步 27B benchmark。未经该对照不能宣称可用或加速。

## E030 Ascend recurrent fused-op 实测（当前软件栈不可用）

- 新增 `scripts/npu_recurrent_probe.py`，按 `torch_npu` schema 对 inplace/functional 两个原生接口执行 B=1,T=4,H=2,D=8 BF16 探针。
- 两个接口均失败：`aclnnRecurrentGatedDeltaRule ... not in libopapi.so`、`ERR01004 OPS invalid pointer`。因此当前 CANN/driver 的 Python 暴露层存在，但底层 `libopapi.so` 没有实现，不能接入 Qwen3.5，也不能宣称 fused recurrent 加速。
- 这排除了当前软件栈上的最高优先级 kernel 候选；后续只有升级 CANN/driver 后才值得重跑。现阶段保留 SDPA + batch=16 为已验证最佳路径。

## E031 Ascend fused RMSNorm 探针（值得进入下一轮）

- 新增 `scripts/rms_norm_probe.py`，在 910B3 上对实际 hidden=5120、batch=16、BF16 输入比较 PyTorch reference 与 `torch_npu.npu_rms_norm`。
- fused 中位延迟 `89.8 us`，reference `252.5 us`，局部加速 **2.81x**；max abs error `0.0625`、mean `0.00155`，argmax 全部一致。
- 这是目前第一个底层可执行且数值可接受的 910B 原生融合算子候选。尚未直接替换模型：需确认 Qwen3.5 RMSNorm 的 epsilon/权重布局和返回 tuple 语义，并用 4 层真实结构模型逐层 logits 对照，再做完整 27B greedy/accuracy/makespan。

## E032 Qwen3.5 模型级 fused RMSNorm（缩小模型通过）

- 新增 `scripts/rms_norm_model_probe.py`，复制 4 层真实 Qwen3.5 结构模型，将全部 11 个 `Qwen3_5RMSNorm` 替换为 `torch_npu.npu_rms_norm`；保持 `1 + weight`、epsilon 和 BF16 输出语义不变。
- 32-token SDPA prefill：reference `0.8263 s`，fused `0.6796 s`，局部模型加速 **1.216x**；logits max/mean error 均为 `0`，argmax 一致，输出 finite。
- 该结果满足缩小模型的数值门槛，但仍不是完整 27B e2e 证据。下一步应在空闲 910B 上做完整模型单题 greedy smoke，再做 8 题 accuracy/makespan；若通过才可进入 128 题全量。

## E033 fused RMSNorm agent smoke（通过，尚未全量）

- `agent_bench.py` 增加 `--fused-rmsnorm`，在完整 Qwen3.5-27B 上替换 161 个 RMSNorm，其他条件固定为 concise + SDPA + BF16 + greedy。
- 1 题 smoke：1/1 正确，makespan=22.385 s；8 题 smoke：8/8 正确，makespan=160.271 s，959 generated tokens，peak/协议记录完整。
- 8 题结果相对既有 concise eager smoke 的 182.359 s 为 1.138x，但 attention backend 同时不同，不能把这部分全部归因于 RMSNorm；需与同条件 concise+SDPA 对照并完成 128 题 v2 accuracy/makespan，才可接纳为最终候选。

## E034 fused RMSNorm 128 题全量（淘汰）

- 完整运行 `results/phase1_concise_sdpa_fused_rmsnorm_128.jsonl`：128/128、审计通过，makespan `2818.777 s`；同条件 concise+SDPA 基线 `2890.923 s`，仅 **1.0256x**，说明局部 RMSNorm 收益被其他算子和调度开销稀释。
- v2 accuracy `121/128=0.9453125`，低于基线 `122/128=0.953125`，不满足质量护栏；失败包含 id=1001 在 fused 路径生成 `14:00` 而基线为正确的 `2:00 PM`，且逐题 greedy token 不一致（compare_runs mismatch=19）。
- 因为 accuracy guardrail 失败，该候选不能进入最终优化组合；保留完整原始、v2、审计和比较证据。fused RMSNorm 可作为未来在更高精度/稳定算子配置下重新评估的候选，但当前不能宣称端到端收益。

## E035 prefill-only fused RMSNorm smoke（淘汰）

- 针对 E034 的数值漂移，新增 `--fused-rmsnorm-prefill`：仅多 token prefill 使用 fused RMSNorm，单 token decode 恢复 reference，期望保留 prefill 收益并避免 decode 漂移。
- 8 题 concise+SDPA smoke 准确率 8/8，但 makespan `170.125 s`，慢于 fused 全路径 smoke `160.271 s`，没有显示端到端收益；因此按 smoke 门槛停止 128 题运行。partial 文件保留但不计优化结果。
- 新实验规则：只有小批量 smoke 同时显示准确率通过和明确端到端收益，才启动 128 题全量；否则只保留探针证据并淘汰候选。

## E036 Ascend fused SwiGLU 模型探针（暂不进入 smoke）

- `torch_npu.npu_swiglu` schema 可执行；在 4 层真实 Qwen3.5 结构模型替换 `silu(gate) * up`，32-token SDPA prefill 从 `0.7267 s` 降至 `0.6851 s`，局部仅 **1.061x**。
- logits max error `0.0601`、mean `0.0125`，argmax 一致。由于局部收益很小且已有 RMSNorm 的模型级替换出现 accuracy 回归，按 E035 smoke 门槛不启动端到端实验；保留为软件栈升级/更高精度配置后的候选。

## E037 当前 910B 最终候选排序（证据汇总）

- **采用**：BF16 + SDPA + concise protocol + 离线 batch=16；128/128 审计通过，v2 accuracy=122/128，makespan=586.012 s，是当前已验证最佳组合。
- **不采用**：fused RMSNorm 全路径（2818.777 s、121/128，质量护栏失败）；prefill-only RMSNorm（8-task 170.125 s，较全路径 smoke 更慢）；fused SwiGLU（局部仅 1.061x 且误差 0.0601）；cache 跨工具轮复用（完整 27B 仅 1.065x、logit max error 7.09）；静态 decode-length 排序（smoke 慢 6.2%）。
- **软件栈阻断**：FP8/INT8 线性层、native recurrent gated-delta-rule 和 graph capture 均由当前 CANN/driver 明确拒绝；INT4 算子可执行但尚无模型级校准和 accuracy 证据。
- **方法约束**：后续只对小批量 smoke 同时显示准确率通过和明确端到端收益的候选做全量；避免把单算子微基准或 partial run 当成优化成果。

## E038 vLLM-NPU 可比性检查（未能启动）

- 当前环境没有安装 `vllm`/`vllm-npu`/`vllm-ascend`。查询到的 `vllm-npu 0.4.2.post2` 依赖 `npu-vllm 0.4.2.post3`，后者硬性要求 `torch==2.1.0` 与 `torch-npu==2.1.0.post10`；当前环境为 torch/torch-npu 2.9.1，不能在现有环境中混装。
- 最新 `vllm-ascend 0.23.0.post1` 构建依赖 `torch-npu==2.10.0.post4`，当前索引无该精确版本，且其 Python/CANN 组合与本实验环境不匹配。未强行安装，避免破坏已验证的 Transformers 基线。
- 因此本轮没有可信的 vLLM-NPU 速度数字；需要单独提供匹配的 vLLM-NPU 镜像/环境后，才能用同一模型、输入和 batch 条件进行对照。

## E039 独立 vLLM-NPU 环境安装（后端组件缺失）

- 按用户要求新建隔离 conda 环境 `vllm-npu-py310`（Python 3.10），未改动 `ascend-e2e`。
- 成功安装 `torch==2.1.0`、`torch-npu==2.1.0.post10`、`npu-vllm==0.4.2.post3`、`vllm-npu==0.4.2.post2` 及其依赖；torch/torch-npu 可导入。
- 导入 `vllm` 失败于 `ModuleNotFoundError: mindie_llm`。系统路径和 PyPI 均没有 MindIE runtime；该组件不是 vllm-npu wheel 的依赖项，无法仅靠 pip 补齐。
- 因 vLLM-NPU 服务未能启动，本轮没有伪造速度数字。需要提供包含 MindIE runtime 的 Ascend/vLLM 容器或宿主机安装路径后，才能继续 benchmark。

## E040 vLLM-Ascend 新环境尝试（CANN ABI 不匹配）

- 新建隔离 conda 环境 `vllm-ascend-py310`，安装 `torch==2.10.0`、`torch-npu==2.10.0.post7`、`torchvision==0.25.0`；现有 `ascend-e2e` 未改动。
- 导入 `torch_npu` 在 backend autoload 阶段失败：`libtorch_npu.so: undefined symbol: aclrtBinaryLoadFromData`。当前宿主 CANN 8.0.RC3/ACL runtime 不提供该 ABI，故尚未安装/启动 vLLM-Ascend 服务。
- 结论：vLLM-Ascend 确实是 NPU 后端，但必须使用与其 torch-npu wheel 匹配的 CANN/driver 容器；Python 隔离环境无法修复宿主 ACL ABI。当前没有可信速度数字。

## E041 vLLM-Ascend 0.18.0 最小运行（已启动）

- 新容器映射 `/dev/davinci2/3` 成功；当前环境为 CANN 8.5.1、torch/torch-npu 2.9.0、vLLM/vLLM-Ascend 0.18.0。将 vLLM 核心对齐到 `0.18.0` 后，`vllm.tracing.instrument` API 通过。
- 单卡 NPU0、BF16、Qwen3.5-27B、`max_model_len=256`、`gpu_memory_utilization=0.98`：权重占 51.083 GB，编译/ACL Graph/warmup 后 KV cache 3,840 tokens（5.78 GiB），初始化约 91.6 s，ACL graph 首次单请求处理 65.8 s。
- 裸文本 prompt 生成 `?\n\n17`，说明请求未使用 Qwen chat template，不能用于 accuracy 结论；该结果只证明 vLLM-Ascend engine、模型加载和 NPU 执行链路已通。后续速度对照必须使用 tokenizer chat template，并分离冷启动编译与稳态请求。

## E042 vLLM-Ascend 128 题对照（完成）

- 新增 `scripts/vllm_agent_bench.py`，使用 Qwen chat template、concise tool protocol、greedy、每轮最多 512 token、最多 6 轮、batch=16；与现有 Phase 2 协议和数据 SHA256 一致。vLLM engine 使用 CANN 8.5.1、BF16、单卡 910B3、ACL Graph/continuous scheduler。
- 全量 128/128，审计通过；初始化/编译单独记录 `load_seconds=131.465 s`，主任务 makespan（排除加载）=`338.426 s`。v2 accuracy=`121/128=0.9453125`，比现有 Transformers 优化结果 `122/128=0.953125` 少 1 题，不能宣称质量等价。
- 相对现有优化 makespan `586.012 s`，vLLM 主任务墙钟约 **1.731x** 更快（减少 247.586 s）；若把 vLLM 冷启动计入一次性服务成本，总计约 469.891 s，仍约 **1.247x** 更快。该比较必须同时报告 accuracy 差异。
- vLLM 额外失败 id=304（无 boxed，prediction=null）；其余失败与既有 concise 结果相同。结果文件：`results/vllm_ascend_concise_batch16_128.scoring_v2.jsonl`，摘要和审计均保留。

### E042 可复现配置与结果明细

- 环境：conda `vllm-ascend-py310`，Python 3.10.22；CANN `8.5.1`；`torch==2.9.0`；`torch-npu==2.9.0`；`vllm==0.18.0`；`vllm-ascend==0.18.0`；单卡逻辑设备 0（宿主设备为 910B3）。
- 环境初始化：`source /usr/local/Ascend/cann-8.5.1/set_env.sh`，补充 driver/common `LD_LIBRARY_PATH`，`ASCEND_RT_VISIBLE_DEVICES=0`。
- runner 命令：`python scripts/vllm_agent_bench.py --limit 128 --batch-size 16 --max-model-len 2048 --output results/vllm_ascend_concise_batch16_128.jsonl`。
- vLLM 参数：`tensor_parallel_size=1`、`dtype=bfloat16`、`max_model_len=2048`、`gpu_memory_utilization=0.98`；引擎启用 chunked prefill、ACL Graph/Piecewise compilation、异步调度。初始化日志显示权重 `51.083 GB`、KV cache `23,040 tokens / 5.78 GiB`，ACL Graph 预热约 15 s，编译/初始化合计约 131.465 s。
- 协议：固定 `data/gsm8k_128.jsonl`（SHA256=`8ad965a8bea67b7a06454324427cebc1233a48624187dcb519eb0190b471fae4`），concise system prompt，原生 Qwen chat template，tool call 必须执行 Python，greedy、`max_tokens=512`、最多 6 轮；v2 重评分和 `audit_run.py` 均通过 128/128。
- 结果摘要：主 makespan `338.4262824319303 s`；`load_seconds=131.4646633900702 s`；若计入冷启动总计约 `469.891 s`；生成 token `17,172`；v2 `121/128`。失败 id：304（截断/无 boxed）、423（截断）、552、952、1035、1048、1161（答案或建模错误）。
- 证据文件：原始 `results/vllm_ascend_concise_batch16_128.jsonl`；v2 `results/vllm_ascend_concise_batch16_128.scoring_v2.jsonl`；审计 `results/vllm_ascend_audit.stdout.log`；摘要 `results/vllm_ascend_summary.json`；runner `scripts/vllm_agent_bench.py`。

## E043 batch=18 调度边界复测（因无收益迹象中止）

- 针对 E020 batch=16 与 E021 batch=20 OOM 之间的容量区间，运行 concise + SDPA + BF16 + FIFO batch=18；只改变 batch size，使用相同 128 题、greedy、6 轮/512 token 协议。记录：`results/phase2_concise_sdpa_batch18_128.jsonl` 和 stdout log。
- 运行约 170.6 s 后仅完成 19/128，首个 18 题批次用约 68.4 s；跨入第二批后累计 73.2 s generation/2814 tokens，id=304 单轮达到 512 token 上限、没有 tool call，使其余任务继续等待。设备占用约 64,681/65,536 MiB，接近容量边缘。基于早期 makespan 已显著落后 batch=16 的 586.0 s，且出现质量/尾延迟风险，按 E035 smoke/收益门槛中止，不把 partial 结果当作全量性能或 accuracy。
- partial 为 19/19 complete task、18 correct；不是 128 题 accuracy 证据。该结果证明 batch=18 在该 FIFO 顺序下没有显示扩批收益，但不能单凭 partial 精确估计全量时间。此前 batch=20 在完成 60 题后尝试再分配 452 MiB、仅余 14 MiB 而 OOM；继续加大固定 batch 既加剧显存峰值，也放大慢请求对同步轮的拖累。
- batch=16 完整数据表明 256 次同步生成轮耗时合计 580.656 s，任务 generation 占 99% 以上；最长同步轮 103.780 s。工具执行仅 4.588 s、prepare 0.648 s。vLLM 对照共生成更多 token（17,172 vs 16,145），主 makespan 仍低至 338.426 s，差距指向 continuous scheduling / per-token decode 路径，而非 prompt 或工具开销。
- 下一步不继续扫描固定 batch 或静态排序；先实现固定 cache 的逐 token greedy 对照，量出 HF `generate()` 与显式 forward 的每 token 开销并验证 token 一致性，再判断能否构造有 slot replacement 的动态 batch scheduler。E027 跨调用 cache 的 27B 短 prompt 仅 1.065x 且 logit max error 7.09，故必须重新验证数值和全流程收益。

## E044 显式逐 token decode loop（正确但更慢，淘汰）

- 新增 `scripts/decode_loop_probe.py`，在同一 Qwen3.5-27B、concise prompt、SDPA、BF16 条件下对比 HF `generate()` 和显式 prefix prefill + `past_key_values` greedy 单 token forward；上限 64 token。结果：`results/decode_loop_probe_64.jsonl` / `.stdout.log`。
- 两路径 64/64 token 完全一致，prompt=364 token；generate=13.122 s，显式路径 prefill=0.828 s、decode=11.010 s、总计=11.856 s；手写循环约快 1.107x（约 9.6%），median token forward=172.4 ms；两者峰值约 54.95 GB，几乎无内存收益。这个改善不足以抵消端到端批处理差距，且64 token 仍是被截断的单题微基准，不能宣称 GSM8K 加速。
- 本次实验可解释此前 CPU 逐 token dispatch 的风险：显式 forward 没有消除 Python/NPU 同步边界，单 token 调用仍占每 token 约 172 ms，整体只获得小幅变化。因此不把 loop 接入 agent runner。下一候选应依靠已有 vLLM scheduler/compiled serving，或实现能跨请求 slot replacement 的低层执行，而不是 Python token loop；任何替代路径仍须用完整 128 题质量与 makespan 对照。

## E045 扩大 vLLM 请求池至 32（通过，仍需继续优化）

- 将 E042 的外层请求批从 16 增为 32，模型、vLLM-Ascend 0.18.0、CANN 8.5.1、BF16、greedy、工具协议和数据不变。32 题 smoke makespan=101.198 s、31/32 正确；同一前 32 题在原 batch=16 全量运行中耗时 135.331 s、30/32 正确，smoke 约快 1.34x。
- 全量结果 `results/vllm_ascend_concise_batch32_128.scoring_v2.jsonl`：128/128，v2=122/128=0.953125，审计通过；makespan=301.510 s，生成 16,782 token。相比 vLLM batch=16 的 338.426 s 加速 1.122x（减少 36.917 s），且 v2 多答对 1 题；比较工具确认协议相同、accuracy guardrail 通过，但 31 个 task 的 greedy token 不同，故不是 bit-equal 输出。
- 本次配置仍未优于 vLLM 自身的优化结果：Transformers/HF batch=16 为 586.012 s，约慢 1.944x。E045 证明增加调度器可见请求池有效，但外层工具轮仍有 batch barrier。下一步实测 128 请求一次提交；请求可排队并由 vLLM 连续调度，但当前日志中 KV cache 只有 23,040 tokens（在 2048 max length 下约 12.2 个满长序列），不等于 128 个 agent 同时占用设备执行。

## E046 Transformers 使用 CANN 8.5.1 对照（无显著收益）

- 为排除 E020 的 Transformers/CANN 8.0 系列与 E042 的 CANN 8.5.1 不同造成比较偏差，在 `ascend-e2e` 中只切换运行时 toolkit 到 CANN 8.5.1，保留 torch/torch-npu 2.9.1、Transformers 5.5.4、SDPA、BF16、batch=16 和前 32 题。
- 32 题完整 smoke makespan=226.328 s、31/32 正确；原工具链同批任务的 completion span=220.379 s、31/32。改变 toolkit 未改善性能，且略慢约 2.7%；不能解释 HF 与 vLLM 的约 1.9x 差距。数据：`results/phase2_concise_sdpa_cann851_batch16_smoke32.jsonl`。

## E047 vLLM 一次提交 128 agent 请求（完成）

- 将 runner 外层 batch 调为 128，让全部固定任务进入一次 `llm.generate()` 请求池；模型/协议/greedy/工具行为相同。128/128 完成，无 OOM；原始准确率 121/128，v2=122/128，makespan=239.194 s，生成 16,629 token；重评分和完整审计已通过。
- 引擎日志明确报告 KV cache=23,040 tokens，在 max_model_len=2,048 下最大并发=12.20 个满长序列。故 API 能同时接收 128 个 agent 并由 scheduler 排队/连续调度，但不能声称 128 条 2,048-token 序列同时驻留执行。实际短请求可超过满长并发数，真实活跃数随 KV 使用变化。
- 相对 batch=32 的 301.510 s 加速 1.261x；当前最佳 vLLM 基准为 239.194 s。相比 HF/Transformers batch=16 的 586.012 s 仍快约 2.45x。完整结果：`results/vllm_ascend_concise_batch128_128.scoring_v2.jsonl`、stdout log、summary 和 audit。

## E048 最短代码提示协议（淘汰）

- 统计 vLLM batch=128 concise 输出：256 轮平均 64.96 token，中位数 51.5、p95=178、最大 512；128 个工具代码含 236 行注释/约 7,947 注释字符。generation 是绝对主耗时，因此新增通用 minimal 协议，要求 Python 程序无注释/解释并尽量短。
- 新增 `MINIMAL_SYSTEM`，同步扩展 HF batch runner 与 vLLM runner 的 protocol 选择，不改评分、工具执行或输出格式。Transformers smoke（batch=16、16 题）16/16 正确，1,846 token，相对 concise 同一前缀 2,045 token 少 9.7%；makespan 63.847 s，相对 66.996 s 快 1.049x。
- vLLM 同协议 smoke（batch=16、16 题）16/16 正确，1,883 token、30.649 s；同一前缀的 vLLM concise batch=16 full run 为 2,042 token、41.175 s，minimal 约快 1.344x。提示缩短同样帮助 vLLM，因此不能据此主张相对优势。
- vLLM minimal batch=128 全量 128/128，v2=121/128、makespan=238.562 s、15,640 token；相对 concise vLLM 239.194 s 仅快 1.003x，质量少 1 题，提示压缩并未给 vLLM 端到端带来实质收益。
- HF minimal batch=16 全量运行在 64/128 后因明显落后而停止：partial 61/64 correct，generation 累计 487.052 s，尚未完成剩余 64 题；这已经消耗 HF concise batch=16 完整 128 题 586.012 s 的 83.1%，不能视为有效加速结果。证据 `results/phase2_minimal_sdpa_batch16_128.jsonl` 保留 partial，不作完整 accuracy/makespan。
- 失败原因：16-task smoke 的短题输出少约 9.7% token，但复杂题 id=428 的 HF minimal 输出仍有大量注释，512 token 截断在 `</tool_call>` 前，tool_count=0；同一题 HF concise 仅 242 token 且答对。通用提示对短题有小收益，却让复杂题输出长尾/截断更严重，所以淘汰 minimal prompt。下一次提示候选应先对 hard-task 集做约束验证，不能只看简单题 smoke；当前继续转向 cache/scheduler/kernel 架构。

## E049 Transformers 原生 continuous batching 适配审计（当前模型不兼容）

- 检查当前 Transformers 5.5.4 的 `generation.continuous_batching`：它的 `PagedAttentionCache` 管理 attention KV block；Qwen3.5 Gated DeltaNet 前向则还读取 `cache_params.layers[layer_idx].conv_states/recurrent_states`，并调用 `update_conv_state`、`update_recurrent_state`。PagedAttentionCache 没有这些 recurrent-state 接口，不能将其直接用于 Qwen3.5 的连续批处理。
- 因此不把 HF continuous batching API 当作可直接开启的优化项。下一次需要先为 48 层混合架构实现并验证 NPU 上的 hybrid paged-KV + per-slot recurrent/conv state cache，再做动态 slot admission 的小模型 logits/token 一致性测试；若该状态接口或算子继续受限，再转向模型库提供的 NPU GDN 实现。历史 `npu_recurrent_gated_delta_rule` 探针已因当前 CANN 缺失底层算子失败，需保留为显式软件栈限制。

## E050 完整模型 recurrent cache dtype 复测（修正 E006 假设）

- 在完整 Qwen3.5-27B、BF16、SDPA 上执行单 prompt prefill 后读取真实 `past_key_values`：48 个线性注意力层的 recurrent state dtype=`torch.bfloat16`，每层 shape=`[1,48,128,128]`，合计 75,497,472 bytes/请求（72 MiB）；conv cache 合计 3,932,160 bytes（3.75 MiB）。这是运行时 tensor 证据，优先于根据算子中间态推断缓存 dtype。
- 修正 `scripts/model_memory.py` 的 recurrent state 每元素字节数为 2 并重跑，`results/model_memory.json` 已更新为 72 MiB/请求。此前 FP32 估算错误，因此 batch=20 OOM 不能归因于“FP32 recurrent cache”；实际还需考虑模型权重、prefill/decode 临时激活、SDPA/GDN workspace 与 allocator。batch18/20 的实测容量边界仍成立。

## E051 vLLM-Ascend GDN kernel 复用探针（单层通过，模型适配被 TBE 阻断）

- 在 vLLM-Ascend Python 3.10/CANN 8.5.1 环境直接调用其 `fused_recurrent_gated_delta_rule`，实际 Qwen3.5 形状 B=1,T=1,H=48,K=V=128；与 vLLM-Ascend 自带 PyTorch reference 对照：输出 max abs error=`7.45e-9`、mean=`1.21e-12`，state max abs error=`0.0078125`，结果 finite；kernel median=`0.417 ms`。这证明该 kernel 本身可执行且数值对齐，单层探针不是完整模型性能证据。
- 新增 `scripts/gdn_kernel_model_probe.py`，尝试在 Transformers 5.5.4 中仅替换 recurrent GDN 函数，并显式转换 HF `[B,H,K,V]` 与 kernel `[B,H,V,K]` state layout。模型加载后第一层 `conv1d` 编译失败：CANN 报 `ModuleNotFoundError: No module named 'tbe'`、`ACL_PRECISION_MODE`/op-store 初始化错误。原因是 vLLM Python 环境的 CANN Python/TBE 组件不完整，不能通过复制 vLLM kernel wheel 修复；不能把这次失败写成 kernel 数值或模型逻辑失败。
- 下一步若要把该候选接入自有框架，需要提供完整匹配的 CANN Python/TBE 环境，或在原 `ascend-e2e` 环境安装与当前 torch-npu 对应的 vLLM-Ascend kernel 依赖；然后先做 4 层真实结构 logits/greedy 对照，再做完整 27B smoke 和 128 题 accuracy/makespan。当前已验证的自有框架最佳仍是 BF16+SDPA+concise+batch=16（586.012 s、122/128），vLLM baseline 为 239.194 s（concise，128 request pool），目标尚未达成。

## E052 GDN kernel 模型级 adapter（集成成功但单题变慢，淘汰）

- 修正 E051 的环境问题：保留 `set_env.sh` 提供的 CANN TBE `PYTHONPATH`，在 Python 3.10 overlay 中加载 Transformers 5.5.4，并用 state layout adapter 将 vLLM kernel 接入 HF Qwen3.5 recurrent path。完整 27B 单题生成 109 token，tool call 正确、输出 finite；结果 `results/gdn_kernel_model_probe.json`，日志 `results/gdn_kernel_model_probe.stdout.log`。
- 同一环境、同一 prompt、同一 token 数的 reference 对照 `results/gdn_model_reference_probe.stdout.log`：reference generation=`42.243 s`，kernel adapter=`46.673 s`，kernel 反而慢约 **10.5%**，两次都正确调用工具并打印 `8`。因此未启动 128 题全量。
- 失败原因不是 kernel 数值：E051 单层 reference 对照误差很小；本次端到端 adapter 每次 recurrent 调用都需要 HF `[B,H,K,V]` 与 kernel `[B,H,V,K]` 的 transpose/contiguous，且当前场景 batch=1、短 decode，kernel launch/转换开销超过算子收益。下一次若继续，应做 batch decode 专用的原生 state layout，避免每层转置，并先以 batch=16 decode-only probe 验证局部收益；当前不接入主 runner。
- 追加 B=16、T=1 的 decode-only probe 未产生可用计时：kernel 在重复调用期间触发 Ascend vector core exception，错误码 `507035`（`rtDeviceSynchronizeWithTimeout`）。设备随后恢复 `Health=OK` 且无残留进程，但该形状在当前 CANN/Triton 组合下不稳定。故 batch kernel 路径正式淘汰，不能接入自有框架；若未来升级 CANN/Triton，需从该形状重新做稳定性、数值和性能三重验证。

## E053 final-turn generation cap（性能候选，通过 v3 质量门槛）

- 在 E020 保存的 batch-generation 计时中，第二轮 8 个 batch 有 7 个耗时 `8.087–12.749 s`；包含 id=423 的 batch 耗时 `103.780 s`。该题第二轮生成 512 token 且无 boxed 答案，其余 15 题最终轮最多 12 token。首轮工具调用仍可能需要长代码，因此不降低它的 512-token 上限。
- 给自有 `agent_batch_bench.py` 增加默认不改变行为的 `--final-max-new-tokens`；只对工具结果后的轮次设为 32。相同前 32 题 smoke：E020 的两个 FIFO batch generation 合计 `219.007 s`，候选整批 makespan `129.912 s`；均为 31/32，唯一失败 id=423，token cap 消除了该 batch 的 512-token 长尾。
- 128 题 full run：`511.197 s`、原始输出 121/128、15,827 token、peak `62,097,888,256` bytes；相对 E020 `586.012 s` 快 `1.146x`。v2 比 E020 少 1 题，新增 id=1001 输出 `14:00`。同一任务在默认 512-token replay 中仍输出 `14:00`，与 final cap 无关；这是跨运行 greedy 轨迹漂移。
- 检查发现 v2 只把 `2:00 PM` 归一到标签 `2`，未识别语义相同的 24 小时制 `14:00`。新增独立 `scoring_v3.py`，只扩展通用整点时间等价关系；保留 v2 文件不变，并用 v3 重评分 E020、E047 和候选。三者均为 122/128，失败 ID 完全相同（423、552、952、1035、1048、1161）；三份 v3 文件的完整 128-task audit 均通过。
- 该候选在 v3 下保持质量护栏，接纳为 E053；但它仍不是 SOTA，长度分组继续作为下一项单变量优化。

## E054 按观测首轮生成长度分组（当前自有最佳）

- E020 首轮 8 个 batch generation 时间与每组最大生成长度高度线性：拟合斜率约 `0.188 s/max-token`、截距 `3.879 s`、`R^2=0.9997`。基于 E053 的首轮长度做离线分组模拟，128 题各组最大长度总和从 `2057` 降到 `1188`；这是筛选依据，性能结论仍来自 e2e。
- 新增自有 runner 调度项 `generation_length`，按同一数据集此前运行中每题首轮 token 数降序排序，模型、提示、SDPA、BF16、greedy、batch=16 和 final cap32 均不变。前 64 题 smoke：FIFO `229.003 s`，长度分组 `171.751 s`（`1.333x`）；v3 都是 62/64，失败 ID 为423、552，peak 约 `60.5 GB`。
- 128 题全量结果 `results/phase2_concise_sdpa_batch16_final32_length128.scoring_v3.jsonl`：审计 128/128，v3 `122/128`，失败 ID 与 E020 相同；makespan `346.604 s`，generation `341.070 s`（首轮 `269.897 s`、第二轮 `71.173 s`），15,696 token，peak `62,098,578,432` bytes。
- 相对 E053 final32 FIFO `511.197 s` 快 `1.476x`，相对 E020 `586.012 s` 快 `1.691x`；仍比 vLLM SOTA `239.194 s` 慢 `1.449x`。这成为当前自有最佳，但目标尚未完成。排序依赖同工作负载先前观测长度；用于本固定离线 workload，不改变模型输出约束。

## E055 真实长 prompt 工具轮 cache 复用探针（正确性不通过）

- 选取实际较长的 id=640：首轮 prompt `459` token，greedy 工具调用 `428` token；工具执行后完整第二轮 prompt 为 `936` token。`generate(return_dict_in_generate=True)` 返回的 cache 长度 `886`，其 token prefix 与完整第二轮 prompt 对齐；只需追加 `50` token。
- 单题局部 prefill 从完整路径 `1.093 s` 降为 cached append `0.719 s`（`1.52x`），peak `55.34 GB`。但 full 与 cached 路径的 logits max abs error=`6.53125`、mean=`0.9820`，argmax 不同。违反固定轨迹正确性门槛，不接入 runner；不能把局部速度收益计作 e2e 优化成果。

## E056 排序 batch=16 + fused RMSNorm（当前自有最佳）

- 在 E054 的同一长度分组、concise、SDPA、BF16、final-turn cap32、batch=16 路径上启用 161 个 `torch_npu.npu_rms_norm`；模型、任务顺序和工具协议不变。64 题 smoke `210.299 s`，相同排序 BF16 前缀约 `230.325 s`，约 `1.095x`，v3 `59/64`，无新增失败。
- 128 题全量 `results/phase2_concise_sdpa_batch16_final32_length_rmsnorm_128.scoring_v3.jsonl`：audit 128/128，v3 `122/128`，失败 ID `[423,552,952,1035,1048,1161]`，makespan `325.242 s`，generation `319.853 s`，prepare `0.678 s`，tool `4.565 s`，15,716 token，peak `62,094,008,832` bytes。
- 相对 E054 `346.604 s` 加速 `1.066x`，相对 E020 `586.012 s` 加速 `1.802x`；仍比 vLLM SOTA `239.194 s` 慢 `1.360x`。在 v3 下 RMSNorm 不引入质量损失，接纳为当前自有最佳。
- E034 的旧 sequential RMSNorm 候选曾因 v2 的 id=1001 `14:00` 评分缺口被标为质量失败；v3 复评为 `122/128`，但其端到端收益只有约 `1.026x`。E056 的新 batch/调度组合才是有效的 RMSNorm 端到端证据。

## E057 排序 batch=18 边界复测（淘汰）

- 在 E056 完整相同配置上仅将 batch size 从 16 改为 18，保留历史首轮生成长度分组、fused RMSNorm、SDPA、BF16 和 final-turn cap32。36 题 smoke：`104.848 s`、v3 `35/36`、唯一失败 id=423、peak `61.60 GB`，因此进入全量。
- 128 题全量审计通过，v3 `122/128`，失败 ID 与 E056 相同；makespan `326.863 s`，generation `321.455 s`，peak `63,020,450,816` bytes。相对 E056 `325.242 s` 慢 `0.5%`，没有端到端收益，且显存余量更小。
- batch=18 淘汰；不再继续扫描固定 batch。后续只有动态 admission/紧凑 live rows 或能够减少 recurrent/decode 固定开销的实现，才值得进入下一轮。

## E058 prefill-only fused RMSNorm（淘汰）

- 在 E056 相同排序前 64 题、batch=16、SDPA、BF16、final-turn cap32 条件下，仅多 token prefill 使用 reference `Qwen3_5RMSNorm._norm` + `(1 + weight)`，单 token decode 使用 `torch_npu.npu_rms_norm`；工具协议与长度分组不变。
- smoke makespan `299.168 s`，scorer v3 `57/64`；同前缀 E056 全路径 fused RMSNorm 为 `230.325 s`、`59/64`。新增失败 id=304、640，端到端更慢，候选淘汰，不扩展全量。
- 该结果不改变 E056 的当前最佳；不继续混用 reference/fused RMSNorm 路径。

## E059 prefill-only fused SwiGLU（淘汰）

- 在 E056 同排序前 64 题上，仅多 token prefill 将 `silu(gate) * up` 替换为 `torch_npu.npu_swiglu(cat(gate, up))`，单 token decode 保持 reference；同时保持 E056 fused RMSNorm、BF16、SDPA、final cap32 和 batch16。
- smoke makespan `286.367 s`，scorer v3 `58/64`；同前缀 E056 为 `214.179 s`、`59/64`。新增失败 id=304，端到端明显变慢，候选淘汰，不扩展全量。

## E060 FP16 计算形态（淘汰）

- 将 E056 完整配置的权重/计算 dtype 单独由 BF16 改为 FP16，在相同排序前 64 题 smoke；其余仍为 SDPA、fused RMSNorm、batch16、final cap32。
- FP16 smoke makespan `225.768 s`，慢于同前缀 BF16 `214.179 s`；scorer v3 `58/64`，新增失败 id=640。候选不扩展全量；BF16 保持当前实现精度。

## E061 Qwen3.5 hybrid cache 行压缩能力探针

- Transformers `DynamicCache.batch_select_indices` 遍历 cache layers，但当前 Qwen3.5 `LinearAttentionLayer` 未实现该方法。按真实 cache 字段手动对 attention `keys/values`、`conv_states`、`recurrent_states` 做 batch `index_select`。
- 4 层真实 Qwen3.5 结构（含 linear/full attention）B=2 prefill 后选择第 2 行：8 个已初始化 state tensor 均与源 cache 对应行逐元素一致；压缩耗时约 `9.44 ms`，后续单 token 输出 finite，peak `3.15 GB`。结果：`results/cache_compaction_probe.json`。
- 该探针只证明 state row selection 可实现，不证明变长序列拼接、slot admission 或完整生成正确。

## E062 每 token finished-row compaction（端到端淘汰）

- 临时实现 per-token greedy loop：生成 token 命中 EOS 后从 hybrid cache 选择仍活跃行继续解码。测试使用 E056 首个排序 batch 16 题和 E056 相同 HF 配置。
- 单 batch replay generation `77.069 s`，相对 E056 `82.075 s` 快约 `6.1%`；但 16 题中 10 题首轮 token 轨迹不同。Scorer v3 得分均为 15/16，唯一失败均为 id=1161，未通过固定轨迹门槛。
- 同排序前 64 题 smoke makespan `280.813 s`，v3 `59/64`，失败集合与 E056 前缀相同；E056 前 64 题 makespan `214.179 s`。id=304 首轮输出从约 110 token 增至 424 token，延长 batch 长尾。候选淘汰，不扩全量；compaction 实验开关已从 runner 移除，证据保留于 `results/phase2_compact_decode_sorted_batch1_replay16_fixed.*` 和 `results/phase2_compact_decode_sorted_smoke64.*`。

## E063 batch16 NPU trace 与 Inductor 能力探针

- 用 E056 首个排序 batch 的真实 16 条问题和 fused RMSNorm 路径采集 8-token profile，关闭 shape/memory trace，输出到 `/tmp/profile_batch16_e056_compact`。profiled wall `37.318 s`（含 profiler 开销），NPU device 时间窗口 `6.693 s`、operator union busy `5.208 s`（`77.8%`）；idle gaps 最大约 `17.3 ms`。这些带插桩比例只用于定位，不作为 e2e 性能估计。
- 设备 duration top 包含 MatMulV3 `1.441 s/400`、Mul `0.737 s/10,720`、EVENT_WAIT `0.449 s/124`、ReduceSum `0.437 s/4,464`、MatMulV2 `0.424 s/3,576`、Slice `0.407 s/6,807`。这表明当前 batch16 时间里有 recurrent/elementwise 小算子和等待成本，但 temporal union 不能证明其依赖关键路径；GEMM 也占显著时长。原首轮 trace 写入 workspace 失败于 `No space left on device`、只剩 CPU 前缀，已删本次不完整目录；缩小 trace 使用 `/tmp` 成功。
- 另对实际 `torch_recurrent_gated_delta_rule` B=16,T=1 试 `torch.compile(..., backend='inductor')`；backend 初始化失败于 Triton `0 active drivers ([]). There should only be one.`，没有编译输出或性能数字。与已记录的 ACL graph capture 失败不同，是当前 torch_npu Inductor/Triton backend 不可用；不在本环境重复。
- 下一步先分析可支持的通用融合机会，必须对参考输出/固定轨迹和端到端 smoke 双重验证；E063 不能支持直接开发新的 recurrent kernel。

## E064 阈值式 live-row compaction（淘汰）

- 为减少 E062 每 token 改变 batch shape 的影响，尝试在当前 batch 有 75% 行完成后才 compact 一次；阈值前保持原 batch，并对已完成行输入 pad token。使用 E056 排序首批16题。
- generation `80.197 s`，与 E056 同 batch `82.075 s` 接近；16题中10题首轮 token 仍不一致，v3 都为 `15/16`，唯一失败 id=1161。未通过固定轨迹门槛且没有明确端到端收益；不扩展到64/128题。
- 证据：`results/phase2_compact_threshold75_replay16.scoring_v3.jsonl`。compaction 试验代码已从 runner 移除。

## E065 试验 runner 清理与审计

- `agent_batch_bench.py` 保留 BF16、final-turn token cap、按生成长度分组和全路径 fused RMSNorm；移除 E058–E060 的 prefill-only/FP16/SwiGLU 开关及 E062/E064 的 compact decode 分支。`--task-ids` 保留用于匹配样本重放。
- `py_compile` 通过；E056 scorer v3 全量 audit 仍为 128/128、无错误。当前自有最佳保持 E056 `325.242 s / 122/128`，项目目标仍未完成。


## E066 fused causal Conv1d 能力检查（不可用）

- 实际 Qwen3.5 decode 形状 B=16、T=1、D=10240、K=4 的接口探针失败：`aclnnFusedCausalConv1d`/workspace symbol 不在 `libopapi.so`。没有数值或性能结果；不接入模型。日志 `/tmp/fused_causal_conv_probe.log`。

## E067 固定 batch 显式 greedy loop：输入准备修正

- 初版手写循环没有复用 Qwen3.5 `_prepare_position_ids_for_generation`，16题有10题 token 不同；这与 E062/E064 的手写循环存在相同遗漏。因此此前“形状变化导致轨迹变化”的归因证据不足，不能据此排除正确实现的 compaction。
- 修正为复用模型 position-ID 和 generation input/update 方法，并设 `logits_to_keep=1`（漏设时一次 OOM，不作为候选能力结论）。固定 batch16 两轮 replay 与 E056 所有 token 完全一致，audit通过、v3为15/16，唯一失败id=1161。
- 首轮 generation `80.865 s`，E056同批 `82.075 s`；整批 `94.341 s`。约1.5%的首轮差异未证明稳定收益，不扩大全量。证据 `results/phase2_explicit_greedy_sorted_replay16_v3.scoring_v3.jsonl`。
- 下一步使用已验证输入准备流程检查 compaction 的第一处轨迹分歧；先区分输入/position错误与实际形状数值差异，再决定是否做性能smoke。当前最佳仍为E056，未超过SOTA。

## E068 position-ID 修正后的 finished-row compaction（验证中）

- 在 E067 已匹配 HF generation 的四维 position-ID/input 准备流程上，仅添加 EOS 后行移除；使用 cache 自带 reorder_cache 同时处理 attention KV、conv/recurrent state，并同步选择 attention mask 和 position IDs。
- 首个排序 batch16 审计通过，makespan 87.955s（E056同批95.226s），两轮 generation 74.944s/12.242s，准确率同为15/16；唯一token差异为id214首轮。此前E062/E064的10题漂移大部分来自错误输入准备，其负面结论不能直接推广至修正实现。
- 仍未通过严格token一致门槛；开展匹配排序前64题的性能/质量诊断，结果不作为正式接纳证据。当前自有最佳仍是E056。

### E068 64题结果

- 完成匹配排序前64题，audit无错误，v3为59/64，失败集合与E056相同，仅id214首轮token不同。makespan211.195s，E056同前缀214.179s，约1.4%改善；不足以证明稳定e2e收益，不扩全量。

## E069 通用单token recurrent路径简化（模型验证中）

- 新增`recurrent_single_step.py`：对T=1保持reference的l2norm、FP32运算顺序与state输出，只删除Python序列循环、零输出buffer和切片写回。T>1回退reference。
- B=1与B=16、H=48/K=V=128局部交替A/B：output/state逐元素一致，最大误差0；B16中位数reference1.006ms、candidate0.670ms；B1为0.929ms/0.702ms。原始样本保存在`results/recurrent_single_step_probe.json`。
- 48层应用时粗估每decode step节省约16ms（48乘局部差值），但并发/调度重叠使这个估计不能直接作为Amdahl占比。仅进入同排序64题smoke，未接纳，当前最佳仍为E056。

### E069 64题smoke完成

- 同排序64题audit通过、v3为59/64，所有两轮token与E056完全一致，总token10255，失败集合相同。makespan174.001s，参考214.179s，改善18.8%（1.231x）；peak62094008832 bytes。
- 已进入128题完整验收，结果未完成前不更新最佳。

### E069 完整128题验收：接纳为当前自有最佳

- `results/recurrent_single_step128.scoring_v3.jsonl` 全量audit通过，128个唯一任务、v3为122/128，失败集合与E056相同；逐题两轮所有token与E056完全一致，总生成15716token，peak62094008832 bytes。
- makespan280.735644s，相对E056325.242017s加速1.159x，相对E020586.012400s加速2.087x。仍慢于vLLM对照239.194363s，项目未完成。
- 相比64题smoke的18.8%降幅，全量降幅13.7%；局部计时受batch长度、prefill占比及运行漂移影响，使用完整端到端结果作为接纳依据，不将算子1.50x当全模型收益。
- 复现：`source scripts/env.sh`后运行`python scripts/agent_batch_bench.py --protocol concise --attention sdpa --batch-size 16 --final-max-new-tokens 32 --fused-rmsnorm --single-step-recurrent --schedule generation_length --order-file results/phase2_concise_sdpa_batch16_final32_128.jsonl --output <new-output>`。历史长度估计文件依赖不变，未在本次makespan计入生成该文件的离线历史运行成本。

## E070 recurrent临时state原位更新（不扩展）

- 在独立FP32临时state上mul_/add_，确保不改输入cache。BF16/FP32输入state的output/state均逐元素一致且输入保持不变；B16的BF16 state仅从0.698ms变0.688ms，FP32 state从0.651ms变0.664ms。收益不明确，不进入e2e。证据`results/recurrent_temporary_state_probe.json`。

## E071 GDN四输入投影合并能力探针（不接纳）

- 同hidden输入的qkv/z/b/a四Linear合为单GEMM，实际N=[10240,6144,48,48]、K=5120。M=1/16/512的局部中位耗时分别从0.122/0.108/0.450ms降到0.080/0.082/0.347ms。
- M16 z输出max误差0.0078125，M512四输出均不bit equal；数值路径改变，且该局部收益不足以支撑当前大差距，未接入模型。证据`results/packed_gdn_projection_probe.json`。

## E072 recurrent乘向量归约改batched matmul（不扩展）

- 两次(state*vector).sum改为vector@state，保持FP32；B16局部中位数0.643ms/0.621ms，仅约3.4%收益，output max误差0.00024414、state max误差4.77e-7。未通过逐元素一致且收益小，不进入e2e。证据`results/recurrent_matmul_reduce_probe.json`。
- E070/E072的临时实现参数已移除，recurrent_single_step.py恢复E069已验证路径。当前最佳仍280.736s/122，未超过SOTA。

## E073 E069组合长上下文验收

- `longctx_probe.py`新增显式RMSNorm/recurrent候选参数，实际启用E069组合，SDPA、BF16、greedy、32-token检索输出；2048/4096/8192三个长度、needle位置0.1/0.5/0.9共9/9正确，未OOM。原始记录`results/longctx_e069.jsonl`。
- 该护栏覆盖E069新增decode实现与RMSNorm组合的长上下文运行，不替代GSM8K全量质量，也不是性能比较。

## E074 chunk prefill去掉row/sub clone（不接纳）

- 参考chunk递推的被写行i不在读取的子矩阵[:i,:i]内；尝试直接使用views，保持数学运算顺序。B=1/16、T=512、H=48/K=V=128，output/state均逐元素一致。
- B1中位数16.13ms降至13.16ms，但B16从59.94ms升至61.83ms；非连续子矩阵读取抵消复制收益。主负载B16没有收益，不集成模型。候选仅在独立probe模块`chunk_without_clones.py`，runner不引用；证据`results/chunk_without_clones_probe.json`。
- 当前最佳仍为E069 280.736s/122，且E073长上下文护栏通过。下一步需要改变较高占比的prefill/decode执行，而不是继续扩大低收益微调。

## E075 GDN prefill chunk-size计算形态（候选验证中）

- 实际H48/K=V128/B16/T512测试chunk=64/32/16，中位数62.40/47.60/45.37ms；B1中位数14.64/13.06/20.22ms。32的output max误差0.00024414、state误差5.36e-7，非bit equal；选32作为模型候选，局部证据`results/chunk_size_probe.json`。
- E069同排序首16题，chunk32审计通过、v3为15/16、唯一失败仍id1161；makespan53.921s，E069同batch80.234s。但5题首轮token不同，因此速度收益包含生成轨迹变化，不宣称kernel等价或纯算子1.49x加速。
- 64题结果：chunk32为v3 `58/64`、makespan `208.163 s`、10,452 token；E069同前缀为`59/64`、10,255 token。新增失败id=304：首轮生成512 token截断、tool_count=0；基线用106 token完成工具调用。chunk32淘汰，不做全量。

## E079 chunk48最终复评（性能不接纳）

- 64题先验显示chunk48 v3 `60/64`，相对E069 `59/64`修复id552且无新增失败，故做了128题验收。全量audit通过、v3 `123/128`、token15,819；makespan `288.597s`，比E069 `280.736s`慢约2.8%。按主指标不接纳，E069仍是当前最佳。
- runner中的chunk size实验开关已移除，保持E069默认chunk64；完整数值结果和脚本固定默认路径均可复现。

## E080 grouped matmul与微型state变体的最终筛选

- grouped projection实际GDN形状M16/K5120/N=[10240,6144,48,48]慢约24%，且z分量非exact（E076）；不接入。
- recurrent临时state原位更新、addcmul及乘向量matmul仅约1–3.4%局部收益或产生浮点误差（E070/E072/E078）；不进入端到端。
- 清理未接纳实验的参数和分支后，`agent_batch_bench.py`只保留已验证的BF16、final-turn cap、长度分组、fused RMSNorm和E069单token recurrent实现。E069全量scorer v3 audit仍128/128、122正确；项目目标未完成。

## E076 GDN grouped projection（实际形状探针淘汰）

- 小BF16接口探针B=16/K=512/N=1024，两路`npu_grouped_matmul`输出与独立Linear逐元素一致，接口可调用。
- 实际GDN输入投影M=16/K=5120/N=[10240,6144,48,48]：4个Linear中位`114.4 us`，grouped matmul `142.1 us`（慢约24%）；z分量max误差`0.0078125`，非exact。候选不接入、不做e2e。证据`results/gdn_grouped_projection_probe.json`。

## E077 GDN chunk size 48（质量通过、性能淘汰）

- 在E069完整配置上只把GDN prefill chunk从64改为48。64题smoke 188.940s、v3 60/64，较E069前缀快；token有变化但未新增失败，于是扩展全量。
- 128题audit通过，v3 `123/128`，失败`[423,952,1035,1048,1161]`，id552由错变对；makespan `288.597 s`，generation `283.278 s`，15,819 token，peak `61,976,304,128` bytes。比E069 `280.736 s`慢2.8%，质量提高不抵端到端时延，淘汰，E069保持最佳。

## E078 recurrent state原位更新与AddCMul（不扩展）

- 临时state在FP32下原位更新，不修改输入cache，output/state逐元素一致。B16局部仅约1–2%收益，未进入e2e。证据`results/recurrent_inplace_state_probe.json`。
- 单独`torch.addcmul`同样exact，B16约2%局部收益，不进入端到端。证据`results/recurrent_addcmul_probe.json`。

## E081 AOT eager recurrent compile（淘汰）

- 对E069的T=1 recurrent函数使用`torch.compile(backend="aot_eager", fullgraph=True)`，不依赖Triton/ACL graph capture。B=1/16输出和state均bit equal，但B16中位数由`0.650 ms`升至`1.827 ms`，慢约2.8倍；不接入。样本`results/recurrent_aot_eager_probe.json`。

## E082 GDN低收益微变体复核（不接纳）

- chunk48全量v3为`123/128`，但makespan `288.597 s`，慢于E069 `280.736 s`；chunk32 64题出现id304工具调用512-token截断，不扩展。chunk-size参数从runner移除，保持E069默认chunk64。
- 模型实际GDN投影的GroupedMatmul慢约24%且z输出max误差`0.0078125`；`npu_bmmV2` state contraction慢约33%并改变数值。两者不接入。
- BF16/FP32临时state原位更新和`addcmul`均逐元素一致，但B16局部收益约1–2%，不足以进入e2e。
- l2norm经RMSNorm缩放转换仍改变数值（B16 output/state max error约`0.00195/0.0121`）且更慢；淘汰并恢复E069 reference l2norm。无缩放的早期探针公式不等价，不作为性能证据。
- 这些探针均未取代E069；当前目标仍是提高自有端到端吞吐并超过vLLM对照，而不是增加局部microbenchmark成绩。

## E083 torch_npu.npu_linear 实际形状对照（不接纳）

- 对`F.linear`与`torch_npu.npu_linear`做BF16相同输入/权重/偏置A/B，形状(M,K,N)=(1,5120,17408)、(16,5120,17408)、(16,5120,6144)。三种均逐元素exact。
- 中位延迟分别为linear `79.86/99.63/46.06 us`、npu_linear `80.84/100.65/48.88 us`；NPU接口略慢，模型当前已使用等价PyTorch路径，无收益不替换。结果`results/npu_linear_probe.json`。

## E084 Gated RMSNorm/SwiGLU融合（当前自有最佳）

- `Qwen3_5RMSNormGated`原路径先对hidden做FP32 RMSNorm与weight缩放，再计算`SiLU(gate) * normalized`。采用`npu_rms_norm`与`npu_swiglu`完成设备融合。真实decode行形状`[768,128]`局部探针约快`2.18x`，最大误差`0.0625`，非逐元素相同，故再做端到端质量门槛。
- E069排序前64题：E069 `280.736 s / 59/64`，融合候选`175.988 s / 59/64`；失败ID一致，peak不变，但候选多生成261 token。
- 128题结果`results/gated_rmsnorm_128.scoring_v3.jsonl` audit通过，v3 `122/128`，失败集合与E069一致，生成15,981 token，peak `62,094,008,832` bytes；makespan `271.120104 s`，generation `265.792 s`。
- 相对E069端到端加速`1.035x`，相对E020 `2.162x`。该组合接纳为当前自有最佳；仍比vLLM SOTA `239.194 s`慢约31.9秒，目标未完成。
- E084 profile同排序首batch16题采8-token短trace：device window `6.122 s`、operator union busy `5.106 s`（83.4%），MatMulV3 `1.439 s/400`、Mul `0.700 s/9,568`、ReduceSum `0.436 s/4,464`、MatMulV2 `0.424 s/3,576`、Slice `0.405 s/6,807`。profile wall受采集开销影响，不作为e2e计时或Amdahl比例。

## E085 E084长上下文护栏

- 启用E084全组合运行独立needle检索：2048/4096/8192长度、位置0.1/0.5/0.9共9/9正确，无OOM。记录`results/longctx_e084.jsonl`。
- 此结果验证E084组合长上下文护栏；不替代GSM8K精度或性能验收。

## E086 z/b/a decode投影打包（模型级淘汰）

- 局部M16测试将GDN z/b/a三个共享hidden的投影合为一组矩阵乘，输出逐元素exact、局部约快27%。为控制显存，不合并更大的qkv投影；仅在T=1 decode使用打包权重，prefill保持原Linear。
- 初次smoke因参数未注册而未启动。有效decode-only z/b/a打包结果见E087；本节局部结果与最终结论不作全量证据。

## E086b GDN四投影打包（端到端容量淘汰）

- M16/K5120/N=[10240,6144,48,48]局部打包GEMM较四个Linear快，但z输出有`0.0078125`误差（E076后续packed probe）。按共享hidden输入合并每层四投影，在runner中尝试权重预打包后进行64题端到端smoke。
- 首轮prefill申请额外`174 MiB`失败，NPU已分配`53.43 GiB`、仅剩`8.71 MiB`，无有效task和makespan。该方案的workspace/峰值显存无法满足当前单卡batch16容量，不接入；runner开关和临时helper已移除，失败记录`results/packed_gdn_smoke64.log`保留。

## E087 z/b/a decode打包全量验收（性能淘汰）

- 只合并GDN z/b/a三个小投影用于T=1 decode，qkv及所有prefill保留原实现。64题smoke v3 `59/64`，失败集合与E084前缀相同，生成token总数相同、peak仅增加约4.5MB；makespan `177.560 s`，对照前缀`271.120 s`。
- 128题完整audit通过，v3 `122/128`，失败集合和15,981生成token与E084完全相同，peak `62,098,538,496` bytes；makespan `281.090 s`，比E084慢约3.7%。局部smoke收益未推广到全量，候选淘汰；runner选项/helper已移除。

## E088 最新结果审计与路线校正

- 检查到 `results/int8_weight_only_probe.log` 于2026-10-18更新，包含M=1/16、K=5120、N=17408两条成功记录。相邻 Ascend 实验目录解释了 API 语义：BF16 activation、对称 per-output-channel INT8 weight、BF16 antiquant scale，使用 `npu_weight_quant_batchmatmul`；它与失败的 INT8 activation/weight `npu_quant_matmul` 是不同算子路径。
- 为保留工作区可复现性，新增 `scripts/int8_weight_only_probe.py`，读取本模型 layer 8 `mlp.gate_proj.weight` 并以固定种子生成 BF16 激活，对 BF16 Linear 与 weight-only INT8 做同步计时和输出比较。脚本语法检查通过，实际910B3运行结果写入 `results/int8_weight_only_probe_real_weight.json`；原始不明来源 log 保留不覆盖。
- 此轮更新纠正了“INT8 GEMM 均不可用”的过宽结论，但尚无模型级量化或端到端收益。后续按 E089 的实测决定该算子只作为可能的内存容量工具，不纳入当前推理路径。

## E089 真实 MLP checkpoint 权重量化层探针（延迟候选淘汰）

- 使用 `model.language_model.layers.8.mlp.gate_proj.weight`（BF16 `[17408,5120]`），按输出通道对称量化为 INT8，scale 为 BF16；输入是固定 seed 的标准正态 BF16 激活，M=1/16。它覆盖真实权重但不覆盖真实前向激活分布，因此仅是层级筛选，不是模型轨迹证据。
- 首次运行曾用上中位数，已统一为8样本标准中位数并重跑。复跑结果 M=1：BF16 Linear `0.132 ms`、INT8 weight-only `0.143 ms`（慢约8.4%），relative RMS `0.94%`，top-1一致；M=16：BF16 `0.132 ms`、INT8 `0.136 ms`（慢约3.2%，低于单次微基准能可靠区分的幅度），relative RMS `0.95%`，通道 top-1 一致15/16。量化权重由 `178,257,920` 降至 `89,128,960` bytes，但单层节省约85 MiB没有形成可确认的延迟收益。
- 不进入任务 smoke 或主 runner。仅保留为潜在显存/并发方向：须先证明完整模型权重确实是 batch=16 扩容瓶颈，且量化后可安全提高有效 batch；否则关闭此路线。证据及复现命令：`scripts/int8_weight_only_probe.py`、`results/int8_weight_only_probe_real_weight.json`。

## E090 INT8 扩批方向暂停与调度路线重排

- E057 已提供当前主配置的固定 batch 边界：batch=18 全量 `326.863 s`，比 batch=16 `325.242 s` 慢约0.5%，峰值显存 `63,020,450,816` bytes；因此仅靠 INT8 权重省显存以增大同步 batch，缺乏已测收益依据。E089 的 M=16 权重量化算子标准中位数仅慢约3.2%，但仍没有稳定端到端收益，不应增加主路径算子成本。
- INT8 weight-only 不进入 e2e 实现，除非未来动态 admission 能证明额外驻留权重空间可提高有效并发，而不是单纯增大同步 batch。
- 下一候选改为离线分析 `results/gated_rmsnorm_128.jsonl` 与 E020 的同步轮数据，按轮统计活跃 agent、实际轮数、最长/中位输出长度及尾部等待上界。分析结果若支持动态 admission，才以 hybrid recurrent/conv state + attention KV slot 管理为实现目标；E049 的 PagedAttentionCache 不兼容性仍是明确约束。该路线尚未实现或验证，不计作优化收益。

## E091 E020/E084 同步轮尾部离线归因

- 对既有 JSONL 按 `(turn, batch_generation_seconds)` 聚合同一同步调用中的题目，不启动模型或占用 NPU。E020/E084 各有16次同步调用；所有128题都恰好两轮，每次调用初始 batch size 为16。此统计是每次调用的输入行数，不是每个 decode step 的未完成行数；E084 批内长度差异意味着 EOS 后的 finished-row compaction 仍可能省去部分 row-token 工作。
- E084 首轮8批的最大输出长度为 `[428,170,137,115,100,91,79,70]`，批内中位数为 `[196.5,155.5,125,107,93.5,81.5,73,62.5]`；末轮最大值为 `[12,32,10,9,9,12,10,10]`，中位数为 `[7,7.5,7,7,7,7.5,7,7]`。按每批最长序列估算，padding 上界首轮4,024 token、末轮699 token，合计4,723 token（约为15,981个有效生成token的29.5%）；这只是可消除工作的粗上界，不是可达到的时间加速。
- E084 的首轮 batch generation 合计204.676 s，末轮61.116 s；全部 generation 为265.792 s，和全量记录相符。E020 对应同步生成合计580.656 s；其中末轮有一个512-token请求令该批耗时103.780 s，正是 E053 final-turn cap32 已处理的长尾，不应重复优化。
- 结论修正：E084 的 padding 上界使 finished-row compaction 值得在当前配置重新验证；E068 基于较早 E056 配置的64题改善约1.4%、仍有一题token差异，不能直接外推或淘汰 E084 候选。本轮以准确的 Qwen3.5 position/cache 更新流程复测 E084 首批16题，结果见 E092；跨批次 admission 属于后续更大改造。
- 复现命令：`python scripts/analyze_batch_tail.py results/phase2_concise_sdpa_batch16_128.jsonl results/gated_rmsnorm_128.jsonl`；结构化输出保留每轮调用的 active rows、最长/中位 token、padding 上界和同步时间。

## E092 E084 finished-row cache compaction replay（淘汰）

- 新增 `scripts/compact_decode_probe.py`，从 E084 同排序的首个16题 batch 构造 concise prompts，加载相同 BF16+SDPA 模型并启用 fused RMSNorm、Gated RMSNorm/SwiGLU 和单步 recurrent。先跑标准 `model.generate()`，再仅替换 greedy `_sample` 循环：EOS 行完成后用 `past_key_values.batch_select_indices` 压缩 hybrid cache，同时选择 attention mask、position IDs 和生成序列索引。
- 16题生成 token 与 `results/gated_rmsnorm_128.jsonl` 对照全部逐 token 完全一致。标准生成耗时 `71.852 s`，compaction `74.296 s`，慢约3.4%；未插桩同步计时下没有兑现 padding 上界。该实现增加了 cache gather/索引管理，候选淘汰，不接入主 runner、不扩展64/128题。
- 复现命令：`source scripts/env.sh && python scripts/compact_decode_probe.py --task-ids 640,1137,214,1030,950,428,866,175,1130,445,1161,611,108,1288,554,249 --order-file results/phase2_concise_sdpa_batch16_final32_128.jsonl --reference-run results/gated_rmsnorm_128.jsonl --output results/compact_decode_e084_replay16.json`。结果 `results/compact_decode_e084_replay16.json`，完整 stdout 保存在同前缀 `.stdout.log`。
- E084 仍是自有最佳；项目目标尚未完成。下一候选只能是跨批次 hybrid-cache admission，先建立收益/实现成本估算，不预设该路线有效。

## E093 跨批 admission 的工作量与服务时间上界

- 从 E084 逐题生成长度得到有效token `15,981`，按 batch16 做理想无 padding 均摊需约`998.8`个 decode step；当前16批同步调用的每批最大长度总和为`1,294` step。理想均摊减少约`22.8%` row-step，约为生成时间的上限信号，不是墙钟预测；它忽略新 prompt prefill、工具依赖、cache 搬运和小 batch 计算效率。
- Python 工具实际累计`4.545 s`，即使完全与 NPU 重叠，也无法单独覆盖 E084 与 E047 的`31.9 s`差距。跨轮 cache 保留/连续 admission 才可能改变主要成本。
- 结合 E084 同步时间对批最大生成长度的线性拟合，斜率约`0.138 s/token`、截距约`5.42 s/call`，但不能把截距当纯 prefill 或用该回归断言 continuous-batching 收益。新的 slot prefill 形状/分批方式尚未实测。
- 结论：存在值得验证的理论工作量空间，但 E092 单纯压缩行数已慢3.4%；要利用空间必须在保持 B16 的情况下及时补槽，并分组处理新 prompt，不能每个 EOS 后串行做单请求 prefill。

## E094 四层 Qwen3.5 hybrid cache slot reset/rebase（语义探针通过）

- 新增 `scripts/cache_slot_reset_probe.py`，使用真实 Qwen3.5 4层结构（3 linear-attention + 1 full-attention），随机权重和 BF16 输入。清零 slot0 的 recurrent/conv state、用 attention mask 屏蔽旧 KV、追加11-token 新 prompt，再对 full-attention KV 按行有效 mask 压缩。
- 与独立 fresh prefill 对照，替换后 logits top-1 一致（max abs error`0.046875`）；再解码一步仍 top-1 一致（max abs error`0.0390625`）。KV 序列长从59 rebase到49；这是单 seed/small model 的接口语义证据，不是完整模型正确性或性能结论。
- 结果：`results/cache_slot_reset_probe.json`；运行：`source scripts/env.sh && python scripts/cache_slot_reset_probe.py`。

## E095 完整 27B hybrid cache slot reset/rebase（单槽 top-1 通过）

- 新增 `scripts/cache_slot_reset_model_probe.py`，使用 E084 配置、真实 E084 首 batch16 prompt 初始化 cache，并尝试将 slot0 替换为按历史生成长度排序的下一题 id=1226。其新 prompt 为393 token；原 KV 序列宽459，替换后临时增至852，按每行 attention mask 重排有效 KV 后压至436。
- full checkpoint 下，新任务 prefill 与独立 fresh prompt 首步 top-1 相同，rebase 后下一步 top-1 也相同；对应 logits max abs error为`0.4375/0.5859375`，不是逐元素相同。cache 从`1,752,170,496` bytes降至`1,728,053,248` bytes，完整模型 peak=`60,535,555,584` bytes。
- 同配置复跑加计时：单请求 fresh prefill=`2.194 s`；在B16 cache中替换一行并处理dummy rows的prompt prefill=`4.217 s`；全16行KV rebase=`0.181 s`。因此不能逐 EOS 串行补槽；需要批量补入多个空槽。单槽/单步 top-1不证明完整greedy轨迹正确。
- 证据：`results/cache_slot_reset_model_probe_timed.json` 和 `.stdout.log`；命令 `source scripts/env.sh && python scripts/cache_slot_reset_model_probe.py --order-file results/phase2_concise_sdpa_batch16_final32_128.jsonl --output results/cache_slot_reset_model_probe_timed.json`。只有固定32题、16-slot 实验显示轨迹稳定且端到端净收益后，才实现主 runner 接入。

## E096 固定 slot admission 原型（OOM，停止扩大）

- 新增 `scripts/streaming_slot_probe.py`，尝试在固定16-slot内首轮生成时补入后续任务：新 prompt 先做成组 prefill，再把新行的 recurrent/conv state 写入空槽、按有效 attention mask 重排旧 KV，保持 batch 行数不变。候选只覆盖32题首轮 tool-call，不执行工具轮，不接入主 runner。
- 第一版把新 prompt 建立为独立 cache 后合并，NPU 在前向申请约98 MiB时 OOM（active约60.01 GiB）；第二版改为复用已有 cache、原位写槽和重排 KV，仍在首轮/补槽前向阶段以同样内存边界 OOM。两次均无 task 输出、无 accuracy/makespan，不能视为性能或质量结果。日志：`results/streaming_slot_e084_32.stdout.log`、`results/streaming_slot_e084_32_reusecache.stdout.log`。
- 结论：当前 Transformers Python cache 合并会产生临时 full-width attention/recurrent workspace，无法在 E084 已接近 HBM 上限的 batch16 上安全完成 admission。保留脚本作为失败复现，但停止继续调阈值、扩题数或接入主 runner；后续需固定容量 paged/hybrid cache 或 CANN/vLLM runtime 层支持后再重开。

## E097 continuous admission 离线理论上界收敛

- 新增 `scripts/simulate_admission.py`，只读取完整 JSONL，不启动模型。E084 `gated_rmsnorm_128.jsonl` 的有效生成token为15,981；固定16行同步批次消耗1,294个最大长度 step，理想连续 admission 为998.8个等效 step，理论减少22.8%。原始 E020 因首轮/末轮长度差异更大，理论减少61.3%，但 E053 的 final-turn cap 已改变尾部结构，不能把该上界直接套到 E084。
- 该输出只描述 row-step 工作量，不含 prompt prefill、cache rebase、工具依赖、设备 workspace 和小批效率。E096 已证明在当前 HBM 边界用 Python cache 合并会 OOM，因此不再把理论上界当作继续实现的理由。
- 当前路线收敛：E084 的剩余差距需要 runtime 级 continuous/hybrid batching；在本软件栈没有固定容量 slot/cache API 前，继续修改 `agent_batch_bench.py` 只能重复 E092/E096 的失败模式。保留模拟器作为未来 CANN/vLLM runtime 变更后的进入门槛。

## E098 npugraph_ex recurrent graph probe（捕获失败）

- 当前环境 `torch._dynamo.list_backends()` 已列出 `npugraph_ex`；旧 E010 的 `torchair reduce-overhead` 输出也提示该模式已弃用。因此新增 `scripts/npugraph_ex_probe.py`，不复用旧 backend，直接用 `torch.compile(..., backend="npugraph_ex", fullgraph=True, dynamic=False)` 测试实际 recurrent decode 形状 B=16,T=1,H=48,K=V=128。
- reference 运行正常，8组样本约`1.01–1.06 ms`；backend 在首次 capture 失败于 `copy_between_host_and_device_opapi`，CANN 返回 `ERR00100/107030`（captured stream 不允许同步 memcpy）。没有 output/state、编译稳态时间或 graph speedup，不能接入模型。
- 结果 `results/npugraph_ex_probe.json` / `.stdout.log`；脚本已通过语法检查。结论：图执行路线不是“backend 名称未切换”问题，当前 recurrent 函数仍包含 capture 不支持的 host/device copy；依赖或模型前向实现变化后再重开 capability probe。

## E099 npugraph_ex 简单设备图对照（后端可用但更慢）

- 新增 `scripts/npugraph_ex_simple_probe.py`，使用无 cache/host copy 的实际设备计算 `silu(x @ w)`，形状 `[16,1024] @ [1024,1024]`，与 eager 逐元素比较。`npugraph_ex` 首次编译约`0.993 s`，输出 bit equal、max error=`0`。
- 稳态 eager 样本约`0.137 ms`，compiled 样本约`0.52 ms`，compiled 约`3.8x`更慢；因此后端本身可捕获，但当前软件栈的 graph overhead/算子选择不适合把简单局部计算迁入图执行。没有模型级或端到端收益，不接入 runner。
- 结果 `results/npugraph_ex_simple_probe.json` / `.stdout.log`。图路线状态改为：recurrent capture 仍失败，简单图可运行但变慢；依赖、编译器或算子选择变化后才重开，不重复当前环境探测。

## E100 final-turn cap12 全量验收（接纳为当前自有最佳）

- 在 E084 相同模型、BF16、SDPA、concise、batch16、固定历史长度排序、fused RMSNorm/Gated RMSNorm、single-step recurrent 配置下，仅将 final turn 的生成上限从32降为12；首轮上限仍为512，工具协议、数据 hash、greedy 和最多6轮不变。原始运行：`results/phase2_concise_sdpa_batch16_final12_length128.jsonl`，派生 v3：`results/phase2_concise_sdpa_batch16_final12_length128.v3.jsonl`。
- 全量 `audit_run.py` 通过（128/128，metadata/summary 完整）。scorer v3 为 `122/128`；失败集合 `[1161, 423, 952, 1035, 552, 1048]` 与 E084 完全一致，没有新增失败。唯一评分派生修正仍是 scorer v3 对 id=1001 的 `2:00 PM` 解析，和 E084 一致。
- 未插桩 makespan `265.030128 s`，generation 记录对应约265.030 s，peak `62,094,008,832` bytes；总生成 token `15,961`。E084 cap32 为 `271.120104 s / 15,981 tokens / 122/128`，因此同环境端到端减少 `6.090 s`（约2.25%），显存峰值不变。相同排序首批16题的 cap12 smoke 也比 cap32 快约9.3%，但不作为全量结论。
- `task 423` 在 cap12 下仍是已知失败，且其第二轮由 cap32 的32 token缩短至12 token；由于失败集合未改变，未引入新的质量损失。该协议改变只适合作为当前离线工作负载候选，不能宣称通用对话质量等价。
- 该结果成为当前自有最佳：`265.030128 s / 122/128`，相对 E047 SOTA `239.194 s` 仍慢 `25.836 s`（约1.108x）。继续保持 runtime 级 continuous/hybrid batching、固定容量 cache 或 recurrent graph 支持为唯一高收益路线；Python cache admission、finished-row compaction 和当前 npugraph_ex capture 均已有失败证据，不重复实现。

## E101 decode-length 调度复核（不接纳）

- 在 E100 相同配置下，对同一固定 32 题 smoke 分别使用现有 `generation_length`（按首轮生成长度排序）和 `decode_length`（按全轮 token 总量排序）。两次均保持 `31/32` 正确；`decode_length` makespan `81.485 s`，`generation_length` `81.806 s`，差异约 `0.4%`，处于单次 NPU smoke 波动范围，不能作为稳定端到端收益证据。
- 结果保留于 `results/e101_decode_length_smoke32.jsonl` 和 `results/e101_generation_length_smoke32.jsonl`。不扩展 128 题，避免用小样本噪声替换全量验收；E100 的 `generation_length` 排序继续作为当前离线 workload 配置。
- 该复核没有改变路线判断：静态排序已接近可获得的批内尾部收益，剩余差距需要固定容量 hybrid cache/continuous batching 或 recurrent graph runtime 支持。

## E102 decode-length 重复测量（确认噪声，不接纳）

- 重复运行 E101 的 `decode_length`、同一 32 题、batch16、cap12 配置，得到 `83.430 s / 31/32`；首次 decode-length 为 `81.485 s`，generation-length 对照为 `81.806 s`。同一候选自身重复波动约 `2.4%`，超过首次两种排序的 `0.4%` 差异。
- 结果 `results/e102_decode_length_smoke32_repeat.jsonl`。这确认 E101 的微小差异是测量噪声范围，decode-length 不扩展到全量，也不改变 E100 当前最佳。

## E103 npugraph_ex 固定 bool wrapper（仍捕获失败）

- 针对 E098 的失败点新增 `scripts/npugraph_ex_const_wrapper_probe.py`：将 recurrent 函数的 `output_final_state` 和 `use_qk_l2norm_in_kernel` 两个 Python bool 固定在 wrapper 闭包中，只把六个 NPU tensor 作为 graph 输入，验证是否能消除 host 参数拷贝。
- wrapper 仍在首次 capture 的 `copy_between_host_and_device_opapi` 报 `ERR00100/107030`，没有编译稳态样本或输出结果。结果 `results/npugraph_ex_const_wrapper_probe.json`，完整日志 `results/npugraph_ex_const_wrapper_probe.stdout.log`。
- 结论：失败不只是 bool 参数传递；当前 Transformers recurrent 实现内部的 dtype/layout 转换或生成中间张量仍触发 capture 不支持的同步 memcpy。图执行路线在该软件栈停止继续修改，只有 CANN/torch_npu 修复或原生无拷贝 recurrent 实现后才重开。

## E104 GDN kernel batch16 probe（环境阻断，未形成性能结论）

- 为验证 E051/E052 留下的 batch16 可能性，扩展 `scripts/gdn_kernel_model_probe.py` 支持 `--batch-size`，保持 vLLM Ascend kernel adapter 和真实 HF 模型前向不变。
- 当前 `ascend-e2e` 环境没有 `vllm`；切换到 `vllm-ascend-py310` 后又与当前 Transformers/PyTorch overlay 不兼容（先后出现缺少 `Qwen3_5ForConditionalGeneration` 和 PyTorch C extension 加载错误）。没有模型加载、kernel 调用或 timing 数据，不能把本轮视为 kernel 淘汰或收益。
- 保留 batch-size 扩展和错误日志 `results/gdn_kernel_model_probe_b16.stdout.log`，待可复现的 vLLM/Transformers 双环境重新建立后再测。当前主 runner 不接入该 adapter；E100 仍为最佳。

## E105 vLLM GDN kernel 独立 batch16 微基准（待 state 语义验证）

- 在独立 `vllm-ascend-py310` 环境直接调用 `fused_recurrent_gated_delta_rule`，真实 decode 形状为 B=16、T=1、H=48、K=V=128，8组每组20次同步调用；输出和 state 均 finite，kernel 中位数 `0.3339 ms/call`。结果 `results/gdn_kernel_batch_probe.json`，脚本 `scripts/gdn_kernel_batch_probe.py`。
- `inplace_final_state=False` 返回 `[1,48,128,128]` 是 API 对 T 个 token 输出 final state 的形状设计，不足以判断 batch state 是否错误。改用 `inplace_final_state=True` 和 `ssm_state_indices=torch.arange(16)` 后，16 行 state 均被更新，形状保持 `[16,48,128,128]`；单看 shape 的疑点已澄清。
- 首次 batch-vs-single 对照曾报告 output/state 不一致；该对照把原位更新后重复计时循环的最终状态与初始 state 的单请求结果比较，参照状态不一致，结论无效。修正后的同初态对照见 E106。
- 复现脚本 `scripts/gdn_kernel_batch_probe.py`，结果 `results/gdn_kernel_batch_probe.json`。E100 仍为当前最佳。

## E106 GDN batch16 同初态 state/output 对照（batch自洽，HF语义未对齐）

- 修正 E105 的验证顺序：先用同一个初始 state 分别执行一次 B16 和16次 B1，并分别保存结果；正确性对照完成后，再在另一份独立 state 上做原位重复计时，避免计时循环演进 state 后污染对照。
- B16 的 16 行 state 全部更新；batch16 与逐行 batch1 的 output 和 state 均 bit-equal，最大误差均为0。输出 shape `[16,1,48,128]`，state shape `[16,48,128,128]`。B16 原位 kernel median `0.363 ms/call`，仅说明该通用接口形状数值正确且可执行，不含 HF 的 transpose/contiguous、权重投影或完整 GDN 调用成本。
- 结果 `results/gdn_kernel_batch_probe.json`；脚本 `scripts/gdn_kernel_batch_probe.py` 语法检查通过。修正后 vLLM kernel 的 batch 语义不再是 blocker，但 E052 真实 HF batch16 adapter 曾触发 vector core exception，且单请求 adapter 比 reference 慢10.5%。下一步须在兼容的 vLLM/Transformers overlay 下重测 HF batch16 token轨迹与端到端前向成本；未完成前不接入主 runner。E100 保持当前最佳。
- 追加独立 FP32 reference recurrence 对照后，kernel 与公式 reference 的 output 最大误差为`3.59375`、state 最大误差为`1.2109375`。虽然 kernel batch-vs-single 自洽，但当前随机输入的 kernel 参数语义（尤其 g/beta 预处理或 layout）尚未与 HF reference 对齐；不能把该 kernel 的微基准时间用于性能推断。需要按 Qwen3.5/vLLM 完整投影后的 q/k/v/g/beta 语义做对照，当前不继续扩大适配。

## E107 真实 Qwen3.5 batch16 GDN 跨环境 replay（正确性未通过）

- 将验证拆为三个脚本：`scripts/gdn_real_batch_capture.py` 在 `ascend-e2e` 加载真实 Qwen3.5-27B 并从真实 B=16、T=1 decode 调用点抓取 CPU `q/k/v/g/beta/state` 和 HF output/final state/logits；`scripts/gdn_real_batch_kernel.py` 在 `vllm-ascend-py310` 只读取相同 artifact 并执行 Ascend kernel；`scripts/gdn_real_batch_compare.py` 做 CPU 数值比较。artifact=`results/gdn_real_batch_artifact.pt`（88 MiB），kernel 输出=`results/gdn_real_batch_kernel.pt`，比较=`results/gdn_real_batch_compare.json`。
- 真实张量形状为 q/k/v `[16,1,48,128]`、g/beta `[16,1,48]`、HF state `[16,48,128,128]`。同一输入下 kernel 输出与 HF reference 不一致：output max abs error=`2.3125`、state max abs error=`24.2083`、output argmax 不一致；kernel 输出均 finite。跨环境 kernel 调用含 state transpose/contiguous，20 次中位数约`0.569 ms`，但正确性未通过，不能将其视为净收益。
- 当前 artifact 保存了 HF 下一 token logits，但 kernel replay 仅替换 GDN recurrence 输出，未把候选输出重新注入完整模型后续 norm/projection/layers，因此尚无候选模型 logits/top-1 对照。也不能用 GDN output tensor 的 argmax 冒充模型 logits top-1。需先解释并修复真实输入下的 output/state 语义差异，再做候选输出注入后的 logits/top-1 与完整调用重复计时。
- E107 暂停在语义对齐门槛：不接入 Agent runner、不做固定轨迹或 128 题验收。此结果是 kernel 当前参数/layout 适配与 HF 不匹配的实证，不足以断言所有迁移实现均无效。E100 cap12 继续作为批处理参考，单独归因为任务预算调整，不作为底层优化收益。

## E108 MegaGDN PTO prefill capability probe（单请求通过，E100 验收未完成）

- 当前 `vllm-ascend-py310` 实际已安装 editable `megagdn-pto`（源码 `/tmp/megagdn-pto`），通过 `VLLM_PTO_PATCH_DIR=/tmp/megagdn-pto/vllm_patch` 注入；该 checkout 使用 `VLLM_PTO_MEGAKERNEL=1`，不是 `VLLM_ASCEND_PTO_CHUNK_GDN`。首次真实运行暴露 wrapper 向 `run_mega_kernel()` 传入已删除的 `stream` 参数；做了最小兼容修补并保留失败日志 `results/megagdn_pto_mega_probe.log`。
- 在本地完整 `model/qwen3.5-27b`、单 NPU、vLLM `0.18.0`、eager、同一 128-token prompt、warmup=1/repeat=2 下，MegaGDN `pto_mega` median TTFT=`335.27 ms`（mean=`335.27 ms`），默认 Triton median TTFT=`399.43 ms`（mean=`399.43 ms`），局部 TTFT 减少约 `16.1%`。结果分别保存于 `results/megagdn_pto_mega_probe_rerun.jsonl` 和 `results/megagdn_triton_probe.jsonl`。
- 两次 probe 都成功加载完整 27B 并生成 1 token；这是单请求、短 prompt 的 prefill capability/局部性能证据，不包含 E100 首批 16 题、concise 协议、固定 greedy token 轨迹，也没有 batch16 的真实题目输入。因此不能据此宣称端到端收益或扩展到 32/128 题。下一步仍需在兼容的 vLLM runner 中复现 E100 首轮 batch16，并同时记录 prefill 墙钟与 token IDs；若收益低于 2% 或轨迹改变则淘汰。

## E109 MegaGDN E100 首批 batch16 首 token 对照（淘汰）

- 新增 `scripts/megagdn_prefill_batch16.py`，复用 E100 首批固定顺序 `[640,1137,214,1030,950,428,866,175,1130,445,1161,611,108,1288,554,249]`、concise system/tool schema 和本地完整 Qwen3.5-27B；warmup 后只生成 1 个 greedy token，不执行工具轮，记录 batch `generate()` 墙钟和每题 token ID。结果保存于 `results/megagdn_triton_e100_batch16_prefill1.json` 与 `results/megagdn_pto_mega_e100_batch16_prefill1_spawn.json`。
- 同一 vLLM-Ascend 0.18.0/eager/单 NPU 条件下，默认 Triton 墙钟 `6.6978 s`，MegaGDN `5.8485 s`，局部 batch16 首轮调用约快 `14.5%`。但 16/16 任务首 token 轨迹均改变：Triton 为 token `248058`，MegaGDN 为 token `0`。MegaGDN 真实 batch16 prefill 输出语义未对齐，速度收益不可采用。
- 按停止条件（任一 task token 轨迹改变即淘汰）停止该候选：不扩展 32/128 题、不接入 Agent runner。该结果证明当前 PTO wrapper/kernel 适配不能用于这组 E100 prompts，不能推广为所有 GDN 实现无效。wrapper 契约为 packed sequence、要求 `cu_seqlens`、零/无 initial state，内部 q/k/v/beta 转 FP16、g 转 FP32、C=128 单 fused launch；具体 stride、`cu_seqlens` 数值、输出/state 张量差异及 candidate logits 均未保存。E109 的计时是在 vLLM Triton 与 vLLM PTO 的 matched runtime 首轮生成对照，不是 E100 HF runner 的 model/e2e A/B；candidate model-only 时间、128题 makespan 和 peak memory 都未测，日志中的 0.90 GiB 是可用 KV cache 预算，不是峰值显存。case 004 已明确这些证据边界。E100 继续保持当前自有参考。

## E110 GDN `chunk_cumsum` 单阶段探针（设备占用，未执行）

- 用户报告显存问题已解决后复测：`/tmp/megagdn-pto/megagdn_pto/kernel_libs.py` 已恢复（checkout 含既有本地修改，未触碰），但 `npu-smi` 仍报 DCMI `ret=-8005`；`torch.npu.is_available()` 为 false 且 `device_count()` 为0，runtime 报 `drvRet=4` / `driver error: internal error`。NPU capture 未启动。Python3.10 CPU-only reference 命令通过：`/remote-home/S45160/miniforge3/envs/vllm-ascend-py310/bin/python scripts/gdn_cumsum_probe.py --cpu-self-test`，Python/HF chunk reference 与 packed row boundary 均 exact。probe 现先检查 PTO source，再导入 NPU/vLLM；当前没有 kernel/model/e2e timing、NPU peak memory、真实 E100 logits 或 token 轨迹，需先恢复容器内 driver/DCMI。

- 本轮只研究 GDN prefill 的 `chunk_cumsum`，不改 decode recurrent、scheduler、prompt 协议或其它五个阶段。契约：HF 输入 `g` 为 `[B,T,H]`、原始 BF16 gate logits在模型内先转 FP32，输出同 shape FP32，按每个序列每个64-token chunk独立沿 T 做 inclusive cumsum；无 recurrent state。PTO packed 适配将输入视为 `[1,B*T,H]`，以 contiguous int32 `cu_seqlens=[0,T,...,B*T]` 定义序列边界，输出仍为 FP32 `[1,B*T,H]`，不应跨序列累加。
- 收益预算：这是 GDN prefill 的一个 Vec-only阶段；候选必须同时计 kernel+launch、wrapper、pack/layout、workspace 和 peak allocation。只有在 E100 16题首轮固定 token 全部一致后才可扩展。即使把当前阶段在所有48层、16个prefill调用、每序列4096 token的测量时间全部视为可消除工作，才可作为2%端到端上限信号；实际 E100 prompt 更短，真实上限会更低。
- 新增 `scripts/gdn_cumsum_capture.py`：复用 E100 首批 task IDs、concise protocol、SDPA、BF16、fused RMSNorm/Gated RMSNorm，在真实 HF GDN prefill 调用点保存 `g`、shape/stride/dtype、prompt lengths 和首 token。新增 `scripts/gdn_cumsum_probe.py`：比较 Python/FP32 reference、HF chunk reference、当前 vLLM/NPU `chunk_local_cumsum` 与 PTO C=64 kernel，分层记录 device/host stage、wrapper、peak delta 和理论上限；任一 output mismatch 在计时和扩展前停止。
- 预定命令：capture 使用 `source scripts/env.sh && python scripts/gdn_cumsum_capture.py --output results/gdn_cumsum_e100_batch16.pt`（`ascend-e2e` Python3.12/Transformers）；CPU reference 可独立运行 `source scripts/env.sh && python scripts/gdn_cumsum_probe.py --cpu-self-test`；NPU/PTO probe 必须使用兼容 vLLM Python3.10 环境：`source /usr/local/Ascend/ascend-toolkit/set_env.sh && export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH:-} && /remote-home/S45160/miniforge3/envs/vllm-ascend-py310/bin/python scripts/gdn_cumsum_probe.py --artifact results/gdn_cumsum_e100_batch16.pt --output results/gdn_cumsum_probe.json`。本次 `npu-smi info` 显示两张910B3各约61.8GiB/64GiB已用且AICore约30%，虽然没有可见进程ID，无法证明独占设备，因此未启动27B加载或kernel编译。候选状态为 blocked_environment，不能据此宣称性能或正确性。
- 复查时设备状态未变化；host `ps` 没有可归属的 Python/vLLM/Ascend 进程，且该设备报告“不支持查询 proc info”，因此无法安全释放或归因这约61.8GiB分配。已在 `ascend-e2e` CPU路径用随机 `B=3,T=130,H=5,C=64` 验证 Python chunk-local cumsum 与 padded HF 语义逐元素相等，并验证 packed `[1,B*T,H]` 按 `cu_seqlens=[0,130,260,390]` 分段不会跨序列累加；这只证明参考公式，不替代NPU kernel或真实E100结果。继续条件仍是获得独占且有足够空闲HBM后执行两条预定命令。
- 第三次连续轮询仍显示两卡约61.8GiB/64GiB占用，AICore非零，且无法从主机或 `npu-smi` 取得归属进程；因此本候选在当前外部设备状态下阻断。恢复条件：设备重置或外部作业释放分配后，重新确认独占状态，再运行 capture → reference/NPU/PTO cumsum 分层验证；在此之前不扩大到32/128题。
- 2026-10-22恢复检查时，设备遥测状态发生变化：`npu-smi info` 在重新 source toolkit 和 driver lib 后仍以 exit 187 报 `dcmi module initialize failed. ret is -8005`，并有 `DrvMngGetConsoleLogLevel failed (ret=4)`；host 无推理进程，容器仅暴露 `/dev/davinci6`、`/dev/davinci7`。因此当前无法读取HBM/利用率或证明独占，未启动 capture。case 007 更新为 telemetry-unavailable 状态；待DCMI恢复并能确认可用HBM后继续原候选，不将此前61.8GiB读数当作本次状态。

## E111 GDN `chunk_cumsum` PTO 单阶段复核（收益上限不足，淘汰）

- 设备恢复后执行真实 E100 首批 capture：`g` shape `[16,459,48]`、stride `[22032,48,1]`、FP32，prompt lengths `[459,410,386,436,380,396,373,389,405,395,396,397,404,380,378,390]`，HF baseline 首 token 16/16 为 `248058`。artifact=`results/gdn_cumsum_e100_batch16.pt`。
- 命令：`source scripts/env.sh && python scripts/gdn_cumsum_capture.py --output results/gdn_cumsum_e100_batch16.pt`；`source /usr/local/Ascend/ascend-toolkit/set_env.sh && export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH:-} && export PYTHONPATH=/tmp/megagdn-pto:${PYTHONPATH:-} && /remote-home/S45160/miniforge3/envs/vllm-ascend-py310/bin/python scripts/gdn_cumsum_probe.py --artifact results/gdn_cumsum_e100_batch16.pt --output results/gdn_cumsum_probe.json --repeats 12`。
- Python reference与HF padded C=64 reference exact。当前 vLLM/NPU实现和PTO C=64候选均finite，二者逐元素exact；相对FP32 reference最大误差均`1.5259e-4`（以`2e-4`容差解释为NPU累加舍入）。当前 device median `0.9223 ms`，PTO kernel+launch `0.3016 ms`，完整wrapper `0.3580 ms`，PTO peak delta `1,410,560 bytes`，cu metadata 68 bytes。
- 即使采用极宽松、不可实现的上限假设（所有48层、16次prefill、每序列4096 token全部执行并完全消除该阶段），基线 cumsum累计仅`4.4546 s`，占E100 `265.030128 s`的`1.6808%`，低于至少2%门槛；真实E100 prompt更短且wrapper/其它阶段仍保留。按收益预算停止，不做固定 logits注入、32题或128题。case 008记录为 rejected；E100不变。

## E112 GDN `scaled_dot_kkt` 单阶段候选准备

- 只替换 chunk size C=64 的 `scaled_dot_kkt`，decode、其它 GDN stages、模型、prompt、scheduler 均不变。输入为实际 Qwen3.5 prefill 的 normalized `k:[B=16,T,H=48,D=128]` BF16、`beta:[16,T,48]` BF16、chunk-local `g_sum:[16,T,48]` FP32；输出 A `[16,T,48,64]`，lower-triangular rows with `j<i`。无 recurrent state；dense HF batch 中序列边界独立，PTO packed `[1,16*T,...]` 使用 `cu_seqlens=[0,T,...,16*T]` 并在每条序列重置 chunk。
- 候选是恢复 checkout 中 PTO `run_scaled_dot_kkt(..., chunk_size=64)` 替换当前 vLLM Triton KKT 单阶段。由于 HF eager GDN 的内部 WY 因子化表达与 vLLM optimized KKT 中间表示不保证相同，单阶段 acceptance contract 是 Python 公式对照、candidate-vs-current vLLM stage output exact/finite；若要整层/模型接入，仍必须完整 HF logits/greedy top-1 和 E100 首 token 对照。case 003/004 的语义/轨迹门槛可迁移；case 008 的Amdahl ceiling gate可迁移；case 001/002仅支持“融合需模型级验收”，其 normalization/SwiGLU机制不迁移。
- 收益预算先测真实首批 shape 与每序列4096-token shape 的 current/PTO stage device time；上限按`baseline_stage_ms * 48层 * 16次prefill / 265.030128s`折算，只有严格大于2%才继续。workspace/output按实测 peak delta 记录；FP16 PTO A 对 FP32 current A 有舍入风险，停止条件为参考误差不可解释、candidate/current不一致、token变化、OOM/设备异常或 ceiling≤2%。验证入口计划为 `scripts/gdn_scaled_dot_kkt_probe.py`，复用 E100 artifact 和分层 stage timers；尚未运行。
- E112 probe environment follow-up (2026-10-22): syntax check passed in the documented `ascend-e2e` environment. The PTO vendored reference is `/tmp/megagdn-pto/kernels/triton_baseline/fla_vendor/chunk_scaled_dot_kkt.py`; its signature uses `g_cumsum`. The probe accepts both `--artifact` and `--input`. Running it against `results/gdn_kkt_e100_batch16.pt` stops before measurement because that vendored file imports `vllm.triton_utils`, which is absent from `ascend-e2e`. The available `vllm-ascend-py310` environment is a separate Python/torch stack and was not mixed into this run. No FP32/current/PTO errors, peak delta, or 4096-token ceiling are reported until the reference can load. This is a setup limitation, separate from any installation-kernel UB claim.
- E112 independent-reference follow-up (2026-10-22): the installed `fla.ops.common.backends.triton_ascend` source was not runnable in `ascend-e2e` because that environment has CPU PyTorch and no active Triton NPU driver. A self-contained port is in `scripts/gdn_scaled_dot_kkt_reference.py`; it has no vLLM/FLA runtime imports and passed a real 910B3 NPU smoke in `vllm-ascend-py310`. The full E100 probe was attempted there with the saved artifact. The first gate exposed the source's `exp2` versus Qwen's natural-exponential gate convention (max error 0.133), so timings were correctly withheld; after correcting the port to safe natural `exp`, full-shape Triton compilation remained active for several minutes and was stopped. No accepted FP32/reference/PTO timing, peak delta, or 4096-token ceiling is claimed. This remains separate from the archived installation failure in `results/vllm_kkt_install_ub_failure.log`.
- E112 completed NPU run (2026-10-22, `vllm-ascend-py310`, saved artifact): standalone Ascend reference versus Python FP32 max_abs_error `4.768e-7`; PTO versus reference max_abs_error `5.227e-4`; both pass the `2e-3` gate and all outputs are finite. Reference device time for `[16,459,48,128]` was `147.257 ms`; PTO direct kernel `0.575 ms`, full wrapper `1.053 ms`; PTO peak delta `139,136,000` bytes. The 4096-token screen was run at `B=1` (`73.989 ms` reference), then conservatively scaled across 48 layers and 16 prefill calls: `56.824 s`, `21.44%` of the 265.03 s E100 baseline. This is an upper-bound screening signal, not a model result; no logits/token replay was run. The candidate remains pending model-level acceptance and a direct installed-vLLM kernel comparison.
- E112 model gate follow-up (2026-10-22): baseline `triton` completed the fixed E100 batch in `6.651 s` and produced token `248058` for all 16 rows (`results/gdn_model_gate_triton.json`). The PTO staged drop-in was initially attempted with C=64, but the existing `chunk_h.cpp` is template-specialized for `GDN_C=128` and failed compilation; the patch was restored to C=128. With C=128, the model run completed in `8.289 s` but produced token `0` for all 16 rows (`results/gdn_model_gate_pto_staged.json`), so the model gate is rejected. The stage-level KKT result does not transfer to the full model because the current drop-in replaces the entire staged GDN pipeline and its output semantics are not yet equivalent. Do not enable this model patch.
- E112 final model-gate disposition: the PTO staged patch is **not usable for acceptance**. The trusted baseline produced `248058` for 16/16 rows in `6.651 s`; the staged replacement produced `0` for 16/16 rows in `8.289 s`. This is a structural semantic failure, not a precision or environment-noise issue. The patch replaces the whole GDN pipeline, while `chunk_h.cpp` is template-specialized for `C=128` and incompatible with the vLLM C=64 configuration. The staged patch must remain disabled. The next experiment is KKT-only integration: keep vLLM cumsum, solve-tril, WY, chunk-H, and chunk-O unchanged, and replace only the scaled-dot-KKT call after verifying its `g_cumsum` contract and C=64 support.
- E112 KKT-only valid worker run (2026-10-22): original Ascend `chunk.py` computes chunk-local natural-log gate cumsum before calling `chunk_scaled_dot_kkt_fwd(k=k,beta=beta,g_cumsum=g,...)`, C=64. The adapter alone converts key/beta to FP16, transposes gate/beta metadata, uses `cu_seqlens.numel()-1` sequences, allocates FP16 PTO output and casts it back to requested FP32. All other vLLM GDN stages remain original. Parent and EngineCore logs confirm KKT-only activation. Result: **16/16 first tokens match 248058**, `7.120497 s` versus baseline `6.651192 s` (7.05% slower, single timed run). This passes only the first-token gate, not logits/decode or end-to-end performance acceptance. Results: `results/gdn_model_gate_pto_kkt_only.json`, full worker log, and `results/gdn_model_gate_kkt_only_comparison.json`. Repo-owned adapter/hook: `scripts/pto_kkt_patch/`. Earlier KKT-only-labelled all-zero output was invalid because the worker hook still installed the rejected staged pipeline; it is superseded by this explicitly routed worker run. The staged patch remains unusable; all-zero tokens alone do not isolate its precise faulty stage.

## E113 E100 decode-only profile and one candidate budget

- Profiled the actual HF E100 configuration (BF16, SDPA, sorted first batch16, fused RMSNorm/gated RMSNorm, single-step recurrent), not the vLLM KKT experiment. Skipped prefill and first decode using profiler schedule; collected steps 2–5 only. Artifacts: `results/profile_e100_decode_only/`, log, and `results/e100_decode_profile_summary.json`.
- Four decode steps: device computing union 493.910 ms, stage window 710.295 ms. Kernel-family sums: Matmul 237.046 ms/1996 launches; convolution 116.231 ms/1344 launches; Mul 49.617 ms/2632 launches; copy 22.529 ms; ReduceSum 14.147 ms. Convolution is 23.53% of computing, 16.36% of the stage window. Sums may overlap; profiler wall and free gaps are not E100 speedup estimates.
- Select exactly one candidate: T=1 causal depthwise convolution update, leaving prefill, recurrent arithmetic, and cache layout unchanged. Source is HF `torch_causal_conv1d_update`: concat cached state/new input, copy updated state, grouped conv1d, last-token slice and SiLU. E066 only rejected the unavailable fused API; do not repeat that API. Local target: halve measured 29.058 ms convolution kernel-family time per batch16 decode; conditional profiled-stage gain about 8.18%, not measured model gain. Reconstructed actual E100 generation/decode counts and 2% threshold are in the summary JSON; batch-size differences and profiling overhead limit extrapolation.
- Acceptance remains local real-input output/cache correctness, then model logits and multi-step greedy trajectory, then fixed workload quality audit and unprofiled makespan. Do not enable a candidate on profile evidence alone. Implementation has not started; this experiment selects and budgets the next candidate only.

## E114 T=1 depthwise causal convolution candidate（已结束：两版均不接纳）

- Implemented `scripts/causal_conv_single_step.py`: T=1 only, concat/read original cache layout, FP32 weighted reduction, cast at convolution boundary before SiLU, identical cache update. Prefill delegates to HF. Screening budget is >=4.214 ms net per batch16 decode across 48 layers to target 2% of E100; no gain assumed from profile alone.
- Local synthetic B1/B16 bias/no-bias tests: output and cache bit equal in all four cases. B16 no-bias wrapper median 0.778→0.335 ms per layer; bias 0.844→0.404 ms. Random inputs do not replace real-input validation.
- E100 first batch 16-token model gate: greedy trajectories 16/16 equal, finite logits; logits max/mean absolute difference 0.4140625/0.00493175. Follow-up checked 720 real-input layer updates: cache 720/720 exact; outputs 662/720 exact, maximum local output difference 0.0009765625. Thus this is not bit-exact. Full tool-round smoke is required before workload expansion; candidate is opt-in and not accepted.
- Artifacts: `results/causal_conv_single_step_probe.json`, `results/causal_conv_model_gate.json`, `results/causal_conv_model_gate_checked.json`. Model gate runtimes include cold/warm differences and are not throughput claims.
- E114 disposition: direct FP32 reduction changed task 428's full greedy trajectory in the fixed first-batch16 smoke (15/16 quality matched). Reject this arithmetic variant; the initial 16-token gate was insufficient. Its evidence is preserved as `causal_conv_reduction_probe.json` and `causal_conv_reduction_model_gate_checked.json`.
- Conservative variant keeps original F.conv1d, restricts T=1 input to last K samples, and keeps original state update. Synthetic local outputs/cache exact; 720/720 real model layer calls exact; full model first16-token logits bit-exact. Fixed matched 16-task complete tool-round runs audited without errors; all tokens/tool outputs equal and quality 15/16 on both sides. Baseline makespan 81.145922 s; candidate 82.124225 s (1.21% slower, single pair), equal peak 62,094,008,832 bytes. B16 local wrapper 0.784291→0.780130 ms/layer gives only ~0.200 ms/48-layer step, far below 4.214 ms budget. No measured performance benefit: reject deployment and do not expand to full128. E100 unchanged. Evidence: `results/causal_conv_window_model_gate_checked.json`, `results/causal_conv_window_matched_comparison.json`, `results/causal_conv_matched_baseline16.jsonl`, `results/causal_conv_window_smoke16.jsonl`. The checked model gate includes shadow-reference overhead and is used only for correctness, never performance.

## E115 Matmul / recurrent state / dispatch attribution（已完成诊断，未启用优化）

- Added opt-in `--attribute-decode` module labels to the E100 short profiler; `scripts/analyze_decode_attribution.py` joins marked CPU regions to NPU kernels using exact `torch_to_npu` flows. Four decode steps: MLP matmul 157.407 ms/768 kernels, projections 71.687 ms/1216, LM head separately 8.448 ms/4; recurrent has zero matmul. Remaining unlabelled kernels are retained as other rather than misassigned.
- Recurrent device total 76.751 ms (Mul 43.200, ReduceSum 14.776, copies 9.113, Add 5.984 ms). State [16,48,128,128] FP32 =48MiB/layer; eager source accounting about12 state-sized passes, 27GiB logical traffic per48-layer step. This is not measured HBM traffic/bandwidth; copies include dtype/layout movement, not all cache writes.
- Recurrent host inclusive markers 201.993 ms; nested ACLNN dispatch sum about29.86 ms. Flow delay contains queue/backlog and is not pure launch latency. Marked device union495.062ms/window952.157ms vs E113493.910/710.295ms: instrumentation substantially inflates host window, so gaps cannot become an Amdahl claim.
- Conclusion: MLP dominates matrix work; recurrent cost lies in state/intermediate elementwise/reduction and dispatch, not recurrent matmul. Next isolate a concrete real-state recurrent segment and check whether reducing tensor passes also reduces full wrapper time; consult failed inplace/addcmul/AOT/graph cases. Do not claim memory bandwidth saturation or a scheduling speedup without new evidence. Report: `results/e100_decode_attribution_report.md`; raw labelled trace and `results/e100_decode_attribution.json` retain attribution and limitations.

## E116 Real recurrent state segment: two-kernel fusion（不接纳）

- Purpose remains reducing E100 fixed128 makespan at unchanged quality. Before new work, reviewed E070/E078 inplace/addcmul (only1–2% local gains), E072 BMM contraction (small gains/numerical changes), BMMV2/AOT/graph failures. Do not repeat those unchanged. New mechanism: Triton memory kernel fuses FP32 state decay and key contraction; update kernel recomputes decay, writes new state and contracts query, omitting state-sized product/outer temporaries. Input cache is not modified; no model patch was installed. Floating-point fusion disabled; reduction order still differs.
- Input evidence: `results/gdn_real_batch_artifact.pt`, actual HF repeated-prompt decode B16, H48,K128,V128, q/k/v/beta BF16, gate FP32; original captured state BF16 is converted to FP32 as current single-step working state. This is not an E100 capture, not a steady-state multi-step FP32 cache validation. Complete wrapper includes normalization, contiguous/casts, gate exp, allocations and output cast. Both sides use same inputs/preparation in NPU Triton-capable `vllm-ascend-py310`; no cross-runtime performance conclusion.
- Tile8 state segment baseline0.398ms vs fused1.988ms; full wrapper0.888ms vs2.558ms. One bounded tile32 follow-up: segment0.399ms vs0.681ms; full wrapper0.882ms vs1.186ms (about34.6% slower). Both output max error3.052e-5 and state max error3.815e-6; input working state unchanged. Values are warmup then10 synchronized host-wall samples, median; not model/e2e timing. Performance instrumentation deliberately includes full allocation/dispatch/synchronization cost, not just kernel launch.
- Conclusion: eliminating logical intermediates does not guarantee faster physical execution; this two-kernel implementation fails the wrapper performance gate and is not exact. Do not expand to model/logits/128 tasks or deploy. This is a rejection of this kernel/tile implementation, not a proof that recurrent fusion cannot work; future attempts require a materially different tiling/reduction mechanism and budget. Existing E100 unchanged. Script `scripts/recurrent_fused_state_probe.py` supports explicit tile/output; artifacts `results/recurrent_fused_state_tile8.json`, `results/recurrent_fused_state_tile32.json` and logs.

## E117 MLP / runtime 机制区分（诊断，不启用优化）

- Before implementation, reviewed E089 weight-only latency rejection, E087 packing, E096 admission OOM, E098/E103 graph capture failures. New diagnostic uses real checkpoint MLP gate/up/down layers8–11 with synthetic BF16 activations, M1/16/64; compares serial-sync, queued same weights, queued rotating weights. Raw host/event/submission samples in `results/mlp_runtime_diagnostic.json`; no quality or E100 speedup claim.
- M16 full MLP host0.5483ms serial vs0.4910ms queued vs0.4927ms rotating. M1/16/64 queued0.4729/0.4910/0.5347ms: small-M FLOPs alone do not explain latency. Individual gate/up repeat-hot ~0.080ms vs rotating~0.173ms, but complete MLP does not retain this difference. This supports a cache/residency concern in isolated microbenchmarks, not measured HBM saturation. Do not sum hot isolated GEMMs or revive INT8/packing without a new mechanism.
- Unlabelled E113 trace flow association: next-host-start-after-prior-device-end gaps208.005ms vs before8.622ms. Dispatch origin is not command-ready time; association does not prove removable launch cost. E100 has no per-layer sync, so artificial serial/queued difference is not deployable gain. Report and conditional experiment decisions: `results/mlp_runtime_experiment_plan.md`; analyzer `scripts/analyze_runtime_gaps.py`.
- Next priority is fixed-shape/teacher-forced E100 runtime host-sensitivity test (interleaved threads8/1/4 and8 control, unprofiled wall/events plus logits); inspect prior phase1 conditions before interpreting. MLP real-activation/layout/residency audit second. Graph only after evidence and exact-segment eligibility checks; no blind full-model capture or new quantization implementation. End-to-end goal and acceptance gates remain unchanged.

## E118 E100 fixed-trajectory host thread sensitivity（诊断未定，控制组漂移）

- `scripts/e100_thread_sensitivity.py` uses actual E100 first-batch16 prompts, original HF generate preparation/cache, BF16 SDPA/fused norms/single-step recurrent. A logits processor forces first12 saved E100 tokens; collect original forward logits before forcing. Fresh cache every condition; 11 decode forwards. Initial nine-trial run retained separately; expanded17-trial run interleaves8/1/8/4/8 and reverse order controls. No candidate model/kernel implementation enabled.
- All17 trials: logits bit-exact to first8-thread reference, finite, identical argmax and forced tokens. Timing records host generate wall, decode-span wall and NPU forward events; score CPU transfer occurs after measured completion. No profiler trace collection in these windows, but hooks/events/forced processor overhead remains identical across conditions.
- Analysis: four paired conditions each for threads1/4. Threads1 savings versus mean adjacent8 controls range -3.35% to+2.82%; threads4 -5.97% to+5.77%. Every pair's adjacent8 controls drift >2% (2.37–16.34%), hence zero stable pairs under the diagnostic threshold. Initial run showed large temporal shift (~1.91s to1.52s for11decode calls), so repeated rather than interpreting first apparent wins. No stable benefit demonstrated; retain threads8.
- Host decode span minus forward-event sum only8.40–13.40ms across11calls, suggesting small generation-between-forward overhead in this replay. Events include host submission gaps inside forward; they cannot isolate kernel-only execution or establish device-bound behavior. Drift cause remains unproven (do not attribute to DVFS/CPU scheduling without telemetry). Thus this does not support graph/runtime implementation yet and does not rule out all host-launch opportunities. Next diagnostic should establish timing stability/telemetry or correlate per-step host/device work under a stable condition, not tune threads until a winner appears.
- Artifacts: `results/e100_thread_sensitivity_initial.json`, `results/e100_thread_sensitivity.json`, `results/e100_thread_sensitivity_analysis.json`, initial/confirmation logs; `scripts/analyze_thread_sensitivity.py` retains all pairs and marks instability rather than deleting negative samples. Goal of causal bottleneck discrimination remains unresolved. E100 default unchanged; no quality or end-to-end speedup claim.
- E118 supplemental completion: fixed-replay four-step profiles8→1→4→8, all17,416 lowered kernels and output tokens identical. Computing union494.270/494.450/494.896/495.586ms (range0.27%); stage712.701/731.701/696.962/742.605ms; free218.431/237.251/202.066/247.019ms. Repeated8 control changes mainly in gaps, not kernel time. This distinguishes stable device work from variable gaps in these profiles, but does not isolate host submission vs dependencies/driver scheduling or prove a removable-launch budget. Separate-profile observations cannot causally label all unprofiled drift. CPU affinity recorded, cpufreq unavailable in container; no DVFS attribution.
- E118 decision: requested thread-sensitivity diagnostic completed with bounded conclusion: logits/trajectory exact, no robust benefit from1/4, E100 threads8 retained. No graph/runtime implementation justified. Further runtime experiment requires a specific gap mechanism/eligible segment, not blind capture retries. Final evidence and limitations: `results/e118_thread_sensitivity_report.md`, `results/e118_thread_kernel_cost.json`, four raw profiles and unprofiled analysis. Broader end-to-end optimization remains unfinished.

## E119 具体间隙机制定位（不做盲目graph）

- Reused E118 four fixed-replay traces; exact host/runtime/device flows partition idle. Idle202–247ms/foursteps: before next host flow174.6–202.1ms; hostflow→runtime marker25.4–51.5ms; after marker2.2–2.4ms. These are clipped temporal intervals, not removable-cost proof. In8a, disjoint idle host coverage includes empty_tensor24.33ms, RMSNorm custom op23.81ms, matmul8.33ms, copy8.17ms, as_strided7.08ms, cat6.71ms, unrecorded68.56ms. Main mechanism location is host producer preparation/dispatch; unrecorded Python/scheduling/profiler components not separated.
- Corrected queue hypothesis: largest gap1.508ms overlaps long enqueue1.989ms, but only1/14,884 enqueue spans>100us. Does not establish persistent backpressure or justify queue tuning.
- Source-level scalar interception proves per-decode SDPA padding_mask.all() Python bool and generation this_peer_finished Python bool. Fourstep8 is_nonzero, inclusive2.343ms;1248 local_scalar calls are NOT all device readbacks. Fixed-left-padding mask-skip bypass keeps logits/tokens exact on five replay runs but one candidate slower0.82%, another faster6.94%; no stable performance gain. Diagnostic-only, not general mask optimization or model acceptance. It does not identify E098/E103 recurrent capture failures' exact source.
- Conclusion: specific host prep/dispatch stages and recurring bool synchronization located; much idle remains non-causally attributed. Do not count200+ms as graph savings; do not claim all host/runtime costs removable. No E100 change. Report `results/e119_gap_mechanism_report.md`; scalar stacks, timestamp partition/host coverage JSONs, mask-check A/B and analyzers preserved. Further implementation requires a stable general mechanism and complete-wrapper budget.

## DeepInfer 官方实现参考（2026-10-22；只读审查）

- Official main commit9bd9cf243fc53ef9afbf8f04d391d989117d6ab9 supports Ascend950, but model listed only DeepSeek-V4.1-Flash, CANN9.2, torch/torch_npu2.13; not our910B3/CANN8.5.1/Qwen3.5. README benchmarks are SM100 multiGPU, not950 or singleNPU. No dependencies installed/default flags changed.
- Verified transferable mechanisms: graph runner preallocated metadata/output and bucket views/memory-pool reuse; worker current-forward overlaps previous-postprocess using events/state; Linear BF16 GEMM explicit output/alignment buffers; Ascend device-side batched metadata. V4 MoE/FP8 scale layouts are not drop-in denseMLP implementations. Platform TASK_QUEUE_ENABLE=0 is its configuration, not proof of HF speedup or persistent queue backpressure.
- Most grounded next candidate experiment is complete BF16 Qwen MLP real-activation buffer/layout boundary comparison (allocation vs reuse, identical arithmetic), measuring all dispatch/copy costs and exact outputs before graph/quantization. Profitability remains unknown; asynchronous scheduler migration lower priority given E118 between-forward cost. DSpark/MTP excluded by project constraints; V4 SWA/EP/EPLB/CP and NVIDIA-only batch invariance not transferred. Detailed source review/conditions: `results/deepinfer_reference_review.md`.

## E120 DeepInfer-inspired complete BF16 MLP buffer/layout probe（不接纳）

- Captured actual E100 firstbatch16 decode activations at layers8/55, three calls each, BF16[16,1,5120] stride[5120,5120,1]. Loaded same real checkpoint gate/up/down weights. Kept final implementation's gate→SiLU→up→multiply→down order and BF16 boundaries. Explicit mm/silu/mul output buffers; separate clone variant includes output ownership cost. No precision change, fusion or weight packing. Persistent2,392,064bytes/layer; no model/default patch.
- Six real outputs finite/bit-exact and output stride identical. Timings:4warmup,9windows×8calls; interleaved baseline/reuse/baseline/clone/baseline, serial and queued. Final queued original~0.486ms vs reuse0.490–0.493ms, no gain; clone also slower. Serial likewise no improvement. Raw event/host samples retained, final ordered profiles separate from initial-order trial.
- Separate4call profiles verify baseline20allocation events→reuse0, clone4; each12GEMM+4SiLU+4Mul, no contiguous/format_cast. Device kernel sum remains~1.87–1.90ms/fourcalls. Hence eliminating allocation-event count did not reduce complete-wrapper cost. F.linear/aclnnMatmul versus mm(out=)/aclnnMm is an API difference, not pure allocator causal proof. Reject this specific buffering implementation; no model/128 expansion and no DeepInfer speedup inference. Original E100 unchanged.
- Evidence: `results/e100_mlp_activations.pt`, `results/mlp_buffer_probe.json`, `results/mlp_buffer_probe_analysis.json`, `results/mlp_buffer_report.md`, final `profile_mlp_buffer_ordered_*` traces. Scripts capture_e100_mlp.py/mlp_buffer_probe.py/analyze_mlp_buffer_probe.py. No actual format conversion identified to optimize; only reopen with a different measured mechanism, not blind buffer/INT8/packing retries.

## E121 bounded BF16 GEMM capability audit（找到NZ候选，未接纳）

- Reviewed installed interfaces/docs/headers on910B3/CANN8.5.1. Existing npu_linear(E083), buffer reuse(E120), grouped projection/INT8 failures not repeated. Default JIT already off. Communication-fused GEMM irrelevant singleNPU; fused_matmul proto BF16 notes910_95 and has add/mul/GELU notSiLU, so no invented fusion test. DeepInfer950 kernels not assumed compatible.
- New mechanism tested on actual E100 layers8/55 three decode activations each: persistent BF16 FRACTAL_NZ weight format29. Full original-order MLP output exact/finite6/6. Layer8 ND0.47731→NZ0.39812ms; layer55 interleaved ND0.47727/NZ0.39693/ND0.47720/NZ0.39665ms (~16.8%local gain). Trace12aclnnMatmulWeightNz kernels/fourMLPs, no runtime format_cast/contiguous. This is distinct from E120 output buffer reuse. Physical contiguous transpose has smaller gain and changes values up to0.00390625; deprioritized.
- One-time NZ preparation~15s/layer probe includes cold compiler; do not extrapolate initialization or ignore it. Full-model peak, startup, prefill/decode/logits/complete trajectory unvalidated. Local~80us/layer×64≈5.1ms/step is conditional budget only, not end-to-end gain. Next candidate may convert MLP weights once while preserving original forward, with memory/ownership and model gates before deployment. E100 unchanged.
- Audit report `results/bf16_gemm_capability_review.md`; initial/layer55JSON and raw NZ trace. Scope complete: capability audit only, not optimization acceptance.

## E122 持久化 BF16 NZ MLP 模型接入与 matched16 gate（不接纳）

- Added opt-in `--mlp-nz`; converted only the192 BF16 gate/up/down weights with `npu_format_cast(...,29)`, one weight at a time; verified each actual format=29, replaced module Parameter, released old reference, synchronized and emptied cache. No forward operation changed. Full 16-task model allocation stayed stable: before/after preparation 54,713,982,976 bytes; preparation peak54,892,241,408 (about179MiB above baseline); model load time26.50s vs11.55s. Generation peak62,099,070,464 vs62,094,008,832 (+~5MiB), no OOM.
- Fixed E100 batch16 tasks/config, matched run audits have no errors. 16/16 round token/tool-output trajectory exact, quality15/16 and total3969 tokens both. Baseline76.976515s; NZ80.804083s, 4.97% slower; single matched pair; no benefit. Local six-input exactness and ~16.8% layer microbench gain fail to transfer to full task. Preparation is outside measured generation makespan but still a service startup cost.
- Candidate rejected at model performance gate; no128 expansion/deployment; E100 unchanged. Results `mlp_nz_model_gate.json` (logits/16-token exact), `mlp_nz_smoke16.jsonl`, `mlp_nz_baseline16.jsonl`, `mlp_nz_matched_gate.json`. This specifically rejects persistent-NZ path in current PyTorch HF wrapper/910B3 conditions, not every possible specialized GEMM implementation.

## E124 BF16 NZ MLP 收益断层归因（A/B 完成，不接纳）

- 按附件方案先做 E124-A：新增 `scripts/e124_nz_e2e_ab.py` 和 `scripts/analyze_e124_nz.py`。每次新进程重新加载模型，固定 E100 首批16题、batch16、concise、SDPA、BF16、fused norms、single-step recurrent、final cap12、generation-length 排序，交替 ND→NZ→NZ→ND。两组配对 ND makespan=`79.5052/78.2747 s`，NZ=`85.3281/81.1845 s`，NZ 慢`7.324%/3.718%`，中位数`5.521%`；两组轨迹 hash 完全一致，任务均15/16正确。生成时间不含模型加载/NZ转换；ND peak=`62,094,008,832`，NZ peak=`62,099,070,464` bytes。NZ 转换约15.0–15.3s，启动 load 约28.4–28.6s，ND load约13.3s。
- E124-B 新增 `scripts/e124_nz_mlp_probe.py`，复用 `results/e100_mlp_activations.pt` 第8、55层的三个真实 `[16,1,5120]` BF16 激活，完整 gate→SiLU→up→mul→down wrapper，ND/NZ 交替各20次，记录 host wall、设备事件、输入/输出 stride 和 peak。两层全部 bit-exact；第8层 NZ host/device 中位数相对 ND分别`-0.368%/-0.177%`，第55层分别`+0.473%/+0.090%`，没有复现 E121 的16.8%局部收益。设备事件仍含排队/host间隙，不能解释成纯 kernel busy。
- 结论：E124-A 稳定复现 E122 的生成回退，E124-B 表明完整真实 MLP wrapper 中收益已消失且层间方向不一致；按停止条件不进入 E124-C，不做布局选择、trace 归因或异步调度改动。E122/E124 NZ 路线继续 rejected，E100 `265.030128 s / 122/128` 保持参考。原始数据：`results/e124_nz_e2e.jsonl`、`results/e124_nz_mlp.jsonl`；报告：`results/e124_nz_gain_gap_report.md`。

## E125 NZ E121 局部收益复现与 decode 边界归因

- E125-A/B 原样复现 E121：新增 `scripts/e125_nz_boundary_probe.py`，固定 E100 layer8/55 的真实 `[16,1,5120]` BF16 激活、三组原始权重、原始 `F.linear -> SiLU -> mul -> F.linear` 顺序、4 warmup、8 calls/window、窗口末同步，ND/NZ 交替20对。两层完整 MLP 均 bit-exact；layer8 NZ `-17.88%`，layer55 `-15.70%` host wall，三组独立 matmul 也均有收益。NZ conversion 约15.1s，不计入稳态窗口。E121 的约16.8%收益可重复，E124-B 的近零结果来自不同时间/调用路径，不能据此否定局部收益。
- E125-C 对相同 E100 首批16题、12个强制 replay token、SDPA/fused norms/single-step recurrent 做 ND/NZ 固定轨迹 profile。两次 replay token exact。ND/NZ 均17,416 kernels；ND 总列示设备时长494,439.6us，NZ510,079.5us。ND regular `aclnnMatmul` 237,218.4us；NZ 为 regular80,591.4us + `aclnnMatmulWeightNz`172,189.9us，只有768个 matmul切到NZ，仍有1,220个 regular matmul。NZ列示总时长高15,639.9us，并出现约45ms transdata路径。Profiler wall 受插桩影响，不作为端到端结论。
- 结论：局部 wrapper收益真实，但完整 decode 中仅部分 matmul切换且存在布局/数据搬运路径；这是收益首次与模型级结果分离的边界。NZ 仍不部署，因为 E124-A 未插桩完整生成稳定慢`3.718%/7.324%`。下一步若继续，只能先把 regular/NZ matmul与具体模块及 transdata 归属映射清楚，不能直接做异步或继续扩大 workload。详细报告：`results/e125_nz_gain_boundary_report.md`。

## E126-E128 Matmul 账本与布局路径核对（停止修改）

- 新增 `scripts/e126_matmul_ledger.py`、`scripts/e126_enrich_ledger.py`、`scripts/analyze_e126_ledger.py`，从 ND/NZ `trace_view.json` 和 `kernel_details.csv` 逐调用导出 `results/e126_matmul_ledger.json` 与 CSV。两边各1,992个 Matmul 调用；NZ 为768个 `aclnnMatmulWeightNz`（64层×3 MLP投影×4 decode steps），其余1,224个为 regular。MLP调用 shape 可由 checkpoint 权重和固定 batch 推导并写入账本：例如 gate/up `[M=16,N=17408,K=5120]`、down `[16,5120,17408]`，BF16，NZ format29；其余调用也保留 layer/module/shape/kernel时长。输入 runtime stride 仍未被该 profiler export记录，明确标为缺失，不猜测。
- 1,224 regular 调用均属于 attention/projection/lm_head，超出 `mlp_nz.py` 仅转换 `.mlp.{gate,up,down}_proj` 的选择器；账本不能证明它们是 shape unsupported，也不能据此盲目转换。
- E127 检查了布局/搬运生命周期：decode trace 中无重复 NZ weight format_cast；192权重在生成前转换一次并跨 step 复用。NZ `trans_TransData_5`约45,099us在每个4-step profile中48次/step，ND 对应 `aclnnConvolution_TransDataFzToDst_TransData`约44,749us，同级路径并非已证实的NZ新增45ms。ND/NZ conversion-like host enqueue均不是权重转换本身，不能从名称推断对象归属。
- E128 不改代码：当前没有找到“可消除的重复权重转换”这一单因素；NZ局部收益真实但E124完整生成仍慢`3.718%/7.324%`，且NZ/regular混合路径和同规模transdata占据完整trace。继续优化前必须新增带 runtime input stride/format 与权重 identity 的低扰动 hook；本轮不扩大 workload、不部署NZ。报告：`results/e126_matmul_ledger_report.md`，逐调用CSV为 `results/e126_matmul_ledger.csv`。

## NZ 路线封存与完整 Decode 热点排序

- NZ 正式记录为“局部有效、端到端无效”的已验证负结果。E121/E125 重复证明 MLP wrapper 和 Matmul 局部收益约15–18%，E124/E122 的完整生成却稳定变慢；正确性和 token 轨迹均通过，但没有 makespan 收益。因此停止 NZ 扩展、权重布局范围扩大、转换复用改动和进一步 profiler 归因。
- 下一轮排序见 `results/next_decode_hotspot_ranking.md`：四步设备归因中 MLP Matmul=`157.407ms/768 kernels`，GDN convolution/transdata=`116.823ms`，recurrent Mul/ReduceSum/copy=`43.200/14.776/9.113ms`，attention/GDN projections=`71.687ms/1216 kernels`，LM head=`8.448ms`。这些是互斥归因表中的局部设备证据，不把嵌套时间相加成端到端结论。
- 新候选必须从可量化假设开始：明确目标阶段、输入/状态契约、可回收时间上限、固定 replay correctness 和完整 decode gate；优先研究 GDN state/convolution 或一个有明确 wrapper 预算的 projection，而不是继续 NZ 或盲目 graph/async。E100 `265.030128s / 122/128` 仍为参考。

## E129 MLP Matmul 性能上界分析（只选下一项假设，未改模型）

- 新增 `scripts/e129_mlp_matmul_roofline.py` 与 `results/e129_mlp_roofline.json/.md`。从 E125 ND trace 通过精确 `torch_to_npu` flow 将设备 kernel 对到 MLP module，再按真实 shape/dtype 聚合；768/768 个 Matmul 均匹配，覆盖4个固定 decode steps。形状：`[M16,N5120,K17408]` down 256次，kernel40.591ms、2.852GFLOP/call、17.99TFLOP/s；`[M16,N17408,K5120]` gate/up合计512次，kernel116.242ms、2.852GFLOP/call、12.56TFLOP/s。总 kernel156.833ms，约2.190TFLOP工作量；与E115 157.407ms在不同trace间相差约0.37%。
- 用`2MNK`、BF16 2-byte流量模型得算术强度15.94FLOP/byte。硬件探针本机大buffer copy为1256.66GB/s、M512 BF16 GEMM为238.91TFLOP/s；据此经验memory roof约20.03TFLOP/s，两个shape效率分别89.8%和62.7%。若每次完整读一次权重，估算streaming floor总109.38ms，实测高47.45ms；这是理想流量预算而非已证明的实现浪费，因为trace没有实测HBM/L2字节。
- 相同`[16,17408,5120]`形状拆op后，gate 256调用核时76.549ms，中位303.14us；up 256调用39.693ms，中位155.16us，跨层约1.95x差异；down接近经验memory roof。匹配的`torch.mm`基准31.34TFLOP/s使用连续[K,N]右操作数，而模型Linear存储[N,K]并使用转置view，故只作有利参考，不能称严格布局上界。硬件属性未提供910B3官方BF16理论峰值，不能由20个Cube core推算；报告明确留空。
- 唯一下一步假设：gate/up同形状核时差可能跟随权重身份或MLP调用位置。用相同真实激活、原始gate/up权重做交替顺序交叉，记录各 Matmul event 与bit-exact输出；若慢跟随调用位置，检查先后/缓存状态；若慢跟随权重，再检查地址对齐、真实stride/格式及tile选择。按“gate达到up当前中位数”计算四步上限约37.88ms，未验证前不做算子实现改动。down_proj不作为当前候选。

## E130 gate/up 调用位置交叉实验（局部诊断，不接入）

- 新增 `scripts/e130_gate_up_crossover.py`，对 E100 捕获的 layer8/55 三组真实 BF16 `[16,1,5120]` 激活，使用原始 ND gate/up/down 权重，保持 gate→SiLU→up→down 语义，做100组同激活交叉顺序，连续提交后一次同步。所有 gate/up/MLP 输出 bit-exact，输入输出 stride一致，两个权重 NPU format=`2`、stride=`[5120,1]`、data_ptr mod512均为0。
- 连续提交下差异跟随调用位置而不是权重：layer8 gate first/second=`203/161us`、up first/second=`205/187us`；layer55 gate=`201/157us`、up=`200/175us`。完整两投影MLP wrapper 的 up→gate 相对 gate→up 中位差分别`-24.3us`和`-16.7us`，bootstrap95% CI分别`[-29.3,-21.9]us`、`[-17.9,-12.9]us`；100对中分别92/93对 up→gate更快。输出bit-exact，输入/权重形状、dtype、stride固定；weights stride `[5120,1]`，format2，data pointer mod512均为0。
- 现有完整模型trace里 layer8 gate/up MatMulV2分别`300.2/158.3us`，layer55为`277.6/154.6us`；host Linear span只差约14us，二者之间正常有SiLU dispatch，没有显式CPU同步。kernel名称相同；当前trace不能证明缓存或带宽原因，也没有提供完整operand物理layout/traffic。位置相关现象在轻量probe里成立，但全模型gap幅度仍未解释。
- 追加 `scripts/e130_same_weight_reuse.py`、`results/e130_same_weight_reuse.json`：相同真实激活下对 gate/up、gate/gate、up/up 交替，检查重复调用。重复相同权重的第二次 Matmul 从首调用约`377–389us`降到`64–66us`且输出exact；不同权重的第二次约`182–188us`。为隔离缓存，另加512MiB scratch read/write并在每对前同步（L2为192MiB）。这证明probe可观察到同权重驻留带来的明显局部复用，也显示任意首个调用比第二个调用慢；不证明实际 gate/up 的 full-model流量命中模式相同。
- E125 trace上下文：gate前最近操作为RMSNorm，up前有SiLU；CPU Linear span相近，NPU kernel名同为MatMulV2；被捕获的异步NPU任务均在stream47，未见MLP Matmul跨stream并行。Profiler没有物理cache hit/miss计数、调用级operand format，也不证明同stream队列外无资源竞争。
- 原先用孤立 gate/up 中位数相减得到的`37.88ms`不可作为可回收预算。E130真实交叉差按64层×4步粗略外推约`4.3–6.2ms`，约为相应四步stage窗口的`0.4–0.7%`，低于2%筛选门槛。当前设备仅约44GB可用，而E100完整模型需约54.7GB权重/分配且生成峰值约62.1GB，因此未做 full Decode A/B。该候选在2%门槛下停止，不改MLP实现；这轮没有形成端到端收益结论。
- E130 scripts/artifact/report: `scripts/e130_gate_up_crossover.py`、`results/e130_gate_up_crossover.json`、`results/e130_gate_up_crossover_report.md`。NZ路线仍封存。

## E123 generation-stage timing（诊断，未启用异步调度）

- 用户要求先测每步 scheduler/model forward/sampling/postprocess 的 host wall 与设备区间，再决定跨步异步调度。新增 `scripts/generation_stage_timing.py`（源码级 HF `_sample` 插桩）、`scripts/time_generation_stages.py`（E100 首批16题、交替未插桩/插桩、前32 token）和 `scripts/analyze_generation_stages.py`。契约与边界见 `results/e123_stage_timing_contract.md`；只使用原始 HF 调用、BF16、SDPA、fused RMSNorm/Gated RMSNorm、single-step recurrent，不改变协议、模型、cache 或调度。
- 三次插桩运行均与未插桩及 E100 保存的首轮 token IDs 完全一致（16/16×32），peak allocation=`59,173,664,768` bytes；设备事件在生成末尾统一读取，避免阶段间同步。插桩/相邻控制 wall 差分别`+0.91%/+0.63%/+0.20%`，相邻未插桩漂移`0.61%/0.38%/0.57%`，测量扰动低于1%。
- 32 decode steps 的 host wall 汇总中，model forward `4,078.9–4,181.6 ms`（约`96.8–96.9%`），scheduler `96.1–99.2 ms`（约`2.24–2.35%`；其中循环条件约`87.7–91.1 ms`，输入准备约`8.1 ms`），postprocess `29.4–32.4 ms`（约`0.70–0.76%`），sampling `6.7–6.9 ms`（约`0.16%`）。设备区间对应 forward `4,143.3–4,245.3 ms`，scheduler `13.4–14.1 ms`，postprocess `2.9–3.8 ms`，sampling `7.4 ms`。阶段设备区间包含队列等待/host间隙，不能解释为 kernel busy 或可直接重叠预算。
- 分布稳定但 scheduler 与 postprocess 合计约3%，低于“稳定关键路径”且依赖逐 token cache/next-token 顺序；没有证据支持跨步异步调度实现。按停止条件不改 Agent scheduler、不做 graph/capture 或 overlap patch；E100 保持`265.030128 s / 122/128`。若运行时/缓存语义改变，优先重新测完整 Agent tool-loop 的 scheduler 和 postprocess，而不是根据本诊断盲改。
