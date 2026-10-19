# 单卡 Qwen3.5-27B / GSM8K 优化实验记录

## 优化路线重构（当前执行版；更新至 2026-10-19）

### 阶段位置、目标与已知边界

优化阶段是递进关系，不是互相排斥的最终目标：Phase 1 优化/理解单请求执行路径；Phase 2 在稳定模型路径上优化多请求调度和 128 题 makespan；Phase 3 再做面向设备的通用 kernel、内存与 launch 优化。Phase 1 的 batch=1 指标用于判断基础执行路径，Phase 2 的 batch 指标衡量调度，Phase 3 的局部算子指标用于验证机制；任何 Phase 3 改动要成为项目收益，最终仍须改善 Phase 2 的端到端任务且守住质量。

当前已完成 Phase 1 的基线与主要执行路径探索，并完成 Phase 2 的离线批处理优化。E100 是当前 128 题工作流中已验收的最好配置，但其中 final-turn cap 12 是任务协议预算调整，收益有限且依赖当前工作负载，不作为底层优化的主要技术贡献。当前工作的重心应进入 Phase 3：寻找能在正确性、显存和兼容性约束下改善模型执行成本的通用优化；hybrid cache/continuous batching 是候选系统方向之一，不预设为唯一答案。

目标是缩短固定 128 题离线 GSM8K 工作流的未插桩 makespan，并保持 scorer v3 至少 `122/128`。当前最佳自有实现冻结为 E100：`265.030128 s / 122/128 / 15,961 tokens`；对照 E047 为 vLLM-Ascend `239.194 s / 122/128 / 16,629 tokens`。差距为 `25.836 s`（自有实现需缩短约 9.75% 才追平该 makespan）。两次 token 轨迹不同，E047/E100 是端到端系统结果，不是执行效率对比。超过 E047 的声明仍须在同一 128 题、未插桩运行中低于 `239.194 s`，同时通过相同评分护栏并报告 token 数、失败 ID 与归因。

E100 配置与结果作为冻结参考，不为刷新基线重跑。HF batch runner 已保存每个同步 generation call 的墙钟时间、输入准备时间和工具时间；这些时间不能直接相加成各题耗时，batch call wall time 是该调用的有效成本。E047 的归档没有 `llm.generate()` 阶段时间，只有总 makespan、逐题 token/轮次和工具时间，因此不能从现有记录推断 prefill/decode/轮间空档占比。E007 的 profiler idle、E093 的 22.8% row-step 上界也都不是可实现的时间收益。

E100 相比 E047 少生成 668 tokens，质量分相同；所以 25.836 秒端到端差距混合了 token 数、模型执行路径与调度差异。任何“执行效率更高/更低”的结论须基于相同 prompt 与输出 token 轨迹的 replay。现有 replay 资产主要验证 HF 局部路径和候选正确性，尚无完整的 HF-vs-vLLM 固定轨迹耗时对照；固定轨迹结果只能回答执行成本，不能替代 128 题端到端验收。

### Phase 3 当前路线：补齐归因，再验证底层候选

| 阶段 | 工作与产物 | 继续条件 | 停止/转向条件 |
|---|---|---|---|
| 0. 冻结 Phase 2 参考 | 固定 E100 配置、数据 hash、scorer v3 失败集合 `[1161,423,952,1035,552,1048]`、15,961 token、265.030 s；E084 cap32 保留作更宽 final-turn budget 的对照，E047 保留作系统参考。 | 只在环境/依赖变化、结果完整性问题或新候选 A/B 时复测必要对照。 | 不因 Phase 3 开始而重跑全量基线；不把 E100 cap12 当作底层 kernel 收益。 |
| 1. 建立 Phase 3 诊断基线 | 离线整理 E100/HF 已有逐批 generation、prompt 长度、有效输出 token、工具轮和已有 profile；指出 prefill/decode/launch/scheduler 哪些已知、哪些未知。只有缺失项会改变候选排序时，才做代表性插桩或给 vLLM runner 加阶段字段。 | 有证据把工作量映射到 prefill、decode、NPU launch/算子、batch tail 或 Agent 轮间成本；局部 profile 不作为收益结论。 | 不把 E007 idle、row-step 理论上界或单算子峰值直接当可实现收益；不能影响决策的 profile 不采。 |
| 2. 选择 Phase 3 候选 | 按诊断和收益预算选择一个通用算子/内存/执行机制候选。近期可核实 E106 的 GDN B16 kernel 语义能否与 HF reference 对齐；也可只读审计 hybrid-cache runtime 能力。二者均需确认当前依赖和接口，不直接扩展为主 runner。 | 候选命中关键路径，理想总收益足以改变决策，且当前环境有可执行实现。 | GDN adapter 既有 E052 慢约10.5%、发生过设备异常；runtime admission 有 E096 OOM。除非找到新的可测实现条件，不重复原形状/原 wrapper 试验。 |
| 3. 分层验证候选 | 单算子/真实模型形状 reference 对照 → 固定输入及 token replay → 16–32 题 batch smoke。算子速度必须连同转换、cache、workspace 和 launch 开销一起量。 | 数值/状态正确，轨迹风险可解释，无 OOM，重复 smoke 收益高于运行波动。 | 局部快但调用路径总成本不降、正确性失败、显存无余量或收益在噪声内则停止，不做全量。 |
| 4. 回到 Phase 2 目标验收 | 只给 smoke 胜者跑完整 128 题；未插桩、单 NPU、同数据和环境口径，运行 audit/scorer v3，报告 makespan、分数/失败 ID、token、峰值显存。 | 更新自有最佳需 `>=122/128` 且快于 E100；声称超过 E047 还需 `<239.194 s`。 | 未改善 128 题 makespan 的 Phase 3 候选可作为算子研究结果记录，但不能宣称项目端到端优化成功。 |

### 候选收益预算

任何编码前先写一行预算：要消除的真实工作、当前量、理想上限、实现成本/显存/数值风险、验证样本、停止阈值。局部算子按 Amdahl 上界折算到 265 秒总 makespan；工作量比例只标为上界信号。没有可观测时间份额、没有可执行路径或理论最大收益不足以改变决策时，不进入实现。

### 当前待办

1. 将 E100 冻结为 Phase 2 端到端参考；报告时把 cap12 标为当前任务配置收益，与底层执行优化分开。
2. 以已有 E100/E084 数据和 profile 建立 Phase 3 成本视图，避免重测已知部分；按证据选定一个可执行的通用底层候选。
3. 对选中候选执行真实形状正确性、固定轨迹和 16–32 题 smoke。当前可先审查 E106 GDN B16 结果能否接到 HF reference；若需重建兼容环境，先验证其是否可能消除 E052 adapter 的转换/调度损失。
4. hybrid-cache/continuous batching 作为系统级候选并行做能力审计，Python cache 合并不重开。任一候选只有通过完整 128 题 Phase 2 指标后才更新最佳。

### 已知停止项

E092 finished-row compaction 比参考慢约 3.4%；E096 Python cache admission OOM；E098/E103 recurrent graph capture 失败；E087 投影打包全量慢于 E084；E089 weight-only INT8 没有稳定延迟收益；静态排序复测处于噪声范围。除非依赖、硬件接口或实现前提发生变化，不再重开这些方向。

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
