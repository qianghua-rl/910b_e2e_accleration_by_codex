# 公平性审计与比较规则

本文件回答两件事：(1) 哪些已有的比较可以用、哪些不能用；(2) 以后做比较必须满足什么。

结论依据是各 `results/**/per_task.jsonl` 首行的 `metadata`（实际运行配置）以及 `results/README.md`，不是日志文件名，也不是 HANDOFF 里的口头描述。

## 1. 比较前必须一致的项

两组运行要互相比较，下列项必须全部相同。任何一项不同，结论只能作为"参考"，不能作为"收益"。

| 项 | metadata 字段 | 说明 |
|---|---|---|
| 题目集合 | `data_sha256` | 同一数据集、同一 sha 前缀 |
| 并发槽位 | `slots` / `batch_size` | 8、12、16 不可混比 |
| 生成上限 | `t_max`、`first_max_new_tokens`、`final_max_new_tokens` | 上限不同，截断不同，准确率不可比 |
| 引擎路径 | `engine`、`graph_decode`、`prefill_group`、`fast_gdn`、`elem_conv`、`gdn_v2` | 只允许"被测的那一项"不同，其余照抄 |
| 测量方式 | 同卡、顺序 ABBA 或同一时段 | 见 README §3.2，单对测量不能给出确定结论 |

只改一项：比较 A/B 时，除了被测项，其它字段必须逐个相同。

## 2. 已有比较的状态

| 比较 | 结果 | 状态 | 依据 |
|---|---|---|---|
| graph vs eager，TP=2，aime2024 / aime2025 / gpqa / bbh / math500 | 同 `slots=12`、`t_max=8192`、caps 6144/2048、同 sha；只有 `graph_decode` 不同 | **可比** | `official/continuous_batching_tp2/*` (eager) 与 `official/continuous_batching_tp2/graph_*` (graph) 的 metadata |
| graph vs eager，TP=2，gsm8k_longctx_128 | 同 `slots=12`、`t_max=16384`、caps 6144/2048 | **可比** | `official/continuous_batching_tp2/gsm8k_longctx_128_graph_*` 与 `gsm8k_longctx_128` |
| 单卡 CB vs 双卡 CB | `t_max` 4096 vs 8192，caps 2048/512 vs 6144/2048 | **不可比**（已在 results/README 说明） | metadata |
| 单卡 CB 正式结果 vs 单卡 ablation | `official/continuous_batching/` 正式结果为 `slots=12`，ablation 多为 `slots=16`，ablation 的 caps 字段为 None | **不可直接比**，需同 slots 重跑 | metadata |
| 静态 batch16 vs CB（E01 vs E07） | 代码版本不同 | **不可比** | records.jsonl E07 备注 |
| `m8_*`（`slots=8`）vs 12 slot 运行 | `slots` 不同 | **不可比**，`m8_*` 本身是为了与 vLLM 对齐的 8 slot 运行，只能和 `m8_*` 比 | metadata |
| `graph_trunc_cap_aime2024_subset` | caps 7168/4096，子集 | **不可与全量 caps 6144/2048 比** | metadata |
| vLLM-Ascend（`official/bench_vllm*`） vs 自研引擎 | vLLM 的 `per_task.jsonl` 首行没有 slots / t_max / caps 字段（`engine=AsyncLLM`） | **配置无法从文件验证**，需从运行脚本或 `logs/vllm/` 补齐后才能比较 | metadata |
| `eager_subset21`（eager，8 slots，21 题，2026-11-11 完成）vs `m8_gpqa_r1/r2`（graph，8 slots） | 相同 slots、t_max、caps、数据 sha、prefill_group、elem_conv、gdn_v2，只差 `graph_decode` | **可比**。结果：eager 7/21，graph 7/21；正确性翻转 6 题（14、46、58、78、81、173）方向相反，净差 0；两次 graph 运行 21/21 逐字一致（确定性），eager 与 graph 逐字一致仅 5/21 | `results/official/continuous_batching_tp2/eager_subset21/`、`m8_gpqa_r1/r2/` 的 metadata 与逐题比较 |
| `unfair_e100_concise_aborted` | `final_max_new_tokens=12` | **不可用**，已移至 `logs/invalid_or_aborted/` | metadata |
| `eager_subset21` 之外的 TP2 eager 基线（records E21 备注"flags 未记录"） | 已能从 metadata 确认与 graph 一致（见第一行） | 备注可以消除 | metadata |

## 3. 已知的错误结论或弱证据（records.jsonl）

- **E03**：同步次数被当作瓶颈的代理，wall 结果反向。以后"减少 X 就会更快"的假设必须直接测 wall。
- **E05**：设备时间下降没有转化为 wall 收益。device 时间不能单独作为收益依据。
- **E07**：静态与 CB 的比较不同代码版本，已标注不可比。
- **E06、E10、E13、E14 等 `evidence_quality: medium` 的记录**：单对或单次测量。要做决定，必须补 ABBA。
- **E18**：device-bound 结论后来被 wall 分解修正。

## 4. 以后做实验的规则

1. **先写假设，再跑**。假设必须是可证伪的句子，并写出预期方向和门槛（默认 wall 收益 ≥ 2%）。
2. **只改一个变量**。改动前，把 metadata 与基线逐字段对照，不一致就先修配置。
3. **决定要用 ABBA**（至少两对，A-B-B-A），并确认无其他进程占用同卡。两组区间不重叠才算确定收益。
4. **每一次运行都要在 metadata 里有 slots、t_max、caps、graph/eager 标志**。vLLM 运行也要写。缺字段的运行不进入比较表。
5. **未完成或中止的运行移到 `logs/invalid_or_aborted/` 或 `results/archive/`**，并在 `results/README.md` 注明"无效"。不能留在正式目录里。
6. **正确性用逐题对比**，不只看正确数（见 README §3.3）。
7. **新记录写入 `experiments/records.jsonl`**，格式见 `schema.md`。`evidence_quality` 按 schema 定义如实填写。

## 5. 待办

- [x] `eager_subset21` 已与 8 slot 的 `m8_gpqa_r1/r2` 配对（见第 2 节），结论：graph 路径在这 21 题上没有净正确率变化。
- [ ] 为 `bench_vllm`、`official/bench_vllm_tp2` 的 `per_task.jsonl` 补写 slots / caps / max_model_len，使 vLLM 对照可验证。
- [ ] 单卡正式结果（slots=12）与 ablation（slots=16）的差异：在同一 slots 下补一组对照，或在报告中注明不可直接比。
- [ ] 把 `records.jsonl` 中 E21 的"flags 未记录"备注更新为"已由 metadata 确认"（本次未修改原始记录）。
