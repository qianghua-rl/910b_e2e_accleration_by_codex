# results 目录说明

每个评测目录下的 `per_task.jsonl` 格式相同：第一行 `metadata`（配置），中间每题一行 `task`，最后一行 `summary`（makespan 等）。注意文件里的 `correct` 字段不可靠（部分结果在修复前写入），正式准确率请用离线评分器重算：选择题 `src/scorer_mc.py`，数学题 `src/scorer_math.py`。

## 顶层：正式评测

| 目录 | 引擎 / 卡数 | 数据集 | 说明 |
|---|---|---|---|
| `continuous_batching/` | 自研 CB 引擎，单卡 | aime2024、aime2025、bbh、gpqa_diamond、math500 | 单卡自研引擎的正式结果。配置：`t_max=4096`，首轮 2048 / 末轮 512 |
| `continuous_batching_tp2/` | 自研 CB 引擎，TP=2 | 同上 | 双卡结果，配置：`t_max=8192`，首轮 6144 / 末轮 2048。另含 `smoke`、`single_ref`（调试用，各 2 题，不作为结果） |
| `bench_vllm/` | vLLM-Ascend，单卡 | 同上 | 单卡 vLLM 基线。AIME 2024/2025 已用 `max_model_len=4096`、首轮 2048、末轮 512 重测（`scripts/retest_vllm_aime.sh`，结果覆盖原文件；原始 AIME 结果因上限过小无效）。其余数据集为原始运行 |
| `bench_vllm_tp2/` | vLLM-Ascend，TP=2 | 同上 | 双卡 vLLM 结果。`math500_aborted/` 是中途中止的部分运行，不可用；`smoke/` 为 2 题冒烟测试 |
| `gsm8k_128_baseline_hf/` | HF 基线 | gsm8k（128 题） | 早期 HF 串行基线，含 `summary.json` |
| `continuous_batching_tp2/gsm8k_longctx_128/` | 自研 CB，TP=2 | gsm8k_longctx_128（128 题，长上下文） | `max_model_len`/`t_max` 16384，有效结果：122/128 |
| `continuous_batching_tp2/gsm8k_longctx_128_max8192/` | 自研 CB，TP=2 | 同上 | 8192 上限，114 题因上下文不足被拒绝，**无效** |
| `bench_vllm_tp2/gsm8k_longctx_128/` | vLLM-Ascend，TP=2 | 同上 | 16384，有效结果：122/128 |
| `bench_vllm_tp2/gsm8k_longctx_128_max8192/` | vLLM-Ascend，TP=2 | 同上 | 8192，114 题被拒绝，**无效** |

单卡和双卡的 `continuous_batching` 与 `bench_vllm` 两组可以成对比较；但 TP=2 的两组配置（token 上限、`max_model_len`）与单卡不同，不能和单卡混在一起比。

## ablations/：消融与 A/B 实验

| 目录 | 内容 |
|---|---|
| `ablations/continuous_batching/` | CB 引擎的消融实验，47 个子目录。命名含义：`abba_*` 为 A/B/B/A 交替重复；`first16*` 为前 16 题子集；`full128*` 为 128 题全量；`*_graph` / `*_eager` 为是否用图执行；`*_pg1` / `*_pg4` / `*_pg8` 为 prefill group 大小；`full128_batchprefill{,4,8}` 为批量 prefill 的不同组大小；`*_fastgdn` / `*_gdnv2` 为 GDN 实现版本；`repeat_*_r2` 为重复运行。其中 `full128_batchprefill`、`graph_first16_graph`、`slots24_fast_r1` 的 per_task 中没有任务记录，属于未完成或失败的运行 |
| `ablations/shift_conv/` | shift_conv 实现方案对比，A、A2、A3、B、C、C2、C3、C4 |
| `ablations/b16/` | batch=16 的对比：`b16_original_noopt`、`b16_original_opt`、`vllm_original_b16` |
| `ablations/gdn_chunk_fast/` | GDN chunk 快速路径的测试输出（`ab_prefill_fast.txt`、`test_output.txt`） |
| `ablations/misc/` | 单次小实验：`split_test`、`prefill_split`、`decode_wall_breakdown`、`slow_batch_steps` |

## profiling/：性能剖析

| 目录 | 内容 |
|---|---|
| `profiling/decode/` | decode 阶段 profiler：`profile_decode_A`、`profile_decode_B`、`profile_decode_b1`、`profile_cb_decode_ops`、`profile_batch16_decode`、`profile_vllm_batch_decode` |
| `profiling/prefill/` | prefill 阶段 profiler：`prefill_v2_ops` |
| `profiling/sync/` | 同步开销 profiler：`profile_sync_stack` |

## microbench/：微基准

| 目录 | 内容 |
|---|---|
| `microbench/sync/sync_micro_v4/` | 同步与算子微基准，最新版本 |

## archive/：旧版本与废弃结果（未删除）

| 目录 | 内容 |
|---|---|
| `archive/sync_micro/` | sync_micro 的 v1、v2、v3 旧版本 |
| `archive/old_cap512/` | 旧的 cap=512 配置结果（aime2024、aime2025、gpqa_diamond、math500、vllm_static） |
| `archive/unfair_e100_concise_aborted/` | 配置不公平且中途中止的运行，结果不完整，仅作留档 |

## 文档

| 文件 | 内容 |
|---|---|
| `REPORT.md` | 自研 CB 引擎与 vLLM-Ascend 的对比报告（Qwen3.5-27B） |
| `excluded_directions.md` | 已排除的优化方向 |

## 与旧版 README 的变化

旧版列出的 `benchmarks/bench/`、`benchmarks/baseline_serial/`、`benchmarks/smoke2/` 在当前目录中已不存在，对应的单卡 CB 正式结果现在位于顶层 `continuous_batching/`。旧版"待确认"中提到的 `bench/` 与 `bench_vllm/` 是否重复的问题已经解决：它们是单卡自研引擎与单卡 vLLM 的一对比较，不是重复数据。
