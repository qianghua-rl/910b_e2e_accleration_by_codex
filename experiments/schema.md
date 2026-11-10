# 实验记录 schema（v0）

每次实验一行 JSON，写入 `experiments/records.jsonl`。字段缺失填 `null`，不要省略键。

| 字段 | 类型 | 含义 |
|---|---|---|
| `id` | str | 短 ID，如 `E01` |
| `date` | str | 日期，YYYY-MM-DD |
| `topic` | str | 所属方向：`decode` / `prefill` / `scheduling` / `memory` / `correctness` / `tp` / `baseline` |
| `hypothesis` | str | 做这个改动前的预期，要求写成可证伪的句子，并写出预期方向 |
| `change` | str | 改动内容（代码路径或 flags） |
| `config` | obj | 运行配置：`device`, `slots`, `prefill_group`, `flags`, `tasks`（如 first16 / full128） |
| `wall_s` | obj | `{"A": x, "B": y}` 或单值；端到端 makespan，单位秒 |
| `decode_ms_per_step` | obj | 解码每步 wall time，毫秒 |
| `device_ms_per_step` | obj | profiler 的 device 时间之和，毫秒（不是 wall） |
| `host_ms_per_step` | obj | Python 侧或同步阻塞时间，毫秒（只有测过才填） |
| `measurement` | str | 测量方法：`ABBA` / `single_pair` / `profiler` / `wrapper_timer` / `makespan` |
| `correct` | obj | 正确数，如 `{"A": "16/16", "B": "16/16"}` |
| `text_identical` | str | 与对照的 final_text 一致数 |
| `decision` | str | `accepted` / `rejected` / `flag_off` / `diagnostic` / `baseline` |
| `conclusion` | str | 文档里的结论，原样或精简 |
| `evidence_quality` | str | `high`（ABBA 或多次重复）/ `medium`（单对或单次）/ `low`（不可比或缺数据） |
| `source` | str | HANDOFF 章节或结果路径 |
| `notes` | str | 不可比、已知偏差、后来被推翻的结论 |

## 归因层需要的派生量（回放时计算）
- `wall_gain = (wall_A - wall_B) / wall_A`
- `device_gain` 同理，用 `device_ms_per_step`
- `host_fraction = host_ms / decode_ms_per_step`
- `dispatch_mismatch = device_gain 显著 且 wall_gain 不显著`：说明瓶颈不在 device
- 决策规则：只有 `wall_gain` 超过门槛（如 2%）且通过 ABBA，才算 accepted
