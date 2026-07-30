# Agent 实验报告: `0722_1805_2026`

**生成时间**: 2026-07-23 10:22:50
**数据来源**: `/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/ssh_agent/output/0722_1805_2026`
**总 Trial 数**: 2

## 实验概览

- **最终阶段**: `hardware_tuning`
- **总 Trial 数**: 2


| Trial | 阶段 | 结果 | 吞吐量 | Reward (均值) | Reward (最大) | 显存峰值% | Agent 工具调用 |
|---|---|---|---:|---:|---:|---:|:---:|
| 1 | hardware_tuning | success | 1188 | -0.9948 | -0.9863 | 61.6% | - |
| 2 | hardware_tuning | success | 1110 | -0.9928 | -0.9902 | 70.0% | 7 |

---

## 逐 Trial 详细分析

### Trial 1: hardware_tuning

- **结果**: `success` | **完成步数**: 6/6

#### 初始参数（基准）

_完整参数见 `trials/0001/parameters.json`_

### 关键指标

| 指标 | 均值 | P95 | 最大值 |
|---|---|---|---|
| 吞吐量 (tok/s) | 1188.0 | 1188.0 | 1188.0 |
| 每步耗时 (s) | 125.7 | 125.7 | 125.7 |
| 生成 tgs | 1996.9 | 1996.9 | 1996.9 |
| Actor MFU | 0.1169 | 0.1169 | 0.1169 |
| **时间瓶颈** | rollout |||

**各阶段耗时 (s):**

| 阶段 | 均值 | P95 | 最大 |
|---|---|---|---|
| rollout | 74.8 | 74.8 | 74.8 |
| actor_log_prob | 14.6 | 14.6 | 14.6 |
| ref_log_prob | 6.2 | 6.2 | 6.2 |
| training | 29.9 | 29.9 | 29.9 |

**稳定性指标:**

| 指标 | 均值 | P95 | 最大 |
|---|---|---|---|
| Reward | -0.9948 | -0.9863 | -0.9863 |
| Reward 斜率 | -0.001953 |||
| Actor PPO KL | 0.00002032 | 0.00002032 | 0.00002032 |
| Actor Entropy | 2.5900 | 2.5900 | 2.5900 |
| Clip Fraction | 0.000000 | 0.000000 | 0.000000 |
| Response Length | 1010.5 | 1010.5 | 1010.5 |

- **显存瓶颈**: ref_log_prob
- **峰值显存**: 61.6%

### Agent 行为分析

_Trial 1 是基准试验，无需 Agent 提出变更建议。_

📄 [训练日志](/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/ssh_agent/output/0722_1805_2026/trials/0001/train.log)

---

### Trial 2: hardware_tuning

- **结果**: `success` | **完成步数**: 6/6

#### 参数变更（相比上一 Trial）

| 参数 | 旧值 | 新值 |
|---|---|---|
| `gpu_memory_utilization` | `0.5` | `0.65` |
| `max_num_batched_tokens` | `None` | `8192` |
| `max_num_seqs` | `None` | `64` |


### 关键指标

| 指标 | 均值 | P95 | 最大值 |
|---|---|---|---|
| 吞吐量 (tok/s) | 1109.9 | 1109.9 | 1109.9 |
| 每步耗时 (s) | 195.9 | 195.9 | 195.9 |
| 生成 tgs | 1563.9 | 1563.9 | 1563.9 |
| Actor MFU | 0.1665 | 0.1665 | 0.1665 |
| **时间瓶颈** | rollout |||

**各阶段耗时 (s):**

| 阶段 | 均值 | P95 | 最大 |
|---|---|---|---|
| rollout | 139.0 | 139.0 | 139.0 |
| actor_log_prob | 16.2 | 16.2 | 16.2 |
| ref_log_prob | 7.7 | 7.7 | 7.7 |
| training | 32.8 | 32.8 | 32.8 |

**稳定性指标:**

| 指标 | 均值 | P95 | 最大 |
|---|---|---|---|
| Reward | -0.9928 | -0.9902 | -0.9902 |
| Reward 斜率 | 0.000000 |||
| Actor PPO KL | 0.00033500 | 0.00033500 | 0.00033500 |
| Actor Entropy | 2.5733 | 2.5733 | 2.5733 |
| Clip Fraction | 0.000130 | 0.000130 | 0.000130 |
| Response Length | 1542.7 | 1542.7 | 1542.7 |

- **显存瓶颈**: ref_log_prob
- **峰值显存**: 70.0%

### Agent 行为分析

**工具调用 (7 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `tuning_strategies` | end_to_end_throughput, low_gpu_utilization | success |
| 1 | `parameter_understanding` | gpu_memory_utilization, max_num_batched_tokens, max_num_seqs, enable_chunked_prefill, enable_prefix_caching, train_batch_size …共 8 个 | success |
| 2 | `memory_estimator` | ref_trial=1, 预测参数: gpu_memory_utilization, max_num_batched_tokens, max_num_seqs | success |
| 2 | `parameter_understanding` | enforce_eager, free_cache_engine, recompute_granularity, recompute_method | success |
| 3 | `memory_estimator` | ref_trial=1, 预测参数: max_num_batched_tokens, max_num_seqs, gpu_memory_utilization | success |
| 3 | `search_verl_docs` | 查询: "max_num_batched_tokens max_num_seqs vllm rollout default" | success |
| 4 | `memory_estimator` | ref_trial=1, 预测参数: max_num_batched_tokens, max_num_seqs, gpu_memory_utilization | success |


#### Agent 决策

- **决策**: `modify`
- **原因**: rollout 占总步时的 59.5% 是首要时间瓶颈，actor MFU 仅 11.7% 表明 GPU 利用率低，且 ref/training 显存均有 >30% 余量。按 low_gpu_utilization 策略：先通过 max_num_batched_tokens + max_num_seqs 提升 rollout 并发度，同时适度提高 gpu_memory_utilization 为 KV-cache 预留空间；memory_estimator 显示非 rollout 阶段不变、rollout 压力比低于基线。
- **置信度**: 0.7

| 参数 | 旧值 | 新值 | 原因 |
|---|---|---|---|
| `gpu_memory_utilization` | `0.5` | `0.65` | 为显式增大 max_num_batched_tokens/max_num_seqs 后的 KV-cache 预留空间，0.5→0.65 适度提升，其他阶段仍有充足 headroom |
| `max_num_batched_tokens` | `None` | `8192` | 当前未设置，vLLM 自行推断；显式设为 8192 可覆盖单条完整序列(1280+5120=6400)并支持部分 batch，提升 prefill/decode 利用率 |
| `max_num_seqs` | `None` | `64` | 当前未设置；train_batch_size=256 × n=4 / 8GPUs ≈ 128 条序列，64 并发可提升 batch 效率且不超 KV-cache 容量 |


### Feasibility 审查

- **#1 判定**: `?` — 

📄 [训练日志](/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/ssh_agent/output/0722_1805_2026/trials/0002/train.log)

---
