# Agent 实验报告: `0723_1118_2026`

**生成时间**: 2026-07-23 12:28:47
**数据来源**: `/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/ssh_agent/output/0723_1118_2026`
**总 Trial 数**: 2

## 实验概览

- **最终阶段**: `hardware_tuning`
- **总 Trial 数**: 2


| Trial | 阶段 | 结果 | 吞吐量 | Reward (均值) | Reward (最大) | 显存峰值% | Agent 工具调用 |
|---|---|---|---:|---:|---:|---:|:---:|
| 1 | hardware_tuning | success | 1475 | -0.9932 | -0.9902 | 62.0% | - |
| 2 | hardware_tuning | success | 1351 | -0.9860 | -0.9824 | 79.1% | 7 |

---

## 逐 Trial 详细分析

### Trial 1: hardware_tuning

- **结果**: `success` | **完成步数**: 6/6

#### 初始参数（基准）

_完整参数见 `trials/0001/parameters.json`_

### 关键指标

| 指标 | 均值 | P95 | 最大值 |
|---|---|---|---|
| 吞吐量 (tok/s) | 1474.9 | 1474.9 | 1474.9 |
| 每步耗时 (s) | 130.5 | 130.5 | 130.5 |
| 生成 tgs | 2486.0 | 2486.0 | 2486.0 |
| Actor MFU | 0.1464 | 0.1464 | 0.1464 |
| **时间瓶颈** | rollout |||

**各阶段耗时 (s):**

| 阶段 | 均值 | P95 | 最大 |
|---|---|---|---|
| rollout | 77.4 | 77.4 | 77.4 |
| actor_log_prob | 15.3 | 15.3 | 15.3 |
| ref_log_prob | 7.0 | 7.0 | 7.0 |
| training | 30.5 | 30.5 | 30.5 |

**稳定性指标:**

| 指标 | 均值 | P95 | 最大 |
|---|---|---|---|
| Reward | -0.9932 | -0.9902 | -0.9902 |
| Reward 斜率 | -0.000488 |||
| Actor PPO KL | 0.00029040 | 0.00029040 | 0.00029040 |
| Actor Entropy | 1.2538 | 1.2538 | 1.2538 |
| Clip Fraction | 0.000002 | 0.000002 | 0.000002 |
| Response Length | 1348.4 | 1348.4 | 1348.4 |

- **显存瓶颈**: ref_log_prob
- **峰值显存**: 62.0%

### Agent 行为分析

_Trial 1 是基准试验，无需 Agent 提出变更建议。_

📄 [训练日志](/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/ssh_agent/output/0723_1118_2026/trials/0001/train.log)

---

### Trial 2: hardware_tuning

- **结果**: `success` | **完成步数**: 6/6

#### 参数变更（相比上一 Trial）

| 参数 | 旧值 | 新值 |
|---|---|---|
| `enable_chunked_prefill` | `None` | `True` |
| `enable_prefix_caching` | `None` | `True` |
| `gpu_memory_utilization` | `0.5` | `0.7` |


### 关键指标

| 指标 | 均值 | P95 | 最大值 |
|---|---|---|---|
| 吞吐量 (tok/s) | 1350.6 | 1350.6 | 1350.6 |
| 每步耗时 (s) | 126.1 | 126.1 | 126.1 |
| 生成 tgs | 2292.4 | 2292.4 | 2292.4 |
| Actor MFU | 0.1309 | 0.1309 | 0.1309 |
| **时间瓶颈** | rollout |||

**各阶段耗时 (s):**

| 阶段 | 均值 | P95 | 最大 |
|---|---|---|---|
| rollout | 74.3 | 74.3 | 74.3 |
| actor_log_prob | 14.9 | 14.9 | 14.9 |
| ref_log_prob | 6.7 | 6.7 | 6.7 |
| training | 30.0 | 30.0 | 30.0 |

**稳定性指标:**

| 指标 | 均值 | P95 | 最大 |
|---|---|---|---|
| Reward | -0.9860 | -0.9824 | -0.9824 |
| Reward 斜率 | -0.000488 |||
| Actor PPO KL | 0.00058883 | 0.00058883 | 0.00058883 |
| Actor Entropy | 1.7963 | 1.7963 | 1.7963 |
| Clip Fraction | 0.000041 | 0.000041 | 0.000041 |
| Response Length | 1174.8 | 1174.8 | 1174.8 |

- **显存瓶颈**: rollout
- **峰值显存**: 79.1%

### Agent 行为分析

**工具调用 (7 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `tuning_strategies` | end_to_end_throughput, ref_logprob_memory_pressure, low_gpu_utilization | success |
| 1 | `parameter_understanding` | gpu_memory_utilization, max_num_batched_tokens, max_num_seqs, enable_chunked_prefill, free_cache_engine, enable_prefix_caching …共 8 个 | success |
| 2 | `memory_estimator` | ref_trial=1, 预测参数: gpu_memory_utilization, free_cache_engine | success |
| 2 | `search_verl_docs` | 查询: "max_num_batched_tokens max_num_seqs default value vllm rollout" | success |
| 3 | `parameter_understanding` | enable_chunked_prefill, enforce_eager | success |
| 3 | `memory_estimator` | ref_trial=1, 预测参数: gpu_memory_utilization, free_cache_engine, enable_chunked_prefill | success |
| 4 | `memory_estimator` | ref_trial=1, 预测参数: gpu_memory_utilization, enable_chunked_prefill | success |


#### Agent 决策

- **决策**: `modify`
- **原因**: Rollout dominates step time at 77.4s (59%). GPU memory utilization of 0.5 leaves substantial KV-cache headroom unused — raising to 0.7 projects 71.89% peak, safely under the 92% limit. Enabling chunked prefill smooths scheduling by splitting prompt prefills into smaller chunks that can interleave with decode. Enabling prefix caching exploits GRPO's n=4 sampling: all four responses share the same prompt, so the prompt KV is computed once and reused, directly reducing prefill work. All three changes are complementary and low-risk for memory.
- **置信度**: 0.7

| 参数 | 旧值 | 新值 | 原因 |
|---|---|---|---|
| `gpu_memory_utilization` | `0.5` | `0.7` | Rollout is the time bottleneck with only 59.15% peak memory. Raising to 0.7 gives vLLM more KV-cache budget for higher concurrency, projecting to 71.89% — well under the 92% throughput limit. |
| `enable_chunked_prefill` | `None` | `True` | Splits long prompt prefills into scheduler-friendly chunks, allowing prefill and decode work to overlap. Reduces single-prefill latency spikes during rollout without meaningful memory impact. |
| `enable_prefix_caching` | `None` | `True` | GRPO with n=4 samples the same prompt four times. Prefix caching computes the shared prompt KV once and reuses it across all four responses, directly cutting redundant prefill computation in the rollout bottleneck. |


### Feasibility 审查

- **#1 判定**: `?` — 

📄 [训练日志](/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/ssh_agent/output/0723_1118_2026/trials/0002/train.log)

---
