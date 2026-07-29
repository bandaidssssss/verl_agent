# Agent 实验报告: `0728_1501_2026`

**生成时间**: 2026-07-29 09:54:56
**数据来源**: `/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/ssh_agent/output/0728_1501_2026`
**总 Trial 数**: 1

## 实验概览

- **最终阶段**: `hardware_tuning`
- **总 Trial 数**: 1


| Trial | 阶段 | 结果 | 吞吐量 | Reward (均值) | Reward (最大) | 显存峰值% | Health 决策 | 完成后的 Agent 工具调用（D + P + F） |
|---|---|---|---:|---:|---:|---:|:---:|:---:|
| 1 | hardware_tuning | success | 1402 | -0.9910 | -0.9824 | 89.0% | - | - |

---

## 逐 Trial 详细分析

### Trial 1: hardware_tuning

- **结果**: `success` | **完成步数**: 10/10

#### 初始参数（基准）

_完整参数见 `trials/0001/parameters.json`_

### 关键指标

| 指标 | 均值 | P95 | 最大值 |
|---|---|---|---|
| 吞吐量 (tok/s) | 1402.0 | 1459.6 | 1459.6 |
| 每步耗时 (s) | 103.5 | 108.3 | 108.3 |
| 生成 tgs | 2105.2 | 2186.8 | 2186.8 |
| Actor MFU | 0.1802 | 0.1907 | 0.1907 |
| **时间瓶颈** | rollout |||

**各阶段耗时 (s):**

| 阶段 | 均值 | P95 | 最大 |
|---|---|---|---|
| rollout | 69.0 | 73.1 | 73.1 |
| actor_log_prob | 9.3 | 9.6 | 9.6 |
| ref_log_prob | 6.1 | 6.3 | 6.3 |
| training | 18.9 | 19.3 | 19.3 |

**稳定性指标:**

| 指标 | 均值 | P95 | 最大 |
|---|---|---|---|
| Reward | -0.9910 | -0.9824 | -0.9824 |
| Reward 斜率 | 0.000000 |||
| Actor PPO KL | 0.00001834 | 0.00034787 | 0.00034787 |
| Actor Entropy | 2.2855 | 2.5222 | 2.5222 |
| Clip Fraction | 0.000008 | 0.000019 | 0.000019 |
| Response Length | 982.7 | 1041.0 | 1041.0 |

- **显存瓶颈**: rollout
- **峰值显存**: 89.0%

### 本 Trial 完成后的 Agent 行为

_这是最后一个 Trial，记录中没有后续 Agent trace。_

📄 [训练日志](/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/ssh_agent/output/0728_1501_2026/trials/0001/train.log)

---
