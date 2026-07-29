# Agent 实验报告: `0729_0955_2026`

**生成时间**: 2026-07-29 14:47:17
**数据来源**: `/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/ssh_agent/output/0729_0955_2026`
**总 Trial 数**: 9

## 实验概览

- **最终阶段**: `stability_tuning`
- **总 Trial 数**: 9


| Trial | 阶段 | 结果 | 吞吐量 | Reward (均值) | Reward (最大) | 显存峰值% | Health 决策 | 完成后的 Agent 工具调用（D + P + F） |
|---|---|---|---:|---:|---:|---:|:---:|:---:|
| 1 | hardware_tuning | success | 1031 | -0.9895 | -0.9863 | 59.1% | - | 0 + 3 + 2 = 5 |
| 2 | hardware_tuning | success | 1128 | -0.9900 | -0.9844 | 79.0% | - | 0 + 6 + 2 = 8 |
| 3 | hardware_tuning | success | 1107 | -0.9906 | -0.9824 | 84.0% | - | 0 + 3 + 3 = 6 |
| 4 | hardware_tuning | success | 1175 | -0.9914 | -0.9863 | 76.5% | - | 0 + 7 + 6 = 13 |
| 5 | hardware_tuning | success | 1515 | -0.9904 | -0.9805 | 78.1% | - | 0 + 3 + 2 = 5 |
| 6 | hardware_tuning | fail | 1509 | -0.9922 | -0.9922 | 99.8% | - | - |
| 7 | stability_tuning | success | 1151 | -0.9454 | -0.8516 | 86.6% | - | 0 + 3 + 1 = 4 |
| 8 | stability_tuning | fail | 1701 | -0.9941 | -0.9922 | 95.9% | - | 6 + 5 + 3 = 14 |
| 9 | stability_tuning | fail | 1597 | -0.9926 | -0.9883 | 98.8% | - | - |

---

## 逐 Trial 详细分析

### Trial 1: hardware_tuning

- **结果**: `success` | **完成步数**: 10/10

#### 初始参数（基准）

_完整参数见 `trials/0001/parameters.json`_

### 关键指标

| 指标 | 均值 | P95 | 最大值 |
|---|---|---|---|
| 吞吐量 (tok/s) | 1030.5 | 1112.3 | 1112.3 |
| 每步耗时 (s) | 108.4 | 112.4 | 112.4 |
| 生成 tgs | 1868.1 | 1992.2 | 1992.2 |
| Actor MFU | 0.0854 | 0.0954 | 0.0954 |
| **时间瓶颈** | rollout |||

**稳定性指标:**

| 指标 | 均值 | P95 | 最大 |
|---|---|---|---|
| Reward | -0.9895 | -0.9863 | -0.9863 |
| Reward 斜率 | 0.000488 |||
| Actor PPO KL | -0.00042065 | 0.00009198 | 0.00009198 |
| Actor Entropy | 1.6738 | 1.9796 | 1.9796 |
| Clip Fraction | 0.000053 | 0.000171 | 0.000171 |
| Response Length | 722.0 | 813.4 | 813.4 |

**分阶段耗时、显存与 GPU 利用率:**

| 阶段 | 耗时均值 (s) | 显存均值 (%) | 显存峰值 (%) | GPU 利用率均值 (%) | GPU 利用率 P95 (%) | GPU 利用率峰值 (%) |
|---|---:|---:|---:|---:|---:|---:|
| rollout | 59.8 | 49.4 | 59.1 | 30.3 | 48.0 | 52.0 |
| actor_log_prob | 13.9 | 21.7 | 49.3 | 28.5 | 70.0 | 73.0 |
| ref_log_prob | 5.3 | 28.4 | 55.4 | 34.3 | 84.0 | 88.0 |
| training | 29.0 | 26.6 | 37.7 | 29.2 | 70.0 | 78.0 |

- **相对最高显存阶段**: rollout （仅表示四阶段中峰值最高，不等于存在显存压力）
- **总体实测峰值**: 59.1%

### 本 Trial 完成后的 Agent 行为

_以下行为用于生成 Trial 2 的候选配置。_

#### Proposal Agent 决策

**Proposal 工具调用 (3 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `parameter_understanding` | actor_rollout_ref.rollout.gpu_memory_utilization, actor_rollout_ref.rollout.max_num_batched_tokens, actor_rollout_ref.rollout.max_num_seqs, actor_rollout_ref.rollout.tensor_model_parallel_size | success |
| 2 | `query_trial_history` | stage=hardware_tuning, limit=5 | success |
| 3 | `memory_estimator` | ref_trial=1, 预测参数: actor_rollout_ref.rollout.gpu_memory_utilization, actor_rollout_ref.rollout.max_num_batched_tokens | success |


- **决策**: `modify`
- **原因**: 当前 rollout 阶段耗时最多（均值 59.8s），且显存峰值仅 59.1%，有较大余量。提高 gpu_memory_utilization 可扩充 KV-cache 预算，提高 max_num_batched_tokens 可增大调度批次，二者协同可提升生成吞吐。内存估算显示 rollout 投影峰值 79.0%，上界 82.0%，风险低。
- **置信度**: 0.65

| 参数 | 旧值 | 新值 | 原因 |
|---|---|---|---|
| `actor_rollout_ref.rollout.gpu_memory_utilization` | `0.5` | `0.7` | 扩充 vLLM 推理引擎的显存预算，使 KV-cache 容量从约 50% 提升至 70%，允许容纳更大的并发批处理。 |
| `actor_rollout_ref.rollout.max_num_batched_tokens` | `8192` | `16384` | 当前 token 上限可能限制批次大小（响应长度 ~722，8192 tokens 仅能容纳约 11 个序列），翻倍后可更好地利用 GPU 并发能力，减少调度波次。 |


#### Feasibility Agent 审查

**审查 #1（attempt=1）**

**Feasibility 工具调用 (2 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `parameter_understanding` | actor_rollout_ref.rollout.gpu_memory_utilization, actor_rollout_ref.rollout.max_num_batched_tokens | success |
| 1 | `memory_estimator` | ref_trial=1, 预测参数: actor_rollout_ref.rollout.gpu_memory_utilization, actor_rollout_ref.rollout.max_num_batched_tokens | success |

- **判定**: `valid`
- **原因**: Rollout 显存上界 82% 低于吞吐安全线 92%，其他阶段无变化，风险可控。max_num_batched_tokens 翻倍效果未校准但估算保持安全，低置信度需真实短跑确认。
- **风险**:
  - rollout 阶段置信度低，max_num_batched_tokens 属于未校准变更，实际显存压力可能略高于估计
  - 最终安全性仍需由真实短跑 resource gate 确认

**预测显存占用 (%):**

| rollout | actor_log_prob | ref_log_prob | training |
|---:|---:|---:|---:|
| 79.00 | 49.26 | 55.40 | 37.66 |

📄 [训练日志](/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/ssh_agent/output/0729_0955_2026/trials/0001/train.log)

---

### Trial 2: hardware_tuning

- **结果**: `success` | **完成步数**: 10/10

#### 参数变更（相比上一 Trial）

| 参数 | 旧值 | 新值 |
|---|---|---|
| `actor_rollout_ref.rollout.gpu_memory_utilization` | `0.5` | `0.7` |
| `actor_rollout_ref.rollout.max_num_batched_tokens` | `8192` | `16384` |


### 关键指标

| 指标 | 均值 | P95 | 最大值 |
|---|---|---|---|
| 吞吐量 (tok/s) | 1127.8 | 1203.0 | 1203.0 |
| 每步耗时 (s) | 112.1 | 120.5 | 120.5 |
| 生成 tgs | 2011.9 | 2102.9 | 2102.9 |
| Actor MFU | 0.0977 | 0.1123 | 0.1123 |
| **时间瓶颈** | rollout |||

**稳定性指标:**

| 指标 | 均值 | P95 | 最大 |
|---|---|---|---|
| Reward | -0.9900 | -0.9844 | -0.9844 |
| Reward 斜率 | -0.000488 |||
| Actor PPO KL | -0.00005204 | 0.00021726 | 0.00021726 |
| Actor Entropy | 1.8624 | 2.0708 | 2.0708 |
| Clip Fraction | 0.000050 | 0.000173 | 0.000173 |
| Response Length | 837.4 | 984.1 | 984.1 |

**分阶段耗时、显存与 GPU 利用率:**

| 阶段 | 耗时均值 (s) | 显存均值 (%) | 显存峰值 (%) | GPU 利用率均值 (%) | GPU 利用率 P95 (%) | GPU 利用率峰值 (%) |
|---|---:|---:|---:|---:|---:|---:|
| rollout | 62.9 | 66.9 | 79.0 | 32.3 | 49.0 | 52.0 |
| actor_log_prob | 14.2 | 21.7 | 49.6 | 30.8 | 69.0 | 74.0 |
| ref_log_prob | 5.7 | 32.7 | 56.7 | 49.9 | 87.0 | 88.0 |
| training | 29.1 | 27.1 | 38.0 | 30.7 | 72.0 | 78.0 |

- **相对最高显存阶段**: rollout （仅表示四阶段中峰值最高，不等于存在显存压力）
- **总体实测峰值**: 79.0%

### 本 Trial 完成后的 Agent 行为

_以下行为用于生成 Trial 3 的候选配置。_

#### Proposal Agent 决策

**Proposal 工具调用 (6 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `parameter_understanding` | actor_rollout_ref.rollout.gpu_memory_utilization, actor_rollout_ref.rollout.max_num_batched_tokens, actor_rollout_ref.rollout.max_num_seqs, actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu | success |
| 2 | `memory_estimator` | ref_trial=2, 预测参数: actor_rollout_ref.rollout.gpu_memory_utilization, actor_rollout_ref.rollout.max_num_seqs | success |
| 3 | `query_trial_history` | stage=hardware_tuning, limit=5 | success |
| 4 | `parameter_understanding` | actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu, actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu, actor_rollout_ref.rollout.max_num_batched_tokens | success |
| 4 | `memory_estimator` | ref_trial=2, 预测参数: actor_rollout_ref.rollout.max_num_batched_tokens | success |
| 5 | `memory_estimator` | ref_trial=2, 预测参数: actor_rollout_ref.rollout.gpu_memory_utilization, actor_rollout_ref.rollout.max_num_batched_tokens | success |


- **决策**: `modify`
- **原因**: Trial 2 通过提升 gpu_memory_utilization 和 max_num_batched_tokens 获得了 9.4% 的吞吐提升，rollout 仍是耗时瓶颈（62.9s）。在安全显存边界内（上限 92%）进一步扩展 rollout 容量，有望延续吞吐改善。memory_estimator 显示 gpu_memory_utilization 提至 0.75 且 max_num_batched_tokens 提至 24576 时，rollout 投影 83.8%、上界 86.8%，风险 low。max_num_batched_tokens 的调整虽未经校准，但历史 Trial 中该参数提升与吞吐正相关。
- **置信度**: 0.65

| 参数 | 旧值 | 新值 | 原因 |
|---|---|---|---|
| `actor_rollout_ref.rollout.gpu_memory_utilization` | `0.7` | `0.75` | 在 Trial 2 的 0.7 基础上小幅增加显存预算，为更大的批次和 KV-cache 提供空间，同时保留安全余量。 |
| `actor_rollout_ref.rollout.max_num_batched_tokens` | `16384` | `24576` | 进一步抬高 token 上限，当前响应平均长度 837、最大 5120，16384 可能限制单批并发序列数；提升后可减少调度波次，提高 GPU 利用率。 |


#### Feasibility Agent 审查

**审查 #1（attempt=1）**

**Feasibility 工具调用 (2 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `parameter_understanding` | actor_rollout_ref.rollout.gpu_memory_utilization, actor_rollout_ref.rollout.max_num_batched_tokens | success |
| 2 | `memory_estimator` | ref_trial=2, 预测参数: actor_rollout_ref.rollout.gpu_memory_utilization, actor_rollout_ref.rollout.max_num_batched_tokens | success |

- **判定**: `valid`
- **原因**: Rollout 显存 upper_bound 86.83% 仍低于吞吐安全线 92.0%，剩余阶段不变且安全余量充足。max_num_batched_tokens 为未校准变更，但整体风险低，候选值得进行短跑测试。
- **风险**:
  - max_num_batched_tokens 属于未校准变更，实际显存压力可能略高于估计
  - 最终安全性仍需由真实短跑 resource gate 确认

**预测显存占用 (%):**

| rollout | actor_log_prob | ref_log_prob | training |
|---:|---:|---:|---:|
| 83.83 | 49.55 | 56.67 | 38.02 |

📄 [训练日志](/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/ssh_agent/output/0729_0955_2026/trials/0002/train.log)

---

### Trial 3: hardware_tuning

- **结果**: `success` | **完成步数**: 10/10

#### 参数变更（相比上一 Trial）

| 参数 | 旧值 | 新值 |
|---|---|---|
| `actor_rollout_ref.rollout.gpu_memory_utilization` | `0.7` | `0.75` |
| `actor_rollout_ref.rollout.max_num_batched_tokens` | `16384` | `24576` |


### 关键指标

| 指标 | 均值 | P95 | 最大值 |
|---|---|---|---|
| 吞吐量 (tok/s) | 1106.8 | 1148.4 | 1148.4 |
| 每步耗时 (s) | 110.3 | 115.1 | 115.1 |
| 生成 tgs | 1999.2 | 2058.6 | 2058.6 |
| Actor MFU | 0.0940 | 0.1014 | 0.1014 |
| **时间瓶颈** | rollout |||

**稳定性指标:**

| 指标 | 均值 | P95 | 最大 |
|---|---|---|---|
| Reward | -0.9906 | -0.9824 | -0.9824 |
| Reward 斜率 | -0.000488 |||
| Actor PPO KL | -0.00005611 | 0.00023962 | 0.00023962 |
| Actor Entropy | 1.7310 | 2.0928 | 2.0928 |
| Clip Fraction | 0.000009 | 0.000015 | 0.000015 |
| Response Length | 802.4 | 883.9 | 883.9 |

**分阶段耗时、显存与 GPU 利用率:**

| 阶段 | 耗时均值 (s) | 显存均值 (%) | 显存峰值 (%) | GPU 利用率均值 (%) | GPU 利用率 P95 (%) | GPU 利用率峰值 (%) |
|---|---:|---:|---:|---:|---:|---:|
| rollout | 61.1 | 64.7 | 84.0 | 29.2 | 48.0 | 56.0 |
| actor_log_prob | 14.1 | 21.2 | 29.5 | 29.1 | 71.0 | 73.0 |
| ref_log_prob | 5.7 | 31.1 | 56.5 | 48.1 | 86.0 | 88.0 |
| training | 29.1 | 27.2 | 39.1 | 30.2 | 73.0 | 83.0 |

- **相对最高显存阶段**: rollout （仅表示四阶段中峰值最高，不等于存在显存压力）
- **总体实测峰值**: 84.0%

### 本 Trial 完成后的 Agent 行为

_以下行为用于生成 Trial 4 的候选配置。_

#### Proposal Agent 决策

**Proposal 工具调用 (3 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `parameter_understanding` | actor_rollout_ref.rollout.tensor_model_parallel_size, actor_rollout_ref.rollout.gpu_memory_utilization, actor_rollout_ref.rollout.max_num_batched_tokens, actor_rollout_ref.rollout.max_num_seqs, actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu, actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu …共 8 个 | success |
| 2 | `query_trial_history` | stage=hardware_tuning, limit=10 | success |
| 3 | `memory_estimator` | ref_trial=2, 预测参数: actor_rollout_ref.rollout.tensor_model_parallel_size | success |


- **决策**: `modify`
- **原因**: rollout 阶段耗时占比最高（~55%），目前 rollout TP=2 引入了 per-token 通信开销。降低 TP 至 1 可消除 TP 通信，有望提升 decode 吞吐；memory_estimator 基于 Trial 2 的投影显示 rollout 峰值 upper_bound 82%（<92% 限制），风险 low，但 confidence low 且变化未经校准，需短跑验证。
- **置信度**: 0.6

| 参数 | 旧值 | 新值 | 原因 |
|---|---|---|---|
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | `2` | `1` | 降低 rollout 张量并行度以消除 per-token 通信，提升生成阶段吞吐；单 GPU 容纳完整模型和 KV cache，预计显存压力增加但仍在安全范围内。 |


#### Feasibility Agent 审查

**审查 #1（attempt=1）**

**Feasibility 工具调用 (3 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `parameter_understanding` | actor_rollout_ref.rollout.tensor_model_parallel_size, actor_rollout_ref.rollout.gpu_memory_utilization | success |
| 1 | `memory_estimator` | ref_trial=2, 预测参数: actor_rollout_ref.rollout.tensor_model_parallel_size | success |
| 2 | `memory_estimator` | ref_trial=2, 预测参数: actor_rollout_ref.rollout.tensor_model_parallel_size | success |

- **判定**: `valid`
- **原因**: Rollout TP 从 2 降至 1 可消除 per-token 通信，有望提升 rollout 吞吐；memory_estimator 基于 Trial 2 的投影显示 rollout 上界 82% 仍低于 92% 安全线，其余阶段不变且安全余量充足。变更虽未校准，但风险低。
- **风险**:
  - rollout.tensor_model_parallel_size 为未校准变更，实际显存可能略高于估计
  - 最终安全性仍需由真实短跑 resource gate 确认

**预测显存占用 (%):**

| rollout | actor_log_prob | ref_log_prob | training |
|---:|---:|---:|---:|
| 79.00 | 49.55 | 56.67 | 38.02 |

📄 [训练日志](/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/ssh_agent/output/0729_0955_2026/trials/0003/train.log)

---

### Trial 4: hardware_tuning

- **结果**: `success` | **完成步数**: 10/10

#### 参数变更（相比上一 Trial）

| 参数 | 旧值 | 新值 |
|---|---|---|
| `actor_rollout_ref.rollout.gpu_memory_utilization` | `0.75` | `0.7` |
| `actor_rollout_ref.rollout.max_num_batched_tokens` | `24576` | `16384` |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | `2` | `1` |


### 关键指标

| 指标 | 均值 | P95 | 最大值 |
|---|---|---|---|
| 吞吐量 (tok/s) | 1174.5 | 1236.5 | 1236.5 |
| 每步耗时 (s) | 107.1 | 109.5 | 109.5 |
| 生成 tgs | 2162.8 | 2301.4 | 2301.4 |
| Actor MFU | 0.0991 | 0.1071 | 0.1071 |
| **时间瓶颈** | rollout |||

**稳定性指标:**

| 指标 | 均值 | P95 | 最大 |
|---|---|---|---|
| Reward | -0.9914 | -0.9863 | -0.9863 |
| Reward 斜率 | -0.000488 |||
| Actor PPO KL | -0.00009549 | 0.00040196 | 0.00040196 |
| Actor Entropy | 1.9082 | 2.0403 | 2.0403 |
| Clip Fraction | 0.000021 | 0.000091 | 0.000091 |
| Response Length | 831.9 | 883.0 | 883.0 |

**分阶段耗时、显存与 GPU 利用率:**

| 阶段 | 耗时均值 (s) | 显存均值 (%) | 显存峰值 (%) | GPU 利用率均值 (%) | GPU 利用率 P95 (%) | GPU 利用率峰值 (%) |
|---|---:|---:|---:|---:|---:|---:|
| rollout | 58.2 | 63.1 | 76.5 | 35.8 | 58.0 | 64.0 |
| actor_log_prob | 14.2 | 19.4 | 27.6 | 29.6 | 71.0 | 74.0 |
| ref_log_prob | 5.7 | 31.2 | 55.7 | 51.1 | 87.0 | 88.0 |
| training | 28.8 | 24.3 | 35.0 | 30.5 | 72.0 | 79.0 |

- **相对最高显存阶段**: rollout （仅表示四阶段中峰值最高，不等于存在显存压力）
- **总体实测峰值**: 76.5%

### 本 Trial 完成后的 Agent 行为

_以下行为用于生成 Trial 5 的候选配置。_

#### Proposal Agent 决策

**Proposal 工具调用 (7 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `parameter_understanding` | actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu, actor_rollout_ref.rollout.gpu_memory_utilization, actor_rollout_ref.rollout.max_num_batched_tokens, actor_rollout_ref.rollout.max_num_seqs, actor_rollout_ref.rollout.tensor_model_parallel_size, actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu …共 7 个 | success |
| 1 | `query_trial_history` | stage=hardware_tuning, limit=10 | success |
| 2 | `memory_estimator` | ref_trial=4, 预测参数: actor_rollout_ref.rollout.gpu_memory_utilization, actor_rollout_ref.rollout.max_num_batched_tokens | success |
| 3 | `memory_estimator` | ref_trial=4, 预测参数: actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu | success |
| 4 | `memory_estimator` | ref_trial=4, 预测参数: actor_rollout_ref.rollout.gpu_memory_utilization | success |
| 5 | `parameter_understanding` | actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu | success |
| 5 | `memory_estimator` | ref_trial=4, 预测参数: actor_rollout_ref.actor.megatron.tensor_model_parallel_size | success |


- **决策**: `modify`
- **原因**: Actor training with TP=2 on a 0.6B model shows severe underutilization (MFU 0.099). Reducing actor TP to 1 eliminates tensor-parallel communication, raises per-rank work, and is projected to improve training throughput. Memory estimator projects training upper bound at 59.3%, well within the 92% limit. This targets the second-largest phase (27% of step time) and is the most promising unexplored direction given rollout is already optimized and previous attempts to raise rollout caps decreased throughput.
- **置信度**: 0.7

| 参数 | 旧值 | 新值 | 原因 |
|---|---|---|---|
| `actor_rollout_ref.actor.megatron.tensor_model_parallel_size` | `2` | `1` | Eliminate tensor-parallel communication in actor training to reduce per-step training time; current MFU 0.099 indicates heavy communication overhead on small model. Memory safety confirmed by estimator (training projected 55.3%, upper 59.3%). |


#### Feasibility Agent 审查

**审查 #1（attempt=1）**

**Feasibility 工具调用 (6 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `parameter_understanding` | actor_rollout_ref.actor.megatron.tensor_model_parallel_size, actor_rollout_ref.actor.megatron.sequence_parallel, actor_rollout_ref.ref.megatron.tensor_model_parallel_size | success |
| 1 | `memory_estimator` | ref_trial=4, 预测参数: actor_rollout_ref.actor.megatron.tensor_model_parallel_size | success |
| 2 | `parameter_understanding` | actor_rollout_ref.actor.ppo_mini_batch_size, actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu | success |
| 2 | `search_verl_docs` | 查询: "sequence_parallel tensor_model_parallel_size 1" | success |
| 3 | `search_verl_docs` | 查询: "sequence_parallel TP=1" | success |
| 4 | `search_verl_docs` | 查询: "megatron sequence_parallel TP 1" | success |

- **判定**: `valid`
- **原因**: 将 actor TP 从 2 降至 1 可消除训练张量并行通信，有望提升训练吞吐；memory_estimator 基于 Trial 4 的投影显示所有阶段 upper_bound 均在 60% 以下，远低于 92% 安全线；Megatron 代码会在 TP=1 时自动关闭 sequence_parallel，因此不存在无效组合。rollout 和 ref 阶段不受影响。
- **风险**:
  - actor_log_prob 与 training 阶段为未校准变更（confidence low），实际显存可能略高于投影
  - 最终安全性仍需由真实短跑 resource gate 确认

**预测显存占用 (%):**

| rollout | actor_log_prob | ref_log_prob | training |
|---:|---:|---:|---:|
| 78.00 | 39.74 | 59.71 | 59.34 |

📄 [训练日志](/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/ssh_agent/output/0729_0955_2026/trials/0004/train.log)

---

### Trial 5: hardware_tuning

- **结果**: `success` | **完成步数**: 10/10

#### 参数变更（相比上一 Trial）

| 参数 | 旧值 | 新值 |
|---|---|---|
| `actor_rollout_ref.actor.megatron.tensor_model_parallel_size` | `2` | `1` |


### 关键指标

| 指标 | 均值 | P95 | 最大值 |
|---|---|---|---|
| 吞吐量 (tok/s) | 1515.2 | 1606.6 | 1606.6 |
| 每步耗时 (s) | 84.3 | 89.8 | 89.8 |
| 生成 tgs | 2220.3 | 2387.6 | 2387.6 |
| Actor MFU | 0.2105 | 0.2311 | 0.2311 |
| **时间瓶颈** | rollout |||

**稳定性指标:**

| 指标 | 均值 | P95 | 最大 |
|---|---|---|---|
| Reward | -0.9904 | -0.9805 | -0.9805 |
| Reward 斜率 | 0.002441 |||
| Actor PPO KL | -0.00033121 | 0.00017239 | 0.00017239 |
| Actor Entropy | 1.8508 | 2.0426 | 2.0426 |
| Clip Fraction | 0.000041 | 0.000252 | 0.000252 |
| Response Length | 846.8 | 955.5 | 955.5 |

**分阶段耗时、显存与 GPU 利用率:**

| 阶段 | 耗时均值 (s) | 显存均值 (%) | 显存峰值 (%) | GPU 利用率均值 (%) | GPU 利用率 P95 (%) | GPU 利用率峰值 (%) |
|---|---:|---:|---:|---:|---:|---:|
| rollout | 57.6 | 64.1 | 78.1 | 36.1 | 58.0 | 63.0 |
| actor_log_prob | 7.8 | 29.6 | 44.6 | 40.4 | 85.0 | 88.0 |
| ref_log_prob | 4.5 | 35.6 | 73.2 | 52.8 | 95.0 | 96.0 |
| training | 14.2 | 35.4 | 60.6 | 41.7 | 88.0 | 91.0 |

- **相对最高显存阶段**: rollout （仅表示四阶段中峰值最高，不等于存在显存压力）
- **总体实测峰值**: 78.1%

### 本 Trial 完成后的 Agent 行为

_以下行为用于生成 Trial 6 的候选配置。_

#### Proposal Agent 决策

**Proposal 工具调用 (3 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `parameter_understanding` | actor_rollout_ref.rollout.gpu_memory_utilization, actor_rollout_ref.rollout.max_num_batched_tokens, actor_rollout_ref.rollout.max_num_seqs, actor_rollout_ref.rollout.tensor_model_parallel_size, actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu, actor_rollout_ref.ref.megatron.tensor_model_parallel_size | success |
| 2 | `query_trial_history` | stage=None, limit=None | success |
| 3 | `memory_estimator` | ref_trial=5, 预测参数: actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu, actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu | success |


- **决策**: `modify`
- **原因**: rollout 占步骤时间的 68%（57.6/84.3s），actor 训练与 log-prob 阶段合计 22.0s。将 actor ppo_micro_batch 和 rollout log_prob_micro_batch 从 2 翻倍至 4，可将梯度累积步数从 4 减半至 2，并减少 actor log-prob 的调度轮次，从而缩短这两个阶段。memory_estimator 显示训练阶段峰值 72.7%（上限 76.7%）、actor log-prob 峰值 53.5%（上限 56.5%），风险均为 low，且参数整除约束满足。
- **置信度**: 0.7

| 参数 | 旧值 | 新值 | 原因 |
|---|---|---|---|
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | `2` | `4` | 扩大训练微批次以减半梯度累积步数，缩短训练阶段。estimator 预测峰值 72.7%（上限 76.7%），在 92% 限制内安全。 |
| `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu` | `2` | `4` | 扩大 actor log-prob 微批次以减少该阶段耗时。estimator 预测峰值 53.5%（上限 56.5%），内存安全。 |


#### Feasibility Agent 审查

**审查 #1（attempt=1）**

**Feasibility 工具调用 (2 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `parameter_understanding` | actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu, actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu | success |
| 1 | `memory_estimator` | ref_trial=5, 预测参数: actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu, actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu | success |

- **判定**: `valid`
- **原因**: 参考 Trial 5 的显存估算器显示：rollout 无改动峰值 78.1%→79.6%（上限），actor_log_prob 44.6%→56.5%，ref_log_prob 不变 73.3%→77.3%，training 60.6%→76.7%。所有阶段上限远低于 92% 安全线，风险低。增大微批次可减少梯度累积步数，预期缩短训练和 log_prob 阶段时间，端到端吞吐可能提升。
- **风险**:
  - actor_log_prob 与 training 阶段估算置信度为 low，实际显存可能略高于投影；最终安全性仍需由真实短跑 resource gate 确认
  - 增大微批次未必线性改善 kernel 利用率，吞吐收益需实测验证

**预测显存占用 (%):**

| rollout | actor_log_prob | ref_log_prob | training |
|---:|---:|---:|---:|
| 79.59 | 56.46 | 77.25 | 76.74 |

📄 [训练日志](/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/ssh_agent/output/0729_0955_2026/trials/0005/train.log)

---

### Trial 6: hardware_tuning

- **结果**: `fail` | **完成步数**: 1/10
- **错误类型**: MEMORY_HEADROOM_EXCEEDED

#### 参数变更（相比上一 Trial）

| 参数 | 旧值 | 新值 |
|---|---|---|
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | `2` | `4` |
| `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu` | `2` | `4` |


### 关键指标

| 指标 | 均值 | P95 | 最大值 |
|---|---|---|---|
| 吞吐量 (tok/s) | 1508.9 | 1508.9 | 1508.9 |
| 每步耗时 (s) | 92.5 | 92.5 | 92.5 |
| 生成 tgs | 2100.2 | 2100.2 | 2100.2 |
| Actor MFU | 0.2229 | 0.2229 | 0.2229 |
| **时间瓶颈** | rollout |||

**稳定性指标:**

| 指标 | 均值 | P95 | 最大 |
|---|---|---|---|
| Reward | -0.9922 | -0.9922 | -0.9922 |
| Reward 斜率 | - |||
| Actor PPO KL | 0.00003144 | 0.00003144 | 0.00003144 |
| Actor Entropy | 1.9421 | 1.9421 | 1.9421 |
| Clip Fraction | 0.000011 | 0.000011 | 0.000011 |
| Response Length | 938.0 | 938.0 | 938.0 |

**分阶段耗时、显存与 GPU 利用率:**

| 阶段 | 耗时均值 (s) | 显存均值 (%) | 显存峰值 (%) | GPU 利用率均值 (%) | GPU 利用率 P95 (%) | GPU 利用率峰值 (%) |
|---|---:|---:|---:|---:|---:|---:|
| rollout | 66.5 | 47.2 | 78.1 | 22.6 | 57.0 | 63.0 |
| actor_log_prob | 6.3 | 49.8 | 74.2 | 55.2 | 87.0 | 87.0 |
| ref_log_prob | 4.8 | 55.0 | 99.8 | 50.2 | 94.0 | 95.0 |
| training | 14.7 | 9.1 | 10.4 | 1.0 | 10.0 | 12.0 |

- **相对最高显存阶段**: ref_log_prob （仅表示四阶段中峰值最高，不等于存在显存压力）
- **总体实测峰值**: 99.8%

### 本 Trial 完成后的 Agent 行为

_该 Trial 完成后没有记录 Diagnosis、Proposal 或 Feasibility trace。_

📄 [训练日志](/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/ssh_agent/output/0729_0955_2026/trials/0006/train.log)

---

### Trial 7: stability_tuning

- **结果**: `success` | **完成步数**: 80/80

#### 参数变更（相比上一 Trial）

| 参数 | 旧值 | 新值 |
|---|---|---|
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | `4` | `2` |
| `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu` | `4` | `2` |


### 关键指标

| 指标 | 均值 | P95 | 最大值 |
|---|---|---|---|
| 吞吐量 (tok/s) | 1150.7 | 1766.3 | 1788.5 |
| 每步耗时 (s) | 77.4 | 100.3 | 104.4 |
| 生成 tgs | 1696.4 | 2546.2 | 2591.5 |
| Actor MFU | 0.1558 | 0.2702 | 0.2728 |
| **时间瓶颈** | rollout |||

**稳定性指标:**

| 指标 | 均值 | P95 | 最大 |
|---|---|---|---|
| Reward | -0.9454 | -0.8711 | -0.8516 |
| Reward 斜率 | 0.000488 |||
| Actor PPO KL | 0.00043554 | 0.00765656 | 0.01391117 |
| Actor Entropy | 1.7523 | 2.5456 | 2.7294 |
| Clip Fraction | 0.000675 | 0.002051 | 0.002606 |
| Response Length | 597.7 | 1224.2 | 1237.4 |

**分阶段耗时、显存与 GPU 利用率:**

| 阶段 | 耗时均值 (s) | 显存均值 (%) | 显存峰值 (%) | GPU 利用率均值 (%) | GPU 利用率 P95 (%) | GPU 利用率峰值 (%) |
|---|---:|---:|---:|---:|---:|---:|
| rollout | 52.3 | 68.2 | 78.1 | 35.1 | 60.0 | 67.0 |
| actor_log_prob | 7.2 | 27.4 | 60.5 | 35.3 | 85.0 | 88.0 |
| ref_log_prob | 4.0 | 39.0 | 86.6 | 50.1 | 95.0 | 96.0 |
| training | 13.7 | 34.4 | 61.9 | 32.5 | 85.0 | 92.0 |

- **相对最高显存阶段**: ref_log_prob （仅表示四阶段中峰值最高，不等于存在显存压力）
- **总体实测峰值**: 86.6%

### 本 Trial 完成后的 Agent 行为

_以下行为用于生成 Trial 8 的候选配置。_

#### Proposal Agent 决策

**Proposal 工具调用 (3 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `parameter_understanding` | actor_rollout_ref.actor.entropy_coeff, actor_rollout_ref.actor.kl_loss_coef, actor_rollout_ref.actor.kl_loss_type, actor_rollout_ref.actor.optim.lr, actor_rollout_ref.actor.optim.lr_warmup_steps, actor_rollout_ref.actor.use_kl_loss …共 7 个 | success |
| 2 | `memory_estimator` | ref_trial=7, 预测参数: actor_rollout_ref.rollout.n | success |
| 2 | `query_trial_history` | stage=stability_tuning, limit=10 | success |


- **决策**: `modify`
- **原因**: Trial 7 的 actor_ppo_kl 均值仅 0.00044，max 0.014，表明策略漂移很小；actor_pg_loss 均值 0.00046，clipfrac 均值 0.00067，说明实际更新幅度极低，reward 斜率仅 0.00049。综合推断学习率偏低且 KL 系数偏高抑制了有效学习。将学习率从 1e-6 温和提升至 3e-6，同时将 kl_loss_coef 从 0.003 降至 0.001，以允许更大的策略更新，同时保留 KL 正则防止塌缩。
- **置信度**: 0.75

| 参数 | 旧值 | 新值 | 原因 |
|---|---|---|---|
| `actor_rollout_ref.actor.optim.lr` | `1e-06` | `3e-06` | 当前 lr 导致 PG loss 和 clipfrac 几乎为零，reward 改善缓慢；温和提升至 3e-6 以加大更新幅度，仍在安全范围内。 |
| `actor_rollout_ref.actor.kl_loss_coef` | `0.003` | `0.001` | KL 均值很低但系数偏高可能限制探索；降低至 0.001 可减轻约束，与更高的 lr 配合促进策略改进。 |


#### Feasibility Agent 审查

**审查 #1（attempt=1）**

**Feasibility 工具调用 (1 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `parameter_understanding` | actor_rollout_ref.actor.optim.lr, actor_rollout_ref.actor.kl_loss_coef, actor_rollout_ref.actor.use_kl_loss, actor_rollout_ref.actor.kl_loss_type | success |

- **判定**: `valid`
- **原因**: Both changes are within stability‑stage parameter ranges. The learning‑rate increase (1e‑6 → 3e‑6) is moderate and addresses near‑zero PG‑loss/clipfrac observed in Trial 7. The KL‑coefficient decrease (0.003 → 0.001) is safe given the measured KL mean of ~0.0004. Neither parameter affects GPU memory; the previously validated memory layout remains unchanged.
- **风险**:
  - Larger policy updates may raise KL and clipfrac; real‑sprint monitoring required.
  - If KL grows faster than expected, the lower coefficient may provide insufficient restraint, but the current KL levels suggest adequate headroom.

**预测显存占用 (%):**

| rollout | actor_log_prob | ref_log_prob | training |
|---:|---:|---:|---:|
| - | - | - | - |

📄 [训练日志](/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/ssh_agent/output/0729_0955_2026/trials/0007/train.log)

---

### Trial 8: stability_tuning

- **结果**: `fail` | **完成步数**: 3/80
- **错误类型**: INCOMPLETE_TRAINING

#### 参数变更（相比上一 Trial）

| 参数 | 旧值 | 新值 |
|---|---|---|
| `actor_rollout_ref.actor.kl_loss_coef` | `0.003` | `0.001` |
| `actor_rollout_ref.actor.optim.lr` | `1e-06` | `3e-06` |


### 关键指标

| 指标 | 均值 | P95 | 最大值 |
|---|---|---|---|
| 吞吐量 (tok/s) | 1700.7 | 1865.3 | 1865.3 |
| 每步耗时 (s) | 106.5 | 110.7 | 110.7 |
| 生成 tgs | 2443.9 | 2645.2 | 2645.2 |
| Actor MFU | 0.2559 | 0.2993 | 0.2993 |
| **时间瓶颈** | rollout |||

**稳定性指标:**

| 指标 | 均值 | P95 | 最大 |
|---|---|---|---|
| Reward | -0.9941 | -0.9922 | -0.9922 |
| Reward 斜率 | -0.001953 |||
| Actor PPO KL | 0.00158186 | 0.00314535 | 0.00314535 |
| Actor Entropy | 2.4288 | 2.6755 | 2.6755 |
| Clip Fraction | 0.000074 | 0.000169 | 0.000169 |
| Response Length | 1266.2 | 1458.8 | 1458.8 |

**分阶段耗时、显存与 GPU 利用率:**

| 阶段 | 耗时均值 (s) | 显存均值 (%) | 显存峰值 (%) | GPU 利用率均值 (%) | GPU 利用率 P95 (%) | GPU 利用率峰值 (%) |
|---|---:|---:|---:|---:|---:|---:|
| rollout | 74.1 | 59.6 | 78.1 | 37.8 | 65.0 | 70.0 |
| actor_log_prob | 9.0 | 32.2 | 44.0 | 53.6 | 87.0 | 88.0 |
| ref_log_prob | 5.6 | 45.1 | 95.9 | 49.2 | 96.0 | 96.0 |
| training | 17.6 | 44.2 | 60.2 | 56.7 | 90.0 | 92.0 |

- **相对最高显存阶段**: ref_log_prob （仅表示四阶段中峰值最高，不等于存在显存压力）
- **总体实测峰值**: 95.9%

### 本 Trial 完成后的 Agent 行为

_以下行为用于生成 Trial 9 的候选配置。_

#### Diagnosis Agent 诊断

**Diagnosis 工具调用 (6 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `read_trial_log_excerpt` | {"trial_id": 8, "pattern": "oom|error|traceback|killed|signal|memory|CUDA", "max_lines": 30} | success |
| 2 | `read_trial_log_excerpt` | {"trial_id": 8, "max_lines": 30} | success |
| 3 | `read_trial_log_excerpt` | {"trial_id": 8, "max_lines": 40} | success |
| 4 | `read_trial_log_excerpt` | {"trial_id": 8, "max_lines": 40, "pattern": "step:4|OOM|killed|signal|error|traceback|CUDA|memory"} | success |
| 5 | `read_trial_log_excerpt` | {"trial_id": 8, "max_lines": 10, "pattern": "step:4"} | success |
| 6 | `read_trial_log_excerpt` | {"trial_id": 8, "max_lines": 30, "pattern": "step:5|step:6|step:7|step:8|exit|terminated|killed|OOM|… | success |

- **失败类型**: `MEMORY_HEADROOM_EXCEEDED`
- **训练子阶段**: `ref_log_prob`
- **置信度**: 0.9
- **原因**: ref_log_prob phase memory peaked at 95.89%, exceeding safe headroom and causing the trial to stall or be killed, leading to incomplete training.
- **证据**:
  - error.type is INCOMPLETE_TRAINING with evidence 'completed 3 of 80 updates'
  - memory_by_phase_pct.ref_log_prob.max = 95.89%, p95 = 93.15%
  - max_observed_memory_pct = 95.89%, memory_bottleneck = ref_log_prob
  - no step:4 or later log lines; trial stopped after step 3 without explicit error message

#### Proposal Agent 决策

**Proposal 工具调用 (5 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `parameter_understanding` | actor_rollout_ref.actor.optim.lr, actor_rollout_ref.actor.kl_loss_coef, actor_rollout_ref.actor.entropy_coeff, actor_rollout_ref.rollout.n, actor_rollout_ref.actor.use_kl_loss, actor_rollout_ref.actor.kl_loss_type | success |
| 2 | `memory_estimator` | ref_trial=7, 预测参数: actor_rollout_ref.actor.optim.lr, actor_rollout_ref.actor.kl_loss_coef | success |
| 2 | `query_trial_history` | stage=stability_tuning, limit=5 | success |
| 3 | `memory_estimator` | ref_trial=7, 预测参数: actor_rollout_ref.actor.optim.lr | success |
| 4 | `search_verl_docs` | 查询: "kl_loss_type low_var_kl alternatives" | success |


- **决策**: `modify`
- **原因**: Trial 8 failed because the lr increase (1e-6→3e-6) combined with a reduced kl_loss_coef (0.003→0.001) caused response length to double (597→1266), pushing ref_log_prob memory to 95.89% and killing the trial. To repair, we raise lr more conservatively to 2e-6 while keeping kl_loss_coef at 0.003 to constrain length growth. A tiny entropy coefficient (0.001) is added to encourage exploration without affecting memory.
- **置信度**: 0.65

| 参数 | 旧值 | 新值 | 原因 |
|---|---|---|---|
| `actor_rollout_ref.actor.optim.lr` | `1e-06` | `2e-06` | A moderate increase from 1e-6 to 2e-6 accelerates learning while limiting the response-length explosion seen at 3e-6. |
| `actor_rollout_ref.actor.entropy_coeff` | `0.0` | `0.001` | A small entropy bonus (from 0 to 0.001) encourages policy diversity and may help escape the -0.945 reward plateau. |


#### Feasibility Agent 审查

**审查 #1（attempt=1）**

**Feasibility 工具调用 (3 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `parameter_understanding` | actor_rollout_ref.actor.optim.lr, actor_rollout_ref.actor.entropy_coeff | success |
| 2 | `memory_estimator` | ref_trial=7, 预测参数: actor_rollout_ref.actor.optim.lr, actor_rollout_ref.actor.entropy_coeff | success |
| 2 | `query_trial_history` | stage=stability_tuning, limit=3 | success |

- **判定**: `valid`
- **原因**: Both changes (lr 1e‑6→2e‑6, entropy_coeff 0.0→0.001) are within stability‑stage ranges and do not alter memory‑critical hardware parameters. The memory estimator (anchored to successful Trial 7) projects no pressure increase; ref_log_prob upper bound is 90.58%, below the 92% throughput and 95% resource limits. The slight lr increase addresses the very small PG‑loss/clipfrac of Trial 7, while the small entropy bonus encourages exploration without risking collapse given the unchanged kl_loss_coef of 0.003.
- **风险**:
  - Ref_log_prob upper bound is only 1.42 pp below the 92% throughput limit; real‑sprint monitoring required.
  - Higher lr + entropy bonus may raise KL faster than expected despite the conservative kl_loss_coef; watch actor_ppo_kl.

**预测显存占用 (%):**

| rollout | actor_log_prob | ref_log_prob | training |
|---:|---:|---:|---:|
| 78.09 | 60.49 | 86.58 | 61.94 |

📄 [训练日志](/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/ssh_agent/output/0729_0955_2026/trials/0008/train.log)

---

### Trial 9: stability_tuning

- **结果**: `fail` | **完成步数**: 5/80
- **错误类型**: OOM

#### 参数变更（相比上一 Trial）

| 参数 | 旧值 | 新值 |
|---|---|---|
| `actor_rollout_ref.actor.entropy_coeff` | `0.0` | `0.001` |
| `actor_rollout_ref.actor.kl_loss_coef` | `0.001` | `0.003` |
| `actor_rollout_ref.actor.optim.lr` | `3e-06` | `2e-06` |


### 关键指标

| 指标 | 均值 | P95 | 最大值 |
|---|---|---|---|
| 吞吐量 (tok/s) | 1596.8 | 1627.1 | 1627.1 |
| 每步耗时 (s) | 101.5 | 106.3 | 106.3 |
| 生成 tgs | 2319.6 | 2387.3 | 2387.3 |
| Actor MFU | 0.2261 | 0.2282 | 0.2282 |
| **时间瓶颈** | rollout |||

**稳定性指标:**

| 指标 | 均值 | P95 | 最大 |
|---|---|---|---|
| Reward | -0.9926 | -0.9883 | -0.9883 |
| Reward 斜率 | -0.000488 |||
| Actor PPO KL | 0.00020424 | 0.00027421 | 0.00027421 |
| Actor Entropy | 3.2800 | 3.6541 | 3.6541 |
| Clip Fraction | 0.000072 | 0.000077 | 0.000077 |
| Response Length | 1111.4 | 1142.3 | 1142.3 |

**分阶段耗时、显存与 GPU 利用率:**

| 阶段 | 耗时均值 (s) | 显存均值 (%) | 显存峰值 (%) | GPU 利用率均值 (%) | GPU 利用率 P95 (%) | GPU 利用率峰值 (%) |
|---|---:|---:|---:|---:|---:|---:|
| rollout | 69.9 | 64.4 | 78.1 | 39.8 | 65.0 | 72.0 |
| actor_log_prob | 8.5 | 30.6 | 44.5 | 46.4 | 87.0 | 88.0 |
| ref_log_prob | 5.2 | 48.4 | 98.8 | 51.3 | 96.0 | 97.0 |
| training | 17.6 | 48.6 | 71.0 | 53.9 | 91.0 | 93.0 |

- **相对最高显存阶段**: ref_log_prob （仅表示四阶段中峰值最高，不等于存在显存压力）
- **总体实测峰值**: 98.8%

### 本 Trial 完成后的 Agent 行为

_这是最后一个 Trial，记录中没有后续 Agent trace。_

📄 [训练日志](/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/ssh_agent/output/0729_0955_2026/trials/0009/train.log)

---
