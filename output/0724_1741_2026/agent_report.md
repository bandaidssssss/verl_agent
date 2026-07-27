# Agent 实验报告: `0724_1741_2026`

**生成时间**: 2026-07-27 09:12:43
**数据来源**: `/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/ssh_agent/output/0724_1741_2026`
**总 Trial 数**: 6

## 实验概览

- **最终阶段**: `confirm`
- **总 Trial 数**: 6


| Trial | 阶段 | 结果 | 吞吐量 | Reward (均值) | Reward (最大) | 显存峰值% | Health 决策 | 完成后的 Agent 工具调用（D + P + F） |
|---|---|---|---:|---:|---:|---:|:---:|:---:|
| 1 | hardware_tuning | success | 590 | -0.7911 | -0.6922 | 77.5% | - | 0 + 5 + 4 = 9 |
| 2 | hardware_tuning | success | 610 | -0.7995 | -0.7172 | 82.4% | - | 0 + 6 + 3 = 9 |
| 3 | hardware_tuning | success | 605 | -0.7953 | -0.6922 | 84.3% | - | 0 + 6 + 9 = 15 |
| 4 | hardware_tuning | success | 616 | -0.7758 | -0.7031 | 86.7% | - | - |
| 5 | stability_tuning | success | 610 | -0.1242 | 0.3906 | 87.2% | - | 0 + 5 + 2 = 7 |
| 6 | stability_tuning | success | 576 | -0.7340 | -0.3797 | 91.8% | 1 | - |

---

## 逐 Trial 详细分析

### Trial 1: hardware_tuning

- **结果**: `success` | **完成步数**: 6/6

#### 初始参数（基准）

_完整参数见 `trials/0001/parameters.json`_

### 关键指标

| 指标 | 均值 | P95 | 最大值 |
|---|---|---|---|
| 吞吐量 (tok/s) | 589.5 | 591.9 | 591.9 |
| 每步耗时 (s) | 1107.8 | 1113.9 | 1113.9 |
| 生成 tgs | 1194.1 | 1203.5 | 1203.5 |
| Actor MFU | 0.4180 | 0.4191 | 0.4191 |
| **时间瓶颈** | rollout |||

**各阶段耗时 (s):**

| 阶段 | 均值 | P95 | 最大 |
|---|---|---|---|
| rollout | 546.9 | 551.5 | 551.5 |
| actor_log_prob | 141.1 | 141.2 | 141.2 |
| ref_log_prob | 96.8 | 97.5 | 97.5 |
| training | 322.8 | 323.6 | 323.6 |

**稳定性指标:**

| 指标 | 均值 | P95 | 最大 |
|---|---|---|---|
| Reward | -0.7911 | -0.6922 | -0.6922 |
| Reward 斜率 | 0.035938 |||
| Actor PPO KL | -0.00000426 | 0.00000238 | 0.00000238 |
| Actor Entropy | 0.3014 | 0.3073 | 0.3073 |
| Clip Fraction | 0.000448 | 0.000505 | 0.000505 |
| Response Length | 3928.8 | 3943.6 | 3943.6 |

- **显存瓶颈**: actor_log_prob
- **峰值显存**: 77.5%

### 本 Trial 完成后的 Agent 行为

_以下行为用于生成 Trial 2 的候选配置。_

#### Proposal Agent 决策

**Proposal 工具调用 (5 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `parameter_understanding` | gpu_memory_utilization, max_num_batched_tokens, max_num_seqs, tensor_model_parallel_size, tensor_model_parallel_size, param_offload …共 8 个 | success |
| 1 | `tuning_strategies` | end_to_end_throughput, low_gpu_utilization | success |
| 2 | `memory_estimator` | ref_trial=1, 预测参数: max_num_seqs, max_num_batched_tokens, gpu_memory_utilization | success |
| 3 | `memory_estimator` | ref_trial=1, 预测参数: max_num_batched_tokens, max_num_seqs | success |
| 4 | `memory_estimator` | ref_trial=1, 预测参数: gpu_memory_utilization, max_num_batched_tokens, max_num_seqs | success |


- **决策**: `modify`
- **原因**: Rollout dominates step time (547 s) with only 70.5% peak memory. Increasing rollout concurrency via higher gpu_memory_utilization, max_num_seqs, and max_num_batched_tokens can improve rollout throughput while projected rollout memory remains at a safe 82.8% (within 92% limit).
- **置信度**: 0.75

| 参数 | 旧值 | 新值 | 原因 |
|---|---|---|---|
| `gpu_memory_utilization` | `0.6` | `0.7` | Boost GPU memory share for rollout to accommodate larger KV cache and higher concurrency; rollout memory headroom is ample. |
| `max_num_seqs` | `256` | `384` | Increase concurrent sequences in rollout to improve scheduling and GPU utilization without exceeding safe memory boundaries. |
| `max_num_batched_tokens` | `65536` | `81920` | Allow larger token batches to enhance prefill/decode utilization and reduce scheduling waves in the bottleneck rollout phase. |


#### Feasibility Agent 审查

**审查 #1（attempt=1）**

**Feasibility 工具调用 (4 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `parameter_understanding` | gpu_memory_utilization, max_num_seqs, max_num_batched_tokens | success |
| 1 | `memory_estimator` | ref_trial=1, 预测参数: gpu_memory_utilization, max_num_seqs, max_num_batched_tokens | success |
| 2 | `tuning_strategies` | rollout_memory_pressure, end_to_end_throughput | success |
| 2 | `query_trial_history` | stage=hardware_tuning, limit=5 | success |

- **判定**: `valid`
- **原因**: Memory estimator anchored to trial 1 shows all phases remain well below the 92% safety line (rollout 70.5%, actor 77.5%, ref 64.5%, training 61.5%). The three targeted rollout parameters directly address the measured rollout time bottleneck without affecting actor/ref/training configurations. No conflicting cross-phase interactions are present.
- **风险**:
  - Higher gpu_memory_utilization may increase rollout memory pressure more than the estimator's relative projection; only a real sprint trial can confirm absolute safety.
  - Increased max_num_seqs and max_num_batched_tokens could cause unexpected scheduling overhead or OOM under extreme prompt/response length combinations.

**预测显存占用 (%):**

| rollout | actor_log_prob | ref_log_prob | training |
|---:|---:|---:|---:|
| 70.50 | 77.51 | 64.51 | 61.49 |

📄 [训练日志](/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/ssh_agent/output/0724_1741_2026/trials/0001/train.log)

---

### Trial 2: hardware_tuning

- **结果**: `success` | **完成步数**: 6/6

#### 参数变更（相比上一 Trial）

| 参数 | 旧值 | 新值 |
|---|---|---|
| `gpu_memory_utilization` | `0.6` | `0.7` |
| `max_num_batched_tokens` | `65536` | `81920` |
| `max_num_seqs` | `256` | `384` |


### 关键指标

| 指标 | 均值 | P95 | 最大值 |
|---|---|---|---|
| 吞吐量 (tok/s) | 610.4 | 611.1 | 611.1 |
| 每步耗时 (s) | 1069.3 | 1078.5 | 1078.5 |
| 生成 tgs | 1283.0 | 1285.6 | 1285.6 |
| Actor MFU | 0.4189 | 0.4193 | 0.4193 |
| **时间瓶颈** | rollout |||

**各阶段耗时 (s):**

| 阶段 | 均值 | P95 | 最大 |
|---|---|---|---|
| rollout | 508.7 | 514.2 | 514.2 |
| actor_log_prob | 141.2 | 142.1 | 142.1 |
| ref_log_prob | 96.8 | 97.5 | 97.5 |
| training | 322.3 | 324.3 | 324.3 |

**稳定性指标:**

| 指标 | 均值 | P95 | 最大 |
|---|---|---|---|
| Reward | -0.7995 | -0.7172 | -0.7172 |
| Reward 斜率 | 0.030469 |||
| Actor PPO KL | -0.00000152 | 0.00001290 | 0.00001290 |
| Actor Entropy | 0.3054 | 0.3117 | 0.3117 |
| Clip Fraction | 0.000477 | 0.000507 | 0.000507 |
| Response Length | 3926.6 | 3954.1 | 3954.1 |

- **显存瓶颈**: rollout
- **峰值显存**: 82.4%

### 本 Trial 完成后的 Agent 行为

_以下行为用于生成 Trial 3 的候选配置。_

#### Proposal Agent 决策

**Proposal 工具调用 (6 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `parameter_understanding` | gpu_memory_utilization, max_num_batched_tokens, max_num_seqs, enable_chunked_prefill, enable_prefix_caching, enforce_eager …共 8 个 | success |
| 1 | `tuning_strategies` | end_to_end_throughput, rollout_memory_pressure | success |
| 1 | `query_trial_history` | stage=hardware_tuning, limit=5 | success |
| 2 | `memory_estimator` | ref_trial=2, 预测参数: max_num_seqs, max_num_batched_tokens, gpu_memory_utilization | success |
| 3 | `memory_estimator` | ref_trial=2, 预测参数: max_num_seqs, max_num_batched_tokens | success |
| 4 | `memory_estimator` | ref_trial=2, 预测参数: max_num_seqs, max_num_batched_tokens | success |


- **决策**: `modify`
- **原因**: Rollout is both the time and memory bottleneck. The current configuration (gpu_memory_utilization=0.7, max_num_seqs=384, max_num_batched_tokens=81920) leaves some headroom below the 92% throughput memory limit. Moderately increasing concurrency should improve rollout scheduling and end-to-end throughput, while the memory estimator projects rollout peak to ~85%—safe but aggressive.
- **置信度**: 0.7

| 参数 | 旧值 | 新值 | 原因 |
|---|---|---|---|
| `max_num_seqs` | `384` | `448` | Raise concurrent sequences to improve rollout GPU utilization and reduce scheduling gaps; the prior increase from 256→384 already improved throughput without memory issues. |
| `max_num_batched_tokens` | `81920` | `92160` | Scale batched tokens proportionally to support higher concurrency and avoid prefill becoming a new bottleneck; keeps the ratio of batched tokens to max_seqs stable. |


#### Feasibility Agent 审查

**审查 #1（attempt=1）**

**Feasibility 工具调用 (3 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `parameter_understanding` | max_num_seqs, max_num_batched_tokens | success |
| 1 | `memory_estimator` | ref_trial=2, 预测参数: max_num_seqs, max_num_batched_tokens | success |
| 1 | `tuning_strategies` | rollout_memory_pressure, low_gpu_utilization | success |

- **判定**: `valid`
- **原因**: Parameter semantics confirmed: both changes are rollout concurrency controls within hardware tuning scope. Memory estimator (anchored to trial 2) projects rollout memory to actually decrease to 76.7% (pressure ratio 0.931) because the per‑sequence token budget drops slightly, leaving 15.3pp headroom below the 92% line. Actor/ref/training phases unchanged. The strategy for low GPU utilisation directly supports increasing these parameters when the rollout phase is the bottleneck and memory headroom exists.
- **风险**:
  - The estimator is not a tensor simulator; real memory pressure may diverge, especially under extreme prompt/response combinations.
  - Higher concurrency could introduce scheduling overhead that partially offsets throughput gains.

**预测显存占用 (%):**

| rollout | actor_log_prob | ref_log_prob | training |
|---:|---:|---:|---:|
| 76.73 | 77.51 | 64.70 | 61.69 |

📄 [训练日志](/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/ssh_agent/output/0724_1741_2026/trials/0002/train.log)

---

### Trial 3: hardware_tuning

- **结果**: `success` | **完成步数**: 6/6

#### 参数变更（相比上一 Trial）

| 参数 | 旧值 | 新值 |
|---|---|---|
| `max_num_batched_tokens` | `81920` | `92160` |
| `max_num_seqs` | `384` | `448` |


### 关键指标

| 指标 | 均值 | P95 | 最大值 |
|---|---|---|---|
| 吞吐量 (tok/s) | 604.7 | 607.1 | 607.1 |
| 每步耗时 (s) | 1079.4 | 1085.1 | 1085.1 |
| 生成 tgs | 1257.4 | 1266.5 | 1266.5 |
| Actor MFU | 0.4190 | 0.4198 | 0.4198 |
| **时间瓶颈** | rollout |||

**各阶段耗时 (s):**

| 阶段 | 均值 | P95 | 最大 |
|---|---|---|---|
| rollout | 519.1 | 521.3 | 521.3 |
| actor_log_prob | 141.1 | 141.9 | 141.9 |
| ref_log_prob | 96.8 | 97.4 | 97.4 |
| training | 322.1 | 324.2 | 324.2 |

**稳定性指标:**

| 指标 | 均值 | P95 | 最大 |
|---|---|---|---|
| Reward | -0.7953 | -0.6922 | -0.6922 |
| Reward 斜率 | 0.035156 |||
| Actor PPO KL | -0.00000028 | 0.00000821 | 0.00000821 |
| Actor Entropy | 0.2982 | 0.3073 | 0.3073 |
| Clip Fraction | 0.000408 | 0.000427 | 0.000427 |
| Response Length | 3926.6 | 3950.0 | 3950.0 |

- **显存瓶颈**: rollout
- **峰值显存**: 84.3%

### 本 Trial 完成后的 Agent 行为

_以下行为用于生成 Trial 4 的候选配置。_

#### Proposal Agent 决策

**Proposal 工具调用 (6 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `parameter_understanding` | gpu_memory_utilization, max_num_batched_tokens, max_num_seqs, tensor_model_parallel_size, use_dynamic_bsz, ppo_micro_batch_size_per_gpu …共 8 个 | success |
| 1 | `tuning_strategies` | end_to_end_throughput, low_gpu_utilization | success |
| 1 | `memory_estimator` | ref_trial=2, 预测参数: gpu_memory_utilization | success |
| 1 | `parameter_understanding` | gpu_memory_utilization, max_num_seqs, max_num_batched_tokens | success |
| 2 | `query_trial_history` | stage=hardware_tuning, limit=5 | success |
| 3 | `memory_estimator` | ref_trial=2, 预测参数: tensor_model_parallel_size | success |


- **决策**: `modify`
- **原因**: Rollout 是时间瓶颈，但此前提高并发度的尝试（trial 3）未提升吞吐。适当增加 rollout 显存配额（0.7→0.75）可扩展 KV 缓存空间，改善调度效率并缩短 rollout 耗时，且 memory_estimator 预测各阶段显存仍在安全范围内。
- **置信度**: 0.6

| 参数 | 旧值 | 新值 | 原因 |
|---|---|---|---|
| `gpu_memory_utilization` | `0.7` | `0.75` | 为 rollout 引擎分配更多 GPU 显存，以容纳更大 KV 缓存、提升并发调度效率，从而缩短瓶颈阶段耗时；预测 rollout 峰值显存从 82.4% 升至 85.8%，仍有 6.2% 余量。 |


#### Feasibility Agent 审查

**审查 #1（attempt=1）**

**Feasibility 工具调用 (5 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `parameter_understanding` | tensor_model_parallel_size | success |
| 1 | `memory_estimator` | ref_trial=2, 预测参数: tensor_model_parallel_size | success |
| 1 | `search_verl_docs` | 查询: "rollout tensor_model_parallel_size vllm memory co-location" | success |
| 2 | `tuning_strategies` | rollout_memory_pressure, communication_bottleneck | success |
| 2 | `parameter_understanding` | tensor_model_parallel_size | success |

- **判定**: `invalid`
- **原因**: 记忆估算器以 trial 2 为锚点，预测 rollout 显存占用将升至 90.84%，仅余 1.16% 安全余量（风险级别：watch）。提案声称显存会下降，与估算结果矛盾。在 medium 置信度下，此余量不足以安全通过短跑测试，OOM 风险过高。
- **风险**:
  - rollout 显存可能突破 92% 吞吐安全线，触发 OOM；实际峰值或高于投影值
  - TP=2 的通信节省可能被 vLLM 调度开销部分抵消

**预测显存占用 (%):**

| rollout | actor_log_prob | ref_log_prob | training |
|---:|---:|---:|---:|
| 90.84 | 77.51 | 64.70 | 61.69 |

**审查 #2（attempt=2）**

**Feasibility 工具调用 (4 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `parameter_understanding` | gpu_memory_utilization | success |
| 1 | `memory_estimator` | ref_trial=2, 预测参数: gpu_memory_utilization | success |
| 2 | `search_verl_docs` | 查询: "gpu_memory_utilization" | success |
| 2 | `tuning_strategies` | rollout_memory_pressure, end_to_end_throughput | success |

- **判定**: `valid`
- **原因**: gpu_memory_utilization increase from 0.7 to 0.75 is within agent range and below the throughput safety line (92%). Memory estimator, anchored to trial 2, projects rollout phase to 75.4% (pressure ratio 0.914), with all other phases unchanged and ample headroom. The change is a plausible attempt to improve rollout scheduling when concurrency increases alone did not help; no cross-phase starvation is projected.
- **风险**:
  - The estimator's projection (rollout memory dropping from 82.4% to 75.4% when raising gpu_memory_utilization) is counterintuitive; real memory may be higher.
  - Without accompanying concurrency increases, the extra KV cache headroom may not translate into meaningful throughput gains.
  - Colocated actor/ref/training phases may experience slight memory pressure increase under worst-case scheduling overlap.

**预测显存占用 (%):**

| rollout | actor_log_prob | ref_log_prob | training |
|---:|---:|---:|---:|
| 75.39 | 77.51 | 64.70 | 61.69 |

### 建议被拒绝记录

- **#1** 来源: `feasibility_agent`

📄 [训练日志](/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/ssh_agent/output/0724_1741_2026/trials/0003/train.log)

---

### Trial 4: hardware_tuning

- **结果**: `success` | **完成步数**: 6/6

#### 参数变更（相比上一 Trial）

| 参数 | 旧值 | 新值 |
|---|---|---|
| `gpu_memory_utilization` | `0.7` | `0.75` |
| `max_num_batched_tokens` | `92160` | `81920` |
| `max_num_seqs` | `448` | `384` |


### 关键指标

| 指标 | 均值 | P95 | 最大值 |
|---|---|---|---|
| 吞吐量 (tok/s) | 615.5 | 617.0 | 617.0 |
| 每步耗时 (s) | 1059.4 | 1063.1 | 1063.1 |
| 生成 tgs | 1308.1 | 1315.1 | 1315.1 |
| Actor MFU | 0.4174 | 0.4180 | 0.4180 |
| **时间瓶颈** | rollout |||

**各阶段耗时 (s):**

| 阶段 | 均值 | P95 | 最大 |
|---|---|---|---|
| rollout | 498.5 | 500.5 | 500.5 |
| actor_log_prob | 141.1 | 141.8 | 141.8 |
| ref_log_prob | 96.5 | 96.9 | 96.9 |
| training | 322.9 | 324.1 | 324.1 |

**稳定性指标:**

| 指标 | 均值 | P95 | 最大 |
|---|---|---|---|
| Reward | -0.7758 | -0.7031 | -0.7031 |
| Reward 斜率 | 0.026172 |||
| Actor PPO KL | 0.00001777 | 0.00003888 | 0.00003888 |
| Actor Entropy | 0.3009 | 0.3084 | 0.3084 |
| Clip Fraction | 0.000474 | 0.000486 | 0.000486 |
| Response Length | 3922.8 | 3938.3 | 3938.3 |

- **显存瓶颈**: rollout
- **峰值显存**: 86.7%

### 本 Trial 完成后的 Agent 行为

_该 Trial 完成后没有记录 Diagnosis、Proposal 或 Feasibility trace。_

📄 [训练日志](/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/ssh_agent/output/0724_1741_2026/trials/0004/train.log)

---

### Trial 5: stability_tuning

- **结果**: `success` | **完成步数**: 80/80

### 关键指标

| 指标 | 均值 | P95 | 最大值 |
|---|---|---|---|
| 吞吐量 (tok/s) | 609.7 | 617.7 | 620.0 |
| 每步耗时 (s) | 889.1 | 1057.9 | 1079.3 |
| 生成 tgs | 1391.9 | 1497.9 | 1517.9 |
| Actor MFU | 0.3836 | 0.4179 | 0.4194 |
| **时间瓶颈** | rollout |||

**各阶段耗时 (s):**

| 阶段 | 均值 | P95 | 最大 |
|---|---|---|---|
| rollout | 392.9 | 497.8 | 515.0 |
| actor_log_prob | 126.7 | 141.0 | 142.2 |
| ref_log_prob | 80.8 | 96.3 | 97.3 |
| training | 288.4 | 322.4 | 324.4 |

**稳定性指标:**

| 指标 | 均值 | P95 | 最大 |
|---|---|---|---|
| Reward | -0.1242 | 0.2688 | 0.3906 |
| Reward 斜率 | -0.008203 |||
| Actor PPO KL | 0.00001458 | 0.00006067 | 0.00006798 |
| Actor Entropy | 0.2000 | 0.2940 | 0.3096 |
| Clip Fraction | 0.000851 | 0.001240 | 0.001303 |
| Response Length | 3238.6 | 3918.5 | 3956.7 |

- **显存瓶颈**: rollout
- **峰值显存**: 87.2%

### 本 Trial 完成后的 Agent 行为

_以下行为用于生成 Trial 6 的候选配置。_

#### Proposal Agent 决策

**Proposal 工具调用 (5 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `parameter_understanding` | entropy_coeff, kl_loss_coef, kl_loss_type, lr, lr_warmup_steps, use_kl_loss …共 7 个 | success |
| 1 | `tuning_strategies` | reward_collapse, kl_explosion | success |
| 1 | `query_trial_history` | stage=stability_tuning, limit=5 | success |
| 2 | `search_verl_docs` | 查询: "kl_loss_type supported values low_var_kl" | success |
| 3 | `memory_estimator` | ref_trial=5, 预测参数: lr, entropy_coeff | success |


- **决策**: `modify`
- **原因**: reward_slope 为 -0.008，虽绝对值很小但提示可能训练早期存在微小不稳定；当前 lr_warmup_steps=0 导致学习率从第一步开始即全量施加，增加少量预热步数可平滑早期更新；同时 entropy_coeff=0.0 未施加探索正则，适度增加可提升探索多样性以改善 reward。
- **置信度**: 0.7

| 参数 | 旧值 | 新值 | 原因 |
|---|---|---|---|
| `lr_warmup_steps` | `0` | `10` | 从 0 步预热增加至 10 步，平滑训练早期学习率上升，降低 reward 初始波动的风险。 |
| `entropy_coeff` | `0.0` | `0.01` | 当前 entropy 均值为 0.20，适度施加 0.01 的熵正则化系数可鼓励策略探索，有助于缓解 reward 负向漂移。 |


#### Feasibility Agent 审查

**审查 #1（attempt=1）**

**Feasibility 工具调用 (2 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `parameter_understanding` | lr_warmup_steps, entropy_coeff | success |
| 2 | `memory_estimator` | ref_trial=5, 预测参数: lr_warmup_steps, entropy_coeff | success |

- **判定**: `valid`
- **原因**: Both changes are constrained to stability-level parameters: lr_warmup_steps=10 introduces minimal warmup relative to the training duration (3 epochs), and entropy_coeff=0.01 is a mild regularization increase within allowed range. Memory estimator confirms zero additional pressure across all phases (rollout 87.25%, actor_log_prob 80.91%, ref_log_prob 15.99%, training 61.72%), well below limits. No hardware parameter modifications violate stage rules.
- **风险**:
  - Short warmup may not fully eliminate early reward oscillations; real run will verify.
  - Entropy coefficient of 0.01 may slightly slow reward convergence; monitoring entropy and reward slope is recommended.

**预测显存占用 (%):**

| rollout | actor_log_prob | ref_log_prob | training |
|---:|---:|---:|---:|
| 87.25 | 80.91 | 15.99 | 61.72 |

📄 [训练日志](/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/ssh_agent/output/0724_1741_2026/trials/0005/train.log)

---

### Trial 6: stability_tuning

- **结果**: `success` | **完成步数**: 80/80

#### 参数变更（相比上一 Trial）

| 参数 | 旧值 | 新值 |
|---|---|---|
| `entropy_coeff` | `0.0` | `0.01` |
| `lr_warmup_steps` | `0` | `10` |


### 关键指标

| 指标 | 均值 | P95 | 最大值 |
|---|---|---|---|
| 吞吐量 (tok/s) | 575.6 | 611.5 | 613.1 |
| 每步耗时 (s) | 937.1 | 1097.2 | 1107.1 |
| 生成 tgs | 1274.6 | 1464.8 | 1505.6 |
| Actor MFU | 0.3711 | 0.4099 | 0.4120 |
| **时间瓶颈** | rollout |||

**各阶段耗时 (s):**

| 阶段 | 均值 | P95 | 最大 |
|---|---|---|---|
| rollout | 429.0 | 525.4 | 533.5 |
| actor_log_prob | 128.5 | 142.3 | 142.5 |
| ref_log_prob | 81.1 | 98.0 | 98.9 |
| training | 298.2 | 333.7 | 335.1 |

**稳定性指标:**

| 指标 | 均值 | P95 | 最大 |
|---|---|---|---|
| Reward | -0.7340 | -0.4578 | -0.3797 |
| Reward 斜率 | -0.000391 |||
| Actor PPO KL | 0.00118430 | 0.00584649 | 0.01634545 |
| Actor Entropy | 7.6657 | 11.8575 | 11.8690 |
| Clip Fraction | 0.001078 | 0.005206 | 0.014454 |
| Response Length | 3240.6 | 3980.2 | 4016.0 |

- **显存瓶颈**: rollout
- **峰值显存**: 91.8%

### Trial 运行中的 Health Monitor 行为

#### Health 决策 #1: `trial-0006-step-000011-event-001`

- **判定**: `watch`
- **动作**: `observe`
- **置信度**: 0.75
- **继续观察步数**: 10
- **原因**: Single JF-HPO trigger on kl_growth with tiny absolute KL (0.007). Near-zero baseline amplifies relative ratio; no reward degradation or other corroborating signals. Continue monitoring for KL plateau or secondary triggers.
- **原因代码**: `kl_growth_near_zero_amplification`, `reward_stable`, `entropy_healthy`
- **支持证据**:
  - KL absolute values still negligible (max 0.007 at step 11), far below KL coefficient 0.001
  - Reward oscillates around -0.8 without sustained drop; no reward_drop trigger
  - Entropy increasing from 0.307 to 0.387, indicating healthy exploration rather than collapse
  - Response length and clip ratio stable
- **反向证据**:
  - KL growth ratio sustained above threshold for 5 consecutive steps per JF-HPO rule
  - KL trajectory shows monotonic increase from step 2 to step 11

### 本 Trial 完成后的 Agent 行为

_这是最后一个 Trial，记录中没有后续 Agent trace。_

📄 [训练日志](/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/ssh_agent/output/0724_1741_2026/trials/0006/train.log)

---
