# Agent 实验报告: `0723_1550_2026`


**生成时间**: 2026-07-24 15:53:25
**数据来源**: `/Users/noone/Desktop/share/ssh_agent/output/0723_1550_2026`
**总 Trial 数**: 6

## 实验概览

- **最终阶段**: `stability_tuning`
- **总 Trial 数**: 6


| Trial | 阶段 | 结果 | 吞吐量 | Reward (均值) | Reward (最大) | 显存峰值% | Health 决策 | 完成后的 Agent 工具调用（D + P + F） |
|---|---|---|---:|---:|---:|---:|:---:|:---:|
| 1 | hardware_tuning | success | 1313 | -0.9922 | -0.9883 | 65.6% | - | 0 + 7 + 2 = 9 |
| 2 | hardware_tuning | success | 1486 | -0.9886 | -0.9844 | 79.1% | - | 0 + 5 + 2 = 7 |
| 3 | hardware_tuning | success | 1052 | -0.9886 | -0.9844 | 89.0% | - | 0 + 6 + 2 = 8 |
| 4 | hardware_tuning | success | 1275 | -0.9906 | -0.9824 | 79.8% | - | - |
| 5 | stability_tuning | fail | 389 | -0.9141 | -0.8516 | 79.1% | 1 | 4 + 3 + 3 = 10 |
| 6 | stability_tuning | fail | 668 | -0.9451 | -0.8672 | 79.1% | - | - |

---

## 逐 Trial 详细分析

### Trial 1: hardware_tuning

- **结果**: `success` | **完成步数**: 6/6

#### 初始参数（基准）

_完整参数见 `trials/0001/parameters.json`_

### 关键指标

| 指标 | 均值 | P95 | 最大值 |
|---|---|---|---|
| 吞吐量 (tok/s) | 1312.8 | 1312.8 | 1312.8 |
| 每步耗时 (s) | 131.1 | 131.1 | 131.1 |
| 生成 tgs | 2209.5 | 2209.5 | 2209.5 |
| Actor MFU | 0.1318 | 0.1318 | 0.1318 |
| **时间瓶颈** | rollout |||

**各阶段耗时 (s):**

| 阶段 | 均值 | P95 | 最大 |
|---|---|---|---|
| rollout | 77.9 | 77.9 | 77.9 |
| actor_log_prob | 14.9 | 14.9 | 14.9 |
| ref_log_prob | 6.7 | 6.7 | 6.7 |
| training | 31.3 | 31.3 | 31.3 |

**稳定性指标:**

| 指标 | 均值 | P95 | 最大 |
|---|---|---|---|
| Reward | -0.9922 | -0.9883 | -0.9883 |
| Reward 斜率 | 0.001953 |||
| Actor PPO KL | 0.00015274 | 0.00015274 | 0.00015274 |
| Actor Entropy | 2.5200 | 2.5200 | 2.5200 |
| Clip Fraction | 0.000019 | 0.000019 | 0.000019 |
| Response Length | 1188.6 | 1188.6 | 1188.6 |

- **显存瓶颈**: ref_log_prob
- **峰值显存**: 65.6%

### 本 Trial 完成后的 Agent 行为

_以下行为用于生成 Trial 2 的候选配置。_

#### Proposal Agent 决策

**Proposal 工具调用 (7 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `parameter_understanding` | tensor_model_parallel_size, tensor_model_parallel_size, gpu_memory_utilization, ppo_micro_batch_size_per_gpu, ppo_mini_batch_size, enable_chunked_prefill …共 8 个 | success |
| 1 | `tuning_strategies` | end_to_end_throughput, rollout_memory_pressure | success |
| 2 | `memory_estimator` | ref_trial=1, 预测参数: gpu_memory_utilization, free_cache_engine | success |
| 3 | `memory_estimator` | ref_trial=1, 预测参数: gpu_memory_utilization | success |
| 4 | `parameter_understanding` | max_num_batched_tokens, max_num_seqs, enforce_eager | success |
| 4 | `tuning_strategies` | low_gpu_utilization | success |
| 5 | `memory_estimator` | ref_trial=1, 预测参数: gpu_memory_utilization, max_num_batched_tokens | success |


- **决策**: `modify`
- **原因**: Rollout is the time bottleneck (77.88 s, 59.4 % of step time) while its peak memory is low (59.15 %). Increasing gpu_memory_utilization from 0.5 to 0.7 allows the scheduler to reserve more KV-cache space, enabling higher concurrency and generation throughput. The memory estimator projects rollout at 71.89 % with ample headroom to the 92 % limit, and other phases are unchanged.
- **置信度**: 0.8

| 参数 | 旧值 | 新值 | 原因 |
|---|---|---|---|
| `gpu_memory_utilization` | `0.5` | `0.7` | Relieves the rollout time bottleneck by granting more KV-cache headroom, directly improving generation throughput without endangering other phases. |


#### Feasibility Agent 审查

**审查 #1（attempt=1）**

**Feasibility 工具调用 (2 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `parameter_understanding` | gpu_memory_utilization | success |
| 1 | `memory_estimator` | ref_trial=1, 预测参数: gpu_memory_utilization | success |

- **判定**: `valid`
- **原因**: Rollout is the time bottleneck; increasing gpu_memory_utilization from 0.5 to 0.7 projects rollout memory to 65.74% (well below 92% safety line) without endangering other phases. Actor, ref, and training memory are unchanged. The change is semantically appropriate for hardware tuning.
- **风险**:
  - Projection is medium-confidence; real run may push rollout memory slightly higher, but large headroom exists.
  - Increased KV cache may marginally slow actor if GPU allocator fragments, but given 26% headroom this is unlikely.

**预测显存占用 (%):**

| rollout | actor_log_prob | ref_log_prob | training |
|---:|---:|---:|---:|
| 65.74 | 31.22 | 65.56 | 38.10 |

📄 [训练日志](/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/ssh_agent/output/0723_1550_2026/trials/0001/train.log)

---

### Trial 2: hardware_tuning

- **结果**: `success` | **完成步数**: 6/6

#### 参数变更（相比上一 Trial）

| 参数 | 旧值 | 新值 |
|---|---|---|
| `gpu_memory_utilization` | `0.5` | `0.7` |


### 关键指标

| 指标 | 均值 | P95 | 最大值 |
|---|---|---|---|
| 吞吐量 (tok/s) | 1485.5 | 1485.5 | 1485.5 |
| 每步耗时 (s) | 123.8 | 123.8 | 123.8 |
| 生成 tgs | 2575.4 | 2575.4 | 2575.4 |
| Actor MFU | 0.1383 | 0.1383 | 0.1383 |
| **时间瓶颈** | rollout |||

**各阶段耗时 (s):**

| 阶段 | 均值 | P95 | 最大 |
|---|---|---|---|
| rollout | 71.4 | 71.4 | 71.4 |
| actor_log_prob | 14.9 | 14.9 | 14.9 |
| ref_log_prob | 6.8 | 6.8 | 6.8 |
| training | 30.4 | 30.4 | 30.4 |

**稳定性指标:**

| 指标 | 均值 | P95 | 最大 |
|---|---|---|---|
| Reward | -0.9886 | -0.9844 | -0.9844 |
| Reward 斜率 | 0.000488 |||
| Actor PPO KL | 0.00007245 | 0.00007245 | 0.00007245 |
| Actor Entropy | 0.8194 | 0.8194 | 0.8194 |
| Clip Fraction | 0.000031 | 0.000031 | 0.000031 |
| Response Length | 1280.7 | 1280.7 | 1280.7 |

- **显存瓶颈**: rollout
- **峰值显存**: 79.1%

### 本 Trial 完成后的 Agent 行为

_以下行为用于生成 Trial 3 的候选配置。_

#### Proposal Agent 决策

**Proposal 工具调用 (5 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `parameter_understanding` | gpu_memory_utilization, enable_chunked_prefill, enable_prefix_caching, tensor_model_parallel_size, max_num_seqs, max_num_batched_tokens | success |
| 2 | `tuning_strategies` | end_to_end_throughput, rollout_memory_pressure | success |
| 3 | `search_verl_docs` | 查询: "max_num_batched_tokens default value" | success |
| 3 | `search_verl_docs` | 查询: "max_num_seqs default value" | success |
| 4 | `memory_estimator` | ref_trial=2, 预测参数: gpu_memory_utilization | success |


- **决策**: `modify`
- **原因**: rollout 阶段时间占比最高且显存峰值 79%，低于 92% 安全线；memory_estimator 投影 gpu_memory_utilization=0.8 后 rollout 显存约 85.4%，风险低，有望通过更多 KV 缓存提升并发与吞吐。
- **置信度**: 0.8

| 参数 | 旧值 | 新值 | 原因 |
|---|---|---|---|
| `gpu_memory_utilization` | `0.7` | `0.8` | 提高 vLLM 可用显存比例，增加 KV 缓存容量，缓解 rollout 瓶颈；历史相似变更已验证收益。 |


#### Feasibility Agent 审查

**审查 #1（attempt=1）**

**Feasibility 工具调用 (2 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `parameter_understanding` | gpu_memory_utilization | success |
| 1 | `memory_estimator` | ref_trial=2, 预测参数: gpu_memory_utilization | success |

- **判定**: `valid`
- **原因**: Rollout is the time bottleneck (71.39 s vs total 123.77 s per step). Raising gpu_memory_utilization from 0.7 to 0.8 is semantically appropriate for hardware tuning and targets the bottleneck. The memory estimator, anchored to trial 2, projects rollout at 72.3% (19.7 pp headroom below the 92% safety line) with no pressure on actor-log-prob (30.73%), ref-log-prob (59.71%), or training (37.92%). Other parameters are unchanged, so cross-phase risk is low.
- **风险**:
  - The memory estimator provides relative pressure, not a tensor-level simulation; a real short-run gate will confirm the projection.
  - At gpu_memory_utilization=0.8 the rollout engine may still compete with actor/ref/training for fragmented memory, but current headroom is large enough to absorb this.

**预测显存占用 (%):**

| rollout | actor_log_prob | ref_log_prob | training |
|---:|---:|---:|---:|
| 72.30 | 30.73 | 59.71 | 37.92 |

📄 [训练日志](/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/ssh_agent/output/0723_1550_2026/trials/0002/train.log)

---

### Trial 3: hardware_tuning

- **结果**: `success` | **完成步数**: 6/6

#### 参数变更（相比上一 Trial）

| 参数 | 旧值 | 新值 |
|---|---|---|
| `gpu_memory_utilization` | `0.7` | `0.8` |


### 关键指标

| 指标 | 均值 | P95 | 最大值 |
|---|---|---|---|
| 吞吐量 (tok/s) | 1052.3 | 1052.3 | 1052.3 |
| 每步耗时 (s) | 110.2 | 110.2 | 110.2 |
| 生成 tgs | 1884.6 | 1884.6 | 1884.6 |
| Actor MFU | 0.0908 | 0.0908 | 0.0908 |
| **时间瓶颈** | rollout |||

**各阶段耗时 (s):**

| 阶段 | 均值 | P95 | 最大 |
|---|---|---|---|
| rollout | 61.5 | 61.5 | 61.5 |
| actor_log_prob | 14.1 | 14.1 | 14.1 |
| ref_log_prob | 5.4 | 5.4 | 5.4 |
| training | 29.0 | 29.0 | 29.0 |

**稳定性指标:**

| 指标 | 均值 | P95 | 最大 |
|---|---|---|---|
| Reward | -0.9886 | -0.9844 | -0.9844 |
| Reward 斜率 | 0.001465 |||
| Actor PPO KL | -0.00037345 | -0.00037345 | -0.00037345 |
| Actor Entropy | 2.0912 | 2.0912 | 2.0912 |
| Clip Fraction | 0.000400 | 0.000400 | 0.000400 |
| Response Length | 750.0 | 750.0 | 750.0 |

- **显存瓶颈**: rollout
- **峰值显存**: 89.0%

### 本 Trial 完成后的 Agent 行为

_以下行为用于生成 Trial 4 的候选配置。_

#### Proposal Agent 决策

**Proposal 工具调用 (6 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `parameter_understanding` | enable_prefix_caching, enable_chunked_prefill, max_num_batched_tokens, max_num_seqs, log_prob_micro_batch_size_per_gpu, ppo_micro_batch_size_per_gpu …共 8 个 | success |
| 1 | `tuning_strategies` | end_to_end_throughput | success |
| 2 | `search_verl_docs` | 查询: "max_num_batched_tokens" | success |
| 2 | `search_verl_docs` | 查询: "max_num_seqs" | success |
| 2 | `search_verl_docs` | 查询: "enable_chunked_prefill" | success |
| 3 | `memory_estimator` | ref_trial=2, 预测参数: max_num_batched_tokens, max_num_seqs | success |


- **决策**: `modify`
- **原因**: rollout 阶段占据 57.7% 的步时，是主要瓶颈。增加 max_num_seqs 和 max_num_batched_tokens 可提升 vLLM 并发度和 GPU 利用率，从而加速生成；显式启用 prefix_caching 可利用 prompt 相似性减少重复计算。内存估算显示 rollout 峰值仅微增至 80.2%，风险低。
- **置信度**: 0.7

| 参数 | 旧值 | 新值 | 原因 |
|---|---|---|---|
| `max_num_seqs` | `None` | `2048` | 提高并发序列数上限，允许同时处理更多 sample，提升 rollout 批次利用率 |
| `max_num_batched_tokens` | `None` | `16384` | 扩大单批 token 上限，配合 max_num_seqs 增加有效批大小，提高生成吞吐 |
| `enable_prefix_caching` | `None` | `True` | 显式启用 prefix caching，复用重复 prompt 前缀的 KV cache，节省重计算开销 |


#### Feasibility Agent 审查

**审查 #1（attempt=1）**

**Feasibility 工具调用 (2 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `parameter_understanding` | max_num_seqs, max_num_batched_tokens, enable_prefix_caching | success |
| 1 | `memory_estimator` | ref_trial=2, 预测参数: max_num_seqs, max_num_batched_tokens, enable_prefix_caching | success |

- **判定**: `valid`
- **原因**: 候选修改仅涉及 rollout 硬件参数（max_num_seqs、max_num_batched_tokens、enable_prefix_caching），均属于 hardware_tuning 合法范围。memory_estimator 以 trial 2 为锚点，投影 rollout 峰值显存 80.16%（+1.1pp），远低于 92% 安全线；actor_log_prob、ref_log_prob、training 不受影响。gpu_memory_utilization 保持 0.7，避免 trial 3 的 OOM 风险。三个参数协同可提升 rollout 并发度与吞吐，且不增加其他阶段压力。
- **风险**:
  - memory_estimator 仅提供相对压力估算，真实显存需短跑验证
  - enable_prefix_caching 收益依赖 prompt 前缀复用度，可能未达预期
  - max_num_seqs=2048 可能导致调度开销，需实测确认吞吐提升

**预测显存占用 (%):**

| rollout | actor_log_prob | ref_log_prob | training |
|---:|---:|---:|---:|
| 80.16 | 30.73 | 59.71 | 37.92 |

📄 [训练日志](/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/ssh_agent/output/0723_1550_2026/trials/0003/train.log)

---

### Trial 4: hardware_tuning

- **结果**: `success` | **完成步数**: 6/6

#### 参数变更（相比上一 Trial）

| 参数 | 旧值 | 新值 |
|---|---|---|
| `enable_prefix_caching` | `None` | `True` |
| `gpu_memory_utilization` | `0.8` | `0.7` |
| `max_num_batched_tokens` | `None` | `16384` |
| `max_num_seqs` | `None` | `2048` |


### 关键指标

| 指标 | 均值 | P95 | 最大值 |
|---|---|---|---|
| 吞吐量 (tok/s) | 1275.4 | 1275.4 | 1275.4 |
| 每步耗时 (s) | 126.6 | 126.6 | 126.6 |
| 生成 tgs | 2156.2 | 2156.2 | 2156.2 |
| Actor MFU | 0.1265 | 0.1265 | 0.1265 |
| **时间瓶颈** | rollout |||

**各阶段耗时 (s):**

| 阶段 | 均值 | P95 | 最大 |
|---|---|---|---|
| rollout | 74.9 | 74.9 | 74.9 |
| actor_log_prob | 14.7 | 14.7 | 14.7 |
| ref_log_prob | 6.6 | 6.6 | 6.6 |
| training | 30.2 | 30.2 | 30.2 |

**稳定性指标:**

| 指标 | 均值 | P95 | 最大 |
|---|---|---|---|
| Reward | -0.9906 | -0.9824 | -0.9824 |
| Reward 斜率 | 0.000000 |||
| Actor PPO KL | 0.00154836 | 0.00154836 | 0.00154836 |
| Actor Entropy | 2.1836 | 2.1836 | 2.1836 |
| Clip Fraction | 0.000044 | 0.000044 | 0.000044 |
| Response Length | 1105.9 | 1105.9 | 1105.9 |

- **显存瓶颈**: rollout
- **峰值显存**: 79.8%

### 本 Trial 完成后的 Agent 行为

_该 Trial 完成后没有记录 Diagnosis、Proposal 或 Feasibility trace。_

📄 [训练日志](/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/ssh_agent/output/0723_1550_2026/trials/0004/train.log)

---

### Trial 5: stability_tuning

- **结果**: `fail` | **完成步数**: 67/80
- **错误类型**: INCOMPLETE_TRAINING

#### 参数变更（相比上一 Trial）

| 参数 | 旧值 | 新值 |
|---|---|---|
| `enable_prefix_caching` | `True` | `None` |
| `max_num_batched_tokens` | `16384` | `None` |
| `max_num_seqs` | `2048` | `None` |


### 关键指标

| 指标 | 均值 | P95 | 最大值 |
|---|---|---|---|
| 吞吐量 (tok/s) | 388.7 | 579.8 | 731.5 |
| 每步耗时 (s) | 86.6 | 97.1 | 112.5 |
| 生成 tgs | 794.2 | 1100.1 | 1290.1 |
| Actor MFU | 0.0258 | 0.0485 | 0.0711 |
| **时间瓶颈** | rollout |||

**各阶段耗时 (s):**

| 阶段 | 均值 | P95 | 最大 |
|---|---|---|---|
| rollout | 42.2 | 51.3 | 63.9 |
| actor_log_prob | 12.7 | 13.2 | 13.8 |
| ref_log_prob | 3.9 | 4.4 | 5.1 |
| training | 27.5 | 28.3 | 29.4 |

**稳定性指标:**

| 指标 | 均值 | P95 | 最大 |
|---|---|---|---|
| Reward | -0.9141 | -0.8750 | -0.8516 |
| Reward 斜率 | -0.000977 |||
| Actor PPO KL | 0.00409274 | 0.01120392 | 0.04042674 |
| Actor Entropy | 1.9271 | 3.4202 | 3.9762 |
| Clip Fraction | 0.002236 | 0.004317 | 0.005292 |
| Response Length | 113.7 | 281.0 | 482.2 |

- **显存瓶颈**: rollout
- **峰值显存**: 79.1%

### Trial 运行中的 Health Monitor 行为

#### Health 决策 #1: `trial-0005-step-000006-event-001`

- **判定**: `watch`
- **动作**: `observe`
- **置信度**: 0.75
- **继续观察步数**: 5
- **原因**: KL divergence warning triggered, but reward is improving and absolute KL remains moderate. No reward deterioration. Continue observing for another 5 updates to see if trend reverses or worsens.
- **原因代码**: `kl_growth_without_reward_drop`, `improving_reward`, `early_training`
- **支持证据**:
  - KL loss grew from ~0.001 to 0.185 over 6 steps, exceeding JF-HPO kl_growth threshold for 5 consecutive steps.
  - Reward improved from -0.992 to -0.978, no reward_drop trigger.
  - Absolute KL is still moderate and reward trajectory is positive.
  - Response length decreased from 993 to 345, but this may be beneficial as reward improves.
- **反向证据**:
  - KL growth ratio sustained >0.15 for 5 steps indicates persistent divergence trend.
  - Response length collapse could signal mode collapse, though not yet reflected in reward.

### 本 Trial 完成后的 Agent 行为

_以下行为用于生成 Trial 6 的候选配置。_

#### Diagnosis Agent 诊断

**Diagnosis 工具调用 (4 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `read_trial_log_excerpt` | {"trial_id": 5, "max_lines": 40} | success |
| 2 | `read_trial_log_excerpt` | {"trial_id": 5, "pattern": "error|oom|killed|traceback|exception", "max_lines": 40} | success |
| 3 | `read_trial_log_excerpt` | {"trial_id": 5, "max_lines": 40, "pattern": "step:67|step:68|finished|stopped|complete|error|end"} | success |
| 4 | `read_trial_log_excerpt` | {"trial_id": 5, "max_lines": 40, "pattern": "stop|terminated|abort|kill|signal|exit"} | success |

- **失败类型**: `UNKNOWN_FAILURE`
- **训练子阶段**: `unknown`
- **置信度**: 0.6
- **原因**: No evidence of memory exhaustion, divergence, or hardware error. Incomplete training likely due to external termination or timeout.
- **证据**:
  - Trial completed only 67 of 80 updates without any recorded error, OOM, NaN, or crash in logs.
  - Max GPU memory usage 79%, well below OOM threshold.
  - Health monitor triggered once at step 6 (watch/observe) but no further health events.
  - Logs show normal step progression until step 67, then training stops without error message.

#### Proposal Agent 决策

**Proposal 工具调用 (3 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `parameter_understanding` | entropy_coeff, kl_loss_coef, kl_loss_type, lr, lr_warmup_steps, use_kl_loss …共 7 个 | success |
| 1 | `tuning_strategies` | reward_collapse, kl_explosion | success |
| 2 | `query_trial_history` | stage=stability_tuning, limit=1 | success |


- **决策**: `modify`
- **原因**: Trial 5 showed reward collapse (-0.914, slope -0.001), KL rising to 0.04, entropy exploding to 1.93, and responses degenerating to mean length 114. Following reward_collapse and kl_explosion strategies, the primary action is to lower learning rate. Adding warmup steps reduces early instability risk.
- **置信度**: 0.75

| 参数 | 旧值 | 新值 | 原因 |
|---|---|---|---|
| `lr` | `3e-06` | `1e-06` | Lower learning rate to reduce policy update magnitude and curb KL drift, entropy explosion, and reward collapse. |
| `lr_warmup_steps` | `None` | `10` | Introduce learning rate warmup to avoid early training shocks; 10 steps is meaningful relative to the 80-step horizon. |


#### Feasibility Agent 审查

**审查 #1（attempt=1）**

**Feasibility 工具调用 (3 次):**

| 轮次 | 工具 | 参数 / 查询内容 | 状态 |
|---|---|---|---|
| 1 | `parameter_understanding` | lr, lr_warmup_steps | success |
| 1 | `tuning_strategies` | reward_collapse, kl_explosion | success |
| 2 | `memory_estimator` | ref_trial=5, 预测参数: lr, lr_warmup_steps | success |

- **判定**: `valid`
- **原因**: Lower learning rate and warmup are the primary recommended actions for reward collapse and KL explosion (strategies confirmed). Parameter ranges are within agent constraints (lr 1e-6, warmup 10 < 80 updates). No hardware changes; memory pressure unchanged from reference trial 5, all phases well within limits.
- **风险**:
  - Slower reward improvement may mask underlying issues; short-run validation needed.
  - Warmup delay could defer learning in a short horizon (80 updates), but 10 steps is reasonable.

**预测显存占用 (%):**

| rollout | actor_log_prob | ref_log_prob | training |
|---:|---:|---:|---:|
| 79.06 | 37.51 | 66.81 | 39.00 |

📄 [训练日志](/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/ssh_agent/output/0723_1550_2026/trials/0005/train.log)

---

### Trial 6: stability_tuning

- **结果**: `fail` | **完成步数**: 67/80
- **错误类型**: INCOMPLETE_TRAINING

#### 参数变更（相比上一 Trial）

| 参数 | 旧值 | 新值 |
|---|---|---|
| `lr` | `3e-06` | `1e-06` |
| `lr_warmup_steps` | `None` | `10` |


### 关键指标

| 指标 | 均值 | P95 | 最大值 |
|---|---|---|---|
| 吞吐量 (tok/s) | 668.0 | 1229.1 | 1244.5 |
| 每步耗时 (s) | 92.1 | 120.2 | 124.2 |
| 生成 tgs | 1308.2 | 2150.2 | 2187.3 |
| Actor MFU | 0.0496 | 0.1124 | 0.1156 |
| **时间瓶颈** | rollout |||

**各阶段耗时 (s):**

| 阶段 | 均值 | P95 | 最大 |
|---|---|---|---|
| rollout | 46.0 | 69.0 | 72.8 |
| actor_log_prob | 13.2 | 14.6 | 14.7 |
| ref_log_prob | 4.5 | 6.2 | 6.3 |
| training | 28.1 | 30.1 | 30.4 |

**稳定性指标:**

| 指标 | 均值 | P95 | 最大 |
|---|---|---|---|
| Reward | -0.9451 | -0.8848 | -0.8672 |
| Reward 斜率 | -0.005371 |||
| Actor PPO KL | 0.00119109 | 0.00645554 | 0.02389571 |
| Actor Entropy | 0.9690 | 2.1454 | 2.2134 |
| Clip Fraction | 0.001062 | 0.002820 | 0.006251 |
| Response Length | 372.6 | 1002.9 | 1020.7 |

- **显存瓶颈**: rollout
- **峰值显存**: 79.1%

### 本 Trial 完成后的 Agent 行为

_这是最后一个 Trial，记录中没有后续 Agent trace。_

📄 [训练日志](/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/ssh_agent/output/0723_1550_2026/trials/0006/train.log)

---
