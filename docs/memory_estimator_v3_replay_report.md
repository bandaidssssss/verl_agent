# Memory Estimator V3 历史回放报告

- 生成日期：2026-08-14
- 成功执行重放：14 组
- 可计算绝对预测误差：12 组
- 因参考 GPU 容量缺失而只能生成相对结果：2 组
- 无法重放：2 组
- false-safe：0 个 phase
- compute phase 上界未覆盖：3 个 phase

## 口径

本报告记录本次 V3 修改完成后实际执行的全部历史回放。主体扫描对每个含有 `trials.jsonl` 的历史目录选择最后一个带非空 proposal changes 的 trial；公式验证期间另外单独重放了 `0807_1735_2026` 的 trial 4、5、6。每次都只向 estimator 暴露目标 trial 之前的历史数据。

- 点估计：`projected_pct`。
- 保守上界：`upper_bound_pct`。上界可以超过 100%，表示风险界已经超过物理容量，不表示 GPU 可以实际分配超过容量。
- 实测：目标 trial 的 `memory_by_phase_pct.max`，结合 `gpu_samples.csv` 的 GPU 容量换算。
- false-safe：点估计/风险判断认为可运行，但目标实测超过该 trial 的显存安全线。
- upper miss：目标实测超过 estimator 的保守上界，不一定等同于 false-safe。

## 汇总

| Run | 目标 / 参考 trial | 目标结果 | 参数变化 | MAE (pct-pt) | 最大低估 (pct-pt) | 上界未覆盖 | false-safe |
|---|---:|---|---|---:|---:|---|---|
| `0723_1118_2026` | 2 / 1 | success | `actor_rollout_ref.rollout.enable_chunked_prefill`: null → true<br>`actor_rollout_ref.rollout.enable_prefix_caching`: null → true<br>`actor_rollout_ref.rollout.gpu_memory_utilization`: 0.5 → 0.7 | 0.40 | 0.44 | 无 | 无 |
| `0723_1550_2026` | 6 / 2 | fail | `actor_rollout_ref.actor.optim.lr`: 3e-06 → 1e-06<br>`actor_rollout_ref.actor.optim.lr_warmup_steps`: null → 10 | 4.38 | 16.72 | actor_log_prob | 无 |
| `0724_1741_2026` | 6 / 5 | success | `actor_rollout_ref.actor.entropy_coeff`: 0.0 → 0.01<br>`actor_rollout_ref.actor.optim.lr_warmup_steps`: 0 → 10 | 13.69 | 48.71 | rollout, ref_log_prob | 无 |
| `0728_1551_2026` | 9 / 6 | fail | `actor_rollout_ref.actor.kl_loss_coef`: 0.003 → 0.01<br>`actor_rollout_ref.actor.optim.lr`: 1e-06 → 5e-07<br>`actor_rollout_ref.actor.optim.lr_warmup_steps`: null → 10 | — | — | 无 | 无 |
| `0729_0955_2026` | 9 / 7 | fail | `actor_rollout_ref.actor.entropy_coeff`: 0.0 → 0.001<br>`actor_rollout_ref.actor.optim.lr`: 1e-06 → 2e-06 | — | — | 无 | 无 |
| `0731_0959_2026` | 8 / 7 | early_stopped | `actor_rollout_ref.actor.entropy_coeff`: 0.0 → 0.01 | 3.59 | 9.67 | rollout | 无 |
| `0731_1702_2026` | 9 / 7 | success | `actor_rollout_ref.actor.optim.lr`: 5e-06 → 3e-06 | 1.04 | 2.55 | rollout | 无 |
| `0806_0914_2026` | 4 / 3 | success | `actor_rollout_ref.rollout.gpu_memory_utilization`: 0.7 → 0.75 | 0.25 | 0.59 | 无 | 无 |
| `0807_1110_2026` | 7 / 5 | fail | `actor_rollout_ref.actor.optim.lr`: 3e-06 → 1e-05 | 5.30 | 0.23 | rollout | 无 |
| `0807_1735_2026` | 4 / 3 | success | `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`: 1 → 2 | 3.53 | 9.81 | 无 | 无 |
| `0807_1735_2026` | 5 / 4 | success | `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu`: 8 → 32<br>`actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu`: 1 → 8 | 18.17 | 37.46 | rollout | 无 |
| `0807_1735_2026` | 6 / 5 | fail | `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu`: 32 → 16<br>`actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu`: 8 → 16 | 11.05 | 22.81 | 无 | 无 |
| `0807_1735_2026` | 7 / 5 | success | `actor_rollout_ref.actor.optim.lr`: 3e-06 → 8e-06<br>`actor_rollout_ref.actor.optim.lr_warmup_steps`: 0 → 5 | 1.75 | 5.12 | ref_log_prob | 无 |
| `0812_1751_2026` | 2 / 1 | early_stopped | `actor_rollout_ref.actor.optim.lr`: 8e-06 → 2e-06 | 0.61 | 0.12 | 无 | 无 |

## 分组明细

### 0723_1118_2026 / trial 2

- 参考 trial：1
- 安全线：92.00%
- 目标结果：success
- 参数变化：`actor_rollout_ref.rollout.enable_chunked_prefill`: null → true<br>`actor_rollout_ref.rollout.enable_prefix_caching`: null → true<br>`actor_rollout_ref.rollout.gpu_memory_utilization`: 0.5 → 0.7
- 总体 MAE：0.40 pct-pt / 0.25 GiB
- 最大低估：0.44 pct-pt / 0.28 GiB
- 上界未覆盖：无
- false-safe：无

| Phase | 参考 % | 点估计 % | 上界 % | 实测 % | 误差 pct-pt | 风险 | 置信度 | 上界覆盖 |
|---|---:|---:|---:|---:|---:|---|---|---|
| rollout | 59.15 | 79.15 | 87.15 | 79.06 | 0.09 | watch | low | 是 |
| actor log-prob | 30.34 | 30.34 | 38.73 | 30.78 | -0.44 | low | low | 是 |
| ref log-prob | 62.00 | 62.00 | 70.39 | 61.04 | 0.96 | low | low | 是 |
| training | 37.95 | 37.95 | 46.34 | 38.06 | -0.11 | low | low | 是 |

### 0723_1550_2026 / trial 6

- 参考 trial：2
- 安全线：92.00%
- 目标结果：fail
- 参数变化：`actor_rollout_ref.actor.optim.lr`: 3e-06 → 1e-06<br>`actor_rollout_ref.actor.optim.lr_warmup_steps`: null → 10
- 总体 MAE：4.38 pct-pt / 2.80 GiB
- 最大低估：16.72 pct-pt / 10.70 GiB
- 上界未覆盖：actor_log_prob
- false-safe：无

| Phase | 参考 % | 点估计 % | 上界 % | 实测 % | 误差 pct-pt | 风险 | 置信度 | 上界覆盖 |
|---|---:|---:|---:|---:|---:|---|---|---|
| rollout | 79.06 | 79.06 | 79.06 | 79.06 | 0.00 | low | high | 是 |
| actor log-prob | 30.73 | 30.73 | 35.73 | 47.45 | -16.72 | low | medium | 否 |
| ref log-prob | 59.71 | 59.71 | 64.71 | 59.06 | 0.65 | low | medium | 是 |
| training | 37.92 | 37.92 | 42.92 | 38.08 | -0.16 | low | medium | 是 |

### 0724_1741_2026 / trial 6

- 参考 trial：5
- 安全线：92.00%
- 目标结果：success
- 参数变化：`actor_rollout_ref.actor.entropy_coeff`: 0.0 → 0.01<br>`actor_rollout_ref.actor.optim.lr_warmup_steps`: 0 → 10
- 总体 MAE：13.69 pct-pt / 8.77 GiB
- 最大低估：48.71 pct-pt / 31.18 GiB
- 上界未覆盖：rollout, ref_log_prob
- false-safe：无

| Phase | 参考 % | 点估计 % | 上界 % | 实测 % | 误差 pct-pt | 风险 | 置信度 | 上界覆盖 |
|---|---:|---:|---:|---:|---:|---|---|---|
| rollout | 87.25 | 87.25 | 87.25 | 91.77 | -4.52 | watch | high | 否 |
| actor log-prob | 80.91 | 80.91 | 85.91 | 81.64 | -0.73 | low | medium | 是 |
| ref log-prob | 15.99 | 15.99 | 20.99 | 64.70 | -48.71 | low | medium | 否 |
| training | 61.72 | 62.18 | 65.32 | 62.99 | -0.81 | low | low | 是 |

### 0728_1551_2026 / trial 9

- 参考 trial：6
- 安全线：92.00%
- 目标结果：fail
- 参数变化：`actor_rollout_ref.actor.kl_loss_coef`: 0.003 → 0.01<br>`actor_rollout_ref.actor.optim.lr`: 1e-06 → 5e-07<br>`actor_rollout_ref.actor.optim.lr_warmup_steps`: null → 10
- 总体 MAE：— pct-pt / — GiB
- 最大低估：— pct-pt / — GiB
- 上界未覆盖：无
- false-safe：无

| Phase | 参考 % | 点估计 % | 上界 % | 实测 % | 误差 pct-pt | 风险 | 置信度 | 上界覆盖 |
|---|---:|---:|---:|---:|---:|---|---|---|
| rollout | 58.93 | — | — | 59.97 | — | unknown_without_absolute_anchor | high | — |
| actor log-prob | 47.66 | — | — | 41.91 | — | unknown_without_absolute_anchor | medium | — |
| ref log-prob | 54.84 | — | — | 59.34 | — | unknown_without_absolute_anchor | medium | — |
| training | 59.88 | — | — | 60.30 | — | unknown_without_absolute_anchor | medium | — |

### 0729_0955_2026 / trial 9

- 参考 trial：7
- 安全线：92.00%
- 目标结果：fail
- 参数变化：`actor_rollout_ref.actor.entropy_coeff`: 0.0 → 0.001<br>`actor_rollout_ref.actor.optim.lr`: 1e-06 → 2e-06
- 总体 MAE：— pct-pt / — GiB
- 最大低估：— pct-pt / — GiB
- 上界未覆盖：无
- false-safe：无

| Phase | 参考 % | 点估计 % | 上界 % | 实测 % | 误差 pct-pt | 风险 | 置信度 | 上界覆盖 |
|---|---:|---:|---:|---:|---:|---|---|---|
| rollout | 78.09 | — | — | 78.09 | — | unknown_without_absolute_anchor | high | — |
| actor log-prob | 60.49 | — | — | 44.50 | — | unknown_without_absolute_anchor | medium | — |
| ref log-prob | 86.58 | — | — | 98.84 | — | unknown_without_absolute_anchor | medium | — |
| training | 61.94 | — | — | 71.03 | — | unknown_without_absolute_anchor | low | — |

### 0731_0959_2026 / trial 8

- 参考 trial：7
- 安全线：92.00%
- 目标结果：early_stopped
- 参数变化：`actor_rollout_ref.actor.entropy_coeff`: 0.0 → 0.01
- 总体 MAE：3.59 pct-pt / 2.30 GiB
- 最大低估：9.67 pct-pt / 6.19 GiB
- 上界未覆盖：rollout
- false-safe：无

| Phase | 参考 % | 点估计 % | 上界 % | 实测 % | 误差 pct-pt | 风险 | 置信度 | 上界覆盖 |
|---|---:|---:|---:|---:|---:|---|---|---|
| rollout | 86.80 | 86.80 | 86.80 | 87.34 | -0.54 | low | high | 否 |
| actor log-prob | 24.63 | 24.63 | 29.63 | 26.56 | -1.93 | low | medium | 是 |
| ref log-prob | 35.99 | 35.99 | 40.99 | 38.21 | -2.22 | low | medium | 是 |
| training | 59.46 | 60.91 | 72.31 | 70.58 | -9.67 | low | medium | 是 |

### 0731_1702_2026 / trial 9

- 参考 trial：7
- 安全线：90.00%
- 目标结果：success
- 参数变化：`actor_rollout_ref.actor.optim.lr`: 5e-06 → 3e-06
- 总体 MAE：1.04 pct-pt / 0.67 GiB
- 最大低估：2.55 pct-pt / 1.64 GiB
- 上界未覆盖：rollout
- false-safe：无

| Phase | 参考 % | 点估计 % | 上界 % | 实测 % | 误差 pct-pt | 风险 | 置信度 | 上界覆盖 |
|---|---:|---:|---:|---:|---:|---|---|---|
| rollout | 81.46 | 81.46 | 81.46 | 84.01 | -2.55 | low | high | 否 |
| actor log-prob | 79.55 | 79.55 | 84.55 | 80.96 | -1.41 | low | medium | 是 |
| ref log-prob | 64.51 | 64.51 | 69.51 | 64.54 | -0.03 | low | medium | 是 |
| training | 61.49 | 61.49 | 66.49 | 61.67 | -0.18 | low | medium | 是 |

### 0806_0914_2026 / trial 4

- 参考 trial：3
- 安全线：90.00%
- 目标结果：success
- 参数变化：`actor_rollout_ref.rollout.gpu_memory_utilization`: 0.7 → 0.75
- 总体 MAE：0.25 pct-pt / 0.16 GiB
- 最大低估：0.59 pct-pt / 0.37 GiB
- 上界未覆盖：无
- false-safe：无

| Phase | 参考 % | 点估计 % | 上界 % | 实测 % | 误差 pct-pt | 风险 | 置信度 | 上界覆盖 |
|---|---:|---:|---:|---:|---:|---|---|---|
| rollout | 79.42 | 83.01 | 86.04 | 83.60 | -0.59 | watch | high | 是 |
| actor log-prob | 19.45 | 19.45 | 27.84 | 19.73 | -0.28 | low | low | 是 |
| ref log-prob | 30.53 | 30.53 | 38.92 | 30.38 | 0.15 | low | low | 是 |
| training | 63.67 | 63.67 | 72.07 | 63.67 | 0.00 | low | low | 是 |

### 0807_1110_2026 / trial 7

- 参考 trial：5
- 安全线：90.00%
- 目标结果：fail
- 参数变化：`actor_rollout_ref.actor.optim.lr`: 3e-06 → 1e-05
- 总体 MAE：5.30 pct-pt / 3.39 GiB
- 最大低估：0.23 pct-pt / 0.15 GiB
- 上界未覆盖：rollout
- false-safe：无

| Phase | 参考 % | 点估计 % | 上界 % | 实测 % | 误差 pct-pt | 风险 | 置信度 | 上界覆盖 |
|---|---:|---:|---:|---:|---:|---|---|---|
| rollout | 58.27 | 58.27 | 58.27 | 58.27 | 0.00 | low | high | 否 |
| actor log-prob | 59.31 | 59.31 | 64.31 | 44.26 | 15.05 | low | medium | 是 |
| ref log-prob | 85.61 | 85.61 | 90.61 | 79.70 | 5.91 | high | medium | 是 |
| training | 59.78 | 59.78 | 64.78 | 60.01 | -0.23 | low | medium | 是 |

### 0807_1735_2026 / trial 4

- 参考 trial：3
- 安全线：90.00%
- 目标结果：success
- 参数变化：`actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`: 1 → 2
- 总体 MAE：3.53 pct-pt / 2.26 GiB
- 最大低估：9.81 pct-pt / 6.28 GiB
- 上界未覆盖：无
- false-safe：无

| Phase | 参考 % | 点估计 % | 上界 % | 实测 % | 误差 pct-pt | 风险 | 置信度 | 上界覆盖 |
|---|---:|---:|---:|---:|---:|---|---|---|
| rollout | 86.26 | 86.26 | 86.26 | 86.19 | 0.07 | watch | high | 是 |
| actor log-prob | 19.43 | 19.43 | 24.43 | 19.41 | 0.02 | low | medium | 是 |
| ref log-prob | 30.45 | 30.45 | 35.45 | 34.66 | -4.21 | low | medium | 是 |
| training | 63.67 | 72.79 | 127.34 | 82.60 | -9.81 | high | low | 是 |

### 0807_1735_2026 / trial 5

- 参考 trial：4
- 安全线：90.00%
- 目标结果：success
- 参数变化：`actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu`: 8 → 32<br>`actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu`: 1 → 8
- 总体 MAE：18.17 pct-pt / 11.63 GiB
- 最大低估：37.46 pct-pt / 23.97 GiB
- 上界未覆盖：rollout
- false-safe：无

| Phase | 参考 % | 点估计 % | 上界 % | 实测 % | 误差 pct-pt | 风险 | 置信度 | 上界覆盖 |
|---|---:|---:|---:|---:|---:|---|---|---|
| rollout | 86.19 | 86.19 | 86.19 | 87.83 | -1.64 | watch | high | 否 |
| actor log-prob | 19.41 | 28.25 | 147.55 | 60.30 | -32.05 | high | low | 是 |
| ref log-prob | 34.66 | 52.37 | 137.97 | 89.83 | -37.46 | high | low | 是 |
| training | 82.60 | 82.60 | 87.60 | 84.13 | -1.53 | watch | medium | 是 |

### 0807_1735_2026 / trial 6

- 参考 trial：5
- 安全线：90.00%
- 目标结果：fail
- 参数变化：`actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu`: 32 → 16<br>`actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu`: 8 → 16
- 总体 MAE：11.05 pct-pt / 7.07 GiB
- 最大低估：22.81 pct-pt / 14.60 GiB
- 上界未覆盖：无
- false-safe：无

| Phase | 参考 % | 点估计 % | 上界 % | 实测 % | 误差 pct-pt | 风险 | 置信度 | 上界覆盖 |
|---|---:|---:|---:|---:|---:|---|---|---|
| rollout | 87.83 | 87.83 | 87.83 | 86.97 | 0.86 | watch | high | 是 |
| actor log-prob | 60.30 | 70.52 | 120.15 | 93.33 | -22.81 | high | low | 是 |
| ref log-prob | 89.83 | 77.88 | 79.32 | 58.40 | 19.48 | low | low | 是 |
| training | 84.13 | 84.13 | 89.13 | 83.08 | 1.05 | watch | medium | 是 |

### 0807_1735_2026 / trial 7

- 参考 trial：5
- 安全线：90.00%
- 目标结果：success
- 参数变化：`actor_rollout_ref.actor.optim.lr`: 3e-06 → 8e-06<br>`actor_rollout_ref.actor.optim.lr_warmup_steps`: 0 → 5
- 总体 MAE：1.75 pct-pt / 1.12 GiB
- 最大低估：5.12 pct-pt / 3.28 GiB
- 上界未覆盖：ref_log_prob
- false-safe：无

| Phase | 参考 % | 点估计 % | 上界 % | 实测 % | 误差 pct-pt | 风险 | 置信度 | 上界覆盖 |
|---|---:|---:|---:|---:|---:|---|---|---|
| rollout | 87.83 | 87.83 | 87.83 | 87.70 | 0.13 | watch | high | 是 |
| actor log-prob | 60.30 | 60.30 | 65.30 | 61.10 | -0.80 | low | medium | 是 |
| ref log-prob | 89.83 | 89.83 | 94.83 | 94.95 | -5.12 | high | medium | 否 |
| training | 84.13 | 84.13 | 89.13 | 85.08 | -0.95 | watch | medium | 是 |

### 0812_1751_2026 / trial 2

- 参考 trial：1
- 安全线：90.00%
- 目标结果：early_stopped
- 参数变化：`actor_rollout_ref.actor.optim.lr`: 8e-06 → 2e-06
- 总体 MAE：0.61 pct-pt / 0.39 GiB
- 最大低估：0.12 pct-pt / 0.08 GiB
- 上界未覆盖：无
- false-safe：无

| Phase | 参考 % | 点估计 % | 上界 % | 实测 % | 误差 pct-pt | 风险 | 置信度 | 上界覆盖 |
|---|---:|---:|---:|---:|---:|---|---|---|
| rollout | 88.50 | 88.50 | 88.50 | 87.77 | 0.73 | watch | high | 是 |
| actor log-prob | 60.33 | 60.33 | 65.33 | 60.45 | -0.12 | low | medium | 是 |
| ref log-prob | 95.85 | 95.85 | 100.85 | 94.27 | 1.58 | high | medium | 是 |
| training | 86.86 | 86.86 | 91.86 | 86.85 | 0.01 | high | medium | 是 |

## 无法重放的历史目录

| Run | 目标 trial | 原因 |
|---|---:|---|
| `0720_1656_2026` | 6 | ValueError: target trial proposal has no integer reference_trial_id; choose a non-baseline trial produced from an earlier empirical reference |
| `0724_1721_2026` | 2 | ValueError: reference trial 1 has no phase memory observations |

## 观察

1. 12 组具有绝对预测值的回放中没有出现 false-safe；另外两组虽完成 estimator 调用，但参考记录缺少 GPU 容量，因此不能计算点估计、上界或 false-safe。
2. log-prob micro-batch 大幅增加时，点估计仍可能明显低估真实峰值。例如 `0807_1735_2026/trial 5` 的 actor/ref 分别低估 32.05/37.46 pct-pt；但新的 activation ratio 上界覆盖了这两个实测峰值，结果也保持 `low` confidence。
3. 少数 compute phase 在配置未发生直接相关变化时仍出现较大的实测漂移。例如 `0724_1741_2026/trial 6` 的 ref log-prob 从参考 15.99% 漂移到实测 64.70%。这类变化无法由静态配置公式解释，更像 phase 标签、采样时机、allocator/rollout 残留或运行时 shape 的变化。
4. 上界仍不是形式化保证：成功回放中仍有 compute upper miss。真实短跑 resource gate 必须保留，尤其是历史 phase 峰值本身不稳定的实验组。
5. 两组失败不是 estimator 计算异常：一个 proposal 缺少整数 `reference_trial_id`，另一个参考 trial 没有 phase memory observation，因此不满足显式实测锚点要求。

## 当前仍需实测校准的部分

- dynamic BSHD 的真实 padded micro-batch shape；
- fused cross-entropy、Transformer Engine、通信 overlap 与 allocator workspace；
- MoE 每个 EP rank 的 routed-token histogram 与最忙 rank；
- MLA activation 生命周期；
- LoRA 每 rank 的真实 trainable numel；
- phase sampler 对瞬时峰值和跨 phase 驻留的归因准确性。
