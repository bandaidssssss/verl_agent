# verl Stage Tuning Agent

这个目录实现了一个参考 OptiCo 的 verl 0.7 GRPO 自动调优闭环：Proposal Agent 每轮生成 2–3 组候选，确定性 Validator 按候选各自的 reference trial 检查硬约束，Feasibility Agent 从合法候选中选择一组，Diagnosis Agent 在失败后归因。Proposal 支持参数查询、调优策略、分阶段显存估算、实时 GPU、verl 本地文档和历史 trial 等主动工具调用；候选批次被 Validator/Feasibility 拒绝后，会在同一个对话中看到逐项拒绝原因并继续推理。

完整架构、源码阅读顺序和调试方法见 [`docs/Agent架构与源码学习指南.md`](docs/Agent架构与源码学习指南.md)。

## 调优状态

1. `hardware_tuning` / `hardware_repair`
   - 每个候选最多运行 20 update。
   - 前 5 update 是 Resource Gate；出现 OOM/NCCL 或 GPU 显存达到硬上限时提前停止。
   - 通过后继续测量去掉 warmup 后的端到端吞吐和四个训练子阶段耗时。
2. `stability_tuning`
   - 当前运行 50 update。
   - 冻结硬件参数，只允许修改 lr、warmup、KL、entropy 和 rollout.n。
   - 每个完整 update 检查 reward 趋势恶化、KL 突变和 entropy 骤降；触发后异步交给 Train Health Agent 复核。
   - Agent 只有返回 `unhealthy + stop` 且置信度达到门槛时，才会在下一个完整 update 边界早停；Agent 失败或证据不足时继续训练。
   - Agent 返回 `observe N` 后，runner 会建立独立于规则 cooldown 的复审期限；第 N 个新 update 到期时即使没有规则再次触发，也会读取当前 trial 的固定指标快照并强制复审。
   - Train Health Agent 可通过 `read_current_trial_metrics` 读取当前运行日志中受限的 reward、KL、entropy、gradient 和生成健康指标；路径和最大可见 step 均由 runner 固定，Agent 不能读取任意文件或未来 step。
   - 主动早停记录为 `early_stopped`，不与 OOM/NCCL 失败混淆。
   - runner 会覆盖基础参数中的 `trainer.save_freq=-1`，在成功 trial 的最后一个 update 保存 Verl 原生 checkpoint；失败或早停 trial 不发布 checkpoint。
3. `confirm`
   - 从 `val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@1` 最高的成功 stability trial 的 checkpoint 恢复，当前训练到全局 step 135；例如从 step 50 恢复时实际再运行 85 update。
   - 配置冻结，记录 reward 到阈值的累计时间、累计 token 和 peak reward。

一次 update 内的监控阶段为 `rollout`、`actor_log_prob`、`ref_log_prob`、`training`。runner 设置 `VERL_LOGGING_LEVEL=DEBUG`，使用 verl 的 `GPUMemoryLogger` 阶段边界，同时调用平台对应的 SMI 每秒采样每张卡。若没有可用 SMI，阶段显存会使用日志观测值；没有可靠数据时输出 `null`。

当 rollout 后端为 vLLM，并且参数中明确设置
`actor_rollout_ref.rollout.disable_log_stats=false` 时，runner 还会自动发现每个
vLLM replica 的 `/metrics` 地址，默认每 5 秒采集一次。采集结果只保留调节
`gpu_memory_utilization`、`max_num_seqs` 和 `max_num_batched_tokens` 所需的活动期指标；
字段为 `true` 或未配置时不会启动监控，也不会创建 `vllm_metrics.csv`。采样间隔可通过
`config/agent_config.json` 的 `vllm_metrics_interval_seconds` 调整。

支持的平台：

- `PLATFORM=V5000`：默认平台，使用 `xpu-smi` 和 `train/env_V5000.sh`。
- `PLATFORM=A100`、`NVIDIA` 或 `CUDA`：使用 `nvidia-smi` 和 `train/env_NVIDIA.sh`。
- `PLATFORM=C550` 或 `METAX`：使用 `mx-smi` 和 `train/env_C550.sh`。
- 特殊环境可通过 `GPU_SMI` 和 `VERL_ENV_SCRIPT` 覆盖监控命令与环境脚本。

## 配置

日常修改集中在两个 JSON 文件：

- `config/base_parameters.json`：由参考脚本 `qwen3_8B_baseline.sh` 转换出的初始 verl Hydra 参数。数据集、模型路径、GPU 数在这里修改。
- `config/agent_config.json`：verl 仓库路径、update 预算、显存阈值和调优轮数。

在线健康监控只保留三类触发信号：

- reward：`health_reward_trend_steps=5` 个 update 内，用大小为 `health_reward_window_size=3` 的多个滑动窗口均值判断持续下降或低位横盘；相对历史最佳窗口至少回撤 `health_reward_trend_min_drawdown=0.15`，并允许 `health_reward_trend_tolerance=0.01` 的噪声。
- KL：相邻 update 的变化同时达到 `health_kl_change_ratio_threshold=0.50` 和 `health_kl_change_absolute_threshold=0.02`，避免接近零时仅因比例放大而误触发。
- entropy：相邻 update 相对下降达到 `health_entropy_drop_ratio_threshold=0.30`，用于捕获突然倒塌，不使用固定绝对 entropy 地板。
- `health_agent_stop_confidence=0.8`：Agent 早停建议的最低置信度。
- `health_agent_shadow_mode=false`：实际应用早停；改为 `true` 可只记录判断、不停止。
- `health_agent_max_calls_per_trial=0`：单个 trial 不限制调用次数；仍受单请求并发和 5-update cooldown 约束。

这些规则只负责触发复核，最终早停仍由 Train Health Agent 读取当前 trial 指标后决定；不存在固定 reward 失败地板，也不会因 KL 或 entropy 的绝对值较高就单独触发。

服务器上的 verl 路径可以直接通过 `VERL_ROOT` 设置。环境脚本只能设置环境变量，不能包含训练命令。API 凭据通过环境变量提供——复制 [`env.sh`](env.sh) 填入实际值后 `source env.sh` 即可。

## 运行

先检查第一轮生成的命令，不启动训练：

```bash
PLATFORM=C550 bash run_circle.sh --dry-run --rules-only
```

运行一个 trial。第一轮使用 `hardware_trial_updates`（当前为 5）的硬件基线，不需要调用 LLM：

```bash
PLATFORM=C550 MAX_TRIALS=1 bash run_circle.sh
```

后续运行会读取 `output/trials.jsonl`，调用 Agent 产生候选并继续状态机：

```bash
PLATFORM=C550 MAX_TRIALS=10 bash run_circle.sh
```

调试 stability 时可以跳过 hardware 搜索。请使用一个新的 `OUTPUT_PATH`，首个 trial 会直接以 `base_parameters.json` 作为冻结硬件基线运行 stability：

```bash
PLATFORM=C550 START_STAGE=stability_tuning \
  OUTPUT_PATH=/absolute/path/to/new-stability-debug-run \
  MAX_TRIALS=1 bash run_circle.sh
```

也可以传 `--start-stage stability_tuning`，或把 `config/agent_config.json` 的 `start_stage` 从 `auto` 改为 `stability_tuning`。默认 `auto` 保持原有 hardware → stability → confirm 状态机。

每次运行的终端标准输出和标准错误也会保存到本次实验目录的
`run_circle.log`；训练子进程的逐 trial 日志仍保存在
`trials/NNNN/train.log`。

默认一次只运行一个 trial，便于检查实际 GPU 环境。确认配置和监控正确后再提高 `--max-trials`。

Proposal 返回由 `min_proposal_candidates` / `max_proposal_candidates` 限制的候选数组；每组候选独立记录 `candidate_id`、`reference_trial_id`，并逐项输出参数的 `from → to`、修改原因和预期指标变化。orchestrator 会按各自 reference trial 构造完整参数并逐组校验，至少两组通过后才交给 Feasibility。Feasibility 只能返回已验证的 `selected_candidate_id`，选中后转换回原有单 proposal/candidate 接口，训练及后续状态机不变。

参考 trial 未显式配置、但位于当前阶段 editable 白名单中的字段，可使用 `from: null` 表示新增 Hydra override；白名单外字段仍会被拒绝，并把原因和 editable 字段列表反馈给 Proposal。默认 `stream_agent_events=true`，终端会实时显示每次 Agent 工具调用、最终回答和审查拒绝。所有 Proposal 轮次都失败时写入 `state.json: proposal_blocked` 和 `last_agent_rejection.json` 后安全退出，不再抛出 traceback。

主要输出：

- `output/trials.jsonl`：轻量、可恢复的 trial 索引，只含状态机和历史初筛字段；每条记录最后写入。
- `output/state.json`：当前调优阶段。
- `output/last_agent_rejection.json`：多轮建议仍未通过时的完整拒绝轨迹。
- `output/trials/NNNN/train.log`：原始 verl 日志。
- `output/trials/NNNN/gpu_samples.csv`：带训练子阶段标签的逐 GPU 采样。
- `output/trials/NNNN/vllm_metrics.csv`：仅在 vLLM stats 明确启用时生成的紧凑 rollout 调度、KV-cache 和 preemption 采样。
- `output/trials/NNNN/health_events.jsonl`：健康规则触发、Agent 决策及停止动作。
- `output/trials/NNNN/health_agent_traces.jsonl`：Train Health Agent 的完整对话、工具和 token trace。
- `output/trials/NNNN/metrics.json`：一次解析得到的 throughput、stability、evaluation、resource 分类指标；运行中原子更新，结束后标为 `final`。`evaluation.latest_metrics` 保存 `config/agent_config.json:evaluation_metrics` 中配置的最新验证/测试分数。`evaluate_at_trial_end=true` 会按当前 stage 的 update 目标设置 `trainer.test_freq`，在正常结束时执行一次验证；提前终止且尚未验证的 trial 不会有该分数。
- `output/trials/NNNN/log_facts.json`：统一提取器在同一次日志扫描中提取的完整 Hydra runtime 参数、模型配置、Megatron resolved runtime、去重后的 rank 参数量和有效序列长度；不属于 memory estimator 输出。
- `output/trials/NNNN/parameters.json` / `parameter_groups.json`：实际传给 Hydra 的显式参数，以及 fixed/throughput/stability/ignored 分类。框架默认值不会补写进 `parameters.json`。
- `output/trials/NNNN/decision.json` / `agent_trace.json`：决策摘要与完整 Agent trace 分开保存。
- `output/trials/NNNN/trial_report.json`：不含逐 step 数组和 trace 的单轮结果摘要。

## 最终指标

`tools/compare_end_to_end_reward.py` 是独立验收入口：

```bash
PYTHONPATH=. python3 tools/compare_end_to_end_reward.py \
  --log base=path/to/base.log \
  --log candidate=path/to/candidate.log \
  --output reward_comparison.json
```

阈值与窗口可以通过 `--thresholds` 和 `--window` 修改。

## 独立脚本

主循环之外，也可以单独执行与 OptiCo 对应的步骤：

```bash
# 单独调用一个 Agent 角色
bash analyzer/run_analyze.sh proposal context.json suggestion.json trace.json

# 单独分析已有训练日志
bash monitor/run_monitor.sh train.log gpu_samples.csv report.json

# 从单个 trial 的原始 artifact 重新生成分类指标
python tools/extract_trial_metrics.py --trial-dir output/trials/0001 \
  --agent-config config/agent_config.json

# 单独运行一个受监控的 trial
PLATFORM=C550 bash train/run_verl.sh \
  parameters.json hardware_tuning 1 20
```

Agent、Validator 和 Memory Estimator 要求 reference trial 的 `log_facts.json`
包含 `runtime_parameters`。缺少这一层的 trial 不做旧格式兼容，必须用上面的
`extract_trial_metrics.py` 从其 `train.log` 重新提取后才能作为 reference。
