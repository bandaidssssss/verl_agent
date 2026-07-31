# verl Stage Tuning Agent

这个目录实现了一个参考 OptiCo 的 verl 0.7 GRPO 自动调优闭环：Proposal Agent 每轮生成 2–3 组候选，确定性 Validator 按候选各自的 reference trial 检查硬约束，Feasibility Agent 从合法候选中选择一组，Diagnosis Agent 在失败后归因。Proposal 支持参数查询、调优策略、分阶段显存估算、实时 GPU、verl 本地文档和历史 trial 等主动工具调用；候选批次被 Validator/Feasibility 拒绝后，会在同一个对话中看到逐项拒绝原因并继续推理。

完整架构、源码阅读顺序和调试方法见 [`docs/Agent架构与源码学习指南.md`](docs/Agent架构与源码学习指南.md)。

## 调优状态

1. `hardware_tuning` / `hardware_repair`
   - 每个候选最多运行 20 update。
   - 前 5 update 是 Resource Gate；出现 OOM/NCCL 或 GPU 显存达到硬上限时提前停止。
   - 通过后继续测量去掉 warmup 后的端到端吞吐和四个训练子阶段耗时。
2. `stability_tuning`
   - 默认运行 80 update。
   - 冻结硬件参数，只允许修改 lr、warmup、KL、entropy 和 rollout.n。
   - 每个完整 update 使用 JF-HPO 规则检查 KL 增长、reward 下降和 reward 连续为零；触发后异步交给 Train Health Agent 复核。
   - Agent 只有返回 `unhealthy + stop` 且置信度达到门槛时，才会在下一个完整 update 边界早停；Agent 失败或证据不足时继续训练。
   - 默认每 5 update 保存 checkpoint，主动早停记录为 `early_stopped`，不与 OOM/NCCL 失败混淆。
3. `confirm`
   - 默认运行 300 update。
   - 配置冻结，记录 reward 到阈值的累计时间、累计 token 和 peak reward。

一次 update 内的监控阶段为 `rollout`、`actor_log_prob`、`ref_log_prob`、`training`。runner 设置 `VERL_LOGGING_LEVEL=DEBUG`，使用 verl 的 `GPUMemoryLogger` 阶段边界，同时调用平台对应的 SMI 每秒采样每张卡。若没有可用 SMI，阶段显存会使用日志观测值；没有可靠数据时输出 `null`。

支持的平台：

- `PLATFORM=V5000`：默认平台，使用 `xpu-smi` 和 `train/env_V5000.sh`。
- `PLATFORM=A100`、`NVIDIA` 或 `CUDA`：使用 `nvidia-smi` 和 `train/env_NVIDIA.sh`。
- `PLATFORM=C550` 或 `METAX`：使用 `mx-smi` 和 `train/env_C550.sh`。
- 特殊环境可通过 `GPU_SMI` 和 `VERL_ENV_SCRIPT` 覆盖监控命令与环境脚本。

## 配置

日常修改集中在两个 JSON 文件：

- `config/base_parameters.json`：由参考脚本 `qwen3_8B_baseline.sh` 转换出的初始 verl Hydra 参数。数据集、模型路径、GPU 数在这里修改。
- `config/agent_config.json`：verl 仓库路径、update 预算、显存阈值和调优轮数。

在线健康监控默认采用 JF-HPO 论文实验参数：

- `health_kl_growth_threshold=0.15`：相邻 update 的 KL loss 增长率超过 15%。
- `health_reward_drop_threshold=0.10`：相邻 update 的 reward 下降率超过 10%。
- `health_consecutive_steps=5`：上述条件或 reward 为零连续 5 个 update 后触发 Agent。
- `health_agent_stop_confidence=0.8`：Agent 早停建议的最低置信度。
- `health_agent_shadow_mode=false`：实际应用早停；改为 `true` 可只记录判断、不停止。
- `health_agent_max_calls_per_trial=0`：单个 trial 不限制调用次数；仍受单请求并发和 5-update cooldown 约束。

服务器上的 verl 路径可以直接通过 `VERL_ROOT` 设置。环境脚本只能设置环境变量，不能包含训练命令。API 凭据通过环境变量提供——复制 [`env.sh`](env.sh) 填入实际值后 `source env.sh` 即可。

## 运行

先检查第一轮生成的命令，不启动训练：

```bash
PLATFORM=C550 bash run_circle.sh --dry-run --rules-only
```

运行一个 trial。第一轮是 20-update 硬件基线，不需要调用 LLM：

```bash
PLATFORM=C550 MAX_TRIALS=1 bash run_circle.sh
```

后续运行会读取 `output/trials.jsonl`，调用 Agent 产生候选并继续状态机：

```bash
PLATFORM=C550 MAX_TRIALS=10 bash run_circle.sh
```

每次运行的终端标准输出和标准错误也会保存到本次实验目录的
`run_circle.log`；训练子进程的逐 trial 日志仍保存在
`trials/NNNN/train.log`。

默认一次只运行一个 trial，便于检查实际 GPU 环境。确认配置和监控正确后再提高 `--max-trials`。

Proposal 返回由 `min_proposal_candidates` / `max_proposal_candidates` 限制的候选数组；每组候选独立记录 `candidate_id`、`reference_trial_id`，并逐项输出参数的 `from → to`、修改原因和预期指标变化。orchestrator 会按各自 reference trial 构造完整参数并逐组校验，至少两组通过后才交给 Feasibility。Feasibility 只能返回已验证的 `selected_candidate_id`，选中后转换回原有单 proposal/candidate 接口，训练及后续状态机不变。

参考 trial 未显式配置、但位于当前阶段 editable 白名单中的字段，可使用 `from: null` 表示新增 Hydra override；白名单外字段仍会被拒绝，并把原因和 editable 字段列表反馈给 Proposal。默认 `stream_agent_events=true`，终端会实时显示每次 Agent 工具调用、最终回答和审查拒绝。所有 Proposal 轮次都失败时写入 `state.json: proposal_blocked` 和 `last_agent_rejection.json` 后安全退出，不再抛出 traceback。

主要输出：

- `output/trials.jsonl`：所有 trial 的参数、指标、建议、Feasibility 和完整 Agent trace。
- `output/state.json`：当前调优阶段。
- `output/last_agent_rejection.json`：多轮建议仍未通过时的完整拒绝轨迹。
- `output/trials/NNNN/train.log`：原始 verl 日志。
- `output/trials/NNNN/gpu_samples.csv`：带训练子阶段标签的逐 GPU 采样。
- `output/trials/NNNN/health_events.jsonl`：JF-HPO 规则触发、Agent 决策及停止动作。
- `output/trials/NNNN/health_agent_traces.jsonl`：Train Health Agent 的完整对话、工具和 token trace。
- `output/trials/NNNN/trial_report.json`：单轮结构化报告与 Agent 工具调用轨迹。

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

# 单独运行一个受监控的 trial
PLATFORM=C550 bash train/run_verl.sh \
  parameters.json hardware_tuning 1 20
```
