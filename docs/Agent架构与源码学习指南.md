# verl Stage Tuning Agent — 对话速览

> 目标：让新对话快速理解项目。以当前工作区源码和 `config/agent_config.json` 为准。

## 1. 一句话说明

这是一个面向 **verl 0.7 GRPO** 的自动超参数调优器：LLM 负责提出实验、审查候选和解释异常；确定性代码负责阶段迁移、参数约束、训练启停与结果落盘。

```text
历史 trial
  → 判断调优阶段
  → Proposal 提出 3 个候选
  → Validator 过滤硬约束
  → Feasibility 选择一个合法候选
  → runner 执行真实 verl 训练
  → 采集显存、吞吐、稳定性和 vLLM 指标
  → 写回历史，进入下一轮
```

第一轮不调用 LLM，直接运行 `base_parameters.json`，建立真实基线。

## 2. 三个调优阶段

```text
hardware_tuning / hardware_repair
  在显存安全前提下提高端到端吞吐；全部失败时进入 repair。

stability_tuning
  冻结硬件参数，调整 lr、warmup、KL、entropy、rollout.n 等训练参数。

confirm
  冻结最终配置，从最佳成功 stability trial 的 checkpoint 恢复并记录 reward 收敛。
```

状态机在 [orchestrator.py](../orchestrator.py)：

- hardware 数量不足或吞吐尚未 plateau：继续 hardware。
- hardware 搜索结束：进入 stability。
- healthy stability trial 足够：进入 confirm。
- stability 达到上限但都不健康：`stopped_unstable`。
- 已存在 confirm trial：`done`。

默认起点：hardware 用吞吐最高的成功 trial；repair 用最新失败 trial；stability/confirm 优先用 terminal reward 最好的 stability trial。

当前实际预算：

```text
hardware 5 updates
stability 50 updates
confirm 训练到全局 step 135（从 stability step 50 恢复时新增 85 updates）
resource gate 从第 1 个 update 后检查
```

调试 stability 时可设置 `start_stage=stability_tuning`（或命令行
`--start-stage stability_tuning`）跳过 hardware 搜索。新输出目录的首个 trial
直接使用 `base_parameters.json` 作为冻结硬件基线；默认 `auto` 不改变原状态机。

## 3. 候选选择闭环

Proposal 当前返回 3 个候选。每个候选可选择不同的 `reference_trial_id`，并记录每项参数的 `from → to`。

```text
Proposal
  ↓
来源校验：reference 存在，from 与 reference 完全一致
  ↓
Validator：阶段白名单、类型、范围、batch/TP/PP 整除、token budget、去重
  ↓
Feasibility：逐项语义和资源风险审查，只选择一个 candidate_id
```

Feasibility 不能修改、合并或创建候选。

整批被拒绝后，具体原因会追加到原 Proposal 对话中继续重试。轮次耗尽时写 `last_agent_rejection.json`，不启动训练。

## 4. 四个 Agent

运行时在 [agents.py](../agents.py)，Prompt 在 `prompts/`。

| Agent | 职责 |
|---|---|
| Proposal | 根据阶段、历史和工具证据设计候选 |
| Feasibility | 从确定性合法候选中选择一个 |
| Diagnosis | 对失败 trial 做根因归类 |
| Train Health | 复核正在运行的 stability trial 是否应早停 |

单个 Agent 内部还有一个工具循环：

```text
LLM → function call → ToolRegistry → 结果加入对话 → LLM 最终 JSON
```

会话 trace 保存 messages、工具调用、请求错误、token 使用和结果。LLM 使用 OpenAI Chat Completions 兼容接口，读取 `API_KEY/OPENAI_API_KEY`、`BASE_URL` 和 `INFER_MODEL`。

## 5. Agent 工具

工具声明在 [agent_tools/skills.json](../agent_tools/skills.json)，执行边界在 [agent_tools/registry.py](../agent_tools/registry.py)。

| 工具 | 用途 |
|---|---|
| `parameter_understanding` | 查询参数语义和约束 |
| `memory_estimator` | 以历史 trial 为锚点估算四阶段显存 |
| `analyze_rollout_metrics` | 判断 vLLM capacity 参数是否真的受限 |
| `live_gpu_snapshot` | 当前主机 GPU 快照，仅作环境证据 |
| `search_verl_docs` | 搜索本地 verl 配置、源码和文档 |
| `query_trial_history` | 查询结构化 trial 历史 |
| `read_trial_log_excerpt` | 读取受限日志片段 |
| `read_trial_metrics` | 读取 reward、KL 等 step 窗口 |
| `read_current_trial_metrics` | Train Health 读取 runner 固定 snapshot step 的当前 trial 指标；不接受任意路径 |

`tuning_strategies` 有实现，但当前没有授权给任何角色。

日志和指标工具不能读取任意路径，只能访问当前 output 历史中记录的文件。

## 6. 显存估算版本边界

- [agent_tools/memory_estimator_V3.py](../agent_tools/memory_estimator_V3.py)：唯一实现，只消费当前格式参考 trial 的结构化 artifact，输出分阶段状态、参考峰值、预测峰值和相对变化百分比。
- `agent_tools/mem_estimator.py`：理论公式参考，不是生产入口。

生产版分别估算：

```text
rollout / actor_log_prob / ref_log_prob / training
```

Agent 只消费每个阶段的 `status`、`reference_peak_mib`、`estimated_peak_mib` 和 `estimated_relative_change_pct`，以及顶层 `safety` 与条件式 `note`。内部 upper 不进入 Agent 上下文；真实短跑的 Resource Gate 仍是最终安全依据。

## 7. Trial 执行与监控

[runner.py](../runner.py) 负责：

1. 构造 Hydra 命令并启动 verl 子进程；
2. 将 stdout 保存为 `train.log`；
3. 跟踪 rollout、actor log-prob、ref log-prob、training 阶段；
4. 调用 SMI 采集逐 GPU 显存和利用率；
5. 检查 OOM、NCCL/分布式错误，以及“每卡总容量减绝对 reserve”得到的显存硬上限；
6. 条件采集 vLLM 指标；
7. stability 阶段运行在线健康监控；
8. 每个 update 原子写入 running `metrics.json`，训练结束后一次解析原始 artifact 并写入 final `metrics.json`。
9. stability 到达目标 update 时保存 Verl 原生 checkpoint；只有最终结果为 `success` 才登记为 trial artifact，失败或早停产生的残留会清理。

每个 trial 使用独立的 `trials/NNNN/checkpoints`，hardware/stability 禁止 Verl `auto` 恢复。confirm 直接从状态机选中的最佳 stability reference 的 `global_step_*` 目录恢复。

平台：

```text
V5000          → xpu-smi    → train/env_V5000.sh
C550 / METAX   → mx-smi     → train/env_C550.sh
NVIDIA / CUDA  → nvidia-smi → train/env_NVIDIA.sh
```

结果分为 `success`、`fail` 和 `early_stopped`。

## 8. 在线健康监控

[health_monitor.py](../health_monitor.py) 在 stability trial 的每个 update 检查：

- 最近 5 个 step 的多个 reward 窗口均值是否持续下降，或在明显回撤后低位横盘；
- KL 是否在相邻 step 同时发生足够大的相对变化和绝对变化；
- entropy 是否相对前一个 step 突然倒塌。

规则只产生事件，不直接停止训练。runner 异步调用 Train Health Agent；真正早停必须满足：

```text
verdict == unhealthy
action == stop
confidence >= 配置门槛
shadow_mode == false
```

满足后在下一个完整 update 边界停止。Agent 调用失败时继续训练。

`observe N` 会建立独立于规则 cooldown 的复审期限；到期后即使没有规则再次触发，
runner 也会生成 `scheduled_followup` 并调用 Agent。Train Health 可读取当前 trial 的
粗粒度窗口和触发附近逐 step 指标。所有阈值都描述相对轨迹或变化；没有固定 reward
失败地板，也不会仅凭 KL/entropy 的绝对值触发。规则只负责唤醒 Agent，停止仍需
Agent 结合 reward、KL、entropy、gradient 和生成指标复核。

## 9. 指标与 vLLM

[metrics.py](../metrics.py) 汇总：

- 四阶段显存、GPU 利用率和耗时；
- 吞吐和 step 时间；
- reward、KL、entropy、clip 等窗口序列；
- terminal reward；
- reward 阈值对应的累计时间/token；
- OOM、NCCL、NaN/Inf 等错误。

[vllm_metrics.py](../vllm_metrics.py) 只有在以下条件同时满足时启用：

```text
actor_rollout_ref.rollout.name == "vllm"
actor_rollout_ref.rollout.disable_log_stats == false
```

它采集 request、KV cache、preemption 和 token rate，用于判断 `gpu_memory_utilization`、`max_num_seqs`、`max_num_batched_tokens` 是否存在直接 binding evidence。缺失指标表示 unknown，不表示 0。

## 10. 最重要的文件

```text
run_circle.sh               Shell 主入口
run_agent.py                Python 主入口
orchestrator.py             状态机、候选闭环、历史写入
agents.py                   LLM 会话与四角色
validator.py                硬约束
runner.py                   训练与在线监控
health_monitor.py           stability 风险规则
metrics.py                  离线指标汇总
vllm_metrics.py             vLLM 指标链路
agent_tools/registry.py     工具权限和执行边界
config/agent_config.json    调优器实际配置
config/base_parameters.json 第一轮 verl 参数
```

## 11. 输出与事实来源

```text
<run_dir>/
├── trials.jsonl                  # 状态机的事实来源
├── state.json                    # 当前状态摘要
├── final_result.json             # confirm 结果
├── last_agent_rejection.json
├── last_agent_error.json
└── trials/NNNN/
    ├── parameters.json
    ├── parameter_groups.json
    ├── metrics.json
    ├── log_facts.json
    ├── decision.json
    ├── agent_trace.json
    ├── command.json
    ├── train.log
    ├── gpu_samples.csv
    ├── vllm_metrics.csv          # 条件开启
    ├── health_events.jsonl
    ├── health_agent_traces.jsonl
    ├── checkpoints/global_step_N/  # 成功 stability trial 的 Verl checkpoint
    └── trial_report.json
```

`trials.jsonl` 是轻量权威索引，状态机和历史初筛只读取它；详细参数、指标与决策按索引中的相对 artifact 路径按需加载。`state.json` 不是完整历史。Agent 指标工具只读 `metrics.json`，不会反复扫描 `train.log`。

指标分类如下：

- `metrics.json.throughput`：吞吐、step 时间、TGS、MFU、各阶段耗时与 vLLM 指标。
- `metrics.json.stability`：reward、KL、entropy、loss、clip、grad norm、学习率和对齐窗口。
- `metrics.json.resource`：每卡容量、各阶段绝对 MiB 峰值、最坏 GPU、monitor 覆盖和绝对 reserve 策略。

`parameter_groups.json` 由 Validator 的同一份白名单生成。ref 的 TP/PP/SP 归入 `ignored`，因为 ref 实际继承 actor 拓扑；`entropy_coeff` 和 `rollout.n` 通过 `cross_effects` 明确记录跨稳定性、吞吐和显存影响。

`log_facts.json` 由 `extract_trial_metrics.py` 所属的统一提取链生成，不属于 memory estimator。train-log parser 在同一遍读取中落盘模型 config、Megatron resolved runtime、去重后的 rank 参数量和稳定 step 的有效序列长度；后续 memory estimator 与 Proposal immutable context 只消费该结构化文件，不直接读取 `train.log`。

## 12. 常用命令

```bash
# 只生成命令
PLATFORM=C550 bash run_circle.sh --dry-run --rules-only

# 运行一个 trial
PLATFORM=C550 MAX_TRIALS=1 bash run_circle.sh

# 跳过 hardware，使用新输出目录直接调试 stability
PLATFORM=C550 START_STAGE=stability_tuning \
  OUTPUT_PATH=/absolute/new-run MAX_TRIALS=1 bash run_circle.sh

# 继续同一实验时必须复用 OUTPUT_PATH
PLATFORM=C550 OUTPUT_PATH=/absolute/run MAX_TRIALS=5 bash run_circle.sh

# 从具体 trial artifacts 重建 Proposal Prompt，不启动训练
python replay_agent_prompts.py \
  --trial-dir output/某次实验/trials/0004 --render-only

# 重新提取一个 trial 的分类指标
python tools/extract_trial_metrics.py --trial-dir output/某次实验/trials/0001 \
  --agent-config config/agent_config.json

# 单元测试
python -m unittest discover -s tests -p 'test_*.py'
```

## 13. 接手时只需记住

1. Agent 只建议，Validator、orchestrator 和 runner 才有执行权。
2. 每个候选有自己的 reference，`from` 必须与它严格一致。
3. Feasibility 只能选择 candidate ID，不能改参数。
4. 显存工具只保留 `memory_estimator_V3.py`，输出分阶段状态、参考峰值、预测峰值和相对变化百分比；内部 upper 只用于顶层 `safety` 判断。
5. 实际预算来自 `agent_config.json`，当前为 5/50/135；confirm 的 135 是全局目标 step。
6. stability 早停是“规则触发 → Agent 复核 → runner 执行”。
7. vLLM 监控必须显式设置 `disable_log_stats=false`。
8. `trials.jsonl` 是权威轻量索引，完整事实位于每个 trial 的 artifact 文件中。
9. stability checkpoint 是 trial artifact；confirm 的 `resume` 元数据记录来源 trial、恢复 step 和本次实际执行 update 数。

排查一次决策时沿这条链即可：

```text
trials.jsonl
  → parameters.json / parameter_groups.json
  → metrics.json 的分类证据
  → decision.json / agent_trace.json
  → canonical candidate
  → command.json
  → train.log + GPU/vLLM/health 数据
  → trial_report.json
  → 下一轮 trials.jsonl
```
