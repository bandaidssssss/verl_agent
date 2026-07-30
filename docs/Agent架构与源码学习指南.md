# verl Stage Tuning Agent — 架构与源码学习指南

> 逐文件源码解读，覆盖全部 Python 模块、配置、提示词模板、CLI 脚本与测试。适合已经了解 verl/GRPO 基本概念、想深入理解这个 Agent 系统如何实现的开发者。

---

## 目录

1. [项目概览](#1-项目概览)
2. [整体架构与数据流](#2-整体架构与数据流)
3. [入口与调度层](#3-入口与调度层)
   - [3.1 run_agent.py — 主入口](#31-run_agentpy--主入口)
   - [3.2 run_circle.sh — Shell 包装器](#32-run_circlesh--shell-包装器)
   - [3.3 orchestrator.py — 调优状态机](#33-orchestratorpy--调优状态机)
4. [Agent 框架层](#4-agent-框架层)
   - [4.1 agents.py — LLM Agent 核心](#41-agentspy--llm-agent-核心)
   - [4.2 prompting.py — 提示词渲染](#42-promptingpy--提示词渲染)
   - [4.3 prompts/ — 角色提示词模板](#43-prompts--角色提示词模板)
5. [工具系统（agent_tools/）](#5-工具系统agent_tools)
   - [5.1 registry.py — ToolRegistry 工具注册中心](#51-registrypy--toolregistry-工具注册中心)
   - [5.2 memory_estimator.py — 分阶段显存估算器](#52-memory_estimatorpy--分阶段显存估算器)
   - [5.3 skills.json — 工具白名单与函数签名](#53-skillsjson--工具白名单与函数签名)
   - [5.4 parameter_docs.json — 参数知识库](#54-parameter_docsjson--参数知识库)
   - [5.5 tuning_strategies.json — 调优策略库](#55-tuning_strategiesjson--调优策略库)
6. [执行与监控层](#6-执行与监控层)
   - [6.1 runner.py — Trial 执行与 GPU 监控](#61-runnerpy--trial-执行与-gpu-监控)
   - [6.2 metrics.py — 训练指标分析引擎](#62-metricspy--训练指标分析引擎)
7. [验证与约束层](#7-验证与约束层)
   - [7.1 validator.py — 确定性参数校验](#71-validatorpy--确定性参数校验)
8. [配置与基础工具层](#8-配置与基础工具层)
   - [8.1 config_utils.py — 通用 IO 与 Hydra 转换](#81-config_utilspy--通用-io-与-hydra-转换)
   - [8.2 config/base_parameters.json — verl 基线参数](#82-configbase_parametersjson--verl-基线参数)
   - [8.3 config/agent_config.json — Agent 行为配置](#83-configagent_configjson--agent-行为配置)
9. [CLI 独立脚本](#9-cli-独立脚本)
   - [9.1 role_cli.py — 单角色执行](#91-role_clipy--单角色执行)
   - [9.2 trial_cli.py — 单 Trial 执行](#92-trial_clipy--单-trial-执行)
   - [9.3 monitor_cli.py — 离线日志分析](#93-monitor_clipy--离线日志分析)
   - [9.4 tools/compare_end_to_end_reward.py — 最终验收对比](#94-toolscompare_end_to_end_rewardpy--最终验收对比)
10. [测试体系](#10-测试体系)
    - [10.1 test_orchestrator.py](#101-test_orchestratorpy)
    - [10.2 test_agents.py](#102-test_agentspy)
    - [10.3 test_agent_tools.py](#103-test_agent_toolspy)
    - [10.4 test_validator.py](#104-test_validatorpy)
11. [核心设计模式与技巧](#11-核心设计模式与技巧)
12. [源码阅读路线建议](#12-源码阅读路线建议)

---

## 1. 项目概览

**verl Stage Tuning Agent** 是一个基于 LLM 的多阶段强化学习超参数自动调优系统，参考了 OptiCo 的设计思想。它针对 **verl 0.7 GRPO**（Group Relative Policy Optimization）训练进行自动参数优化。

### 核心能力一句话总结

> 状态机决定当前阶段 → Proposal Agent 在同一对话中调用工具并提出候选 → Validator 检查硬约束 → Feasibility Agent 独立审查 → 拒绝证据回灌 → 通过后启动短跑训练 → 真实指标写回历史。

### 两个嵌套循环

理解这个系统最关键的是理解**两个不同层次的循环**:

1. **内层 — LLM 主动工具调用循环**（`agents.py` `LLMRoleAgent.run()`）: Agent 在一次推理中可以多次调用工具（查询参数、估算显存、搜索文档），每次工具结果追加到对话中，直到 Agent 认为证据足够后输出最终 JSON。

2. **外层 — Proposal 与 Validator/Feasibility 对抗循环**（`orchestrator.py` `_propose_candidate()`）: Proposal 被拒绝后，拒绝原因注入同一个对话，Agent 看到自己的失败过程后重新提议，最多 3 轮。

### 技术栈

| 组件 | 技术 |
|------|------|
| 语言 | Python 3.10+ |
| LLM 调用 | OpenAI SDK（兼容 Responses API，含 function tools） |
| 训练框架 | verl 0.7 + Megatron + vLLM |
| 配置系统 | Hydra（通过命令行 override） |
| GPU 监控 | xpu-smi（V5000）/ nvidia-smi（NVIDIA/C550） |

---

## 2. 整体架构与数据流

```text
┌─ run_circle.sh ─────────────────────────────────────────────┐
│  环境变量 → 平台检测 → 选择环境脚本 → 启动 Python            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─ run_agent.py ──────────────────────────────────────────────┐
│  解析 CLI 参数 → 加载两个 JSON 配置 → 创建 TuningOrchestrator │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─ TuningOrchestrator ────────────────────────────────────────┐
│                                                              │
│  run() 主循环:                                               │
│    ┌─ determine_stage(trials) ─── 读历史 → 判定阶段          │
│    │   ↓                                                     │
│    ├─ _starting_parameters() ─── 选基线参数                  │
│    │   ↓                                                     │
│    ├─ _propose_candidate() ─── Agent 协作闭环                │
│    │   │  ┌─ Diagnosis Agent (如上次失败)                    │
│    │   │  ├─ Proposal Agent (LLM + 7 工具)                  │
│    │   │  ├─ validate_candidate() (确定性规则)               │
│    │   │  └─ Feasibility Agent (LLM 审查)                   │
│    │   │     被拒 → rejection_feedback 注入对话 → 重试       │
│    │   ↓                                                     │
│    ├─ run_trial() ─── 启动训练 + GPU 监控 + 分析             │
│    │   ↓                                                     │
│    └─ 追加 trials.jsonl, 更新 state.json                    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 关键数据文件

| 文件 | 格式 | 内容 |
|------|------|------|
| `output/trials.jsonl` | JSONL（追加） | 所有 trial 参数、指标、Agent trace |
| `output/state.json` | JSON | 当前阶段、最后 trial ID |
| `output/last_agent_rejection.json` | JSON | 多轮仍失败时的完整拒绝轨迹 |
| `output/final_result.json` | JSON | confirm 阶段最终结果 |
| `output/trials/NNNN/train.log` | 文本 | verl 原始日志 |
| `output/trials/NNNN/gpu_samples.csv` | CSV | 逐 GPU 每秒采样 |
| `output/trials/NNNN/trial_report.json` | JSON | 单 trial 结构化报告 |
| `output/trials/NNNN/parameters.json` | JSON | 该 trial 使用的参数 |
| `output/trials/NNNN/command.json` | JSON | 执行的命令和 cwd |

---

## 3. 入口与调度层

### 3.1 `run_agent.py` — 主入口

**文件**: [run_agent.py](../run_agent.py) | 约 50 行

这是整个系统的主入口，职责非常清晰——解析参数、合并环境变量、创建 Orchestrator 并启动。

```python
# 关键代码段 (lines 11-12): 包路径处理
# 当 python3 run_agent.py 直接执行时 __package__ 为 None，
# 手动将项目根目录加入 sys.path，保证相对导入 (from config_utils import ...) 正常工作
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
```

**环境变量覆盖优先级**（lines 32-41）:

```python
# 环境变量 > 配置文件，适用于 CI/CD 和不同平台灵活切换
if os.getenv("PLATFORM"):
    agent_config["platform"] = os.environ["PLATFORM"]
if os.getenv("VERL_ROOT"):
    agent_config["verl_root"] = os.environ["VERL_ROOT"]
if os.getenv("OUTPUT_PATH"):
    agent_config["output_dir"] = os.environ["OUTPUT_PATH"]
```

**`--rules-only` 模式**（line 40-41）: 设置 `agent_mode = "rules"`，跳过所有 LLM API 调用。Agent 返回确定性结果（`decision: "keep"`）。用于:
- 基线测试（不调参，直接跑）
- 确定性校验测试
- 快速验证基础设施是否正常工作

**命令行参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--base-config` | `config/base_parameters.json` | verl Hydra 基础参数 |
| `--agent-config` | `config/agent_config.json` | Agent 行为配置 |
| `--max-trials` | 1 | 本次最多执行几个 trial |
| `--dry-run` | False | 只生成命令不启动训练 |
| `--rules-only` | False | 跳过 LLM，纯确定性规则 |

### 3.2 `run_circle.sh` — Shell 包装器

**文件**: [run_circle.sh](../run_circle.sh) | 约 25 行

负责三件事:

**① 平台自动检测**（lines 11-18）: 根据 `PLATFORM` 环境变量选择对应的环境脚本（conda 激活、环境变量设置等）。

```bash
case "${PLATFORM_UPPER}" in
    V5000) export VERL_ENV_SCRIPT="${SCRIPT_DIR}/train/env_V5000.sh" ;;
    C550|METAX) export VERL_ENV_SCRIPT="${SCRIPT_DIR}/train/env_C550.sh" ;;
    A100|NVIDIA|CUDA) export VERL_ENV_SCRIPT="${SCRIPT_DIR}/train/env_NVIDIA.sh" ;;
    *) echo "Unsupported PLATFORM=${PLATFORM}" >&2; exit 2 ;;
esac
```

**② 环境变量透传**: `MAX_TRIALS`, `OUTPUT_PATH`, `PYTHONPATH` 全部导出。

**③ 参数转发**: `"$@"` 将 `--dry-run`、`--rules-only` 等全部传给 `run_agent.py`。

### 3.3 `orchestrator.py` — 调优状态机

**文件**: [orchestrator.py](../orchestrator.py) | 约 380 行

这是系统最核心的模块，包含三部分: 阶段判定逻辑、Agent 协作闭环、主循环。

#### 3.3.1 辅助函数

```python
def _metric_mean(trial, *path) -> float | None:
    """从嵌套字典中提取指标的 mean 值。
    例如 _metric_mean(trial, "performance", "throughput") → trial["performance"]["throughput"]["mean"]
    """
    value = trial
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    if isinstance(value, Mapping):
        value = value.get("mean")
    return float(value) if isinstance(value, (int, float)) else None
```

`_compact_trial()`: 从完整 trial 记录中提取 Agent 需要的字段（trial_id, stage, result, parameters, performance, stability 等），去掉原始日志路径等冗余信息。这样发送给 LLM 的上下文更精简。

#### 3.3.2 阶段判定: `determine_stage()`

```python
def determine_stage(trials, config) -> str:
```

决策树（按优先级）:

```text
已有 confirm trial?                                    → "done"
有 hardware trial?    → 无成功者                       → "hardware_repair"
                      → 成功不足 min(2)                 → "hardware_tuning"
                      → 未达 max(6) 且非 plateau       → "hardware_tuning"
检查 stability trial  → 达 max(4) 且无 healthy         → "stopped_unstable"
                      → healthy 达 min(2)              → "confirm"
                      → 否则                            → "stability_tuning"
```

**Plateau 检测** (`hardware_plateaued`, lines 60-72):

```python
def hardware_plateaued(trials, config) -> bool:
    # 最后 plateau_rounds 个成功 trial 的吞吐改善均不超过 2%
    best_before = max(scores[:-plateau_rounds])
    return all(score <= best_before * 1.02 for score in scores[-plateau_rounds:])
```

**稳定性健康检查** (`stability_healthy`, lines 75-84):

```python
def stability_healthy(trial, config) -> bool:
    # 两个条件: 最后两个完整 reward window 的斜率不低于 -0.01，
    # 所有完整 window 的 PPO KL 不超过 0.1
    return (slope >= -0.01) and (kl_max <= 0.1)
```

#### 3.3.3 Agent 协作闭环: `_propose_candidate()`

这是整个系统设计最精巧的部分（lines 191-303）。伪代码:

```python
def _propose_candidate(self, stage, current, trials):
    diagnosis = self._diagnosis(trials)  # 只在上次失败时触发

    proposal_conversation = None  # 关键: 拒绝后复用同一个对话
    for attempt in range(1, max_validation_rounds + 1):  # 默认 3 轮
        # 1. Proposal Agent 推理（首次给 context，后续用已有对话）
        proposal_run = self.agents.propose(
            context if first_attempt else None,
            proposal_conversation,  # 已有对话 → continue
        )
        proposal_conversation = proposal_run.conversation

        # 2. decision="keep"/"stop" → 不修改，直接返回
        if proposal["decision"] in {"keep", "stop"}:
            return current, proposal, {"verdict": "valid"}, trace

        # 3. 应用修改 → 确定性校验
        candidate = apply_changes(current, proposal["changes"])
        validation = validate_candidate(candidate, proposal["changes"], ...)
        if not validation.valid:
            # 拒绝原因注入对话，continue 让 Agent 看到失败
            proposal_conversation.add_user_message(
                rejection_feedback(attempt, proposal, candidate,
                                   "deterministic_validator", validation_result)
            )
            continue

        # 4. Feasibility Agent LLM 审查
        review_run = self.agents.review({candidate 的上下文})
        if review["verdict"] == "valid":
            return candidate, proposal, review, trace

        # 拒绝原因注入对话
        proposal_conversation.add_user_message(
            rejection_feedback(attempt, proposal, candidate,
                               "feasibility_agent", review)
        )

    # 全部失败 → 保存完整轨迹 → 抛出异常
    write_json("output/last_agent_rejection.json", trace)
    raise AgentError(f"no feasible proposal after {max_rounds} rounds")
```

**对话连续性是关键设计**（lines 240-243, 298-300）: 拒绝消息是结构化的 Markdown，包含:
- 第几次建议被拒绝
- 拒绝来源（Validator 还是 Feasibility）
- 原始 proposal JSON
- 修改后的完整参数
- 具体拒绝原因

Agent 在下一轮推理时能看到所有这些，从而不会重复同样的错误。

#### 3.3.4 主循环: `run()`

```python
def run(self, max_trials=1, dry_run=False):
    for _ in range(max_trials):
        trials = self.trials()          # 读取历史
        stage = determine_stage(trials)  # 判定阶段
        if stage in {"done", "stopped_unstable"}:
            break

        parameters = self._starting_parameters(stage, trials)

        # 这些情况跳过 Agent 调用:
        # - 第一个硬件 trial (无历史训练证据的基线)
        # - confirm 阶段 (参数冻结)
        first_hardware = not _hardware_trials(trials)
        if not first_hardware and stage != "confirm":
            parameters, proposal, review, trace = self._propose_candidate(...)
            # decision="keep"/"stop" → 推进阶段
            if proposal["decision"] in {"keep", "stop"}:
                # 尝试进入下一阶段或停止
                ...

        report = run_trial(parameters, self.config, trial_id, stage, updates, dry_run)
        # 持久化
        append_jsonl(self.history_path, report)
```

**基线策略**: 只有第一个硬件 trial 在没有训练证据时直接运行。第一个 stability trial 会读取最佳 hardware trial 的训练稳定性时序后调用 Agent；若 Agent 决定 `keep`，仍会用未修改参数运行该 stability 基线。

**阶段终止条件**:

| 条件 | 动作 |
|------|------|
| `confirm` 完成 | 保存 `final_result.json`，退出 |
| `stopped_unstable` | 退出 |
| Proposal `decision="keep"/"stop"` + 有最佳参数 | 推进到下一阶段 |
| Proposal `decision="keep"/"stop"` + 无最佳参数 | `stopped_no_candidate` |

---

## 4. Agent 框架层

### 4.1 `agents.py` — LLM Agent 核心

**文件**: [agents.py](../agents.py) | 约 307 行

包含五个关键元素: 数据类、JSON 提取、LLMRoleAgent、AgentSet、rules 模式。

#### 4.1.1 数据结构

**`AgentConversation`** (lines 60-82):

```python
@dataclass
class AgentConversation:
    role: str                    # "proposal" | "feasibility" | "diagnosis"
    context: dict                # 创建对话时的上下文快照
    messages: list[dict]         # [{"role": "user"/"assistant"/"system", "content": "..."}]
    tool_trace: list[dict]       # 工具调用记录
    usage: dict                  # {"input_tokens": N, "output_tokens": N, "total_tokens": N, "api_calls": N}
    completed_turns: int         # 完成的 Agent 推理轮数（不含中间工具调用）
```

`as_trace()` 方法全部使用 `copy.deepcopy`，确保返回的轨迹不受后续修改影响。

**`AgentRun`** (lines 85-93):

```python
@dataclass
class AgentRun:
    result: dict              # Agent 最终输出的 JSON
    conversation: AgentConversation
```

#### 4.1.2 JSON 提取: `_extract_json()`

```python
def _extract_json(text: str) -> dict:
    # 策略 1: 匹配 ```json ... ``` 代码块
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.S)

    # 策略 2: 找第一个 { 到最后一个 }
    start, end = stripped.find("{"), stripped.rfind("}")

    # 策略 3: 直接 parse，失败抛 AgentError
    return json.loads(stripped)
```

这种容错设计很重要——LLM 有时会在 JSON 前后加 Markdown 格式或额外解释。

#### 4.1.3 LLMRoleAgent 核心方法

**`run()`** — 工具调用循环（lines 161-238）:

```python
def run(self, context=None, conversation=None) -> AgentRun:
    # 创建或复用对话
    if conversation is None:
        conversation = self.new_conversation(context)

    for tool_round in range(max_tool_rounds + 1):  # 默认最多 7 轮 (6+1)
        request = {
            "model": self.model,
            "input": conversation.messages,  # 完整的对话历史
            "max_output_tokens": 4096,
        }
        # 如果还有工具轮次，附加 tools 定义
        if schemas and tool_round < max_tool_rounds:
            request["tools"] = schemas      # OpenAI function calling 格式
            request["tool_choice"] = "auto"

        response = client.responses.create(**request)  # OpenAI Responses API

        calls = _tool_calls(response)
        if calls:
            # 逐条执行工具，结果作为 user message 注入对话
            for call in calls:
                result = registry.execute(role, name, arguments, runtime)
                conversation.messages.append({
                    "role": "user",
                    "content": f"The result of tool `{name}` is:\n```json\n{json.dumps(result)}\n```"
                })
            continue  # 让 LLM 继续推理

        # 没有工具调用 → 解析最终 JSON
        return AgentRun(_extract_json(_response_text(response)), conversation)
```

**关键设计细节**:

- 工具结果以**代码块**形式返回，方便 LLM 阅读结构化数据
- `tool_trace` 记录每次调用的参数、结果和状态（success/error）
- 错误时不中断循环，而是将错误信息返回给 LLM 让其自行修正
- API 使用 OpenAI Responses API（`client.responses.create`），因为需要 function tools 支持

**`_get_client()`** — LLM 客户端初始化（lines 115-136）:

```python
kwargs = {
    "api_key": self.api_key,              # API_KEY 或 OPENAI_API_KEY
    "timeout": 120.0,                     # llm_timeout_seconds
    "max_retries": 0,                     # 由 Agent 应用层统一执行有限重试
    "default_headers": {"User-Agent": "curl/7.81.0"},  # 模拟 curl
}
if self.base_url:
    kwargs["base_url"] = self.base_url    # 支持自定义端点
self._client = OpenAI(**kwargs)
```

**`client_factory` 注入**（line 112）: 构造函数接受可选的 `client_factory` 参数。在测试中，通过注入 `FakeClient` 完全绕过网络调用，测试 Agent 的推理逻辑。这是依赖注入模式的经典应用。

#### 4.1.4 AgentSet

```python
class AgentSet:
    def __init__(self, root, mode, agent_config, history_path):
        self.registry = ToolRegistry(root, self.config, self.history_path)
        # 三个角色各自独立的 LLMRoleAgent 实例
        self.proposal   = LLMRoleAgent("proposal",   "prompts/proposal.md",   ...)
        self.feasibility = LLMRoleAgent("feasibility", "prompts/feasibility.md", ...)
        self.diagnosis  = LLMRoleAgent("diagnosis",   "prompts/diagnosis.md",   ...)
```

统一入口:
- `propose(context, conversation=None)` — 支持新对话和继续已有对话
- `review(context)` — 审查候选配置
- `diagnose(context)` — 失败归因

**Rules 模式实现** (lines 261-306):

```python
if self.mode == "rules":
    return self._rules_run("proposal", rule_context,
        {"decision": "keep", "reason": "rules mode keeps the current configuration",
         "changes": {}})
```

当 `--rules-only` 或 `agent_mode: "rules"` 时，所有 Agent 方法返回确定性结果，跳过 LLM API 调用。Diagnosis 在 rules 模式下做确定性分类:
- 已知错误类型 → confidence=1.0
- `UNKNOWN_FAILURE` → confidence=0.3

### 4.2 `prompting.py` — 提示词渲染

**文件**: [prompting.py](../prompting.py) | 约 122 行

#### 核心函数

**`render_prompt(template, context, tool_definitions)`** — 模板变量替换:

```python
replacements = {
    "CURRENT_STAGE": f"`{context.get('current_stage', 'unknown')}`",
    "MODE": f"`{context.get('mode', 'unknown')}`",
    "CURRENT_PARAMETERS": json_block(context.get("current_parameters")),
    "EDITABLE_PARAMETERS": json_block(context.get("editable_parameters")),
    "CONSTRAINTS": json_block(context.get("constraints")),
    "DIAGNOSIS": json_block(context.get("diagnosis")),
    "TRIAL_HISTORY": trial_history_table(context.get("recent_trials", [])),
    "AVAILABLE_TOOLS": available_tools_markdown(tool_definitions),
    # ... 更多变量
}
# 最后清理未替换的占位符 → "未提供"
return re.sub(r"\{[A-Z][A-Z0-9_]*\}", "未提供", rendered)
```

**`json_block(value)`** — 格式化 JSON 为 Markdown 代码块:
```python
def json_block(value):
    if value in (None, {}, []):
        return "未提供"
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n```"
```

**`trial_history_table(trials)`** — 将历史转为表格:
```
|Trial|阶段|结果|修改|吞吐|Step(s)|显存瓶颈|峰值显存%|失败类型|
|---:|---|---|---|---:|---:|---|---:|---|
| 1 | hardware_tuning | success | baseline/keep | 12.3 | 45.2 | training | 78.5 | - |
```

表格化让 LLM 能快速比较不同 trial，而不需要解析嵌套 JSON。

**`rejection_feedback(attempt, proposal, candidate, source, result)`** — 构造拒绝反馈:

```python
return (
    f"## 第 {attempt} 次建议被拒绝\n\n"
    f"- 拒绝来源：`{source}`\n\n"           # deterministic_validator | feasibility_agent
    f"- Proposed changes:\n{json_block(proposal)}\n\n"
    f"- The modified parameters are:\n{json_block(candidate)}\n\n"
    f"- Validation result:\n{json_block(result)}\n\n"
    "请把这次失败及其原因作为后续推理证据。你可以继续调用工具核查参数、显存或 verl 文档，"
    "然后提出不同且可行的新建议。最终仍只输出约定的 JSON 对象。"
)
```

### 4.3 `prompts/` — 角色提示词模板

三个 Markdown 文件定义角色行为规范。关键设计原则: **中文系统提示词 + 英文 JSON 输出字段名**。

#### 4.3.1 `prompts/proposal.md`

**输出格式**:
```json
{
  "decision": "modify|keep|stop",
  "reason": "基于观测证据的简短因果说明",
  "changes": {"完整 Hydra 参数名": "新值"},
  "expected_effect": {"指标": "increase|decrease|stable"},
  "confidence": 0.0
}
```

**阶段特定指令**:
- `hardware_repair`: 只修复 diagnosis 指明的子阶段，优先降低资源压力
- `hardware_tuning`: 端到端吞吐是目标，只调整当前瓶颈参数组
- `stability_tuning`: 冻结硬件参数，根据 reward/KL/entropy 调整优化行为
- `confirm`: 不修改

**安全约束**:
- 一次修改不超过 `max_parameter_changes`
- 整除关系联动计入数量
- 不能输出历史中已有完整配置
- 被拒绝后必须正面处理原因，不原样重复

#### 4.3.2 `prompts/feasibility.md`

**输出格式**:
```json
{
  "verdict": "valid|invalid",
  "reason": "基于独立证据的简短说明",
  "risks": ["仍需由短跑测试验证的风险"],
  "predicted_memory_pct": {"rollout": null, "actor_log_prob": null, "ref_log_prob": null, "training": null}
}
```

**核心约束**:
- 必须独立查询参数影响，不直接相信 Proposal 的理由
- Hardware 候选必须调用 `memory_estimator`
- 检查 actor、rollout、ref 共置时的跨阶段影响
- 拒绝局部改善但可能降低端到端吞吐的修改
- Stability 阶段修改硬件参数 → invalid

#### 4.3.3 `prompts/diagnosis.md`

**预定义标签**:
```
OOM_ROLLOUT, OOM_ACTOR_LOGPROB, OOM_REF_LOGPROB, OOM_TRAINING,
MEMORY_HEADROOM_EXCEEDED, NCCL_OR_DISTRIBUTED_FAILURE, NAN_OR_INF,
KL_EXPLOSION, REWARD_COLLAPSE,
LOW_THROUGHPUT_ROLLOUT/ACTOR_LOGPROB/REF/TRAINING,
UNKNOWN_FAILURE
```

提供了明确的诊断优先级: 先看结构化指标（`failure_phase`, `memory_bottleneck`），不足时调用 `read_trial_log_excerpt`，不确定时降低 confidence 而不是编造。

---

## 5. 工具系统（agent_tools/）

### 5.1 `registry.py` — ToolRegistry 工具注册中心

**文件**: [registry.py](../agent_tools/registry.py) | 约 378 行

工具注册中心是 Agent 与外部世界交互的唯一通道。没有通用 shell 工具，所有工具都有固定参数和输出格式。

#### 5.1.1 ToolRuntime

```python
@dataclass(frozen=True)
class ToolRuntime:
    root: Path                    # 项目根目录
    agent_config: Mapping         # Agent 配置
    context: Mapping              # 当前上下文（包含 current_parameters 等）
    history_path: Path            # trials.jsonl 路径
```

`frozen=True` 确保工具执行不会意外修改运行时状态。

#### 5.1.2 初始化和工具注册

```python
class ToolRegistry:
    def __init__(self, root, agent_config, history_path):
        self._skill_config = _load_json("agent_tools/skills.json")
        self._parameter_docs = _load_json("agent_tools/parameter_docs.json")["data"]
        self._strategies = _load_json("agent_tools/tuning_strategies.json")["data"]

        # 工具名 → 处理函数的映射
        self._handlers = {
            "parameter_understanding": self._parameter_understanding,
            "tuning_strategies": self._tuning_strategies,
            "memory_estimator": self._memory_estimator,
            "live_gpu_snapshot": self._live_gpu_snapshot,
            "search_verl_docs": self._search_verl_docs,
            "query_trial_history": self._query_trial_history,
            "read_trial_log_excerpt": self._read_trial_log_excerpt,
        }
```

#### 5.1.3 权限控制

```python
def execute(self, role, name, arguments, runtime):
    allowed = {skill["name"]: skill for skill in self.definitions(role)}
    if name not in allowed or name not in self._handlers:
        raise ToolError(f"tool {name!r} is not allowed for role {role!r}")
```

每个工具在 `skills.json` 中定义了 `roles` 列表。例如 `read_trial_log_excerpt` 只有 `["diagnosis"]`，Proposal Agent 即使构造同名调用也会被拒绝。

#### 5.1.4 七个工具详解

**① `parameter_understanding`** (lines 106-113):
```python
def _parameter_understanding(self, arguments, runtime):
    items = arguments.get("items")             # 1-8 个 Hydra 参数名
    found = {item: self._parameter_docs[item] for item in items if item in self._parameter_docs}
    missing = [item for item in items if item not in self._parameter_docs]
    return {"parameters": found, "unknown_parameters": missing}
```

**② `tuning_strategies`** (lines 115-122):
```python
def _tuning_strategies(self, arguments, runtime):
    items = arguments.get("items")             # 1-4 个场景名
    found = {item: self._strategies[item] for item in items if item in self._strategies}
    return {"strategies": found, "unknown_strategies": missing, "available": sorted(self._strategies)}
```

**③ `memory_estimator`** (lines 124-160):
```python
def _memory_estimator(self, arguments, runtime):
    # 从 context 提取当前参数和候选参数
    current = runtime.context.get("current_parameters") or ...
    candidate = dict(runtime.context.get("candidate_parameters") or current)
    candidate.update(arguments.get("changes", {}))

    # 获取显存限制阈值
    limit = float(limits.get("throughput") or limits.get("resource") or 92.0)

    # 调用核心估算逻辑（agent_tools/memory_estimator.py）
    result = estimate_phase_memory(current, candidate, trials, limit, reference_id)
    return result
```

**④ `live_gpu_snapshot`** (lines 162-236):
```python
def _live_gpu_snapshot(self, arguments, runtime):
    # 不接受任何参数（arguments 必须为空）
    if arguments:
        raise ToolError("live_gpu_snapshot takes no arguments")

    # 自动检测平台 → 选择 xpu-smi 或 nvidia-smi
    executable = os.getenv("GPU_SMI") or shutil.which(default)
    # 执行查询，超时 1-10 秒
    # 对 V5000 有旧版 xpu-smi 回退解析
    # 返回每张 GPU 的显存和利用率 + summary
```

重要: 返回值明确包含 `"interpretation": "Instantaneous host occupancy only; use trial phase samples for tuning decisions."`

**⑤ `search_verl_docs`** (lines 238-293):
```python
def _search_verl_docs(self, arguments, runtime):
    query = arguments["query"]
    # 搜索范围: verl/trainer/config/, verl/workers/, docs/, examples/
    # 文件类型: .py, .yaml, .yml, .md, .rst
    # 上限: 5000 文件, 单文件 1MB
    # 评分: 完整短语匹配=20分 + 每个词条=1分
    # 返回前 N 个匹配 + 上下文片段（±1 行）
```

这是一个**本地 grep 实现**，不依赖外部搜索引擎。关键词长度 ≥3 才计入评分，避免噪声。

**⑥ `query_trial_history`** (lines 295-340):
```python
def _query_trial_history(self, arguments, runtime):
    # 支持按 stage/result/failure_type 筛选
    # 支持按 trial_id/throughput/reward/memory 排序
    # 限制 1-10 条
    # 可选 include_parameters（返回完整参数）
```

**⑦ `read_trial_log_excerpt`** (lines 342-377):
```python
def _read_trial_log_excerpt(self, arguments, runtime):
    # 安全: 日志路径必须在 output 目录下（防止路径穿越）
    log_path.relative_to(allowed_root)  # 不通过抛异常
    # 支持正则过滤或返回尾部 N 行
    # diagnosis 角色专用
```

### 5.2 `memory_estimator.py` — 分阶段显存估算器

**文件**: [memory_estimator.py](../agent_tools/memory_estimator.py)

这是项目中**算法最密集**的模块。核心思想不是精确模拟，而是以显式
`reference_trial_id` 的分阶段实测峰值为锚点，进行**分量感知的相对投影**。

#### 5.2.1 四个阶段的不同压力模型

verl GRPO 的显存不是单一阶段，四个子阶段有不同的主要影响因素:

```python
PHASES = ("rollout", "actor_log_prob", "ref_log_prob", "training")
```

**Rollout 阶段**:

```python
# vLLM 的显存预算是 rollout 峰值的主控量
budget_ratio = (candidate_util / reference_util) ** 0.86
```

`0.86` 来自当前 C550 历史点
`0.5→0.7→0.8 == 59.15%→79.06%→88.97%` 的局部校准。
`max_num_batched_tokens`、`max_num_seqs`、prefix caching 等是容量上限或调度
行为；没有匹配历史时不再直接缩放点估计，而是进入
`uncalibrated_changes` 并扩大不确定区间。

**Actor/Ref Log-Prob 阶段**:

```python
total_ratio = (
    0.55                              # 固定 runtime/allocator
    + 0.25 * sharded_model_ratio      # 模型分片
    + 0.20 * activation_ratio         # micro/token/并行与 packing
)
```

micro batch、sequence length 和 TP/PP 只缩放 activation/model 分量，不再乘到
整个阶段峰值。param offload 等没有匹配历史的跨阶段行为只增加不确定性。
Ref 阶段因现有采样自身波动更大，使用比 Actor 更宽的基础上界余量。

**Training 阶段**:

```python
total_ratio = (
    0.30                              # 固定 runtime
    + 0.25 * model_gradient_ratio     # 模型与梯度
    + 0.25 * optimizer_ratio          # optimizer/distributed/offload
    + 0.20 * activation_ratio         # micro/token/recompute
)
```

distributed optimizer 和 optimizer offload 只影响 optimizer 分量；recompute
只影响 activation 分量，从而避免多个布尔因子连乘后错误地将整个 training
峰值压低六成以上。

#### 5.2.2 经验锚点投影

```python
def estimate_phase_memory(current_parameters, candidate_parameters, trials, ...):
    # 1. 找显式 reference trial
    reference = _reference_trial(current_parameters, trials, reference_trial_id)

    # 2. 对每个阶段计算分量比例
    for phase in PHASES:
        projection = _phase_projection(phase, reference_params, candidate_params)
        projected = reference_peak_memory * projection["ratio"]
        upper_bound = projected + projection["uncertainty_pct"]

        # 3. 风险根据不确定性上界，而不是点估计判断
        if upper_bound >= memory_limit_pct:       risk = "high"
        elif memory_limit_pct - upper_bound < 5:  risk = "watch"
        else:                                     risk = "low"
```

每个阶段会同时返回：

- `projected_pct`：点估计
- `uncertainty_pct` / `upper_bound_pct`：保守上界
- `risk`：按上界计算的风险
- `confidence`：阶段置信度
- `component_shares` / `drivers`：比例来源
- `uncalibrated_changes`：没有匹配历史、未进入点估计的参数

#### 5.2.3 工具输入和参考 Trial 校验

Agent 调用必须同时提供：

```json
{
  "changes": {"parameter": {"from": 0.5, "to": 0.7}},
  "parameters": {"parameter": 0.7},
  "reference_trial_id": 1
}
```

registry 会检查 `from` 是否与参考 Trial 严格一致、`parameters` 是否与 `to`
一致，以及参考 Trial 是否存在分阶段显存。非法对象不再静默退回默认值。

#### 5.2.4 局限性声明

1. 分量比例仍是知识先验，不是 tensor-allocation 模拟
2. scheduler cap、跨阶段 offload 等缺少配对历史时只进入不确定性
3. 配置最大长度只是 workload 上界，实际 token 分布可能更低
4. 实时 SMI 快照不能替代分阶段采样
5. 真实 short resource-gate trial 仍是最终内存权威

### 5.3 `skills.json` — 工具白名单与函数签名

**文件**: [skills.json](../agent_tools/skills.json)

7 个工具的完整元数据。每个工具定义:
- `name`: 暴露给 LLM 的函数名
- `handler`: 本地 Python 方法名
- `roles`: 允许使用的角色列表（proposal/feasibility/diagnosis）
- `description`: 自然语言描述，用于 LLM 理解何时调用
- `parameters`: JSON Schema 格式的参数约束，直接传给 OpenAI function calling

**角色权限矩阵**:

| 工具 | Proposal | Feasibility | Diagnosis |
|------|:---:|:---:|:---:|
| parameter_understanding | ✓ | ✓ | ✓ |
| tuning_strategies | ✓ | ✓ | ✓ |
| memory_estimator | ✓ | ✓ | ✗ |
| live_gpu_snapshot | ✓ | ✓ | ✓ |
| search_verl_docs | ✓ | ✓ | ✓ |
| query_trial_history | ✓ | ✓ | ✓ |
| read_trial_log_excerpt | ✗ | ✗ | ✓ |

### 5.4 `parameter_docs.json` — 参数知识库

**文件**: [parameter_docs.json](../agent_tools/parameter_docs.json)

约 40 个 Hydra 参数的详细文档，每个参数包含:

| 字段 | 例值 | 说明 |
|------|------|------|
| `stage` | `"hardware"` | 所属阶段 |
| `type` | `"int"` / `"float"` / `"bool"` / `"string"` | 数据类型 |
| `impact` | `"Per-GPU PPO update micro batch..."` | 参数影响概述 |
| `increase` | `"Usually improves training utilization..."` | 增大参数的影响 |
| `decrease` | `"Lowers training memory..."` | 减小参数的影响 |
| `interactions` | `["actor.ppo_mini_batch_size", ...]` | 关联参数 |
| `constraints` | `["ppo_mini_batch_size must be divisible..."]` | 硬约束 |

这是一个**结构化的领域知识库**。LLM 不确定参数语义时查询这个，而不是凭空猜测。

### 5.5 `tuning_strategies.json` — 调优策略库

**文件**: [tuning_strategies.json](../agent_tools/tuning_strategies.json)

10 种调优场景的策略模板，每个包含 `evidence`（触发条件）、`ordered_actions`（优先级操作序列）和 `do_not`（禁止操作）。

| 场景 | 阶段 | 核心建议 |
|------|------|----------|
| `rollout_memory_pressure` | hardware | 降 gpu_memory_utilization → 降 batched_tokens/segs → 启用 chunked prefill |
| `actor_logprob_memory_pressure` | hardware | 降 log_prob micro batch → 保持 remove_padding → 考虑 actor param offload |
| `ref_logprob_memory_pressure` | hardware | 降 ref micro batch → 保持 ref param_offload → 增 ref TP/PP |
| `training_memory_pressure` | hardware | 降 PPO micro batch → 增强 recompute → 开启 sequence_parallel/offload |
| `low_gpu_utilization` | hardware | 找最慢阶段 → 只增加该阶段控制的参数 → 一次一个整除步 |
| `communication_bottleneck` | hardware | 保持最小可行 TP/PP → 比较端到端时间而非单阶段 |
| `end_to_end_throughput` | hardware | 优化 time_bottleneck → 保持 token budget 固定 → 拒绝局部优化 |
| `kl_explosion` | stability | 降 lr → 增 KL loss coef → 加 warmup |
| `reward_collapse` | stability | 降 lr 优先 → 检查 KL/entropy/pg_loss 联动 → 不碰硬件参数 |
| `platform_portability` | hardware | 平台脚本在 LLM 参数空间之外 → 换硬件必须重跑 Resource Gate |

---

## 6. 执行与监控层

### 6.1 `runner.py` — Trial 执行与 GPU 监控

**文件**: [runner.py](../runner.py) | 约 290 行

负责启动训练进程、实时监控 GPU、日志分析和安全终止。

#### 6.1.1 `build_command()` — 构建 verl 命令

```python
def build_command(parameters, agent_config, trial_id, updates):
    # 1. 仅注入 Trial 运行步数；experiment_name、logger、save_freq、
    #    test_freq、val_before_train 沿用 base/current parameters
    run_parameters["trainer.total_training_steps"] = updates

    # 2. 构建 Hydra 命令行
    command = ["python3", "-m", "verl.trainer.main_ppo",
               f"--config-path={...}", f"--config-name={...}",
               *hydra_overrides(run_parameters)]

    # 3. 可选: 通过环境脚本包装
    # 使用 bash -lc + source 在子进程中激活环境
    if environment_script:
        command = ["bash", "-lc", 'source "$1"; shift; exec "$@"',
                   "verl-agent", script, *command]
```

**安全点**: `save_freq=-1` 和 `test_freq=-1` 是重要的默认覆盖——调优 trial 不需要 checkpoint 和验证，节省磁盘和时间。

#### 6.1.2 `PhaseTracker` — 训练阶段实时跟踪

```python
class PhaseTracker:
    PHASE_START = {
        "Before generate_sequences": "rollout",
        "Before compute_log_prob": "actor_log_prob",
        "Before compute_ref_log_prob": "ref_log_prob",
        "Before update_actor": "training",
    }
    PHASE_END = ("After generate_sequences", "After compute_log_prob", ...)
```

通过扫描 verl 日志中的 `GPUMemoryLogger` 阶段标记来跟踪。使用 `threading.Lock` 保证与 GPU sampler 线程的线程安全。

#### 6.1.3 `GPUSampler` — GPU 监控线程

```python
class GPUSampler(threading.Thread):
    def run(self):
        with open(output_path, "w") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "phase", "gpu_index",
                            "memory_used_mb", "memory_total_mb", "utilization_pct"])
            while not stop_event.is_set():
                for fields in self._query_rows():  # 调用 nvidia-smi / xpu-smi
                    writer.writerow([now, tracker.get(), gpu_index,
                                    used, total, utilization])
                time.sleep(interval)  # 每秒一次
```

**平台兼容**:
- V5000: `xpu-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits`
- 旧版 V5000 xpu-smi 回退: 解析人类可读表格格式
- NVIDIA/C550: `nvidia-smi` 相同参数

**安全特性**:
- `daemon=True` — 主进程退出时自动终止
- 采集失败不崩溃 — `try/except (OSError, ValueError, SubprocessError): pass`
- 查询超时 `max(1.0, interval)` — 防止 SMI 命令挂起

#### 6.1.4 `run_trial()` — 主执行流程

```python
def run_trial(parameters, agent_config, trial_id, stage, updates, dry_run=False):
    # 1. 创建输出目录 output/trials/NNNN/

    # 2. dry_run → 返回命令信息，不执行
    if dry_run:
        return {"trial_id": ..., "command": command, "cwd": str(cwd), ...}

    # 3. 启动训练子进程
    process = subprocess.Popen(
        command, cwd=cwd, env=env,
        stdout=PIPE, stderr=STDOUT,   # 合并 stderr → stdout
        text=True, bufsize=1,          # 行缓冲
        start_new_session=True,        # 新进程组（便于 kill 整个进程树）
    )

    # 4. 启动 GPU sampler 线程
    sampler.start()

    # 5. 逐行读取 stdout:
    gate_updates = 5  # Resource Gate 前 5 步
    hard_limit = 95.0  # 显存硬上限
    for line in process.stdout:
        log_handle.write(line)
        tracker.update_from_log(line)

        # 致命日志检测 (OOM, NCCL 错误)
        if FATAL_RE.search(line):
            stop_reason = "fatal_log"
            _terminate(process)  # SIGTERM → 10s → SIGKILL
            break

        # Resource Gate 检测
        if step >= gate_updates and max_memory_pct >= hard_limit:
            stop_reason = "resource_gate_memory_limit"
            _terminate(process)
            break

    # 6. 停止 sampler，等待进程结束

    # 7. 调用 metrics.analyze_trial() 分析
    metrics = analyze_trial(log_path, samples_path, ...)

    # 8. 补充元数据
    metrics["trial_id"] = trial_id
    metrics["failure_phase"] = tracker.get()  # 失败时的训练阶段

    # 9. 显存余量检查（throughput 模式用更严格阈值）
    if observed_memory > throughput_memory_limit_pct (92%):
        metrics["error"] = {"type": "MEMORY_HEADROOM_EXCEEDED", ...}

    return metrics
```

**进程终止** (`_terminate`, lines 161-171):
```python
def _terminate(process):
    os.killpg(process.pid, signal.SIGTERM)   # 先优雅终止
    process.wait(timeout=10)
    # 超时 → 强制 kill
    os.killpg(process.pid, signal.SIGKILL)
```

使用 `os.killpg` 而非 `process.kill()`，因为 verl 会启动多个子进程（Ray workers 等）。

**Resource Gate**（lines 228-229, 242-244）: 前 5 个 update 是资源门控期。如果在此期间的显存已达到 95% 硬上限，立即终止——继续下去只会 OOM。

**FATAL 日志模式**（lines 20-24）:
```python
FATAL_RE = re.compile(
    r"CUDA out of memory|OutOfMemoryError|ChildFailedError|DistBackendError|"
    r"NCCL[^\n]{0,80}(?:WARN|ERROR|unhandled|failed|failure)",
    re.I,
)
```

### 6.2 `metrics.py` — 训练指标分析引擎

**文件**: [metrics.py](../metrics.py) | 约 232 行

离线分析训练日志和 GPU 采样数据，生成结构化报告。

#### 6.2.1 核心解析函数

**`parse_step_records(log_path)`** — 解析 verl 日志中的 step 级指标:

```python
# 日志格式: "step:5 ... critic/rewards/mean:0.23 perf/throughput:12.3 ..."
STEP_RE = re.compile(r"step:(\d+)")
PAIR_RE = re.compile(rf"([^\s:]+):({NUMBER})")
# 返回 {step_num: {"critic/rewards/mean": 0.23, "perf/throughput": 12.3, ...}}
```

**`parse_phase_memory_from_log(log_path)`** — 从 `GPUMemoryLogger` 提取分阶段显存:

```python
MEMORY_RE = re.compile(
    r"(?P<when>Before|After) (?P<name>generate_sequences|...),.*?"
    r"device memory used/total \(GB\): (?P<used>{NUMBER})/(?P<total>{NUMBER})"
)
```

**`parse_gpu_samples(csv_path)`** — 从 GPU 采样 CSV 读取数据:
```python
# 按 phase 分组返回显存百分比列表和利用率列表
# memory["rollout"] = [78.5, 79.1, ...]  # 该阶段所有采样点
```

**`detect_error(log_path)`** — 致命错误检测:

```python
FATAL_PATTERNS = {
    "OOM": r"CUDA out of memory|torch\.OutOfMemoryError|...",
    "NCCL_OR_DISTRIBUTED_FAILURE": r"ChildFailedError|DistBackendError|...",
    "NAN_OR_INF": r"\b(?:nan|inf)\b.*(?:loss|gradient|reward)|...",
}
```

**`compute_threshold_stats(records, thresholds, window)`** — Reward 阈值统计:

```python
# 对每个 step 用滑动窗口平均平滑 reward
# 记录首次达到各阈值 (0.0, 0.1, 0.2, 0.3) 时的累计时间和 token
# 这是最终验收的核心指标——"训练多久能达到目标 reward"
```

#### 6.2.2 `analyze_trial()` — 综合报告

```python
def analyze_trial(log_path, gpu_samples_path, warmup_updates=5,
                  reward_window=5, reward_thresholds=(0.0, 0.1, 0.2, 0.3)):
    # warmup_updates: 前 5 步丢弃（不稳定）
    stable_rows = [row for step, row in records.items() if step > warmup_updates]

    return {
        "updates_completed": max(records, default=0),
        "result": "fail" if error else ("success" if records else "fail"),
        "error": {"type": error_type, "evidence": error_evidence},

        # 四阶段显存 (mean/p95/max)
        "memory_by_phase_pct": {
            "rollout": {"mean": ..., "p95": ..., "max": ...},
            "actor_log_prob": {...}, "ref_log_prob": {...}, "training": {...}
        },

        # 性能指标
        "performance": {
            "throughput": {...},           # perf/throughput
            "time_per_step_s": {...},      # perf/time_per_step
            "phase_duration_s": {          # 各阶段耗时
                "rollout": ..., "actor_log_prob": ...,
                "ref_log_prob": ..., "training": ...
            },
            "time_bottleneck": "rollout",  # 最慢阶段
            "generation_tgs": {...},       # 生成速度
            "actor_mfu": {...},            # 模型算力利用率
        },

        # 资源瓶颈
        "resource": {
            "memory_bottleneck": "training",    # 显存峰值阶段
            "max_observed_memory_pct": 78.5,
        },

        # 稳定性时序：每个 window 对应连续的 5 个 update；
        # metrics 的数组下标与 windows 一一对应。
        "stability": {
            "warmup_updates": 5,
            "window_size": 5,
            "windows": [{"start_step": 6, "end_step": 10, "sample_count": 5}],
            "metrics": {
                "critic/rewards/mean": [...],
                "actor/ppo_kl": [...],
                "actor/pg_clipfrac": [...],
                "actor/entropy": [...],
                "actor/lr": [...],
                "response_length/clip_ratio": [...],
            },
        },

        # 端到端 reward 阈值
        "end_to_end_reward": {
            "thresholds": {
                "0.0": {"step": 10, "cumulative_time_s": 123.4, ...},
                "0.1": {"step": 45, ...}, ...
            },
            "peak_reward": 0.35,
        }
    }
```

---

## 7. 验证与约束层

### 7.1 `validator.py` — 确定性参数校验

**文件**: [validator.py](../validator.py) | 约 180 行

在 LLM 提案进入 Feasibility 审查之前，先过确定性规则。这是**防御性编程**——不该交给不可靠的 LLM 的事情，就用程序检查。

#### 7.1.1 参数白名单

```python
HARDWARE_PARAMETERS = {  # ~30 个硬件参数，包含:
    "data.train_batch_size",          # 训练批大小
    "data.max_prompt_length",         # 最大提示长度
    "actor_rollout_ref.actor.*",      # Actor Megatron 配置
    "actor_rollout_ref.rollout.*",    # Rollout vLLM 配置
    "actor_rollout_ref.ref.*",        # 参考模型配置
}

STABILITY_PARAMETERS = {  # 7 个稳定性参数:
    "actor_rollout_ref.actor.optim.lr",           # 学习率
    "actor_rollout_ref.actor.optim.lr_warmup_steps", # warmup 步数
    "actor_rollout_ref.actor.use_kl_loss",        # 是否使用 KL 损失
    "actor_rollout_ref.actor.kl_loss_coef",       # KL 系数
    "actor_rollout_ref.actor.kl_loss_type",       # KL 类型
    "actor_rollout_ref.actor.entropy_coeff",      # 熵系数
    "actor_rollout_ref.rollout.n",                # 每 prompt 采样数
}
```

#### 7.1.2 `validate_candidate()` 检查项（按执行顺序）

```python
def validate_candidate(parameters, changes, stage, agent_config, base_parameters, history):
    violations = []

    # 1. 修改数量: 不超过 max_parameter_changes (默认 3)
    if len(changes) > max_changes: ...

    # 2. 阶段白名单: 每个修改参数必须在 editable_parameters(stage)
    for key in changes:
        if key not in editable: ...

    # 3. 类型检查: 比对 base_parameters 中的原始类型
    for key in changes:
        if key in base_parameters:
            if isinstance(original, bool) and not isinstance(value, bool): ...
            elif isinstance(original, int) and value is not int: ...
            elif isinstance(original, float) and not isinstance(value, (int, float)): ...

    # 4. 范围检查: RANGES 字典定义每个参数的合法范围
    for key, (lo, hi) in RANGES.items():
        if parameters[key] not in [lo, hi]: ...

    # 5. 必填参数: 6 个核心参数不能缺失
    for key in ["data.train_batch_size", "actor_rollout_ref.rollout.n",
                 "actor_rollout_ref.actor.ppo_mini_batch_size",
                 "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
                 "trainer.n_gpus_per_node", "trainer.nnodes"]:
        if key not in parameters: ...

    # 6. 整除约束:
    #    - train_batch_size * n % ppo_mini_batch_size == 0
    #    - ppo_mini_batch_size % ppo_micro_batch_size_per_gpu == 0
    #    - num_gpus % (actor_TP * actor_PP) == 0
    #    - num_gpus % rollout_TP == 0
    #    - num_gpus % (ref_TP * ref_PP) == 0

    # 7. Token Budget (硬件阶段):
    #    候选的 train_batch_size * n * (prompt + response) 不能偏离基线
    baseline = hardware_token_budget(base_parameters)
    candidate = hardware_token_budget(parameters)
    if abs(candidate - baseline) / baseline > tolerance (0.0): ...

    # 8. 去重: JSON 序列化比较，不允许多次运行同一配置
    for trial in history:
        if json.dumps(parameters, sort_keys=True) == json.dumps(previous, sort_keys=True): ...

    return ValidationResult(not violations, violations)
```

**Token Budget** 是一个关键设计:

```python
def hardware_token_budget(parameters):
    return (
        train_batch_size * rollout_n * (max_prompt_length + max_response_length)
    )
```

在硬件调优阶段，改变 batch_size、序列长度或 n 会改变工作量。保持 token budget 不变确保了吞吐比较是公平的——不是在"做更少工作"的前提下获得虚假提升。

---

## 8. 配置与基础工具层

### 8.1 `config_utils.py` — 通用 IO 与 Hydra 转换

**文件**: [config_utils.py](../config_utils.py) | 约 62 行

纯工具函数，无状态:

| 函数 | 说明 |
|------|------|
| `load_json(path)` | 读取 JSON 文件，文件不存在抛异常 |
| `write_json(path, value)` | 写入 JSON（自动创建父目录，末尾换行） |
| `append_jsonl(path, value)` | 追加一行 JSONL（`default=str` 处理非标准类型） |
| `read_jsonl(path)` | 读取 JSONL，文件不存在返回 `[]` |
| `hydra_value(value)` | Python 值 → Hydra 命令行字符串 |
| `hydra_overrides(parameters)` | `{key: value}` → `["key=value", ...]` |
| `apply_changes(current, changes)` | 合并参数修改（浅层 dict.update） |
| `changed_parameters(before, after)` | 提取被修改的键值 |

Hydra 值转换逻辑:
```python
def hydra_value(value):
    if isinstance(value, bool):   return "True" if value else "False"
    if value is None:             return "null"
    if isinstance(value, list):   return "[" + ",".join(hydra_value(v) for v in value) + "]"
    return str(value)
```

### 8.2 `config/base_parameters.json` — verl 基线参数

**文件**: [base_parameters.json](../config/base_parameters.json)

从 `qwen3_8B_baseline.sh` 转换出的初始 verl Hydra 参数，约 55 项。包含:
- 数据: `train_batch_size=256`, `max_prompt_length=1280`, `max_response_length=5120`
- 模型: `Qwen3-8B`, TP=4, PP=1
- 优化器: `lr=3e-6`, 无 warmup
- Megatron 配置: sequence_parallel, distributed_optimizer, remove_padding, optimizer/param offload 全部启用
- vLLM: `gpu_memory_utilization=0.6`, `max_num_batched_tokens=65536`
- GRPO: `n=4`, `kl_loss_coef=0.003`
- 基础设施: 8 GPUs, 1 node

### 8.3 `config/agent_config.json` — Agent 行为配置

**文件**: [agent_config.json](../config/agent_config.json)

约 30 个配置项，分类如下:

| 类别 | 关键参数 | 默认值 | 说明 |
|------|----------|--------|------|
| 平台 | `platform` | `"V5000"` | GPU 平台 |
| | `verl_root` | 本地路径 | verl 仓库路径 |
| Trial 预算 | `hardware_trial_updates` | 20 | 硬件 trial 步数 |
| | `stability_trial_updates` | 80 | 稳定性 trial 步数 |
| | `confirm_trial_updates` | 300 | 确认 trial 步数 |
| 显存 | `resource_memory_limit_pct` | 95.0 | Resource Gate 硬上限 |
| | `throughput_memory_limit_pct` | 92.0 | 吞吐模式更严格限制 |
| 阶段控制 | `min_hardware_trials` | 2 | 最少硬件 trial 数 |
| | `max_hardware_trials` | 6 | 最多硬件 trial 数 |
| | `plateau_rounds` | 2 | 平台期判断窗口 |
| | `min_throughput_improvement` | 0.02 | 最小改善阈值 (2%) |
| Agent | `max_validation_rounds` | 3 | Proposal 最多重试 |
| | `max_tool_rounds` | 6 | 工具调用最多轮次 |
| | `max_parameter_changes` | 3 | 单次最多修改参数 |
| LLM | `llm_timeout_seconds` | 120 | API 超时 |
| | `llm_max_output_tokens` | 4096 | 最大输出 token |
| 稳定性 | `kl_warning` | 0.1 | KL 预警阈值 |
| | `reward_collapse_slope` | -0.01 | Reward 衰退斜率 |

---

## 9. CLI 独立脚本

### 9.1 `role_cli.py` — 单角色执行

**文件**: [role_cli.py](../role_cli.py) | 约 59 行

允许单独调用一个 Agent 角色（proposal/feasibility/diagnosis），不启动训练:

```bash
# 独立运行 Proposal Agent
python3 role_cli.py --role proposal --context context.json \
  --output suggestion.json --trace-output trace.json

# Rules-only 模式（不调用 LLM）
python3 role_cli.py --role feasibility --context context.json --rules-only
```

用于:
- 调试 Agent 提示词
- 测试工具调用链
- 保存完整的对话轨迹供分析

### 9.2 `trial_cli.py` — 单 Trial 执行

**文件**: [trial_cli.py](../trial_cli.py) | 约 40 行

独立运行一个受监控的 trial:

```bash
python3 trial_cli.py --parameters params.json --agent-config config.json \
  --stage hardware_tuning --trial-id 1 --updates 20 --dry-run
```

这绕过了 Orchestrator 状态机，直接调用 `runner.run_trial()`。适合手动测试参数组合。

### 9.3 `monitor_cli.py` — 离线日志分析

**文件**: [monitor_cli.py](../monitor_cli.py) | 约 41 行

对已有训练日志进行离线分析:

```bash
python3 monitor_cli.py --log train.log --gpu-samples gpu_samples.csv \
  --output report.json --warmup-updates 5 --reward-window 5
```

这绕过了训练执行，直接调用 `metrics.analyze_trial()`。

### 9.4 `tools/compare_end_to_end_reward.py` — 最终验收对比

**文件**: [compare_end_to_end_reward.py](../tools/compare_end_to_end_reward.py) | 约 82 行

独立验收入口，比较多个配置的端到端 reward 收敛性能:

```bash
PYTHONPATH=. python3 tools/compare_end_to_end_reward.py \
  --log base=path/to/base.log \
  --log candidate=path/to/candidate.log \
  --output reward_comparison.json \
  --thresholds 0.0 0.1 0.2 0.3 \
  --window 5
```

输出每组的:
- 达到各 reward 阈值的 step、时间、token
- Peak reward 及达到时的累计时间和 token
- 总时间和总 token

使用滑动窗口平均避免单步噪声干扰。

---

## 10. 测试体系

### 10.1 `test_orchestrator.py`

**文件**: [test_orchestrator.py](../tests/test_orchestrator.py)

测试阶段状态机和拒绝-重试闭环:

```python
# 测试数据工厂
def hardware_trial(trial_id, throughput, result="success"):
    return {"trial_id": trial_id, "stage": "hardware_tuning",
            "result": result, "performance": {"throughput": {"mean": throughput}}}

class OrchestratorStageTest:
    def test_initial_and_failed_hardware_stage(self):
        self.assertEqual(determine_stage([], config), "hardware_tuning")
        self.assertEqual(determine_stage([failed], config), "hardware_repair")

    def test_plateau_moves_to_stability(self):
        # 4 个成功硬件 trial，最后 2 个改善 < 2% → stability

    def test_two_healthy_stability_trials_move_to_confirm(self):
        # 2 个健康稳定性 trial → confirm

class RejectionConversationTest:
    def test_validator_rejection_is_added_to_same_proposal_conversation(self):
        # 创建 FakeAgents: 第一次返回 ppo_micro_batch=3（被 Validator 拒绝）
        #                   第二次返回 ppo_micro_batch=2（通过）
        # 验证: agent 在同一对话中看到了第 1 次的拒绝消息
```

### 10.2 `test_agents.py`

**文件**: [test_agents.py](../tests/test_agents.py)

使用 **FakeClient 依赖注入**模拟 LLM API:

```python
class FakeResponses:
    def __init__(self, responses): self.responses = list(responses)
    def create(self, **kwargs): return self.responses.pop(0)

class FakeClient:
    def __init__(self, responses): self.responses = FakeResponses(responses)

# 注入到 LLMRoleAgent
agent = LLMRoleAgent("proposal", prompt_path, registry, config,
                      client_factory=lambda: FakeClient([tool_response, json_response]))
```

测试:
- 工具结果正确注入对话
- 拒绝消息在同一对话中传递
- 多轮工具调用循环

关键设计: 测试不发送任何 API 请求，完全验证 Agent 的推理逻辑。

### 10.3 `test_agent_tools.py`

**文件**: [test_agent_tools.py](../tests/test_agent_tools.py)

测试每个工具的权限、输入验证和输出:

```python
class AgentToolsTest:
    def test_parameter_understanding_is_allowlisted(self):
        # 验证: 未知参数 → unknown_parameters
        # 验证: read_trial_log_excerpt 对 proposal 角色被拒绝

    def test_memory_estimator_uses_phase_observation_anchor(self):
        # 注入带 phase memory 的 trial，验证投影计算

    def test_search_verl_docs_is_bounded_to_configured_root(self):
        # 创建临时 verl 目录，验证搜索结果

    def test_trial_history_query_can_return_successful_parameters(self):
        # 写入测试 JSONL，验证筛选和排序
```

### 10.4 `test_validator.py`

**文件**: [test_validator.py](../tests/test_validator.py)

```python
class ValidatorTest:
    def test_valid_micro_batch_change(self):
        # ppo_micro_batch_size_per_gpu=2 → valid

    def test_rejects_stability_hardware_change(self):
        # stability 阶段改 max_num_seqs → invalid

    def test_rejects_changed_hardware_token_budget(self):
        # 改 rollout.n → token budget 变化 → invalid
```

---

## 11. 核心设计模式与技巧

### 11.1 三阶段状态机

```
hardware_tuning ──→ stability_tuning ──→ confirm ──→ done
       ↑                    ↑
  hardware_repair      stopped_unstable
```

每个阶段明确定义:
- **目标**: 吞吐最大化 → 稳定性保证 → 最终确认
- **可编辑参数集**: 硬件 30 个 → 稳定性 7 个 → 无
- **Trial 预算**: 20 → 80 → 300 updates
- **转换条件**: 数量/plateau/健康检查

### 11.2 双层拒绝-重试循环

- **内层（Agent 推理层）**: LLM 调用工具 → 分析结果 → 再调用工具 → 输出决策
- **外层（协作对抗层）**: Proposal → Validator → Feasibility，被拒则反馈注入对话重试

### 11.3 对话连续性

被拒绝后不重建对话，而是将拒绝原因注入当前对话的 `messages` 列表。Agent 在下一轮推理时能看到自己的失败过程。这是实现"从错误中学习"的关键机制。

### 11.4 LLM 与确定性程序的职责分离

| 职责 | 谁负责 | 原因 |
|------|--------|------|
| 生成候选参数 | LLM (Proposal) | 需要推理和知识检索 |
| 类型/范围/整除检查 | 程序 (Validator) | 确定可证明，不应委托 LLM |
| 语义/跨阶段 trade-off | LLM (Feasibility) | 需要综合判断上下文 |
| 进程管理和监控 | 程序 (Runner) | 安全关键，不能给 LLM |
| 指标计算 | 程序 (Metrics) | 数值计算，需要确定性 |
| 阶段决策 | 程序 (Orchestrator) | 基于数值阈值的规则 |

### 11.5 工具安全设计

不给 LLM 通用 shell 工具。每个工具:
- 固定参数格式（JSON Schema）
- 固定输出结构
- 白名单权限控制（按角色）
- 输入验证和边界检查
- 超时和资源限制

### 11.6 经验锚点估算

显存估算器不是精确模拟器，而是基于**已有观测数据的相对投影**。明确标注方法、参考 trial、置信度和局限性。没有观测时明确返回 `confidence: "low"` 和 `projected_pct: null`。

### 11.7 配置分层

```
配置文件 (JSON) → 命令行参数 → 环境变量
     ↑                              ↑
   默认值                         覆盖值
```

环境变量优先级最高，方便 CI/CD 和不同平台灵活切换。

### 11.8 依赖注入与可测试性

- `LLMRoleAgent` 接受 `client_factory` 注入 → 测试用 FakeClient
- `ToolRegistry` 接受 history_path 参数 → 测试用临时目录
- `TuningOrchestrator` 的 `agents` 属性可替换 → 测试用 FakeAgents

---

## 12. 源码阅读路线建议

### 快速入门（30 分钟）

1. [README.md](../README.md) — 项目概览、运行方式
2. [run_agent.py](../run_agent.py) — 入口→配置→Orchestrator 连线
3. [orchestrator.py](../orchestrator.py) `determine_stage()` — 阶段判定（lines 87-105）
4. [orchestrator.py](../orchestrator.py) `run()` — 主循环（lines 305-378）
5. [config/agent_config.json](../config/agent_config.json) — 控制参数一览

### 深入理解（2 小时）

6. [agents.py](../agents.py) `LLMRoleAgent.run()` — 工具调用循环（lines 161-238）
7. [orchestrator.py](../orchestrator.py) `_propose_candidate()` — 协作闭环（lines 191-303）
8. [runner.py](../runner.py) `run_trial()` — 训练执行 + 监控（lines 174-289）
9. [validator.py](../validator.py) `validate_candidate()` — 确定性校验（lines 94-179）
10. [agent_tools/registry.py](../agent_tools/registry.py) — 工具注册与执行（全部）
11. [agent_tools/memory_estimator.py](../agent_tools/memory_estimator.py) — 显存估算算法（全部）
12. [prompts/proposal.md](../prompts/proposal.md) — Proposal 提示词

### 全面掌握（4 小时+）

13. [metrics.py](../metrics.py) — 指标分析全流程
14. [prompting.py](../prompting.py) — 提示词渲染与拒绝反馈
15. [agent_tools/parameter_docs.json](../agent_tools/parameter_docs.json) — 参数知识库结构
16. [agent_tools/tuning_strategies.json](../agent_tools/tuning_strategies.json) — 策略库结构
17. [tests/test_agents.py](../tests/test_agents.py) — FakeClient 测试模式
18. [tests/test_orchestrator.py](../tests/test_orchestrator.py) — 状态机 + 拒绝循环测试
19. [tools/compare_end_to_end_reward.py](../tools/compare_end_to_end_reward.py) — 最终验收逻辑

### 每步自检问题

| 读完 | 应能回答 |
|------|----------|
| 状态机 | 为什么现在处于 hardware 而不是 stability？ |
| Validator | 候选是否在数学和阶段规则上合法？ |
| Agent 对话 | LLM 是主动查到证据，还是直接猜的？ |
| Feasibility | 为什么合法候选仍可能不值得运行？ |
| Runner | 训练何时提前停止，phase memory 从哪里来？ |
| Metrics | 什么指标决定成功、最优和阶段切换？ |
| 显存估算器 | 为什么不能直接相信估算结果作为最终内存使用？ |
