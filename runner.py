from __future__ import annotations

import csv
import os
import queue
import re
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from config_utils import append_jsonl, hydra_overrides, write_json, write_json_atomic
from health_monitor import OnlineHealthMonitor, parse_online_step
from metrics import (
    build_running_metrics,
    legacy_metrics_from_structured,
    parse_step_line,
)
from tools.extract_trial_metrics import extract_trial_metrics
from validator import parameter_groups
from vllm_metrics import (
    DISABLE_LOG_STATS_PARAMETER,
    ROLLOUT_ENGINE_PARAMETER,
    VLLMMetricsSampler,
    summarize_vllm_metrics,
    vllm_metrics_enabled,
)


FATAL_RE = re.compile(
    r"CUDA out of memory|OutOfMemoryError|ChildFailedError|DistBackendError|"
    r"NCCL[^\n]{0,80}(?:WARN|ERROR|unhandled|failed|failure)",
    re.I,
)
PHASE_START = {
    "Before generate_sequences": "rollout",
    "Before compute_log_prob": "actor_log_prob",
    "Before compute_ref_log_prob": "ref_log_prob",
    "Before update_actor": "training",
}
PHASE_END = tuple(value.replace("Before", "After", 1) for value in PHASE_START)
# MetaX C550's verl build does not emit ``Before generate_sequences``.  Instead,
# it reports that rollout has been initialized once, then emits
# ``After update_actor`` immediately before each following rollout.  Keep the
# sampler in rollout until the next explicit compute phase begins.
C550_ROLLOUT_START = ("After rollout init", "After update_actor")


class PhaseTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._phase = "startup"

    def update_from_log(self, line: str) -> None:
        with self._lock:
            for marker, phase in PHASE_START.items():
                if marker in line:
                    self._phase = phase
                    return
            if any(marker in line for marker in C550_ROLLOUT_START):
                self._phase = "rollout"
                return
            if any(marker in line for marker in PHASE_END):
                self._phase = "between_phases"

    def get(self) -> str:
        with self._lock:
            return self._phase


class GPUSampler(threading.Thread):
    def __init__(self, output_path: Path, tracker: PhaseTracker, interval: float, platform: str) -> None:
        super().__init__(daemon=True)
        self.output_path = output_path
        self.tracker = tracker
        self.interval = interval
        self.stop_event = threading.Event()
        self.max_memory_pct = 0.0
        self.max_memory_used_mib = 0.0
        self._state_lock = threading.Lock()
        self.latest_rows: list[dict[str, float | str]] = []
        self.samples_written = 0
        self.sample_errors = 0
        self.last_sample_timestamp: float | None = None
        self.platform = os.getenv("PLATFORM", platform).upper()
        configured_smi = os.getenv("GPU_SMI")
        default_smi = "xpu-smi" if self.platform == "V5000" else "mx-smi" if self.platform in {"C550", "METAX"} else "nvidia-smi"
        self.executable = configured_smi or shutil.which(default_smi)

    @staticmethod
    def _number(value: str) -> float:
        match = re.search(r"[-+]?\d*\.?\d+", value.replace(",", ""))
        if not match:
            raise ValueError(f"cannot parse numeric GPU field: {value}")
        return float(match.group(0))

    def _query_rows(self) -> list[list[str]]:
        proc = subprocess.run(
            [
                self.executable,
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=max(1.0, self.interval),
            check=False,
        )
        rows: list[list[str]] = []
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                fields = [part.strip() for part in line.split(",")]
                if len(fields) == 4:
                    rows.append(fields)
        if rows:
            return rows

        # Fall back to human-readable table parsing.
        proc = subprocess.run(
            [self.executable], capture_output=True, text=True, timeout=max(1.0, self.interval), check=False
        )
        output = proc.stdout

        # mx-smi (C550 / METAX): each GPU spans two table lines.
        if self.platform in {"C550", "METAX"}:
            table_lines = [line for line in output.splitlines() if line.strip().startswith("|")]
            idx = 0
            while idx + 1 < len(table_lines):
                gpu_line = table_lines[idx]
                mem_line = table_lines[idx + 1]
                gpu_match = re.match(r"\|\s*(\d+)\s+", gpu_line)
                if not gpu_match:
                    idx += 1
                    continue
                util_match = re.search(r"(\d+)%\s+(?:Disabled|Enabled)", gpu_line)
                mem_match = re.search(r"(\d+)/(\d+)\s*MiB", mem_line)
                if not mem_match:
                    idx += 2
                    continue
                rows.append([
                    gpu_match.group(1),
                    mem_match.group(1),
                    mem_match.group(2),
                    util_match.group(1) if util_match else "0",
                ])
                idx += 2
            return rows

        # Older xpu-smi versions only expose the human-readable table.
        for index, line in enumerate(part for part in output.splitlines() if "Default" in part):
            fields = line.split()
            if len(fields) >= 13:
                rows.append([str(index), fields[8], fields[10], fields[12]])
        return rows

    def run(self) -> None:
        if not self.executable:
            return
        with self.output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["timestamp", "phase", "gpu_index", "memory_used_mb", "memory_total_mb", "utilization_pct"]
            )
            while not self.stop_event.is_set():
                try:
                    now = time.time()
                    for fields in self._query_rows():
                        used, total = self._number(fields[1]), self._number(fields[2])
                        utilization = self._number(fields[3])
                        if total > 0:
                            memory_pct = 100.0 * used / total
                            with self._state_lock:
                                self.max_memory_pct = max(self.max_memory_pct, memory_pct)
                                self.max_memory_used_mib = max(self.max_memory_used_mib, used)
                                self.latest_rows.append(
                                    {
                                        "gpu_index": str(fields[0]),
                                        "memory_used_mb": used,
                                        "memory_total_mb": total,
                                        "memory_pct": memory_pct,
                                        "utilization_pct": utilization,
                                    }
                                )
                            writer.writerow([now, self.tracker.get(), fields[0], used, total, utilization])
                            with self._state_lock:
                                self.samples_written += 1
                                self.last_sample_timestamp = now
                    with self._state_lock:
                        if self.latest_rows:
                            latest_by_gpu = {}
                            for row in self.latest_rows:
                                latest_by_gpu[str(row["gpu_index"])] = row
                            self.latest_rows = list(latest_by_gpu.values())
                    handle.flush()
                except (OSError, ValueError, subprocess.SubprocessError):
                    with self._state_lock:
                        self.sample_errors += 1
                self.stop_event.wait(self.interval)

    def stop(self) -> None:
        self.stop_event.set()

    def snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "available": bool(self.executable),
                "executable": self.executable,
                "platform": self.platform,
                "samples_written": self.samples_written,
                "sample_errors": self.sample_errors,
                "last_sample_timestamp": self.last_sample_timestamp,
                "max_memory_pct": self.max_memory_pct,
                "max_memory_used_mib": self.max_memory_used_mib,
                "gpus": [dict(row) for row in self.latest_rows],
            }

    def reserve_exceeded(self, reserve_mib: float) -> bool:
        with self._state_lock:
            return any(
                float(row["memory_total_mb"]) - float(row["memory_used_mb"])
                < reserve_mib
                for row in self.latest_rows
            )


HealthDecider = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class HealthAgentWorker:
    """Run at most one health review at a time without blocking stdout reads."""

    def __init__(self, decider: HealthDecider) -> None:
        self.decider = decider
        self._results: queue.Queue[dict[str, Any]] = queue.Queue()
        self._lock = threading.Lock()
        self._pending = False

    def submit(self, event_id: str, context: Mapping[str, Any]) -> bool:
        with self._lock:
            if self._pending:
                return False
            self._pending = True

        def invoke() -> None:
            try:
                payload = dict(self.decider(context))
                result = {"event_id": event_id, "ok": True, "payload": payload}
            except Exception as exc:  # Agent failures must never stop training.
                result = {
                    "event_id": event_id,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            self._results.put(result)

        threading.Thread(target=invoke, name=f"health-agent-{event_id}", daemon=True).start()
        return True

    def poll(self) -> dict[str, Any] | None:
        try:
            result = self._results.get_nowait()
        except queue.Empty:
            return None
        with self._lock:
            self._pending = False
        return result

    def pending(self) -> bool:
        with self._lock:
            return self._pending


@dataclass
class PendingHealthObservation:
    origin_event_id: str
    origin_step: int
    due_step: int
    original_decision: dict[str, Any]
    review_submitted: bool = False


class HealthReviewSchedule:
    """Track an Agent-requested observation independently of rule cooldowns."""

    def __init__(self) -> None:
        self.pending: PendingHealthObservation | None = None

    def observe(
        self,
        event_id: str,
        origin_step: int,
        observe_for_updates: int,
        decision: Mapping[str, Any],
    ) -> None:
        if observe_for_updates < 1:
            raise ValueError("observe_for_updates must be positive")
        self.pending = PendingHealthObservation(
            origin_event_id=event_id,
            origin_step=int(origin_step),
            due_step=int(origin_step) + int(observe_for_updates),
            original_decision=dict(decision),
        )

    def due(self, step: int) -> bool:
        return bool(
            self.pending is not None
            and not self.pending.review_submitted
            and int(step) >= self.pending.due_step
        )

    def active(self) -> bool:
        return self.pending is not None

    def mark_submitted(self) -> None:
        if self.pending is not None:
            self.pending.review_submitted = True

    def clear(self) -> None:
        self.pending = None

    def snapshot(self) -> dict[str, Any] | None:
        return asdict(self.pending) if self.pending is not None else None


def build_command(
    parameters: Mapping[str, Any],
    agent_config: Mapping[str, Any],
    trial_id: int,
    updates: int,
    stage: str | None = None,
) -> tuple[list[str], Path]:
    verl_root = Path(os.getenv("VERL_ROOT", str(agent_config["verl_root"]))).expanduser().resolve()
    run_parameters = dict(parameters)
    run_parameters.update(
        {
            "trainer.total_training_steps": updates,
            "trainer.total_epochs": max(1, int(parameters.get("trainer.total_epochs", 1))),
            "trainer.logger": parameters.get("trainer.logger", ["console"]),
        }
    )
    command = [
        "python3",
        "-m",
        "verl.trainer.main_ppo",
        f"--config-path={agent_config.get('config_path', 'config')}",
        f"--config-name={agent_config.get('config_name', 'ppo_megatron_trainer.yaml')}",
        *hydra_overrides(run_parameters),
    ]
    environment_script = os.getenv("VERL_ENV_SCRIPT") or agent_config.get("environment_script")
    if environment_script:
        script = str(Path(str(environment_script)).expanduser().resolve())
        command = ["bash", "-lc", 'source "$1"; shift; exec "$@"', "verl-agent", script, *command]
    return command, verl_root


def _terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _resource_gate_enabled(stage: str) -> bool:
    """Every training stage must preserve the configured absolute reserve."""
    return stage in {"hardware_tuning", "hardware_repair", "stability_tuning", "confirm"}


def run_trial(
    parameters: Mapping[str, Any],
    agent_config: Mapping[str, Any],
    trial_id: int,
    stage: str,
    updates: int,
    dry_run: bool = False,
    health_decider: HealthDecider | None = None,
) -> dict[str, Any]:
    output_root = Path(os.getenv("OUTPUT_PATH", str(agent_config["output_dir"]))).expanduser().resolve()
    trial_dir = output_root / "trials" / f"{trial_id:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    log_path = trial_dir / "train.log"
    samples_path = trial_dir / "gpu_samples.csv"
    vllm_metrics_path = trial_dir / "vllm_metrics.csv"
    health_events_path = trial_dir / "health_events.jsonl"
    health_traces_path = trial_dir / "health_agent_traces.jsonl"
    metrics_path = trial_dir / "metrics.json"
    platform = os.getenv("PLATFORM", str(agent_config.get("platform", "V5000")))
    write_json(trial_dir / "parameters.json", dict(parameters))
    write_json(trial_dir / "parameter_groups.json", parameter_groups(parameters, stage))

    command, cwd = build_command(parameters, agent_config, trial_id, updates, stage)
    write_json(trial_dir / "command.json", {"cwd": str(cwd), "argv": command})
    if dry_run:
        return {
            "trial_id": trial_id,
            "stage": stage,
            "platform": platform,
            "dry_run": True,
            "updates_target": updates,
            "parameters": dict(parameters),
            "command": command,
            "cwd": str(cwd),
        }
    if not cwd.exists():
        raise FileNotFoundError(f"verl_root does not exist: {cwd}")

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("VERL_LOGGING_LEVEL", "DEBUG")
    tracker = PhaseTracker()
    sampler = GPUSampler(
        samples_path,
        tracker,
        float(agent_config.get("monitor_interval_seconds", 1.0)),
        platform,
    )
    collect_vllm_metrics = vllm_metrics_enabled(parameters)
    vllm_sampler = (
        VLLMMetricsSampler(
            vllm_metrics_path,
            float(agent_config.get("vllm_metrics_interval_seconds", 5.0)),
        )
        if collect_vllm_metrics
        else None
    )
    health_monitor = None
    health_worker = None
    if stage == "stability_tuning" and bool(agent_config.get("health_monitor_enabled", True)):
        health_monitor = OnlineHealthMonitor(
            kl_metric=str(agent_config.get("health_kl_metric", "actor/kl_loss")),
            reward_metric=str(agent_config.get("health_reward_metric", "critic/rewards/mean")),
            reward_trend_steps=int(agent_config.get("health_reward_trend_steps", 5)),
            warmup_updates=int(agent_config.get("health_warmup_updates", 0)),
            cooldown_updates=int(agent_config.get("health_agent_cooldown_updates", 5)),
            reward_window_size=int(agent_config.get("health_reward_window_size", 3)),
            reward_trend_min_drawdown=float(
                agent_config.get("health_reward_trend_min_drawdown", 0.15)
            ),
            reward_trend_tolerance=float(
                agent_config.get("health_reward_trend_tolerance", 0.01)
            ),
            kl_change_ratio_threshold=float(
                agent_config.get("health_kl_change_ratio_threshold", 0.50)
            ),
            kl_change_absolute_threshold=float(
                agent_config.get("health_kl_change_absolute_threshold", 0.02)
            ),
            entropy_drop_ratio_threshold=float(
                agent_config.get("health_entropy_drop_ratio_threshold", 0.30)
            ),
        )
        if health_decider is not None:
            health_worker = HealthAgentWorker(health_decider)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    sampler.start()
    if vllm_sampler is not None:
        vllm_sampler.start()
    stop_reason = None
    health_events: list[dict[str, Any]] = []
    health_decisions: list[dict[str, Any]] = []
    health_agent_calls = 0
    pending_health_stop: dict[str, Any] | None = None
    health_review_schedule = HealthReviewSchedule()
    submitted_health_contexts: dict[str, dict[str, Any]] = {}
    # Zero means unlimited; cooldown and the single-flight worker still prevent storms.
    max_health_calls = max(0, int(agent_config.get("health_agent_max_calls_per_trial", 0)))
    stop_confidence = float(agent_config.get("health_agent_stop_confidence", 0.8))
    shadow_mode = bool(agent_config.get("health_agent_shadow_mode", False))
    gate_updates = int(agent_config.get("resource_gate_updates", 5))
    resource_reserve_mib = float(agent_config.get("resource_memory_reserve_mib", 3277))
    throughput_reserve_mib = float(agent_config.get("throughput_memory_reserve_mib", 6554))
    if resource_reserve_mib < 0 or throughput_reserve_mib < resource_reserve_mib:
        raise ValueError(
            "throughput_memory_reserve_mib must be >= resource_memory_reserve_mib >= 0"
        )
    online_records: dict[int, dict[str, float]] = {}

    def submit_health_review(event: Mapping[str, Any], step: int) -> tuple[bool, str | None]:
        nonlocal health_agent_calls
        if health_worker is None:
            return False, "agent_unavailable"
        if max_health_calls > 0 and health_agent_calls >= max_health_calls:
            return False, "max_calls_reached"
        event_id = str(event["event_id"])
        context = {
            "current_stage": stage,
            "mode": str(event.get("review_type", "online_health_review")),
            "health_event": dict(event),
            "active_trial": {
                "trial_id": trial_id,
                "stage": stage,
                "log_path": str(log_path),
                "metrics_path": str(metrics_path),
                "snapshot_step": int(step),
            },
            "recent_trials": [],
        }
        if not health_worker.submit(event_id, context):
            return False, "review_pending"
        submitted_health_contexts[event_id] = context
        health_agent_calls += 1
        return True, None

    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            assert process.stdout is not None
            for line in process.stdout:
                log_handle.write(line)
                log_handle.flush()
                tracker.update_from_log(line)
                if vllm_sampler is not None:
                    vllm_sampler.observe_log_line(line)

                agent_result = health_worker.poll() if health_worker is not None else None
                if agent_result is not None:
                    event_id = str(agent_result["event_id"])
                    submitted_context = submitted_health_contexts.pop(event_id, {})
                    active_trial = submitted_context.get("active_trial", {})
                    decision_step = (
                        active_trial.get("snapshot_step")
                        if isinstance(active_trial, Mapping)
                        else None
                    )
                    review_type = submitted_context.get("mode", "online_health_review")
                    if agent_result.get("ok"):
                        payload = agent_result.get("payload", {})
                        decision = payload.get("decision", payload)
                        trace = payload.get("trace")
                        if not isinstance(decision, Mapping):
                            decision = {}
                        compact_decision = {
                            "event_id": event_id,
                            "review_type": review_type,
                            "snapshot_step": decision_step,
                            "verdict": decision.get("verdict"),
                            "action": decision.get("action"),
                            "confidence": decision.get("confidence"),
                            "reason_codes": decision.get("reason_codes", []),
                            "evidence": decision.get("evidence", []),
                            "counterevidence": decision.get("counterevidence", []),
                            "observe_for_updates": decision.get("observe_for_updates", 0),
                            "reason": decision.get("reason"),
                        }
                        health_decisions.append(compact_decision)
                        append_jsonl(
                            health_events_path,
                            {"record_type": "agent_decision", **compact_decision},
                        )
                        if trace is not None:
                            append_jsonl(
                                health_traces_path,
                                {"event_id": event_id, "trace": trace},
                            )
                        observe_for = decision.get("observe_for_updates", 0)
                        if (
                            decision.get("action") == "observe"
                            and isinstance(observe_for, int)
                            and observe_for > 0
                            and isinstance(decision_step, int)
                        ):
                            health_review_schedule.observe(
                                event_id,
                                decision_step,
                                observe_for,
                                compact_decision,
                            )
                        else:
                            health_review_schedule.clear()
                        confidence = decision.get("confidence")
                        should_stop = (
                            decision.get("verdict") == "unhealthy"
                            and decision.get("action") == "stop"
                            and isinstance(confidence, (int, float))
                            and float(confidence) >= stop_confidence
                        )
                        if should_stop:
                            compact_decision["shadow_mode"] = shadow_mode
                            if not shadow_mode:
                                pending_health_stop = compact_decision
                    else:
                        failure = {
                            "event_id": event_id,
                            "review_type": review_type,
                            "snapshot_step": decision_step,
                            "verdict": "insufficient_evidence",
                            "action": "continue",
                            "confidence": 0.0,
                            "reason_codes": ["HEALTH_AGENT_ERROR"],
                            "evidence": [],
                            "counterevidence": [],
                            "observe_for_updates": 0,
                            "reason": agent_result.get("error"),
                        }
                        health_decisions.append(failure)
                        append_jsonl(health_events_path, {"record_type": "agent_error", **failure})
                        health_review_schedule.clear()

                if FATAL_RE.search(line):
                    stop_reason = "fatal_log"
                    _terminate(process)
                    break

                parsed_step = parse_online_step(line)
                current_step = parsed_step[0] if parsed_step is not None else None
                parsed_metrics = parse_step_line(line)
                if parsed_metrics is not None:
                    metric_step, metric_values = parsed_metrics
                    online_records[metric_step] = metric_values
                    write_json_atomic(
                        metrics_path,
                        build_running_metrics(
                            online_records,
                            snapshot_step=metric_step,
                            resource_snapshot=sampler.snapshot(),
                            expected_gpu_count=int(
                                parameters.get("trainer.n_gpus_per_node", 1)
                            ),
                            resource_reserve_mib=resource_reserve_mib,
                            throughput_reserve_mib=throughput_reserve_mib,
                        ),
                    )
                if current_step is not None and pending_health_stop is not None:
                    stop_reason = "health_agent_early_stop"
                    append_jsonl(
                        health_events_path,
                        {
                            "record_type": "stop_applied",
                            "step": current_step,
                            "decision": pending_health_stop,
                        },
                    )
                    _terminate(process)
                    break
                if (
                    _resource_gate_enabled(stage)
                    and current_step is not None
                    and current_step >= gate_updates
                    and sampler.reserve_exceeded(resource_reserve_mib)
                ):
                    stop_reason = "resource_gate_memory_limit"
                    _terminate(process)
                    break
                if parsed_step is not None and health_monitor is not None:
                    step, step_metrics = parsed_step
                    trigger = health_monitor.add_step(step, step_metrics)
                    scheduled_due = health_review_schedule.due(step)
                    if trigger is not None or scheduled_due:
                        event_id = f"trial-{trial_id:04d}-step-{step:06d}-event-{len(health_events) + 1:03d}"
                        review_type = (
                            "scheduled_followup"
                            if scheduled_due
                            else "risk_escalation"
                            if health_review_schedule.active()
                            and trigger is not None
                            and trigger.get("severity") == "high"
                            else "rule_trigger"
                        )
                        event = {
                            "event_id": event_id,
                            "trial_id": trial_id,
                            "stage": stage,
                            "updates_target": updates,
                            "review_type": review_type,
                            "trigger": trigger or health_monitor.review_snapshot(),
                            "observation": health_review_schedule.snapshot(),
                            "resource_snapshot": sampler.snapshot(),
                            "parameters": dict(parameters),
                        }
                        health_events.append(event)
                        append_jsonl(
                            health_events_path,
                            {
                                "record_type": (
                                    "rule_trigger" if trigger is not None else "scheduled_review"
                                ),
                                **event,
                            },
                        )
                        should_submit = bool(
                            scheduled_due
                            or not health_review_schedule.active()
                            or review_type == "risk_escalation"
                        )
                        if should_submit:
                            submitted, reason = submit_health_review(event, step)
                            if submitted:
                                if scheduled_due or review_type == "risk_escalation":
                                    health_review_schedule.mark_submitted()
                            else:
                                append_jsonl(
                                    health_events_path,
                                    {
                                        "record_type": "agent_not_submitted",
                                        "event_id": event_id,
                                        "step": step,
                                        "reason": reason,
                                    },
                                )
                                if scheduled_due and reason in {
                                    "agent_unavailable",
                                    "max_calls_reached",
                                }:
                                    health_review_schedule.mark_submitted()
                        else:
                            append_jsonl(
                                health_events_path,
                                {
                                    "record_type": "agent_not_submitted",
                                    "event_id": event_id,
                                    "step": step,
                                    "reason": "observation_active",
                                },
                            )
        return_code = process.wait()
    finally:
        sampler.stop()
        sampler.join(timeout=5)
        if vllm_sampler is not None:
            vllm_sampler.stop()
            vllm_sampler.join(timeout=5)
        _terminate(process)

    vllm_summary = summarize_vllm_metrics(
        vllm_metrics_path if collect_vllm_metrics else None
    )
    structured_metrics = extract_trial_metrics(
        trial_dir,
        agent_config,
        parameters=parameters,
        expected_gpu_count=int(parameters.get("trainer.n_gpus_per_node", 1)),
        vllm_summary=vllm_summary,
        monitor=sampler.snapshot(),
        write_metrics=False,
    )
    metrics = legacy_metrics_from_structured(structured_metrics)
    metrics.update(
        {
            "trial_id": trial_id,
            "stage": stage,
            "platform": platform,
            "updates_target": updates,
            "parameters": dict(parameters),
            "return_code": return_code,
            "stop_reason": stop_reason,
            "failure_phase": tracker.get() if stop_reason or return_code != 0 else None,
            "log_path": str(log_path),
            "gpu_samples_path": str(samples_path) if samples_path.exists() else None,
            "vllm_metrics_path": (
                str(vllm_metrics_path) if collect_vllm_metrics and vllm_metrics_path.exists() else None
            ),
            "rollout_engine": {
                "name": parameters.get(ROLLOUT_ENGINE_PARAMETER),
                "disable_log_stats": parameters.get(DISABLE_LOG_STATS_PARAMETER),
                "monitor": (
                    vllm_sampler.snapshot()
                    if vllm_sampler is not None
                    else {
                        "enabled": False,
                        "reason": (
                            "rollout engine is not vllm"
                            if str(parameters.get(ROLLOUT_ENGINE_PARAMETER, "")).lower() != "vllm"
                            else f"{DISABLE_LOG_STATS_PARAMETER} must be explicitly false"
                        ),
                    }
                ),
                "metrics": vllm_summary,
            },
            "health_events_path": str(health_events_path) if health_events_path.exists() else None,
            "health_agent_traces_path": str(health_traces_path) if health_traces_path.exists() else None,
            "health_monitor": health_monitor.summary() if health_monitor is not None else {"enabled": False},
            "health_events": health_events,
            "health_decisions": health_decisions,
        }
    )
    health_early_stopped = stop_reason == "health_agent_early_stop"
    if metrics["updates_completed"] < updates and not health_early_stopped and not metrics["error"].get("type"):
        metrics["error"] = {
            "type": "INCOMPLETE_TRAINING",
            "evidence": [f"completed {metrics['updates_completed']} of {updates} updates"],
        }
    resource_summary = structured_metrics.get("resource", {}).get("summary", {})
    resource_monitor = structured_metrics.get("resource", {}).get("monitor", {})
    if resource_summary.get("resource_limit_exceeded") is True:
        if metrics["error"].get("type") in {None, "INCOMPLETE_TRAINING"}:
            metrics["error"] = {
                "type": "MEMORY_HEADROOM_EXCEEDED",
                "evidence": [
                    "observed device memory left less than "
                    f"resource_memory_reserve_mib={resource_reserve_mib:.0f}"
                ],
            }
        metrics["result"] = "fail"
    if resource_monitor.get("coverage_complete") is not True:
        if not metrics["error"].get("type"):
            metrics["error"] = {
                "type": "RESOURCE_MONITOR_INCOMPLETE",
                "evidence": [
                    "resource safety cannot be certified because GPU monitor coverage is incomplete"
                ],
            }
        metrics["result"] = "fail"
    if stage.startswith("hardware") and resource_summary.get("throughput_limit_exceeded") is True:
        metrics["error"] = {
            "type": "MEMORY_HEADROOM_EXCEEDED",
            "evidence": [
                "observed device memory left less than "
                f"throughput_memory_reserve_mib={throughput_reserve_mib:.0f}"
            ],
        }
        metrics["result"] = "fail"
    if health_early_stopped:
        metrics["result"] = "early_stopped"
        metrics["termination"] = {
            "type": "health_policy",
            "step": metrics["updates_completed"],
            "decision": pending_health_stop,
        }
    elif return_code != 0 or stop_reason or metrics["updates_completed"] < updates:
        metrics["result"] = "fail"
    structured_metrics["result"] = metrics["result"]
    structured_metrics["error"] = dict(metrics.get("error", {}))
    structured_metrics["stop_reason"] = stop_reason
    structured_metrics["updates_target"] = updates
    write_json_atomic(metrics_path, structured_metrics)
    write_json(trial_dir / "trial_report.json", metrics)
    return metrics
