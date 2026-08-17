from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from agent_tools.memory_estimator_V3 import estimate_phase_memory
from metrics import (
    STABILITY_QUERY_METRICS,
    build_metric_windows,
    records_from_metric_steps,
)
from trial_storage import (
    hydrate_trial,
    read_metrics_for_trial,
    read_trial_indexes,
    read_trials,
    resolve_artifact,
)
from vllm_metrics import assess_rollout_metrics, summarize_vllm_metrics
from validator import IGNORED_PARAMETERS


@dataclass(frozen=True)
class ToolRuntime:
    root: Path
    agent_config: Mapping[str, Any]
    context: Mapping[str, Any]
    history_path: Path


class ToolError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _number(value: str) -> float:
    match = re.search(r"[-+]?\d*\.?\d+", value.replace(",", ""))
    if not match:
        raise ValueError(f"cannot parse numeric GPU field: {value}")
    return float(match.group(0))


def _nested_metric(trial: Mapping[str, Any], *path: str) -> float | None:
    value: Any = trial
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    if isinstance(value, Mapping):
        value = value.get("mean")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _last_stability_reward(trial: Mapping[str, Any]) -> float | None:
    stability = trial.get("stability")
    if not isinstance(stability, Mapping):
        return None
    windows = stability.get("windows")
    metrics = stability.get("metrics")
    rewards = metrics.get("critic/rewards/mean") if isinstance(metrics, Mapping) else None
    window_size = stability.get("window_size")
    if isinstance(windows, list) and isinstance(rewards, list):
        for window, reward in reversed(list(zip(windows, rewards))):
            if not isinstance(window, Mapping) or not isinstance(reward, (int, float)):
                continue
            required = window_size if isinstance(window_size, int) else window.get("sample_count")
            if window.get("sample_count") == required:
                return float(reward)
        return None
    return _nested_metric(trial, "stability", "reward")


def _normalize_memory_changes(
    changes: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """检查是不是仅仅包含支持的参数。Validate the only supported memory-estimator change representation."""
    targets: dict[str, Any] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for raw_key, value in changes.items():
        key = str(raw_key)
        if not isinstance(value, Mapping):
            raise ToolError(f"changes[{key!r}] must contain 'from' and 'to'")
        missing = sorted({"from", "to"} - set(value))
        extra = sorted(set(value) - {"from", "to"})
        if missing or extra:
            raise ToolError(
                f"changes[{key!r}] must contain only 'from' and 'to'; "
                f"missing={missing}, extra={extra}"
            )
        targets[key] = value["to"]
        metadata[key] = {"from": value["from"], "to": value["to"]}
    return targets, metadata


class ToolRegistry:
    def __init__(self, root: str | Path, agent_config: Mapping[str, Any], history_path: str | Path) -> None:
        self.root = Path(root).resolve()
        self.agent_config = dict(agent_config)
        self.history_path = Path(history_path).expanduser().resolve()
        tool_root = self.root / "agent_tools"
        self._skill_config = _load_json(tool_root / "skills.json")
        parameter_doc_config = _load_json(tool_root / "parameter_docs.json")
        self._parameter_docs = parameter_doc_config["data"]
        self._known_parameters = frozenset(parameter_doc_config["known_parameters"])
        self._strategies = _load_json(tool_root / "tuning_strategies.json")["data"]
        self._handlers: dict[str, Callable[[Mapping[str, Any], ToolRuntime], Any]] = {
            "parameter_understanding": self._parameter_understanding,
            "tuning_strategies": self._tuning_strategies,
            "memory_estimator": self._memory_estimator,
            "analyze_rollout_metrics": self._analyze_rollout_metrics,
            "live_gpu_snapshot": self._live_gpu_snapshot,
            "search_verl_docs": self._search_verl_docs,
            "query_trial_history": self._query_trial_history,
            "read_trial_log_excerpt": self._read_trial_log_excerpt,
            "read_trial_metrics": self._read_trial_metrics,
            "read_current_trial_metrics": self._read_current_trial_metrics,
        }

    def definitions(self, role: str) -> list[dict[str, Any]]:
        return [skill for skill in self._skill_config["skills"] if role in skill.get("roles", [])]

    def api_schemas(self, role: str) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": skill["name"],
                    "description": skill["description"],
                    "parameters": skill["parameters"],
                },
            }
            for skill in self.definitions(role)
        ]

    def runtime(self, context: Mapping[str, Any]) -> ToolRuntime:
        return ToolRuntime(self.root, self.agent_config, context, self.history_path)

    def execute(self, role: str, name: str, arguments: Mapping[str, Any], runtime: ToolRuntime) -> Any:
        allowed = {skill["name"]: skill for skill in self.definitions(role)}
        if name not in allowed or name not in self._handlers:
            raise ToolError(f"tool {name!r} is not allowed for role {role!r}")
        if not isinstance(arguments, Mapping):
            raise ToolError("tool arguments must be an object")
        return self._handlers[name](arguments, runtime)

    def _parameter_understanding(self, arguments: Mapping[str, Any], runtime: ToolRuntime) -> dict[str, Any]:
        del runtime
        items = arguments.get("items")
        if not isinstance(items, list) or not items or len(items) > 8:
            raise ToolError("items must contain 1-8 parameter names")

        found: dict[str, Any] = {}
        missing: list[str] = []
        for item in items:
            key = str(item)
            if key not in self._known_parameters:
                missing.append(key)
                continue
            document = self._parameter_docs.get(key)
            found[key] = (
                {"status": "documented", **document}
                if document
                else {"status": "known_no_special_context"}
            )
        return {"parameters": found, "unknown_parameters": missing}

    def _tuning_strategies(self, arguments: Mapping[str, Any], runtime: ToolRuntime) -> dict[str, Any]:
        del runtime
        items = arguments.get("items")
        if not isinstance(items, list) or not items or len(items) > 4:
            raise ToolError("items must contain 1-4 strategy names")
        found = {str(item): self._strategies[str(item)] for item in items if str(item) in self._strategies}
        missing = [str(item) for item in items if str(item) not in self._strategies]
        return {"strategies": found, "unknown_strategies": missing, "available": sorted(self._strategies)}

    def _memory_estimator(self, arguments: Mapping[str, Any], runtime: ToolRuntime) -> dict[str, Any]:
        extra_arguments = sorted(
            set(arguments) - {"changes", "reference_trial_id"}
        )
        if extra_arguments:
            raise ToolError(
                "memory_estimator accepts only changes and reference_trial_id; "
                f"unexpected={extra_arguments}"
            )
        changes = arguments.get("changes")
        if not isinstance(changes, Mapping) or not changes:
            raise ToolError("changes must contain at least one parameter")
        normalized_changes, change_metadata = _normalize_memory_changes(changes)
        ignored = sorted(set(normalized_changes) & IGNORED_PARAMETERS)
        if ignored:
            raise ToolError(
                "reference Megatron topology follows actor topology and cannot be "
                f"estimated as an independent change: {ignored}"
            )
        reference_id = arguments.get("reference_trial_id")
        if not isinstance(reference_id, int) or isinstance(reference_id, bool):
            raise ToolError("reference_trial_id must be an integer")
        reference_index = next(
            (
                row
                for row in read_trial_indexes(runtime.history_path)
                if row.get("trial_id") == reference_id
            ),
            None,
        )
        if reference_index is None:
            raise ToolError(f"reference trial {reference_id} was not found")
        if reference_index.get("schema_version") != 2:
            raise ToolError(
                f"reference trial {reference_id} does not use the current trial schema"
            )
        artifacts = reference_index.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ToolError(f"reference trial {reference_id} has no artifact map")
        for artifact_name in ("parameters", "metrics", "log_facts"):
            relative = artifacts.get(artifact_name)
            if not isinstance(relative, str):
                raise ToolError(
                    f"reference trial {reference_id} is missing {artifact_name} artifact"
                )
            path = resolve_artifact(runtime.history_path, relative)
            if not path.is_file():
                raise ToolError(
                    f"reference trial {reference_id} artifact is missing: {relative}"
                )
        try:
            reference = hydrate_trial(reference_index, runtime.history_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ToolError(
                f"cannot load reference trial {reference_id} artifacts: {exc}"
            ) from exc
        reference_parameters = reference.get("parameters")
        if not isinstance(reference_parameters, Mapping):
            raise ToolError(f"reference trial {reference_id} has no parameters")
        for key, metadata in change_metadata.items():
            observed = reference_parameters.get(key)
            if metadata["from"] != observed:
                raise ToolError(
                    f"changes[{key!r}].from={metadata['from']!r} does not match "
                    f"reference trial {reference_id} value {observed!r}"
                )

        candidate = dict(reference_parameters)
        candidate.update(normalized_changes)
        trials: list[dict[str, Any]] = []
        for row in read_trial_indexes(runtime.history_path):
            if row.get("schema_version") != 2:
                continue
            try:
                hydrated = hydrate_trial(row, runtime.history_path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if (
                isinstance(hydrated.get("parameters"), Mapping)
                and isinstance(hydrated.get("structured_metrics"), Mapping)
                and isinstance(hydrated.get("log_facts"), Mapping)
            ):
                trials.append(hydrated)
        try:
            result = estimate_phase_memory(
                reference,
                candidate,
                trials,
            )
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        return result

    def _analyze_rollout_metrics(
        self, arguments: Mapping[str, Any], runtime: ToolRuntime
    ) -> dict[str, Any]:
        trial_id = arguments.get("trial_id")
        if not isinstance(trial_id, int) or isinstance(trial_id, bool):
            raise ToolError("trial_id must be an integer")
        trial = next(
            (row for row in read_trials(runtime.history_path) if row.get("trial_id") == trial_id),
            None,
        )
        if trial is None:
            return {"available": False, "trial_id": trial_id, "error": "trial not found"}

        parameters = trial.get("parameters")
        if not isinstance(parameters, Mapping):
            return {
                "available": False,
                "trial_id": trial_id,
                "error": "trial has no recorded parameters",
            }
        rollout_engine = trial.get("rollout_engine")
        structured = trial.get("structured_metrics")
        throughput = (
            structured.get("throughput") if isinstance(structured, Mapping) else None
        )
        recorded_summary = (
            throughput.get("vllm") if isinstance(throughput, Mapping) else None
        )
        if not isinstance(recorded_summary, Mapping):
            recorded_summary = (
                rollout_engine.get("metrics")
                if isinstance(rollout_engine, Mapping)
                else None
            )
        summary = dict(recorded_summary) if isinstance(recorded_summary, Mapping) else {}
        metrics_path_value = trial.get("vllm_metrics_path")
        if metrics_path_value:
            metrics_path = Path(str(metrics_path_value)).expanduser().resolve()
            allowed_root = runtime.history_path.parent.resolve()
            try:
                metrics_path.relative_to(allowed_root)
            except ValueError as exc:
                raise ToolError(
                    "recorded vLLM metrics path is outside the configured output directory"
                ) from exc
            if metrics_path.is_file():
                summary = summarize_vllm_metrics(metrics_path)

        rollout_memory = trial.get("memory_by_phase_pct", {}).get("rollout", {})
        rollout_util = trial.get("gpu_utilization_by_phase_pct", {}).get("rollout", {})
        memory_peak = (
            float(rollout_memory["max"])
            if isinstance(rollout_memory, Mapping)
            and isinstance(rollout_memory.get("max"), (int, float))
            else None
        )
        utilization_mean = (
            float(rollout_util["mean"])
            if isinstance(rollout_util, Mapping)
            and isinstance(rollout_util.get("mean"), (int, float))
            else None
        )
        resource = structured.get("resource") if isinstance(structured, Mapping) else None
        devices = resource.get("devices") if isinstance(resource, Mapping) else None
        totals = [
            float(row["total_memory_mib"])
            for row in devices or []
            if isinstance(row, Mapping)
            and isinstance(row.get("total_memory_mib"), (int, float))
        ]
        if totals:
            capacity_mib = min(totals)
            reserve_mib = float(
                runtime.agent_config.get("throughput_memory_reserve_mib", 6554)
            )
            display_limit_pct = 100.0 * (capacity_mib - reserve_mib) / capacity_mib
        else:
            display_limit_pct = 92.0
        assessment = assess_rollout_metrics(
            summary,
            parameters,
            rollout_memory_peak_pct=memory_peak,
            rollout_gpu_util_mean_pct=utilization_mean,
            memory_limit_pct=display_limit_pct,
        )
        monitor = rollout_engine.get("monitor") if isinstance(rollout_engine, Mapping) else None
        return {
            "available": bool(summary.get("available")),
            "trial_id": trial_id,
            "monitor": monitor,
            "metrics": summary,
            "assessment": assessment,
        }

    def _live_gpu_snapshot(self, arguments: Mapping[str, Any], runtime: ToolRuntime) -> dict[str, Any]:
        if arguments:
            raise ToolError("live_gpu_snapshot takes no arguments")
        platform = os.getenv("PLATFORM", str(runtime.agent_config.get("platform", "V5000"))).upper()
        configured = os.getenv("GPU_SMI")
        default = "xpu-smi" if platform == "V5000" else "mx-smi" if platform in {"C550", "METAX"} else "nvidia-smi"
        executable = configured or shutil.which(default)
        if not executable:
            return {
                "available": False,
                "platform": platform,
                "error": f"{default} was not found; set GPU_SMI to a compatible executable",
                "interpretation": "No live snapshot. Use phase-tagged trial memory instead.",
            }
        timeout = min(10.0, max(1.0, float(runtime.agent_config.get("tool_timeout_seconds", 5.0))))
        try:
            proc = subprocess.run(
                [
                    executable,
                    "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            rows = []
            if proc.returncode == 0:
                for line in proc.stdout.splitlines():
                    fields = [part.strip() for part in line.split(",")]
                    if len(fields) == 4:
                        rows.append(fields)
            if not rows:
                proc = subprocess.run(
                    [executable], capture_output=True, text=True, timeout=timeout, check=False
                )
                output = proc.stdout
                if platform in {"C550", "METAX"}:
                    table_lines = [line for line in output.splitlines() if line.strip().startswith("|")]
                    i = 0
                    while i + 1 < len(table_lines):
                        gpu_line = table_lines[i]
                        mem_line = table_lines[i + 1]
                        gpu_match = re.match(r"\|\s*(\d+)\s+", gpu_line)
                        if not gpu_match:
                            i += 1
                            continue
                        util_match = re.search(r"(\d+)%\s+(?:Disabled|Enabled)", gpu_line)
                        mem_match = re.search(r"(\d+)/(\d+)\s*MiB", mem_line)
                        if not mem_match:
                            i += 2
                            continue
                        rows.append([
                            gpu_match.group(1),
                            mem_match.group(1),
                            mem_match.group(2),
                            util_match.group(1) if util_match else "0",
                        ])
                        i += 2
                elif platform == "V5000":
                    for index, line in enumerate(part for part in output.splitlines() if "Default" in part):
                        fields = line.split()
                        if len(fields) >= 13:
                            rows.append([str(index), fields[8], fields[10], fields[12]])
            gpus = []
            for row in rows:
                used, total, utilization = _number(row[1]), _number(row[2]), _number(row[3])
                gpus.append(
                    {
                        "index": row[0],
                        "memory_used_mb": used,
                        "memory_total_mb": total,
                        "memory_pct": round(100.0 * used / total, 2) if total else None,
                        "utilization_pct": utilization,
                    }
                )
            if not gpus:
                return {
                    "available": False,
                    "platform": platform,
                    "executable": executable,
                    "error": (proc.stderr or proc.stdout or "GPU query returned no parseable rows")[-1000:],
                    "interpretation": "Do not infer phase memory from this failed snapshot.",
                }
            return {
                "available": True,
                "timestamp_unix": time.time(),
                "platform": platform,
                "executable": executable,
                "gpus": gpus,
                "summary": {
                    "max_memory_pct": max(gpu["memory_pct"] for gpu in gpus if gpu["memory_pct"] is not None),
                    "mean_utilization_pct": round(sum(gpu["utilization_pct"] for gpu in gpus) / len(gpus), 2),
                },
                "interpretation": "Instantaneous host occupancy only; use trial phase samples for tuning decisions.",
            }
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return {"available": False, "platform": platform, "executable": executable, "error": str(exc)}

    def _search_verl_docs(self, arguments: Mapping[str, Any], runtime: ToolRuntime) -> dict[str, Any]:
        query = arguments.get("query")
        if not isinstance(query, str) or len(query.strip()) < 2:
            raise ToolError("query must contain at least two characters")
        limit = arguments.get("max_results", 6)
        if not isinstance(limit, int):
            raise ToolError("max_results must be an integer")
        limit = max(1, min(10, limit))
        verl_root = Path(os.getenv("VERL_ROOT", str(runtime.agent_config.get("verl_root", "")))).expanduser().resolve()
        if not verl_root.is_dir():
            return {"available": False, "verl_root": str(verl_root), "error": "verl_root does not exist"}
        candidates = [
            verl_root / "verl" / "trainer" / "config",
            verl_root / "verl" / "workers",
            verl_root / "docs",
            verl_root / "examples",
        ]
        allowed_roots = [path.resolve() for path in candidates if path.is_dir()]
        suffixes = {".py", ".yaml", ".yml", ".md", ".rst"}
        phrase = query.strip().lower()
        terms = [term for term in re.findall(r"[a-zA-Z0-9_.]+", phrase) if len(term) >= 3]
        matches: list[tuple[int, str, int, str]] = []
        files_scanned = 0
        for allowed in allowed_roots:
            for path in allowed.rglob("*"):
                if files_scanned >= 5000:
                    break
                if not path.is_file() or path.suffix.lower() not in suffixes:
                    continue
                try:
                    if path.stat().st_size > 1_000_000:
                        continue
                    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                except OSError:
                    continue
                files_scanned += 1
                for index, line in enumerate(lines):
                    lowered = line.lower()
                    score = (20 if phrase in lowered else 0) + sum(1 for term in terms if term in lowered)
                    if score <= 0:
                        continue
                    start, end = max(0, index - 1), min(len(lines), index + 2)
                    snippet = "\n".join(lines[start:end]).strip()
                    relative = str(path.relative_to(verl_root))
                    matches.append((score, relative, index + 1, snippet[:700]))
        matches.sort(key=lambda item: (-item[0], item[1], item[2]))
        return {
            "available": True,
            "verl_root": str(verl_root),
            "query": query,
            "files_scanned": files_scanned,
            "matches": [
                {"path": path, "line": line, "score": score, "snippet": snippet}
                for score, path, line, snippet in matches[:limit]
            ],
        }

    def _query_trial_history(self, arguments: Mapping[str, Any], runtime: ToolRuntime) -> dict[str, Any]:
        trials = read_trial_indexes(runtime.history_path)
        stage = arguments.get("stage")
        result = arguments.get("result")
        failure_type = arguments.get("failure_type")
        if stage:
            trials = [trial for trial in trials if trial.get("stage") == stage]
        if result:
            trials = [trial for trial in trials if trial.get("result") == result]
        if failure_type:
            trials = [trial for trial in trials if trial.get("error", {}).get("type") == failure_type]
        sort_by = arguments.get("sort_by", "trial_id")
        metric_paths = {
            "throughput": ("performance", "throughput"),
            "memory": ("resource", "max_used_mib"),
        }
        if sort_by == "trial_id":
            trials.sort(key=lambda trial: int(trial.get("trial_id", 0)), reverse=True)
        elif sort_by == "reward":
            trials.sort(
                key=lambda trial: (
                    _nested_metric(trial, "scores", "terminal_reward")
                    or _last_stability_reward(trial)
                    or float("-inf")
                ),
                reverse=True,
            )
        elif sort_by in metric_paths:
            path = metric_paths[sort_by]
            score_key = "throughput_mean" if sort_by == "throughput" else None
            trials.sort(
                key=lambda trial: (
                    _nested_metric(trial, "scores", score_key)
                    if score_key
                    else _nested_metric(trial, *path)
                )
                or _nested_metric(trial, *path)
                or float("-inf"),
                reverse=True,
            )
        else:
            raise ToolError(f"unsupported sort_by: {sort_by}")
        limit = arguments.get("limit", 5)
        if not isinstance(limit, int):
            raise ToolError("limit must be an integer")
        include_parameters = bool(arguments.get("include_parameters", False))
        selected = []
        for trial in trials[: max(1, min(10, limit))]:
            hydrated = (
                hydrate_trial(trial, runtime.history_path)
                if include_parameters
                else trial
            )
            row = {
                "trial_id": trial.get("trial_id"),
                "stage": trial.get("stage"),
                "result": trial.get("result"),
                "changes": trial.get("changes", trial.get("proposal", {}).get("changes")),
                "scores": trial.get("scores"),
                "resource": trial.get("resource"),
                "error": trial.get("error"),
            }
            if include_parameters:
                row["parameters"] = hydrated.get("parameters")
            selected.append(row)
        return {"history_path": str(runtime.history_path), "matched": len(trials), "trials": selected}

    def _read_trial_log_excerpt(self, arguments: Mapping[str, Any], runtime: ToolRuntime) -> dict[str, Any]:
        trial_id = arguments.get("trial_id")
        if not isinstance(trial_id, int):
            raise ToolError("trial_id must be an integer")
        trial = next((row for row in read_trials(runtime.history_path) if row.get("trial_id") == trial_id), None)
        if not trial or not trial.get("log_path"):
            return {"available": False, "trial_id": trial_id, "error": "trial or recorded log_path not found"}
        log_path = Path(str(trial["log_path"])).expanduser().resolve()
        allowed_root = runtime.history_path.parent.resolve()
        try:
            log_path.relative_to(allowed_root)
        except ValueError as exc:
            raise ToolError("recorded log path is outside the configured output directory") from exc
        if not log_path.is_file():
            return {"available": False, "trial_id": trial_id, "log_path": str(log_path), "error": "log not found"}
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        max_lines = arguments.get("max_lines", 20)
        if not isinstance(max_lines, int):
            raise ToolError("max_lines must be an integer")
        max_lines = max(1, min(40, max_lines))
        pattern = arguments.get("pattern")
        if pattern:
            if not isinstance(pattern, str) or len(pattern) > 120:
                raise ToolError("pattern must be a string no longer than 120 characters")
            selected = [(index + 1, line) for index, line in enumerate(lines) if pattern.lower() in line.lower()]
            selected = selected[:max_lines]
        else:
            start = max(0, len(lines) - max_lines)
            selected = [(index + 1, lines[index]) for index in range(start, len(lines))]
        return {
            "available": True,
            "trial_id": trial_id,
            "log_path": str(log_path),
            "pattern": pattern,
            "lines": [{"line": index, "text": line[:1200]} for index, line in selected],
        }

    @staticmethod
    def _metric_query_arguments(
        arguments: Mapping[str, Any],
        *,
        snapshot_step: int | None = None,
    ) -> tuple[list[str], int | None, int | None, int]:
        metrics = arguments.get("metrics")
        if not isinstance(metrics, list) or not metrics or len(metrics) > len(STABILITY_QUERY_METRICS):
            raise ToolError("metrics must be a non-empty list within the allowed metric limit")
        requested = [str(metric) for metric in metrics]
        unknown = sorted(set(requested) - set(STABILITY_QUERY_METRICS))
        if unknown:
            raise ToolError(f"unsupported metrics: {unknown}")
        if len(set(requested)) != len(requested):
            raise ToolError("metrics must not contain duplicates")

        start_step = arguments.get("start_step")
        end_step = arguments.get("end_step")
        if start_step is not None and (not isinstance(start_step, int) or start_step < 1):
            raise ToolError("start_step must be a positive integer")
        if end_step is not None and (not isinstance(end_step, int) or end_step < 1):
            raise ToolError("end_step must be a positive integer")
        if isinstance(start_step, int) and isinstance(end_step, int) and start_step > end_step:
            raise ToolError("start_step must not exceed end_step")
        if snapshot_step is not None:
            if end_step is not None and end_step > snapshot_step:
                raise ToolError(
                    f"end_step must not exceed the fixed snapshot_step {snapshot_step}"
                )
            end_step = snapshot_step if end_step is None else end_step
        window_size = arguments.get("window_size", 5)
        if not isinstance(window_size, int) or not 1 <= window_size <= 20:
            raise ToolError("window_size must be an integer from 1 to 20")
        return requested, start_step, end_step, window_size

    @staticmethod
    def _metric_result_from_records(
        records: Mapping[int, Mapping[str, float]],
        *,
        source_path: Path,
        trial_id: int,
        requested: list[str],
        start_step: int | None,
        end_step: int | None,
        window_size: int,
        snapshot_step: int | None = None,
    ) -> dict[str, Any]:
        series = build_metric_windows(
            records,
            requested,
            window_size,
            start_step=start_step,
            end_step=end_step,
        )
        if len(series["windows"]) > 32:
            raise ToolError("query would return more than 32 windows; narrow the step range or increase window_size")
        result = {
            "available": True,
            "trial_id": trial_id,
            "metrics_path": str(source_path),
            **series,
            "missing_metrics": [
                metric
                for metric, values in series["metrics"].items()
                if not any(value is not None for value in values)
            ],
        }
        if snapshot_step is not None:
            result["snapshot_step"] = snapshot_step
            result["latest_available_step"] = max(
                (step for step in records if step <= snapshot_step), default=None
            )
        return result

    def _read_trial_metrics(self, arguments: Mapping[str, Any], runtime: ToolRuntime) -> dict[str, Any]:
        """Read bounded stability steps exclusively from metrics.json."""
        trial_id = arguments.get("trial_id")
        if not isinstance(trial_id, int):
            raise ToolError("trial_id must be an integer")
        requested, start_step, end_step, window_size = self._metric_query_arguments(
            arguments
        )

        trial = next((row for row in read_trials(runtime.history_path) if row.get("trial_id") == trial_id), None)
        if not trial:
            return {"available": False, "trial_id": trial_id, "error": "trial not found"}
        structured = read_metrics_for_trial(trial, runtime.history_path)
        stability = structured.get("stability") if isinstance(structured, Mapping) else None
        records = records_from_metric_steps(
            stability.get("steps") if isinstance(stability, Mapping) else None
        )
        artifacts = trial.get("artifacts")
        metrics_relative = artifacts.get("metrics") if isinstance(artifacts, Mapping) else None
        if isinstance(metrics_relative, str):
            metrics_path = resolve_artifact(runtime.history_path, metrics_relative)
            if not structured:
                return {
                    "available": False,
                    "trial_id": trial_id,
                    "metrics_path": str(metrics_path),
                    "error": "metrics artifact is missing or empty",
                }
            return self._metric_result_from_records(
                records,
                source_path=metrics_path,
                trial_id=trial_id,
                requested=requested,
                start_step=start_step,
                end_step=end_step,
                window_size=window_size,
            )
        return {
            "available": False,
            "trial_id": trial_id,
            "error": "trial does not use the current metrics artifact schema",
        }

    def _read_current_trial_metrics(
        self, arguments: Mapping[str, Any], runtime: ToolRuntime
    ) -> dict[str, Any]:
        """Read a reproducible metric snapshot from the active runner-owned trial."""
        active_trial = runtime.context.get("active_trial")
        if not isinstance(active_trial, Mapping):
            return {"available": False, "error": "active trial context is not available"}
        trial_id = active_trial.get("trial_id")
        snapshot_step = active_trial.get("snapshot_step")
        raw_metrics_path = active_trial.get("metrics_path")
        if not isinstance(trial_id, int) or isinstance(trial_id, bool):
            raise ToolError("active trial context has an invalid trial_id")
        if not isinstance(snapshot_step, int) or isinstance(snapshot_step, bool) or snapshot_step < 1:
            raise ToolError("active trial context has an invalid snapshot_step")
        allowed_root = runtime.history_path.parent.resolve()
        requested, start_step, end_step, window_size = self._metric_query_arguments(
            arguments,
            snapshot_step=snapshot_step,
        )
        if isinstance(raw_metrics_path, str) and raw_metrics_path:
            metrics_path = Path(raw_metrics_path).expanduser().resolve()
            try:
                metrics_path.relative_to(allowed_root)
            except ValueError as exc:
                raise ToolError("active trial metrics path is outside the configured output directory") from exc
            if metrics_path.is_file():
                try:
                    structured = _load_json(metrics_path)
                except (OSError, json.JSONDecodeError) as exc:
                    raise ToolError(f"cannot read active metrics snapshot: {exc}") from exc
                stability = structured.get("stability")
                records = records_from_metric_steps(
                    stability.get("steps") if isinstance(stability, Mapping) else None
                )
                return self._metric_result_from_records(
                    records,
                    source_path=metrics_path,
                    trial_id=trial_id,
                    requested=requested,
                    start_step=start_step,
                    end_step=end_step,
                    window_size=window_size,
                    snapshot_step=snapshot_step,
                )
            return {
                "available": False,
                "trial_id": trial_id,
                "snapshot_step": snapshot_step,
                "metrics_path": str(metrics_path),
                "error": "active metrics snapshot is not available",
            }
        return {
            "available": False,
            "trial_id": trial_id,
            "snapshot_step": snapshot_step,
            "error": "active metrics snapshot is not available",
        }
