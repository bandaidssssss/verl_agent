from __future__ import annotations

import math
import re
from collections import deque
from statistics import mean
from typing import Any, Mapping


NUMBER = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
PAIR_RE = re.compile(rf"([^\s:]+):({NUMBER})")
STEP_RE = re.compile(r"(?<![\w/])step:(\d+)")


DEFAULT_REWARD_WINDOW_SIZE = 3
DEFAULT_REWARD_TREND_STEPS = 5
DEFAULT_REWARD_TREND_MIN_DRAWDOWN = 0.15
DEFAULT_REWARD_TREND_TOLERANCE = 0.01
DEFAULT_KL_CHANGE_RATIO_THRESHOLD = 0.50
DEFAULT_KL_CHANGE_ABSOLUTE_THRESHOLD = 0.02
DEFAULT_ENTROPY_DROP_RATIO_THRESHOLD = 0.30
ENTROPY_RATIO_FLOOR = 1e-3


def parse_online_step(line: str) -> tuple[int, dict[str, float]] | None:
    """Parse one verl step summary without waiting for the trial to finish."""
    step_match = STEP_RE.search(line)
    if not step_match:
        return None
    metrics = {key: float(value) for key, value in PAIR_RE.findall(line)}
    if not metrics:
        return None
    return int(step_match.group(1)), metrics


class OnlineHealthMonitor:
    """Small deterministic trigger layer for online training-health reviews.

    The monitor only raises structured events. It never stops a process and never
    calls an Agent. Reward trend degradation, sudden KL change, and sudden
    entropy collapse are the only rule types; a decision layer reviews triggers.
    """

    def __init__(
        self,
        *,
        kl_metric: str = "actor/kl_loss",
        reward_metric: str = "critic/rewards/mean",
        reward_trend_steps: int = DEFAULT_REWARD_TREND_STEPS,
        ratio_epsilon: float = 1e-12,
        warmup_updates: int = 0,
        cooldown_updates: int | None = None,
        reward_window_size: int = DEFAULT_REWARD_WINDOW_SIZE,
        reward_trend_min_drawdown: float = DEFAULT_REWARD_TREND_MIN_DRAWDOWN,
        reward_trend_tolerance: float = DEFAULT_REWARD_TREND_TOLERANCE,
        kl_change_ratio_threshold: float = DEFAULT_KL_CHANGE_RATIO_THRESHOLD,
        kl_change_absolute_threshold: float = DEFAULT_KL_CHANGE_ABSOLUTE_THRESHOLD,
        entropy_drop_ratio_threshold: float = DEFAULT_ENTROPY_DROP_RATIO_THRESHOLD,
    ) -> None:
        if any(
            threshold < 0
            for threshold in (
                reward_trend_min_drawdown,
                kl_change_ratio_threshold,
                kl_change_absolute_threshold,
                entropy_drop_ratio_threshold,
            )
        ):
            raise ValueError("health thresholds must be non-negative")
        if reward_trend_steps < 2:
            raise ValueError("reward_trend_steps must be at least two")
        if reward_window_size < 1 or reward_window_size >= reward_trend_steps:
            raise ValueError(
                "reward_window_size must be positive and smaller than reward_trend_steps"
            )
        self.kl_metric = kl_metric
        self.reward_metric = reward_metric
        self.reward_trend_steps = int(reward_trend_steps)
        self.ratio_epsilon = max(float(ratio_epsilon), 1e-30)
        self.warmup_updates = max(0, int(warmup_updates))
        self.cooldown_updates = max(
            1,
            int(cooldown_updates if cooldown_updates is not None else reward_trend_steps),
        )
        self.reward_window_size = int(reward_window_size)
        self.reward_trend_min_drawdown = float(reward_trend_min_drawdown)
        self.reward_trend_tolerance = max(0.0, float(reward_trend_tolerance))
        self.kl_change_ratio_threshold = float(kl_change_ratio_threshold)
        self.kl_change_absolute_threshold = float(kl_change_absolute_threshold)
        self.entropy_drop_ratio_threshold = float(entropy_drop_ratio_threshold)

        self.previous_kl: float | None = None
        self.previous_entropy: float | None = None
        self.counters = {
            "kl_sudden_change": 0,
            "entropy_collapse": 0,
            "reward_trend": 0,
        }
        self.last_step = 0
        self.last_trigger_step: int | None = None
        self.trigger_count = 0
        self.recent_steps: deque[dict[str, Any]] = deque(
            maxlen=max(10, 2 * self.reward_trend_steps, 3 * self.reward_window_size)
        )
        self.reward_window: deque[float] = deque(maxlen=self.reward_window_size)
        self.reward_window_means: deque[float] = deque(
            maxlen=self.reward_trend_steps - self.reward_window_size + 1
        )
        self.best_reward_window_mean: float | None = None
        self.current_rules: list[dict[str, Any]] = []

    @staticmethod
    def _finite(value: Any) -> float | None:
        if not isinstance(value, (int, float)):
            return None
        number = float(value)
        return number if math.isfinite(number) else None

    def _ratio(self, delta: float, previous: float) -> float:
        return delta / max(abs(previous), self.ratio_epsilon)

    def _advance_counter(self, name: str, active: bool) -> int:
        self.counters[name] = self.counters[name] + 1 if active else 0
        return self.counters[name]

    def add_step(self, step: int, metrics: Mapping[str, float]) -> dict[str, Any] | None:
        kl = self._finite(metrics.get(self.kl_metric))
        reward = self._finite(metrics.get(self.reward_metric))
        entropy = self._finite(metrics.get("actor/entropy"))
        kl_change = None
        kl_change_ratio = None
        entropy_drop_ratio = None
        reward_window_mean = None
        reward_window_drawdown = None
        reward_trend_kind = None
        previous_kl = self.previous_kl
        previous_entropy = self.previous_entropy

        if kl is not None and previous_kl is not None:
            kl_change = kl - previous_kl
            kl_change_ratio = abs(self._ratio(kl_change, previous_kl))
        if entropy is not None and previous_entropy is not None:
            entropy_drop_ratio = (previous_entropy - entropy) / max(
                abs(previous_entropy),
                ENTROPY_RATIO_FLOOR,
            )

        eligible = step > self.warmup_updates
        kl_change_active = bool(
            eligible
            and kl_change is not None
            and kl_change_ratio is not None
            and abs(kl_change) >= self.kl_change_absolute_threshold
            and kl_change_ratio >= self.kl_change_ratio_threshold
        )
        entropy_collapse_active = bool(
            eligible
            and entropy_drop_ratio is not None
            and entropy_drop_ratio >= self.entropy_drop_ratio_threshold
        )
        if reward is not None:
            self.reward_window.append(reward)
            if len(self.reward_window) == self.reward_window_size:
                reward_window_mean = mean(self.reward_window)
                self.reward_window_means.append(reward_window_mean)
                if self.best_reward_window_mean is None:
                    self.best_reward_window_mean = reward_window_mean
                else:
                    self.best_reward_window_mean = max(
                        self.best_reward_window_mean,
                        reward_window_mean,
                    )
                reward_window_drawdown = self.best_reward_window_mean - reward_window_mean

        window_means = list(self.reward_window_means)
        trend_ready = len(window_means) == self.reward_window_means.maxlen
        trend_non_improving = bool(
            trend_ready
            and all(
                current <= previous + self.reward_trend_tolerance
                for previous, current in zip(window_means, window_means[1:])
            )
        )
        reward_trend_active = bool(
            eligible
            and trend_non_improving
            and reward_window_drawdown is not None
            and reward_window_drawdown >= self.reward_trend_min_drawdown
        )
        if reward_trend_active:
            reward_trend_kind = (
                "decline"
                if window_means[-1] < window_means[0] - self.reward_trend_tolerance
                else "low_plateau"
            )

        self._advance_counter("kl_sudden_change", kl_change_active)
        self._advance_counter("entropy_collapse", entropy_collapse_active)
        self._advance_counter("reward_trend", reward_trend_active)
        self.last_step = max(self.last_step, int(step))

        selected_metrics = {
            key: float(value)
            for key, value in metrics.items()
            if key
            in {
                self.kl_metric,
                self.reward_metric,
                "actor/ppo_kl",
                "actor/entropy",
                "actor/grad_norm",
                "actor/pg_clipfrac",
                "response_length/mean",
                "response_length/clip_ratio",
                "response/aborted_ratio",
                "perf/max_memory_allocated_gb",
                "perf/max_memory_reserved_gb",
                "perf/cpu_memory_used_gb",
                "perf/throughput",
                "perf/time_per_step",
            }
        }
        row = {
            "step": int(step),
            "metrics": selected_metrics,
            "derived": {
                "kl_change": kl_change,
                "kl_change_ratio": kl_change_ratio,
                "entropy_drop_ratio": entropy_drop_ratio,
                "reward_window_mean": reward_window_mean,
                "recent_reward_window_means": window_means,
                "best_reward_window_mean": self.best_reward_window_mean,
                "reward_window_drawdown": reward_window_drawdown,
                "reward_trend_kind": reward_trend_kind,
            },
            "counters": dict(self.counters),
        }
        self.recent_steps.append(row)

        if kl is not None:
            self.previous_kl = kl
        if entropy is not None:
            self.previous_entropy = entropy

        rules = []
        if kl_change_active:
            rules.append(
                {
                    "name": "kl_sudden_change",
                    "metric": self.kl_metric,
                    "previous_value": previous_kl,
                    "observed_value": kl,
                    "observed_change": kl_change,
                    "observed_change_ratio": kl_change_ratio,
                    "minimum_absolute_change": self.kl_change_absolute_threshold,
                    "minimum_change_ratio": self.kl_change_ratio_threshold,
                }
            )
        if entropy_collapse_active:
            rules.append(
                {
                    "name": "entropy_sudden_collapse",
                    "metric": "actor/entropy",
                    "previous_value": previous_entropy,
                    "observed_value": entropy,
                    "observed_drop_ratio": entropy_drop_ratio,
                    "minimum_drop_ratio": self.entropy_drop_ratio_threshold,
                }
            )
        if reward_trend_active:
            rules.append(
                {
                    "name": "reward_trend_degradation",
                    "metric": self.reward_metric,
                    "trend_kind": reward_trend_kind,
                    "recent_window_means": window_means,
                    "recent_window_mean": reward_window_mean,
                    "best_window_mean": self.best_reward_window_mean,
                    "observed_drawdown": reward_window_drawdown,
                    "minimum_drawdown": self.reward_trend_min_drawdown,
                    "non_improvement_tolerance": self.reward_trend_tolerance,
                    "window_size": self.reward_window_size,
                    "observation_steps": self.reward_trend_steps,
                }
            )
        self.current_rules = rules
        if not rules:
            return None

        if self.last_trigger_step is not None and step - self.last_trigger_step < self.cooldown_updates:
            return None

        self.last_trigger_step = int(step)
        self.trigger_count += 1
        return {
            "source": "online_health_monitor",
            "step": int(step),
            "severity": "high",
            "rules": rules,
            "operational_parameters": {
                "reward_trend_steps": self.reward_trend_steps,
                "reward_window_size": self.reward_window_size,
                "reward_trend_min_drawdown": self.reward_trend_min_drawdown,
                "reward_trend_tolerance": self.reward_trend_tolerance,
                "kl_change_ratio_threshold": self.kl_change_ratio_threshold,
                "kl_change_absolute_threshold": self.kl_change_absolute_threshold,
                "entropy_drop_ratio_threshold": self.entropy_drop_ratio_threshold,
            },
            "recent_steps": list(self.recent_steps),
        }

    def review_snapshot(self) -> dict[str, Any]:
        """Return current monitor evidence even when no new rule event fired."""
        return {
            "source": "online_health_monitor",
            "step": self.last_step,
            "severity": "high" if self.current_rules else "warning",
            "rules": list(self.current_rules),
            "counters": dict(self.counters),
            "recent_steps": list(self.recent_steps),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "kl_metric": self.kl_metric,
            "reward_metric": self.reward_metric,
            "warmup_updates": self.warmup_updates,
            "cooldown_updates": self.cooldown_updates,
            "operational_parameters": {
                "reward_trend_steps": self.reward_trend_steps,
                "reward_window_size": self.reward_window_size,
                "reward_trend_min_drawdown": self.reward_trend_min_drawdown,
                "reward_trend_tolerance": self.reward_trend_tolerance,
                "kl_change_ratio_threshold": self.kl_change_ratio_threshold,
                "kl_change_absolute_threshold": self.kl_change_absolute_threshold,
                "entropy_drop_ratio_threshold": self.entropy_drop_ratio_threshold,
            },
            "last_step": self.last_step,
            "trigger_count": self.trigger_count,
            "counters": dict(self.counters),
        }
