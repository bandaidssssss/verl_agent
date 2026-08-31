from __future__ import annotations

import unittest
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prompt_context import (  # noqa: E402
    compact_candidate_for_prompt,
    compact_reference_history,
    compact_trial_for_prompt,
)


class PromptContextTest(unittest.TestCase):
    def test_hardware_context_selects_only_configured_metrics(self) -> None:
        trial = {
            "trial_id": 3,
            "stage": "hardware_tuning",
            "result": "success",
            "updates_completed": 20,
            "changes": {"rollout.max_num_seqs": {"from": 128, "to": 256}},
            "parameters": {"rollout.max_num_seqs": 256},
            "log_facts": {
                "runtime_parameters": {
                    "available": True,
                    "source": "train.log:resolved_hydra_config",
                    "values": {"rollout.max_num_seqs": 256},
                }
            },
            "structured_metrics": {
                "resource": {
                    "by_phase": {
                        phase: {
                            "mean_used_mib": 10,
                            "p95_used_mib": 20,
                            "max_used_mib": 30,
                            "max_used_gpu_index": "7",
                            "per_gpu_max_used_mib": {"7": 30},
                        }
                        for phase in (
                            "rollout",
                            "actor_log_prob",
                            "ref_log_prob",
                            "training",
                        )
                    }
                },
                "throughput": {
                    "summary": {
                        "throughput": {"mean": 1, "p95": 2, "max": 3},
                        "actor_mfu": {"mean": 0.1, "p95": 0.2, "max": 0.3},
                        "total_num_tokens": {"mean": 100, "p95": 200, "max": 300},
                    }
                },
            },
        }

        compact = compact_trial_for_prompt(
            trial,
            "hardware_tuning",
            ["rollout.max_num_seqs", "rollout.gpu_memory_utilization"],
            hardware_summary_metrics={
                "throughput": "throughput.summary.throughput",
                "actor_mfu": "throughput.summary.actor_mfu",
                "total_num_tokens": "throughput.summary.total_num_tokens",
            },
        )

        rollout = compact["metrics"]["phase_memory_mib"]["rollout"]
        self.assertEqual(
            set(rollout), {"mean_used_mib", "p95_used_mib", "max_used_mib"}
        )
        self.assertNotIn("max_used_gpu_index", str(compact))
        self.assertEqual(
            compact["metrics"]["summary"]["total_num_tokens"]["max"], 300
        )
        self.assertEqual(
            compact["editable_parameter_values"]["rollout.max_num_seqs"],
            {
                "configured_value": 256,
                "explicitly_configured": True,
                "effective_value": 256,
                "effective_source": "train.log:resolved_hydra_config",
            },
        )
        self.assertEqual(
            compact["editable_parameter_values"]["rollout.gpu_memory_utilization"],
            {
                "configured_value": None,
                "explicitly_configured": False,
                "effective_value": None,
                "effective_source": "unavailable",
            },
        )
        self.assertNotIn("missing_metrics", compact)

    def test_stability_context_uses_windows_and_reports_exact_missing_paths(self) -> None:
        trial = {
            "trial_id": 4,
            "stage": "stability_tuning",
            "result": "success",
            "parameters": {"actor.lr": 1e-6},
            "structured_metrics": {
                "stability": {
                    "step_range": [4, 80],
                    "windows": [{"start_step": 4, "end_step": 8}],
                    "window_metrics": {"critic/rewards/mean": [0.1]},
                    "terminal_window": {"start_step": 76, "end_step": 80},
                    "terminal_metrics": {"critic/rewards/mean": 0.1},
                    "steps": [{"step": 4, "critic/rewards/mean": 0.1}],
                    "health": {"healthy": True},
                },
                "evaluation": {
                    "steps": [
                        {
                            "step": 80,
                            "metrics": {
                                "val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@1": 0.375,
                            },
                        }
                    ],
                    "latest_metrics": {
                        "val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@1": 0.375,
                    },
                },
            },
        }

        compact = compact_trial_for_prompt(
            trial,
            "stability_tuning",
            ["actor.lr"],
            stability_metrics=("critic/rewards/mean", "actor/ppo_kl"),
        )

        self.assertNotIn("steps", compact["metrics"])
        self.assertNotIn("health", compact["metrics"])
        self.assertEqual(
            compact["metrics"]["window_metrics"]["critic/rewards/mean"], [0.1]
        )
        self.assertEqual(
            compact["metrics"]["evaluation"],
            {
                "latest_metrics": {
                    "val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@1": 0.375,
                },
                "latest_step": 80,
            },
        )
        self.assertEqual(
            compact["missing_metrics"],
            [
                "stability.window_metrics.actor/ppo_kl",
                "stability.terminal_metrics.actor/ppo_kl",
            ],
        )

    def test_missing_override_uses_observed_effective_value(self) -> None:
        key = "actor_rollout_ref.actor.ppo_max_token_len_per_gpu"
        compact = compact_trial_for_prompt(
            {
                "trial_id": 2,
                "stage": "hardware_tuning",
                "result": "success",
                "parameters": {},
                "log_facts": {
                    "runtime_parameters": {
                        "available": True,
                        "source": "train.log:resolved_hydra_config",
                        "values": {key: 16384},
                    }
                },
                "structured_metrics": {},
            },
            "hardware_tuning",
            [key],
            hardware_summary_metrics={},
        )

        self.assertEqual(
            compact["editable_parameter_values"][key],
            {
                "configured_value": None,
                "explicitly_configured": False,
                "effective_value": 16384,
                "effective_source": "train.log:resolved_hydra_config",
            },
        )

    def test_history_keeps_required_reference_outside_recent_limit(self) -> None:
        trials = [
            {"trial_id": trial_id, "parameters": {}, "structured_metrics": {}}
            for trial_id in range(1, 6)
        ]
        history = compact_reference_history(
            trials,
            "hardware_tuning",
            [],
            required_trial_ids=[1, 4],
            limit=2,
        )
        self.assertEqual([row["trial_id"] for row in history], [1, 4, 5])

    def test_feasibility_candidate_omits_internal_parameter_maps(self) -> None:
        compact = compact_candidate_for_prompt(
            {
                "candidate_id": "candidate_a",
                "reference_trial_id": 3,
                "reference_reason": "measured baseline",
                "reason": "test one hypothesis",
                "changes": {"x": {"from": 1, "to": 2}},
                "target_changes": {"x": 2},
                "candidate_parameters": {"x": 2, "large": "map"},
                "reference_trial": {"parameters": {"x": 1}},
                "expected_effect": {"throughput": "increase"},
                "confidence": 0.7,
            }
        )
        self.assertEqual(
            set(compact),
            {
                "candidate_id",
                "reference_trial_id",
                "reference_reason",
                "reason",
                "changes",
                "expected_effect",
                "confidence",
            },
        )


if __name__ == "__main__":
    unittest.main()
