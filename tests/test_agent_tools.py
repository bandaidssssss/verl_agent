from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_tools import ToolRegistry
from config_utils import load_json


ROOT = Path(__file__).resolve().parents[1]


class AgentToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.temp_root = Path(self.temp_dir.name)
        self.history_path = self.temp_root / "output" / "trials.jsonl"
        self.history_path.parent.mkdir(parents=True)
        self.config = load_json(ROOT / "config" / "agent_config.json")
        self.base = load_json(ROOT / "config" / "base_parameters.json")

    def registry(self, **overrides: object) -> ToolRegistry:
        config = {**self.config, **overrides}
        return ToolRegistry(ROOT, config, self.history_path)

    def test_parameter_understanding_is_allowlisted(self) -> None:
        registry = self.registry()
        runtime = registry.runtime({})
        result = registry.execute(
            "proposal",
            "parameter_understanding",
            {"items": ["actor_rollout_ref.rollout.gpu_memory_utilization", "missing.parameter"]},
            runtime,
        )
        self.assertIn("actor_rollout_ref.rollout.gpu_memory_utilization", result["parameters"])
        self.assertEqual(result["unknown_parameters"], ["missing.parameter"])
        with self.assertRaises(RuntimeError):
            registry.execute("proposal", "read_trial_log_excerpt", {"trial_id": 1}, runtime)

    def test_memory_estimator_uses_phase_observation_anchor(self) -> None:
        current_micro = self.base[
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"
        ]
        target_micro = current_micro * 2
        trial = {
            "trial_id": 1,
            "parameters": self.base,
            "memory_by_phase_pct": {
                "rollout": {"max": 70.0},
                "actor_log_prob": {"max": 60.0},
                "ref_log_prob": {"max": 55.0},
                "training": {"max": 72.0},
            },
        }
        context = {
            "current_parameters": self.base,
            "recent_trials": [trial],
            "constraints": {"throughput_memory_limit_pct": 92.0},
        }
        registry = self.registry()
        result = registry.execute(
            "proposal",
            "memory_estimator",
            {
                "changes": {
                    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": {
                        "from": current_micro,
                        "to": target_micro,
                    }
                },
                "parameters": {
                    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": target_micro
                },
                "reference_trial_id": 1,
            },
            registry.runtime(context),
        )
        self.assertEqual(result["method"], "empirical_component_relative")
        self.assertEqual(result["reference_trial_id"], 1)
        self.assertGreater(result["phases"]["training"]["projected_pct"], 72.0)
        self.assertEqual(result["phases"]["training"]["risk"], "watch")
        self.assertGreater(
            result["phases"]["training"]["upper_bound_pct"],
            result["phases"]["training"]["projected_pct"],
        )
        self.assertAlmostEqual(result["phases"]["rollout"]["projected_pct"], 70.0)

    def test_memory_estimator_normalizes_proposal_style_changes(self) -> None:
        reference = dict(self.base)
        reference["actor_rollout_ref.rollout.gpu_memory_utilization"] = 0.5
        trial = {
            "trial_id": 1,
            "parameters": reference,
            "memory_by_phase_pct": {
                "rollout": {"max": 59.14764404296875},
                "actor_log_prob": {"max": 31.22100830078125},
                "ref_log_prob": {"max": 65.56396484375},
                "training": {"max": 38.10272216796875},
            },
        }
        context = {
            "current_parameters": reference,
            "candidate_parameters": {
                **reference,
                "actor_rollout_ref.rollout.gpu_memory_utilization": 0.7,
            },
            "recent_trials": [trial],
            "constraints": {"throughput_memory_limit_pct": 92.0},
        }
        registry = self.registry()
        result = registry.execute(
            "feasibility",
            "memory_estimator",
            {
                "changes": {
                    "actor_rollout_ref.rollout.gpu_memory_utilization": {
                        "from": 0.5,
                        "to": 0.7,
                    }
                },
                "parameters": {
                    "actor_rollout_ref.rollout.gpu_memory_utilization": 0.7
                },
                "reference_trial_id": 1,
            },
            registry.runtime(context),
        )
        rollout = result["phases"]["rollout"]
        self.assertGreater(rollout["pressure_ratio"], 1.0)
        self.assertAlmostEqual(rollout["projected_pct"], 79.0, delta=0.2)
        self.assertEqual(
            result["candidate_changes"][
                "actor_rollout_ref.rollout.gpu_memory_utilization"
            ],
            0.7,
        )
        self.assertEqual(rollout["model"], "vllm_budget_relative")
        self.assertEqual(rollout["risk"], "low")
        self.assertEqual(
            result["phases"]["ref_log_prob"]["uncertainty_pct"],
            4.0,
        )

    def test_memory_estimator_treats_rollout_caps_as_uncalibrated(self) -> None:
        key = "actor_rollout_ref.rollout.max_num_seqs"
        reference = {**self.base, key: 256}
        trial = {
            "trial_id": 1,
            "parameters": reference,
            "memory_by_phase_pct": {
                phase: {"max": value}
                for phase, value in {
                    "rollout": 79.0,
                    "actor_log_prob": 31.0,
                    "ref_log_prob": 62.0,
                    "training": 38.0,
                }.items()
            },
        }
        registry = self.registry()
        result = registry.execute(
            "proposal",
            "memory_estimator",
            {
                "changes": {key: {"from": 256, "to": 512}},
                "parameters": {key: 512},
                "reference_trial_id": 1,
            },
            registry.runtime(
                {"current_parameters": reference, "recent_trials": [trial]}
            ),
        )
        rollout = result["phases"]["rollout"]
        self.assertEqual(rollout["projected_pct"], 79.0)
        self.assertIn(key, rollout["uncalibrated_changes"])
        self.assertGreater(rollout["upper_bound_pct"], 79.0)
        self.assertEqual(rollout["confidence"], "low")

    def test_training_component_savings_do_not_multiply_full_peak(self) -> None:
        distributed_key = (
            "actor_rollout_ref.actor.megatron.use_distributed_optimizer"
        )
        optimizer_offload_key = (
            "actor_rollout_ref.actor.megatron.optimizer_offload"
        )
        recompute_key = (
            "actor_rollout_ref.actor.megatron."
            "override_transformer_config.recompute_granularity"
        )
        reference = {
            **self.base,
            distributed_key: False,
            optimizer_offload_key: False,
            recompute_key: None,
        }
        trial = {
            "trial_id": 1,
            "parameters": reference,
            "memory_by_phase_pct": {
                phase: {"max": 80.0}
                for phase in ("rollout", "actor_log_prob", "ref_log_prob", "training")
            },
        }
        changes = {
            distributed_key: {"from": False, "to": True},
            optimizer_offload_key: {"from": False, "to": True},
            recompute_key: {"from": None, "to": "full"},
        }
        targets = {key: value["to"] for key, value in changes.items()}
        registry = self.registry()
        result = registry.execute(
            "proposal",
            "memory_estimator",
            {
                "changes": changes,
                "parameters": targets,
                "reference_trial_id": 1,
            },
            registry.runtime(
                {"current_parameters": reference, "recent_trials": [trial]}
            ),
        )
        training = result["phases"]["training"]
        self.assertEqual(
            training["model"],
            "fixed_model_optimizer_activation_components",
        )
        self.assertGreater(training["pressure_ratio"], 0.60)
        self.assertLess(training["pressure_ratio"], 0.75)
        self.assertEqual(
            sum(training["component_shares"].values()),
            1.0,
        )

    def test_memory_estimator_rejects_reference_mismatch(self) -> None:
        trial = {
            "trial_id": 1,
            "parameters": self.base,
            "memory_by_phase_pct": {
                phase: {"max": 50.0}
                for phase in ("rollout", "actor_log_prob", "ref_log_prob", "training")
            },
        }
        context = {
            "current_parameters": self.base,
            "recent_trials": [trial],
        }
        registry = self.registry()
        with self.assertRaisesRegex(RuntimeError, "does not match reference trial"):
            registry.execute(
                "proposal",
                "memory_estimator",
                {
                    "changes": {
                        "actor_rollout_ref.rollout.gpu_memory_utilization": {
                            "from": 0.4,
                            "to": 0.7,
                        }
                    },
                    "parameters": {
                        "actor_rollout_ref.rollout.gpu_memory_utilization": 0.7
                    },
                    "reference_trial_id": 1,
                },
                registry.runtime(context),
            )

    def test_search_verl_docs_is_bounded_to_configured_root(self) -> None:
        source = self.temp_root / "verl" / "workers" / "config" / "actor.py"
        source.parent.mkdir(parents=True)
        source.write_text("ppo_micro_batch_size_per_gpu controls local batches\n", encoding="utf-8")
        registry = self.registry(verl_root=str(self.temp_root))
        result = registry.execute(
            "proposal",
            "search_verl_docs",
            {"query": "ppo_micro_batch_size_per_gpu", "max_results": 3},
            registry.runtime({}),
        )
        self.assertTrue(result["available"])
        self.assertEqual(result["matches"][0]["path"], "verl/workers/config/actor.py")

    def test_trial_history_query_can_return_successful_parameters(self) -> None:
        rows = [
            {"trial_id": 1, "stage": "hardware_tuning", "result": "fail", "parameters": {"x": 1}},
            {
                "trial_id": 2,
                "stage": "hardware_tuning",
                "result": "success",
                "parameters": {"x": 2},
                "performance": {"throughput": {"mean": 12.0}},
            },
        ]
        self.history_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        registry = self.registry()
        result = registry.execute(
            "proposal",
            "query_trial_history",
            {"result": "success", "sort_by": "throughput", "include_parameters": True},
            registry.runtime({}),
        )
        self.assertEqual(result["matched"], 1)
        self.assertEqual(result["trials"][0]["parameters"], {"x": 2})

    def test_proposal_can_read_bounded_trial_metric_windows(self) -> None:
        log_path = self.history_path.parent / "trials" / "0001" / "train.log"
        log_path.parent.mkdir(parents=True)
        log_path.write_text(
            "\n".join(
                f"step:{step} - critic/rewards/mean:{step / 10} - actor/ppo_kl:{step / 100} - actor/lr:0.000003"
                for step in range(1, 11)
            ),
            encoding="utf-8",
        )
        self.history_path.write_text(
            json.dumps({"trial_id": 1, "log_path": str(log_path)}) + "\n",
            encoding="utf-8",
        )
        registry = self.registry()
        result = registry.execute(
            "proposal",
            "read_trial_metrics",
            {
                "trial_id": 1,
                "metrics": ["critic/rewards/mean", "actor/ppo_kl", "actor/lr"],
                "window_size": 5,
            },
            registry.runtime({}),
        )
        self.assertTrue(result["available"])
        self.assertEqual(result["windows"][0], {"start_step": 1, "end_step": 5, "sample_count": 5})
        self.assertAlmostEqual(result["metrics"]["critic/rewards/mean"][1], 0.8)

    def test_train_health_can_read_immutable_current_trial_snapshot(self) -> None:
        log_path = self.history_path.parent / "trials" / "0008" / "train.log"
        log_path.parent.mkdir(parents=True)
        log_path.write_text(
            "\n".join(
                f"step:{step} - critic/rewards/mean:{-step / 10} - actor/kl_loss:{step / 100}"
                for step in range(1, 8)
            ),
            encoding="utf-8",
        )
        registry = self.registry()
        runtime = registry.runtime(
            {
                "active_trial": {
                    "trial_id": 8,
                    "log_path": str(log_path),
                    "snapshot_step": 5,
                }
            }
        )
        result = registry.execute(
            "train_health",
            "read_current_trial_metrics",
            {
                "metrics": ["critic/rewards/mean", "actor/kl_loss"],
                "window_size": 1,
            },
            runtime,
        )
        self.assertTrue(result["available"])
        self.assertEqual(result["snapshot_step"], 5)
        self.assertEqual(result["latest_available_step"], 5)
        self.assertEqual(result["step_range"], [1, 5])
        self.assertEqual(len(result["windows"]), 5)
        with self.assertRaisesRegex(RuntimeError, "snapshot_step 5"):
            registry.execute(
                "train_health",
                "read_current_trial_metrics",
                {
                    "metrics": ["critic/rewards/mean"],
                    "end_step": 6,
                },
                runtime,
            )

    def test_current_trial_metrics_rejects_runner_path_outside_output(self) -> None:
        outside = self.temp_root / "outside.log"
        outside.write_text("step:1 - critic/rewards/mean:0.1\n", encoding="utf-8")
        registry = self.registry()
        runtime = registry.runtime(
            {
                "active_trial": {
                    "trial_id": 8,
                    "log_path": str(outside),
                    "snapshot_step": 1,
                }
            }
        )
        with self.assertRaisesRegex(RuntimeError, "outside the configured output"):
            registry.execute(
                "train_health",
                "read_current_trial_metrics",
                {"metrics": ["critic/rewards/mean"]},
                runtime,
            )

    def test_rollout_metrics_skill_uses_recorded_vllm_summary(self) -> None:
        trial = {
            "trial_id": 4,
            "parameters": {
                **self.base,
                "actor_rollout_ref.rollout.disable_log_stats": False,
                "actor_rollout_ref.rollout.max_num_seqs": 256,
                "actor_rollout_ref.rollout.max_num_batched_tokens": 65536,
            },
            "memory_by_phase_pct": {"rollout": {"max": 70.0}},
            "gpu_utilization_by_phase_pct": {"rollout": {"mean": 60.0}},
            "rollout_engine": {
                "monitor": {"enabled": True},
                "metrics": {
                    "available": True,
                    "requests_running": {"mean": 220.0, "p95": 256.0, "max": 256.0},
                    "requests_waiting": {"mean": 100.0, "p95": 500.0, "max": 581.0},
                    "kv_cache_usage_pct": {"mean": 40.0, "p95": 50.0, "max": 55.0},
                    "waiting_positive_fraction": 1.0,
                    "preemptions_total": 0.0,
                    "missing_metrics": ["iteration_tokens_p95_upper_bound"],
                },
            },
        }
        self.history_path.write_text(json.dumps(trial) + "\n", encoding="utf-8")
        registry = self.registry()
        result = registry.execute(
            "proposal",
            "analyze_rollout_metrics",
            {"trial_id": 4},
            registry.runtime({}),
        )
        self.assertTrue(result["available"])
        knobs = result["assessment"]["knobs"]
        self.assertEqual(
            knobs["actor_rollout_ref.rollout.max_num_seqs"]["status"],
            "binding_consider_increase_one_step",
        )
        self.assertEqual(
            knobs["actor_rollout_ref.rollout.max_num_batched_tokens"]["status"],
            "unknown_metric_not_exported",
        )


if __name__ == "__main__":
    unittest.main()
