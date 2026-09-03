from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent_tools import ToolRegistry
from config_utils import load_json, write_json
from runtime_parameters import resolve_batching_parameters
from trial_storage import trial_artifacts


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

    def write_memory_trial(
        self,
        parameters: dict[str, object],
        *,
        trial_id: int = 1,
        phase_peaks: dict[str, float] | None = None,
        runtime_overrides: dict[str, object] | None = None,
    ) -> None:
        peaks = phase_peaks or {
            "rollout": 46000.0,
            "actor_log_prob": 40000.0,
            "ref_log_prob": 39000.0,
            "training": 47000.0,
        }
        trial_dir = self.history_path.parent / "trials" / f"{trial_id:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        write_json(trial_dir / "parameters.json", parameters)
        write_json(
            trial_dir / "metrics.json",
            {
                "status": "final",
                "resource": {
                    "devices": [{"gpu_index": "0", "total_memory_mib": 65536}],
                    "by_phase": {
                        phase: {
                            "max_used_mib": peak,
                            "max_used_gpu_total_mib": 65536,
                            "source": "gpu_samples.csv",
                        }
                        for phase, peak in peaks.items()
                    },
                    "summary": {},
                },
                "throughput": {"summary": {}, "phase_duration_s": {}, "steps": []},
                "stability": {"steps": [], "window_metrics": {}},
                "error": {"type": None, "evidence": []},
            },
        )
        runtime_values = {
            **parameters,
            **resolve_batching_parameters(parameters)["values"],
            **(runtime_overrides or {}),
        }
        write_json(
            trial_dir / "log_facts.json",
            {
                "source": {
                    "train_log": "train.log",
                    "parser_version": 1,
                    "warnings": [],
                },
                "model_config": {
                    "model_type": "llama",
                    "hidden_size": 64,
                    "num_hidden_layers": 8,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 2,
                    "intermediate_size": 192,
                    "vocab_size": 1024,
                    "torch_dtype": "bfloat16",
                },
                "runtime_parameters": {
                    "available": True,
                    "source": "train.log:resolved_hydra_config",
                    "values": runtime_values,
                },
                "megatron": {
                    "resolved_config": {
                        "tensor_model_parallel_size": 1,
                        "pipeline_model_parallel_size": 1,
                        "bf16": True,
                    },
                    "rank_parameter_counts": [
                        {"tensor_rank": 0, "pipeline_rank": 0, "parameters": 1000000}
                    ],
                    "parameter_summary": {
                        "complete_tp_pp_coverage": True,
                        "most_loaded_shard_parameters": 1000000,
                        "total_parameters": 1000000,
                        "total_parameters_source": "sum_unique_logged_tp_pp_shards",
                        "reference_topology": {
                            "tensor_model_parallel_size": 1,
                            "pipeline_model_parallel_size": 1,
                            "expert_model_parallel_size": 1,
                            "expert_tensor_parallel_size": 1,
                        },
                    },
                },
                "workload": {
                    "sequence_length": {
                        "point_tokens": 128,
                        "upper_tokens": 192,
                        "configured_upper_tokens": 192,
                        "source": "train_log_stable_step_p95_mean_tokens",
                        "sampled_steps": 5,
                    }
                },
            },
        )
        write_json(trial_dir / "trial_report.json", {"trial_id": trial_id})
        row = {
            "trial_id": trial_id,
            "stage": "hardware_tuning",
            "result": "success",
            "artifacts": trial_artifacts(trial_id),
        }
        self.history_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    def test_parameter_understanding_is_allowlisted(self) -> None:
        registry = self.registry()
        runtime = registry.runtime({})
        result = registry.execute(
            "proposal",
            "parameter_understanding",
            {
                "items": [
                    "actor_rollout_ref.rollout.gpu_memory_utilization",
                    "data.train_batch_size",
                    "missing.parameter",
                ]
            },
            runtime,
        )
        documented = result["parameters"]["actor_rollout_ref.rollout.gpu_memory_utilization"]
        self.assertEqual(documented["status"], "documented")
        self.assertIn("non_obvious_effects", documented)
        self.assertNotIn("type", documented)
        self.assertNotIn("impact", documented)
        self.assertNotIn("increase", documented)
        self.assertNotIn("decrease", documented)
        self.assertEqual(
            result["parameters"]["data.train_batch_size"],
            {"status": "known_no_special_context"},
        )
        self.assertEqual(result["unknown_parameters"], ["missing.parameter"])
        with self.assertRaises(RuntimeError):
            registry.execute("proposal", "read_trial_log_excerpt", {"trial_id": 1}, runtime)

    def test_parameter_understanding_describes_shared_actor_ref_authority(self) -> None:
        registry = self.registry()
        result = registry.execute(
            "proposal",
            "parameter_understanding",
            {
                "items": [
                    "actor_rollout_ref.actor.megatron.tensor_model_parallel_size",
                    "actor_rollout_ref.ref.megatron.tensor_model_parallel_size",
                    "actor_rollout_ref.actor.entropy_coeff",
                ]
            },
            registry.runtime({}),
        )
        actor = result["parameters"]["actor_rollout_ref.actor.megatron.tensor_model_parallel_size"]
        ref = result["parameters"]["actor_rollout_ref.ref.megatron.tensor_model_parallel_size"]
        entropy = result["parameters"]["actor_rollout_ref.actor.entropy_coeff"]
        self.assertIn("both actor and ref", actor["non_obvious_effects"][0])
        self.assertIn("Ignored at runtime", ref["runtime_authority"])
        self.assertEqual(ref["not_tunable_reason"], "Retained only as Hydra provenance.")
        self.assertIn("exactly zero", entropy["non_obvious_effects"][0])

    def test_memory_estimator_uses_phase_observation_anchor(self) -> None:
        current_micro = 2
        target_micro = current_micro * 2
        reference = {
            **self.base,
            "actor_rollout_ref.actor.use_dynamic_bsz": False,
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": current_micro,
        }
        self.write_memory_trial(reference)
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
                "reference_trial_id": 1,
            },
            registry.runtime({}),
        )
        self.assertEqual(result["reference_trial_id"], 1)
        training = result["phases"]["training"]
        self.assertEqual(training["status"], "estimated")
        self.assertGreater(
            training["estimated_relative_change_pct"], 0
        )
        self.assertEqual(result["phases"]["rollout"]["status"], "unaffected")
        self.assertEqual(
            result["phases"]["rollout"]["estimated_relative_change_pct"], 0.0
        )

    def test_memory_estimator_normalizes_proposal_style_changes(self) -> None:
        reference = dict(self.base)
        reference["actor_rollout_ref.rollout.gpu_memory_utilization"] = 0.6
        self.write_memory_trial(reference)
        registry = self.registry()
        result = registry.execute(
            "feasibility",
            "memory_estimator",
            {
                "changes": {
                    "actor_rollout_ref.rollout.gpu_memory_utilization": {
                        "from": 0.6,
                        "to": 0.7,
                    }
                },
                "reference_trial_id": 1,
            },
            registry.runtime({}),
        )
        rollout = result["phases"]["rollout"]
        self.assertEqual(rollout["status"], "estimated")
        self.assertGreater(rollout["estimated_relative_change_pct"], 0)
        self.assertEqual(rollout["reference_peak_mib"], 46000.0)
        self.assertGreater(rollout["estimated_peak_mib"], 46000.0)
        self.assertEqual(result["safety"], "within_limit")
        self.assertIn("does not certify an unevaluated target", result["note"])
        serialized = json.dumps(result, sort_keys=True)
        for removed_field in (
            "projected_memory_mb",
            "upper_bound_memory_mb",
            "memory_limit_pct",
            '"risk"',
        ):
            self.assertNotIn(removed_field, serialized)
        self.assertEqual(result["phases"]["ref_log_prob"]["status"], "unaffected")

    def test_memory_estimator_treats_rollout_caps_as_uncalibrated(self) -> None:
        key = "actor_rollout_ref.rollout.max_num_seqs"
        reference = {**self.base, key: 256}
        self.write_memory_trial(reference)
        registry = self.registry()
        result = registry.execute(
            "proposal",
            "memory_estimator",
            {
                "changes": {key: {"from": 256, "to": 512}},
                "reference_trial_id": 1,
            },
            registry.runtime({}),
        )
        rollout = result["phases"]["rollout"]
        self.assertEqual(rollout["status"], "unmodeled")
        self.assertIsNone(rollout["estimated_peak_mib"])
        self.assertIsNone(rollout["estimated_relative_change_pct"])
        self.assertEqual(result["safety"], "unknown")
        self.assertNotIn("evaluate a larger target", result["note"])

    def test_memory_estimator_marks_dynamic_micro_batch_as_inactive(self) -> None:
        key = "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"
        self.write_memory_trial(self.base)
        result = self.registry().execute(
            "proposal",
            "memory_estimator",
            {
                "changes": {key: {"from": None, "to": 4}},
                "reference_trial_id": 1,
            },
            self.registry().runtime({}),
        )

        training = result["phases"]["training"]
        self.assertEqual(training["status"], "inactive")
        self.assertEqual(training["estimated_relative_change_pct"], 0.0)
        self.assertEqual(training["reference_peak_mib"], training["estimated_peak_mib"])
        self.assertEqual(result["safety"], "within_limit")
        self.assertNotIn("evaluate a larger target", result["note"])

    def test_memory_estimator_marks_fixed_batch_max_tokens_as_inactive(self) -> None:
        dynamic_key = "actor_rollout_ref.actor.use_dynamic_bsz"
        micro_key = "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"
        max_tokens_key = "actor_rollout_ref.actor.ppo_max_token_len_per_gpu"
        reference = {
            **self.base,
            dynamic_key: False,
            micro_key: 2,
        }
        self.write_memory_trial(reference)
        result = self.registry().execute(
            "proposal",
            "memory_estimator",
            {
                "changes": {max_tokens_key: {"from": 24576, "to": 32768}},
                "reference_trial_id": 1,
            },
            self.registry().runtime({}),
        )

        training = result["phases"]["training"]
        self.assertEqual(training["status"], "inactive")
        self.assertEqual(training["estimated_relative_change_pct"], 0.0)
        self.assertNotIn("evaluate a larger target", result["note"])

    def test_memory_estimator_allows_inactive_companion_in_increase_note(self) -> None:
        micro_key = "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"
        max_tokens_key = "actor_rollout_ref.actor.ppo_max_token_len_per_gpu"
        reference = {
            **self.base,
            "actor_rollout_ref.actor.use_dynamic_bsz": False,
            micro_key: 2,
            max_tokens_key: 24576,
        }
        self.write_memory_trial(
            reference,
            phase_peaks={
                "rollout": 20000.0,
                "actor_log_prob": 20000.0,
                "ref_log_prob": 20000.0,
                "training": 20000.0,
            },
        )
        result = self.registry().execute(
            "proposal",
            "memory_estimator",
            {
                "changes": {
                    micro_key: {"from": 2, "to": 4},
                    max_tokens_key: {"from": 24576, "to": 32768},
                },
                "reference_trial_id": 1,
            },
            self.registry().runtime({}),
        )

        self.assertEqual(result["phases"]["training"]["status"], "estimated")
        self.assertIn("evaluate a larger target", result["note"])

    def test_memory_estimator_marks_missing_unaffected_phase_unavailable(self) -> None:
        key = "actor_rollout_ref.rollout.gpu_memory_utilization"
        reference = {**self.base, key: 0.6}
        self.write_memory_trial(reference)
        metrics_path = self.history_path.parent / "trials" / "0001" / "metrics.json"
        metrics = load_json(metrics_path)
        del metrics["resource"]["by_phase"]["ref_log_prob"]
        write_json(metrics_path, metrics)
        result = self.registry().execute(
            "proposal",
            "memory_estimator",
            {
                "changes": {key: {"from": 0.6, "to": 0.7}},
                "reference_trial_id": 1,
            },
            self.registry().runtime({}),
        )

        ref_log_prob = result["phases"]["ref_log_prob"]
        self.assertEqual(ref_log_prob["status"], "unavailable")
        self.assertIsNone(ref_log_prob["estimated_peak_mib"])
        self.assertIsNone(ref_log_prob["estimated_relative_change_pct"])
        self.assertEqual(result["safety"], "unknown")
        self.assertNotIn("evaluate a larger target", result["note"])

    def test_memory_estimator_marks_unsafe_candidate(self) -> None:
        key = "actor_rollout_ref.rollout.gpu_memory_utilization"
        reference = {**self.base, key: 0.6}
        self.write_memory_trial(reference)
        result = self.registry().execute(
            "proposal",
            "memory_estimator",
            {
                "changes": {key: {"from": 0.6, "to": 0.9}},
                "reference_trial_id": 1,
            },
            self.registry().runtime({}),
        )

        self.assertEqual(result["safety"], "exceeds_limit")
        self.assertIn("not memory-feasible", result["note"])
        self.assertNotIn("evaluate a larger target", result["note"])

    def test_memory_estimator_marks_missing_affected_phase_unavailable(self) -> None:
        key = "actor_rollout_ref.actor.ppo_max_token_len_per_gpu"
        self.write_memory_trial(self.base)
        metrics_path = self.history_path.parent / "trials" / "0001" / "metrics.json"
        metrics = load_json(metrics_path)
        del metrics["resource"]["by_phase"]["training"]
        write_json(metrics_path, metrics)
        result = self.registry().execute(
            "proposal",
            "memory_estimator",
            {
                "changes": {key: {"from": 24576, "to": 32768}},
                "reference_trial_id": 1,
            },
            self.registry().runtime({}),
        )

        training = result["phases"]["training"]
        self.assertEqual(training["status"], "unavailable")
        self.assertIsNone(training["estimated_peak_mib"])
        self.assertIsNone(training["estimated_relative_change_pct"])
        self.assertEqual(result["safety"], "unknown")

    def test_dynamic_batch_and_remove_padding_use_modeled_activation_deltas(self) -> None:
        dynamic_key = "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz"
        padding_key = "actor_rollout_ref.actor.megatron.use_remove_padding"
        reference = {**self.base, padding_key: True, dynamic_key: False}
        self.write_memory_trial(reference)
        result = self.registry().execute(
            "proposal",
            "memory_estimator",
            {
                "changes": {
                    dynamic_key: {"from": False, "to": True},
                    padding_key: {"from": True, "to": False},
                },
                "reference_trial_id": 1,
            },
            self.registry().runtime({}),
        )

        actor_log_prob = result["phases"]["actor_log_prob"]
        self.assertEqual(actor_log_prob["status"], "estimated")
        self.assertGreater(
            actor_log_prob["estimated_relative_change_pct"], 0.0
        )

    def test_explicit_actor_switch_uses_observed_reference_controls(self) -> None:
        actor_dynamic = "actor_rollout_ref.actor.use_dynamic_bsz"
        actor_cap = "actor_rollout_ref.actor.ppo_max_token_len_per_gpu"
        rollout_dynamic = (
            "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz"
        )
        ref_dynamic = "actor_rollout_ref.ref.log_prob_use_dynamic_bsz"
        reference = {
            **self.base,
            actor_dynamic: False,
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 8,
        }
        for key in (actor_cap, rollout_dynamic, ref_dynamic):
            reference.pop(key, None)
        runtime = {
            actor_dynamic: False,
            actor_cap: 16384,
            rollout_dynamic: False,
            "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": 16384,
            ref_dynamic: False,
            "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu": 16384,
        }
        self.write_memory_trial(reference, runtime_overrides=runtime)
        result = self.registry().execute(
            "proposal",
            "memory_estimator",
            {
                "changes": {
                    actor_dynamic: {"from": False, "to": True},
                    actor_cap: {"from": None, "to": 6400},
                    rollout_dynamic: {"from": None, "to": False},
                    ref_dynamic: {"from": None, "to": False},
                },
                "reference_trial_id": 1,
            },
            self.registry().runtime({}),
        )

        candidate_batching = result["resolved_batching"]["candidate"]
        self.assertTrue(candidate_batching["training"]["dynamic"])
        self.assertEqual(
            candidate_batching["training"]["max_token_len_per_gpu"], 6400
        )
        self.assertFalse(candidate_batching["actor_log_prob"]["dynamic"])
        self.assertFalse(candidate_batching["ref_log_prob"]["dynamic"])
        self.assertEqual(result["phases"]["training"]["status"], "estimated")

    def test_memory_estimator_rejects_log_facts_without_runtime_snapshot(self) -> None:
        key = "actor_rollout_ref.rollout.gpu_memory_utilization"
        reference = {**self.base, key: 0.6}
        self.write_memory_trial(reference)
        facts_path = self.history_path.parent / "trials" / "0001" / "log_facts.json"
        facts = load_json(facts_path)
        facts.pop("runtime_parameters")
        write_json(facts_path, facts)

        with self.assertRaisesRegex(Exception, "re-extract"):
            self.registry().execute(
                "proposal",
                "memory_estimator",
                {
                    "changes": {key: {"from": 0.6, "to": 0.7}},
                    "reference_trial_id": 1,
                },
                self.registry().runtime({}),
            )

    def test_entropy_zero_disables_training_workspace_only(self) -> None:
        key = "actor_rollout_ref.actor.entropy_coeff"
        reference = {**self.base, key: 0.01}
        self.write_memory_trial(reference)
        registry = self.registry()
        result = registry.execute(
            "proposal",
            "memory_estimator",
            {
                "changes": {key: {"from": 0.01, "to": 0.0}},
                "reference_trial_id": 1,
            },
            registry.runtime({}),
        )
        training = result["phases"]["training"]
        self.assertEqual(training["status"], "estimated")
        self.assertLess(training["estimated_relative_change_pct"], 0)
        self.assertEqual(result["phases"]["rollout"]["status"], "unaffected")

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
        self.write_memory_trial(reference)
        changes = {
            distributed_key: {"from": False, "to": True},
            optimizer_offload_key: {"from": False, "to": True},
            recompute_key: {"from": None, "to": "full"},
        }
        registry = self.registry()
        result = registry.execute(
            "proposal",
            "memory_estimator",
            {
                "changes": changes,
                "reference_trial_id": 1,
            },
            registry.runtime({}),
        )
        training = result["phases"]["training"]
        self.assertEqual(training["status"], "estimated")
        self.assertLess(training["estimated_relative_change_pct"], 0)

    def test_memory_estimator_rejects_reference_mismatch(self) -> None:
        self.write_memory_trial(self.base)
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
                    "reference_trial_id": 1,
                },
                registry.runtime({}),
            )

    def test_memory_estimator_schema_has_only_changes_and_reference(self) -> None:
        schema = next(
            item["function"]["parameters"]
            for item in self.registry().api_schemas("proposal")
            if item["function"]["name"] == "memory_estimator"
        )
        self.assertEqual(
            set(schema["properties"]), {"changes", "reference_trial_id"}
        )
        self.assertEqual(
            set(schema["required"]), {"changes", "reference_trial_id"}
        )

    def test_memory_estimator_rejects_ref_topology_change(self) -> None:
        self.write_memory_trial(self.base)
        key = "actor_rollout_ref.ref.megatron.tensor_model_parallel_size"
        with self.assertRaisesRegex(RuntimeError, "follows actor topology"):
            self.registry().execute(
                "proposal",
                "memory_estimator",
                {
                    "changes": {key: {"from": self.base.get(key), "to": 2}},
                    "reference_trial_id": 1,
                },
                self.registry().runtime({}),
            )

    def test_memory_estimator_requires_log_facts_artifact(self) -> None:
        self.write_memory_trial(self.base)
        (self.history_path.parent / "trials" / "0001" / "log_facts.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "artifact is missing"):
            self.registry().execute(
                "proposal",
                "memory_estimator",
                {
                    "changes": {
                        "actor_rollout_ref.rollout.gpu_memory_utilization": {
                            "from": self.base[
                                "actor_rollout_ref.rollout.gpu_memory_utilization"
                            ],
                            "to": 0.7,
                        }
                    },
                    "reference_trial_id": 1,
                },
                self.registry().runtime({}),
            )

    def test_rollout_n_has_cross_phase_memory_effects(self) -> None:
        key = "actor_rollout_ref.rollout.n"
        reference = {**self.base, key: 2}
        self.write_memory_trial(reference)
        result = self.registry().execute(
            "proposal",
            "memory_estimator",
            {
                "changes": {key: {"from": 2, "to": 4}},
                "reference_trial_id": 1,
            },
            self.registry().runtime({}),
        )
        rollout = result["phases"]["rollout"]
        training = result["phases"]["training"]
        self.assertEqual(rollout["status"], "unmodeled")
        self.assertIsNone(rollout["estimated_relative_change_pct"])
        self.assertEqual(training["status"], "estimated")
        self.assertGreaterEqual(training["estimated_relative_change_pct"], 0)
        self.assertEqual(result["safety"], "unknown")

    def test_repeated_memory_estimates_never_open_train_log(self) -> None:
        key = "actor_rollout_ref.rollout.gpu_memory_utilization"
        reference = {**self.base, key: 0.5}
        self.write_memory_trial(reference)
        trial_log = self.history_path.parent / "trials" / "0001" / "train.log"
        trial_log.write_text("must not be read", encoding="utf-8")
        original_open = Path.open

        def guarded_open(path: Path, *args, **kwargs):
            if path.resolve() == trial_log.resolve():
                raise AssertionError("memory estimator opened train.log")
            return original_open(path, *args, **kwargs)

        arguments = {
            "changes": {key: {"from": 0.5, "to": 0.7}},
            "reference_trial_id": 1,
        }
        registry = self.registry()
        with mock.patch.object(Path, "open", guarded_open):
            registry.execute(
                "proposal", "memory_estimator", arguments, registry.runtime({})
            )
            registry.execute(
                "proposal", "memory_estimator", arguments, registry.runtime({})
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

    def test_trial_history_query_returns_ordered_reference_parameters_and_metrics(self) -> None:
        parameter_key = "actor_rollout_ref.rollout.max_num_seqs"
        rows = [
            {
                "trial_id": 1,
                "stage": "hardware_tuning",
                "result": "fail",
                "changes": {parameter_key: {"from": 64, "to": 128}},
                "artifacts": trial_artifacts(1),
            },
            {
                "trial_id": 2,
                "stage": "hardware_tuning",
                "result": "success",
                "changes": {parameter_key: {"from": 128, "to": 256}},
                "artifacts": trial_artifacts(2),
            },
        ]
        for trial_id in (1, 2):
            trial_dir = self.history_path.parent / "trials" / f"{trial_id:04d}"
            trial_dir.mkdir(parents=True)
            write_json(
                trial_dir / "parameters.json",
                {parameter_key: trial_id * 128, "data.train_batch_size": 64},
            )
            write_json(
                trial_dir / "log_facts.json",
                {
                    "runtime_parameters": {
                        "available": True,
                        "source": "train.log:resolved_hydra_config",
                        "values": {parameter_key: trial_id * 128},
                    }
                },
            )
            write_json(
                trial_dir / "metrics.json",
                {
                    "resource": {
                        "by_phase": {
                            phase: {
                                "mean_used_mib": trial_id * 1000,
                                "p95_used_mib": trial_id * 1100,
                                "max_used_mib": trial_id * 1200,
                                "max_used_gpu_index": "7",
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
                            "throughput": {
                                "mean": trial_id * 10,
                                "p95": trial_id * 11,
                                "max": trial_id * 12,
                            },
                            "actor_mfu": {
                                "mean": trial_id / 10,
                                "p95": trial_id / 9,
                                "max": trial_id / 8,
                            },
                        }
                    },
                },
            )
        self.history_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        registry = self.registry()
        result = registry.execute(
            "proposal",
            "query_trial_history",
            {"reference_trial_ids": [2, 99, 1], "stage": "hardware"},
            registry.runtime({}),
        )
        self.assertEqual(result["stage"], "hardware")
        references = result["reference_trials"]
        self.assertEqual([row["trial_id"] for row in references], [2, 99, 1])
        self.assertEqual(
            references[0]["parameters"][parameter_key],
            {
                "configured_value": 256,
                "explicitly_configured": True,
                "effective_value": 256,
                "effective_source": "train.log:resolved_hydra_config",
            },
        )
        self.assertNotIn("data.train_batch_size", references[0]["parameters"])
        self.assertEqual(
            references[0]["metrics"]["summary"]["throughput"]["mean"], 20
        )
        self.assertNotIn("gpu_index", json.dumps(references[0]))
        self.assertFalse(references[1]["available"])
        self.assertEqual(references[1]["error"], "reference trial not found")

    def test_trial_history_query_rejects_duplicate_reference_ids(self) -> None:
        registry = self.registry()
        with self.assertRaisesRegex(RuntimeError, "must not contain duplicates"):
            registry.execute(
                "proposal",
                "query_trial_history",
                {"reference_trial_ids": [1, 1], "stage": "hardware"},
                registry.runtime({}),
            )

    def test_trial_history_schema_is_reference_batch_only(self) -> None:
        registry = self.registry()
        schema = next(
            row["function"]["parameters"]
            for row in registry.api_schemas("proposal")
            if row["function"]["name"] == "query_trial_history"
        )
        self.assertEqual(
            set(schema["properties"]), {"reference_trial_ids", "stage"}
        )
        self.assertEqual(
            set(schema["required"]), {"reference_trial_ids", "stage"}
        )

    def test_trial_history_query_selects_stability_parameters_and_metrics(self) -> None:
        learning_rate = "actor_rollout_ref.actor.optim.lr"
        trial_dir = self.history_path.parent / "trials" / "0001"
        trial_dir.mkdir(parents=True)
        write_json(
            trial_dir / "parameters.json",
            {
                learning_rate: 3e-6,
                "actor_rollout_ref.rollout.max_num_seqs": 256,
            },
        )
        write_json(
            trial_dir / "log_facts.json",
            {
                "runtime_parameters": {
                    "available": True,
                    "source": "train.log:resolved_hydra_config",
                    "values": {learning_rate: 3e-6},
                }
            },
        )
        write_json(
            trial_dir / "metrics.json",
            {
                "stability": {
                    "step_range": [4, 8],
                    "windows": [
                        {"start_step": 4, "end_step": 8, "sample_count": 5}
                    ],
                    "window_metrics": {"critic/rewards/mean": [0.25]},
                    "terminal_window": {
                        "start_step": 4,
                        "end_step": 8,
                        "sample_count": 5,
                    },
                    "terminal_metrics": {"critic/rewards/mean": 0.25},
                },
                "evaluation": {
                    "steps": [
                        {
                            "step": 4,
                            "metrics": {
                                "val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@1": 0.35,
                            },
                        },
                        {
                            "step": 8,
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
        )
        self.history_path.write_text(
            json.dumps(
                {
                    "trial_id": 1,
                    "stage": "hardware_tuning",
                    "result": "success",
                    "changes": {},
                    "artifacts": trial_artifacts(1),
                }
            )
            + "\n",
            encoding="utf-8",
        )

        registry = self.registry()
        result = registry.execute(
            "proposal",
            "query_trial_history",
            {"reference_trial_ids": [1], "stage": "stability"},
            registry.runtime({}),
        )

        reference = result["reference_trials"][0]
        self.assertEqual(
            reference["parameters"][learning_rate],
            {
                "configured_value": 3e-6,
                "explicitly_configured": True,
                "effective_value": 3e-6,
                "effective_source": "train.log:resolved_hydra_config",
            },
        )
        self.assertNotIn(
            "actor_rollout_ref.rollout.max_num_seqs", reference["parameters"]
        )
        self.assertEqual(
            reference["metrics"]["terminal_metrics"]["critic/rewards/mean"],
            0.25,
        )
        self.assertEqual(
            reference["metrics"]["evaluation"],
            {
                "metric": "val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@1",
                "steps": [
                    {"step": 4, "value": 0.35},
                    {"step": 8, "value": 0.375},
                ],
            },
        )
        self.assertIn(
            "stability.window_metrics.actor/ppo_kl",
            reference["missing_metrics"],
        )

    def test_proposal_can_read_bounded_trial_metric_windows(self) -> None:
        trial_dir = self.history_path.parent / "trials" / "0001"
        trial_dir.mkdir(parents=True)
        write_json(
            trial_dir / "metrics.json",
            {
                "stability": {
                    "steps": [
                        {
                            "step": step,
                            "metrics": {
                                "critic/rewards/mean": step / 10,
                                "actor/ppo_kl": step / 100,
                                "actor/lr": 0.000003,
                            },
                        }
                        for step in range(1, 11)
                    ]
                },
                "evaluation": {
                    "steps": [
                        {
                            "step": step,
                            "metrics": {
                                "val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@1": value,
                            },
                        }
                        for step, value in ((1, 0.2), (5, 0.3), (10, 0.25))
                    ]
                },
            },
        )
        self.history_path.write_text(
            json.dumps(
                {
                    "trial_id": 1,
                    "artifacts": {"metrics": "trials/0001/metrics.json"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        registry = self.registry()
        result = registry.execute(
            "proposal",
            "read_trial_metrics",
            {
                "trial_id": 1,
                "metrics": ["critic/rewards/mean", "actor/ppo_kl", "actor/lr"],
                "start_step": 2,
                "end_step": 9,
                "window_size": 5,
            },
            registry.runtime({}),
        )
        self.assertTrue(result["available"])
        self.assertEqual(result["windows"][0], {"start_step": 2, "end_step": 6, "sample_count": 5})
        self.assertAlmostEqual(result["metrics"]["critic/rewards/mean"][1], 0.8)
        self.assertEqual(
            result["evaluation"],
            {
                "metric": "val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@1",
                "steps": [{"step": 5, "value": 0.3}],
            },
        )

    def test_v2_trial_metrics_never_reparse_train_log(self) -> None:
        trial_dir = self.history_path.parent / "trials" / "0001"
        trial_dir.mkdir(parents=True)
        (trial_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "status": "final",
                    "stability": {
                        "steps": [
                            {
                                "step": step,
                                "metrics": {"critic/rewards/mean": step / 10},
                            }
                            for step in range(1, 6)
                        ]
                    },
                    "throughput": {},
                    "resource": {},
                }
            ),
            encoding="utf-8",
        )
        (trial_dir / "parameters.json").write_text("{}", encoding="utf-8")
        self.history_path.write_text(
            json.dumps(
                {
                    "trial_id": 1,
                    "stage": "stability_tuning",
                    "result": "success",
                    "artifacts": {
                        "metrics": "trials/0001/metrics.json",
                        "parameters": "trials/0001/parameters.json",
                        "log": "trials/0001/train.log",
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        registry = self.registry()
        original_open = Path.open

        def guarded_open(path: Path, *args, **kwargs):
            if path.name == "train.log":
                raise AssertionError("train.log must not be parsed")
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", guarded_open):
            result = registry.execute(
                "proposal",
                "read_trial_metrics",
                {
                    "trial_id": 1,
                    "metrics": ["critic/rewards/mean"],
                    "window_size": 5,
                },
                registry.runtime({}),
            )
        self.assertTrue(result["available"])
        self.assertEqual(result["step_range"], [1, 5])
        self.assertIn("metrics_path", result)

    def test_train_health_can_read_immutable_current_trial_snapshot(self) -> None:
        metrics_path = self.history_path.parent / "trials" / "0008" / "metrics.json"
        metrics_path.parent.mkdir(parents=True)
        write_json(
            metrics_path,
            {
                "status": "running",
                "latest_step": 5,
                "stability": {
                    "steps": [
                        {
                            "step": step,
                            "metrics": {
                                "critic/rewards/mean": -step / 10,
                                "actor/kl_loss": step / 100,
                            },
                        }
                        for step in range(1, 6)
                    ]
                },
                "evaluation": {
                    "steps": [
                        {
                            "step": step,
                            "metrics": {
                                "val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@1": value,
                            },
                        }
                        for step, value in ((2, 0.2), (5, 0.3), (6, 0.25))
                    ]
                },
            },
        )
        registry = self.registry()
        runtime = registry.runtime(
            {
                "active_trial": {
                    "trial_id": 8,
                    "metrics_path": str(metrics_path),
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
        self.assertEqual(
            result["evaluation"],
            {
                "metric": "val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@1",
                "steps": [
                    {"step": 2, "value": 0.2},
                    {"step": 5, "value": 0.3},
                ],
            },
        )
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
        outside = self.temp_root / "outside.json"
        outside.write_text("{}", encoding="utf-8")
        registry = self.registry()
        runtime = registry.runtime(
            {
                "active_trial": {
                    "trial_id": 8,
                    "metrics_path": str(outside),
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
        trial_id = 4
        trial_dir = self.history_path.parent / "trials" / "0004"
        trial_dir.mkdir(parents=True)
        parameters = {
            **self.base,
            "actor_rollout_ref.rollout.disable_log_stats": False,
            "actor_rollout_ref.rollout.max_num_seqs": 256,
            "actor_rollout_ref.rollout.max_num_batched_tokens": 65536,
        }
        write_json(trial_dir / "parameters.json", parameters)
        vllm = {
            "available": True,
            "requests_running": {"mean": 220.0, "p95": 256.0, "max": 256.0},
            "requests_waiting": {"mean": 100.0, "p95": 500.0, "max": 581.0},
            "kv_cache_usage_pct": {"mean": 40.0, "p95": 50.0, "max": 55.0},
            "waiting_positive_fraction": 1.0,
            "preemptions_total": 0.0,
            "missing_metrics": ["iteration_tokens_p95_upper_bound"],
        }
        write_json(
            trial_dir / "metrics.json",
            {
                "throughput": {"vllm": vllm, "summary": {}, "phase_duration_s": {}},
                "stability": {"steps": [], "window_metrics": {}},
                "resource": {
                    "devices": [{"gpu_index": "0", "total_memory_mib": 65536}],
                    "by_phase": {
                        "rollout": {
                            "max_used_mib": 45875.2,
                            "max_used_gpu_total_mib": 65536,
                        }
                    },
                    "utilization_by_phase_pct": {"rollout": {"mean": 60.0}},
                    "summary": {},
                },
                "error": {"type": None},
            },
        )
        write_json(
            trial_dir / "trial_report.json",
            {
                "trial_id": trial_id,
                "result": "success",
                "rollout_engine": {
                    "monitor": {"enabled": True}
                },
            },
        )
        self.history_path.write_text(
            json.dumps(
                {
                    "trial_id": trial_id,
                    "result": "success",
                    "artifacts": trial_artifacts(trial_id),
                }
            )
            + "\n",
            encoding="utf-8",
        )
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
            "binding_increase_if_memory_feasible",
        )
        self.assertEqual(
            knobs["actor_rollout_ref.rollout.max_num_batched_tokens"]["status"],
            "unknown_metric_not_exported",
        )


if __name__ == "__main__":
    unittest.main()
