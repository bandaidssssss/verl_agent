from __future__ import annotations

import unittest
from pathlib import Path

from config_utils import load_json
from validator import IGNORED_PARAMETERS, parameter_groups, validate_candidate


class ValidatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.base = load_json(root / "config" / "base_parameters.json")
        cls.config = load_json(root / "config" / "agent_config.json")

    def test_valid_micro_batch_change(self) -> None:
        candidate = dict(self.base)
        candidate["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"] = 2
        result = validate_candidate(
            candidate,
            {"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 2},
            "hardware_tuning",
            self.config,
            self.base,
            [],
        )
        self.assertTrue(result.valid, result.violations)

    def test_dynamic_batching_does_not_require_inactive_micro_batches(self) -> None:
        candidate = dict(self.base)
        candidate.pop("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu", None)
        candidate.pop("actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu", None)
        candidate.pop("actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu", None)
        result = validate_candidate(
            candidate,
            {},
            "hardware_tuning",
            self.config,
            self.base,
            [],
        )
        self.assertTrue(result.valid, result.violations)

    def test_fixed_batching_requires_actor_micro_batch(self) -> None:
        candidate = {
            **self.base,
            "actor_rollout_ref.actor.use_dynamic_bsz": False,
        }
        candidate.pop("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu", None)
        result = validate_candidate(
            candidate,
            {"actor_rollout_ref.actor.use_dynamic_bsz": False},
            "hardware_tuning",
            self.config,
            self.base,
            [],
        )
        self.assertFalse(result.valid)
        self.assertTrue(
            any("ppo_micro_batch_size_per_gpu" in row for row in result.violations)
        )

    def test_dynamic_batching_requires_max_token_controls(self) -> None:
        candidate = dict(self.base)
        candidate.pop("actor_rollout_ref.ref.log_prob_max_token_len_per_gpu")
        result = validate_candidate(
            candidate,
            {},
            "hardware_tuning",
            self.config,
            self.base,
            [],
        )
        self.assertFalse(result.valid)
        self.assertTrue(
            any("ref.log_prob_max_token_len_per_gpu" in row for row in result.violations)
        )

    def test_rejects_stability_hardware_change(self) -> None:
        candidate = dict(self.base)
        candidate["actor_rollout_ref.rollout.max_num_seqs"] = 128
        result = validate_candidate(
            candidate,
            {"actor_rollout_ref.rollout.max_num_seqs": 128},
            "stability_tuning",
            self.config,
            self.base,
            [],
        )
        self.assertFalse(result.valid)

    def test_rejects_changed_hardware_token_budget(self) -> None:
        candidate = dict(self.base)
        candidate["actor_rollout_ref.rollout.n"] = 8
        result = validate_candidate(
            candidate,
            {"actor_rollout_ref.rollout.n": 8},
            "hardware_tuning",
            self.config,
            self.base,
            [],
        )
        self.assertFalse(result.valid)

    def test_parameter_groups_share_validator_policy(self) -> None:
        groups = parameter_groups(self.base, "stability_tuning")
        self.assertIn("actor_rollout_ref.actor.optim.lr", groups["stability_tuning"])
        self.assertIn(
            "actor_rollout_ref.actor.ppo_max_token_len_per_gpu",
            groups["throughput_tuning"],
        )
        self.assertIn(
            "actor_rollout_ref.actor.ppo_max_token_len_per_gpu",
            groups["fixed"],
        )
        self.assertNotIn("actor_rollout_ref.actor.optim.lr", groups["fixed"])
        self.assertEqual(set(groups["ignored"]), IGNORED_PARAMETERS & set(self.base))
        self.assertIn(
            "training_memory",
            groups["cross_effects"]["actor_rollout_ref.actor.entropy_coeff"],
        )

    def test_ref_topology_is_not_editable(self) -> None:
        key = "actor_rollout_ref.ref.megatron.tensor_model_parallel_size"
        candidate = {**self.base, key: 4}
        result = validate_candidate(
            candidate, {key: 4}, "hardware_tuning", self.config, self.base, []
        )
        self.assertFalse(result.valid)

    def test_ignored_ref_topology_does_not_make_a_configuration_unique(self) -> None:
        previous = dict(self.base)
        candidate = dict(previous)
        key = "actor_rollout_ref.ref.megatron.tensor_model_parallel_size"
        candidate[key] = int(previous[key]) * 2
        result = validate_candidate(
            candidate,
            {},
            "confirm",
            self.config,
            self.base,
            [{"trial_id": 3, "parameters": previous}],
        )
        self.assertFalse(result.valid)
        self.assertTrue(any("duplicates trial 3" in row for row in result.violations))


if __name__ == "__main__":
    unittest.main()
