from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from unittest import mock

from agents import AgentConversation, AgentResponseError, AgentRun
from config_utils import load_json
from orchestrator import (
    TuningOrchestrator,
    _feasibility_selection_violations,
    _normalize_proposal_changes,
    best_stability_trial,
    determine_stage,
    stability_healthy,
)


ROOT = Path(__file__).resolve().parents[1]


def hardware_trial(trial_id: int, throughput: float, result: str = "success") -> dict:
    return {
        "trial_id": trial_id,
        "stage": "hardware_tuning",
        "result": result,
        "performance": {"throughput": {"mean": throughput}},
    }


def stability_trial(trial_id: int, reward: float, kl: float = 0.02) -> dict:
    return {
        "trial_id": trial_id,
        "stage": "stability_tuning",
        "result": "success",
        "stability": {
            "window_size": 5,
            "windows": [{"start_step": 6, "end_step": 10, "sample_count": 5}],
            "metrics": {
                "critic/rewards/mean": [reward],
                "actor/ppo_kl": [kl],
            },
        },
    }


class OrchestratorStageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "min_hardware_trials": 2,
            "max_hardware_trials": 6,
            "plateau_rounds": 2,
            "min_stability_trials": 2,
            "max_stability_trials": 4,
        }

    def test_initial_and_failed_hardware_stage(self) -> None:
        self.assertEqual(determine_stage([], self.config), "hardware_tuning")
        failed = {"trial_id": 1, "stage": "hardware_tuning", "result": "fail"}
        self.assertEqual(determine_stage([failed], self.config), "hardware_repair")

    def test_direct_stability_mode_skips_hardware_trials(self) -> None:
        config = {**self.config, "start_stage": "stability_tuning"}
        self.assertEqual(determine_stage([], config), "stability_tuning")
        self.assertEqual(
            determine_stage([stability_trial(1, 0.1)], config),
            "stability_tuning",
        )

    def test_plateau_moves_to_stability(self) -> None:
        trials = [
            hardware_trial(1, 100),
            hardware_trial(2, 110),
            hardware_trial(3, 111),
            hardware_trial(4, 109),
            hardware_trial(5, 108),
        ]
        self.assertEqual(determine_stage(trials, self.config), "stability_tuning")

    def test_latest_hardware_best_resets_plateau_patience(self) -> None:
        trials = [
            hardware_trial(1, 100),
            hardware_trial(2, 110),
            hardware_trial(3, 109),
            hardware_trial(4, 108),
            hardware_trial(5, 111),
        ]
        self.assertEqual(determine_stage(trials, self.config), "hardware_tuning")

    def test_two_healthy_stability_trials_move_to_confirm(self) -> None:
        trials = [hardware_trial(index, 100 + index) for index in range(1, 7)]
        trials.extend([stability_trial(7, 0.1), stability_trial(8, 0.2)])
        self.assertEqual(determine_stage(trials, self.config), "confirm")

    def test_best_stability_uses_terminal_reward_mean(self) -> None:
        earlier = stability_trial(7, 0.2)
        earlier["stability"]["terminal_metrics"] = {
            "critic/rewards/mean": 0.25
        }
        later = stability_trial(8, 0.4)
        later["stability"]["windows"].append({"start_step": 11, "end_step": 15, "sample_count": 5})
        later["stability"]["metrics"]["critic/rewards/mean"].append(0.5)
        later["stability"]["metrics"]["actor/ppo_kl"].append(0.02)
        later["stability"]["terminal_metrics"] = {
            "critic/rewards/mean": 0.1
        }
        self.assertEqual(best_stability_trial([earlier, later])["trial_id"], 7)

    def test_best_stability_ignores_incomplete_terminal_window(self) -> None:
        earlier = stability_trial(7, 0.2)
        later = stability_trial(8, 0.4)
        later["stability"]["windows"].append(
            {"start_step": 11, "end_step": 12, "sample_count": 2}
        )
        later["stability"]["metrics"]["critic/rewards/mean"].append(0.1)
        later["stability"]["metrics"]["actor/ppo_kl"].append(0.02)
        self.assertEqual(best_stability_trial([earlier, later])["trial_id"], 8)

    def test_single_low_reward_is_not_treated_as_a_fixed_failure_floor(self) -> None:
        low_reward = stability_trial(7, -1.0, kl=0.01)
        low_reward["stability"]["terminal_metrics"] = {
            "critic/rewards/mean": -1.0
        }
        self.assertTrue(stability_healthy(low_reward, self.config))

    def test_persistent_window_decline_is_not_healthy(self) -> None:
        collapsed = stability_trial(7, 0.5, kl=0.01)
        for index, reward in enumerate([0.4, 0.2, -0.2, -0.5, -0.5], 2):
            collapsed["stability"]["windows"].append(
                {
                    "start_step": index * 5 + 1,
                    "end_step": (index + 1) * 5,
                    "sample_count": 5,
                }
            )
            collapsed["stability"]["metrics"]["critic/rewards/mean"].append(reward)
            collapsed["stability"]["metrics"]["actor/ppo_kl"].append(0.01)
        self.assertFalse(stability_healthy(collapsed, self.config))

    def test_sudden_kl_window_change_is_not_healthy(self) -> None:
        unstable = stability_trial(7, 0.2, kl=0.01)
        unstable["stability"]["windows"].append(
            {"start_step": 11, "end_step": 15, "sample_count": 5}
        )
        unstable["stability"]["metrics"]["critic/rewards/mean"].append(0.2)
        unstable["stability"]["metrics"]["actor/ppo_kl"].append(0.01)
        unstable["stability"]["metrics"]["actor/kl_loss"] = [0.01, 0.04]
        self.assertFalse(stability_healthy(unstable, self.config))

    def test_sudden_entropy_window_collapse_is_not_healthy(self) -> None:
        unstable = stability_trial(7, 0.2, kl=0.01)
        unstable["stability"]["windows"].append(
            {"start_step": 11, "end_step": 15, "sample_count": 5}
        )
        unstable["stability"]["metrics"]["critic/rewards/mean"].append(0.2)
        unstable["stability"]["metrics"]["actor/ppo_kl"].append(0.01)
        unstable["stability"]["metrics"]["actor/entropy"] = [0.5, 0.2]
        self.assertFalse(stability_healthy(unstable, self.config))


class DirectStabilityRunTest(unittest.TestCase):
    def test_first_direct_trial_uses_base_parameters_without_proposal(self) -> None:
        base = load_json(ROOT / "config" / "base_parameters.json")
        config = load_json(ROOT / "config" / "agent_config.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            config.update(
                {
                    "output_dir": temp_dir,
                    "start_stage": "stability_tuning",
                    "stream_agent_events": False,
                }
            )
            orchestrator = TuningOrchestrator(ROOT, base, config)
            with mock.patch("orchestrator.run_trial", return_value={}) as run_trial:
                reports = orchestrator.run(max_trials=1, dry_run=True)
        self.assertEqual(len(reports), 1)
        self.assertEqual(run_trial.call_args.args[0], base)
        self.assertEqual(run_trial.call_args.args[3], "stability_tuning")


class ConfirmCheckpointRunTest(unittest.TestCase):
    @staticmethod
    def _trial(trial_id: int, reward: float, parameters, checkpoint: Path) -> dict:
        trial = stability_trial(trial_id, reward)
        trial.update(
            {
                "parameters": parameters,
                "checkpoint": {
                    "global_step": 50,
                    "path": str(checkpoint),
                },
            }
        )
        return trial

    def test_confirm_resumes_the_best_stability_reference_checkpoint(self) -> None:
        base = load_json(ROOT / "config" / "base_parameters.json")
        config = load_json(ROOT / "config" / "agent_config.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            checkpoint_1 = output / "trials/0001/checkpoints/global_step_50"
            checkpoint_2 = output / "trials/0002/checkpoints/global_step_50"
            for checkpoint in (checkpoint_1, checkpoint_2):
                (checkpoint / "actor").mkdir(parents=True)
                (checkpoint / "data.pt").write_bytes(b"checkpoint")
            config.update(
                {
                    "output_dir": str(output),
                    "start_stage": "stability_tuning",
                    "min_stability_trials": 2,
                    "stream_agent_events": False,
                }
            )
            trials = [
                self._trial(1, 0.1, base, checkpoint_1),
                self._trial(2, 0.4, base, checkpoint_2),
            ]
            orchestrator = TuningOrchestrator(ROOT, base, config)
            orchestrator.trials = lambda: trials
            orchestrator.trial_indexes = lambda: trials
            with mock.patch("orchestrator.run_trial", return_value={}) as run_trial:
                reports = orchestrator.run(max_trials=1, dry_run=True)

        self.assertEqual(len(reports), 1)
        self.assertEqual(run_trial.call_args.args[3], "confirm")
        self.assertEqual(run_trial.call_args.args[4], 135)
        self.assertEqual(
            run_trial.call_args.kwargs["resume_checkpoint"],
            {
                "source_trial_id": 2,
                "global_step": 50,
                "path": str(checkpoint_2),
            },
        )
        self.assertEqual(reports[0]["proposal"]["reference_trial_id"], 2)

class ProposalProvenanceTest(unittest.TestCase):
    def test_structured_change_is_converted_to_executable_target(self) -> None:
        current = {"actor_rollout_ref.rollout.n": 3}
        reference = {"source": "trial", "trial_id": 7}
        proposal = {
            "decision": "modify",
            "reference_trial_id": 7,
            "reference_reason": "trial 7 is the selected stability baseline",
            "changes": {
                "actor_rollout_ref.rollout.n": {
                    "from": 3,
                    "to": 5,
                    "reason": "increase within-prompt comparisons",
                }
            },
            "expected_effect": {"reward_stability": "increase"},
        }
        targets, details, violations = _normalize_proposal_changes(
            proposal, current, reference
        )
        self.assertEqual(violations, [])
        self.assertEqual(targets, {"actor_rollout_ref.rollout.n": 5})
        self.assertEqual(details["actor_rollout_ref.rollout.n"]["from"], 3)
        self.assertEqual(details["actor_rollout_ref.rollout.n"]["to"], 5)

    def test_wrong_reference_and_from_value_are_rejected(self) -> None:
        proposal = {
            "decision": "modify",
            "reference_trial_id": 8,
            "reference_reason": "wrong trial",
            "changes": {
                "actor_rollout_ref.rollout.n": {
                    "from": 4,
                    "to": 5,
                    "reason": "test",
                }
            },
            "expected_effect": {"reward_stability": "increase"},
        }
        _, _, violations = _normalize_proposal_changes(
            proposal,
            {"actor_rollout_ref.rollout.n": 3},
            {"source": "trial", "trial_id": 7},
        )
        self.assertTrue(any("reference_trial_id" in row for row in violations))
        self.assertTrue(any("from must equal reference value" in row for row in violations))

    def test_null_from_adds_an_unset_override(self) -> None:
        proposal = {
            "decision": "modify",
            "reference_trial_id": 7,
            "reference_reason": "trial 7 is the current hardware reference",
            "changes": {
                "actor_rollout_ref.rollout.max_num_batched_tokens": {
                    "from": None,
                    "to": 8192,
                    "reason": "set an explicit scheduler token ceiling",
                }
            },
            "expected_effect": {"rollout_throughput": "increase"},
        }
        targets, details, violations = _normalize_proposal_changes(
            proposal,
            {"actor_rollout_ref.rollout.gpu_memory_utilization": 0.5},
            {"source": "trial", "trial_id": 7},
        )
        self.assertEqual(violations, [])
        self.assertEqual(
            targets,
            {"actor_rollout_ref.rollout.max_num_batched_tokens": 8192},
        )
        self.assertIsNone(
            details["actor_rollout_ref.rollout.max_num_batched_tokens"]["from"]
        )


class FeasibilitySelectionTest(unittest.TestCase):
    def test_selection_must_cover_and_choose_a_valid_canonical_candidate(self) -> None:
        candidate_ids = {"actor_batch", "rollout_memory"}
        valid_review = {
            "verdict": "valid",
            "selected_candidate_id": "rollout_memory",
            "candidate_reviews": [
                {"candidate_id": "actor_batch", "verdict": "invalid"},
                {"candidate_id": "rollout_memory", "verdict": "valid"},
            ],
        }
        self.assertEqual(
            _feasibility_selection_violations(valid_review, candidate_ids),
            [],
        )

        invalid_review = {
            "verdict": "valid",
            "selected_candidate_id": "invented_candidate",
            "candidate_reviews": [
                {"candidate_id": "rollout_memory", "verdict": "valid"},
            ],
        }
        violations = _feasibility_selection_violations(
            invalid_review, candidate_ids
        )
        self.assertTrue(
            any("selected_candidate_id" in row for row in violations)
        )
        self.assertTrue(
            any("cover exactly" in row for row in violations)
        )


class ProposalContextTest(unittest.TestCase):
    def test_hardware_proposal_receives_compact_reference_metrics(self) -> None:
        base = load_json(ROOT / "config" / "base_parameters.json")
        config = load_json(ROOT / "config" / "agent_config.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            config.update({"output_dir": temp_dir, "stream_agent_events": False})
            orchestrator = TuningOrchestrator(ROOT, base, config)

            class FakeAgents:
                def propose(self, context=None, conversation=None):
                    assert context is not None
                    assert "current_parameters" not in context
                    assert "reference_trial" not in context
                    assert context["default_reference"]["trial_id"] == 1
                    compact = context["compact_reference_history"][0]
                    assert compact["trial_id"] == 1
                    assert compact["metrics"]["summary"]["throughput"]["mean"] == 1.0
                    assert "max_used_gpu_index" not in str(compact)
                    conversation = AgentConversation("proposal", dict(context), [])
                    return AgentRun(
                        {
                            "decision": "stop",
                            "reason": "hardware stage has no further responsible experiment",
                            "candidates": [],
                        },
                        conversation,
                    )

                def diagnose(self, context):
                    raise AssertionError("successful trial should not be diagnosed")

            orchestrator.agents = FakeAgents()
            trials = [
                {
                    "trial_id": 1,
                    "stage": "hardware_tuning",
                    "result": "success",
                    "parameters": base,
                    "performance": {"throughput": {"mean": 1.0}},
                    "structured_metrics": {
                        "resource": {
                            "devices": [],
                            "by_phase": {
                                phase: {
                                    "mean_used_mib": 10,
                                    "p95_used_mib": 20,
                                    "max_used_mib": 30,
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
                                "throughput": {"mean": 1.0, "p95": 1.1, "max": 1.2},
                                "actor_mfu": {"mean": 0.1, "p95": 0.2, "max": 0.3},
                            }
                        },
                    },
                }
            ]
            _, proposal, _, _ = orchestrator._propose_candidate("hardware_tuning", base, trials)
            self.assertEqual(proposal["decision"], "stop")

    def test_first_stability_trial_runs_automatic_baseline_without_proposal(self) -> None:
        base = load_json(ROOT / "config" / "base_parameters.json")
        config = load_json(ROOT / "config" / "agent_config.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            config.update(
                {
                    "output_dir": temp_dir,
                    "stream_agent_events": False,
                    "min_hardware_trials": 1,
                    "max_hardware_trials": 1,
                }
            )
            orchestrator = TuningOrchestrator(ROOT, base, config)
            trials = [
                {
                    "trial_id": 1,
                    "stage": "hardware_tuning",
                    "result": "success",
                    "parameters": base,
                    "performance": {"throughput": {"mean": 1.0}},
                    "stability": {
                        "window_size": 5,
                        "windows": [{"start_step": 6, "end_step": 10, "sample_count": 5}],
                        "metrics": {"critic/rewards/mean": [0.1]},
                    },
                }
            ]
            orchestrator.trials = lambda: trials
            orchestrator.trial_indexes = lambda: trials

            class FakeAgents:
                def propose(self, context=None, conversation=None):
                    raise AssertionError("automatic stability baseline must skip Proposal")

                def diagnose(self, context):
                    raise AssertionError("successful trial should not be diagnosed")

            orchestrator.agents = FakeAgents()
            with mock.patch("orchestrator.run_trial", return_value={}) as run_trial:
                reports = orchestrator.run(max_trials=1, dry_run=True)
            self.assertEqual(len(reports), 1)
            self.assertEqual(run_trial.call_args.args[3], "stability_tuning")
            self.assertEqual(reports[0]["proposal"]["decision"], "baseline")
            self.assertEqual(reports[0]["proposal"]["source"], "orchestrator")


class ProposalDecisionTransitionTest(unittest.TestCase):
    def _hardware_trial(self, parameters):
        return {
            "trial_id": 1,
            "stage": "hardware_tuning",
            "result": "success",
            "parameters": parameters,
            "performance": {"throughput": {"mean": 1.0}},
        }

    def test_keep_is_rejected_then_stop_is_accepted_in_same_conversation(self) -> None:
        base = load_json(ROOT / "config" / "base_parameters.json")
        config = load_json(ROOT / "config" / "agent_config.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            config.update({"output_dir": temp_dir, "stream_agent_events": False})
            orchestrator = TuningOrchestrator(ROOT, base, config)
            trials = [self._hardware_trial(base)]

            class FakeAgents:
                def __init__(self):
                    self.calls = 0

                def propose(self, context=None, conversation=None):
                    self.calls += 1
                    active = conversation or AgentConversation("proposal", dict(context), [])
                    if self.calls == 1:
                        result = {"decision": "keep", "reason": "no candidate", "candidates": []}
                    else:
                        assert any(
                            "keep is not a valid Proposal decision" in row["content"]
                            for row in active.messages
                            if row.get("role") == "user"
                        )
                        result = {
                            "decision": "stop",
                            "reason": "hardware stage is complete",
                            "candidates": [],
                        }
                    return AgentRun(result, active)

                def diagnose(self, context):
                    raise AssertionError("successful trial should not be diagnosed")

            fake = FakeAgents()
            orchestrator.agents = fake
            _, proposal, _, trace = orchestrator._propose_candidate(
                "hardware_tuning", base, trials
            )
        self.assertEqual(fake.calls, 2)
        self.assertEqual(proposal["decision"], "stop")
        self.assertEqual(trace["rejections"][0]["proposal"]["decision"], "keep")

    def test_hardware_stop_runs_stability_baseline_and_preserves_trigger(self) -> None:
        base = load_json(ROOT / "config" / "base_parameters.json")
        config = load_json(ROOT / "config" / "agent_config.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            config.update({"output_dir": temp_dir, "stream_agent_events": False})
            orchestrator = TuningOrchestrator(ROOT, base, config)
            trials = [self._hardware_trial(base)]
            orchestrator.trials = lambda: trials
            orchestrator.trial_indexes = lambda: trials

            class StopAgents:
                def propose(self, context=None, conversation=None):
                    return AgentRun(
                        {
                            "decision": "stop",
                            "reason": "hardware search is complete",
                            "candidates": [],
                        },
                        AgentConversation("proposal", dict(context), []),
                    )

                def diagnose(self, context):
                    raise AssertionError("successful trial should not be diagnosed")

            orchestrator.agents = StopAgents()
            with mock.patch("orchestrator.run_trial", return_value={}) as run_trial:
                reports = orchestrator.run(max_trials=1, dry_run=True)
        self.assertEqual(run_trial.call_args.args[3], "stability_tuning")
        self.assertEqual(reports[0]["proposal"]["decision"], "baseline")
        self.assertEqual(
            reports[0]["proposal"]["transition_trigger"],
            {"decision": "stop", "reason": "hardware search is complete"},
        )

class RejectionConversationTest(unittest.TestCase):
    def test_feasibility_selects_valid_candidate_with_its_own_reference(self) -> None:
        base = load_json(ROOT / "config" / "base_parameters.json")
        alternate = {
            **base,
            "actor_rollout_ref.rollout.gpu_memory_utilization": 0.6,
        }
        config = load_json(ROOT / "config" / "agent_config.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            config.update(
                {
                    "output_dir": temp_dir,
                    "max_validation_rounds": 3,
                    "min_proposal_candidates": 2,
                    "max_proposal_candidates": 2,
                    "stream_agent_events": False,
                }
            )
            orchestrator = TuningOrchestrator(ROOT, base, config)

            class FakeAgents:
                def __init__(self) -> None:
                    self.calls = 0

                def propose(self, context=None, conversation=None):
                    self.calls += 1
                    if conversation is None:
                        conversation = AgentConversation("proposal", dict(context), [{"role": "user", "content": "start"}])
                    else:
                        assert any(
                            "Proposal Batch Attempt 1 Was Rejected" in row["content"]
                            for row in conversation.messages
                        )
                    if self.calls == 1:
                        candidates = [
                            {
                                "candidate_id": "bad_token_limit",
                                "reference_trial_id": 1,
                                "reference_reason": "trial 1 is the baseline",
                                "reason": "invalid token limit",
                                "changes": {
                                    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": {
                                        "from": base["actor_rollout_ref.actor.ppo_max_token_len_per_gpu"],
                                        "to": 0,
                                        "reason": "first attempt",
                                    }
                                },
                                "expected_effect": {"throughput": "increase"},
                            },
                            {
                                "candidate_id": "missing_reference",
                                "reference_trial_id": 99,
                                "reference_reason": "nonexistent trial",
                                "reason": "invalid reference",
                                "changes": {
                                    "actor_rollout_ref.rollout.gpu_memory_utilization": {
                                        "from": 0.6,
                                        "to": 0.65,
                                        "reason": "first attempt",
                                    }
                                },
                                "expected_effect": {"rollout_throughput": "increase"},
                            },
                        ]
                    else:
                        candidates = [
                            {
                                "candidate_id": "actor_tokens",
                                "reference_trial_id": 1,
                                "reference_reason": "trial 1 measured actor behavior",
                                "reason": "increase actor token batching",
                                "changes": {
                                    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": {
                                        "from": base["actor_rollout_ref.actor.ppo_max_token_len_per_gpu"],
                                        "to": base["actor_rollout_ref.actor.ppo_max_token_len_per_gpu"] * 2,
                                        "reason": "test actor throughput",
                                    }
                                },
                                "expected_effect": {"throughput": "increase"},
                            },
                            {
                                "candidate_id": "rollout_memory",
                                "reference_trial_id": 2,
                                "reference_reason": "trial 2 measured the alternate rollout baseline",
                                "reason": "increase rollout memory budget",
                                "changes": {
                                    "actor_rollout_ref.rollout.gpu_memory_utilization": {
                                        "from": 0.6,
                                        "to": 0.65,
                                        "reason": "test rollout concurrency",
                                    }
                                },
                                "expected_effect": {"rollout_throughput": "increase"},
                            },
                        ]
                    result = {
                        "decision": "modify",
                        "reason": f"attempt {self.calls}",
                        "candidates": candidates,
                    }
                    conversation.messages.append({"role": "assistant", "content": json.dumps(result)})
                    conversation.completed_turns += 1
                    return AgentRun(result, conversation)

                def review(self, context):
                    assert [row["candidate_id"] for row in context["candidates"]] == [
                        "actor_tokens",
                        "rollout_memory",
                    ]
                    assert context["candidates"][1]["reference_trial_id"] == 2
                    assert "candidate_parameters" not in context["candidates"][0]
                    assert "reference_trial" not in context["candidates"][0]
                    assert [
                        row["trial_id"]
                        for row in context["compact_reference_history"]
                    ] == [1, 2]
                    result = {
                        "verdict": "valid",
                        "selected_candidate_id": "rollout_memory",
                        "reason": "candidate 2 has the better evidence/risk trade-off",
                        "candidate_reviews": [
                            {"candidate_id": "actor_tokens", "verdict": "valid"},
                            {"candidate_id": "rollout_memory", "verdict": "valid"},
                        ],
                        "risks": [],
                    }
                    conversation = AgentConversation("feasibility", dict(context), [])
                    return AgentRun(result, conversation)

                def diagnose(self, context):
                    raise AssertionError("successful previous trial should not be diagnosed")

            fake = FakeAgents()
            orchestrator.agents = fake
            trials = [
                {
                    "trial_id": 1,
                    "stage": "hardware_tuning",
                    "result": "success",
                    "parameters": base,
                    "performance": {"throughput": {"mean": 1.0}},
                },
                {
                    "trial_id": 2,
                    "stage": "hardware_tuning",
                    "result": "success",
                    "parameters": alternate,
                    "performance": {"throughput": {"mean": 0.9}},
                },
            ]
            candidate, proposal, review, trace = orchestrator._propose_candidate(
                "hardware_tuning", base, trials
            )
            self.assertEqual(
                candidate["actor_rollout_ref.rollout.gpu_memory_utilization"],
                0.65,
            )
            self.assertEqual(
                candidate["actor_rollout_ref.actor.ppo_max_token_len_per_gpu"],
                alternate["actor_rollout_ref.actor.ppo_max_token_len_per_gpu"],
            )
            self.assertEqual(fake.calls, 2)
            self.assertEqual(proposal["candidate_id"], "rollout_memory")
            self.assertEqual(proposal["reference_trial_id"], 2)
            self.assertEqual(review["verdict"], "valid")
            self.assertEqual(review["selected_candidate_id"], "rollout_memory")
            self.assertEqual(trace["rejections"][0]["source"], "deterministic_validator")
            self.assertTrue(
                any(
                    "Proposal Batch Attempt 1 Was Rejected" in row["content"]
                    for row in trace["proposal_conversation"]["messages"]
                )
            )

    def test_exhausted_proposals_return_blocked_instead_of_raising(self) -> None:
        base = load_json(ROOT / "config" / "base_parameters.json")
        config = load_json(ROOT / "config" / "agent_config.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            config.update(
                {
                    "output_dir": temp_dir,
                    "max_validation_rounds": 2,
                    "stream_agent_events": False,
                }
            )
            orchestrator = TuningOrchestrator(ROOT, base, config)

            class InvalidAgents:
                def propose(self, context=None, conversation=None):
                    if conversation is None:
                        conversation = AgentConversation(
                            "proposal", dict(context), [{"role": "user", "content": "start"}]
                        )
                    result = {
                        "decision": "keep",
                        "reason": "no candidate",
                        "candidates": [],
                    }
                    conversation.messages.append(
                        {"role": "assistant", "content": json.dumps(result)}
                    )
                    return AgentRun(result, conversation)

                def review(self, context):
                    raise AssertionError("invalid proposal must not reach feasibility")

                def diagnose(self, context):
                    raise AssertionError("successful previous trial should not be diagnosed")

            orchestrator.agents = InvalidAgents()
            trials = [
                {
                    "trial_id": 1,
                    "stage": "hardware_tuning",
                    "result": "success",
                    "parameters": base,
                    "performance": {"throughput": {"mean": 1.0}},
                }
            ]
            _, proposal, review, trace = orchestrator._propose_candidate(
                "hardware_tuning", base, trials
            )
            self.assertEqual(proposal["decision"], "blocked")
            self.assertEqual(review["verdict"], "blocked")
            self.assertEqual(len(trace["rejections"]), 2)
            self.assertTrue((Path(temp_dir) / "last_agent_rejection.json").exists())

    def test_exhausted_response_repairs_write_blocked_state(self) -> None:
        base = load_json(ROOT / "config" / "base_parameters.json")
        config = load_json(ROOT / "config" / "agent_config.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            config.update(
                {
                    "output_dir": temp_dir,
                    "stream_agent_events": False,
                }
            )
            orchestrator = TuningOrchestrator(ROOT, base, config)
            trials = [
                {
                    "trial_id": 1,
                    "stage": "hardware_tuning",
                    "result": "success",
                    "parameters": base,
                    "performance": {"throughput": {"mean": 1.0}},
                }
            ]
            orchestrator.trials = lambda: trials

            class InvalidResponseAgents:
                def propose(self, context=None, conversation=None):
                    active = conversation or AgentConversation(
                        "proposal",
                        dict(context),
                        [{"role": "assistant", "content": "not json"}],
                    )
                    raise AgentResponseError(
                        "proposal",
                        "proposal response repair exhausted",
                        "not json",
                        2,
                        active,
                    )

                def diagnose(self, context):
                    raise AssertionError("successful previous trial should not be diagnosed")

            orchestrator.agents = InvalidResponseAgents()

            reports = orchestrator.run(max_trials=1)

            self.assertEqual(reports, [])
            state = load_json(Path(temp_dir) / "state.json")
            self.assertEqual(state["current_stage"], "agent_response_blocked")
            self.assertEqual(state["agent_role"], "proposal")
            self.assertTrue((Path(temp_dir) / "last_agent_error.json").exists())


if __name__ == "__main__":
    unittest.main()
