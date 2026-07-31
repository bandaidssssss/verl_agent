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
            "reward_collapse_slope": -0.01,
            "kl_warning": 0.1,
        }

    def test_initial_and_failed_hardware_stage(self) -> None:
        self.assertEqual(determine_stage([], self.config), "hardware_tuning")
        failed = {"trial_id": 1, "stage": "hardware_tuning", "result": "fail"}
        self.assertEqual(determine_stage([failed], self.config), "hardware_repair")

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

    def test_best_stability_legacy_fallback_uses_last_reported_window(self) -> None:
        earlier = stability_trial(7, 0.2)
        later = stability_trial(8, 0.4)
        later["stability"]["windows"].append(
            {"start_step": 11, "end_step": 12, "sample_count": 2}
        )
        later["stability"]["metrics"]["critic/rewards/mean"].append(0.1)
        later["stability"]["metrics"]["actor/ppo_kl"].append(0.02)
        self.assertEqual(best_stability_trial([earlier, later])["trial_id"], 7)


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


class ProposalSeriesContextTest(unittest.TestCase):
    def test_hardware_proposal_receives_reference_metric_windows_without_summary(self) -> None:
        base = load_json(ROOT / "config" / "base_parameters.json")
        config = load_json(ROOT / "config" / "agent_config.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            config.update({"output_dir": temp_dir, "stream_agent_events": False})
            orchestrator = TuningOrchestrator(ROOT, base, config)

            class FakeAgents:
                def propose(self, context=None, conversation=None):
                    assert context is not None
                    assert "stability" not in context["reference_trial"]
                    series = context["reference_stability_series"]
                    assert series["trial_id"] == 1
                    assert series["metrics"]["critic/rewards/mean"] == [0.1]
                    conversation = AgentConversation("proposal", dict(context), [])
                    return AgentRun({"decision": "keep", "reason": "test"}, conversation)

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
                    "stability": {
                        "window_size": 5,
                        "windows": [{"start_step": 6, "end_step": 10, "sample_count": 5}],
                        "metrics": {"critic/rewards/mean": [0.1]},
                    },
                }
            ]
            _, proposal, _, _ = orchestrator._propose_candidate("hardware_tuning", base, trials)
            self.assertEqual(proposal["decision"], "keep")

    def test_first_stability_trial_receives_hardware_series_and_keep_runs_baseline(self) -> None:
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

            class FakeAgents:
                def propose(self, context=None, conversation=None):
                    assert context["current_stage"] == "stability_tuning"
                    assert context["reference_stability_series"]["stage"] == "hardware_tuning"
                    return AgentRun(
                        {"decision": "keep", "reason": "hardware trajectory is the stability baseline"},
                        AgentConversation("proposal", dict(context), []),
                    )

                def diagnose(self, context):
                    raise AssertionError("successful trial should not be diagnosed")

            orchestrator.agents = FakeAgents()
            with mock.patch("orchestrator.run_trial", return_value={}) as run_trial:
                reports = orchestrator.run(max_trials=1, dry_run=True)
            self.assertEqual(len(reports), 1)
            self.assertEqual(run_trial.call_args.args[3], "stability_tuning")


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
                                "candidate_id": "bad_divisibility",
                                "reference_trial_id": 1,
                                "reference_reason": "trial 1 is the baseline",
                                "reason": "invalid micro batch",
                                "changes": {
                                    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": {
                                        "from": base["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"],
                                        "to": 3,
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
                                "candidate_id": "actor_batch",
                                "reference_trial_id": 1,
                                "reference_reason": "trial 1 measured actor behavior",
                                "reason": "increase actor micro batching",
                                "changes": {
                                    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": {
                                        "from": base["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"],
                                        "to": base["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"] * 2,
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
                        "actor_batch",
                        "rollout_memory",
                    ]
                    assert context["candidates"][1]["reference_trial_id"] == 2
                    result = {
                        "verdict": "valid",
                        "selected_candidate_id": "rollout_memory",
                        "reason": "candidate 2 has the better evidence/risk trade-off",
                        "candidate_reviews": [
                            {"candidate_id": "actor_batch", "verdict": "valid"},
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
                candidate["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"],
                alternate["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"],
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
                        "reference_trial_id": 1,
                        "changes": {
                            "actor_rollout_ref.rollout.gpu_memory_utilization": 0.7
                        },
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
