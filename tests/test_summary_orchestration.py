from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents import AgentConversation, AgentError, AgentRun
from config_utils import load_json
from orchestrator import TuningOrchestrator


ROOT = Path(__file__).resolve().parents[1]


def summary_context(run_id: str, *, agent_trial: bool = True) -> dict:
    return {
        "run_id": run_id,
        "run_context": {
            "run_id": run_id,
            "algorithm": "grpo",
            "model": "Qwen3-8B",
            "train_dataset": "math/train.parquet",
            "evaluation_dataset": "math/test.parquet",
            "platform": "C550",
            "workload": {},
        },
        "stage_objectives": {},
        "trial_graph_rule": "resolve references by ID",
        "trials": [
            {
                "trial_id": 2 if agent_trial else 1,
                "stage_group": "hardware",
                "source": "agent" if agent_trial else "reference_only",
            }
        ],
        "warnings": [],
    }


def agent_result(trial_id: int = 2) -> dict:
    return {
        "hardware": {
            "problems": [],
            "useful_directions": [
                {"direction": "increase rollout concurrency", "trial_ids": [trial_id]}
            ],
            "ineffective_directions": [],
        },
        "stability": {
            "problems": [],
            "useful_directions": [],
            "ineffective_directions": [],
        },
    }


class SummaryOrchestrationTests(unittest.TestCase):
    def _orchestrator(self, directory: str) -> TuningOrchestrator:
        base = load_json(ROOT / "config/base_parameters.json")
        config = load_json(ROOT / "config/agent_config.json")
        config.update(
            {
                "output_dir": str(Path(directory) / "output/0903_1200_2026"),
                "stream_agent_events": False,
            }
        )
        return TuningOrchestrator(ROOT, base, config)

    def test_valid_summary_is_promoted_and_indexed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = self._orchestrator(directory)
            context = summary_context(orchestrator.output_dir.name)
            trace_path = orchestrator.output_dir / "trials/0002/agent_trace.json"
            trace_path.parent.mkdir(parents=True)
            trace_path.write_text(
                '{"proposal_conversation": {"messages": []}}\n', encoding="utf-8"
            )
            conversation = AgentConversation("summary", {"trial": context}, [])
            orchestrator.agents.summarize = mock.Mock(
                return_value=AgentRun(agent_result(), conversation)
            )

            with mock.patch(
                "orchestrator.build_summary_context", return_value=context
            ):
                result = orchestrator._summarize_run(2)

            self.assertEqual(result["run_context"], context["run_context"])
            self.assertTrue(
                (orchestrator.output_dir / "summary/summary_result.json").is_file()
            )
            self.assertTrue(
                (orchestrator.output_dir.parent / "summary_index.json").is_file()
            )
            trace = load_json(trace_path)
            self.assertIn("proposal_conversation", trace)
            self.assertEqual(trace["summary"]["result"], result)
            for name in (
                "summary_context.json",
                "summary_trace.json",
                "rendered_summary.md",
                "summary_status.json",
                "last_error.json",
            ):
                self.assertFalse((orchestrator.output_dir / "summary" / name).exists())

    def test_reference_only_trial_still_calls_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = self._orchestrator(directory)
            context = summary_context(
                orchestrator.output_dir.name, agent_trial=False
            )
            trace_path = orchestrator.output_dir / "trials/0001/agent_trace.json"
            trace_path.parent.mkdir(parents=True)
            trace_path.write_text("{}\n", encoding="utf-8")
            conversation = AgentConversation("summary", {"trial": context}, [])
            orchestrator.agents.summarize = mock.Mock(
                return_value=AgentRun(agent_result(1), conversation)
            )

            with mock.patch(
                "orchestrator.build_summary_context", return_value=context
            ):
                result = orchestrator._summarize_run(1)

            orchestrator.agents.summarize.assert_called_once_with({"trial": context})
            self.assertEqual(result["hardware"]["useful_directions"][0]["trial_ids"], [1])

    def test_summary_failure_does_not_persist_failure_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = self._orchestrator(directory)
            context = summary_context(orchestrator.output_dir.name)
            orchestrator.agents.summarize = mock.Mock(
                side_effect=AgentError("temporary failure")
            )

            with mock.patch(
                "orchestrator.build_summary_context", return_value=context
            ):
                result = orchestrator._summarize_run(2)

            self.assertIsNone(result)
            self.assertFalse((orchestrator.output_dir / "summary").exists())
            self.assertFalse(
                (orchestrator.output_dir / "trials/0002/agent_trace.json").exists()
            )

    def test_run_invokes_summary_once_after_the_trial_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = self._orchestrator(directory)
            orchestrator._summarize_run = mock.Mock()
            base = load_json(ROOT / "config/base_parameters.json")
            orchestrator.trials = mock.Mock(side_effect=[[], [{}]])
            orchestrator._starting_point = mock.Mock(
                return_value=(
                    base,
                    {
                        "source": "base_parameters",
                        "trial_id": None,
                        "selection_reason": "test baseline",
                    },
                )
            )

            def report_for_trial(
                parameters, config, trial_id, stage, updates, **kwargs
            ):
                return {
                    "trial_id": trial_id,
                    "stage": stage,
                    "result": "success",
                    "updates_target": updates,
                    "updates_completed": updates,
                    "updates_executed": updates,
                    "parameters": parameters,
                    "performance": {"throughput": {"mean": float(trial_id)}},
                    "resource": {},
                    "error": {},
                }

            with mock.patch(
                "orchestrator._runs_automatic_baseline", return_value=True
            ), mock.patch("orchestrator.run_trial", side_effect=report_for_trial):
                orchestrator.run(max_trials=2, dry_run=False)

            orchestrator._summarize_run.assert_called_once_with(2)

    def test_run_without_a_new_trial_does_not_invoke_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = self._orchestrator(directory)
            orchestrator._summarize_run = mock.Mock()

            orchestrator.run(max_trials=0, dry_run=False)

            orchestrator._summarize_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
