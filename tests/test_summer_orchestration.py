from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents import AgentConversation, AgentError, AgentRun
from config_utils import load_json
from orchestrator import TuningOrchestrator


ROOT = Path(__file__).resolve().parents[1]


def summer_context(run_id: str, *, agent_trial: bool = True) -> dict:
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


class SummerOrchestrationTests(unittest.TestCase):
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
            context = summer_context(orchestrator.output_dir.name)
            trace_path = orchestrator.output_dir / "trials/0002/agent_trace.json"
            trace_path.parent.mkdir(parents=True)
            trace_path.write_text(
                '{"proposal_conversation": {"messages": []}}\n', encoding="utf-8"
            )
            conversation = AgentConversation("summer", {"trial": context}, [])
            orchestrator.agents.summarize = mock.Mock(
                return_value=AgentRun(agent_result(), conversation)
            )

            with mock.patch(
                "orchestrator.build_summer_context", return_value=context
            ):
                result = orchestrator._summarize_run(2)

            self.assertEqual(result["run_context"], context["run_context"])
            self.assertTrue(
                (orchestrator.output_dir / "summer/summer_result.json").is_file()
            )
            self.assertTrue(
                (orchestrator.output_dir.parent / "summer_index.json").is_file()
            )
            trace = load_json(trace_path)
            self.assertIn("proposal_conversation", trace)
            self.assertEqual(trace["summer"]["result"], result)
            for name in (
                "summer_context.json",
                "summer_trace.json",
                "rendered_summer.md",
                "summer_status.json",
                "last_error.json",
            ):
                self.assertFalse((orchestrator.output_dir / "summer" / name).exists())

    def test_reference_only_trial_still_calls_summer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = self._orchestrator(directory)
            context = summer_context(
                orchestrator.output_dir.name, agent_trial=False
            )
            trace_path = orchestrator.output_dir / "trials/0001/agent_trace.json"
            trace_path.parent.mkdir(parents=True)
            trace_path.write_text("{}\n", encoding="utf-8")
            conversation = AgentConversation("summer", {"trial": context}, [])
            orchestrator.agents.summarize = mock.Mock(
                return_value=AgentRun(agent_result(1), conversation)
            )

            with mock.patch(
                "orchestrator.build_summer_context", return_value=context
            ):
                result = orchestrator._summarize_run(1)

            orchestrator.agents.summarize.assert_called_once_with({"trial": context})
            self.assertEqual(result["hardware"]["useful_directions"][0]["trial_ids"], [1])

    def test_summer_failure_does_not_persist_failure_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = self._orchestrator(directory)
            context = summer_context(orchestrator.output_dir.name)
            orchestrator.agents.summarize = mock.Mock(
                side_effect=AgentError("temporary failure")
            )

            with mock.patch(
                "orchestrator.build_summer_context", return_value=context
            ):
                result = orchestrator._summarize_run(2)

            self.assertIsNone(result)
            self.assertFalse((orchestrator.output_dir / "summer").exists())
            self.assertFalse(
                (orchestrator.output_dir / "trials/0002/agent_trace.json").exists()
            )

    def test_run_invokes_summer_after_each_persisted_trial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = self._orchestrator(directory)
            orchestrator._summarize_run = mock.Mock()
            report = {
                "trial_id": 1,
                "stage": "hardware_tuning",
                "result": "success",
                "updates_target": 5,
                "updates_completed": 5,
                "updates_executed": 5,
                "parameters": load_json(ROOT / "config/base_parameters.json"),
                "performance": {"throughput": {"mean": 1.0}},
                "resource": {},
                "error": {},
            }

            with mock.patch("orchestrator.run_trial", return_value=report):
                orchestrator.run(max_trials=1, dry_run=False)

            orchestrator._summarize_run.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
