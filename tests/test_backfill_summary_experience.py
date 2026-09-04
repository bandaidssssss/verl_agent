from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents import AgentConversation, AgentRun
from config_utils import append_jsonl, load_json, write_json
from tools.backfill_summary_experience import backfill_run, discover_run_dirs
from trial_storage import trial_artifacts


def summary_result() -> dict:
    return {
        "hardware": {
            "problems": [],
            "useful_directions": [
                {"direction": "increase rollout concurrency", "trial_ids": [2]}
            ],
            "ineffective_directions": [],
        },
        "stability": {
            "problems": [],
            "useful_directions": [],
            "ineffective_directions": [],
        },
    }


class BackfillSummaryExperienceTests(unittest.TestCase):
    def test_discovers_only_recorded_run_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "output"
            append_jsonl(output_root / "0901/trials.jsonl", {"trial_id": 1})
            (output_root / "not-a-run").mkdir(parents=True)

            runs = discover_run_dirs(output_root)

            self.assertEqual([path.name for path in runs], ["0901"])

    def test_backfill_publishes_summary_and_merges_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "output/0901_1200_2026"
            append_jsonl(
                run_dir / "trials.jsonl",
                {"trial_id": 1, "artifacts": trial_artifacts(1)},
            )
            append_jsonl(
                run_dir / "trials.jsonl",
                {"trial_id": 2, "artifacts": trial_artifacts(2)},
            )
            trace_path = run_dir / "trials/0002/agent_trace.json"
            write_json(trace_path, {"proposal_conversation": {"messages": []}})
            context = {
                "run_context": {
                    "run_id": run_dir.name,
                    "algorithm": "grpo",
                    "model": "Qwen3-8B",
                },
                "trials": [],
            }
            conversation = AgentConversation("summary", {"trial": context}, [])
            fake_agents = mock.Mock()
            fake_agents.summarize.return_value = AgentRun(
                summary_result(), conversation
            )

            with mock.patch(
                "tools.backfill_summary_experience.build_summary_context",
                return_value=context,
            ), mock.patch(
                "tools.backfill_summary_experience.AgentSet",
                return_value=fake_agents,
            ):
                outcome = backfill_run(run_dir, {"stream_agent_events": False})

            result_path = run_dir / "summary/summary_result.json"
            self.assertEqual(outcome["status"], "created")
            self.assertEqual(
                load_json(result_path)["run_context"], context["run_context"]
            )
            trace = load_json(trace_path)
            self.assertIn("proposal_conversation", trace)
            self.assertEqual(trace["summary"]["result"], load_json(result_path))
            self.assertEqual(
                sorted(path.name for path in result_path.parent.iterdir()),
                ["summary_result.json"],
            )

    def test_existing_summary_is_skipped_without_calling_agent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "output/0901_1200_2026"
            result_path = run_dir / "summary/summary_result.json"
            write_json(result_path, {"existing": True})

            with mock.patch(
                "tools.backfill_summary_experience.AgentSet"
            ) as agent_set:
                outcome = backfill_run(run_dir, {})

            self.assertEqual(outcome["status"], "skipped")
            agent_set.assert_not_called()
            self.assertEqual(load_json(result_path), {"existing": True})


if __name__ == "__main__":
    unittest.main()
