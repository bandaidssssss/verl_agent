from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent_tools import ToolRegistry
from config_utils import write_json
from summary_store import query_summaries, rebuild_summary_index


def summary(run_id: str, model: str, direction: str) -> dict:
    return {
        "run_context": {
            "run_id": run_id,
            "algorithm": "grpo",
            "model": model,
            "train_dataset": "math/train.parquet",
            "evaluation_dataset": "math/test.parquet",
            "platform": "C550",
            "workload": {},
        },
        "hardware": {
            "problems": [],
            "useful_directions": [{"direction": direction, "trial_ids": [2]}],
            "ineffective_directions": [],
        },
        "stability": {
            "problems": [],
            "useful_directions": [],
            "ineffective_directions": [],
        },
    }


class SummaryStoreTests(unittest.TestCase):
    def test_rebuilds_index_and_queries_prior_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "output"
            write_json(
                output_root / "0901_1200_2026/summer/summer_result.json",
                summary("0901_1200_2026", "Qwen3-8B", "increase rollout concurrency"),
            )
            write_json(
                output_root / "0902_1200_2026/summer/summer_result.json",
                summary("0902_1200_2026", "OtherModel", "increase actor batch size"),
            )

            index_path = rebuild_summary_index(output_root)
            result = query_summaries(
                output_root,
                stage="hardware",
                query="rollout",
                max_results=5,
                current_context={
                    "immutable_context": {
                        "model": {"model_path": "/models/Qwen3-8B"},
                        "hardware": {"platform": "C550"},
                        "workload": {"algorithm": "grpo"},
                    }
                },
                exclude_run_id="0903_1200_2026",
            )

            self.assertTrue(index_path.is_file())
            self.assertTrue(result["found"])
            self.assertEqual(len(result["results"]), 1)
            self.assertEqual(
                result["results"][0]["run_context"]["run_id"],
                "0901_1200_2026",
            )
            self.assertIn("must never be used", result["interpretation"])

    def test_proposal_tool_excludes_current_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "output"
            current = output_root / "0903_1200_2026"
            write_json(
                output_root / "0901_1200_2026/summer/summer_result.json",
                summary("0901_1200_2026", "Qwen3-8B", "increase rollout concurrency"),
            )
            write_json(
                current / "summer/summer_result.json",
                summary("0903_1200_2026", "Qwen3-8B", "current run direction"),
            )
            rebuild_summary_index(output_root)
            registry = ToolRegistry(
                Path(__file__).resolve().parents[1],
                {},
                current / "trials.jsonl",
            )

            result = registry.execute(
                "proposal",
                "query_tuning_summaries",
                {"stage": "hardware", "max_results": 5},
                registry.runtime({}),
            )

            self.assertEqual(len(result["results"]), 1)
            self.assertEqual(
                result["results"][0]["run_context"]["run_id"],
                "0901_1200_2026",
            )
            self.assertEqual(registry.definitions("summer"), [])

    def test_missing_index_returns_not_found_without_scanning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "output"
            write_json(
                output_root / "0901_1200_2026/summer/summer_result.json",
                summary("0901_1200_2026", "Qwen3-8B", "increase rollout concurrency"),
            )

            result = query_summaries(output_root, stage="hardware")

            self.assertEqual(
                result,
                {
                    "found": False,
                    "stage": "hardware",
                    "query": "",
                    "results": [],
                },
            )


if __name__ == "__main__":
    unittest.main()
