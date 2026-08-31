from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config_utils import write_json
from trial_storage import (
    build_trial_index,
    hydrate_trial,
    trial_artifacts,
)


class TrialStorageTest(unittest.TestCase):
    def test_index_is_small_and_hydrates_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            history = run_dir / "trials.jsonl"
            trial_dir = run_dir / "trials" / "0001"
            trial_dir.mkdir(parents=True)
            report = {
                "trial_id": 1,
                "stage": "hardware_tuning",
                "result": "success",
                "updates_target": 10,
                "updates_completed": 10,
                "parameters": {"x": 1},
                "performance": {"throughput": {"mean": 12.0}},
                "resource": {"max_used_mib": 50000},
                "proposal": {"reference_trial_id": None, "changes": {}},
                "error": {"type": None, "evidence": []},
                "agent_trace": {"messages": ["large"]},
                "health_events": [{"step": 1}],
                "health_decisions": [{"action": "continue"}],
                "stability": {"steps": [{"step": 1}]},
            }
            index = build_trial_index(report)
            write_json(trial_dir / "trial_report.json", {"trial_id": 1, "result": "success"})
            write_json(trial_dir / "parameters.json", {"x": 1})
            write_json(
                trial_dir / "metrics.json",
                {
                    "latest_step": 10,
                    "throughput": {"summary": {"throughput": {"mean": 12.0}}, "phase_duration_s": {}},
                    "stability": {"steps": [], "window_metrics": {}},
                    "resource": {"by_phase": {}, "summary": {"max_used_mib": 50000}},
                    "error": {"type": None, "evidence": []},
                },
            )
            write_json(trial_dir / "decision.json", {"proposal": report["proposal"]})
            write_json(
                trial_dir / "log_facts.json",
                {
                    "megatron": {"resolved_config": {"bf16": True}},
                },
            )
            hydrated = hydrate_trial(index, history)
        self.assertNotIn("parameters", index)
        self.assertNotIn("agent_trace", index)
        self.assertNotIn("health_events", index)
        self.assertNotIn("stability", index)
        self.assertEqual(index["artifacts"], trial_artifacts(1))
        self.assertEqual(hydrated["parameters"], {"x": 1})
        self.assertTrue(
            hydrated["log_facts"]["megatron"]["resolved_config"]["bf16"]
        )
        self.assertEqual(hydrated["performance"]["throughput"]["mean"], 12.0)

    def test_checkpoint_artifact_round_trips_as_a_safe_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            history = run_dir / "trials.jsonl"
            checkpoint = (
                run_dir
                / "trials"
                / "0003"
                / "checkpoints"
                / "global_step_50"
            )
            (checkpoint / "actor").mkdir(parents=True)
            (checkpoint / "data.pt").write_bytes(b"checkpoint")
            report = {
                "trial_id": 3,
                "stage": "stability_tuning",
                "result": "success",
                "updates_target": 50,
                "updates_completed": 50,
                "updates_executed": 50,
                "checkpoint": {"global_step": 50, "path": str(checkpoint)},
                "evaluation": {
                    "latest_metrics": {
                        "val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@1": 0.375,
                    }
                },
                "proposal": {"reference_trial_id": 2, "changes": {}},
            }
            index = build_trial_index(report, stability_healthy=True)
            hydrated = hydrate_trial(index, history)

        self.assertEqual(
            index["artifacts"]["checkpoint"],
            "trials/0003/checkpoints/global_step_50",
        )
        self.assertEqual(index["checkpoint"], {"global_step": 50})
        self.assertEqual(index["scores"]["evaluation_score"], 0.375)
        self.assertEqual(hydrated["checkpoint"]["path"], str(checkpoint.resolve()))

    def test_checkpoint_artifact_cannot_escape_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            history = Path(directory) / "trials.jsonl"
            index = build_trial_index(
                {
                    "trial_id": 1,
                    "stage": "stability_tuning",
                    "result": "success",
                    "checkpoint": {"global_step": 50},
                }
            )
            index["artifacts"]["checkpoint"] = "../outside/global_step_50"
            with self.assertRaisesRegex(ValueError, "outside the configured output"):
                hydrate_trial(index, history)
