from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from config_utils import write_json
from trial_storage import build_trial_index, hydrate_trial, trial_artifacts


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
