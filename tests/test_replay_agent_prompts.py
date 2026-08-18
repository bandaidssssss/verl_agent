from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from replay_agent_prompts import load_prior_trials, load_replay_case
from trial_storage import trial_artifacts


class ReplayArtifactTests(unittest.TestCase):
    def _write_case(self, root: Path) -> Path:
        run_dir = root / "run"
        trial_one = run_dir / "trials" / "0001"
        trial_two = run_dir / "trials" / "0002"
        trial_one.mkdir(parents=True)
        trial_two.mkdir(parents=True)
        index = {
            "schema_version": 2,
            "trial_id": 1,
            "stage": "hardware_tuning",
            "result": "success",
            "changes": {},
            "artifacts": trial_artifacts(1),
        }
        (run_dir / "trials.jsonl").write_text(json.dumps(index) + "\n")
        (trial_one / "parameters.json").write_text(json.dumps({"x": 1}))
        (trial_one / "metrics.json").write_text(
            json.dumps(
                {
                    "throughput": {
                        "summary": {
                            "throughput": {"mean": 1, "p95": 2, "max": 3}
                        }
                    }
                }
            )
        )

        (trial_two / "trial_report.json").write_text(
            json.dumps({"trial_id": 2, "stage": "hardware_tuning"})
        )
        (trial_two / "parameters.json").write_text(json.dumps({"x": 2}))
        (trial_two / "metrics.json").write_text(json.dumps({"stability": {}}))
        (trial_two / "decision.json").write_text(
            json.dumps(
                {
                    "proposal": {"decision": "modify", "changes": {}},
                    "feasibility": {"verdict": "valid"},
                }
            )
        )
        (trial_two / "agent_trace.json").write_text(
            json.dumps(
                {
                    "proposal_conversation": {
                        "context": {"current_stage": "hardware_tuning"}
                    }
                }
            )
        )
        return trial_two

    def test_loads_target_and_hydrates_only_prior_trial_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target_dir = self._write_case(Path(directory))

            case = load_replay_case(target_dir)
            history = load_prior_trials(case)

            self.assertEqual(case.trial_id, 2)
            self.assertEqual(case.parameters, {"x": 2})
            self.assertEqual(case.decision["proposal"]["decision"], "modify")
            self.assertEqual([trial["trial_id"] for trial in history], [1])
            self.assertEqual(history[0]["parameters"], {"x": 1})
            self.assertEqual(
                history[0]["structured_metrics"]["throughput"]["summary"][
                    "throughput"
                ]["mean"],
                1,
            )

    def test_rejects_history_with_missing_prior_trial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target_dir = self._write_case(Path(directory))
            case = load_replay_case(target_dir)
            (case.run_dir / "trials.jsonl").write_text("")

            with self.assertRaisesRegex(ValueError, r"missing trial IDs \[1\]"):
                load_prior_trials(case)


if __name__ == "__main__":
    unittest.main()
