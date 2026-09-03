from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.replay_summer_prompt import (
    build_summer_context,
    resolve_run_dir,
    validate_summer_result,
)
from trial_storage import trial_artifacts


class SummerReplayTests(unittest.TestCase):
    def _write_trial(
        self,
        run_dir: Path,
        *,
        trial_id: int,
        stage: str,
        proposal: dict,
        metrics: dict,
        result: str = "success",
        resource: dict | None = None,
    ) -> dict:
        trial_dir = run_dir / "trials" / f"{trial_id:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        index = {
            "trial_id": trial_id,
            "stage": stage,
            "result": result,
            "updates_target": 10,
            "updates_completed": 10,
            "updates_executed": 10,
            "reference_trial_id": proposal.get("reference_trial_id"),
            "changes": proposal.get("changes", {}),
            "resource": resource or {"resource_safe": True},
            "artifacts": trial_artifacts(trial_id),
        }
        (trial_dir / "trial_report.json").write_text(
            json.dumps(index), encoding="utf-8"
        )
        (trial_dir / "metrics.json").write_text(
            json.dumps(metrics), encoding="utf-8"
        )
        (trial_dir / "decision.json").write_text(
            json.dumps({"proposal": proposal}), encoding="utf-8"
        )
        return index

    @staticmethod
    def _hardware_metrics(throughput: float) -> dict:
        return {
            "resource": {
                "by_phase": {
                    name: {
                        "mean_used_mib": 10,
                        "p95_used_mib": 11,
                        "max_used_mib": 12,
                    }
                    for name in (
                        "rollout",
                        "actor_log_prob",
                        "ref_log_prob",
                        "training",
                    )
                }
            },
            "throughput": {
                "summary": {
                    "throughput": {
                        "mean": throughput,
                        "p95": throughput,
                        "max": throughput,
                    },
                    "actor_mfu": {"mean": 0.2, "p95": 0.3, "max": 0.4},
                }
            },
        }

    @staticmethod
    def _stability_metrics(score: float) -> dict:
        names = (
            "critic/rewards/mean",
            "actor/ppo_kl",
            "actor/kl_loss",
            "actor/entropy",
            "actor/pg_loss",
            "actor/pg_clipfrac",
            "actor/lr",
        )
        return {
            "stability": {
                "step_range": {"start": 1, "end": 10},
                "windows": [{"start": 1, "end": 10}],
                "terminal_window": {"start": 6, "end": 10},
                "window_metrics": {name: [0.1] for name in names},
                "terminal_metrics": {name: 0.1 for name in names},
            },
            "evaluation": {
                "steps": {
                    "10": {"test_score/openai/gpt-oss-120b": score}
                }
            },
        }

    def _write_run(self, root: Path) -> Path:
        run_dir = root / "0903_1333_2026"
        run_dir.mkdir()
        rows = [
            self._write_trial(
                run_dir,
                trial_id=1,
                stage="hardware_tuning",
                proposal={"decision": "baseline", "source": "orchestrator"},
                metrics=self._hardware_metrics(1.0),
            ),
            self._write_trial(
                run_dir,
                trial_id=2,
                stage="hardware_tuning",
                proposal={
                    "decision": "modify",
                    "reference_trial_id": 1,
                    "reason": "remove a measured throughput bottleneck",
                    "expected_effect": {"summary.throughput.mean": "increase"},
                    "changes": {"rollout.n": {"from": 4, "to": 8}},
                },
                metrics=self._hardware_metrics(1.2),
            ),
            self._write_trial(
                run_dir,
                trial_id=3,
                stage="stability_tuning",
                proposal={"decision": "baseline", "source": "orchestrator"},
                metrics=self._stability_metrics(0.3),
            ),
            self._write_trial(
                run_dir,
                trial_id=4,
                stage="stability_tuning",
                proposal={
                    "decision": "modify",
                    "reference_trial_id": 3,
                    "reason": "reduce KL drift",
                    "expected_effect": {"evaluation": "increase"},
                    "changes": {"actor.kl_coef": {"from": 0.01, "to": 0.02}},
                },
                metrics=self._stability_metrics(0.4),
            ),
            self._write_trial(
                run_dir,
                trial_id=5,
                stage="confirm",
                proposal={
                    "decision": "modify",
                    "reference_trial_id": 4,
                    "reason": "confirmation only",
                    "changes": {"x": {"from": 1, "to": 2}},
                },
                metrics=self._stability_metrics(0.4),
            ),
        ]
        (run_dir / "trials.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return run_dir

    def test_context_contains_each_tuning_trial_once_with_stage_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._write_run(Path(directory))

            context = build_summer_context(run_dir)

            self.assertEqual([row["trial_id"] for row in context["trials"]], [1, 2, 3, 4])
            hardware = context["trials"][1]
            self.assertEqual(hardware["source"], "agent")
            self.assertEqual(hardware["reference_trial_id"], 1)
            self.assertEqual(hardware["changes"], {"rollout.n": {"from": 4, "to": 8}})
            self.assertEqual(
                hardware["agent_hypothesis"]["reason"],
                "remove a measured throughput bottleneck",
            )
            self.assertIn("summary", hardware["metrics"])
            self.assertNotIn("evaluation", hardware["metrics"])
            stability = context["trials"][3]
            self.assertIn("evaluation", stability["metrics"])
            self.assertIn("window_metrics", stability["metrics"])
            self.assertNotIn("summary", stability["metrics"])
            self.assertEqual(context["warnings"], [])

    def test_resolves_exact_or_unambiguous_date_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._write_run(root)

            self.assertEqual(resolve_run_dir(root, run_dir.name), run_dir.resolve())
            self.assertEqual(resolve_run_dir(root, "0903_1333"), run_dir.resolve())
            second = root / "0903_1333_2027"
            second.mkdir()
            (second / "trials.jsonl").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                resolve_run_dir(root, "0903_1333")

    def test_validation_rejects_reference_or_wrong_stage_citations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context = build_summer_context(self._write_run(Path(directory)))
            valid = {
                "hardware": {
                    "problems": [{"problem": "low throughput", "trial_ids": [2]}],
                    "useful_directions": [{"direction": "more concurrency", "trial_ids": [2]}],
                    "ineffective_directions": [],
                },
                "stability": {
                    "problems": [{"problem": "KL drift", "trial_ids": [4]}],
                    "useful_directions": [{"direction": "stronger regularization", "trial_ids": [4]}],
                    "ineffective_directions": [],
                },
            }
            self.assertEqual(validate_summer_result(valid, context), [])

            invalid = json.loads(json.dumps(valid))
            invalid["hardware"]["problems"][0]["trial_ids"] = [1, 4]
            violations = validate_summer_result(invalid, context)
            self.assertTrue(any("[1, 4]" in item for item in violations))


if __name__ == "__main__":
    unittest.main()
