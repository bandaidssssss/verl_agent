from __future__ import annotations

import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from runner import (
    GPUSampler,
    HealthAgentWorker,
    HealthReviewSchedule,
    PhaseTracker,
    _checkpoint_validation_errors,
    _cleanup_unpublished_checkpoints,
    _finalize_stability_checkpoint,
    _resource_gate_enabled,
    build_command,
    run_trial,
)


class BuildCommandTest(unittest.TestCase):
    def test_resource_gate_applies_to_every_training_stage(self) -> None:
        self.assertTrue(_resource_gate_enabled("stability_tuning"))
        self.assertTrue(_resource_gate_enabled("hardware_tuning"))
        self.assertTrue(_resource_gate_enabled("confirm"))

    def test_evaluate_at_trial_end_uses_the_stage_update_target(self) -> None:
        parameters = {
            "trainer.total_epochs": 1,
            "trainer.test_freq": -1,
        }
        agent_config = {
            "verl_root": "/tmp/verl",
            "evaluate_at_trial_end": True,
        }

        command, _ = build_command(
            parameters,
            agent_config,
            trial_id=1,
            updates=50,
            stage="stability_tuning",
        )

        self.assertIn("trainer.test_freq=50", command)
        self.assertNotIn("trainer.test_freq=-1", command)

    def test_stability_uses_isolated_final_checkpoint(self) -> None:
        parameters = {
            "trainer.total_epochs": 2,
            "trainer.logger": ["console", "wandb"],
            "trainer.experiment_name": "base-experiment",
            "trainer.save_freq": -1,
            "trainer.test_freq": 7,
            "trainer.val_before_train": True,
        }
        agent_config = {
            "verl_root": "/tmp/verl",
            "config_path": "config",
            "config_name": "ppo_megatron_trainer.yaml",
            "environment_script": None,
        }

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {}, clear=True
        ):
            checkpoint_dir = Path(directory) / "checkpoints"
            command, _ = build_command(
                parameters,
                agent_config,
                trial_id=3,
                updates=80,
                stage="stability_tuning",
                checkpoint_dir=checkpoint_dir,
            )

        self.assertIn("trainer.total_training_steps=80", command)
        self.assertIn("trainer.experiment_name=base-experiment", command)
        self.assertIn("trainer.save_freq=80", command)
        self.assertIn(f"trainer.default_local_dir={checkpoint_dir.resolve()}", command)
        self.assertIn("trainer.resume_mode=disable", command)
        self.assertIn("trainer.resume_from_path=null", command)
        self.assertIn("trainer.max_actor_ckpt_to_keep=1", command)
        self.assertIn("trainer.max_critic_ckpt_to_keep=1", command)
        self.assertIn("trainer.test_freq=7", command)
        self.assertIn("trainer.val_before_train=True", command)

    def test_confirm_resumes_to_configured_global_target_without_saving(self) -> None:
        parameters = {
            "trainer.logger": ["console"],
            "trainer.save_freq": 5,
        }
        agent_config = {
            "verl_root": "/tmp/verl",
            "config_path": "config",
            "config_name": "ppo_megatron_trainer.yaml",
            "environment_script": None,
        }
        checkpoint = Path("/tmp/run/trials/0007/checkpoints/global_step_50")
        command, _ = build_command(
            parameters,
            agent_config,
            trial_id=9,
            updates=135,
            stage="confirm",
            checkpoint_dir="/tmp/run/trials/0009/checkpoints",
            resume_checkpoint={
                "source_trial_id": 7,
                "global_step": 50,
                "path": checkpoint,
            },
        )

        self.assertIn("trainer.total_training_steps=135", command)
        self.assertIn("trainer.save_freq=-1", command)
        self.assertIn("trainer.resume_mode=resume_path", command)
        self.assertIn(f"trainer.resume_from_path={checkpoint.resolve()}", command)
        self.assertIn("trainer.del_local_ckpt_after_load=False", command)

    def test_confirm_dry_run_reports_only_new_updates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            checkpoint = output / "trials" / "0007" / "checkpoints" / "global_step_50"
            (checkpoint / "actor").mkdir(parents=True)
            (checkpoint / "data.pt").write_bytes(b"checkpoint")
            config = {
                "output_dir": str(output),
                "verl_root": "/tmp/verl",
                "config_path": "config",
                "config_name": "ppo_megatron_trainer.yaml",
                "environment_script": None,
                "platform": "C550",
            }
            report = run_trial(
                {"trainer.logger": ["console"]},
                config,
                trial_id=9,
                stage="confirm",
                updates=135,
                dry_run=True,
                resume_checkpoint={
                    "source_trial_id": 7,
                    "global_step": 50,
                    "path": str(checkpoint),
                },
            )

        self.assertEqual(report["updates_target"], 135)
        self.assertEqual(report["resume"]["global_step"], 50)
        self.assertEqual(report["resume"]["updates_executed"], 85)

    def test_checkpoint_validation_and_failed_trial_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_root = Path(directory) / "checkpoints"
            checkpoint = checkpoint_root / "global_step_50"
            (checkpoint / "actor").mkdir(parents=True)
            self.assertEqual(
                _checkpoint_validation_errors(checkpoint),
                [f"dataloader checkpoint is missing: {checkpoint / 'data.pt'}"],
            )
            (checkpoint / "data.pt").write_bytes(b"checkpoint")
            self.assertEqual(_checkpoint_validation_errors(checkpoint), [])
            _cleanup_unpublished_checkpoints(checkpoint_root)
            self.assertFalse(checkpoint_root.exists())

    def test_only_successful_stability_checkpoint_is_published(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_root = Path(directory) / "success" / "checkpoints"
            checkpoint = checkpoint_root / "global_step_50"
            (checkpoint / "actor").mkdir(parents=True)
            (checkpoint / "data.pt").write_bytes(b"checkpoint")
            success = {"result": "success", "error": {}}
            _finalize_stability_checkpoint(checkpoint_root, 50, success)
            self.assertEqual(
                success["checkpoint"],
                {"global_step": 50, "path": str(checkpoint)},
            )
            self.assertTrue(checkpoint_root.exists())

            failed_root = Path(directory) / "failed" / "checkpoints"
            failed_checkpoint = failed_root / "global_step_50"
            (failed_checkpoint / "actor").mkdir(parents=True)
            (failed_checkpoint / "data.pt").write_bytes(b"checkpoint")
            failed = {"result": "fail", "error": {"type": "OOM"}}
            _finalize_stability_checkpoint(failed_root, 50, failed)
            self.assertNotIn("checkpoint", failed)
            self.assertFalse(failed_root.exists())

    def test_missing_final_checkpoint_turns_success_into_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_root = Path(directory) / "checkpoints"
            metrics = {"result": "success", "error": {}}
            _finalize_stability_checkpoint(checkpoint_root, 50, metrics)
        self.assertEqual(metrics["result"], "fail")
        self.assertEqual(metrics["error"]["type"], "CHECKPOINT_MISSING")


class GPUSamplerTest(unittest.TestCase):
    def test_phase_tracker_recognizes_c550_rollout_boundaries(self) -> None:
        tracker = PhaseTracker()
        tracker.update_from_log("DEBUG:After rollout init, device memory used/total (GB): 5.26/63.59")
        self.assertEqual(tracker.get(), "rollout")

        tracker.update_from_log("DEBUG:compute_log_prob Before compute_log_prob")
        self.assertEqual(tracker.get(), "actor_log_prob")

        tracker.update_from_log("DEBUG:update_actor After update_actor")
        self.assertEqual(tracker.get(), "rollout")

    def test_nvidia_csv_query(self) -> None:
        response = subprocess.CompletedProcess([], 0, "0, 1024, 8192, 75\n", "")
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"PLATFORM": "A100", "GPU_SMI": ""}
        ), mock.patch("runner.shutil.which", return_value="/usr/bin/nvidia-smi"), mock.patch(
            "runner.subprocess.run", return_value=response
        ):
            sampler = GPUSampler(Path(directory) / "gpu.csv", PhaseTracker(), 1.0, "A100")
            rows = sampler._query_rows()
        self.assertEqual(rows, [["0", "1024", "8192", "75"]])

    def test_v5000_human_table_fallback(self) -> None:
        query = subprocess.CompletedProcess([], 1, "", "unsupported query")
        table = subprocess.CompletedProcess([], 0, "0 Default a b c d e f 1024 x 2048 y 50\n", "")
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"PLATFORM": "V5000", "GPU_SMI": ""}
        ), mock.patch("runner.shutil.which", return_value="/usr/bin/xpu-smi"), mock.patch(
            "runner.subprocess.run", side_effect=[query, table]
        ):
            sampler = GPUSampler(Path(directory) / "gpu.csv", PhaseTracker(), 1.0, "V5000")
            rows = sampler._query_rows()
        self.assertEqual(rows, [["0", "1024", "2048", "50"]])


class HealthAgentWorkerTest(unittest.TestCase):
    def test_agent_failure_is_returned_without_raising_in_caller(self) -> None:
        def fail(_context):
            raise RuntimeError("agent unavailable")

        worker = HealthAgentWorker(fail)
        self.assertTrue(worker.submit("event-1", {"health_event": {}}))
        result = None
        for _ in range(100):
            result = worker.poll()
            if result is not None:
                break
            time.sleep(0.001)
        self.assertIsNotNone(result)
        self.assertFalse((result or {}).get("ok"))
        self.assertIn("agent unavailable", (result or {}).get("error", ""))


class HealthReviewScheduleTest(unittest.TestCase):
    def test_observation_is_due_without_a_new_rule_trigger(self) -> None:
        schedule = HealthReviewSchedule()
        schedule.observe(
            "event-step-12",
            origin_step=12,
            observe_for_updates=3,
            decision={"action": "observe", "observe_for_updates": 3},
        )
        self.assertFalse(schedule.due(14))
        self.assertTrue(schedule.due(15))
        self.assertEqual((schedule.snapshot() or {})["due_step"], 15)
        schedule.mark_submitted()
        self.assertFalse(schedule.due(16))

    def test_a_followup_observation_replaces_the_previous_deadline(self) -> None:
        schedule = HealthReviewSchedule()
        schedule.observe("event-1", 7, 5, {"action": "observe"})
        schedule.observe("event-2", 12, 3, {"action": "observe"})
        snapshot = schedule.snapshot() or {}
        self.assertEqual(snapshot["origin_event_id"], "event-2")
        self.assertEqual(snapshot["due_step"], 15)


if __name__ == "__main__":
    unittest.main()
