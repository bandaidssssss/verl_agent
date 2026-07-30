from __future__ import annotations

import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from runner import GPUSampler, HealthAgentWorker, PhaseTracker, _resource_gate_enabled, build_command


class BuildCommandTest(unittest.TestCase):
    def test_stability_does_not_apply_the_preemptive_memory_gate(self) -> None:
        self.assertFalse(_resource_gate_enabled("stability_tuning"))
        self.assertTrue(_resource_gate_enabled("hardware_tuning"))
        self.assertTrue(_resource_gate_enabled("confirm"))

    def test_preserves_base_trainer_runtime_fields_in_stability_stage(self) -> None:
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

        with mock.patch.dict(os.environ, {}, clear=True):
            command, _ = build_command(
                parameters,
                agent_config,
                trial_id=3,
                updates=80,
                stage="stability_tuning",
            )

        self.assertIn("trainer.total_training_steps=80", command)
        self.assertIn("trainer.experiment_name=base-experiment", command)
        self.assertIn("trainer.save_freq=-1", command)
        self.assertIn("trainer.test_freq=7", command)
        self.assertIn("trainer.val_before_train=True", command)


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


if __name__ == "__main__":
    unittest.main()
