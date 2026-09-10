from __future__ import annotations

import os
import subprocess
import unittest
from unittest import mock

from distributed_launch import Topology, existing_address, prepare_cluster, ray_environment, topology, wait_for_head


class DistributedLaunchTest(unittest.TestCase):
    def test_platform_allocation_overrides_single_node_config(self):
        self.assertEqual(topology({"trainer.nnodes": 1}, {"POD_RANK": "1", "WORLD_SIZE": "4"}), Topology(4, 1, 8))

    def test_explicit_topology_and_slurm(self):
        self.assertEqual(topology({}, {"NNODES": "3", "NODE_RANK": "2", "GPUS_PER_NODE": "4"}), Topology(3, 2, 4))
        self.assertEqual(topology({}, {"SLURM_NNODES": "2", "SLURM_NODEID": "1"}), Topology(2, 1, 8))

    def test_plain_world_size_does_not_mean_node_count(self):
        self.assertEqual(topology({}, {"WORLD_SIZE": "16", "RANK": "0"}), Topology(1, 0, 8))

    def test_invalid_or_per_gpu_launch_rejected(self):
        for env in ({"NNODES": "2"}, {"NNODES": "2", "NODE_RANK": "2"}, {"LOCAL_WORLD_SIZE": "8"}, {"GPUS_PER_NODE": "0"}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                topology({}, env)

    def test_rank_variables_not_inherited_by_ray(self):
        self.assertEqual(ray_environment({"RANK": "1", "WORLD_SIZE": "2", "MASTER_ADDR": "head", "RAY_ADDRESS": "old", "MACA_PATH": "/opt/maca"}), {"MACA_PATH": "/opt/maca"})

    def test_dry_run_does_not_start_ray_or_probe_gpu(self):
        for rank in (0, 1):
            params = {}
            with mock.patch.dict(os.environ, {"POD_RANK": str(rank), "WORLD_SIZE": "2"}, clear=True), mock.patch("distributed_launch.subprocess.run") as run:
                self.assertEqual(prepare_cluster(params, dry_run=True), rank == 0)
                run.assert_not_called()
            self.assertEqual(params, {"trainer.nnodes": 2, "trainer.n_gpus_per_node": 8})

    def test_single_node_needs_no_ray(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch("distributed_launch.subprocess.run") as run:
            self.assertTrue(prepare_cluster({}))
            run.assert_not_called()

    def test_head_waits_before_driver_and_exports_gcs_address(self):
        env = {"POD_RANK": "0", "WORLD_SIZE": "2", "RANK": "0", "MASTER_ADDR": "head", "MASTER_PORT": "7788"}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("distributed_launch.socket.gethostbyname", return_value="10.0.0.1"), mock.patch("distributed_launch.subprocess.run", return_value=subprocess.CompletedProcess([], 0)) as run:
            self.assertTrue(prepare_cluster({}))
            commands = [call.args[0] for call in run.call_args_list]
            self.assertIn("--head", commands[1])
            self.assertIn("--port=6379", commands[1])
            self.assertEqual(commands[2][-3:], ["head:6379", "2", "8"])
            self.assertEqual(os.environ["RAY_ADDRESS"], "head:6379")
            self.assertNotIn("WORLD_SIZE", os.environ)
            self.assertNotIn("MASTER_PORT", run.call_args_list[1].kwargs["env"])

    def test_worker_only_joins_and_blocks(self):
        env = {"POD_RANK": "1", "WORLD_SIZE": "2", "MASTER_ADDR": "head"}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("distributed_launch.wait_for_head") as wait, mock.patch("distributed_launch.subprocess.run", return_value=subprocess.CompletedProcess([], 0)) as run:
            self.assertFalse(prepare_cluster({}))
            wait.assert_called_once_with("head", 6379, 300)
            self.assertIn("--block", run.call_args.args[0])
            self.assertIn("--address=head:6379", run.call_args.args[0])
            self.assertNotIn("--head", run.call_args.args[0])

    def test_wrong_device_count_stops_before_ray(self):
        with mock.patch.dict(os.environ, {"NNODES": "2", "NODE_RANK": "0", "MASTER_ADDR": "head"}, clear=True), mock.patch("distributed_launch.subprocess.run", return_value=subprocess.CompletedProcess([], 1)) as run:
            with self.assertRaisesRegex(RuntimeError, "Visible GPU count"):
                prepare_cluster({})
            self.assertEqual(run.call_count, 1)

    def test_missing_workers_prevent_driver_start(self):
        env = {"POD_RANK": "0", "WORLD_SIZE": "2", "MASTER_ADDR": "head"}
        results = [subprocess.CompletedProcess([], 0), subprocess.CompletedProcess([], 0), subprocess.TimeoutExpired("ready", 300)]
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("distributed_launch.socket.gethostbyname", return_value="10.0.0.1"), mock.patch("distributed_launch.subprocess.run", side_effect=results):
            with self.assertRaisesRegex(TimeoutError, "all Ray GPU nodes"):
                prepare_cluster({})

    def test_unreachable_head_has_timeout(self):
        with mock.patch("distributed_launch.socket.create_connection", side_effect=OSError), mock.patch("distributed_launch.time.monotonic", side_effect=[0, 10]):
            with self.assertRaises(TimeoutError):
                wait_for_head("head", 6379, 1)

    def test_existing_cluster_only_checks_resources_and_exports_address(self):
        env = {"RAY_CLUSTER_MODE": "existing", "RAY_ADDRESS": "10.200.111.147:6379"}
        parameters = {"trainer.nnodes": 2, "trainer.n_gpus_per_node": 8}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("distributed_launch.subprocess.run") as run:
            self.assertTrue(prepare_cluster(parameters))
            run.assert_called_once()
            self.assertEqual(run.call_args.args[0][-3:], [env["RAY_ADDRESS"], "2", "8"])
            self.assertNotEqual(run.call_args.args[0][0], "ray")
            self.assertEqual(os.environ["RAY_ADDRESS"], env["RAY_ADDRESS"])

    def test_existing_cluster_worker_does_nothing(self):
        with mock.patch.dict(os.environ, {"RAY_CLUSTER_MODE": "existing", "POD_RANK": "1", "WORLD_SIZE": "2"}, clear=True), mock.patch("distributed_launch.subprocess.run") as run:
            self.assertFalse(prepare_cluster({}))
            run.assert_not_called()

    def test_existing_cluster_resource_failure_blocks_training(self):
        with mock.patch.dict(os.environ, {"RAY_CLUSTER_MODE": "existing", "RAY_ADDRESS": "head:6379"}, clear=True), mock.patch("distributed_launch.subprocess.run", side_effect=subprocess.CalledProcessError(1, "check")):
            with self.assertRaises(subprocess.CalledProcessError):
                prepare_cluster({"trainer.nnodes": 2})

    def test_existing_cluster_rejects_local_address(self):
        with mock.patch.dict(os.environ, {"RAY_CLUSTER_MODE": "existing", "RAY_ADDRESS": "local"}, clear=True), mock.patch("distributed_launch.subprocess.run") as run:
            with self.assertRaises(ValueError):
                prepare_cluster({})
            run.assert_not_called()

    def test_discovery_uses_actual_address_instead_of_master(self):
        result = subprocess.CompletedProcess([], 0, stdout="some Ray output\nRAY_DISCOVERED_ADDRESS=10.1.2.3:7777\n")
        env = {"RAY_CLUSTER_MODE": "existing", "MASTER_ADDR": "old-head", "RAY_ADDRESS": "auto"}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("distributed_launch.subprocess.run", side_effect=[result, subprocess.CompletedProcess([], 0)]) as run:
            self.assertTrue(prepare_cluster({"trainer.nnodes": 2}))
            self.assertNotIn("RAY_ADDRESS", run.call_args_list[0].kwargs["env"])
            self.assertEqual(run.call_args_list[1].args[0][-3:], ["10.1.2.3:7777", "2", "8"])
            self.assertEqual(os.environ["RAY_ADDRESS"], "10.1.2.3:7777")

    def test_unset_address_discovers_and_explicit_address_wins(self):
        result = subprocess.CompletedProcess([], 0, stdout="RAY_DISCOVERED_ADDRESS=new:1234\n")
        with mock.patch("distributed_launch.subprocess.run", return_value=result) as run:
            self.assertEqual(existing_address({"MASTER_ADDR": "old"}), "new:1234")
            run.assert_called_once()
            self.assertEqual(existing_address({"RAY_ADDRESS": "manual:9999"}), "manual:9999")
            self.assertEqual(existing_address({"RAY_HEAD_ADDR": "manual", "RAY_PORT": "9998"}), "manual:9998")
            run.assert_called_once()

    def test_failed_discovery_does_not_start_or_verify_cluster(self):
        for error in (subprocess.CalledProcessError(1, "discover", stderr="No cluster"), subprocess.TimeoutExpired("discover", 30)):
            with self.subTest(error=error), mock.patch.dict(os.environ, {"RAY_CLUSTER_MODE": "existing"}, clear=True), mock.patch("distributed_launch.subprocess.run", side_effect=error) as run:
                with self.assertRaisesRegex(RuntimeError, "Could not discover"):
                    prepare_cluster({})
                run.assert_called_once()


    def test_platform_fallback_after_missing_local_ray(self):
        results = [subprocess.CalledProcessError(1, "discover", stderr="No local Ray"),
                   subprocess.CompletedProcess([], 0, stdout="RAY_DISCOVERED_ADDRESS=10.2.3.4:6380\n")]
        with mock.patch("distributed_launch.subprocess.run", side_effect=results) as run:
            self.assertEqual(existing_address({"MASTER_ADDR": "current-head", "RAY_PORT": "6380", "MASTER_PORT": "7788"}), "10.2.3.4:6380")
            self.assertEqual([c.args[0][-1] for c in run.call_args_list], ["auto", "current-head:6380"])

    def test_all_discovery_candidates_fail_with_actionable_error(self):
        with mock.patch("distributed_launch.subprocess.run", side_effect=subprocess.TimeoutExpired("probe", 30)) as run:
            with self.assertRaisesRegex(RuntimeError, "Tried: auto, head:6379"):
                existing_address({"MASTER_ADDR": "head"})
            self.assertEqual(run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
