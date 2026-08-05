from __future__ import annotations

import csv
import tempfile
import threading
import unittest
from pathlib import Path

from vllm_metrics import (
    VLLMMetricsSampler,
    assess_rollout_metrics,
    compact_metric_row,
    metric_snapshot,
    summarize_vllm_metrics,
    vllm_metrics_enabled,
)


class VLLMMetricsTest(unittest.TestCase):
    def test_monitor_requires_vllm_and_explicit_false(self) -> None:
        base = {"actor_rollout_ref.rollout.name": "vllm"}
        self.assertFalse(vllm_metrics_enabled(base))
        self.assertFalse(
            vllm_metrics_enabled(
                {**base, "actor_rollout_ref.rollout.disable_log_stats": True}
            )
        )
        self.assertTrue(
            vllm_metrics_enabled(
                {**base, "actor_rollout_ref.rollout.disable_log_stats": False}
            )
        )
        self.assertFalse(
            vllm_metrics_enabled(
                {
                    "actor_rollout_ref.rollout.name": "sglang",
                    "actor_rollout_ref.rollout.disable_log_stats": False,
                }
            )
        )

    def test_parses_only_named_prometheus_metrics(self) -> None:
        data = """
# HELP vllm:num_requests_running Number of running requests.
vllm:num_requests_running{engine="0"} 256.0
vllm:num_requests_waiting{engine="0"} 581.0
vllm:kv_cache_usage_perc{engine="0"} 0.445
vllm:prompt_tokens_total{engine="0"} 1000
vllm:generation_tokens_total{engine="0"} 5000
vllm:cache_config_info{gpu_memory_utilization="0.8",num_gpu_blocks="100"} 1
""".strip()
        snapshot = metric_snapshot(data)
        self.assertEqual(snapshot["requests_running"], 256.0)
        self.assertEqual(snapshot["requests_waiting"], 581.0)
        self.assertAlmostEqual(snapshot["kv_cache_usage_pct"], 44.5)
        self.assertIsNone(snapshot["preemptions_total"])
        self.assertEqual(snapshot["iteration_tokens_buckets"], {})
        self.assertNotIn("gpu_memory_utilization", snapshot)

    def test_counter_deltas_and_iteration_histogram_are_compacted(self) -> None:
        previous = {
            "requests_running": 10.0,
            "requests_waiting": 2.0,
            "kv_cache_usage_pct": 40.0,
            "preemptions_total": 1.0,
            "preemptions_source": "prometheus",
            "prompt_tokens_total": 1000.0,
            "generation_tokens_total": 5000.0,
            "iteration_tokens_count": 0.0,
            "iteration_tokens_sum": 0.0,
            "iteration_tokens_buckets": {
                "2048.0": 0.0,
                "4096.0": 0.0,
                "+Inf": 0.0,
            },
        }
        current = {
            **previous,
            "requests_running": 256.0,
            "requests_waiting": 50.0,
            "kv_cache_usage_pct": 75.0,
            "preemptions_total": 3.0,
            "prompt_tokens_total": 1200.0,
            "generation_tokens_total": 7000.0,
            "iteration_tokens_count": 100.0,
            "iteration_tokens_sum": 300000.0,
            "iteration_tokens_buckets": {
                "2048.0": 20.0,
                "4096.0": 96.0,
                "+Inf": 100.0,
            },
        }
        row = compact_metric_row(current, previous, elapsed=10.0)
        self.assertEqual(row["preemptions_delta"], 2.0)
        self.assertEqual(row["prompt_tokens_per_s"], 20.0)
        self.assertEqual(row["generation_tokens_per_s"], 200.0)
        self.assertEqual(row["iteration_tokens_mean"], 3000.0)
        self.assertEqual(row["iteration_tokens_p95_upper_bound"], 4096.0)

        log_fallback = compact_metric_row(
            {
                **current,
                "preemptions_total": 4.0,
                "preemptions_source": "train_log",
            },
            {**previous, "preemptions_total": None},
            elapsed=10.0,
        )
        self.assertEqual(log_fallback["preemptions_delta"], 4.0)
        self.assertEqual(log_fallback["preemptions_source"], "train_log")

    def test_discovers_all_replica_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sampler = VLLMMetricsSampler(Path(directory) / "vllm.csv")
            sampler.observe_log_line(
                "vLLMHttpServer, replica_rank: 1, master address: 10.0.0.2, "
                "master port: 4567, data parallel master port: 5000"
            )
            sampler.observe_log_line(
                "vLLMHttpServer, replica_rank: 0, master address: 10.0.0.1, "
                "master port: 3456, data parallel master port: 5001"
            )
            sampler.observe_log_line(
                "AgentLoopManager: ['10.0.1.1:7001', '10.0.1.2:7002']"
            )
            endpoints = sampler.snapshot()["endpoints"]
        self.assertEqual(
            endpoints,
            [
                {"replica_rank": "0", "server": "10.0.1.1:7001"},
                {"replica_rank": "1", "server": "10.0.1.2:7002"},
            ],
        )

    def test_sampler_drops_idle_scrapes(self) -> None:
        idle = """
vllm:num_requests_running{engine="0"} 0
vllm:num_requests_waiting{engine="0"} 0
vllm:kv_cache_usage_perc{engine="0"} 0
vllm:prompt_tokens_total{engine="0"} 0
vllm:generation_tokens_total{engine="0"} 0
""".strip()
        active = """
vllm:num_requests_running{engine="0"} 16
vllm:num_requests_waiting{engine="0"} 8
vllm:kv_cache_usage_perc{engine="0"} 0.5
vllm:prompt_tokens_total{engine="0"} 100
vllm:generation_tokens_total{engine="0"} 500
""".strip()
        active_seen = threading.Event()
        calls = 0

        def fetch(_server: str, _timeout: float) -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                return idle
            active_seen.set()
            return active

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vllm_metrics.csv"
            sampler = VLLMMetricsSampler(path, interval=0.01, fetcher=fetch)
            sampler.observe_log_line(
                "AgentLoopManager: ['10.0.0.1:7001']"
            )
            sampler.start()
            self.assertTrue(active_seen.wait(timeout=1.0))
            sampler.stop()
            sampler.join(timeout=1.0)
            with path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["requests_running"], "16")

    def test_summary_and_binding_assessment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vllm_metrics.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "timestamp",
                        "replica_rank",
                        "server",
                        "requests_running",
                        "requests_waiting",
                        "kv_cache_usage_pct",
                        "preemptions_delta",
                        "preemptions_source",
                        "prompt_tokens_per_s",
                        "generation_tokens_per_s",
                        "iteration_tokens_mean",
                        "iteration_tokens_p95_upper_bound",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "timestamp": 1,
                        "replica_rank": 0,
                        "server": "10.0.0.1:1234",
                        "requests_running": 256,
                        "requests_waiting": 581,
                        "kv_cache_usage_pct": 44.5,
                        "preemptions_delta": 0,
                        "preemptions_source": "prometheus",
                        "prompt_tokens_per_s": 100,
                        "generation_tokens_per_s": 1000,
                        "iteration_tokens_mean": "",
                        "iteration_tokens_p95_upper_bound": "",
                    }
                )
            summary = summarize_vllm_metrics(path)

        assessment = assess_rollout_metrics(
            summary,
            {
                "actor_rollout_ref.rollout.gpu_memory_utilization": 0.8,
                "actor_rollout_ref.rollout.max_num_seqs": 256,
                "actor_rollout_ref.rollout.max_num_batched_tokens": 65536,
            },
            rollout_memory_peak_pct=70.0,
            rollout_gpu_util_mean_pct=60.0,
            memory_limit_pct=92.0,
        )
        self.assertEqual(
            assessment["knobs"]["actor_rollout_ref.rollout.max_num_seqs"]["status"],
            "binding_consider_increase_one_step",
        )
        self.assertEqual(
            assessment["knobs"]["actor_rollout_ref.rollout.max_num_batched_tokens"]["status"],
            "unknown_metric_not_exported",
        )
        self.assertEqual(
            assessment["knobs"]["actor_rollout_ref.rollout.gpu_memory_utilization"]["status"],
            "not_binding_do_not_raise_for_low_compute_utilization",
        )


if __name__ == "__main__":
    unittest.main()
