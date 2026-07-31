from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from metrics import analyze_trial, compute_threshold_stats, parse_step_records


class MetricsTest(unittest.TestCase):
    def test_parses_steps_and_phase_memory(self) -> None:
        text = """
Before generate_sequences, memory allocated (GB): 2.0, memory reserved (GB): 3.0, device memory used/total (GB): 32.0/64.0
After generate_sequences, memory allocated (GB): 2.0, memory reserved (GB): 3.0, device memory used/total (GB): 48.0/64.0
Before compute_log_prob, memory allocated (GB): 2.0, memory reserved (GB): 3.0, device memory used/total (GB): 50.0/64.0
After compute_log_prob, memory allocated (GB): 2.0, memory reserved (GB): 3.0, device memory used/total (GB): 58.0/64.0
step:1 - critic/rewards/mean:-0.2 - actor/ppo_kl:0.01 - actor/entropy:0.3 - actor/pg_loss:0.02 - actor/pg_clipfrac:0.1 - timing_s/gen:10 - timing_s/old_log_prob:4 - timing_s/ref:2 - timing_s/update_actor:8 - perf/time_per_step:24 - perf/total_num_tokens:1000 - perf/throughput:5
step:2 - critic/rewards/mean:0.2 - actor/ppo_kl:0.02 - actor/entropy:0.2 - actor/pg_loss:0.01 - actor/pg_clipfrac:0.2 - timing_s/gen:9 - timing_s/old_log_prob:4 - timing_s/ref:2 - timing_s/update_actor:7 - perf/time_per_step:22 - perf/total_num_tokens:1100 - perf/throughput:6
""".strip()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.log"
            path.write_text(text, encoding="utf-8")
            report = analyze_trial(path, None, warmup_updates=0)
        self.assertEqual(report["updates_completed"], 2)
        self.assertEqual(report["performance"]["time_bottleneck"], "rollout")
        self.assertAlmostEqual(report["memory_by_phase_pct"]["actor_log_prob"]["max"], 90.625)
        stability = report["stability"]
        self.assertEqual(stability["window_size"], 5)
        self.assertEqual(stability["windows"], [{"start_step": 1, "end_step": 2, "sample_count": 2}])
        self.assertAlmostEqual(stability["metrics"]["critic/rewards/mean"][0], 0.0)

    def test_stability_metrics_are_aligned_in_five_step_windows(self) -> None:
        text = "\n".join(
            f"step:{step} - critic/rewards/mean:{step / 10} - actor/ppo_kl:{step / 100} - "
            f"actor/pg_clipfrac:{step / 1000} - actor/entropy:{1 - step / 100} - "
            f"actor/lr:0.000003 - response_length/clip_ratio:{step / 1000}"
            for step in range(1, 11)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.log"
            path.write_text(text, encoding="utf-8")
            report = analyze_trial(path, None, warmup_updates=0, stability_window_size=5)
        stability = report["stability"]
        self.assertEqual(
            stability["windows"],
            [
                {"start_step": 1, "end_step": 5, "sample_count": 5},
                {"start_step": 6, "end_step": 10, "sample_count": 5},
            ],
        )
        self.assertAlmostEqual(stability["metrics"]["critic/rewards/mean"][0], 0.3)
        self.assertAlmostEqual(stability["metrics"]["critic/rewards/mean"][1], 0.8)
        self.assertEqual(
            stability["terminal_window"],
            {"start_step": 6, "end_step": 10, "sample_count": 5},
        )
        self.assertAlmostEqual(
            stability["terminal_metrics"]["critic/rewards/mean"], 0.8
        )

    def test_terminal_metrics_use_the_actual_final_five_updates(self) -> None:
        text = "\n".join(
            f"step:{step} - critic/rewards/mean:{step / 10}"
            for step in range(1, 8)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.log"
            path.write_text(text, encoding="utf-8")
            report = analyze_trial(
                path,
                None,
                warmup_updates=0,
                stability_window_size=5,
            )
        stability = report["stability"]
        self.assertEqual(
            stability["windows"],
            [
                {"start_step": 1, "end_step": 5, "sample_count": 5},
                {"start_step": 6, "end_step": 7, "sample_count": 2},
            ],
        )
        self.assertEqual(
            stability["terminal_window"],
            {"start_step": 3, "end_step": 7, "sample_count": 5},
        )
        self.assertAlmostEqual(
            stability["terminal_metrics"]["critic/rewards/mean"], 0.5
        )

    def test_threshold_stats(self) -> None:
        records = {
            1: {"critic/rewards/mean": -0.1, "perf/time_per_step": 2, "perf/total_num_tokens": 10},
            2: {"critic/rewards/mean": 0.1, "perf/time_per_step": 3, "perf/total_num_tokens": 20},
            3: {"critic/rewards/mean": 0.2, "perf/time_per_step": 4, "perf/total_num_tokens": 30},
        }
        result = compute_threshold_stats(records, [0.0, 0.1], window=1)
        self.assertEqual(result["0.0"]["step"], 2)
        self.assertEqual(result["0.1"]["cumulative_tokens"], 30)

    def test_parses_c550_rollout_offload_memory(self) -> None:
        text = """
Before rollout offload, memory allocated (GB): 27.68, memory reserved (GB): 33.73, device memory used/total (GB): 38.42/63.59
After rollout offload, memory allocated (GB): 27.68, memory reserved (GB): 27.78, device memory used/total (GB): 6.91/63.59
step:1 - critic/rewards/mean:0.1 - perf/time_per_step:2 - perf/total_num_tokens:10
""".strip()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.log"
            path.write_text(text, encoding="utf-8")
            report = analyze_trial(path, None, warmup_updates=0)
        self.assertAlmostEqual(report["memory_by_phase_pct"]["rollout"]["max"], 100.0 * 38.42 / 63.59)

    def test_normal_nccl_configuration_is_not_failure(self) -> None:
        text = """
ray init kwargs: {'env_vars': {'NCCL_CUMEM_ENABLE': '0'}}
config: {'nccl_timeout': 600}
step:1 - critic/rewards/mean:0.1 - perf/time_per_step:2 - perf/total_num_tokens:10
""".strip()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.log"
            path.write_text(text, encoding="utf-8")
            report = analyze_trial(path, None, warmup_updates=0)
        self.assertEqual(report["result"], "success")


if __name__ == "__main__":
    unittest.main()
