from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from metrics import (
    analyze_trial,
    build_running_metrics,
    build_structured_metrics,
    compute_threshold_stats,
)
from tools.extract_trial_metrics import extract_trial_metrics


class MetricsTest(unittest.TestCase):
    def test_resolved_hydra_config_is_flattened_without_cross_scope_matches(self) -> None:
        text = "\n".join(
            (
                "\x1b[32m2026-08-19 (TaskRunner pid=42) {'actor_rollout_ref': {\x1b[0m",
                "2026-08-19 (TaskRunner pid=42) 'actor': {'use_dynamic_bsz': False, "
                "'ppo_max_token_len_per_gpu': 16384},",
                "2026-08-19 (TaskRunner pid=42) 'rollout': {"
                "'log_prob_use_dynamic_bsz': False, "
                "'log_prob_max_token_len_per_gpu': 16384},",
                "2026-08-19 (TaskRunner pid=42) 'ref': {"
                "'log_prob_use_dynamic_bsz': False, "
                "'log_prob_max_token_len_per_gpu': 16384}},",
                "2026-08-19 (TaskRunner pid=42) 'critic': {"
                "'ppo_max_token_len_per_gpu': 32768}}",
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "train.log"
            log.write_text(text, encoding="utf-8")
            metrics = build_structured_metrics(log, None, parameters={})

        runtime = metrics["log_facts"]["runtime_parameters"]
        values = runtime["values"]
        self.assertTrue(runtime["available"])
        self.assertEqual(
            values["actor_rollout_ref.actor.ppo_max_token_len_per_gpu"],
            16384,
        )
        self.assertEqual(
            values[
                "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu"
            ],
            16384,
        )
        self.assertEqual(
            values["actor_rollout_ref.ref.log_prob_max_token_len_per_gpu"],
            16384,
        )
        self.assertEqual(values["critic.ppo_max_token_len_per_gpu"], 32768)

    def test_unified_extractor_reads_train_log_once_and_writes_log_facts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trial_dir = Path(directory)
            log = trial_dir / "train.log"
            log.write_text(
                "model config: {'model_type': 'llama', 'hidden_size': 64, "
                "'num_hidden_layers': 8, 'num_attention_heads': 8}\n"
                "TransformerConfig(tensor_model_parallel_size=1, "
                "pipeline_model_parallel_size=1, bf16=True)\n"
                "number of parameters on (tensor, pipeline) model parallel rank "
                "(0, 0): 300\n"
                "step:1 - perf/throughput:10 - perf/total_num_tokens:128\n",
                encoding="utf-8",
            )
            parameters = {
                "data.train_batch_size": 1,
                "data.max_prompt_length": 64,
                "data.max_response_length": 64,
                "actor_rollout_ref.rollout.n": 1,
            }
            original_open = Path.open
            train_log_opens = 0

            def tracked_open(path: Path, *args, **kwargs):
                nonlocal train_log_opens
                if path.resolve() == log.resolve():
                    train_log_opens += 1
                return original_open(path, *args, **kwargs)

            with mock.patch.object(Path, "open", tracked_open):
                metrics = extract_trial_metrics(
                    trial_dir, {}, parameters=parameters
                )
            facts = __import__("json").loads(
                (trial_dir / "log_facts.json").read_text(encoding="utf-8")
            )
        self.assertEqual(train_log_opens, 1)
        self.assertEqual(metrics["source"]["log_facts"], "log_facts.json")
        self.assertEqual(facts["model_config"]["hidden_size"], 64)
        self.assertEqual(
            facts["megatron"]["parameter_summary"]["total_parameters"], 300
        )

    def test_running_snapshot_never_exceeds_snapshot_step(self) -> None:
        metrics = build_running_metrics(
            {
                1: {"critic/rewards/mean": 0.1},
                2: {"critic/rewards/mean": 0.2},
                3: {"critic/rewards/mean": 0.3},
            },
            snapshot_step=2,
        )
        self.assertEqual(metrics["latest_step"], 2)
        self.assertEqual(
            [row["step"] for row in metrics["stability"]["steps"]], [1, 2]
        )

    def test_single_log_pass_extracts_log_facts(self) -> None:
        text = "\n".join(
            (
                "TF config: TransformerConfig(tensor_model_parallel_size=2, "
                "pipeline_model_parallel_size=1, sequence_parallel=True)",
                "number of parameters on (tensor, pipeline) model parallel rank (0, 0): 300",
                "number of parameters on (tensor, pipeline) model parallel rank (1, 0): 300",
                "step:6 - perf/total_num_tokens:640 - critic/rewards/mean:0.1",
            )
        )
        parameters = {
            "data.train_batch_size": 10,
            "data.max_prompt_length": 64,
            "data.max_response_length": 128,
            "actor_rollout_ref.rollout.n": 2,
        }
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "train.log"
            log.write_text(text, encoding="utf-8")
            metrics = build_structured_metrics(log, None, parameters=parameters)
        facts = metrics["log_facts"]
        self.assertEqual(facts["source"]["train_log"], "train.log")
        self.assertEqual(
            facts["megatron"]["resolved_config"]["tensor_model_parallel_size"], 2
        )
        self.assertEqual(
            facts["megatron"]["parameter_summary"]["total_parameters"], 600
        )
        self.assertEqual(
            facts["workload"]["sequence_length"]["point_tokens"], 32.0
        )

    def test_structured_metrics_classify_steps_and_absolute_memory(self) -> None:
        text = (
            "step:1 - critic/rewards/mean:0.2 - actor/entropy:0.3 - "
            "perf/throughput:12 - perf/time_per_step:4 - timing_s/gen:2"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "train.log"
            samples = root / "gpu_samples.csv"
            log.write_text(text, encoding="utf-8")
            samples.write_text(
                "timestamp,phase,gpu_index,memory_used_mb,memory_total_mb,utilization_pct\n"
                "1,rollout,0,50000,65536,80\n",
                encoding="utf-8",
            )
            metrics = build_structured_metrics(
                log,
                samples,
                warmup_updates=0,
                expected_gpu_count=1,
                resource_reserve_mib=3277,
                throughput_reserve_mib=6554,
            )
        self.assertEqual(metrics["throughput"]["steps"][0]["metrics"]["perf/throughput"], 12)
        self.assertEqual(metrics["stability"]["steps"][0]["metrics"]["actor/entropy"], 0.3)
        self.assertEqual(metrics["resource"]["by_phase"]["rollout"]["max_used_mib"], 50000)
        self.assertEqual(
            metrics["resource"]["policy"]["effective_resource_limit_mib_by_gpu"]["0"],
            65536 - 3277,
        )

    def test_extracts_math_test_score_without_overwriting_same_step_metrics(self) -> None:
        metric = "val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@1"
        text = "\n".join(
            (
                "step:5 - perf/throughput:12 - critic/rewards/mean:0.2",
                f"step:5 - {metric}:0.375",
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "train.log"
            log.write_text(text, encoding="utf-8")
            metrics = build_structured_metrics(log, None, warmup_updates=0)

        self.assertEqual(metrics["throughput"]["steps"][0]["metrics"]["perf/throughput"], 12)
        self.assertEqual(metrics["stability"]["steps"][0]["metrics"]["critic/rewards/mean"], 0.2)
        self.assertEqual(metrics["evaluation"]["latest_metrics"][metric], 0.375)
        self.assertEqual(
            metrics["evaluation"]["steps"],
            [{"step": 5, "metrics": {metric: 0.375}}],
        )

    def test_incomplete_monitor_coverage_is_never_marked_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "train.log"
            log.write_text("step:1 - perf/throughput:2 - critic/rewards/mean:0.1\n")
            samples = root / "gpu_samples.csv"
            samples.write_text(
                "timestamp,phase,gpu_index,memory_used_mb,memory_total_mb,utilization_pct\n"
                "1,rollout,0,800,1000,90\n"
            )
            metrics = build_structured_metrics(
                log,
                samples,
                expected_gpu_count=2,
                resource_reserve_mib=100,
                throughput_reserve_mib=200,
            )
        self.assertFalse(metrics["resource"]["monitor"]["coverage_complete"])
        self.assertIsNone(metrics["resource"]["summary"]["resource_safe"])
        self.assertIsNone(metrics["resource"]["summary"]["throughput_safe"])

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
