from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


from agent_tools.memory_estimator_V3 import (
    ACTOR_LOG_PROB_DYNAMIC_KEY,
    ACTOR_LOG_PROB_MAX_TOKENS_KEY,
    ACTOR_CP_KEY,
    ACTOR_OPTIMIZER_OFFLOAD_KEY,
    ACTOR_SP_KEY,
    ACTOR_TP_KEY,
    LORA_RANK_KEY,
    REF_LOG_PROB_DYNAMIC_KEY,
    REF_LOG_PROB_MAX_TOKENS_KEY,
    REF_PARAM_OFFLOAD_KEY,
    REF_SP_KEY,
    REF_TP_KEY,
    ROLLOUT_N_KEY,
    _activation_bytes,
    _component_dependencies,
    _extract_log_context,
    _phase_residency_components,
    _parameter_footprint,
    _runtime_args,
)
from tools.extract_log_facts import LogFactsAccumulator


class MemoryEstimatorV3Tests(unittest.TestCase):
    def test_v3_consumes_log_facts_without_log_parsing(self) -> None:
        context = _extract_log_context(
            {
                "schema_version": 2,
                "log_path": "/path/that/must/not/be/read/train.log",
                "log_facts": {
                    "schema_version": 1,
                    "source": {
                        "train_log": "train.log",
                        "parser_version": 1,
                        "warnings": [],
                    },
                    "model_config": {"hidden_size": 64},
                    "megatron": {
                        "resolved_config": {"bf16": True},
                        "rank_parameter_counts": [],
                        "parameter_summary": {"total_parameters": 100},
                    },
                    "workload": {
                        "sequence_length": {
                            "point_tokens": 32,
                            "upper_tokens": 64,
                        }
                    },
                },
            },
            self._parameters(),
        )
        self.assertEqual(context["model_config"]["hidden_size"], 64)
        self.assertEqual(context["resolved"]["bf16"], True)
        self.assertEqual(context["parameter_profile"]["total_parameters"], 100)

    def _architecture(self, **updates):
        architecture = {
            "hidden_size": 64.0,
            "num_layers": 8,
            "num_attention_heads": 8,
            "ffn_hidden_size": 192.0,
            "padded_vocab_size": 1024,
            "kv_channels": 8.0,
            "num_query_groups": 2,
            "group_query_attention": True,
            "swiglu": True,
            "untie_embeddings_and_output_weights": False,
            "num_experts": None,
            "moe_ffn_hidden_size": None,
            "moe_shared_expert_intermediate_size": 0.0,
            "moe_layer_pattern": [0] * 8,
            "multi_latent_attention": False,
            "mtp_num_layers": 0,
            "bytes_per_weight": 2,
            "activation_dtype_bytes": 2,
            "parameter_profile": {},
        }
        architecture.update(updates)
        return architecture

    def _parameters(self, **updates):
        parameters = {
            "trainer.n_gpus_per_node": 8,
            "trainer.nnodes": 1,
            "data.train_batch_size": 64,
            "data.max_prompt_length": 64,
            "data.max_response_length": 64,
            ROLLOUT_N_KEY: 1,
            ACTOR_TP_KEY: 2,
            "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size": 2,
            "actor_rollout_ref.actor.megatron.context_parallel_size": 1,
            "actor_rollout_ref.actor.megatron.expert_model_parallel_size": 1,
            "actor_rollout_ref.actor.megatron.expert_tensor_parallel_size": 2,
            ACTOR_SP_KEY: True,
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 2,
            "actor_rollout_ref.actor.ppo_mini_batch_size": 16,
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 2,
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 2,
        }
        parameters.update(updates)
        return parameters

    def _runtime(self, phase, parameters=None):
        return _runtime_args(
            phase,
            parameters or self._parameters(),
            {"resolved": {}},
            {"point_tokens": 128, "upper_tokens": 192},
        )

    def test_logged_parameter_count_reconstructs_unique_tp_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "train.log"
            log_path.write_text(
                "\n".join(
                    (
                        "TF config: TransformerConfig("
                        "tensor_model_parallel_size=2, "
                        "pipeline_model_parallel_size=1, "
                        "expert_model_parallel_size=1, "
                        "expert_tensor_parallel_size=2, "
                        "sequence_parallel=True)",
                        " > number of parameters on (tensor, pipeline) model parallel "
                        "rank (0, 0): 300",
                        " > number of parameters on (tensor, pipeline) model parallel "
                        "rank (1, 0): 300 [repeated 3x across cluster]",
                        # A second actor/ref initialization must not double-count.
                        " > number of parameters on (tensor, pipeline) model parallel "
                        "rank (0, 0): 300",
                    )
                ),
                encoding="utf-8",
            )
            accumulator = LogFactsAccumulator(
                {"data.max_prompt_length": 4, "data.max_response_length": 8},
                log_path,
            )
            for line in log_path.read_text(encoding="utf-8").splitlines(True):
                accumulator.consume(line)
            facts = accumulator.finalize({})

        profile = facts["megatron"]["parameter_summary"]
        self.assertEqual(profile["matched_log_lines"], 3)
        self.assertEqual(profile["observed_tp_pp_shard_count"], 2)
        self.assertTrue(profile["complete_tp_pp_coverage"])
        self.assertEqual(profile["most_loaded_shard_parameters"], 300)
        self.assertEqual(profile["total_parameters"], 600)

    def test_rank_parameter_conflict_is_explicit(self) -> None:
        accumulator = LogFactsAccumulator(
            {"data.max_prompt_length": 4, "data.max_response_length": 8},
            "train.log",
        )
        for line in (
            "TransformerConfig(tensor_model_parallel_size=1, "
            "pipeline_model_parallel_size=1)\n",
            "number of parameters on (tensor, pipeline) model parallel rank "
            "(0, 0): 300\n",
            "number of parameters on (tensor, pipeline) model parallel rank "
            "(0, 0): 301\n",
        ):
            accumulator.consume(line)
        facts = accumulator.finalize({})
        summary = facts["megatron"]["parameter_summary"]
        warnings = facts["source"]["warnings"]
        self.assertFalse(summary["complete_tp_pp_coverage"])
        self.assertIsNone(summary["total_parameters"])
        self.assertTrue(any("conflicting" in warning for warning in warnings))
        self.assertEqual(
            facts["megatron"]["rank_parameter_counts"][0]["conflicting_values"],
            [300, 301],
        )

    def test_logged_fixed_parameters_are_reused_then_resharded(self) -> None:
        architecture = {
            "hidden_size": 8.0,
            "num_layers": 2,
            "num_attention_heads": 2,
            "ffn_hidden_size": 16.0,
            "padded_vocab_size": 32,
            "kv_channels": 4.0,
            "num_query_groups": 2,
            "group_query_attention": True,
            "swiglu": True,
            "untie_embeddings_and_output_weights": False,
            "moe_layer_pattern": [0, 0],
            "multi_latent_attention": False,
            "mtp_num_layers": 0,
            "parameter_profile": {
                "most_loaded_shard_parameters": 300,
                "total_parameters": 600,
                "reference_topology": {
                    "tensor_model_parallel_size": 2,
                    "pipeline_model_parallel_size": 1,
                    "expert_model_parallel_size": 1,
                    "expert_tensor_parallel_size": 2,
                },
            },
        }
        reference_runtime = {
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "expert_tensor_parallel_size": 2,
        }
        candidate_runtime = {
            **reference_runtime,
            "tensor_model_parallel_size": 1,
            "expert_tensor_parallel_size": 1,
        }

        reference = _parameter_footprint(architecture, reference_runtime)
        candidate = _parameter_footprint(architecture, candidate_runtime)

        self.assertEqual(reference["most_loaded_shard_parameters"], 300)
        self.assertEqual(reference["total_parameters"], 600)
        self.assertEqual(
            reference["parameter_source"],
            "reference_train_log_exact_shard_unchanged_topology",
        )
        self.assertEqual(candidate["most_loaded_shard_parameters"], 600)
        self.assertEqual(candidate["total_parameters"], 600)
        self.assertEqual(
            candidate["parameter_source"],
            "reference_train_log_total_resharded_for_candidate_topology",
        )

    def test_ref_runtime_uses_actor_megatron_topology(self) -> None:
        parameters = {
            "trainer.n_gpus_per_node": 8,
            "trainer.nnodes": 1,
            "data.train_batch_size": 256,
            "actor_rollout_ref.rollout.n": 4,
            ACTOR_TP_KEY: 1,
            "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size": 1,
            ACTOR_SP_KEY: True,
            REF_TP_KEY: 2,
            "actor_rollout_ref.ref.megatron.pipeline_model_parallel_size": 1,
            REF_SP_KEY: True,
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 16,
            "actor_rollout_ref.ref.megatron.param_offload": True,
        }
        runtime = _runtime_args(
            "ref_log_prob",
            parameters,
            {"resolved": {}},
            {"point_tokens": 128, "upper_tokens": 256},
        )

        self.assertEqual(runtime["micro_batch_size"], 16)
        self.assertEqual(runtime["tensor_model_parallel_size"], 1)
        self.assertEqual(runtime["expert_tensor_parallel_size"], 1)
        self.assertFalse(runtime["sequence_parallel"])
        self.assertTrue(runtime["param_offload"])
        self.assertEqual(
            runtime["sources"]["tensor_model_parallel_size"],
            f"parameters:{ACTOR_TP_KEY}",
        )
        self.assertEqual(
            runtime["sources"]["sequence_parallel"],
            "framework_forced_false_at_tp1",
        )

    def test_ref_dependencies_ignore_ref_topology_knobs(self) -> None:
        dependencies = _component_dependencies("ref_log_prob")["all"]
        self.assertIn(ACTOR_TP_KEY, dependencies)
        self.assertIn(ACTOR_SP_KEY, dependencies)
        self.assertNotIn(REF_TP_KEY, dependencies)
        self.assertNotIn(REF_SP_KEY, dependencies)

    def test_log_prob_dependencies_include_real_dynamic_cap_and_cp_keys(self) -> None:
        actor = _component_dependencies("actor_log_prob")
        ref = _component_dependencies("ref_log_prob")

        self.assertIn(ACTOR_CP_KEY, actor["activation"])
        self.assertIn(ACTOR_LOG_PROB_DYNAMIC_KEY, actor["activation"])
        self.assertIn(ACTOR_LOG_PROB_MAX_TOKENS_KEY, actor["activation"])
        self.assertIn(ACTOR_CP_KEY, ref["activation"])
        self.assertIn(REF_LOG_PROB_DYNAMIC_KEY, ref["activation"])
        self.assertIn(REF_LOG_PROB_MAX_TOKENS_KEY, ref["activation"])

    def test_actor_dynamic_batch_uses_rollout_keys_and_cp_rank_cap(self) -> None:
        parameters = self._parameters(
            **{
                "actor_rollout_ref.actor.megatron.context_parallel_size": 2,
                "actor_rollout_ref.actor.use_dynamic_bsz": False,
                ACTOR_LOG_PROB_DYNAMIC_KEY: True,
                ACTOR_LOG_PROB_MAX_TOKENS_KEY: 100,
                "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 99,
            }
        )
        runtime = self._runtime("actor_log_prob", parameters)

        self.assertTrue(runtime["dynamic_batch"])
        self.assertEqual(runtime["max_token_len_per_gpu"], 100)
        self.assertEqual(runtime["tokens_per_cp_rank"], 100)
        self.assertEqual(
            runtime["sources"]["dynamic_batch"],
            f"parameters:{ACTOR_LOG_PROB_DYNAMIC_KEY}",
        )

    def test_ref_dynamic_batch_uses_ref_keys(self) -> None:
        parameters = self._parameters(
            **{
                "actor_rollout_ref.actor.use_dynamic_bsz": False,
                REF_LOG_PROB_DYNAMIC_KEY: True,
                REF_LOG_PROB_MAX_TOKENS_KEY: 80,
            }
        )
        runtime = self._runtime("ref_log_prob", parameters)

        self.assertTrue(runtime["dynamic_batch"])
        self.assertEqual(runtime["max_token_len_per_gpu"], 80)
        self.assertEqual(runtime["tokens_per_cp_rank"], 80)

    def test_cp_reduces_fixed_workload_log_prob_activation(self) -> None:
        cp1 = self._parameters()
        cp2 = self._parameters(
            **{"actor_rollout_ref.actor.megatron.context_parallel_size": 2}
        )
        runtime1 = self._runtime("actor_log_prob", cp1)
        runtime2 = self._runtime("actor_log_prob", cp2)
        activation1, _ = _activation_bytes(self._architecture(), runtime1)
        activation2, _ = _activation_bytes(self._architecture(), runtime2)

        self.assertEqual(runtime1["tokens_per_cp_rank"], 256)
        self.assertEqual(runtime2["tokens_per_cp_rank"], 128)
        self.assertLess(activation2, activation1)

    def test_log_prob_body_is_forward_only_and_does_not_scale_with_layers(self) -> None:
        runtime = self._runtime("ref_log_prob")
        small, small_details = _activation_bytes(
            self._architecture(num_layers=2, moe_layer_pattern=[0, 0]), runtime
        )
        large, large_details = _activation_bytes(self._architecture(), runtime)

        self.assertEqual(small, large)
        self.assertEqual(
            small_details["formula"],
            "forward_only_one_live_layer_plus_last_stage_logits",
        )
        self.assertEqual(
            small_details["body_live_bytes"], large_details["body_live_bytes"]
        )

    def test_nonfused_actor_has_two_vocab_copies_ref_has_one(self) -> None:
        actor_runtime = self._runtime("actor_log_prob")
        ref_runtime = self._runtime("ref_log_prob")
        actor, actor_details = _activation_bytes(self._architecture(), actor_runtime)
        ref, ref_details = _activation_bytes(self._architecture(), ref_runtime)

        self.assertEqual(actor_details["vocab_logits_copies"], 2)
        self.assertEqual(ref_details["vocab_logits_copies"], 1)
        self.assertGreater(actor, ref)

    def test_nonsp_activation_uses_actual_ffn_ratio(self) -> None:
        runtime = self._runtime(
            "ref_log_prob", self._parameters(**{ACTOR_SP_KEY: False})
        )
        _, narrow = _activation_bytes(
            self._architecture(ffn_hidden_size=128.0), runtime
        )
        _, wide = _activation_bytes(self._architecture(ffn_hidden_size=320.0), runtime)

        self.assertGreater(wide["body_live_bytes"], narrow["body_live_bytes"])

    def test_training_pp_last_stage_includes_vocab_logits(self) -> None:
        runtime = self._runtime("training")
        _, details = _activation_bytes(self._architecture(), runtime)

        self.assertEqual(runtime["pipeline_model_parallel_size"], 2)
        self.assertGreater(details["vocab_logits_one_copy_bytes"], 0)
        self.assertGreater(
            details["stage_peak_bytes"][-1], details["stage_body_bytes"][-1]
        )

    def test_ref_param_offload_removes_second_copy_from_actor_phase(self) -> None:
        architecture = self._architecture()
        resident = self._runtime("actor_log_prob")
        offloaded = self._runtime(
            "actor_log_prob",
            self._parameters(**{REF_PARAM_OFFLOAD_KEY: True}),
        )
        resident_footprint = _parameter_footprint(architecture, resident)
        offloaded_footprint = _parameter_footprint(architecture, offloaded)
        resident_values, resident_details = _phase_residency_components(
            "actor_log_prob", architecture, resident, resident_footprint
        )
        offloaded_values, offloaded_details = _phase_residency_components(
            "actor_log_prob", architecture, offloaded, offloaded_footprint
        )

        self.assertEqual(resident_details["resident_model_copies"], 2)
        self.assertEqual(offloaded_details["resident_model_copies"], 1)
        self.assertLess(
            offloaded_values["resident_model_weights_mb"],
            resident_values["resident_model_weights_mb"],
        )

    def test_training_microbatch_count_includes_rollout_n(self) -> None:
        runtime = self._runtime(
            "training",
            self._parameters(
                **{
                    ROLLOUT_N_KEY: 4,
                    "actor_rollout_ref.actor.ppo_mini_batch_size": 16,
                    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 2,
                }
            ),
        )
        # world=8, TP=2, PP=2 => DP=2; (16 * 4 / 2) / 2 = 16.
        self.assertEqual(runtime["num_microbatches"], 16)

    def test_training_optimizer_offload_does_not_remove_active_optimizer(self) -> None:
        architecture = self._architecture()
        off = self._runtime("training", self._parameters())
        on = self._runtime(
            "training",
            self._parameters(**{ACTOR_OPTIMIZER_OFFLOAD_KEY: True}),
        )
        footprint_off = _parameter_footprint(architecture, off)
        footprint_on = _parameter_footprint(architecture, on)
        values_off, details_off = _phase_residency_components(
            "training", architecture, off, footprint_off
        )
        values_on, details_on = _phase_residency_components(
            "training", architecture, on, footprint_on
        )

        self.assertGreater(values_off["resident_optimizer_state_mb"], 0)
        self.assertEqual(
            values_off["resident_optimizer_state_mb"],
            values_on["resident_optimizer_state_mb"],
        )
        self.assertTrue(details_off["optimizer_resident"])
        self.assertTrue(details_on["optimizer_resident"])

    def test_full_recompute_reduces_training_saved_activation(self) -> None:
        architecture = self._architecture()
        no_recompute = self._runtime("training")
        full_recompute = self._runtime(
            "training",
            self._parameters(
                **{
                    "actor_rollout_ref.actor.megatron."
                    "override_transformer_config.recompute_granularity": "full",
                    "actor_rollout_ref.actor.megatron."
                    "override_transformer_config.recompute_method": "uniform",
                    "actor_rollout_ref.actor.megatron."
                    "override_transformer_config.recompute_num_layers": 4,
                }
            ),
        )
        normal, _ = _activation_bytes(architecture, no_recompute)
        recomputed, details = _activation_bytes(architecture, full_recompute)

        self.assertLess(recomputed, normal)
        self.assertEqual(details["recompute_granularity"], "full")

    def test_lora_ref_reuses_one_model_and_training_state_is_adapter_only(self) -> None:
        architecture = self._architecture()
        full_runtime = self._runtime("training")
        lora_runtime = self._runtime("training", self._parameters(**{LORA_RANK_KEY: 4}))
        full_footprint = _parameter_footprint(architecture, full_runtime)
        lora_footprint = _parameter_footprint(architecture, lora_runtime)
        full_values, full_details = _phase_residency_components(
            "training", architecture, full_runtime, full_footprint
        )
        lora_values, lora_details = _phase_residency_components(
            "training", architecture, lora_runtime, lora_footprint
        )

        self.assertEqual(full_details["resident_model_copies"], 2)
        self.assertEqual(lora_details["resident_model_copies"], 1)
        self.assertLess(
            lora_values["resident_gradients_mb"],
            full_values["resident_gradients_mb"],
        )
        self.assertLess(
            lora_values["resident_optimizer_state_mb"],
            full_values["resident_optimizer_state_mb"],
        )


if __name__ == "__main__":
    unittest.main()
