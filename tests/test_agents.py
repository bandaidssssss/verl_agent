from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agent_tools import ToolRegistry
from agents import AgentResponseError, AgentSet, LLMRoleAgent
from config_utils import load_json


ROOT = Path(__file__).resolve().parents[1]


class FakeResponses:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []

    def create(self, **kwargs: object) -> object:
        self.requests.append(copy.deepcopy(kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClient:
    def __init__(self, responses: list[object]) -> None:
        self.responses = FakeResponses(responses)
        self.chat = SimpleNamespace(completions=self.responses)


def response_with_tool(name: str, arguments: str) -> object:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(name=name, arguments=arguments)
                        )
                    ],
                )
            )
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=3, total_tokens=13),
    )


def response_with_json(text: str) -> object:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, tool_calls=[])
            )
        ],
        usage=SimpleNamespace(input_tokens=12, output_tokens=5, total_tokens=17),
    )


class LLMRoleAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        config = load_json(ROOT / "config" / "agent_config.json")
        self.registry = ToolRegistry(ROOT, config, Path(self.temp_dir.name) / "trials.jsonl")
        self.config = {**config, "max_tool_rounds": 3, "stream_agent_events": False}
        self.context = {
            "current_stage": "hardware_tuning",
            "mode": "hardware_tuning",
            "current_parameters": load_json(ROOT / "config" / "base_parameters.json"),
            "editable_parameters": ["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"],
            "constraints": {"max_parameter_changes": 3},
            "recent_trials": [],
        }

    def test_tool_result_is_added_before_final_decision(self) -> None:
        fake = FakeClient(
            [
                response_with_tool(
                    "parameter_understanding",
                    '{"items":["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"]}',
                ),
                response_with_json(
                    '{"decision":"modify","reason":"two hypotheses","candidates":[{"candidate_id":"micro_batch","reference_trial_id":1,"reference_reason":"measured baseline","reason":"tested","changes":{"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu":{"from":1,"to":2,"reason":"increase throughput"}},"expected_effect":{"throughput":"increase"}},{"candidate_id":"rollout_limit","reference_trial_id":1,"reference_reason":"measured baseline","reason":"test rollout scheduling","changes":{"actor_rollout_ref.rollout.max_num_seqs":{"from":128,"to":256,"reason":"increase scheduler concurrency"}},"expected_effect":{"rollout_throughput":"increase"}}]}'
                ),
            ]
        )
        agent = LLMRoleAgent(
            "proposal",
            ROOT / "prompts" / "proposal.md",
            self.registry,
            self.config,
            client_factory=lambda: fake,
        )
        run = agent.run(self.context)
        self.assertEqual(run.result["decision"], "modify")
        self.assertEqual(run.conversation.tool_trace[0]["status"], "success")
        self.assertEqual(len(fake.responses.requests), 2)
        second_messages = fake.responses.requests[1]["messages"]
        self.assertTrue(any("result of tool" in message["content"] for message in second_messages))
        self.assertEqual(
            fake.responses.requests[0]["response_format"], {"type": "json_object"}
        )

    def test_rejection_continues_the_same_conversation(self) -> None:
        fake = FakeClient(
            [
                response_with_json('{"decision":"modify","reason":"first","candidates":[{"candidate_id":"a","reference_trial_id":1,"reference_reason":"baseline","reason":"first a","changes":{"x":{"from":0,"to":1,"reason":"first attempt"}}},{"candidate_id":"b","reference_trial_id":1,"reference_reason":"baseline","reason":"first b","changes":{"y":{"from":0,"to":1,"reason":"first attempt"}}}]}'),
                response_with_json('{"decision":"modify","reason":"second","candidates":[{"candidate_id":"a","reference_trial_id":1,"reference_reason":"baseline","reason":"second a","changes":{"x":{"from":0,"to":2,"reason":"second attempt"}}},{"candidate_id":"b","reference_trial_id":1,"reference_reason":"baseline","reason":"second b","changes":{"y":{"from":0,"to":2,"reason":"second attempt"}}}]}'),
            ]
        )
        agent = LLMRoleAgent(
            "proposal",
            ROOT / "prompts" / "proposal.md",
            self.registry,
            self.config,
            client_factory=lambda: fake,
        )
        first = agent.run(self.context)
        first.conversation.add_user_message("## Proposal Batch Attempt 1 Was Rejected\nValidator says x=1 is invalid")
        second = agent.run(conversation=first.conversation)
        self.assertEqual(second.result["candidates"][0]["changes"]["x"]["to"], 2)
        self.assertEqual(second.conversation.completed_turns, 2)
        request_messages = fake.responses.requests[1]["messages"]
        self.assertTrue(
            any("Proposal Batch Attempt 1 Was Rejected" in message["content"] for message in request_messages)
        )
        self.assertTrue(any('"reason":"first"' in message["content"] for message in request_messages))

    def test_invalid_json_is_repaired_in_the_same_conversation(self) -> None:
        fake = FakeClient(
            [
                response_with_json(
                    '{"decision":"keep"}\n{"decision":"stop"}'
                ),
                response_with_json(
                    '{"decision":"keep","reason":"corrected","changes":{}}'
                ),
            ]
        )
        agent = LLMRoleAgent(
            "proposal",
            ROOT / "prompts" / "proposal.md",
            self.registry,
            self.config,
            client_factory=lambda: fake,
        )

        run = agent.run(self.context)

        self.assertEqual(run.result["reason"], "corrected")
        self.assertEqual(len(fake.responses.requests), 2)
        repair_messages = fake.responses.requests[1]["messages"]
        self.assertTrue(
            any("could not be parsed as the final JSON object" in message["content"] for message in repair_messages)
        )
        self.assertNotIn("tools", fake.responses.requests[1])

    def test_invalid_json_exhaustion_raises_typed_response_error(self) -> None:
        fake = FakeClient(
            [
                response_with_json("not json"),
                response_with_json("still not json"),
                response_with_json("also not json"),
            ]
        )
        agent = LLMRoleAgent(
            "proposal",
            ROOT / "prompts" / "proposal.md",
            self.registry,
            self.config,
            client_factory=lambda: fake,
        )

        with self.assertRaises(AgentResponseError) as raised:
            agent.run(self.context)

        self.assertEqual(raised.exception.role, "proposal")
        self.assertEqual(raised.exception.repair_attempts, 2)
        self.assertEqual(len(fake.responses.requests), 3)
        self.assertEqual(
            raised.exception.trace["messages"][-1]["content"], "also not json"
        )

    def test_api_request_error_is_retried_with_the_json_retry_limit(self) -> None:
        fake = FakeClient(
            [
                RuntimeError("temporary API failure"),
                response_with_json(
                    '{"decision":"keep","reason":"request recovered","changes":{}}'
                ),
            ]
        )
        agent = LLMRoleAgent(
            "proposal",
            ROOT / "prompts" / "proposal.md",
            self.registry,
            self.config,
            client_factory=lambda: fake,
        )

        run = agent.run(self.context)

        self.assertEqual(run.result["reason"], "request recovered")
        self.assertEqual(len(fake.responses.requests), 2)
        self.assertEqual(len(run.conversation.request_errors), 1)

    def test_api_request_retry_exhaustion_raises_response_error(self) -> None:
        fake = FakeClient(
            [
                RuntimeError("API failure 1"),
                RuntimeError("API failure 2"),
                RuntimeError("API failure 3"),
            ]
        )
        agent = LLMRoleAgent(
            "proposal",
            ROOT / "prompts" / "proposal.md",
            self.registry,
            self.config,
            client_factory=lambda: fake,
        )

        with self.assertRaises(AgentResponseError) as raised:
            agent.run(self.context)

        self.assertEqual(raised.exception.error_type, "agent_request_failed")
        self.assertEqual(raised.exception.repair_attempts, 2)
        self.assertEqual(len(fake.responses.requests), 3)
        self.assertEqual(len(raised.exception.trace["request_errors"]), 3)


class TrainHealthRulesAgentTest(unittest.TestCase):
    def test_rules_mode_accepts_jf_hpo_trigger(self) -> None:
        config = load_json(ROOT / "config" / "agent_config.json")
        config["stream_agent_events"] = False
        with tempfile.TemporaryDirectory() as directory:
            agents = AgentSet(
                ROOT,
                "rules",
                config,
                Path(directory) / "trials.jsonl",
            )
            run = agents.assess_health(
                {
                    "current_stage": "stability_tuning",
                    "health_event": {
                        "rules": [{"name": "jf_hpo_kl_growth"}],
                    },
                    "recent_trials": [],
                }
            )
        self.assertEqual(run.result["verdict"], "unhealthy")
        self.assertEqual(run.result["action"], "stop")


class DiagnosisRulesAgentTest(unittest.TestCase):
    def test_memory_bottleneck_alone_does_not_assign_failure_substage(self) -> None:
        config = load_json(ROOT / "config" / "agent_config.json")
        config["stream_agent_events"] = False
        with tempfile.TemporaryDirectory() as directory:
            agents = AgentSet(
                ROOT,
                "rules",
                config,
                Path(directory) / "trials.jsonl",
            )
            run = agents.diagnose(
                {
                    "trial": {
                        "error": {
                            "type": "CUDA_OUT_OF_MEMORY",
                            "evidence": ["allocation failed without a classified phase"],
                        },
                        "resource": {"memory_bottleneck": "rollout"},
                    }
                }
            )

        self.assertEqual(run.result["training_substage"], "unknown")


if __name__ == "__main__":
    unittest.main()
