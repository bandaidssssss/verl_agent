from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from agent_tools import ToolRegistry
from prompting import render_prompt


class AgentError(RuntimeError):
    pass


class AgentResponseError(AgentError):
    """Raised after an agent repeatedly returns an unusable final response."""

    def __init__(
        self,
        role: str,
        reason: str,
        raw_response: str,
        repair_attempts: int,
        conversation: AgentConversation,
        error_type: str = "invalid_agent_response",
    ) -> None:
        super().__init__(reason)
        self.role = role
        self.reason = reason
        self.raw_response = raw_response
        self.repair_attempts = repair_attempts
        self.error_type = error_type
        self.trace = conversation.as_trace()

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent_role": self.role,
            "error_type": self.error_type,
            "reason": self.reason,
            "repair_attempts": self.repair_attempts,
            "raw_response": self.raw_response,
            "agent_trace": copy.deepcopy(self.trace),
        }


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.S)
    if fenced:
        stripped = fenced.group(1)
    else:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise AgentError(f"agent did not return valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise AgentError("agent response must be a JSON object")
    return value


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _response_text(response: Any) -> str:
    choice = response.choices[0]
    content = choice.message.content
    return content or ""


def _tool_calls(response: Any) -> list[Any]:
    choice = response.choices[0]
    raw = list(choice.message.tool_calls or [])
    normalized: list[dict[str, Any]] = []
    for tc in raw:
        func = _field(tc, "function", {})
        normalized.append({
            "name": _field(func, "name", ""),
            "arguments": _field(func, "arguments", "{}"),
        })
    return normalized


@dataclass
class AgentConversation:
    role: str
    context: dict[str, Any]
    messages: list[dict[str, Any]]
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    request_errors: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(
        default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "api_calls": 0}
    )
    completed_turns: int = 0

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def as_trace(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "context": copy.deepcopy(self.context),
            "messages": copy.deepcopy(self.messages),
            "tool_calls": copy.deepcopy(self.tool_trace),
            "request_errors": copy.deepcopy(self.request_errors),
            "usage": dict(self.usage),
            "completed_turns": self.completed_turns,
        }


@dataclass
class AgentRun:
    result: dict[str, Any]
    conversation: AgentConversation

    def as_trace(self) -> dict[str, Any]:
        trace = self.conversation.as_trace()
        trace["result"] = copy.deepcopy(self.result)
        return trace


class LLMRoleAgent:
    def __init__(
        self,
        role: str,
        prompt_path: str | Path,
        registry: ToolRegistry,
        agent_config: Mapping[str, Any],
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.role = role
        self.prompt_template = Path(prompt_path).read_text(encoding="utf-8")
        self.registry = registry
        self.config = dict(agent_config)
        self.api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model = os.getenv("INFER_MODEL", str(self.config.get("infer_model", "gpt-5")))
        self._client_factory = client_factory
        self._client: Any = None

    def _stream_event(self, event: str, payload: Mapping[str, Any]) -> None:
        if not bool(self.config.get("stream_agent_events", True)):
            return
        print(
            f"\n[Agent:{self.role}] {event}\n"
            + json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str),
            flush=True,
        )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if self._client_factory:
            self._client = self._client_factory()
            return self._client
        if not self.api_key:
            raise AgentError("API_KEY or OPENAI_API_KEY is required for LLM agent decisions")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise AgentError("the openai package is required for LLM agent decisions") from exc
        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": float(self.config.get("llm_timeout_seconds", 120.0)),
            # Application-level retries below are authoritative so request
            # attempts and JSON repairs share one bounded retry policy.
            "max_retries": 0,
            "default_headers": {"User-Agent": "curl/7.81.0"},
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)
        return self._client

    def new_conversation(self, context: Mapping[str, Any]) -> AgentConversation:
        definitions = self.registry.definitions(self.role)
        prompt = render_prompt(self.prompt_template, context, definitions)
        return AgentConversation(
            role=self.role,
            context=copy.deepcopy(dict(context)),
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        "Analyze the current evidence. Call tools proactively when more information "
                        "is needed; once the evidence is sufficient, output the required JSON object."
                    ),
                },
            ],
        )

    @staticmethod
    def _record_usage(conversation: AgentConversation, response: Any) -> None:
        usage = _field(response, "usage")
        conversation.usage["api_calls"] += 1
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            value = _field(usage, name, 0) if usage else 0
            conversation.usage[name] += int(value or 0)

    def run(
        self,
        context: Mapping[str, Any] | None = None,
        conversation: AgentConversation | None = None,
    ) -> AgentRun:
        if conversation is None:
            if context is None:
                raise AgentError("a new agent run requires context")
            conversation = self.new_conversation(context)
        elif conversation.role != self.role:
            raise AgentError(f"cannot use {conversation.role} conversation with {self.role} agent")

        client = self._get_client()
        runtime = self.registry.runtime(conversation.context)
        schemas = self.registry.api_schemas(self.role)
        max_tool_rounds = max(0, int(self.config.get("max_tool_rounds", 6)))
        max_response_repairs = max(0, int(self.config.get("max_response_repair_rounds", 2)))
        max_output_tokens = int(self.config.get("llm_max_output_tokens", 4096))
        enable_json_response_format = bool(
            self.config.get("enable_json_response_format", False)
        )
        tool_round = 0
        response_repairs = 0

        while True:
            request: dict[str, Any] = {
                "model": self.model,
                "messages": conversation.messages,
                "max_tokens": max_output_tokens,
            }
            if enable_json_response_format:
                request["response_format"] = {"type": "json_object"}
            if schemas and tool_round < max_tool_rounds and response_repairs == 0:
                request["tools"] = schemas
                request["tool_choice"] = "auto"
            request_retries = 0
            while True:
                try:
                    response = client.chat.completions.create(**request)
                    break
                except Exception as exc:
                    request_error = {
                        "retry_attempt": request_retries + 1,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    conversation.request_errors.append(request_error)
                    self._stream_event("request_rejected", request_error)
                    if request_retries >= max_response_repairs:
                        raise AgentResponseError(
                            self.role,
                            (
                                f"{self.role} API request failed after "
                                f"{request_retries} retry attempts: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                            "",
                            request_retries,
                            conversation,
                            error_type="agent_request_failed",
                        ) from exc
                    request_retries += 1
            self._record_usage(conversation, response)
            calls = _tool_calls(response)
            if calls:
                if tool_round >= max_tool_rounds:
                    raise AgentError(f"{self.role} exceeded max_tool_rounds={max_tool_rounds}")
                summary = ["I need to call the following tools:"]
                for call in calls:
                    summary.append(f"- {_field(call, 'name')}: {_field(call, 'arguments', '{}')}")
                conversation.messages.append({"role": "assistant", "content": "\n".join(summary)})
                for call in calls:
                    name = str(_field(call, "name", ""))
                    raw_arguments = _field(call, "arguments", "{}")
                    self._stream_event(
                        "tool_call",
                        {"name": name, "arguments": raw_arguments},
                    )
                    try:
                        arguments = raw_arguments if isinstance(raw_arguments, Mapping) else json.loads(raw_arguments or "{}")
                        result = self.registry.execute(self.role, name, arguments, runtime)
                        trace = {
                            "tool_round": tool_round + 1,
                            "name": name,
                            "arguments": copy.deepcopy(dict(arguments)),
                            "status": "success",
                            "result": copy.deepcopy(result),
                        }
                    except (AgentError, ValueError, TypeError, json.JSONDecodeError, RuntimeError) as exc:
                        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                        trace = {
                            "tool_round": tool_round + 1,
                            "name": name,
                            "arguments": raw_arguments,
                            "status": "error",
                            "result": result,
                        }
                    conversation.tool_trace.append(trace)
                    self._stream_event(
                        "tool_result",
                        {
                            "name": name,
                            "status": trace["status"],
                            "error": result.get("error") if isinstance(result, Mapping) else None,
                        },
                    )
                    conversation.messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"The result of tool `{name}` is:\n```json\n"
                                + json.dumps(result, ensure_ascii=False, default=str)
                                + "\n```"
                            ),
                        }
                    )
                tool_round += 1
                continue

            output_text = _response_text(response)
            finish_reason = _field(response.choices[0], "finish_reason")
            try:
                if finish_reason == "length":
                    raise AgentError(
                        "agent response was truncated because finish_reason='length'"
                    )
                result = _extract_json(output_text)
            except AgentError as exc:
                conversation.messages.append(
                    {"role": "assistant", "content": output_text}
                )
                self._stream_event(
                    "answer_rejected",
                    {
                        "reason": str(exc),
                        "repair_attempt": response_repairs + 1,
                        "raw_response": output_text,
                    },
                )
                if response_repairs >= max_response_repairs:
                    raise AgentResponseError(
                        self.role,
                        (
                            f"{self.role} did not return valid JSON after "
                            f"{response_repairs} repair attempts: {exc}"
                        ),
                        output_text,
                        response_repairs,
                        conversation,
                    ) from exc
                response_repairs += 1
                conversation.add_user_message(
                    "The previous response could not be parsed as the final JSON object. "
                    f"Parse error: {exc}. "
                    "Correct the response and output exactly one complete, valid JSON object that "
                    "matches the required schema. Do not call tools and do not output Markdown, "
                    "code fences, or additional explanation."
                )
                continue
            self._stream_event("answer", result)
            conversation.messages.append({"role": "assistant", "content": output_text})
            conversation.completed_turns += 1
            return AgentRun(result, conversation)


class AgentSet:
    def __init__(
        self,
        root: str | Path,
        mode: str = "llm",
        agent_config: Mapping[str, Any] | None = None,
        history_path: str | Path | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        prompt_root = self.root / "prompts"
        self.mode = mode
        self.config = dict(agent_config or {})
        default_history = self.root / str(self.config.get("output_dir", "output")) / "trials.jsonl"
        self.history_path = Path(history_path).resolve() if history_path else default_history.resolve()
        self.registry = ToolRegistry(self.root, self.config, self.history_path)
        self.proposal = LLMRoleAgent("proposal", prompt_root / "proposal.md", self.registry, self.config)
        self.feasibility = LLMRoleAgent("feasibility", prompt_root / "feasibility.md", self.registry, self.config)
        self.diagnosis = LLMRoleAgent("diagnosis", prompt_root / "diagnosis.md", self.registry, self.config)
        self.train_health = LLMRoleAgent(
            "train_health", prompt_root / "train_health.md", self.registry, self.config
        )

    @staticmethod
    def _rules_run(role: str, context: Mapping[str, Any], result: dict[str, Any]) -> AgentRun:
        conversation = AgentConversation(role, copy.deepcopy(dict(context)), [])
        conversation.completed_turns = 1
        conversation.messages.append({"role": "assistant", "content": json.dumps(result, ensure_ascii=False)})
        return AgentRun(result, conversation)

    def propose(
        self,
        context: Mapping[str, Any] | None = None,
        conversation: AgentConversation | None = None,
    ) -> AgentRun:
        if self.mode == "rules":
            rule_context = context or (conversation.context if conversation else {})
            return self._rules_run(
                "proposal",
                rule_context,
                {"decision": "keep", "reason": "rules mode keeps the current configuration", "changes": {}},
            )
        return self.proposal.run(context, conversation)

    def review(self, context: Mapping[str, Any]) -> AgentRun:
        if self.mode == "rules":
            return self._rules_run(
                "feasibility",
                context,
                {"verdict": "valid", "reason": "deterministic validation passed", "risks": []},
            )
        return self.feasibility.run(context)

    def diagnose(self, context: Mapping[str, Any]) -> AgentRun:
        if self.mode == "rules":
            error_type = context.get("trial", {}).get("error", {}).get("type") or "UNKNOWN_FAILURE"
            return self._rules_run(
                "diagnosis",
                context,
                {
                    "failure_type": error_type,
                    "training_substage": context.get("trial", {}).get("failure_phase") or "unknown",
                    "evidence": context.get("trial", {}).get("error", {}).get("evidence", []),
                    "reason": "deterministic log classification",
                    "confidence": 1.0 if error_type != "UNKNOWN_FAILURE" else 0.3,
                },
            )
        return self.diagnosis.run(context)

    def assess_health(self, context: Mapping[str, Any]) -> AgentRun:
        if self.mode == "rules":
            rules = context.get("health_event", {}).get("rules", [])
            names = [str(row.get("name")) for row in rules if isinstance(row, Mapping)]
            return self._rules_run(
                "train_health",
                context,
                {
                    "verdict": "unhealthy",
                    "action": "stop",
                    "confidence": 1.0,
                    "reason_codes": names or ["JF_HPO_RULE_TRIGGERED"],
                    "evidence": ["JF-HPO deterministic early-stop condition was satisfied"],
                    "counterevidence": [],
                    "observe_for_updates": 0,
                    "reason": "rules mode accepts the configured JF-HPO trigger",
                },
            )

        run = self.train_health.run(context)
        result = run.result
        verdicts = {"healthy", "watch", "unhealthy", "insufficient_evidence"}
        actions = {"continue", "observe", "stop"}
        verdict = result.get("verdict")
        action = result.get("action")
        confidence = result.get("confidence")
        if verdict not in verdicts:
            raise AgentError(f"train_health returned invalid verdict: {verdict!r}")
        if action not in actions:
            raise AgentError(f"train_health returned invalid action: {action!r}")
        if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
            raise AgentError("train_health confidence must be between 0 and 1")
        if action == "stop" and verdict != "unhealthy":
            raise AgentError("train_health may recommend stop only with verdict=unhealthy")
        observe = result.get("observe_for_updates", 0)
        if not isinstance(observe, int) or observe < 0:
            raise AgentError("train_health observe_for_updates must be a non-negative integer")
        return run
