# -*- coding: utf-8 -*-
from collections.abc import Iterator
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from copaw.agents.routing_learned_router import (
    clear_learned_router_artifact_cache,
)
from copaw.agents.routing_chat_model import RoutingChatModel, RoutingEndpoint
from copaw.config.config import AgentsLLMRoutingConfig


class DummyStructuredOutput(BaseModel):
    value: str


class DummyFormatter:
    pass


class DummyModel:
    def __init__(
        self,
        provider_id: str,
        model_name: str,
        *,
        fail_on_call: bool = False,
        finish_reason: str = "stop",
    ):
        self.provider_id = provider_id
        self.model_name = model_name
        self.stream = True
        self.fail_on_call = fail_on_call
        self.finish_reason = finish_reason

    async def __call__(self, *args, **kwargs):
        if self.fail_on_call:
            raise RuntimeError("call failed")
        return SimpleNamespace(
            provider_id=self.provider_id,
            model_name=self.model_name,
            args=args,
            kwargs=kwargs,
            finish_reason=self.finish_reason,
            metadata={"finish_reason": self.finish_reason},
        )


@pytest.fixture(autouse=True)
def _reset_learned_router(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    monkeypatch.delenv("COPAW_ROUTING_LEARNED_ROUTER_ARTIFACT", raising=False)
    monkeypatch.delenv("COPAW_ROUTING_EVENT_LOG_PATH", raising=False)
    clear_learned_router_artifact_cache()
    yield
    clear_learned_router_artifact_cache()


def _endpoint(
    provider_id: str,
    model_name: str,
    *,
    fail_on_call: bool = False,
) -> RoutingEndpoint:
    def _load():
        return (
            DummyModel(
                provider_id,
                model_name,
                fail_on_call=fail_on_call,
            ),
            DummyFormatter(),
        )

    return RoutingEndpoint(
        provider_id=provider_id,
        model_name=model_name,
        formatter_family=DummyFormatter,
        loader=_load,
    )


def _failing_endpoint(provider_id: str, model_name: str) -> RoutingEndpoint:
    def _load():
        raise RuntimeError("load failed")

    return RoutingEndpoint(
        provider_id=provider_id,
        model_name=model_name,
        formatter_family=DummyFormatter,
        loader=_load,
    )


@pytest.mark.asyncio
async def test_default_local_first_uses_local_route() -> None:
    model = RoutingChatModel(
        local_endpoint=_endpoint("local-provider", "local-model"),
        cloud_endpoint=_endpoint("cloud-provider", "cloud-model"),
        routing_cfg=AgentsLLMRoutingConfig(enabled=True, mode="local_first"),
    )

    response = await model(
        messages=[{"role": "user", "content": "hello"}],
        tools=[],
    )

    assert response.provider_id == "local-provider"
    assert response.model_name == "local-model"


@pytest.mark.asyncio
async def test_hard_guardrail_override_beats_learned_router(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    artifact_path = _write_artifact(tmp_path, intercept=-2.0)
    event_path = tmp_path / "routing-events.jsonl"
    monkeypatch.setenv(
        "COPAW_ROUTING_LEARNED_ROUTER_ARTIFACT",
        str(artifact_path),
    )
    monkeypatch.setenv("COPAW_ROUTING_EVENT_LOG_PATH", str(event_path))
    clear_learned_router_artifact_cache()

    model = RoutingChatModel(
        local_endpoint=_endpoint("local-provider", "local-model"),
        cloud_endpoint=_endpoint("cloud-provider", "cloud-model"),
        routing_cfg=AgentsLLMRoutingConfig(enabled=True, mode="local_first"),
        request_context={
            "agent_id": "agent-1",
            "session_id": "session-1",
        },
    )

    response = await model(
        messages=[{"role": "user", "content": "Return only JSON."}],
        tools=[],
        structured_model=DummyStructuredOutput,
    )

    event = _read_single_event(event_path)
    assert response.provider_id == "cloud-provider"
    assert event["hard_override_reason"] == "structured_output"
    assert event["chosen_route"] == "cloud"
    assert event["feature_flags"]["hard_rule_result"] == "structured_output"


@pytest.mark.asyncio
async def test_structured_output_forces_cloud_route() -> None:
    model = RoutingChatModel(
        local_endpoint=_endpoint("local-provider", "local-model"),
        cloud_endpoint=_endpoint("cloud-provider", "cloud-model"),
        routing_cfg=AgentsLLMRoutingConfig(enabled=True, mode="local_first"),
    )

    response = await model(
        messages=[{"role": "user", "content": "extract a schema"}],
        tools=[],
        structured_model=DummyStructuredOutput,
    )

    assert response.provider_id == "cloud-provider"
    assert response.model_name == "cloud-model"


@pytest.mark.asyncio
async def test_strict_format_prompt_forces_cloud_route() -> None:
    model = RoutingChatModel(
        local_endpoint=_endpoint("local-provider", "local-model"),
        cloud_endpoint=_endpoint("cloud-provider", "cloud-model"),
        routing_cfg=AgentsLLMRoutingConfig(enabled=True, mode="local_first"),
    )

    response = await model(
        messages=[
            {
                "role": "user",
                "content": (
                    "Return only a JSON object with keys "
                    "project and status."
                ),
            },
        ],
        tools=[],
    )

    assert response.provider_id == "cloud-provider"
    assert response.model_name == "cloud-model"


@pytest.mark.asyncio
async def test_freshness_sensitive_prompt_forces_cloud_route() -> None:
    model = RoutingChatModel(
        local_endpoint=_endpoint("local-provider", "local-model"),
        cloud_endpoint=_endpoint("cloud-provider", "cloud-model"),
        routing_cfg=AgentsLLMRoutingConfig(enabled=True, mode="local_first"),
    )

    response = await model(
        messages=[
            {
                "role": "user",
                "content": (
                    "Compare the latest prices of AMD and NVDA this week."
                ),
            },
        ],
        tools=[],
    )

    assert response.provider_id == "cloud-provider"
    assert response.model_name == "cloud-model"


@pytest.mark.asyncio
async def test_missing_artifact_falls_back_to_rule_based_local() -> None:
    model = RoutingChatModel(
        local_endpoint=_endpoint("local-provider", "local-model"),
        cloud_endpoint=_endpoint("cloud-provider", "cloud-model"),
        routing_cfg=AgentsLLMRoutingConfig(enabled=True, mode="local_first"),
    )

    response = await model(
        messages=[{"role": "user", "content": "hello"}],
        tools=[],
    )

    assert response.provider_id == "local-provider"
    assert response.model_name == "local-model"


@pytest.mark.asyncio
async def test_learned_router_threshold_routes_to_cloud(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    artifact_path = _write_artifact(tmp_path, intercept=1.0)
    event_path = tmp_path / "routing-events.jsonl"
    monkeypatch.setenv(
        "COPAW_ROUTING_LEARNED_ROUTER_ARTIFACT",
        str(artifact_path),
    )
    monkeypatch.setenv("COPAW_ROUTING_EVENT_LOG_PATH", str(event_path))
    clear_learned_router_artifact_cache()

    model = RoutingChatModel(
        local_endpoint=_endpoint("local-provider", "local-model"),
        cloud_endpoint=_endpoint("cloud-provider", "cloud-model"),
        routing_cfg=AgentsLLMRoutingConfig(enabled=True, mode="local_first"),
        request_context={
            "request_id": "req-1",
            "agent_id": "agent-2",
            "session_id": "session-2",
        },
    )

    response = await model(
        messages=[{"role": "user", "content": "hello"}],
        tools=[],
    )

    event = _read_single_event(event_path)
    assert response.provider_id == "cloud-provider"
    assert event["chosen_route"] == "cloud"
    assert event["decision_source"] == "learned_router"
    assert event["learned_score"] >= 0.65


@pytest.mark.asyncio
async def test_recent_tool_context_forces_cloud_route() -> None:
    model = RoutingChatModel(
        local_endpoint=_endpoint("local-provider", "local-model"),
        cloud_endpoint=_endpoint("cloud-provider", "cloud-model"),
        routing_cfg=AgentsLLMRoutingConfig(enabled=True, mode="local_first"),
    )

    response = await model(
        messages=[
            {"role": "assistant", "tool_calls": [{"id": "call-1"}]},
            {"role": "tool", "content": "tool result"},
        ],
        tools=[],
    )

    assert response.provider_id == "cloud-provider"
    assert response.model_name == "cloud-model"


@pytest.mark.asyncio
async def test_local_load_failure_falls_back_to_cloud() -> None:
    model = RoutingChatModel(
        local_endpoint=_failing_endpoint("local-provider", "local-model"),
        cloud_endpoint=_endpoint("cloud-provider", "cloud-model"),
        routing_cfg=AgentsLLMRoutingConfig(enabled=True, mode="local_first"),
    )

    response = await model(
        messages=[{"role": "user", "content": "hello"}],
        tools=[],
    )

    assert response.provider_id == "cloud-provider"
    assert response.model_name == "cloud-model"


@pytest.mark.asyncio
async def test_invocation_fallback_reuses_secondary_slot_and_logs_event(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    artifact_path = _write_artifact(tmp_path, intercept=-1.0)
    event_path = tmp_path / "routing-events.jsonl"
    monkeypatch.setenv(
        "COPAW_ROUTING_LEARNED_ROUTER_ARTIFACT",
        str(artifact_path),
    )
    monkeypatch.setenv("COPAW_ROUTING_EVENT_LOG_PATH", str(event_path))
    clear_learned_router_artifact_cache()

    model = RoutingChatModel(
        local_endpoint=_endpoint(
            "local-provider",
            "local-model",
            fail_on_call=True,
        ),
        cloud_endpoint=_endpoint("cloud-provider", "cloud-model"),
        routing_cfg=AgentsLLMRoutingConfig(enabled=True, mode="local_first"),
        request_context={"agent_id": "agent-3", "session_id": "session-3"},
    )

    response = await model(
        messages=[{"role": "user", "content": "hello"}],
        tools=[],
    )

    event = _read_single_event(event_path)
    assert response.provider_id == "cloud-provider"
    assert event["chosen_route"] == "cloud"
    assert event["fallback_used"] is True
    assert "fallback:local_call_error" in event["reasons"]


@pytest.mark.asyncio
async def test_routing_event_contains_required_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    artifact_path = _write_artifact(tmp_path, intercept=1.0)
    event_path = tmp_path / "routing-events.jsonl"
    monkeypatch.setenv(
        "COPAW_ROUTING_LEARNED_ROUTER_ARTIFACT",
        str(artifact_path),
    )
    monkeypatch.setenv("COPAW_ROUTING_EVENT_LOG_PATH", str(event_path))
    clear_learned_router_artifact_cache()

    model = RoutingChatModel(
        local_endpoint=_endpoint("local-provider", "local-model"),
        cloud_endpoint=_endpoint("cloud-provider", "cloud-model"),
        routing_cfg=AgentsLLMRoutingConfig(enabled=True, mode="local_first"),
        request_context={
            "request_id": "req-99",
            "agent_id": "agent-99",
            "session_id": "session-99",
        },
    )

    await model(messages=[{"role": "user", "content": "hello"}], tools=[])

    event = _read_single_event(event_path)
    assert event["request_id"] == "req-99"
    assert event["agent_id"] == "agent-99"
    assert event["session_id"] == "session-99"
    assert event["chosen_route"] == "cloud"
    assert event["hard_override_reason"] is None
    assert isinstance(event["learned_score"], float)
    assert isinstance(event["feature_flags"], dict)
    assert event["local_slot"]["provider_id"] == "local-provider"
    assert event["cloud_slot"]["provider_id"] == "cloud-provider"
    assert isinstance(event["fallback_used"], bool)
    assert event["latency_ms"] >= 0
    assert event["finish_reason"] == "stop"


def _write_artifact(tmp_path: Path, *, intercept: float) -> str:
    artifact_path = tmp_path / "learned_router_v1.json"
    payload = {
        "version": "copaw-learned-router-v1",
        "threshold": 0.65,
        "n_features": 4096,
        "ngram_range": [1, 2],
        "coefficients": [0.0] * (4096 + 8),
        "intercept": intercept,
        "structured_feature_names": [
            "prompt_chars",
            "message_count",
            "non_text",
            "recent_tool_context",
            "tool_choice",
            "freshness_flag",
            "strict_format_flag",
            "hard_rule_result",
        ],
        "metadata": {"trainer": "unit-test"},
    }
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")
    return str(artifact_path)


def _read_single_event(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    return json.loads(lines[0])
