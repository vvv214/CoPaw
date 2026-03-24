# -*- coding: utf-8 -*-
"""Tests for routing benchmark script helpers."""

from __future__ import annotations

# pylint: disable=wrong-import-position

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "routing"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import (  # noqa: E402
    CODING_DASHSCOPE_BASE_URL,
    DASHSCOPE_BASE_URL,
    ProviderEndpointConfig,
    build_openai_compatible_headers,
    build_provider_benchmark_runtime,
    extract_finish_reason,
    normalize_chat_result,
    normalize_usage,
    resolve_provider_endpoint,
    supports_benchmark_provider,
)


def test_build_openai_compatible_headers_for_dashscope() -> None:
    headers = build_openai_compatible_headers(
        base_url=DASHSCOPE_BASE_URL,
        api_key="sk-test",
    )

    assert headers["Authorization"] == "Bearer sk-test"
    assert "x-dashscope-agentapp" in headers
    assert "X-DashScope-Cdpl" not in headers


def test_build_openai_compatible_headers_for_coding_plan() -> None:
    headers = build_openai_compatible_headers(
        base_url=CODING_DASHSCOPE_BASE_URL,
        api_key="sk-sp-test",
    )

    assert headers["Authorization"] == "Bearer sk-sp-test"
    assert "X-DashScope-Cdpl" in headers
    assert "x-dashscope-agentapp" not in headers


def test_build_openai_compatible_headers_for_generic_openai() -> None:
    headers = build_openai_compatible_headers(
        base_url="http://127.0.0.1:8102/v1",
        api_key="copaw-local",
    )

    assert headers == {
        "Content-Type": "application/json",
        "Authorization": "Bearer copaw-local",
    }


def test_resolve_provider_endpoint_uses_default_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_provider_config(
        tmp_path,
        provider_id="openai",
        payload={
            "id": "openai",
            "base_url": "https://api.openai.com/v1",
            "api_key": "sk-test",
            "require_api_key": True,
            "is_local": False,
            "models": [{"id": "gpt-5-mini", "name": "GPT-5 Mini"}],
            "extra_models": [],
        },
    )
    monkeypatch.setattr("_common.PROVIDER_CONFIG_ROOT", tmp_path)

    endpoint = resolve_provider_endpoint("openai")

    assert endpoint.provider_id == "openai"
    assert endpoint.base_url == "https://api.openai.com/v1"
    assert endpoint.api_key == "sk-test"
    assert endpoint.model == "gpt-5-mini"
    assert endpoint.require_api_key is True
    assert endpoint.is_local is False


def test_resolve_provider_endpoint_prefers_extra_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_provider_config(
        tmp_path,
        provider_id="custom-openai",
        payload={
            "id": "custom-openai",
            "base_url": "https://example.com/v1",
            "api_key": "sk-custom",
            "require_api_key": True,
            "is_local": False,
            "models": [{"id": "fallback-model", "name": "Fallback Model"}],
            "extra_models": [{"id": "preferred-model", "name": "Preferred"}],
        },
        is_custom=True,
    )
    monkeypatch.setattr("_common.PROVIDER_CONFIG_ROOT", tmp_path)

    endpoint = resolve_provider_endpoint("custom-openai")

    assert endpoint.model == "preferred-model"


def test_resolve_provider_endpoint_rejects_missing_required_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_provider_config(
        tmp_path,
        provider_id="dashscope",
        payload={
            "id": "dashscope",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": "",
            "require_api_key": True,
            "is_local": False,
            "models": [{"id": "qwen3-max", "name": "Qwen3 Max"}],
            "extra_models": [],
        },
    )
    monkeypatch.setattr("_common.PROVIDER_CONFIG_ROOT", tmp_path)

    with pytest.raises(ValueError, match="requires an API key"):
        resolve_provider_endpoint("dashscope")


def test_supports_benchmark_provider_accepts_anthropic_and_gemini() -> None:
    assert supports_benchmark_provider(
        {"id": "anthropic", "chat_model": "AnthropicChatModel"},
    )
    assert supports_benchmark_provider(
        {"id": "gemini", "chat_model": "GeminiChatModel"},
    )


def test_supports_benchmark_provider_rejects_unknown_chat_model() -> None:
    assert not supports_benchmark_provider(
        {"id": "custom", "chat_model": "CustomChatModel"},
    )


@pytest.mark.asyncio
async def test_normalize_chat_result_handles_structured_usage() -> None:
    result = SimpleNamespace(
        content=[
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ],
        finish_reason="length",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=4,
            time=0.25,
            metadata={"provider": "test"},
        ),
    )

    normalized = await normalize_chat_result(result)

    assert normalized.response_text == "hello\nworld"
    assert normalized.finish_reason == "length"
    assert normalized.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
        "time": 0.25,
        "metadata": {"provider": "test"},
    }


@pytest.mark.asyncio
async def test_normalize_chat_result_handles_streaming_snapshots() -> None:
    async def fake_stream():
        yield SimpleNamespace(
            content=[{"type": "text", "text": "hel"}],
            finish_reason="streaming",
            usage=None,
        )
        yield SimpleNamespace(
            content=[{"type": "text", "text": "hello"}],
            metadata={"finish_reason": "stop"},
            usage=SimpleNamespace(input_tokens=3, output_tokens=1),
        )

    normalized = await normalize_chat_result(fake_stream())

    assert normalized.response_text == "hello"
    assert normalized.finish_reason == "stop"
    assert normalized.usage == {
        "prompt_tokens": 3,
        "completion_tokens": 1,
        "total_tokens": 4,
    }


def test_extract_finish_reason_falls_back_to_metadata() -> None:
    assert (
        extract_finish_reason(
            SimpleNamespace(
                finish_reason=None,
                metadata={"finish_reason": "tool_use"},
            ),
        )
        == "tool_use"
    )


def test_normalize_usage_passthrough_dict() -> None:
    payload = {"prompt_tokens": 1, "completion_tokens": 2}

    assert normalize_usage(payload) is payload


def test_build_provider_benchmark_runtime_clones_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProvider:
        def __init__(self, *, provider_id: str, base_url: str) -> None:
            self.id = provider_id
            self.base_url = base_url
            self.chat_model = "OpenAIChatModel"

        def model_copy(self, deep: bool = True) -> "FakeProvider":
            assert deep is True
            return FakeProvider(
                provider_id=self.id,
                base_url=self.base_url,
            )

        def get_chat_model_instance(self, model_id: str) -> str:
            return f"remote:{model_id}@{self.base_url}"

    class FakeManager:
        def get_provider(self, provider_id: str) -> FakeProvider | None:
            assert provider_id == "openai"
            return FakeProvider(
                provider_id="openai",
                base_url="https://original.example/v1",
            )

    def fake_resolve_provider_endpoint(
        *args,
        **kwargs,
    ) -> ProviderEndpointConfig:
        del args, kwargs
        return ProviderEndpointConfig(
            provider_id="openai",
            base_url="https://override.example/v1",
            model="gpt-test",
            api_key="sk-test",
            require_api_key=True,
            is_local=False,
            chat_model="OpenAIChatModel",
        )

    def fake_load_provider_config(provider_id: str) -> dict[str, str]:
        return {
            "id": provider_id,
            "chat_model": "OpenAIChatModel",
        }

    def fake_get_instance() -> FakeManager:
        return FakeManager()

    def passthrough_retry_chat_model(runner: str) -> str:
        return runner

    monkeypatch.setattr(
        "_common.resolve_provider_endpoint",
        fake_resolve_provider_endpoint,
    )
    monkeypatch.setattr(
        "_common.load_provider_config",
        fake_load_provider_config,
    )
    monkeypatch.setattr(
        "_common.ProviderManager.get_instance",
        fake_get_instance,
    )
    monkeypatch.setattr(
        "_common.RetryChatModel",
        passthrough_retry_chat_model,
    )

    runtime = build_provider_benchmark_runtime("openai")

    assert runtime.base_url == "https://override.example/v1"
    assert runtime.model == "gpt-test"
    assert runtime.runner == "remote:gpt-test@https://override.example/v1"


def test_build_provider_benchmark_runtime_uses_local_factory_for_llamacpp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLocalProvider:
        def __init__(self) -> None:
            self.id = "llamacpp"
            self.base_url = ""
            self.chat_model = "OpenAIChatModel"

        def model_copy(self, deep: bool = True) -> "FakeLocalProvider":
            assert deep is True
            return FakeLocalProvider()

        def get_chat_model_instance(self, model_id: str) -> str:
            raise AssertionError("local factory path should be used")

    class FakeManager:
        def get_provider(self, provider_id: str) -> FakeLocalProvider | None:
            assert provider_id == "llamacpp"
            return FakeLocalProvider()

    def fake_resolve_provider_endpoint(
        *args,
        **kwargs,
    ) -> ProviderEndpointConfig:
        del args, kwargs
        return ProviderEndpointConfig(
            provider_id="llamacpp",
            base_url="",
            model="tiny-local",
            api_key="",
            require_api_key=False,
            is_local=True,
            chat_model="OpenAIChatModel",
        )

    def fake_load_provider_config(provider_id: str) -> dict[str, Any]:
        return {
            "id": provider_id,
            "chat_model": "OpenAIChatModel",
            "is_local": True,
        }

    def fake_get_instance() -> FakeManager:
        return FakeManager()

    def fake_create_local_chat_model(**kwargs) -> dict[str, Any]:
        return kwargs

    def passthrough_retry_chat_model(
        runner: dict[str, Any],
    ) -> dict[str, Any]:
        return runner

    monkeypatch.setattr(
        "_common.resolve_provider_endpoint",
        fake_resolve_provider_endpoint,
    )
    monkeypatch.setattr(
        "_common.load_provider_config",
        fake_load_provider_config,
    )
    monkeypatch.setattr(
        "_common.ProviderManager.get_instance",
        fake_get_instance,
    )
    monkeypatch.setattr(
        "_common.create_local_chat_model",
        fake_create_local_chat_model,
    )
    monkeypatch.setattr(
        "_common.RetryChatModel",
        passthrough_retry_chat_model,
    )

    runtime = build_provider_benchmark_runtime("llamacpp")

    assert runtime.model == "tiny-local"
    assert runtime.runner == {
        "model_id": "tiny-local",
        "stream": True,
        "generate_kwargs": {"max_tokens": None},
    }


def _write_provider_config(
    root: Path,
    *,
    provider_id: str,
    payload: dict,
    is_custom: bool = False,
) -> None:
    subdir = "custom" if is_custom else "builtin"
    path = root / subdir / f"{provider_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
