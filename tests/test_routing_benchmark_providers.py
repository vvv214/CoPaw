# -*- coding: utf-8 -*-
"""Tests for benchmark provider readiness helpers."""

from __future__ import annotations

# pylint: disable=wrong-import-position

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "routing"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from check_benchmark_providers import (  # noqa: E402
    classify_provider_record,
    classify_runtime_scope,
)


def test_classify_runtime_scope_detects_ollama_as_local() -> None:
    assert (
        classify_runtime_scope(
            provider_id="ollama",
            base_url="http://localhost:11434/v1",
            is_local=False,
        )
        == "local"
    )


def test_classify_provider_record_marks_codingplan_interactive_only() -> None:
    record = classify_provider_record(
        {
            "id": "aliyun-codingplan",
            "name": "Aliyun Coding Plan",
            "chat_model": "OpenAIChatModel",
            "base_url": "https://coding.dashscope.aliyuncs.com/v1",
            "api_key": "sk-sp-test",
            "require_api_key": True,
            "is_local": False,
            "models": [{"id": "qwen3.5-plus", "name": "Qwen3.5 Plus"}],
            "extra_models": [],
        },
    )

    assert record["runtime_scope"] == "cloud"
    assert record["benchmark_role"] == "cloud_candidate"
    assert record["status"] == "interactive_only"


def test_classify_provider_record_marks_openai_missing_api_key() -> None:
    record = classify_provider_record(
        {
            "id": "openai",
            "name": "OpenAI",
            "chat_model": "OpenAIChatModel",
            "base_url": "https://api.openai.com/v1",
            "api_key": "",
            "require_api_key": True,
            "is_local": False,
            "models": [{"id": "gpt-5-mini", "name": "GPT-5 Mini"}],
            "extra_models": [],
        },
    )

    assert record["status"] == "missing_api_key"
    assert record["benchmark_role"] == "cloud_candidate"


def test_classify_provider_record_marks_ollama_local_only() -> None:
    record = classify_provider_record(
        {
            "id": "ollama",
            "name": "Ollama",
            "chat_model": "OpenAIChatModel",
            "base_url": "http://localhost:11434/v1",
            "api_key": "",
            "require_api_key": False,
            "is_local": False,
            "models": [],
            "extra_models": [{"id": "qwen3:latest", "name": "Qwen3"}],
        },
    )

    assert record["runtime_scope"] == "local"
    assert record["benchmark_role"] == "local_candidate"
    assert record["status"] == "local_only"


def test_classify_provider_record_marks_anthropic_supported() -> None:
    record = classify_provider_record(
        {
            "id": "anthropic",
            "name": "Anthropic",
            "chat_model": "AnthropicChatModel",
            "base_url": "https://api.anthropic.com",
            "api_key": "",
            "require_api_key": True,
            "is_local": False,
            "models": [{"id": "claude-sonnet-4-5", "name": "Claude"}],
            "extra_models": [],
        },
    )

    assert record["benchmark_supported"] is True
    assert record["benchmark_role"] == "cloud_candidate"
    assert record["status"] == "missing_api_key"


def test_classify_provider_record_marks_unknown_chat_model_unsupported() -> (
    None
):
    record = classify_provider_record(
        {
            "id": "custom-router",
            "name": "Custom Router",
            "chat_model": "RouterChatModel",
            "base_url": "https://example.com",
            "api_key": "sk-test",
            "require_api_key": True,
            "is_local": False,
            "models": [{"id": "router-1", "name": "Router 1"}],
            "extra_models": [],
        },
    )

    assert record["benchmark_supported"] is False
    assert record["benchmark_role"] == "unsupported"
    assert record["status"] == "unsupported_provider"
