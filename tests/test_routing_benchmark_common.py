# -*- coding: utf-8 -*-
"""Tests for routing benchmark script helpers."""

from __future__ import annotations

# pylint: disable=wrong-import-position

import json
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "routing"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import (  # noqa: E402
    CODING_DASHSCOPE_BASE_URL,
    DASHSCOPE_BASE_URL,
    build_openai_compatible_headers,
    resolve_provider_endpoint,
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
