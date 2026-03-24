# -*- coding: utf-8 -*-
"""Tests for routing benchmark script helpers."""

from __future__ import annotations

# pylint: disable=wrong-import-position

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "routing"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import (  # noqa: E402
    CODING_DASHSCOPE_BASE_URL,
    DASHSCOPE_BASE_URL,
    build_openai_compatible_headers,
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
