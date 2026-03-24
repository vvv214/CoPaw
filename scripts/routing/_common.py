#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for routing benchmark scripts."""

# pylint: disable=wrong-import-position

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from copaw.agents.routing_learned_router import (
    RoutingSignals,
    build_structured_feature_values,
    build_text_feature_counts,
)
from copaw.constant import SECRET_DIR

DEFAULT_CASES_PATH = SCRIPT_DIR / "benchmark_cases_v1.jsonl"
DEFAULT_ARTIFACT_PATH = SCRIPT_DIR / "artifacts" / "learned_router_v1.json"
DEFAULT_OUTPUT_ROOT = Path("/bigtemp/nkp2mr/shared-benchmarks/copaw-routing")
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:18102/v1"
DEFAULT_LOCAL_MODEL = "qwen2.5-32b-awq-local"
DEFAULT_LOCAL_API_KEY = "copaw-local"
DEFAULT_CLOUD_BASE_URL = "https://coding.dashscope.aliyuncs.com/v1"
DEFAULT_CLOUD_MODEL = "qwen3.5-plus"
DEFAULT_CLOUD_API_KEY_ENV = "DASHSCOPE_API_KEY"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
CODING_DASHSCOPE_BASE_URL = "https://coding.dashscope.aliyuncs.com/v1"
PROVIDER_CONFIG_ROOT = SECRET_DIR / "providers"

FRESHNESS_KEYWORDS = (
    "latest",
    "current price",
    "current prices",
    "today",
    "this week",
    "right now",
    "as of ",
    "stock price",
    "stock prices",
    "market price",
    "market prices",
    "breaking news",
    "news today",
    "weather",
    "live score",
    "schedule today",
)

STRICT_FORMAT_KEYWORDS = (
    "json object",
    "valid json",
    "return json",
    "return only json",
    "respond with json",
    "only json",
    "output json",
)

JSON_FENCE_PATTERN = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$",
    flags=re.DOTALL,
)
BULLET_PATTERN = re.compile(r"^\s*(?:[-*]|\d+\.)\s+", flags=re.MULTILINE)


@dataclass(frozen=True)
class ProviderEndpointConfig:
    provider_id: str
    base_url: str
    model: str
    api_key: str
    require_api_key: bool
    is_local: bool


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def build_openai_compatible_headers(
    *,
    base_url: str,
    api_key: str,
) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    metadata = json.dumps(
        {
            "agentType": "CoPaw",
            "deployType": "UnKnown",
            "moduleCode": "model",
            "agentCode": "UnKnown",
        },
        ensure_ascii=False,
    )
    normalized_base_url = base_url.rstrip("/")
    if normalized_base_url == DASHSCOPE_BASE_URL:
        headers["x-dashscope-agentapp"] = metadata
    elif normalized_base_url == CODING_DASHSCOPE_BASE_URL:
        headers["X-DashScope-Cdpl"] = metadata
    return headers


def resolve_provider_endpoint(
    provider_id: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> ProviderEndpointConfig:
    config = load_provider_config(provider_id)
    resolved_base_url = (base_url or str(config.get("base_url") or "")).strip()
    if not resolved_base_url:
        raise ValueError(
            f"Provider '{provider_id}' does not have a configured base URL.",
        )

    resolved_api_key = (api_key or str(config.get("api_key") or "")).strip()
    require_api_key = bool(config.get("require_api_key", True))
    if require_api_key and not resolved_api_key:
        raise ValueError(
            f"Provider '{provider_id}' requires an API key but none is set.",
        )

    resolved_model = (model or default_model_for_provider(config)).strip()
    if not resolved_model:
        raise ValueError(
            f"Provider '{provider_id}' does not have any configured models.",
        )

    return ProviderEndpointConfig(
        provider_id=provider_id,
        base_url=resolved_base_url,
        model=resolved_model,
        api_key=resolved_api_key,
        require_api_key=require_api_key,
        is_local=bool(config.get("is_local", False)),
    )


def load_provider_config(provider_id: str) -> dict[str, Any]:
    for path in provider_config_paths(provider_id):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    raise FileNotFoundError(
        f"Provider configuration for '{provider_id}' was not found.",
    )


def provider_config_paths(provider_id: str) -> list[Path]:
    return [
        PROVIDER_CONFIG_ROOT / "builtin" / f"{provider_id}.json",
        PROVIDER_CONFIG_ROOT / "custom" / f"{provider_id}.json",
    ]


def default_model_for_provider(config: dict[str, Any]) -> str:
    for field_name in ("extra_models", "models"):
        items = config.get(field_name) or []
        for item in items:
            model_id = str(item.get("id") or "").strip()
            if model_id:
                return model_id
    return ""


def latest_user_text(messages: list[dict[str, Any]]) -> str:
    texts = [
        message["content"]
        for message in messages
        if message.get("role") == "user"
        and isinstance(message.get("content"), str)
    ]
    return texts[-1] if texts else ""


def build_signals_for_case(case: dict[str, Any]) -> RoutingSignals:
    messages = case["messages"]
    text = " ".join(
        message["content"]
        for message in messages
        if message.get("role") == "user"
        and isinstance(message.get("content"), str)
    )
    return RoutingSignals(
        text=text,
        prompt_chars=len(text),
        message_count=len(messages),
        non_text=any(
            message.get("role") == "user"
            and message.get("content") not in (None, "")
            and not isinstance(message.get("content"), str)
            for message in messages
        ),
        recent_tool_context=has_recent_tool_context(messages),
        tool_choice=str(case.get("tool_choice") or "auto"),
        freshness_flag=looks_freshness_sensitive(text),
        strict_format_flag=looks_strict_format_request(text),
        structured_output_requested=bool(
            case.get("hard_guardrail_reason") == "structured_output",
        ),
        hard_rule_result=str(case.get("hard_guardrail_reason") or ""),
    )


def build_dense_feature_row(case: dict[str, Any]) -> list[float]:
    signals = build_signals_for_case(case)
    text_features = build_text_feature_counts(signals.text)
    structured_values = build_structured_feature_values(signals)
    total_dimensions = 4096 + len(structured_values)
    row = [0.0] * total_dimensions
    for index, value in text_features.items():
        row[index] = value
    for offset, feature_name in enumerate(structured_values):
        row[4096 + offset] = structured_values[feature_name]
    return row


def score_case_response(
    case: dict[str, Any],
    *,
    response_text: str,
    finish_reason: str | None,
) -> dict[str, Any]:
    rubric = dict(case.get("rubric") or {})
    normalized = normalize_text(response_text)
    checks: list[dict[str, Any]] = []

    def record_check(name: str, passed: bool, detail: Any) -> None:
        checks.append(
            {
                "name": name,
                "passed": passed,
                "detail": detail,
            },
        )

    if rubric.get("must_contain"):
        needles = [str(item).lower() for item in rubric["must_contain"]]
        passed = all(needle in normalized for needle in needles)
        record_check("must_contain", passed, needles)

    if rubric.get("must_contain_any"):
        needles = [str(item).lower() for item in rubric["must_contain_any"]]
        passed = any(needle in normalized for needle in needles)
        record_check("must_contain_any", passed, needles)

    if rubric.get("must_not_contain"):
        needles = [str(item).lower() for item in rubric["must_not_contain"]]
        passed = all(needle not in normalized for needle in needles)
        record_check("must_not_contain", passed, needles)

    if rubric.get("regex_all"):
        patterns = [str(item) for item in rubric["regex_all"]]
        passed = all(
            re.search(pattern, response_text, flags=re.IGNORECASE) is not None
            for pattern in patterns
        )
        record_check("regex_all", passed, patterns)

    if rubric.get("bullet_min") is not None:
        bullet_count = len(BULLET_PATTERN.findall(response_text))
        minimum = int(rubric["bullet_min"])
        record_check("bullet_min", bullet_count >= minimum, bullet_count)

    if rubric.get("json_keys"):
        parsed = try_parse_json_object(response_text)
        required_keys = [str(item) for item in rubric["json_keys"]]
        passed = isinstance(parsed, dict) and all(
            key in parsed for key in required_keys
        )
        record_check("json_keys", passed, required_keys)

    if rubric.get("keyword_counts"):
        keyword_counts = dict(rubric["keyword_counts"])
        passed = True
        details: dict[str, int] = {}
        for keyword, minimum in keyword_counts.items():
            count = normalized.count(str(keyword).lower())
            details[str(keyword)] = count
            if count < int(minimum):
                passed = False
        record_check("keyword_counts", passed, details)

    truncation = str(finish_reason or "").lower() == "length"
    if not rubric.get("allow_truncation", False):
        record_check("not_truncated", not truncation, finish_reason or "stop")

    if not checks:
        checks.append({"name": "default", "passed": True, "detail": "n/a"})

    passed_checks = sum(1 for check in checks if check["passed"])
    score = round(passed_checks / len(checks), 3)
    return {
        "score": score,
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "truncation": truncation,
    }


def try_parse_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    match = JSON_FENCE_PATTERN.match(cleaned)
    if match:
        cleaned = match.group(1).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def looks_freshness_sensitive(text: str) -> bool:
    normalized = text.lower()
    return any(keyword in normalized for keyword in FRESHNESS_KEYWORDS)


def looks_strict_format_request(text: str) -> bool:
    normalized = text.lower()
    return any(keyword in normalized for keyword in STRICT_FORMAT_KEYWORDS)


def has_recent_tool_context(messages: list[dict[str, Any]]) -> bool:
    non_system_messages = [
        message for message in messages if message.get("role") != "system"
    ]
    if not non_system_messages:
        return False
    last_message = non_system_messages[-1]
    if last_message.get("role") == "tool":
        return True
    if last_message.get("role") == "assistant" and last_message.get(
        "tool_calls",
    ):
        return True
    if len(non_system_messages) < 2:
        return False
    previous_message = non_system_messages[-2]
    return bool(
        previous_message.get("role") == "assistant"
        and previous_message.get("tool_calls")
        and last_message.get("role") == "tool",
    )
