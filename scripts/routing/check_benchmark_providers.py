#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect configured providers for routing benchmark readiness."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import httpx

from _common import (
    PROVIDER_CONFIG_ROOT,
    build_openai_compatible_headers,
    default_model_for_provider,
    load_provider_config,
)


LOCAL_HOST_PREFIXES = (
    "http://127.0.0.1",
    "https://127.0.0.1",
    "http://localhost",
    "https://localhost",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider-id",
        action="append",
        default=[],
        help="Specific provider id to inspect. Repeatable.",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Run a tiny chat completion probe for ready providers.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="HTTP timeout in seconds for probe mode.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSONL output path.",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    provider_ids = list(dict.fromkeys(args.provider_id))
    records = load_provider_records(provider_ids=provider_ids or None)

    if args.probe:
        async with httpx.AsyncClient(timeout=args.timeout) as client:
            records = [
                await maybe_probe_provider(client, record)
                for record in records
            ]

    for record in records:
        print(json.dumps(record, ensure_ascii=True))

    summary = build_summary(records)
    print(json.dumps({"summary": summary}, ensure_ascii=True))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    return 0


def load_provider_records(
    *,
    provider_ids: list[str] | None,
) -> list[dict[str, Any]]:
    selected = set(provider_ids or [])
    records: list[dict[str, Any]] = []
    for path in sorted(_provider_config_files()):
        provider_id = path.stem
        if selected and provider_id not in selected:
            continue
        config = load_provider_config(provider_id)
        records.append(classify_provider_record(config))
    return records


def classify_provider_record(config: dict[str, Any]) -> dict[str, Any]:
    provider_id = str(config.get("id") or "")
    base_url = str(config.get("base_url") or "")
    chat_model = str(config.get("chat_model") or "")
    require_api_key = bool(config.get("require_api_key", True))
    has_api_key = bool(str(config.get("api_key") or "").strip())
    is_local = bool(config.get("is_local", False))
    runtime_scope = classify_runtime_scope(
        provider_id=provider_id,
        base_url=base_url,
        is_local=is_local,
    )
    openai_compatible = chat_model == "OpenAIChatModel"
    default_model = default_model_for_provider(config)
    benchmark_role = classify_benchmark_role(
        runtime_scope=runtime_scope,
        openai_compatible=openai_compatible,
    )
    automation_safe = provider_id != "aliyun-codingplan"
    status, note = classify_status(
        provider_id=provider_id,
        base_url=base_url,
        require_api_key=require_api_key,
        has_api_key=has_api_key,
        openai_compatible=openai_compatible,
        automation_safe=automation_safe,
        default_model=default_model,
    )
    return {
        "provider_id": provider_id,
        "name": str(config.get("name") or provider_id),
        "chat_model": chat_model,
        "base_url": base_url,
        "runtime_scope": runtime_scope,
        "benchmark_role": benchmark_role,
        "openai_compatible": openai_compatible,
        "automation_safe": automation_safe,
        "require_api_key": require_api_key,
        "has_api_key": has_api_key,
        "default_model": default_model,
        "status": status,
        "note": note,
    }


def classify_runtime_scope(
    *,
    provider_id: str,
    base_url: str,
    is_local: bool,
) -> str:
    normalized = base_url.rstrip("/").lower()
    if is_local or provider_id == "ollama":
        return "local"
    if any(normalized.startswith(prefix) for prefix in LOCAL_HOST_PREFIXES):
        return "local"
    return "cloud"


def classify_benchmark_role(
    *,
    runtime_scope: str,
    openai_compatible: bool,
) -> str:
    if not openai_compatible:
        return "unsupported"
    if runtime_scope == "cloud":
        return "cloud_candidate"
    return "local_candidate"


def classify_status(
    *,
    provider_id: str,
    base_url: str,
    require_api_key: bool,
    has_api_key: bool,
    openai_compatible: bool,
    automation_safe: bool,
    default_model: str,
) -> tuple[str, str]:
    status = "ready_for_probe"
    note = "Provider is ready for automated probe."
    if not openai_compatible:
        status = "unsupported_chat_model"
        note = (
            "Current routing benchmark scripts only support "
            "OpenAI-compatible endpoints."
        )
    elif not base_url:
        status = "missing_base_url"
        note = "Provider base URL is not configured."
    elif require_api_key and not has_api_key:
        status = "missing_api_key"
        note = "Provider requires an API key."
    elif not default_model:
        status = "missing_model"
        note = "Provider does not have any configured model."
    elif not automation_safe:
        status = "interactive_only"
        note = (
            "Configured, but not suitable as the default automated benchmark "
            "baseline."
        )
    elif provider_id == "ollama":
        status = "local_only"
        note = "Local OpenAI-compatible provider."

    return status, note


async def maybe_probe_provider(
    client: httpx.AsyncClient,
    record: dict[str, Any],
) -> dict[str, Any]:
    if record["status"] not in {
        "ready_for_probe",
        "local_only",
        "interactive_only",
    }:
        return record

    payload = {
        "model": record["default_model"],
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "temperature": 0.0,
        "max_tokens": 4,
    }
    url = record["base_url"].rstrip("/") + "/chat/completions"
    headers = build_openai_compatible_headers(
        base_url=record["base_url"],
        api_key=_provider_api_key(record["provider_id"]),
    )
    started_at = asyncio.get_running_loop().time()
    updated = dict(record)
    try:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        body = response.json()
        choice = (body.get("choices") or [{}])[0]
        updated.update(
            {
                "probe_ok": True,
                "probe_latency_s": round(
                    asyncio.get_running_loop().time() - started_at,
                    3,
                ),
                "probe_finish_reason": choice.get("finish_reason"),
            },
        )
    except Exception as exc:  # pragma: no cover - network guardrail
        updated.update(
            {
                "probe_ok": False,
                "probe_latency_s": round(
                    asyncio.get_running_loop().time() - started_at,
                    3,
                ),
                "probe_error": f"{type(exc).__name__}: {exc}",
            },
        )
    return updated


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    ready_cloud_candidates = 0
    probed_ok = 0
    for record in records:
        status = str(record.get("status") or "")
        status_counts[status] = status_counts.get(status, 0) + 1
        if (
            record.get("benchmark_role") == "cloud_candidate"
            and status == "ready_for_probe"
        ):
            ready_cloud_candidates += 1
        if record.get("probe_ok") is True:
            probed_ok += 1
    return {
        "providers": len(records),
        "status_counts": status_counts,
        "ready_cloud_candidates": ready_cloud_candidates,
        "probed_ok": probed_ok,
    }


def _provider_config_files() -> list[Path]:
    paths: list[Path] = []
    for subdir in ("builtin", "custom"):
        directory = PROVIDER_CONFIG_ROOT / subdir
        if directory.exists():
            paths.extend(directory.glob("*.json"))
    return paths


def _provider_api_key(provider_id: str) -> str:
    return str(load_provider_config(provider_id).get("api_key") or "").strip()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
