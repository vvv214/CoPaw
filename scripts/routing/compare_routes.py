#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare local and cloud OpenAI-compatible routes on one case pack."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from _common import (
    BenchmarkInvocationResult,
    DEFAULT_CASES_PATH,
    DEFAULT_CLOUD_API_KEY_ENV,
    DEFAULT_CLOUD_BASE_URL,
    DEFAULT_CLOUD_MODEL,
    DEFAULT_LOCAL_API_KEY,
    DEFAULT_LOCAL_BASE_URL,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_OUTPUT_ROOT,
    build_provider_benchmark_runtime,
    build_openai_compatible_headers,
    invoke_provider_benchmark,
    load_jsonl,
    score_case_response,
    write_jsonl,
)


@dataclass(frozen=True)
class EndpointConfig:
    route: str
    kind: str
    base_url: str
    model: str
    api_key: str
    provider_id: str = ""
    runtime: Any | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH))
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
    )
    parser.add_argument("--run-name", default="")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--local-provider-id", default="")
    parser.add_argument("--local-base-url", default=DEFAULT_LOCAL_BASE_URL)
    parser.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--local-api-key", default=DEFAULT_LOCAL_API_KEY)
    parser.add_argument("--cloud-provider-id", default="")
    parser.add_argument("--cloud-base-url", default="")
    parser.add_argument("--cloud-model", default="")
    parser.add_argument("--cloud-api-key", default="")
    parser.add_argument(
        "--cloud-api-key-env",
        default=DEFAULT_CLOUD_API_KEY_ENV,
    )
    parser.add_argument("--control-provider-id", default="")
    parser.add_argument("--control-base-url", default="")
    parser.add_argument("--control-model", default="")
    parser.add_argument("--control-api-key", default="")
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    endpoints = [
        build_endpoint_config(
            route="local",
            provider_id=args.local_provider_id,
            base_url=args.local_base_url,
            model=args.local_model,
            api_key=args.local_api_key,
        ),
        build_endpoint_config(
            route="cloud",
            provider_id=args.cloud_provider_id,
            base_url=args.cloud_base_url or DEFAULT_CLOUD_BASE_URL,
            model=args.cloud_model or DEFAULT_CLOUD_MODEL,
            api_key=(
                args.cloud_api_key or os.getenv(args.cloud_api_key_env, "")
            ),
            require_api_key_env=(
                args.cloud_api_key_env if not args.cloud_provider_id else ""
            ),
        ),
    ]
    if args.control_provider_id or (
        args.control_base_url and args.control_model
    ):
        endpoints.append(
            build_endpoint_config(
                route="control",
                provider_id=args.control_provider_id,
                base_url=args.control_base_url,
                model=args.control_model,
                api_key=args.control_api_key,
            ),
        )

    cases = load_jsonl(Path(args.cases))
    output_dir = build_output_dir(
        Path(args.output_root),
        run_name=args.run_name,
    )
    compare_records: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        for case in cases:
            results = await asyncio.gather(
                *[
                    probe_endpoint(
                        client,
                        endpoint=endpoint,
                        case=case,
                        temperature=args.temperature,
                        timeout=args.timeout,
                    )
                    for endpoint in endpoints
                ],
            )
            compare_records.append(
                {
                    "id": case["id"],
                    "bucket": case["bucket"],
                    "gold_route": case["gold_route"],
                    "allow_local": case["allow_local"],
                    "messages": case["messages"],
                    "rubric": case["rubric"],
                    "max_tokens": case.get("max_tokens", 256),
                    "hard_guardrail_reason": case.get(
                        "hard_guardrail_reason",
                        "",
                    ),
                    "results": {result["route"]: result for result in results},
                },
            )

    compare_path = output_dir / "compare.jsonl"
    summary_path = output_dir / "summary.json"
    write_jsonl(compare_path, compare_records)
    summary_path.write_text(
        json.dumps(build_summary(compare_records), ensure_ascii=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"compare_jsonl={compare_path}")
    print(f"summary_json={summary_path}")
    return 0


async def probe_endpoint(
    client: httpx.AsyncClient,
    *,
    endpoint: EndpointConfig,
    case: dict[str, Any],
    temperature: float,
    timeout: float,
) -> dict[str, Any]:
    started_at = asyncio.get_running_loop().time()
    record: dict[str, Any] = {
        "route": endpoint.route,
        "base_url": endpoint.base_url,
        "model": endpoint.model,
    }
    try:
        if endpoint.kind == "provider_model":
            assert endpoint.runtime is not None
            invocation = await asyncio.wait_for(
                invoke_provider_benchmark(
                    endpoint.runtime,
                    messages=case["messages"],
                    temperature=temperature,
                    max_tokens=case.get("max_tokens", 256),
                ),
                timeout=timeout,
            )
        else:
            invocation = await probe_http_endpoint(
                client,
                endpoint=endpoint,
                case=case,
                temperature=temperature,
            )

        rubric_result = score_case_response(
            case,
            response_text=invocation.response_text,
            finish_reason=invocation.finish_reason,
        )
        record.update(
            {
                "ok": True,
                "latency_s": round(
                    asyncio.get_running_loop().time() - started_at,
                    3,
                ),
                "finish_reason": invocation.finish_reason,
                "usage": invocation.usage,
                "response_text": invocation.response_text,
                "truncation": rubric_result["truncation"],
                "score": rubric_result["score"],
                "passed": rubric_result["passed"],
                "checks": rubric_result["checks"],
            },
        )
    except Exception as exc:  # pragma: no cover - endpoint/network guardrail
        record.update(
            {
                "ok": False,
                "latency_s": round(
                    asyncio.get_running_loop().time() - started_at,
                    3,
                ),
                "finish_reason": "error",
                "usage": None,
                "response_text": "",
                "truncation": False,
                "score": 0.0,
                "passed": False,
                "checks": [],
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
    return record


def build_endpoint_config(
    *,
    route: str,
    provider_id: str,
    base_url: str,
    model: str,
    api_key: str,
    require_api_key_env: str = "",
) -> EndpointConfig:
    if provider_id:
        try:
            runtime = build_provider_benchmark_runtime(
                provider_id,
                model=model or None,
                base_url=base_url or None,
                api_key=api_key or None,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        return EndpointConfig(
            route=route,
            kind="provider_model",
            base_url=runtime.base_url,
            model=runtime.model,
            api_key="",
            provider_id=runtime.provider_id,
            runtime=runtime,
        )

    if require_api_key_env and not api_key:
        raise SystemExit(
            "Missing cloud API key. Pass --cloud-api-key or set "
            f"{require_api_key_env}.",
        )

    return EndpointConfig(
        route=route,
        kind="openai_compatible_http",
        base_url=base_url,
        model=model,
        api_key=api_key,
    )


async def probe_http_endpoint(
    client: httpx.AsyncClient,
    *,
    endpoint: EndpointConfig,
    case: dict[str, Any],
    temperature: float,
) -> BenchmarkInvocationResult:
    url = endpoint.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": endpoint.model,
        "messages": case["messages"],
        "temperature": temperature,
        "max_tokens": case.get("max_tokens", 256),
    }
    headers = build_openai_compatible_headers(
        base_url=endpoint.base_url,
        api_key=endpoint.api_key,
    )
    response = await client.post(url, json=payload, headers=headers)
    response.raise_for_status()
    body = response.json()
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return BenchmarkInvocationResult(
        response_text=_summarize_content(message.get("content")),
        finish_reason=choice.get("finish_reason") or "stop",
        usage=body.get("usage"),
    )


def build_output_dir(root: Path, *, run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    suffix = run_name.strip() or "compare"
    output_dir = (
        root / datetime.now().strftime("%Y-%m-%d") / (f"{timestamp}-{suffix}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    buckets = sorted({record["bucket"] for record in records})
    routes = sorted(
        {route for record in records for route in record["results"]},
    )
    summary: dict[str, Any] = {"cases": len(records), "routes": {}}
    for route in routes:
        route_records = [
            record["results"][route]
            for record in records
            if route in record["results"]
        ]
        summary["routes"][route] = {
            "avg_latency_s": round(
                sum(result["latency_s"] for result in route_records)
                / max(1, len(route_records)),
                3,
            ),
            "pass_rate": round(
                sum(1 for result in route_records if result["passed"])
                / max(1, len(route_records)),
                3,
            ),
            "truncation_rate": round(
                sum(1 for result in route_records if result["truncation"])
                / max(1, len(route_records)),
                3,
            ),
        }

    summary["buckets"] = {}
    for bucket in buckets:
        bucket_records = [
            record for record in records if record["bucket"] == bucket
        ]
        summary["buckets"][bucket] = {
            route: round(
                sum(
                    1
                    for record in bucket_records
                    if record["results"].get(route, {}).get("passed")
                )
                / max(1, len(bucket_records)),
                3,
            )
            for route in routes
        }
    return summary


def _summarize_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part.strip() for part in parts if part.strip())
    return ""


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
