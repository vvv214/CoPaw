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
    DEFAULT_CASES_PATH,
    DEFAULT_CLOUD_API_KEY_ENV,
    DEFAULT_CLOUD_BASE_URL,
    DEFAULT_CLOUD_MODEL,
    DEFAULT_LOCAL_API_KEY,
    DEFAULT_LOCAL_BASE_URL,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_OUTPUT_ROOT,
    load_jsonl,
    score_case_response,
    write_jsonl,
)


@dataclass(frozen=True)
class EndpointConfig:
    route: str
    base_url: str
    model: str
    api_key: str


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
    parser.add_argument("--local-base-url", default=DEFAULT_LOCAL_BASE_URL)
    parser.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--local-api-key", default=DEFAULT_LOCAL_API_KEY)
    parser.add_argument("--cloud-base-url", default=DEFAULT_CLOUD_BASE_URL)
    parser.add_argument("--cloud-model", default=DEFAULT_CLOUD_MODEL)
    parser.add_argument("--cloud-api-key", default="")
    parser.add_argument(
        "--cloud-api-key-env",
        default=DEFAULT_CLOUD_API_KEY_ENV,
    )
    parser.add_argument("--control-base-url", default="")
    parser.add_argument("--control-model", default="")
    parser.add_argument("--control-api-key", default="")
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    cloud_api_key = args.cloud_api_key or os.getenv(args.cloud_api_key_env, "")
    if not cloud_api_key:
        raise SystemExit(
            "Missing cloud API key. Pass --cloud-api-key or set "
            f"{args.cloud_api_key_env}.",
        )

    endpoints = [
        EndpointConfig(
            route="local",
            base_url=args.local_base_url,
            model=args.local_model,
            api_key=args.local_api_key,
        ),
        EndpointConfig(
            route="cloud",
            base_url=args.cloud_base_url,
            model=args.cloud_model,
            api_key=cloud_api_key,
        ),
    ]
    if args.control_base_url and args.control_model:
        endpoints.append(
            EndpointConfig(
                route="control",
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
                    )
                    for endpoint in endpoints
                ]
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
                    "results": {
                        result["route"]: result for result in results
                    },
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
) -> dict[str, Any]:
    url = endpoint.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": endpoint.model,
        "messages": case["messages"],
        "temperature": temperature,
        "max_tokens": case.get("max_tokens", 256),
    }
    headers = {"Content-Type": "application/json"}
    if endpoint.api_key:
        headers["Authorization"] = f"Bearer {endpoint.api_key}"

    started_at = asyncio.get_running_loop().time()
    record: dict[str, Any] = {
        "route": endpoint.route,
        "base_url": endpoint.base_url,
        "model": endpoint.model,
    }
    try:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        body = response.json()
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        response_text = _summarize_content(message.get("content"))
        finish_reason = choice.get("finish_reason")
        rubric_result = score_case_response(
            case,
            response_text=response_text,
            finish_reason=finish_reason,
        )
        record.update(
            {
                "ok": True,
                "latency_s": round(
                    asyncio.get_running_loop().time() - started_at,
                    3,
                ),
                "finish_reason": finish_reason,
                "usage": body.get("usage"),
                "response_text": response_text,
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


def build_output_dir(root: Path, *, run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    suffix = run_name.strip() or "compare"
    output_dir = root / datetime.now().strftime("%Y-%m-%d") / (
        f"{timestamp}-{suffix}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    buckets = sorted({record["bucket"] for record in records})
    routes = sorted(
        {
            route
            for record in records
            for route in record["results"]
        }
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
