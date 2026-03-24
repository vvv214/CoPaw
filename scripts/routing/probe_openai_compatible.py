#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe a benchmark endpoint with a small case suite."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from _common import (
    BenchmarkInvocationResult,
    build_openai_compatible_headers,
    build_provider_benchmark_runtime,
    invoke_provider_benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default="",
        help="Endpoint base URL, e.g. http://127.0.0.1:8102/v1",
    )
    parser.add_argument(
        "--provider-id",
        default="",
        help="Configured CoPaw provider id to resolve base URL and API key",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Bearer token if required by the endpoint",
    )
    parser.add_argument("--model", default="", help="Served model name")
    parser.add_argument(
        "--cases",
        required=True,
        help="JSONL file with probe cases",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSONL output path",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def load_cases(path: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


async def main() -> int:
    args = parse_args()
    runtime = None
    base_url = args.base_url
    api_key = args.api_key
    model = args.model

    if args.provider_id:
        try:
            runtime = build_provider_benchmark_runtime(
                args.provider_id,
                model=args.model or None,
                base_url=args.base_url or None,
                api_key=args.api_key or None,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        base_url = runtime.base_url
        model = runtime.model
        api_key = ""

    if not base_url:
        raise SystemExit("Missing --base-url or --provider-id.")
    if not model:
        raise SystemExit(
            "Missing --model or --provider-id with configured models.",
        )

    cases = load_cases(args.cases)
    output_path = Path(args.output) if args.output else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        if output_path is None:
            await _run_probe_cases(
                cases,
                client=client,
                runtime=runtime,
                base_url=base_url,
                api_key=api_key,
                model=model,
                temperature=args.temperature,
                timeout=args.timeout,
                output_handle=None,
            )
        else:
            with output_path.open("w", encoding="utf-8") as output_handle:
                await _run_probe_cases(
                    cases,
                    client=client,
                    runtime=runtime,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    temperature=args.temperature,
                    timeout=args.timeout,
                    output_handle=output_handle,
                )

    return 0


async def _run_probe_cases(
    cases: list[dict[str, Any]],
    *,
    client: httpx.AsyncClient,
    runtime,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    timeout: float,
    output_handle,
) -> None:
    for case in cases:
        started_at = time.perf_counter()
        record: dict[str, Any] = {
            "id": case["id"],
            "category": case.get("category", ""),
            "model": model,
            "endpoint": base_url,
        }
        try:
            if runtime is not None:
                response = await asyncio.wait_for(
                    invoke_provider_benchmark(
                        runtime,
                        messages=case["messages"],
                        temperature=temperature,
                        max_tokens=case.get("max_tokens", 256),
                    ),
                    timeout=timeout,
                )
            else:
                response = await _probe_http_endpoint(
                    client,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    case=case,
                    temperature=temperature,
                )

            elapsed = time.perf_counter() - started_at
            record.update(
                {
                    "ok": True,
                    "latency_s": round(elapsed, 3),
                    "usage": response.usage,
                    "finish_reason": response.finish_reason,
                    "response_text": response.response_text,
                },
            )
        except httpx.HTTPStatusError as exc:
            elapsed = time.perf_counter() - started_at
            record.update(
                {
                    "ok": False,
                    "latency_s": round(elapsed, 3),
                    "error": f"HTTP {exc.response.status_code}",
                    "error_body": exc.response.text,
                },
            )
        except Exception as exc:  # pragma: no cover - probe script guardrail
            elapsed = time.perf_counter() - started_at
            record.update(
                {
                    "ok": False,
                    "latency_s": round(elapsed, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )

        line = json.dumps(record, ensure_ascii=True)
        print(line)
        if output_handle is not None:
            output_handle.write(line + "\n")
            output_handle.flush()


async def _probe_http_endpoint(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    api_key: str,
    model: str,
    case: dict[str, Any],
    temperature: float,
) -> BenchmarkInvocationResult:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": case["messages"],
        "temperature": temperature,
        "max_tokens": case.get("max_tokens", 256),
    }
    headers = build_openai_compatible_headers(
        base_url=base_url,
        api_key=api_key,
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


def _summarize_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(part.strip() for part in parts if part.strip())
    return ""


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
