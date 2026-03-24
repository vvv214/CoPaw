#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe an OpenAI-compatible chat endpoint with a small case suite."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        required=True,
        help="Endpoint base URL, e.g. http://127.0.0.1:8102/v1",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Bearer token if required by the endpoint",
    )
    parser.add_argument("--model", required=True, help="Served model name")
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


def build_request(
    case: dict[str, Any],
    model: str,
    temperature: float,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": case["messages"],
        "temperature": temperature,
        "max_tokens": case.get("max_tokens", 256),
    }


def post_json(
    url: str,
    payload: dict[str, Any],
    api_key: str,
    timeout: float,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def summarize_response(result: dict[str, Any]) -> str:
    choices = result.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(part.strip() for part in parts if part.strip())
    return ""


def main() -> int:
    args = parse_args()
    cases = load_cases(args.cases)
    output_path = Path(args.output) if args.output else None
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    if output_path is None:
        _run_probe_cases(
            cases,
            endpoint=endpoint,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
            timeout=args.timeout,
            output_handle=None,
        )
    else:
        with output_path.open("w", encoding="utf-8") as output_handle:
            _run_probe_cases(
                cases,
                endpoint=endpoint,
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                temperature=args.temperature,
                timeout=args.timeout,
                output_handle=output_handle,
            )

    return 0


def _run_probe_cases(
    cases: list[dict[str, Any]],
    *,
    endpoint: str,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    timeout: float,
    output_handle,
) -> None:
    for case in cases:
        payload = build_request(case, model, temperature)
        started_at = time.perf_counter()
        record: dict[str, Any] = {
            "id": case["id"],
            "category": case.get("category", ""),
            "model": model,
            "endpoint": base_url,
        }
        try:
            response = post_json(
                endpoint,
                payload,
                api_key=api_key,
                timeout=timeout,
            )
            elapsed = time.perf_counter() - started_at
            record.update(
                {
                    "ok": True,
                    "latency_s": round(elapsed, 3),
                    "usage": response.get("usage"),
                    "finish_reason": (
                        (response.get("choices") or [{}])[0].get(
                            "finish_reason",
                        )
                    ),
                    "response_text": summarize_response(response),
                },
            )
        except urllib.error.HTTPError as exc:
            elapsed = time.perf_counter() - started_at
            record.update(
                {
                    "ok": False,
                    "latency_s": round(elapsed, 3),
                    "error": f"HTTP {exc.code}",
                    "error_body": exc.read().decode(
                        "utf-8",
                        errors="replace",
                    ),
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


if __name__ == "__main__":
    sys.exit(main())
