#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Report learned-router metrics from labeled benchmark artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from _common import (
    DEFAULT_ARTIFACT_PATH,
    build_signals_for_case,
    load_jsonl,
)
from copaw.agents.routing_learned_router import (
    LearnedRouterArtifact,
    predict_route_with_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT_PATH))
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels = load_jsonl(Path(args.labels))
    artifact_path = Path(args.artifact)
    with artifact_path.open("r", encoding="utf-8") as handle:
        artifact = LearnedRouterArtifact.from_dict(json.load(handle))

    predicted_routes: list[str] = []
    cloud_needed_labels = [
        record for record in labels if record["label"] == "cloud"
    ]
    cheap_local = [record for record in labels if record["bucket"] == "cheap-local"]
    inference_latencies_ms: list[float] = []
    hard_guardrail_total = 0
    hard_guardrail_misses = 0
    cloud_hits = 0
    for record in labels:
        hard_guardrail = str(record.get("hard_guardrail_reason") or "")
        if hard_guardrail:
            predicted_route = "cloud"
            hard_guardrail_total += 1
        else:
            started_at = time.perf_counter()
            predicted_route = predict_route_with_artifact(
                artifact,
                build_signals_for_case(record),
            ).route
            inference_latencies_ms.append(
                (time.perf_counter() - started_at) * 1000.0
            )
        predicted_routes.append(predicted_route)
        if hard_guardrail and predicted_route != "cloud":
            hard_guardrail_misses += 1
        if record["label"] == "cloud" and predicted_route == "cloud":
            cloud_hits += 1

    local_latency = mean_latency(cheap_local, route="local")
    cloud_latency = mean_latency(cheap_local, route="cloud")
    latency_advantage = 0.0
    if local_latency > 0 and cloud_latency > 0:
        latency_advantage = 1.0 - (local_latency / cloud_latency)

    report = {
        "cases": len(labels),
        "hard_guardrail_no_local_rate": (
            1.0
            if hard_guardrail_total == 0
            else 1.0 - (hard_guardrail_misses / hard_guardrail_total)
        ),
        "cloud_needed_recall": (
            0.0
            if not cloud_needed_labels
            else cloud_hits / len(cloud_needed_labels)
        ),
        "cheap_local_latency_advantage": latency_advantage,
        "learned_router_median_overhead_ms": (
            statistics.median(inference_latencies_ms)
            if inference_latencies_ms
            else 0.0
        ),
        "artifact": str(artifact_path),
        "artifact_metadata": artifact.metadata,
    }
    output_path = (
        Path(args.output)
        if args.output
        else Path(args.labels).with_name("report.json")
    )
    output_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"report_json={output_path}")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


def mean_latency(records: list[dict], *, route: str) -> float:
    values = [
        float(record["results"][route]["latency_s"])
        for record in records
        if route in record["results"]
    ]
    return sum(values) / max(1, len(values))


if __name__ == "__main__":
    raise SystemExit(main())
