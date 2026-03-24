#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate compare results into routing labels."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import load_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", required=True)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    compare_path = Path(args.compare)
    output_path = (
        Path(args.output)
        if args.output
        else compare_path.with_name("labels.jsonl")
    )
    compare_records = load_jsonl(compare_path)
    label_records = [label_record(record) for record in compare_records]
    write_jsonl(output_path, label_records)
    print(f"labels_jsonl={output_path}")
    return 0


def label_record(record: dict) -> dict:
    local = dict(record["results"].get("local") or {})
    cloud = dict(record["results"].get("cloud") or {})
    hard_guardrail_reason = str(record.get("hard_guardrail_reason") or "")

    if hard_guardrail_reason:
        label = "cloud"
        label_reason = f"hard_guardrail:{hard_guardrail_reason}"
    elif not local.get("ok", False):
        label = "cloud"
        label_reason = "local_error"
    elif local.get("truncation", False):
        label = "cloud"
        label_reason = "local_truncation"
    elif (not local.get("passed", False)) and cloud.get("passed", False):
        label = "cloud"
        label_reason = "local_fail_cloud_pass"
    else:
        label = "local"
        label_reason = "local_ok"

    return {
        "id": record["id"],
        "bucket": record["bucket"],
        "gold_route": record["gold_route"],
        "allow_local": record["allow_local"],
        "messages": record["messages"],
        "rubric": record["rubric"],
        "max_tokens": record["max_tokens"],
        "hard_guardrail_reason": hard_guardrail_reason,
        "label": label,
        "label_reason": label_reason,
        "results": record["results"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
