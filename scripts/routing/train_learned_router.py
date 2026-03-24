#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the learned router artifact from labeled benchmark data."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from _common import DEFAULT_ARTIFACT_PATH, build_dense_feature_row, load_jsonl


FEATURE_DIMENSIONS = 4096 + 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", default=str(DEFAULT_ARTIFACT_PATH))
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.12)
    parser.add_argument("--l2", type=float, default=0.02)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels = load_jsonl(Path(args.labels))
    rows = [build_dense_feature_row(record) for record in labels]
    targets = [1 if record["label"] == "cloud" else 0 for record in labels]

    coefficients, intercept, trainer_name = train_router(
        rows,
        targets,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    probabilities = [
        sigmoid(dot(coefficients, row) + intercept) for row in rows
    ]
    predictions = [
        1 if probability >= args.threshold else 0
        for probability in probabilities
    ]
    accuracy = sum(
        1 for prediction, target in zip(predictions, targets) if prediction == target
    ) / max(1, len(targets))
    cloud_targets = [target for target in targets if target == 1]
    cloud_hits = sum(
        1
        for prediction, target in zip(predictions, targets)
        if target == 1 and prediction == 1
    )
    cloud_recall = cloud_hits / max(1, len(cloud_targets))

    artifact = {
        "version": "copaw-learned-router-v1",
        "threshold": args.threshold,
        "n_features": 4096,
        "ngram_range": [1, 2],
        "coefficients": coefficients,
        "intercept": intercept,
        "structured_feature_names": [
            "prompt_chars",
            "message_count",
            "non_text",
            "recent_tool_context",
            "tool_choice",
            "freshness_flag",
            "strict_format_flag",
            "hard_rule_result",
        ],
        "metadata": {
            "trainer": trainer_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "training_cases": len(labels),
            "accuracy": round(accuracy, 3),
            "cloud_recall": round(cloud_recall, 3),
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"artifact_json={output_path}")
    print(f"trainer={trainer_name}")
    print(f"accuracy={accuracy:.3f}")
    print(f"cloud_recall={cloud_recall:.3f}")
    return 0


def train_router(
    rows: list[list[float]],
    targets: list[int],
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> tuple[list[float], float, str]:
    try:
        return train_router_with_sklearn(rows, targets)
    except Exception:
        coefficients, intercept = train_router_with_gradient_descent(
            rows,
            targets,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
        return coefficients, intercept, "batch_gradient_descent"


def train_router_with_sklearn(
    rows: list[list[float]],
    targets: list[int],
) -> tuple[list[float], float, str]:
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    model = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        random_state=0,
    )
    model.fit(rows, targets)
    coefficients = [
        float(value) for value in model.coef_[0].tolist()
    ]
    intercept = float(model.intercept_[0])
    return coefficients, intercept, "sklearn.LogisticRegression"


def train_router_with_gradient_descent(
    rows: list[list[float]],
    targets: list[int],
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> tuple[list[float], float]:
    coefficients = [0.0] * FEATURE_DIMENSIONS
    intercept = 0.0
    count = max(1, len(rows))
    for _ in range(epochs):
        gradients = [0.0] * FEATURE_DIMENSIONS
        intercept_gradient = 0.0
        for row, target in zip(rows, targets):
            probability = sigmoid(dot(coefficients, row) + intercept)
            error = probability - float(target)
            intercept_gradient += error
            for index, value in enumerate(row):
                if value != 0.0:
                    gradients[index] += error * value
        scale = 1.0 / count
        for index, gradient in enumerate(gradients):
            coefficients[index] -= learning_rate * (
                gradient * scale + l2 * coefficients[index]
            )
        intercept -= learning_rate * intercept_gradient * scale
    return coefficients, intercept


def dot(lhs: list[float], rhs: list[float]) -> float:
    return sum(left * right for left, right in zip(lhs, rhs))


def sigmoid(value: float) -> float:
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


if __name__ == "__main__":
    raise SystemExit(main())
