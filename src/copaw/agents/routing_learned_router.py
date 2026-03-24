# -*- coding: utf-8 -*-
"""Portable learned-router artifact loading and prediction."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

Route = Literal["local", "cloud"]

DEFAULT_ROUTING_THRESHOLD = 0.65
DEFAULT_TEXT_FEATURE_DIMENSIONS = 4096
DEFAULT_NGRAM_RANGE = (1, 2)
LEARNED_ROUTER_ARTIFACT_ENV = "COPAW_ROUTING_LEARNED_ROUTER_ARTIFACT"

STRUCTURED_FEATURE_NAMES = (
    "prompt_chars",
    "message_count",
    "non_text",
    "recent_tool_context",
    "tool_choice",
    "freshness_flag",
    "strict_format_flag",
    "hard_rule_result",
)

TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True)
class RoutingSignals:
    text: str
    prompt_chars: int
    message_count: int
    non_text: bool
    recent_tool_context: bool
    tool_choice: str
    freshness_flag: bool
    strict_format_flag: bool
    structured_output_requested: bool
    hard_rule_result: str = ""


@dataclass(frozen=True)
class LearnedRoutePrediction:
    route: Route
    p_cloud: float
    threshold: float


@dataclass(frozen=True)
class LearnedRouterArtifact:
    version: str
    threshold: float
    n_features: int
    ngram_range: tuple[int, int]
    coefficients: tuple[float, ...]
    intercept: float
    structured_feature_names: tuple[str, ...]
    metadata: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LearnedRouterArtifact":
        n_features = int(payload["n_features"])
        coefficients = tuple(float(value) for value in payload["coefficients"])
        structured_feature_names = tuple(
            str(name) for name in payload["structured_feature_names"]
        )
        expected_dimensions = n_features + len(structured_feature_names)
        if len(coefficients) != expected_dimensions:
            raise ValueError(
                "Learned router coefficient count does not match feature "
                f"layout: expected {expected_dimensions}, "
                f"got {len(coefficients)}",
            )
        ngram_range_raw = payload.get("ngram_range", DEFAULT_NGRAM_RANGE)
        if not isinstance(ngram_range_raw, list | tuple):
            raise ValueError("ngram_range must be a length-2 array.")
        if len(ngram_range_raw) != 2:
            raise ValueError("ngram_range must contain exactly two integers.")
        return cls(
            version=str(payload.get("version", "copaw-learned-router-v1")),
            threshold=float(
                payload.get("threshold", DEFAULT_ROUTING_THRESHOLD),
            ),
            n_features=n_features,
            ngram_range=(int(ngram_range_raw[0]), int(ngram_range_raw[1])),
            coefficients=coefficients,
            intercept=float(payload.get("intercept", 0.0)),
            structured_feature_names=structured_feature_names,
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "threshold": self.threshold,
            "n_features": self.n_features,
            "ngram_range": list(self.ngram_range),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "structured_feature_names": list(self.structured_feature_names),
            "metadata": self.metadata,
        }


def resolve_default_learned_router_artifact_path() -> Path:
    env_value = os.getenv(LEARNED_ROUTER_ARTIFACT_ENV, "").strip()
    if env_value:
        return Path(env_value).expanduser()

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "pyproject.toml"
        if candidate.exists():
            return (
                parent / "scripts" / "routing" / "artifacts"
                / "learned_router_v1.json"
            )

    return Path("scripts/routing/artifacts/learned_router_v1.json")


@lru_cache(maxsize=4)
def _load_artifact_cached(path_str: str) -> LearnedRouterArtifact:
    path = Path(path_str)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return LearnedRouterArtifact.from_dict(payload)


def clear_learned_router_artifact_cache() -> None:
    """Clear the in-process artifact cache.

    Useful in tests that swap artifact files or environment variables.
    """

    _load_artifact_cached.cache_clear()


def load_learned_router_artifact(
    path: Path | None = None,
) -> LearnedRouterArtifact | None:
    artifact_path = (path or resolve_default_learned_router_artifact_path())
    if not artifact_path.exists():
        return None
    try:
        return _load_artifact_cached(str(artifact_path.resolve()))
    except Exception:
        logger.warning(
            "Failed to load learned router artifact from %s.",
            artifact_path,
            exc_info=True,
        )
        return None


def predict_route_with_artifact(
    artifact: LearnedRouterArtifact,
    signals: RoutingSignals,
) -> LearnedRoutePrediction:
    linear_score = artifact.intercept
    for index, value in build_text_feature_counts(
        signals.text,
        n_features=artifact.n_features,
        ngram_range=artifact.ngram_range,
    ).items():
        linear_score += artifact.coefficients[index] * value

    structured_values = build_structured_feature_values(signals)
    offset = artifact.n_features
    for structured_index, feature_name in enumerate(
        artifact.structured_feature_names,
    ):
        linear_score += (
            artifact.coefficients[offset + structured_index]
            * structured_values.get(feature_name, 0.0)
        )

    p_cloud = _sigmoid(linear_score)
    return LearnedRoutePrediction(
        route="cloud" if p_cloud >= artifact.threshold else "local",
        p_cloud=p_cloud,
        threshold=artifact.threshold,
    )


def build_text_feature_counts(
    text: str,
    *,
    n_features: int = DEFAULT_TEXT_FEATURE_DIMENSIONS,
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
) -> dict[int, float]:
    features: dict[int, float] = {}
    tokens = _tokenize_text(text)
    if not tokens:
        return features

    min_n, max_n = ngram_range
    for n in range(min_n, max_n + 1):
        for index in range(0, len(tokens) - n + 1):
            ngram = " ".join(tokens[index : index + n])
            feature_index = _stable_hash(ngram) % n_features
            features[feature_index] = features.get(feature_index, 0.0) + 1.0
    return features


def build_structured_feature_values(
    signals: RoutingSignals,
) -> dict[str, float]:
    return {
        "prompt_chars": min(float(signals.prompt_chars), 20000.0) / 1000.0,
        "message_count": min(float(signals.message_count), 100.0),
        "non_text": 1.0 if signals.non_text else 0.0,
        "recent_tool_context": (
            1.0 if signals.recent_tool_context else 0.0
        ),
        "tool_choice": _encode_tool_choice(signals.tool_choice),
        "freshness_flag": 1.0 if signals.freshness_flag else 0.0,
        "strict_format_flag": 1.0 if signals.strict_format_flag else 0.0,
        "hard_rule_result": 1.0 if signals.hard_rule_result else 0.0,
    }


def build_feature_flags(signals: RoutingSignals) -> dict[str, Any]:
    return {
        "prompt_chars": signals.prompt_chars,
        "message_count": signals.message_count,
        "non_text": signals.non_text,
        "recent_tool_context": signals.recent_tool_context,
        "tool_choice": signals.tool_choice,
        "freshness_flag": signals.freshness_flag,
        "strict_format_flag": signals.strict_format_flag,
        "structured_output_requested": signals.structured_output_requested,
        "hard_rule_result": signals.hard_rule_result,
    }


def _tokenize_text(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def _encode_tool_choice(tool_choice: str) -> float:
    normalized = (tool_choice or "").strip().lower()
    if normalized == "required":
        return 1.0
    if normalized == "none":
        return -1.0
    return 0.0


def _stable_hash(text: str) -> int:
    digest = hashlib.blake2b(
        text.encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)
