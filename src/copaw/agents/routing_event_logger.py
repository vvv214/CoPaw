# -*- coding: utf-8 -*-
"""Lightweight JSONL event sink for routing telemetry."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

ROUTING_EVENT_LOG_ENV = "COPAW_ROUTING_EVENT_LOG_PATH"


def resolve_default_routing_event_log_path() -> Path:
    env_value = os.getenv(ROUTING_EVENT_LOG_ENV, "").strip()
    if env_value:
        return Path(env_value).expanduser()

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "pyproject.toml"
        if candidate.exists():
            return parent / "logs" / "routing-events.jsonl"

    return Path("logs/routing-events.jsonl")


class RoutingEventSink:
    """Append-only JSONL sink used by routing decisions."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or resolve_default_routing_event_log_path()
        self._lock = Lock()

    @property
    def path(self) -> Path:
        return self._path

    def emit(self, event: dict[str, Any]) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(event, ensure_ascii=True)
            with self._lock:
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        except Exception:
            logger.warning(
                "Failed to write routing event to %s.",
                self._path,
                exc_info=True,
            )
