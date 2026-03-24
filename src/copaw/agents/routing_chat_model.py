# -*- coding: utf-8 -*-
"""ChatModel router for local/cloud model selection."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Literal, Type

from agentscope.formatter import FormatterBase
from agentscope.model import ChatModelBase
from agentscope.model._model_response import ChatResponse
from pydantic import BaseModel

from ..config.config import AgentsLLMRoutingConfig
from .routing_event_logger import RoutingEventSink
from .routing_learned_router import (
    LearnedRoutePrediction,
    RoutingSignals,
    build_feature_flags,
    load_learned_router_artifact,
    predict_route_with_artifact,
)

logger = logging.getLogger(__name__)


Route = Literal["local", "cloud"]

LONG_PROMPT_CHAR_THRESHOLD = 6000
LONG_CONVERSATION_MESSAGE_THRESHOLD = 24

FRESHNESS_KEYWORDS = (
    "latest",
    "current price",
    "current prices",
    "today",
    "this week",
    "right now",
    "as of ",
    "stock price",
    "stock prices",
    "market price",
    "market prices",
    "breaking news",
    "news today",
    "weather",
    "live score",
    "schedule today",
)

STRICT_FORMAT_KEYWORDS = (
    "json object",
    "valid json",
    "return json",
    "return only json",
    "respond with json",
    "only json",
    "output json",
)


@dataclass
class RoutingDecision:
    route: Route
    reasons: list[str] = field(default_factory=list)
    source: str = "rule_based"
    learned_score: float | None = None
    hard_override_reason: str | None = None


class RoutingPolicy:
    """Routing policy: hard guardrails first, softer rules as fallback."""

    def __init__(self, cfg: AgentsLLMRoutingConfig):
        self.cfg = cfg

    def hard_guardrail(
        self,
        *,
        signals: RoutingSignals,
    ) -> RoutingDecision | None:
        cloud_reasons: list[tuple[bool, str]] = [
            (signals.structured_output_requested, "structured_output"),
            (signals.strict_format_flag, "prompt:strict_format"),
            (signals.freshness_flag, "prompt:freshness_sensitive"),
            (signals.non_text, "user_content:non_text"),
            (signals.tool_choice == "required", "tool_choice:required"),
            (signals.recent_tool_context, "recent_tool_context"),
        ]
        for condition, reason in cloud_reasons:
            if condition:
                return RoutingDecision(
                    route="cloud",
                    reasons=[reason],
                    source="hard_guardrail",
                    hard_override_reason=reason,
                )
        return None

    def fallback_decide(
        self,
        *,
        signals: RoutingSignals,
    ) -> RoutingDecision:
        cloud_reasons: list[tuple[bool, str]] = [
            (
                signals.prompt_chars >= LONG_PROMPT_CHAR_THRESHOLD,
                f"prompt_chars>={LONG_PROMPT_CHAR_THRESHOLD}",
            ),
            (
                signals.message_count >= LONG_CONVERSATION_MESSAGE_THRESHOLD,
                f"message_count>={LONG_CONVERSATION_MESSAGE_THRESHOLD}",
            ),
        ]
        for condition, reason in cloud_reasons:
            if condition:
                return RoutingDecision(
                    route="cloud",
                    reasons=[reason],
                    source="rule_based",
                )

        if getattr(self.cfg, "mode", "local_first") == "cloud_first":
            return RoutingDecision(
                route="cloud",
                reasons=["mode:cloud_first"],
                source="rule_based",
            )

        return RoutingDecision(
            route="local",
            reasons=["mode:local_first"],
            source="rule_based",
        )


@dataclass(frozen=True)
class RoutingEndpoint:
    provider_id: str
    model_name: str
    formatter_family: Type[FormatterBase]
    loader: Callable[[], tuple[ChatModelBase, FormatterBase]]
    _model: ChatModelBase | None = field(default=None, init=False, repr=False)
    _formatter: FormatterBase | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._formatter is not None:
            return
        model, formatter = self.loader()
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_formatter", formatter)

    @property
    def model(self) -> ChatModelBase:
        self._ensure_loaded()
        assert self._model is not None
        return self._model

    @property
    def formatter(self) -> FormatterBase:
        self._ensure_loaded()
        assert self._formatter is not None
        return self._formatter


class RoutingChatModel(ChatModelBase):
    """A ChatModelBase that routes between local and cloud slots."""

    def __init__(
        self,
        *,
        local_endpoint: RoutingEndpoint,
        cloud_endpoint: RoutingEndpoint,
        routing_cfg: AgentsLLMRoutingConfig,
        request_context: dict[str, str] | None = None,
        routing_event_sink: RoutingEventSink | None = None,
    ) -> None:
        super().__init__(
            model_name="routing",
            stream=True,
        )
        self.local_endpoint = local_endpoint
        self.cloud_endpoint = cloud_endpoint
        self.routing_cfg = routing_cfg
        self.policy = RoutingPolicy(routing_cfg)
        self.request_context = dict(request_context or {})
        self.learned_router_artifact = load_learned_router_artifact()
        self.routing_event_sink = routing_event_sink or RoutingEventSink()

    async def __call__(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: Literal["auto", "none", "required"] | str | None = None,
        structured_model: Type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        request_id = self.request_context.get("request_id") or str(
            uuid.uuid4(),
        )
        started_at = time.perf_counter()
        signals = _build_routing_signals(
            messages=messages,
            tool_choice=tool_choice,
            structured_output_requested=structured_model is not None,
        )
        decision = self._decide_route(signals)
        fallback_used = False
        finish_reason = "error"

        (
            endpoint,
            decision,
            load_fallback_used,
        ) = self._load_endpoint_with_fallback(decision)
        fallback_used = fallback_used or load_fallback_used

        logger.debug(
            "LLM routing decision: route=%s provider=%s model=%s source=%s "
            "reasons=%s learned_score=%s",
            decision.route,
            endpoint.provider_id,
            endpoint.model_name,
            decision.source,
            ",".join(decision.reasons),
            (
                f"{decision.learned_score:.3f}"
                if decision.learned_score is not None
                else "none"
            ),
        )

        try:
            result = await endpoint.model(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                structured_model=structured_model,
                **kwargs,
            )
        except Exception:
            fallback = self._secondary_endpoint(decision.route)
            if fallback is None:
                self._emit_routing_event(
                    request_id=request_id,
                    decision=decision,
                    signals=signals,
                    fallback_used=fallback_used,
                    latency_ms=_elapsed_ms(started_at),
                    finish_reason=finish_reason,
                )
                raise

            fallback_route: Route = (
                "cloud" if decision.route == "local" else "local"
            )
            logger.warning(
                "Primary routed model invocation failed; retrying with %s "
                "(provider=%s, model=%s).",
                fallback_route,
                fallback.provider_id,
                fallback.model_name,
                exc_info=True,
            )
            fallback_used = True
            fallback_reason = (
                "fallback:local_call_error"
                if fallback_route == "cloud"
                else "fallback:cloud_call_error"
            )
            decision = RoutingDecision(
                route=fallback_route,
                reasons=[
                    *decision.reasons,
                    fallback_reason,
                ],
                source=decision.source,
                learned_score=decision.learned_score,
                hard_override_reason=decision.hard_override_reason,
            )
            result = await fallback.model(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                structured_model=structured_model,
                **kwargs,
            )

        if isinstance(result, AsyncGenerator):
            return self._wrap_stream(
                stream=result,
                request_id=request_id,
                signals=signals,
                decision=decision,
                fallback_used=fallback_used,
                started_at=started_at,
            )

        finish_reason = _extract_finish_reason(result)
        self._emit_routing_event(
            request_id=request_id,
            decision=decision,
            signals=signals,
            fallback_used=fallback_used,
            latency_ms=_elapsed_ms(started_at),
            finish_reason=finish_reason,
        )
        return result

    def _primary_endpoint(self, route: Route) -> RoutingEndpoint:
        return self.local_endpoint if route == "local" else self.cloud_endpoint

    def _secondary_endpoint(self, route: Route) -> RoutingEndpoint | None:
        fallback_route: Route = "cloud" if route == "local" else "local"
        fallback = self._primary_endpoint(fallback_route)
        primary = self._primary_endpoint(route)
        if (
            fallback.provider_id == primary.provider_id
            and fallback.model_name == primary.model_name
        ):
            return None
        return fallback

    def _decide_route(self, signals: RoutingSignals) -> RoutingDecision:
        hard_guardrail = self.policy.hard_guardrail(signals=signals)
        if hard_guardrail is not None:
            return hard_guardrail

        if self.learned_router_artifact is not None:
            try:
                prediction = predict_route_with_artifact(
                    self.learned_router_artifact,
                    signals,
                )
                return self._decision_from_prediction(prediction)
            except Exception:
                logger.warning(
                    "Learned router prediction failed; falling back to "
                    "rule-based routing.",
                    exc_info=True,
                )

        return self.policy.fallback_decide(signals=signals)

    def _decision_from_prediction(
        self,
        prediction: LearnedRoutePrediction,
    ) -> RoutingDecision:
        threshold = prediction.threshold
        score = prediction.p_cloud
        comparator = ">=" if prediction.route == "cloud" else "<"
        return RoutingDecision(
            route=prediction.route,
            reasons=[
                f"learned:p_cloud={score:.3f}",
                f"learned:p_cloud{comparator}{threshold:.2f}",
            ],
            source="learned_router",
            learned_score=score,
        )

    def _load_endpoint_with_fallback(
        self,
        decision: RoutingDecision,
    ) -> tuple[RoutingEndpoint, RoutingDecision, bool]:
        endpoint = self._primary_endpoint(decision.route)
        try:
            _ = endpoint.model
            return endpoint, decision, False
        except Exception:
            fallback = self._secondary_endpoint(decision.route)
            if fallback is None:
                raise

            fallback_route: Route = (
                "cloud" if decision.route == "local" else "local"
            )
            logger.warning(
                "Primary routed model load failed; falling back to %s "
                "(provider=%s, model=%s).",
                fallback_route,
                fallback.provider_id,
                fallback.model_name,
                exc_info=True,
            )
            _ = fallback.model
            return (
                fallback,
                RoutingDecision(
                    route=fallback_route,
                    reasons=[
                        *decision.reasons,
                        f"fallback:{decision.route}_load_error",
                    ],
                    source=decision.source,
                    learned_score=decision.learned_score,
                    hard_override_reason=decision.hard_override_reason,
                ),
                True,
            )

    async def _wrap_stream(
        self,
        *,
        stream: AsyncGenerator[ChatResponse, None],
        request_id: str,
        signals: RoutingSignals,
        decision: RoutingDecision,
        fallback_used: bool,
        started_at: float,
    ) -> AsyncGenerator[ChatResponse, None]:
        finish_reason = "stream_complete"
        failed_exc: Exception | None = None
        try:
            async for chunk in stream:
                finish_reason = _extract_finish_reason(chunk)
                yield chunk
        except Exception as exc:
            failed_exc = exc
            finish_reason = "error"
            raise
        finally:
            await stream.aclose()
            self._emit_routing_event(
                request_id=request_id,
                decision=decision,
                signals=signals,
                fallback_used=fallback_used,
                latency_ms=_elapsed_ms(started_at),
                finish_reason=finish_reason,
                error=(
                    f"{type(failed_exc).__name__}: {failed_exc}"
                    if failed_exc is not None
                    else None
                ),
            )

    def _emit_routing_event(
        self,
        *,
        request_id: str,
        decision: RoutingDecision,
        signals: RoutingSignals,
        fallback_used: bool,
        latency_ms: int,
        finish_reason: str,
        error: str | None = None,
    ) -> None:
        feature_flags = build_feature_flags(
            RoutingSignals(
                text=signals.text,
                prompt_chars=signals.prompt_chars,
                message_count=signals.message_count,
                non_text=signals.non_text,
                recent_tool_context=signals.recent_tool_context,
                tool_choice=signals.tool_choice,
                freshness_flag=signals.freshness_flag,
                strict_format_flag=signals.strict_format_flag,
                structured_output_requested=(
                    signals.structured_output_requested
                ),
                hard_rule_result=decision.hard_override_reason or "",
            ),
        )
        event: dict[str, Any] = {
            "request_id": request_id,
            "agent_id": self.request_context.get("agent_id", ""),
            "session_id": self.request_context.get("session_id", ""),
            "chosen_route": decision.route,
            "hard_override_reason": decision.hard_override_reason,
            "learned_score": decision.learned_score,
            "feature_flags": feature_flags,
            "local_slot": {
                "provider_id": self.local_endpoint.provider_id,
                "model_name": self.local_endpoint.model_name,
            },
            "cloud_slot": {
                "provider_id": self.cloud_endpoint.provider_id,
                "model_name": self.cloud_endpoint.model_name,
            },
            "fallback_used": fallback_used,
            "latency_ms": latency_ms,
            "finish_reason": finish_reason,
            "decision_source": decision.source,
            "reasons": decision.reasons,
        }
        if error is not None:
            event["error"] = error
        self.routing_event_sink.emit(event)


def _has_recent_tool_context(messages: list[dict]) -> bool:
    """Detect whether the current turn is in the middle of tool execution."""
    non_system_messages = [
        message for message in messages if message.get("role") != "system"
    ]
    if not non_system_messages:
        return False

    last_message = non_system_messages[-1]
    if last_message.get("role") == "tool":
        return True
    if last_message.get("role") == "assistant" and last_message.get(
        "tool_calls",
    ):
        return True

    if len(non_system_messages) < 2:
        return False

    previous_message = non_system_messages[-2]
    return bool(
        previous_message.get("role") == "assistant"
        and previous_message.get("tool_calls")
        and last_message.get("role") == "tool",
    )


def _build_routing_signals(
    *,
    messages: list[dict],
    tool_choice: Literal["auto", "none", "required"] | str | None,
    structured_output_requested: bool,
) -> RoutingSignals:
    text = " ".join(
        message["content"]
        for message in messages
        if message.get("role") == "user"
        and isinstance(message.get("content"), str)
    )
    has_non_text_user_content = any(
        message.get("role") == "user"
        and message.get("content") not in (None, "")
        and not isinstance(message.get("content"), str)
        for message in messages
    )
    return RoutingSignals(
        text=text,
        prompt_chars=len(text),
        message_count=len(messages),
        non_text=has_non_text_user_content,
        recent_tool_context=_has_recent_tool_context(messages),
        tool_choice=str(tool_choice or "auto"),
        freshness_flag=_looks_freshness_sensitive(text),
        strict_format_flag=_looks_strict_format_request(text),
        structured_output_requested=structured_output_requested,
    )


def _looks_freshness_sensitive(text: str) -> bool:
    normalized = text.lower()
    return any(keyword in normalized for keyword in FRESHNESS_KEYWORDS)


def _looks_strict_format_request(text: str) -> bool:
    normalized = text.lower()
    return any(keyword in normalized for keyword in STRICT_FORMAT_KEYWORDS)


def _extract_finish_reason(result: Any) -> str:
    finish_reason = getattr(result, "finish_reason", None)
    if isinstance(finish_reason, str) and finish_reason:
        return finish_reason

    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        metadata_reason = metadata.get("finish_reason")
        if isinstance(metadata_reason, str) and metadata_reason:
            return metadata_reason
    return "stop"


def _elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)
