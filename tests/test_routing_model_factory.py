# -*- coding: utf-8 -*-
from types import SimpleNamespace

import pytest
from agentscope.model import OpenAIChatModel

import copaw.agents.model_factory as model_factory
from copaw.agents.routing_chat_model import RoutingChatModel
from copaw.config.config import AgentsLLMRoutingConfig
from copaw.providers.models import (
    ModelSlotConfig,
    ProviderSettings,
    ProvidersData,
    ResolvedModelConfig,
)


def _providers_data(*, active_llm: ModelSlotConfig) -> ProvidersData:
    return ProvidersData(
        providers={
            "openai": ProviderSettings(
                base_url="https://api.openai.com/v1",
                api_key="sk-test",
            ),
            "aliyun-codingplan": ProviderSettings(
                base_url="https://coding.dashscope.aliyuncs.com/v1",
                api_key="sk-sp-test",
            ),
        },
        active_llm=active_llm,
    )


def _patch_common_routing_mocks(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, str, bool]]:
    created: list[tuple[str, str, bool]] = []

    class FakeChatModel:
        def __init__(self, provider_id: str, model_name: str, is_local: bool):
            self.provider_id = provider_id
            self.model_name = model_name
            self.is_local = is_local
            self.stream = True

        async def __call__(self, *args, **kwargs):
            return SimpleNamespace(
                provider_id=self.provider_id,
                model_name=self.model_name,
                is_local=self.is_local,
                args=args,
                kwargs=kwargs,
            )

    def fake_create_model_instance_for_provider(
        llm_cfg,
        provider_id,
        *,
        providers_data,  # noqa: ARG001
    ):
        created.append((provider_id, llm_cfg.model, llm_cfg.is_local))
        return (
            FakeChatModel(
                provider_id=provider_id,
                model_name=llm_cfg.model,
                is_local=llm_cfg.is_local,
            ),
            OpenAIChatModel,
        )

    monkeypatch.setattr(
        model_factory,
        "_create_model_instance_for_provider",
        fake_create_model_instance_for_provider,
    )
    monkeypatch.setattr(
        model_factory,
        "_create_formatter_instance",
        lambda chat_model_class: SimpleNamespace(
            formatter_for=chat_model_class.__name__,
        ),
    )
    monkeypatch.setattr(
        model_factory,
        "_create_formatter_from_family",
        lambda formatter_family: SimpleNamespace(
            formatter_for=formatter_family.__name__,
        ),
    )
    return created


def test_create_model_and_formatter_uses_routing_with_active_cloud_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routing_cfg = AgentsLLMRoutingConfig(
        enabled=True,
        mode="local_first",
        local=ModelSlotConfig(
            provider_id="llamacpp",
            model="Qwen2.5-0.5B-Instruct-GGUF",
        ),
        cloud=None,
    )
    providers_data = _providers_data(
        active_llm=ModelSlotConfig(provider_id="openai", model="gpt-5"),
    )
    created = _patch_common_routing_mocks(monkeypatch)

    monkeypatch.setattr(
        model_factory,
        "load_config",
        lambda: SimpleNamespace(
            agents=SimpleNamespace(llm_routing=routing_cfg),
        ),
    )
    monkeypatch.setattr(
        model_factory,
        "load_providers_json",
        lambda: providers_data,
    )
    monkeypatch.setattr(
        model_factory,
        "get_active_llm_config",
        lambda: (_ for _ in ()).throw(
            AssertionError("routing path should be used"),
        ),
    )

    model, formatter = model_factory.create_model_and_formatter()

    assert isinstance(model, RoutingChatModel)
    assert model.local_endpoint.provider_id == "llamacpp"
    assert model.local_endpoint.model_name == "Qwen2.5-0.5B-Instruct-GGUF"
    assert model.cloud_endpoint.provider_id == "openai"
    assert model.cloud_endpoint.model_name == "gpt-5"
    assert formatter.formatter_for == "OpenAIChatFormatter"
    assert created == []


@pytest.mark.asyncio
async def test_create_model_and_formatter_loads_only_selected_cloud_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routing_cfg = AgentsLLMRoutingConfig(
        enabled=True,
        mode="cloud_first",
        local=ModelSlotConfig(provider_id="mlx", model="Qwen3-4B"),
        cloud=ModelSlotConfig(
            provider_id="aliyun-codingplan",
            model="qwen3.5-plus",
        ),
    )
    providers_data = _providers_data(
        active_llm=ModelSlotConfig(provider_id="mlx", model="Qwen3-4B"),
    )
    created = _patch_common_routing_mocks(monkeypatch)

    monkeypatch.setattr(
        model_factory,
        "load_config",
        lambda: SimpleNamespace(
            agents=SimpleNamespace(llm_routing=routing_cfg),
        ),
    )
    monkeypatch.setattr(
        model_factory,
        "load_providers_json",
        lambda: providers_data,
    )
    monkeypatch.setattr(
        model_factory,
        "get_active_llm_config",
        lambda: (_ for _ in ()).throw(
            AssertionError("routing path should be used"),
        ),
    )

    model, _ = model_factory.create_model_and_formatter()

    assert isinstance(model, RoutingChatModel)
    assert created == []

    response = await model(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
    )

    assert response.provider_id == "aliyun-codingplan"
    assert response.model_name == "qwen3.5-plus"
    assert created == [
        ("aliyun-codingplan", "qwen3.5-plus", False),
    ]


@pytest.mark.asyncio
async def test_create_model_and_formatter_loads_local_route_on_first_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routing_cfg = AgentsLLMRoutingConfig(
        enabled=True,
        mode="local_first",
        local=ModelSlotConfig(
            provider_id="llamacpp",
            model="Qwen2.5-0.5B-Instruct-GGUF",
        ),
        cloud=None,
    )
    providers_data = _providers_data(
        active_llm=ModelSlotConfig(provider_id="openai", model="gpt-5"),
    )
    created = _patch_common_routing_mocks(monkeypatch)

    monkeypatch.setattr(
        model_factory,
        "load_config",
        lambda: SimpleNamespace(
            agents=SimpleNamespace(llm_routing=routing_cfg),
        ),
    )
    monkeypatch.setattr(
        model_factory,
        "load_providers_json",
        lambda: providers_data,
    )
    monkeypatch.setattr(
        model_factory,
        "get_active_llm_config",
        lambda: (_ for _ in ()).throw(
            AssertionError("routing path should be used"),
        ),
    )

    model, _ = model_factory.create_model_and_formatter()

    assert isinstance(model, RoutingChatModel)
    assert created == []

    response = await model(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
    )

    assert response.provider_id == "llamacpp"
    assert response.model_name == "Qwen2.5-0.5B-Instruct-GGUF"
    assert created == [
        ("llamacpp", "Qwen2.5-0.5B-Instruct-GGUF", True),
    ]


def test_create_model_and_formatter_uses_explicit_cloud_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routing_cfg = AgentsLLMRoutingConfig(
        enabled=True,
        mode="cloud_first",
        local=ModelSlotConfig(provider_id="mlx", model="Qwen3-4B"),
        cloud=ModelSlotConfig(
            provider_id="aliyun-codingplan",
            model="qwen3.5-plus",
        ),
    )
    providers_data = _providers_data(
        active_llm=ModelSlotConfig(provider_id="mlx", model="Qwen3-4B"),
    )
    created = _patch_common_routing_mocks(monkeypatch)

    monkeypatch.setattr(
        model_factory,
        "load_config",
        lambda: SimpleNamespace(
            agents=SimpleNamespace(llm_routing=routing_cfg),
        ),
    )
    monkeypatch.setattr(
        model_factory,
        "load_providers_json",
        lambda: providers_data,
    )
    monkeypatch.setattr(
        model_factory,
        "get_active_llm_config",
        lambda: (_ for _ in ()).throw(
            AssertionError("routing path should be used"),
        ),
    )

    model, _ = model_factory.create_model_and_formatter()

    assert isinstance(model, RoutingChatModel)
    assert model.local_endpoint.provider_id == "mlx"
    assert model.cloud_endpoint.provider_id == "aliyun-codingplan"
    assert model.routing_cfg.mode == "cloud_first"
    assert created == []


def test_create_model_and_formatter_uses_active_model_when_routing_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routing_cfg = AgentsLLMRoutingConfig(enabled=False)
    resolved = ResolvedModelConfig(
        model="gpt-5-mini",
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
        is_local=False,
    )

    monkeypatch.setattr(
        model_factory,
        "load_config",
        lambda: SimpleNamespace(
            agents=SimpleNamespace(llm_routing=routing_cfg),
        ),
    )
    monkeypatch.setattr(
        model_factory,
        "load_providers_json",
        lambda: _providers_data(
            active_llm=ModelSlotConfig(
                provider_id="openai",
                model="gpt-5-mini",
            ),
        ),
    )
    monkeypatch.setattr(model_factory, "get_active_llm_config", lambda: resolved)
    monkeypatch.setattr(
        model_factory,
        "_create_model_instance",
        lambda llm_cfg: (SimpleNamespace(model_name=llm_cfg.model), OpenAIChatModel),
    )
    monkeypatch.setattr(
        model_factory,
        "_create_formatter_instance",
        lambda chat_model_class: SimpleNamespace(
            formatter_for=chat_model_class.__name__,
        ),
    )
    monkeypatch.setattr(
        model_factory,
        "_create_formatter_from_family",
        lambda formatter_family: SimpleNamespace(
            formatter_for=formatter_family.__name__,
        ),
    )

    model, formatter = model_factory.create_model_and_formatter()

    assert not isinstance(model, RoutingChatModel)
    assert model.model_name == "gpt-5-mini"
    assert formatter.formatter_for == "OpenAIChatModel"
