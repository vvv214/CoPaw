import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { MenuProps } from "antd";
import { Button, Dropdown, message } from "antd";
import {
  ApartmentOutlined,
  CheckOutlined,
  DownOutlined,
} from "@ant-design/icons";
import { useTranslation } from "react-i18next";
import { providerApi } from "../../api/modules/provider";
import type {
  ActiveModelsInfo,
  LLMRoutingConfig,
  LLMRoutingMode,
  ModelSlotConfig,
  ProviderInfo,
} from "../../api/types/provider";
import styles from "./index.module.less";

const ROUTING_LOCAL_FIRST = "routing:local_first";
const ROUTING_CLOUD_FIRST = "routing:cloud_first";
const ROUTING_LOCAL_SLOT_PREFIX = "routing-slot:local:";
const ROUTING_CLOUD_SLOT_PREFIX = "routing-slot:cloud:";

interface SlotOption {
  key: string;
  slot: ModelSlotConfig;
  label: string;
}

function hasConfiguredSlot(
  slot?: ModelSlotConfig | null,
): slot is ModelSlotConfig {
  return Boolean(slot?.provider_id && slot?.model);
}

function encodeModelValue(slot: ModelSlotConfig): string {
  return `model:${encodeURIComponent(slot.provider_id)}:${encodeURIComponent(
    slot.model,
  )}`;
}

function decodeModelValue(value: string): ModelSlotConfig {
  const [, providerId = "", model = ""] = value.split(":");
  return {
    provider_id: decodeURIComponent(providerId),
    model: decodeURIComponent(model),
  };
}

function encodeRoutingSlotValue(
  kind: "local" | "cloud",
  slot: ModelSlotConfig,
): string {
  const prefix =
    kind === "local" ? ROUTING_LOCAL_SLOT_PREFIX : ROUTING_CLOUD_SLOT_PREFIX;
  return `${prefix}${encodeURIComponent(slot.provider_id)}:${encodeURIComponent(
    slot.model,
  )}`;
}

function decodeRoutingSlotValue(value: string): {
  kind: "local" | "cloud";
  slot: ModelSlotConfig;
} {
  const kind = value.startsWith(ROUTING_LOCAL_SLOT_PREFIX) ? "local" : "cloud";
  const prefix =
    kind === "local" ? ROUTING_LOCAL_SLOT_PREFIX : ROUTING_CLOUD_SLOT_PREFIX;
  const payload = value.slice(prefix.length);
  const [providerId = "", model = ""] = payload.split(":");
  return {
    kind,
    slot: {
      provider_id: decodeURIComponent(providerId),
      model: decodeURIComponent(model),
    },
  };
}

function isEligibleProvider(provider: ProviderInfo): boolean {
  const hasModels = provider.models.length + provider.extra_models.length > 0;
  if (!hasModels) return false;
  if (provider.id === "ollama") return !!provider.base_url;
  if (provider.is_local) return true;
  if (provider.is_custom) return !!provider.base_url;
  if (provider.require_api_key === false) return !!provider.base_url;
  return !!provider.api_key;
}

function getModelName(
  providers: ProviderInfo[],
  slot?: ModelSlotConfig | null,
): string {
  if (!hasConfiguredSlot(slot)) {
    return "";
  }
  const provider = providers.find((item) => item.id === slot.provider_id);
  const model = [
    ...(provider?.models ?? []),
    ...(provider?.extra_models ?? []),
  ].find((item) => item.id === slot.model);
  return model?.name ?? slot.model;
}

function getModelMenuLabel(
  providers: ProviderInfo[],
  slot: ModelSlotConfig,
): string {
  const provider = providers.find((item) => item.id === slot.provider_id);
  const providerName = provider?.name ?? slot.provider_id;
  const modelName = getModelName(providers, slot);
  return `${providerName} / ${modelName}`;
}

function buildSlotOptions(
  providers: ProviderInfo[],
  kind: "local" | "cloud",
): SlotOption[] {
  return providers
    .filter(isEligibleProvider)
    .filter((provider) =>
      kind === "local" ? provider.is_local : !provider.is_local,
    )
    .flatMap((provider) =>
      [...provider.models, ...provider.extra_models].map((model) => {
        const slot = { provider_id: provider.id, model: model.id };
        return {
          key: encodeRoutingSlotValue(kind, slot),
          slot,
          label: `${provider.name} / ${model.name}`,
        };
      }),
    );
}

export default function ModelRoutingSelector() {
  const { t } = useTranslation();
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [activeModels, setActiveModels] = useState<ActiveModelsInfo | null>(
    null,
  );
  const [routingConfig, setRoutingConfig] = useState<LLMRoutingConfig | null>(
    null,
  );
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [open, setOpen] = useState(false);
  const keepOpenRef = useRef(false);

  const reload = useCallback(async () => {
    setLoading(true);
    try {
      const [providerData, activeData, routingData] = await Promise.all([
        providerApi.listProviders(),
        providerApi.getActiveModels(),
        providerApi.getLlmRoutingConfig(),
      ]);
      setProviders(providerData);
      setActiveModels(activeData);
      setRoutingConfig(routingData);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  const eligibleProviders = useMemo(
    () => providers.filter(isEligibleProvider),
    [providers],
  );

  const modelOptions = useMemo(
    () =>
      eligibleProviders.flatMap((provider) =>
        [...provider.models, ...provider.extra_models].map((model) => ({
          key: encodeModelValue({
            provider_id: provider.id,
            model: model.id,
          }),
          label: `${provider.name} / ${model.name}`,
        })),
      ),
    [eligibleProviders],
  );

  const localOptions = useMemo(
    () => buildSlotOptions(providers, "local"),
    [providers],
  );
  const cloudOptions = useMemo(
    () => buildSlotOptions(providers, "cloud"),
    [providers],
  );

  const effectiveLocalSlot = useMemo(() => {
    if (hasConfiguredSlot(routingConfig?.local)) {
      return routingConfig.local;
    }
    return localOptions[0]?.slot ?? null;
  }, [localOptions, routingConfig]);

  const effectiveCloudSlot = useMemo(() => {
    if (hasConfiguredSlot(routingConfig?.cloud)) {
      return routingConfig.cloud;
    }
    return cloudOptions[0]?.slot ?? null;
  }, [cloudOptions, routingConfig]);

  const selectedKey = useMemo(() => {
    if (routingConfig?.enabled) {
      return routingConfig.mode === "cloud_first"
        ? ROUTING_CLOUD_FIRST
        : ROUTING_LOCAL_FIRST;
    }
    if (hasConfiguredSlot(activeModels?.active_llm)) {
      return encodeModelValue(activeModels.active_llm);
    }
    return "";
  }, [activeModels, routingConfig]);

  const selectedLocalKey = hasConfiguredSlot(effectiveLocalSlot)
    ? encodeRoutingSlotValue("local", effectiveLocalSlot)
    : "";
  const selectedCloudKey = hasConfiguredSlot(effectiveCloudSlot)
    ? encodeRoutingSlotValue("cloud", effectiveCloudSlot)
    : "";

  const canEnableRouting =
    hasConfiguredSlot(effectiveLocalSlot) &&
    hasConfiguredSlot(effectiveCloudSlot);

  const keepDropdownOpen = useCallback(() => {
    keepOpenRef.current = true;
    setOpen(true);
  }, []);

  const handleEnableRouting = useCallback(
    async (mode: LLMRoutingMode) => {
      if (!canEnableRouting) {
        message.warning(t("chatModelSelector.configureRoutingFirst"));
        keepDropdownOpen();
        return;
      }
      await providerApi.setLlmRoutingConfig({
        enabled: true,
        mode,
        local: effectiveLocalSlot!,
        cloud: effectiveCloudSlot!,
      });
      message.success(
        t(
          mode === "cloud_first"
            ? "chatModelSelector.routingCloudFirstEnabled"
            : "chatModelSelector.routingLocalFirstEnabled",
        ),
      );
      keepDropdownOpen();
    },
    [
      canEnableRouting,
      effectiveCloudSlot,
      effectiveLocalSlot,
      keepDropdownOpen,
      t,
    ],
  );

  const handleSelectRoutingSlot = useCallback(
    async (kind: "local" | "cloud", slot: ModelSlotConfig) => {
      await providerApi.setLlmRoutingConfig({
        enabled: routingConfig?.enabled ?? false,
        mode: routingConfig?.mode ?? "local_first",
        local: kind === "local" ? slot : effectiveLocalSlot ?? slot,
        cloud: kind === "cloud" ? slot : effectiveCloudSlot ?? slot,
      });
      message.success(t("chatModelSelector.routingModelUpdated"));
      keepDropdownOpen();
    },
    [
      effectiveCloudSlot,
      effectiveLocalSlot,
      keepDropdownOpen,
      routingConfig,
      t,
    ],
  );

  const handleSelectModel = useCallback(
    async (slot: ModelSlotConfig) => {
      const requests: Array<Promise<unknown>> = [
        providerApi.setActiveLlm(slot),
      ];
      if (routingConfig?.enabled) {
        requests.push(
          providerApi.setLlmRoutingConfig({
            ...routingConfig,
            enabled: false,
          }),
        );
      }
      await Promise.all(requests);
      message.success(
        routingConfig?.enabled
          ? t("chatModelSelector.routingDisabled")
          : t("models.llmModelUpdated"),
      );
      setOpen(false);
    },
    [routingConfig, t],
  );

  const handleMenuClick = async ({ key }: { key: string }) => {
    const keepOpen =
      key === ROUTING_LOCAL_FIRST ||
      key === ROUTING_CLOUD_FIRST ||
      key.startsWith(ROUTING_LOCAL_SLOT_PREFIX) ||
      key.startsWith(ROUTING_CLOUD_SLOT_PREFIX);
    keepOpenRef.current = keepOpen;
    setSaving(true);
    try {
      if (key === ROUTING_LOCAL_FIRST) {
        await handleEnableRouting("local_first");
      } else if (key === ROUTING_CLOUD_FIRST) {
        await handleEnableRouting("cloud_first");
      } else if (
        key.startsWith(ROUTING_LOCAL_SLOT_PREFIX) ||
        key.startsWith(ROUTING_CLOUD_SLOT_PREFIX)
      ) {
        const { kind, slot } = decodeRoutingSlotValue(key);
        await handleSelectRoutingSlot(kind, slot);
      } else {
        await handleSelectModel(decodeModelValue(key));
      }
      await reload();
    } catch (error) {
      const errMsg =
        error instanceof Error
          ? error.message
          : t("chatModelSelector.updateFailed");
      message.error(errMsg);
    } finally {
      setSaving(false);
      if (!keepOpen) {
        setOpen(false);
      }
    }
  };

  const menuItems = useMemo<MenuProps["items"]>(() => {
    const routingChildren: NonNullable<MenuProps["items"]> = [
      {
        key: ROUTING_LOCAL_FIRST,
        label: t("chatModelSelector.localFirst"),
        icon: selectedKey === ROUTING_LOCAL_FIRST ? <CheckOutlined /> : null,
      },
      {
        key: ROUTING_CLOUD_FIRST,
        label: t("chatModelSelector.cloudFirst"),
        icon: selectedKey === ROUTING_CLOUD_FIRST ? <CheckOutlined /> : null,
      },
    ];

    if (localOptions.length > 1) {
      routingChildren.push({
        key: "routing-local-model",
        label: t("chatModelSelector.localModel"),
        children: localOptions.map((option) => ({
          key: option.key,
          label: option.label,
          icon: selectedLocalKey === option.key ? <CheckOutlined /> : null,
        })),
      });
    }

    if (cloudOptions.length > 1) {
      routingChildren.push({
        key: "routing-cloud-model",
        label: t("chatModelSelector.cloudModel"),
        children: cloudOptions.map((option) => ({
          key: option.key,
          label: option.label,
          icon: selectedCloudKey === option.key ? <CheckOutlined /> : null,
        })),
      });
    }

    return [
      {
        type: "group",
        label: t("chatModelSelector.modelsGroup"),
        children: modelOptions.map((option) => ({
          key: option.key,
          label: option.label,
          icon: selectedKey === option.key ? <CheckOutlined /> : null,
        })),
      },
      {
        type: "divider",
      },
      {
        type: "group",
        label: t("chatModelSelector.routingGroup"),
        children: routingChildren,
      },
    ];
  }, [
    cloudOptions,
    localOptions,
    modelOptions,
    selectedCloudKey,
    selectedKey,
    selectedLocalKey,
    t,
  ]);

  const triggerLabel = routingConfig?.enabled
    ? t("chatModelSelector.triggerRouting")
    : getModelName(providers, activeModels?.active_llm) ||
      t("chatModelSelector.placeholder");

  const triggerBadge = routingConfig?.enabled
    ? t(
        routingConfig.mode === "cloud_first"
          ? "chatModelSelector.cloudFirst"
          : "chatModelSelector.localFirst",
      )
    : hasConfiguredSlot(activeModels?.active_llm)
    ? getModelMenuLabel(providers, activeModels.active_llm)
    : "";

  return (
    <Dropdown
      trigger={["click"]}
      open={open}
      onOpenChange={(nextOpen) => {
        if (!nextOpen && keepOpenRef.current) {
          keepOpenRef.current = false;
          setOpen(true);
          return;
        }
        setOpen(nextOpen);
      }}
      menu={{
        items: menuItems,
        onClick: ({ key }) => void handleMenuClick({ key: String(key) }),
      }}
      overlayClassName={styles.routingMenu}
      placement="bottomRight"
    >
      <Button
        className={styles.routingTrigger}
        loading={loading || saving}
        type="default"
      >
        <span className={styles.routingTriggerInner}>
          <span className={styles.routingTriggerLead}>
            <ApartmentOutlined className={styles.routingTriggerIcon} />
            <span className={styles.routingTriggerLabel}>{triggerLabel}</span>
          </span>
          {triggerBadge ? (
            <span className={styles.routingTriggerBadge}>{triggerBadge}</span>
          ) : null}
          <DownOutlined className={styles.routingTriggerCaret} />
        </span>
      </Button>
    </Dropdown>
  );
}
