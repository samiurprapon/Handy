import React, { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { commands, type TranscriptionRuntime } from "@/bindings";
import { useSettings } from "@/hooks/useSettings";
import { Dropdown, type DropdownOption } from "@/components/ui/Dropdown";
import { SettingContainer } from "@/components/ui/SettingContainer";

interface TranscriptionRuntimeSelectorProps {
  descriptionMode?: "tooltip" | "inline";
  grouped?: boolean;
}

/**
 * Picks the engine that runs transcription models. Rendered only where
 * WhisperKit can run (macOS on Apple Silicon); elsewhere there is nothing
 * to choose.
 */
export const TranscriptionRuntimeSelector: React.FC<
  TranscriptionRuntimeSelectorProps
> = ({ descriptionMode = "tooltip", grouped = false }) => {
  const { t } = useTranslation();
  const { getSetting, updateSetting, isUpdating } = useSettings();
  const [available, setAvailable] = useState(false);
  const selected = getSetting("transcription_runtime") ?? "transcribe_cpp";

  useEffect(() => {
    let active = true;
    commands
      .isWhisperkitAvailable()
      .then((result) => {
        if (active) setAvailable(result);
      })
      .catch(() => {
        if (active) setAvailable(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const options = useMemo<DropdownOption[]>(
    () => [
      {
        value: "transcribe_cpp",
        label: t(
          "settings.advanced.transcriptionRuntime.options.transcribeCpp",
        ),
      },
      {
        value: "whisperkit",
        label: t("settings.advanced.transcriptionRuntime.options.whisperKit"),
      },
    ],
    [t],
  );

  if (!available) return null;

  return (
    <SettingContainer
      title={t("settings.advanced.transcriptionRuntime.title")}
      description={t("settings.advanced.transcriptionRuntime.description")}
      descriptionMode={descriptionMode}
      grouped={grouped}
      layout="horizontal"
    >
      <Dropdown
        options={options}
        selectedValue={selected}
        onSelect={(value) =>
          updateSetting("transcription_runtime", value as TranscriptionRuntime)
        }
        disabled={isUpdating("transcription_runtime")}
      />
    </SettingContainer>
  );
};
