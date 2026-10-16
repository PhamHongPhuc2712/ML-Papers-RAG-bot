"use client";

import { applyTheme, type Theme } from "@/lib/theme";
import { useRootData } from "@/lib/use-root-data";
import { Icon, type IconName } from "./Icon";

const CHOICES: { value: Theme; label: string; icon: IconName }[] = [
  { value: "system", label: "System", icon: "monitor" },
  { value: "light", label: "Light", icon: "sun" },
  { value: "dark", label: "Dark", icon: "moon" },
];

export function ThemeToggle() {
  const stored = useRootData("theme");
  const theme: Theme = stored === "light" || stored === "dark" ? stored : "system";
  return (
    <div className="theme-choice" role="group" aria-label="Theme">
      {CHOICES.map((choice) => (
        <button key={choice.value} type="button" aria-pressed={theme === choice.value}
          onClick={() => applyTheme(choice.value)}>
          <Icon name={choice.icon} size={15} />
          {choice.label}
        </button>
      ))}
    </div>
  );
}
