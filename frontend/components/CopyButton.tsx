"use client";

import { useState } from "react";
import { Icon } from "./Icon";

/** A button that puts `text` on the clipboard and says so for two seconds. */
export function CopyButton({ text, label, className }: { text: string; label: string; className?: string }) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    await navigator.clipboard.writeText(text).catch(() => undefined);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 2000);
  }

  return (
    <button type="button" className={className} onClick={() => void copy()}>
      <Icon name={copied ? "check" : "copy"} size={16} /> {copied ? "Copied" : label}
    </button>
  );
}
