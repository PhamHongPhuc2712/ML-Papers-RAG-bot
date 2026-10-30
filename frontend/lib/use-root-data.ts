"use client";

import { useSyncExternalStore } from "react";

function subscribe(onChange: () => void): () => void {
  const observer = new MutationObserver(onChange);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "data-sidebar"] });
  return () => observer.disconnect();
}

/** A `data-*` value on <html>, set before paint by the shell's inline script and changed by the
 *  theme and sidebar controls; the server render (and hydration) sees `undefined`. */
export function useRootData(key: "theme" | "sidebar"): string | undefined {
  return useSyncExternalStore(subscribe, () => document.documentElement.dataset[key], () => undefined);
}
