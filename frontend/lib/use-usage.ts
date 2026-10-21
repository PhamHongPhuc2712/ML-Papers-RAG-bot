"use client";

import { useEffect, useState } from "react";
import { api } from "./api";
import type { Usage } from "./usage";

/** The account's answers today (GET /v1/me), read again whenever a turn ends ("chats-changed"). */
export function useUsage(): Usage | null {
  const [usage, setUsage] = useState<Usage | null>(null);
  useEffect(() => {
    let live = true;
    const load = () => {
      api.me().then((me) => { if (live) setUsage(me.usage ?? null); }).catch(() => undefined);
    };
    load();
    window.addEventListener("chats-changed", load);
    return () => {
      live = false;
      window.removeEventListener("chats-changed", load);
    };
  }, []);
  return usage;
}
