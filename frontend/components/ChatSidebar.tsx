"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { ApiError, api, loginRedirect, type ChatList } from "@/lib/api";

export function ChatSidebar() {
  const router = useRouter();
  const pathname = usePathname();
  const [chats, setChats] = useState<ChatList["items"]>([]);
  const [failed, setFailed] = useState(false);

  const load = useCallback(() => {
    api.chats().then(
      (list) => {
        setChats(list.items);
        setFailed(false);
      },
      (error: unknown) => {
        if (error instanceof ApiError && error.status === 401) router.replace(loginRedirect(window.location.pathname, true));
        // Anything else (a database outage is a 503) is said, never shown as an empty history.
        else setFailed(true);
      },
    );
  }, [router]);

  useEffect(() => {
    load();
    window.addEventListener("chats-changed", load);
    return () => window.removeEventListener("chats-changed", load);
  }, [load]);

  async function remove(id: string, title: string) {
    if (!window.confirm(`Delete “${title}”? This cannot be undone.`)) return;
    await api.deleteChat(id);
    if (window.location.pathname === `/chat/${id}`) router.replace("/");
    load();
  }

  return (
    <nav aria-label="Chats" style={{ borderRight: "1px solid var(--border)", paddingRight: 12 }}>
      <p><Link href="/">New chat</Link></p>
      {failed ? (
        <p className="banner error" role="alert">
          Your chats could not be loaded. <button type="button" onClick={load}>Retry</button>
        </p>
      ) : null}
      <ul style={{ listStyle: "none", padding: 0 }}>
        {chats.map((chat) => (
          <li key={chat.id} style={{ display: "flex", gap: 4, alignItems: "center" }}>
            <Link href={`/chat/${chat.id}`} aria-current={pathname === `/chat/${chat.id}` ? "page" : undefined} style={{ flex: 1 }}>
              {chat.title}
            </Link>
            <button type="button" aria-label={`Delete chat ${chat.title}`} onClick={() => void remove(chat.id, chat.title)}>×</button>
          </li>
        ))}
      </ul>
    </nav>
  );
}
