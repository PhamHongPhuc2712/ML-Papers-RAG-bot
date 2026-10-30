"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { ApiError, api, sessionEnded, type ChatList as Chats } from "@/lib/api";
import { groupChats } from "@/lib/chat-groups";
import { Icon } from "./Icon";

/** The signed-in person's chats, grouped by when they were last used. */
export function ChatList({ onNavigate }: { onNavigate: () => void }) {
  const router = useRouter();
  const pathname = usePathname();
  const [chats, setChats] = useState<Chats["items"]>([]);
  const [failed, setFailed] = useState(false);

  const load = useCallback(() => {
    api.chats().then(
      (list) => {
        setChats(list.items);
        setFailed(false);
      },
      (error: unknown) => {
        if (error instanceof ApiError && error.status === 401) sessionEnded(router, window.location.pathname);
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
    // The History page lists the same chats.
    window.dispatchEvent(new Event("chats-changed"));
  }

  return (
    <nav aria-label="Chats" className="chat-list">
      {failed ? (
        <p className="banner error" role="alert">
          Your chats could not be loaded. <button type="button" onClick={load}>Retry</button>
        </p>
      ) : null}
      {!failed && chats.length === 0 ? <p className="muted" style={{ padding: "0 10px", fontSize: "0.875rem" }}>Your chats appear here.</p> : null}
      {groupChats(chats).map((group) => (
        <section key={group.label} className="chat-group" aria-label={group.label}>
          <h2>{group.label}</h2>
          <ul>
            {group.chats.map((chat) => (
              <li key={chat.id} className="chat-item">
                <Link href={`/chat/${chat.id}`} onClick={onNavigate} title={chat.title}
                  aria-current={pathname === `/chat/${chat.id}` ? "page" : undefined}>
                  {chat.title}
                </Link>
                <button type="button" className="icon-button" aria-label={`Delete chat ${chat.title}`}
                  onClick={() => void remove(chat.id, chat.title)}>
                  <Icon name="trash" size={16} />
                </button>
              </li>
            ))}
          </ul>
        </section>
      ))}
    </nav>
  );
}
