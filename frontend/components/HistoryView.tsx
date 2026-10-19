"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { ApiError, api, sessionEnded, type ChatList } from "@/lib/api";
import { chatAge, groupChats, matchChats } from "@/lib/chat-groups";
import { Icon } from "./Icon";

// The API's most per request; a longer history shows its newest this many.
const LIMIT = 100;

/** The History page's list: a title search in the browser, chats grouped by day, each with its age
 *  and a delete action. The API lists ids, titles and times only, so there is no answer preview. */
export function HistoryView() {
  const router = useRouter();
  const [chats, setChats] = useState<ChatList["items"] | null>(null);
  const [failed, setFailed] = useState(false);
  const [query, setQuery] = useState("");

  const load = useCallback(() => {
    api.chats(LIMIT).then(
      (list) => {
        setChats(list.items);
        setFailed(false);
      },
      (error: unknown) => {
        if (error instanceof ApiError && error.status === 401) sessionEnded(router, "/history");
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
    await api.deleteChat(id).catch(() => undefined);
    window.dispatchEvent(new Event("chats-changed"));
  }

  if (failed) {
    return (
      <div role="alert" className="banner error">
        The service can’t be reached right now. Your chats are safe. Try again in a minute.{" "}
        <button type="button" onClick={load}>Try again</button>
      </div>
    );
  }
  if (chats === null) return <p className="muted" role="status">Loading your chats…</p>;
  if (chats.length === 0) {
    return (
      <div className="empty-state">
        <p>Your chats appear here after your first question.</p>
        <Link href="/" className="button-link primary">Ask a question</Link>
      </div>
    );
  }
  const shown = matchChats(chats, query);
  const groups = groupChats(shown);
  return (
    <>
      <div className="history-search">
        <Icon name="search" size={18} />
        <label htmlFor="search-chats" className="visually-hidden">Search your chats</label>
        <input id="search-chats" type="search" placeholder="Search your chats" value={query}
          onChange={(e) => setQuery(e.target.value)} />
      </div>
      <p className="visually-hidden" role="status">
        {query.trim() ? `${shown.length} of ${chats.length} chats match` : ""}
      </p>
      {shown.length === 0 ? <p className="muted">No chat titles match “{query.trim()}”.</p> : null}
      {groups.map((group) => (
        <section key={group.label} aria-label={group.label} className="history-group">
          <h2>{group.label}</h2>
          <ul>
            {group.chats.map((chat) => (
              <li key={chat.id}>
                <div>
                  <Link href={`/chat/${chat.id}`}>{chat.title}</Link>
                  <p>{chatAge(chat.updated_at)}</p>
                </div>
                <button type="button" className="icon-button" aria-label={`Delete chat ${chat.title}`}
                  onClick={() => void remove(chat.id, chat.title)}>
                  <Icon name="trash" size={18} />
                </button>
              </li>
            ))}
          </ul>
        </section>
      ))}
      {chats.length === LIMIT ? <p className="muted">Showing your {LIMIT} most recently used chats.</p> : null}
    </>
  );
}
