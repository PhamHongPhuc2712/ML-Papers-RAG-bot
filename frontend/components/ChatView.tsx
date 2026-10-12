"use client";

import { usePathname, useRouter } from "next/navigation";
import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { ApiError, api, errorFrom, loginRedirect, streamTurn, type ApiErrorBody, type ChatMessage } from "@/lib/api";
import { STAGES } from "@/lib/chat-labels";
import { readEvents } from "@/lib/sse";
import { Answer } from "./Answer";
import { EvidencePanel } from "./EvidencePanel";

const LIMIT = 2000;
type Failure = ApiErrorBody & { requestId: string; message_text: string };

export function ChatView({ chatId, about }: { chatId: string | null; about: string | null }) {
  const router = useRouter();
  const pathname = usePathname();
  const [id, setId] = useState<string | null>(chatId);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState(about ? "In this paper, " : "");
  const [stage, setStage] = useState<string | null>(null);
  const [failure, setFailure] = useState<Failure | null>(null);
  const [focus, setFocus] = useState<{ message: string; evidence: string } | null>(null);
  const abort = useRef<AbortController | null>(null);

  function expired() {
    router.replace(loginRedirect(pathname, true));
  }

  useEffect(() => {
    if (!chatId) return;
    api.chat(chatId).then(
      (detail) => setMessages(detail.messages),
      (error: unknown) => {
        if (error instanceof ApiError && error.status === 401) expired();
        else if (error instanceof ApiError && error.status === 404) router.replace("/chat");
      },
    );
    return () => abort.current?.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatId]);

  async function send(text: string, requestId: string) {
    setFailure(null);
    setStage("searching");
    const controller = new AbortController();
    abort.current = controller;
    const optimistic: ChatMessage = {
      id: requestId, ordinal: messages.length, role: "user", status: "complete", text,
      claims: [], evidence: [], standalone_query: null, warnings: [], error_code: null,
      corpus_release_id: null, created_at: new Date().toISOString(),
    };
    setMessages((previous) => (previous.some((m) => m.id === requestId) ? previous : [...previous, optimistic]));
    try {
      const response = await streamTurn({ request_id: requestId, chat_id: id, message: text }, controller.signal);
      if (!response.ok || !response.body) {
        const error = await errorFrom(response);
        if (error.status === 401) return expired();
        throw error;
      }
      for await (const event of readEvents(response.body)) {
        const data = event.data as Record<string, unknown>;
        if (event.event === "chat" && typeof data.chat_id === "string" && !id) {
          setId(data.chat_id);
          window.history.replaceState(null, "", `/chat/${data.chat_id}`);
          window.dispatchEvent(new Event("chats-changed"));
        } else if (event.event === "status" && typeof data.stage === "string") {
          setStage(data.stage);
        } else if (event.event === "final") {
          setMessages((previous) => [...previous, data as unknown as ChatMessage]);
          window.dispatchEvent(new Event("chats-changed"));
        } else if (event.event === "error") {
          const body = (data as { error: ApiErrorBody }).error;
          setFailure({ ...body, requestId, message_text: text });
        }
      }
    } catch (error) {
      if (controller.signal.aborted) return;
      const body = error instanceof ApiError ? error.body : { code: "network_error", message: "The connection dropped. Try again.", retryable: true };
      setFailure({ ...body, requestId, message_text: text });
    } finally {
      setStage(null);
      abort.current = null;
    }
  }

  function submit(event?: FormEvent) {
    event?.preventDefault();
    const text = draft.trim();
    if (!text || text.length > LIMIT || stage) return;
    setDraft("");
    void send(text, crypto.randomUUID());
  }

  function onKey(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  }

  const running = stage !== null;
  return (
    <div style={{ display: "grid", gap: 16 }}>
      <ol aria-label="Conversation" style={{ listStyle: "none", padding: 0, display: "grid", gap: 12 }}>
        {messages.map((message) =>
          message.role === "user" ? (
            <li key={message.id}><p style={{ margin: 0, fontWeight: 600 }}>{message.text}</p></li>
          ) : message.status === "complete" || message.status === "insufficient" ? (
            <li key={message.id}>
              <Answer message={message} onCite={(evidence) => setFocus({ message: message.id, evidence })} />
              {focus?.message === message.id ? <EvidencePanel message={message} focused={focus.evidence} /> : null}
            </li>
          ) : (
            <li key={message.id} className="muted">This answer was not completed ({message.error_code ?? message.status}).</li>
          ),
        )}
      </ol>
      <div role="status" aria-live="polite">{stage ? <p className="muted">{STAGES[stage] ?? "Working…"}</p> : null}</div>
      {failure ? (
        <div role="alert" className="banner error">
          {failure.message}{" "}
          {failure.retryable ? (
            <button type="button" onClick={() => void send(failure.message_text, failure.requestId)}>Retry</button>
          ) : null}
        </div>
      ) : null}
      <form onSubmit={submit}>
        <label htmlFor="composer">Ask about ML papers</label>
        <textarea id="composer" rows={3} value={draft} onChange={(e) => setDraft(e.target.value)} onKeyDown={onKey}
          aria-describedby="composer-count" />
        <p id="composer-count" className="muted" style={{ margin: "4px 0" }}>{draft.length} / {LIMIT}</p>
        <button className="primary" type="submit" disabled={running || !draft.trim() || draft.length > LIMIT}>Send</button>
      </form>
    </div>
  );
}
