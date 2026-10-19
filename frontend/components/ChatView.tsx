"use client";

import { usePathname, useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { ApiError, api, errorFrom, loginRedirect, streamTurn, type ApiErrorBody, type ChatMessage, type Reference } from "@/lib/api";
import { STAGES } from "@/lib/chat-labels";
import { corpusLine, withFilter, type AskFilter, type Corpus } from "@/lib/corpus";
import { readEvents } from "@/lib/sse";
import { Answer } from "./Answer";
import { Composer } from "./Composer";
import { EvidencePanel } from "./EvidencePanel";
import { Hero } from "./Hero";
import { References } from "./References";
import { Suggestions } from "./Suggestions";

type Failure = ApiErrorBody & { requestId: string; message_text: string };

type Props = { chatId: string | null; about: string | null; initialDraft?: string | null; corpus?: Corpus | null };

export function ChatView({ chatId, about, initialDraft = null, corpus = null }: Props) {
  const router = useRouter();
  const pathname = usePathname();
  const [id, setId] = useState<string | null>(chatId);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState(initialDraft ?? (about ? "In this paper, " : ""));
  const [stage, setStage] = useState<string | null>(null);
  // The search step, from the `references` frame, shown while the answer is written.
  const [searched, setSearched] = useState<{ query: string; read: number } | null>(null);
  const [failure, setFailure] = useState<Failure | null>(null);
  const [loadFailed, setLoadFailed] = useState(false);
  const [focus, setFocus] = useState<{ message: string; evidence: string } | null>(null);
  const abort = useRef<AbortController | null>(null);
  // A new chat's first answer moves the URL to /chat/<id> in place; "New chat" (`/`) then lands on
  // this same mounted view, which starts over rather than keep the previous conversation.
  const [path, setPath] = useState(pathname);
  if (path !== pathname) {
    setPath(pathname);
    if (chatId === null && !pathname.startsWith("/chat/")) {
      setId(null);
      setMessages([]);
      setFailure(null);
      setFocus(null);
      setDraft("");
    }
  }

  function expired() {
    router.replace(loginRedirect(pathname, true));
  }

  function loadChat(id: string) {
    api.chat(id).then(
      (detail) => {
        setMessages(detail.messages);
        setLoadFailed(false);
      },
      (error: unknown) => {
        if (error instanceof ApiError && error.status === 401) expired();
        else if (error instanceof ApiError && error.status === 404) router.replace("/");
        else setLoadFailed(true);
      },
    );
  }

  useEffect(() => {
    if (!chatId) return;
    loadChat(chatId);
    return () => abort.current?.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatId]);

  async function send(text: string, requestId: string) {
    setFailure(null);
    setStage("searching");
    setSearched(null);
    const controller = new AbortController();
    abort.current = controller;
    const optimistic: ChatMessage = {
      id: requestId, ordinal: messages.length, role: "user", status: "complete", text,
      claims: [], evidence: [], references: [], standalone_query: null, warnings: [], error_code: null,
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
        } else if (event.event === "references" && typeof data.query === "string" && Array.isArray(data.items)) {
          setSearched({ query: data.query, read: (data.items as Reference[]).filter((item) => item.read).length });
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
      setSearched(null);
      abort.current = null;
    }
  }

  function ask(text: string, filter: AskFilter) {
    if (stage) return;
    setDraft("");
    void send(withFilter(text, filter), crypto.randomUUID());
  }

  const running = stage !== null;
  const empty = !id && messages.length === 0;
  const composer = (
    <Composer value={draft} onChange={setDraft} onSubmit={ask} running={running} onStop={() => abort.current?.abort()}
      placeholder={empty ? "Ask a question, or ask for papers on a topic" : "Ask a follow-up"}
      venues={corpus?.venues} years={corpus?.years} autoFocus={empty} />
  );
  if (empty) {
    const line = corpusLine(corpus);
    return (
      <div className="ask-home">
        <Hero />
        {composer}
        <Suggestions onPick={(question) => ask(question, { venue: null, year: null })} />
        {line ? <p className="corpus-line">{line}</p> : null}
      </div>
    );
  }
  return (
    <div className="thread">
      <ol aria-label="Conversation" style={{ listStyle: "none", padding: 0, display: "grid", gap: 12 }}>
        {messages.map((message) =>
          message.role === "user" ? (
            <li key={message.id}><p style={{ margin: 0, fontWeight: 600 }}>{message.text}</p></li>
          ) : message.status === "complete" || message.status === "insufficient" ? (
            <li key={message.id}>
              <Answer message={message} onCite={(evidence) => setFocus({ message: message.id, evidence })} />
              <References message={message} />
              {focus?.message === message.id ? <EvidencePanel message={message} focused={focus.evidence} /> : null}
            </li>
          ) : (
            <li key={message.id} className="muted">This answer was not completed ({message.error_code ?? message.status}).</li>
          ),
        )}
      </ol>
      <div role="status" aria-live="polite">
        {stage && searched ? (
          <p className="muted" style={{ margin: 0 }}>Searched for “{searched.query}” · reading {searched.read} paper{searched.read === 1 ? "" : "s"}</p>
        ) : null}
        {stage ? <p className="muted">{STAGES[stage] ?? "Working…"}</p> : null}
      </div>
      {loadFailed && chatId ? (
        <p role="alert" className="banner error">
          This chat could not be loaded. <button type="button" onClick={() => loadChat(chatId)}>Retry</button>
        </p>
      ) : null}
      {failure ? (
        <div role="alert" className="banner error">
          {failure.message}{" "}
          {failure.retryable ? (
            <button type="button" onClick={() => void send(failure.message_text, failure.requestId)}>Retry</button>
          ) : null}
        </div>
      ) : null}
      <div className="composer-dock">{composer}</div>
    </div>
  );
}
