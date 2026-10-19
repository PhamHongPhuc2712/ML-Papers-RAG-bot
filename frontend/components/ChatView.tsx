"use client";

import { usePathname, useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { ApiError, api, errorFrom, sessionEnded, streamTurn, type ApiErrorBody, type ChatMessage, type Reference } from "@/lib/api";
import { ERRORS } from "@/lib/chat-labels";
import { corpusLine, withFilter, type AskFilter, type Corpus } from "@/lib/corpus";
import { readEvents } from "@/lib/sse";
import { Answer } from "./Answer";
import { Composer } from "./Composer";
import { Hero } from "./Hero";
import { ReferencesPanel, type PanelState } from "./ReferencesPanel";
import { LiveSteps, type LiveSearch } from "./StepsBox";
import { Suggestions } from "./Suggestions";

type Failure = ApiErrorBody & { requestId: string; message_text: string };

type Props = {
  chatId: string | null;
  /** A question to start with: put in the box, or asked at once with `autoSend` (from a paper page). */
  initialDraft?: string | null;
  autoSend?: boolean;
  corpus?: Corpus | null;
};

export function ChatView({ chatId, initialDraft = null, autoSend = false, corpus = null }: Props) {
  const router = useRouter();
  const pathname = usePathname();
  const [id, setId] = useState<string | null>(chatId);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState(autoSend ? "" : (initialDraft ?? ""));
  const [stage, setStage] = useState<string | null>(null);
  // The search step, from the `references` frame, shown while the answer is written.
  const [searched, setSearched] = useState<LiveSearch>(null);
  // A turn stopped here, by Stop: its question can be asked again.
  const [stopped, setStopped] = useState<{ requestId: string; text: string } | null>(null);
  const [failure, setFailure] = useState<Failure | null>(null);
  const [loadFailed, setLoadFailed] = useState(false);
  const [panel, setPanel] = useState<PanelState | null>(null);
  const abort = useRef<AbortController | null>(null);
  const autoSent = useRef(false);
  // The question to bring to the top of the thread once it is drawn: a new one, or a loaded chat's last.
  const reveal = useRef<{ id: string; smooth: boolean } | null>(null);
  // A new chat's first answer moves the URL to /chat/<id> in place; "New chat" (`/`) then lands on
  // this same mounted view, which starts over rather than keep the previous conversation.
  const [path, setPath] = useState(pathname);
  if (path !== pathname) {
    setPath(pathname);
    if (chatId === null && !pathname.startsWith("/chat/")) {
      setId(null);
      setMessages([]);
      setFailure(null);
      setPanel(null);
      setStopped(null);
      setDraft("");
    }
  }

  function expired() {
    sessionEnded(router, pathname);
  }

  function loadChat(id: string) {
    api.chat(id).then(
      (detail) => {
        setMessages(detail.messages);
        const last = [...detail.messages].reverse().find((m) => m.role === "user");
        if (last) reveal.current = { id: last.id, smooth: false };
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

  useEffect(() => {
    if (!autoSend || !initialDraft || autoSent.current) return;
    autoSent.current = true;
    // Asked once: Back or a reload must not ask it again, so `?q=…&send=1` leaves the history.
    window.history.replaceState(null, "", "/");
    void send(initialDraft, crypto.randomUUID());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const target = reveal.current;
    if (!target) return;
    const element = document.querySelector(`[data-message="${target.id}"]`);
    if (!element) return;
    reveal.current = null;
    // The answer is then read from its start, below its question, not from under the ask box.
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    element.scrollIntoView({ block: "start", behavior: target.smooth && !still ? "smooth" : "auto" });
  }, [messages]);

  async function send(text: string, requestId: string) {
    setFailure(null);
    setStopped(null);
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
    reveal.current = { id: requestId, smooth: true };
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
          const items = data.items as Reference[];
          setSearched({ query: data.query, ranked: items.length, read: items.filter((item) => item.read).length });
        } else if (event.event === "final") {
          setMessages((previous) => [...previous, data as unknown as ChatMessage]);
          window.dispatchEvent(new Event("chats-changed"));
        } else if (event.event === "error") {
          const body = (data as { error: ApiErrorBody }).error;
          setFailure({ ...body, requestId, message_text: text });
        }
      }
    } catch (error) {
      if (controller.signal.aborted) {
        setStopped({ requestId, text });
        return;
      }
      const body = error instanceof ApiError ? error.body : { code: "network_error", message: "The connection dropped. Try again.", retryable: true };
      setFailure({ ...body, requestId, message_text: text });
    } finally {
      setStage(null);
      setSearched(null);
      abort.current = null;
    }
  }

  function askAgain(text: string, requestId: string) {
    // The stopped question is asked again as a new turn, in place of its bubble.
    setMessages((previous) => previous.filter((m) => m.id !== requestId));
    void send(text, crypto.randomUUID());
  }

  function cite(message: ChatMessage, evidenceId: string) {
    const paperId = (message.evidence ?? []).find((item) => item.id === evidenceId)?.paper_id ?? null;
    setPanel({ messageId: message.id, paperId, evidenceId });
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
  const latest = [...messages].reverse().find((m) => m.role === "assistant" && (m.references ?? []).length > 0);
  const panelMessage = panel ? messages.find((m) => m.id === panel.messageId) : undefined;
  const title = messages.find((m) => m.role === "user")?.text ?? "New chat";
  // The question a stored failed or stopped answer belongs to, for asking it again.
  const question = (index: number) => (messages[index - 1]?.role === "user" ? messages[index - 1]?.text : undefined);
  return (
    <div className={panelMessage ? "conversation with-panel" : "conversation"}>
      <div className="thread-column">
        <header className="thread-header">
          <h1 className="thread-title">{title}</h1>
          <button type="button" className="chip-button" disabled={!latest}
            aria-pressed={Boolean(panel && latest && panel.messageId === latest.id && !panel.paperId)}
            onClick={() => latest && setPanel(panel && panel.messageId === latest.id && !panel.paperId ? null
              : { messageId: latest.id, paperId: null, evidenceId: null })}>
            References{latest ? ` (${latest.references?.length})` : ""}
          </button>
        </header>
        <div className="thread">
          <ol aria-label="Conversation" className="messages">
            {messages.map((message, index) =>
              message.role === "user" ? (
                <li key={message.id} className="question" data-message={message.id}><p className="bubble">{message.text}</p></li>
              ) : message.status === "complete" || message.status === "insufficient" ? (
                <li key={message.id}>
                  <Answer message={message} selected={panel?.messageId === message.id ? panel.evidenceId : null}
                    onCite={(evidenceId) => cite(message, evidenceId)}
                    onReferences={() => setPanel({ messageId: message.id, paperId: null, evidenceId: null })} />
                </li>
              ) : (
                <li key={message.id} className="turn-note">
                  {message.status === "interrupted" ? "Stopped." : "This answer could not be finished."}
                  {question(index) && !running ? (
                    <button type="button" onClick={() => void send(question(index) ?? "", crypto.randomUUID())}>
                      {message.status === "interrupted" ? "Ask again" : "Try again"}
                    </button>
                  ) : null}
                </li>
              ),
            )}
            {stopped && !running ? (
              <li className="turn-note">
                Stopped.
                <button type="button" onClick={() => askAgain(stopped.text, stopped.requestId)}>Ask again</button>
              </li>
            ) : null}
          </ol>
          <div role="status" aria-live="polite" className="live">
            {stage ? (
              <>
                <LiveSteps stage={stage} searched={searched} />
                <div className="skeleton" aria-hidden="true"><span /><span /><span /></div>
              </>
            ) : null}
          </div>
          {loadFailed && chatId ? (
            <p role="alert" className="banner error">
              This chat could not be loaded. <button type="button" onClick={() => loadChat(chatId)}>Retry</button>
            </p>
          ) : null}
          {failure?.code === "llm_spend_cap" ? (
            // Not a fault: today's budget is spent, and asking works again after the reset (board 11).
            <p role="status" className="banner warn">{failure.message}</p>
          ) : failure ? (
            <div role="alert" className="banner error">
              <p>{ERRORS[failure.code] ?? failure.message}{failure.retryable ? " Your question is kept." : ""}</p>
              {failure.retryable ? (
                <button type="button" className="primary" onClick={() => void send(failure.message_text, failure.requestId)}>Try again</button>
              ) : null}
            </div>
          ) : null}
          <div className="composer-dock">{composer}</div>
        </div>
      </div>
      {panel && panelMessage ? (
        <ReferencesPanel message={panelMessage} state={panel}
          onOpenPaper={(paperId) => setPanel({ ...panel, paperId, evidenceId: null })}
          onBack={() => setPanel({ ...panel, paperId: null, evidenceId: null })}
          onClose={() => setPanel(null)} />
      ) : null}
    </div>
  );
}
