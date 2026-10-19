// A scripted stand-in for the API, for browser tests: no GPU, database or money.
import { createServer } from "node:http";
import { randomUUID } from "node:crypto";
import { PAPERS, PASSWORD, USER } from "./fixtures.mjs";

const PORT = Number(process.env.MOCK_BACKEND_PORT ?? 8799);
let state;
function reset() {
  state = { expired: false, deleted: false, databaseDown: false, chats: new Map(), timedOut: new Set() };
}
reset();

const json = (res, status, body) => {
  res.writeHead(status, { "content-type": "application/json" });
  res.end(JSON.stringify(body));
};
const error = (res, status, code, message, retryable = false) =>
  json(res, status, { error: { code, message, retryable }, request_id: randomUUID() });
const body = (req) =>
  new Promise((resolve) => {
    let raw = "";
    req.on("data", (chunk) => (raw += chunk));
    req.on("end", () => resolve(raw ? JSON.parse(raw) : {}));
  });
const authed = (req) => req.headers.authorization === "Bearer token-alice" && !state.expired && !state.deleted;
const summary = (p) => ({
  title: p.title, authors: p.authors, venue: p.venue, year: p.year,
  abstract: p.abstract, pdf_url: p.pdf_url, fulltext_indexed: p.fulltext_indexed,
});
const searchResponse = (items) => ({
  request_id: randomUUID(),
  corpus_release_id: "mock-release",
  items: items.map((p, i) => ({ paper_id: p.id, score: 1 - i / 10, scores: {}, rank: i + 1 })),
  papers: Object.fromEntries(items.map((p) => [p.id, summary(p)])),
  next_cursor: null,
  degraded: false,
  warnings: [],
});
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function stream(req, res, payload) {
  const send = (event, data) => res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  let chat = payload.chat_id ? state.chats.get(payload.chat_id) : null;
  if (payload.chat_id && !chat) return error(res, 404, "chat_not_found", "no such chat");
  if (!chat) {
    chat = { id: randomUUID(), title: payload.message.slice(0, 80), created_at: new Date().toISOString(), messages: [] };
    state.chats.set(chat.id, chat);
  }
  const follow = chat.messages.length > 0;
  const now = new Date().toISOString();
  chat.updated_at = now;
  const user = { id: randomUUID(), ordinal: chat.messages.length, role: "user", status: "complete", text: payload.message,
    claims: [], evidence: [], references: [], standalone_query: null, warnings: [], error_code: null, corpus_release_id: null, created_at: now };
  chat.messages.push(user);
  const messageId = randomUUID();
  res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-store" });
  send("chat", { chat_id: chat.id, message_id: messageId });
  send("status", { stage: follow ? "condensing" : "searching" });
  const pause = /slow/i.test(payload.message) ? 700 : 150;
  await sleep(pause);
  if (/timeout/i.test(payload.message) && !state.timedOut.has(payload.message)) {
    // Both search branches missed their deadlines, once; asking again succeeds.
    state.timedOut.add(payload.message);
    chat.messages.push({ ...user, id: messageId, ordinal: user.ordinal + 1, role: "assistant", status: "failed", text: "", error_code: "candidates_unavailable" });
    send("error", { error: { code: "candidates_unavailable", message: "Something went wrong while answering. Try again.", retryable: true } });
    return res.end();
  }
  if (/budget/i.test(payload.message)) {
    chat.messages.push({ ...user, id: messageId, ordinal: user.ordinal + 1, role: "assistant", status: "failed", text: "", error_code: "llm_spend_cap" });
    send("error", { error: { code: "llm_spend_cap", message: "Today's answer budget is used up. It resets at 00:00 UTC.", retryable: false } });
    return res.end();
  }
  const query = follow ? `${payload.message} (contrastive sentence embeddings)` : payload.message;
  if (follow) send("status", { stage: "searching", query });
  await sleep(pause);
  // The turn's ranked papers, the first two read (the real turn reads 10 of 20).
  const references = PAPERS.map((p, i) => ({ paper_id: p.id, rank: i + 1, read: i < 2, title: p.title, venue: p.venue, year: p.year }));
  send("references", { query, items: references });
  send("evidence", { items: [{ id: "E1", paper_id: PAPERS[0].id, section: "Method", page_start: 3, page_end: 3 }], warnings: [] });
  send("status", { stage: "writing" });
  await sleep(pause);
  const insufficient = /protein/i.test(payload.message);
  const evidence = [
    { id: "E1", paper_id: PAPERS[0].id, title: PAPERS[0].title, venue: "ACL", year: 2024, section: "Method",
      page_start: 3, page_end: 3, text: "Each sentence is paired with a dropout-noised copy of itself; other sentences in the batch are negatives.",
      abstract_only: false, cited: !insufficient },
    { id: "E2", paper_id: PAPERS[1].id, title: PAPERS[1].title, venue: "ICLR", year: 2025, section: "Abstract",
      page_start: null, page_end: null, text: PAPERS[1].abstract, abstract_only: true, cited: false },
  ];
  const assistant = {
    id: messageId, ordinal: user.ordinal + 1, role: "assistant",
    status: insufficient ? "insufficient" : "complete",
    text: insufficient ? "" : "Sentence pairs are formed from a sentence and a noised copy of it.",
    claims: insufficient ? [] : [{ text: "Sentence pairs are formed from a sentence and a noised copy of it.", evidence_ids: ["E1"] }],
    // "partial" answers as if one search branch had run out of time.
    evidence, references, standalone_query: query, warnings: /partial/i.test(payload.message) ? ["lexical_timeout"] : [], error_code: null, corpus_release_id: "mock-release", created_at: now,
  };
  chat.messages.push(assistant);
  send("final", assistant);
  res.end();
}

createServer(async (req, res) => {
  const url = new URL(req.url, `http://127.0.0.1:${PORT}`);
  const path = url.pathname;
  if (path === "/__control/reset") { reset(); return json(res, 200, { ok: true }); }
  if (path === "/__control/database") { state.databaseDown = url.searchParams.get("down") === "1"; return json(res, 200, { ok: true }); }
  if (path === "/__control/expire") { state.expired = true; return json(res, 200, { ok: true }); }
  if (path === "/__control/seed-history") {
    // Three answered chats last used today, 3 days ago and 30 days ago, for the History page.
    for (const [title, days] of [["What loss does SimCSE use?", 0], ["Graph transformers for molecules", 3], ["Speculative decoding speedups", 30]]) {
      const at = new Date(Date.now() - days * 86400000 - 3600000).toISOString();
      const id = randomUUID();
      state.chats.set(id, { id, title, created_at: at, updated_at: at, messages: [
        { id: randomUUID(), ordinal: 0, role: "user", status: "complete", text: title, claims: [], evidence: [], references: [],
          standalone_query: null, warnings: [], error_code: null, corpus_release_id: null, created_at: at },
      ] });
    }
    return json(res, 200, { ok: true });
  }
  if (path === "/v1/auth/token" && req.method === "POST") {
    const payload = await body(req);
    if (!state.deleted && payload.email?.toLowerCase() === USER.email && payload.password === PASSWORD) {
      state.expired = false;
      return json(res, 200, { access_token: "token-alice", token_type: "bearer", expires_in: 3600, request_id: randomUUID() });
    }
    return error(res, 401, "invalid_credentials", "email or password is incorrect");
  }
  if (path === "/v1/corpus/coverage") {
    return json(res, 200, { request_id: randomUUID(), as_of: "2026-10-11T00:00:00Z", venue_years: [
      { venue: "ACL", year: 2024, papers: 1 }, { venue: "ACL", year: 2025, papers: 1 }, { venue: "ICLR", year: 2025, papers: 1 }],
      totals: { papers: 3, with_abstract: 3, with_fulltext: 2, failed_parse: 0, venue_years: 3 } });
  }
  const paper = path.match(/^\/v1\/papers\/([^/]+)(\/related)?$/);
  if (paper) {
    const found = PAPERS.find((p) => p.id === paper[1]);
    if (!found) return error(res, 404, "paper_not_found", "no such paper");
    if (paper[2]) return json(res, 200, searchResponse(PAPERS.filter((p) => p.id !== found.id)));
    return json(res, 200, { request_id: randomUUID(), paper_id: found.id, redirected_from: null, paper: summary(found),
      acceptance_type: "poster", identifiers: [], source: { pdf_url: found.pdf_url, landing_url: null, available: Boolean(found.pdf_url) }, parse_status: "parsed" });
  }
  if (path.startsWith("/v1/me") || path.startsWith("/v1/chats")) {
    if (!req.headers.authorization) return error(res, 401, "authentication_required", "sign in to continue");
    if (!authed(req)) return error(res, 401, state.expired ? "token_expired" : "token_invalid", "your session is not valid; sign in again");
  }
  if (state.databaseDown && path.startsWith("/v1/chats") && req.method === "GET") {
    return error(res, 503, "database_unavailable", "database is unavailable", true);
  }
  if (path === "/v1/me" && req.method === "DELETE") {
    const payload = await body(req);
    if (payload.password !== PASSWORD) return error(res, 401, "invalid_credentials", "the password is incorrect");
    state.deleted = true;
    state.chats.clear();
    res.writeHead(204);
    return res.end();
  }
  if (path === "/v1/me") return json(res, 200, { ...USER, request_id: randomUUID() });
  if (path === "/v1/chats" && req.method === "GET") {
    const items = [...state.chats.values()].sort((a, b) => b.updated_at.localeCompare(a.updated_at))
      .map(({ id, title, created_at, updated_at }) => ({ id, title, created_at, updated_at }));
    return json(res, 200, { request_id: randomUUID(), items });
  }
  if (path === "/v1/chats/stream" && req.method === "POST") return stream(req, res, await body(req));
  const chat = path.match(/^\/v1\/chats\/([^/]+)$/);
  if (chat) {
    const found = state.chats.get(chat[1]);
    if (!found) return error(res, 404, "chat_not_found", "no such chat");
    if (req.method === "DELETE") { state.chats.delete(found.id); res.writeHead(204); return res.end(); }
    const { messages, ...rest } = found;
    return json(res, 200, { request_id: randomUUID(), chat: { id: rest.id, title: rest.title, created_at: rest.created_at, updated_at: rest.updated_at }, messages });
  }
  return error(res, 404, "not_found", "no such endpoint");
}).listen(PORT, "127.0.0.1");
