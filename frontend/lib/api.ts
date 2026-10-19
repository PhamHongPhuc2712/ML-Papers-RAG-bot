import type { components } from "./api-types";

type Schemas = components["schemas"];
export type SearchResponse = Schemas["SearchResponse"];
export type PaperDetail = Schemas["PaperDetail"];
export type ChatList = Schemas["ChatList"];
export type ChatDetail = Schemas["ChatDetail"];
export type ChatMessage = Schemas["ChatMessage"];
export type StoredEvidence = Schemas["StoredEvidence"];
export type Reference = Schemas["Reference"];
export type ApiErrorBody = { code: string; message: string; retryable: boolean };

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly body: ApiErrorBody,
  ) {
    super(body.message);
  }
}

export async function errorFrom(response: Response): Promise<ApiError> {
  const payload = (await response.json().catch(() => null)) as { error?: ApiErrorBody } | null;
  return new ApiError(
    response.status,
    payload?.error ?? {
      code: `http_${response.status}`,
      message: "The request failed.",
      retryable: response.status >= 500,
    },
  );
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api/v1/${path}`, {
    ...init,
    headers: { "content-type": "application/json", ...init?.headers },
    cache: "no-store",
  });
  if (!response.ok) throw await errorFrom(response);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  paper: (id: string) => call<PaperDetail>(`papers/${encodeURIComponent(id)}`),
  chats: () => call<ChatList>("chats"),
  chat: (id: string) => call<ChatDetail>(`chats/${encodeURIComponent(id)}`),
  deleteChat: (id: string) => call<void>(`chats/${encodeURIComponent(id)}`, { method: "DELETE" }),
};

export type TurnBody = { request_id: string; chat_id: string | null; message: string };

export function streamTurn(body: TurnBody, signal?: AbortSignal): Promise<Response> {
  return fetch("/api/v1/chats/stream", {
    method: "POST",
    headers: { "content-type": "application/json", accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });
}

export function loginRedirect(next: string, expired: boolean): string {
  const query = new URLSearchParams({ next });
  if (expired) query.set("expired", "1");
  return `/login?${query.toString()}`;
}
