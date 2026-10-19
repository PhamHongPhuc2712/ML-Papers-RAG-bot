import { cookies } from "next/headers";
import { SESSION_COOKIE, backendUrl } from "@/lib/server";
import { endsSession } from "@/lib/session-errors";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type Context = { params: Promise<{ path: string[] }> };

const REQUEST_HEADERS = ["content-type", "accept"];
const RESPONSE_HEADERS = ["content-type", "cache-control"];

function failure(status: number, code: string, message: string, retryable: boolean): Response {
  return Response.json({ error: { code, message, retryable } }, { status });
}

async function forward(request: Request, { params }: Context): Promise<Response> {
  const { path } = await params;
  // Tokens are issued only to /api/session, which keeps them out of page scripts.
  if (path[0] === "auth") return failure(404, "not_found", "No such endpoint.", false);
  const search = new URL(request.url).search;
  const target = `${backendUrl()}/v1/${path.map(encodeURIComponent).join("/")}${search}`;
  const headers = new Headers();
  for (const name of REQUEST_HEADERS) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  const jar = await cookies();
  const token = jar.get(SESSION_COOKIE)?.value;
  if (token) headers.set("authorization", `Bearer ${token}`);
  const init: RequestInit = { method: request.method, headers, cache: "no-store", redirect: "manual" };
  if (request.method !== "GET" && request.method !== "HEAD") init.body = await request.text();
  let upstream: Response;
  try {
    upstream = await fetch(target, init);
  } catch {
    return failure(503, "backend_unreachable", "The service can’t be reached right now. Your chats are safe. Try again in a minute.", true);
  }
  const out = new Headers();
  for (const name of RESPONSE_HEADERS) {
    const value = upstream.headers.get(name);
    if (value) out.set(name, value);
  }
  if (upstream.status === 401) {
    // An error body is small: read it to tell an ended session from a wrong password.
    const text = await upstream.text();
    if (token && endsSession(upstream.status, text)) jar.delete(SESSION_COOKIE);
    return new Response(text, { status: upstream.status, headers: out });
  }
  // The body streams through unchanged: chat events arrive as the backend sends them.
  return new Response(upstream.body, { status: upstream.status, headers: out });
}

export const GET = forward;
export const POST = forward;
export const DELETE = forward;
