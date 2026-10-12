import { cookies } from "next/headers";
import { SESSION_COOKIE, backendUrl, cookieSecure } from "@/lib/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const unreachable = {
  error: { code: "backend_unreachable", message: "The service is not reachable.", retryable: true },
};

export async function POST(request: Request): Promise<Response> {
  const body = (await request.json().catch(() => null)) as { email?: unknown; password?: unknown } | null;
  if (typeof body?.email !== "string" || typeof body.password !== "string") {
    return Response.json(
      { error: { code: "validation_error", message: "Email and password are required.", retryable: false } },
      { status: 422 },
    );
  }
  let upstream: Response;
  try {
    upstream = await fetch(`${backendUrl()}/v1/auth/token`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ email: body.email, password: body.password }),
      cache: "no-store",
    });
  } catch {
    return Response.json(unreachable, { status: 503 });
  }
  const payload = (await upstream.json().catch(() => null)) as
    | { access_token?: string; expires_in?: number }
    | null;
  if (!upstream.ok || !payload?.access_token) {
    return Response.json(payload ?? unreachable, { status: upstream.ok ? 502 : upstream.status });
  }
  (await cookies()).set(SESSION_COOKIE, payload.access_token, {
    httpOnly: true,
    sameSite: "lax",
    secure: cookieSecure(),
    path: "/",
    maxAge: payload.expires_in ?? 3600,
  });
  return Response.json({ ok: true });
}

export async function DELETE(): Promise<Response> {
  (await cookies()).delete(SESSION_COOKIE);
  return new Response(null, { status: 204 });
}
