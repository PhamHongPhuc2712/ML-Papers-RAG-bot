import { cookies } from "next/headers";
import { safeNext } from "@/lib/next-path";
import { LOGIN_EMAIL_COOKIE, backendUrl, cookieSecure } from "@/lib/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const unreachable = {
  error: { code: "backend_unreachable", message: "The service is not reachable.", retryable: true },
};

type Outcome = { ok: true; status: number; body: unknown } | { ok: false; status: number; body: unknown };

/** Ask the API to email a sign-in code (sign-up plan, A2). The email is kept in a short-lived
 *  httpOnly cookie, so the code step knows it without putting it in a URL. */
async function askForCode(email: unknown): Promise<Outcome> {
  if (typeof email !== "string") {
    return { ok: false, status: 422, body: { error: { code: "validation_error", message: "An email is required.", retryable: false } } };
  }
  let upstream: Response;
  try {
    upstream = await fetch(`${backendUrl()}/v1/auth/code`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ email }),
      cache: "no-store",
    });
  } catch {
    return { ok: false, status: 503, body: unreachable };
  }
  const body = await upstream.json().catch(() => unreachable);
  // A 429 means a code went out less than a minute ago: the code step is still the right place.
  if (upstream.ok || upstream.status === 429) {
    (await cookies()).set(LOGIN_EMAIL_COOKIE, email.trim().toLowerCase(), {
      httpOnly: true,
      sameSite: "lax",
      secure: cookieSecure(),
      path: "/",
      maxAge: 15 * 60,
    });
  }
  return { ok: upstream.ok, status: upstream.status, body };
}

/** The form posted before the page's script ran: ask, then return to the page's code step. */
async function fromForm(request: Request): Promise<Response> {
  const form = await request.formData();
  const next = safeNext(typeof form.get("next") === "string" ? (form.get("next") as string) : null);
  const outcome = await askForCode(form.get("email"));
  const params = new URLSearchParams({ next });
  if (outcome.ok || outcome.status === 429) params.set("step", "code");
  else params.set("error", outcome.status === 422 ? "email" : "unreachable");
  return new Response(null, { status: 303, headers: { location: `/login?${params}` } });
}

export async function POST(request: Request): Promise<Response> {
  if (request.headers.get("content-type")?.startsWith("application/x-www-form-urlencoded")) return fromForm(request);
  const payload = (await request.json().catch(() => null)) as { email?: unknown } | null;
  const outcome = await askForCode(payload?.email);
  return Response.json(outcome.body, { status: outcome.status });
}
