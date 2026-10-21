"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState, type FormEvent } from "react";
import { LOGIN_MESSAGES, isLoginError, type LoginError } from "@/lib/login-messages";

type ApiError = { error?: { code?: string; retry_after?: number } };

function messageFor(status: number | undefined, body: ApiError | null): string {
  const code = body?.error?.code;
  if (isLoginError(code)) return LOGIN_MESSAGES[code];
  if (status === 422) return LOGIN_MESSAGES.email;
  return LOGIN_MESSAGES.unreachable;
}

/** Sign in or sign up by emailed code (canvas boards 2–3): an email, then the 6-digit code sent to
 *  it. An email without an account gets one on its first correct code. Both steps are real forms,
 *  so they work before the page's script runs; the script posts JSON and keeps the page. */
export function LoginForm({
  next,
  initialStep = "email",
  initialEmail = "",
  initialError = null,
}: {
  next: string;
  initialStep?: "email" | "code";
  initialEmail?: string;
  initialError?: LoginError | null;
}) {
  const router = useRouter();
  const [step, setStep] = useState(initialStep);
  const [email, setEmail] = useState(initialEmail);
  const [error, setError] = useState<string | null>(initialError ? LOGIN_MESSAGES[initialError] : null);
  const [busy, setBusy] = useState(false);
  const [resendIn, setResendIn] = useState(0);

  useEffect(() => {
    if (resendIn <= 0) return;
    const timer = setTimeout(() => setResendIn((left) => left - 1), 1000);
    return () => clearTimeout(timer);
  }, [resendIn]);

  async function ask(address: string): Promise<void> {
    setBusy(true);
    setError(null);
    const response = await fetch("/api/session/code", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ email: address }),
    }).catch(() => null);
    const body = ((await response?.json().catch(() => null)) ?? null) as (ApiError & { resend_after?: number }) | null;
    setBusy(false);
    if (response?.ok || response?.status === 429) {
      setEmail(address.trim().toLowerCase());
      setStep("code");
      setResendIn(response.ok ? (body?.resend_after ?? 60) : (body?.error?.retry_after ?? 60));
      if (!response.ok) setError(`A code was sent less than a minute ago. You can ask for a new one in ${body?.error?.retry_after ?? 60} s.`);
      return;
    }
    setError(messageFor(response?.status, body));
  }

  async function submitEmail(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await ask(String(new FormData(event.currentTarget).get("email") ?? ""));
  }

  async function submitCode(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const code = String(new FormData(event.currentTarget).get("code") ?? "").replace(/\D/g, "");
    setBusy(true);
    setError(null);
    const response = await fetch("/api/session", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ email, code }),
    }).catch(() => null);
    if (response?.ok) {
      router.replace(next);
      router.refresh();
      return;
    }
    const body = ((await response?.json().catch(() => null)) ?? null) as ApiError | null;
    setBusy(false);
    setError(response?.status === 422 ? LOGIN_MESSAGES.invalid_code : messageFor(response?.status, body));
  }

  async function resend(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await ask(email);
  }

  const alert = error ? (
    <p id="login-error" className="banner error" role="alert">
      {error}
    </p>
  ) : null;

  if (step === "email") {
    return (
      <form className="auth-form" method="post" action="/api/session/code" onSubmit={submitEmail}
        aria-describedby={error ? "login-error" : undefined}>
        <input type="hidden" name="next" value={next} />
        <p>
          <label htmlFor="email">Email</label>
          <input id="email" name="email" type="email" autoComplete="email" defaultValue={email} required />
        </p>
        {alert}
        <button className="primary" type="submit" disabled={busy}>
          {busy ? "Sending a code…" : "Continue"}
        </button>
        <p className="auth-fine">We email you a 6-digit code. No password needed; a new email gets a new account.</p>
      </form>
    );
  }

  return (
    <div className="auth-steps">
      <p className="auth-note" role="status">
        We sent a 6-digit code to <strong>{email}</strong>. It works once, for 10 minutes.
      </p>
      <form className="auth-form" method="post" action="/api/session" onSubmit={submitCode}
        aria-describedby={error ? "login-error" : undefined}>
        <input type="hidden" name="next" value={next} />
        <input type="hidden" name="email" value={email} />
        <p>
          <label htmlFor="code">Code</label>
          <input id="code" name="code" type="text" inputMode="numeric" autoComplete="one-time-code"
            pattern="[0-9 ]{6,7}" maxLength={7} required autoFocus />
        </p>
        {alert}
        <button className="primary" type="submit" disabled={busy}>
          {busy ? "Checking…" : "Log in"}
        </button>
      </form>
      <form className="auth-links" method="post" action="/api/session/code" onSubmit={resend}>
        <input type="hidden" name="next" value={next} />
        <input type="hidden" name="email" value={email} />
        <button type="submit" className="link-button" disabled={busy || resendIn > 0}>
          {resendIn > 0 ? `Send a new code in ${resendIn} s` : "Send a new code"}
        </button>
        <a href={`/login?${new URLSearchParams({ next, step: "email" })}`}
          onClick={(event) => { event.preventDefault(); setStep("email"); setError(null); }}>
          Use another email
        </a>
      </form>
    </div>
  );
}
