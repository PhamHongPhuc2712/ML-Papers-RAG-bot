"use client";

import { useRouter } from "next/navigation";
import { useState, type FormEvent } from "react";

const MESSAGES = {
  credentials: "Email or password is incorrect.",
  unreachable: "The service is not reachable. Try again in a moment.",
};

/** Email and password. Posted by the script as JSON; before the script runs, the browser posts the
 *  form itself to the same route, which redirects (`initialError` is how a failure comes back). */
export function LoginForm({ next, initialError = null }: { next: string; initialError?: keyof typeof MESSAGES | null }) {
  const router = useRouter();
  const [error, setError] = useState<string | null>(initialError ? MESSAGES[initialError] : null);
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    setBusy(true);
    setError(null);
    const response = await fetch("/api/session", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ email: form.get("email"), password: form.get("password") }),
    }).catch(() => null);
    setBusy(false);
    if (response?.ok) {
      router.replace(next);
      router.refresh();
      return;
    }
    setError(response?.status === 401 ? MESSAGES.credentials : MESSAGES.unreachable);
  }

  return (
    <form className="auth-form" method="post" action="/api/session" onSubmit={submit}
      aria-describedby={error ? "login-error" : undefined}>
      <input type="hidden" name="next" value={next} />
      <p>
        <label htmlFor="email">Email</label>
        <input id="email" name="email" type="email" autoComplete="username" required />
      </p>
      <p>
        <label htmlFor="password">Password</label>
        <input id="password" name="password" type="password" autoComplete="current-password" required />
      </p>
      {error ? (
        <p id="login-error" className="banner error" role="alert">
          {error}
        </p>
      ) : null}
      <button className="primary" type="submit" disabled={busy}>
        {busy ? "Logging in…" : "Log in"}
      </button>
    </form>
  );
}
