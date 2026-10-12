"use client";

import { useRouter } from "next/navigation";
import { useState, type FormEvent } from "react";

export function LoginForm({ next }: { next: string }) {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
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
    setError(
      response?.status === 401
        ? "Email or password is incorrect."
        : "The service is not reachable. Try again in a moment.",
    );
  }

  return (
    <form onSubmit={submit} aria-describedby={error ? "login-error" : undefined}>
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
        {busy ? "Signing in…" : "Sign in"}
      </button>
    </form>
  );
}
