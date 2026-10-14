"use client";

import { useRouter } from "next/navigation";
import { useState, type FormEvent } from "react";

export function DeleteAccount() {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const password = new FormData(event.currentTarget).get("password");
    setBusy(true);
    setError(null);
    const response = await fetch("/api/v1/me", {
      method: "DELETE",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ password }),
    }).catch(() => null);
    if (response?.status === 204) {
      await fetch("/api/session", { method: "DELETE" }).catch(() => null);
      router.replace("/");
      router.refresh();
      return;
    }
    setBusy(false);
    setError(
      response?.status === 401
        ? "The password is incorrect."
        : "The account could not be deleted. Try again in a moment.",
    );
  }

  return (
    <form onSubmit={submit} aria-describedby={error ? "delete-error" : undefined}>
      <p>
        <label htmlFor="delete-password">Password</label>
        <input
          id="delete-password"
          name="password"
          type="password"
          autoComplete="current-password"
          required
        />
      </p>
      {error ? (
        <p id="delete-error" className="banner error" role="alert">
          {error}
        </p>
      ) : null}
      <button className="primary" type="submit" disabled={busy}>
        {busy ? "Deleting…" : "Delete my account"}
      </button>
    </form>
  );
}
