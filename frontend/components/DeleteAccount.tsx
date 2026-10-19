"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState, type FormEvent } from "react";

/** Deleting the account (canvas board 10): a button that asks for the password before anything goes. */
export function DeleteAccount() {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const field = useRef<HTMLInputElement>(null);
  const opener = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (open) field.current?.focus();
  }, [open]);

  function cancel() {
    setOpen(false);
    setError(null);
    // The opener is rendered again once the form is gone.
    window.requestAnimationFrame(() => opener.current?.focus());
  }

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

  if (!open) {
    return (
      <div className="settings-row">
        <p>Delete your account and all of its chats. This cannot be undone.</p>
        <button ref={opener} type="button" className="danger" onClick={() => setOpen(true)}>Delete account</button>
      </div>
    );
  }
  return (
    <form className="delete-account" onSubmit={submit} aria-describedby={error ? "delete-error" : "delete-note"}>
      <p id="delete-note">Deleting your account removes all of your chats at once. This cannot be undone. Enter your password to confirm.</p>
      <p>
        <label htmlFor="delete-password">Password</label>
        <input ref={field} id="delete-password" name="password" type="password" autoComplete="current-password" required />
      </p>
      {error ? (
        <p id="delete-error" className="banner error" role="alert">
          {error}
        </p>
      ) : null}
      <div className="form-actions">
        <button type="button" onClick={cancel} disabled={busy}>Cancel</button>
        <button className="danger solid" type="submit" disabled={busy}>
          {busy ? "Deleting…" : "Delete my account"}
        </button>
      </div>
    </form>
  );
}
