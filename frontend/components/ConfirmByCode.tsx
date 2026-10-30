"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import { LOGIN_MESSAGES, isLoginError } from "@/lib/login-messages";

type Props = {
  /** The row's own words and its button (canvas board 10). */
  description: string;
  action: string;
  /** What deleting removes, said again in the open form before the code is asked for. */
  warning: string;
  confirm: string;
  /** The API path the code is sent to as DELETE {code}: "me" or "chats". */
  path: "me" | "chats";
  onDone: () => void | Promise<void>;
};

/** A delete confirmed by a code emailed to the account (sign-up plan, A4): the button sends the
 *  code (POST /v1/me/code), the form takes it, and nothing is deleted before it is right. */
export function ConfirmByCode({ description, action, warning, confirm, path, onDone }: Props) {
  const [open, setOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const field = useRef<HTMLInputElement>(null);
  const opener = useRef<HTMLButtonElement>(null);
  const id = `confirm-${path}`;

  useEffect(() => {
    if (open) field.current?.focus();
  }, [open]);

  async function sendCode(): Promise<boolean> {
    setBusy(true);
    setError(null);
    const response = await fetch("/api/v1/me/code", { method: "POST" }).catch(() => null);
    setBusy(false);
    if (response?.ok) return true;
    const body = (await response?.json().catch(() => null)) as { error?: { code?: string; retry_after?: number } } | null;
    if (response?.status === 429) {
      // A code went out less than a minute ago: the one already sent still works.
      setError(`A code was sent less than a minute ago. Use it, or ask again in ${body?.error?.retry_after ?? 60} s.`);
      return true;
    }
    setError(LOGIN_MESSAGES.mail_unavailable);
    return false;
  }

  async function start() {
    if (await sendCode()) setOpen(true);
  }

  function cancel() {
    setOpen(false);
    setError(null);
    // The opener is rendered again once the form is gone.
    window.requestAnimationFrame(() => opener.current?.focus());
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const code = String(new FormData(event.currentTarget).get("code") ?? "").replace(/\D/g, "");
    setBusy(true);
    setError(null);
    const response = await fetch(`/api/v1/${path}`, {
      method: "DELETE",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ code }),
    }).catch(() => null);
    if (response?.status === 204) {
      setOpen(false);
      setBusy(false);
      await onDone();
      return;
    }
    const body = (await response?.json().catch(() => null)) as { error?: { code?: string } } | null;
    setBusy(false);
    const known = body?.error?.code;
    setError(response?.status === 422 ? LOGIN_MESSAGES.invalid_code
      : isLoginError(known) ? LOGIN_MESSAGES[known] : "That did not work. Try again in a moment.");
  }

  if (!open) {
    return (
      <div className="settings-row">
        <p>{description}</p>
        <button ref={opener} type="button" className="danger" onClick={() => void start()} disabled={busy}>
          {busy ? "Sending a code…" : action}
        </button>
        {error ? <p id={`${id}-error`} className="banner error" role="alert">{error}</p> : null}
      </div>
    );
  }
  return (
    <form className="delete-account" onSubmit={submit} aria-describedby={error ? `${id}-error` : `${id}-note`}>
      <p id={`${id}-note`}>{warning} We emailed you a 6-digit code; enter it to confirm.</p>
      <p>
        <label htmlFor={`${id}-code`}>Code</label>
        <input ref={field} id={`${id}-code`} name="code" type="text" inputMode="numeric" autoComplete="one-time-code"
          pattern="[0-9 ]{6,7}" maxLength={7} required />
      </p>
      {error ? (
        <p id={`${id}-error`} className="banner error" role="alert">
          {error}
        </p>
      ) : null}
      <div className="form-actions">
        <button type="button" onClick={cancel} disabled={busy}>Cancel</button>
        <button className="danger solid" type="submit" disabled={busy}>
          {busy ? "Deleting…" : confirm}
        </button>
      </div>
    </form>
  );
}
