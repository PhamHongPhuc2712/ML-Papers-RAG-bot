"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useId, useRef, useState } from "react";
import { Icon } from "./Icon";
import { ThemeToggle } from "./ThemeToggle";

/** The sidebar's bottom: who is signed in, the account page, the theme and signing out. */
export function AccountMenu({ email }: { email: string | null }) {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  const button = useRef<HTMLButtonElement>(null);
  const id = useId();

  useEffect(() => {
    if (!open) return;
    box.current?.querySelector<HTMLElement>(".menu a, .menu button")?.focus();
    const outside = (event: PointerEvent) => {
      if (!box.current?.contains(event.target as Node)) setOpen(false);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false);
        button.current?.focus();
      }
    };
    document.addEventListener("pointerdown", outside);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("pointerdown", outside);
      document.removeEventListener("keydown", escape);
    };
  }, [open]);

  async function signOut() {
    await fetch("/api/session", { method: "DELETE" });
    router.push("/login");
    router.refresh();
  }

  const name = email ?? "Your account";
  return (
    <div className="sidebar-bottom" ref={box}>
      {open ? (
        <div className="menu" id={id}>
          {email ? <p className="menu-section" style={{ margin: 0 }}>{email}</p> : null}
          <Link className="menu-item" href="/account" onClick={() => setOpen(false)}>
            <Icon name="user" size={18} /> Account
          </Link>
          <hr />
          <p className="menu-section" style={{ margin: 0 }}>Theme</p>
          <ThemeToggle />
          <hr />
          <button type="button" className="menu-item" onClick={() => void signOut()}>
            <Icon name="signout" size={18} /> Sign out
          </button>
        </div>
      ) : null}
      <button ref={button} type="button" className="account-button" aria-expanded={open} aria-controls={id}
        aria-label={`Account menu for ${name}`} onClick={() => setOpen((value) => !value)}>
        <span className="avatar" aria-hidden="true">{name.charAt(0).toUpperCase()}</span>
        <span className="account-email">{name}</span>
        <Icon name="chevrons" size={16} className="label" />
      </button>
    </div>
  );
}
