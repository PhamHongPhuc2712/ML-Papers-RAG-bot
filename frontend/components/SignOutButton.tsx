"use client";

import { useRouter } from "next/navigation";
import type { ReactNode } from "react";

/** Ends the session (the cookie is dropped by `/api/session`) and goes to log in. */
export function SignOutButton({ className, children }: { className?: string; children: ReactNode }) {
  const router = useRouter();

  async function signOut() {
    await fetch("/api/session", { method: "DELETE" });
    router.push("/login");
    router.refresh();
  }

  return <button type="button" className={className} onClick={() => void signOut()}>{children}</button>;
}
