"use client";

import { useRouter } from "next/navigation";

export function SignOutButton() {
  const router = useRouter();
  return (
    <button
      type="button"
      onClick={async () => {
        await fetch("/api/session", { method: "DELETE" });
        router.push("/login");
        router.refresh();
      }}
    >
      Sign out
    </button>
  );
}
