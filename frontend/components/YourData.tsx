"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { ConfirmByCode } from "./ConfirmByCode";

/** Settings' "Your data" (canvas board 10, sign-up plan A4): delete all chats, delete the account,
 *  each confirmed by a code emailed to the account. */
export function YourData() {
  const router = useRouter();
  const [cleared, setCleared] = useState(false);
  return (
    <div className="settings-data">
      <ConfirmByCode
        path="chats"
        description="Delete all of your chats. Today's answers stay counted."
        action="Delete all chats"
        warning="Every chat and its answers will be deleted. This cannot be undone."
        confirm="Delete all chats"
        onDone={() => {
          setCleared(true);
          window.dispatchEvent(new Event("chats-changed"));
          router.refresh();
        }}
      />
      {cleared ? <p role="status" className="settings-done">All of your chats are deleted.</p> : null}
      <ConfirmByCode
        path="me"
        description="Delete your account and all of its chats. This cannot be undone."
        action="Delete account"
        warning="Deleting your account removes all of your chats at once. This cannot be undone."
        confirm="Delete my account"
        onDone={async () => {
          await fetch("/api/session", { method: "DELETE" }).catch(() => null);
          router.replace("/");
          router.refresh();
        }}
      />
    </div>
  );
}
