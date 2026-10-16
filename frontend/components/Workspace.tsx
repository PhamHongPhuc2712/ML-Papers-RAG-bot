import type { ReactNode } from "react";

/** A conversation's column inside the shell (the sidebar holds the chat history since D1). */
export function Workspace({ children }: { children: ReactNode }) {
  return <main style={{ maxWidth: 820 }}>{children}</main>;
}
