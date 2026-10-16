import type { ReactNode } from "react";
import { ChatSidebar } from "./ChatSidebar";

/** The signed-in page: the chat history beside one conversation. */
export function Workspace({ children }: { children: ReactNode }) {
  return (
    <main style={{ display: "grid", gridTemplateColumns: "minmax(160px, 240px) 1fr", gap: 16, maxWidth: 1200 }}>
      <ChatSidebar />
      <section>{children}</section>
    </main>
  );
}
