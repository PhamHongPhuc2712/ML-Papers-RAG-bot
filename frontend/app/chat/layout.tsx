import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import type { ReactNode } from "react";
import { ChatSidebar } from "@/components/ChatSidebar";
import { SESSION_COOKIE } from "@/lib/server";

export default async function ChatLayout({ children }: { children: ReactNode }) {
  if (!(await cookies()).get(SESSION_COOKIE)?.value) redirect("/login?next=%2Fchat");
  return (
    <main style={{ display: "grid", gridTemplateColumns: "minmax(160px, 240px) 1fr", gap: 16, maxWidth: 1200 }}>
      <ChatSidebar />
      <section>{children}</section>
    </main>
  );
}
