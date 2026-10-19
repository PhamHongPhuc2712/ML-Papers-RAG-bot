import { cookies } from "next/headers";
import { ChatView } from "@/components/ChatView";
import { SignedOutHome } from "@/components/SignedOutHome";
import { Workspace } from "@/components/Workspace";
import { SESSION_COOKIE, corpus } from "@/lib/server";

type Props = { searchParams: Promise<{ q?: string; send?: string }> };

/** The one entry point (RAG-only plan, U3; canvas boards 1 and 4). `?q=` is a question kept across log in;
 *  with `send=1` (a paper page's ask box) it is asked at once. */
export default async function Home({ searchParams }: Props) {
  const [{ q, send }, facts] = await Promise.all([searchParams, corpus()]);
  if (!(await cookies()).get(SESSION_COOKIE)?.value) return <SignedOutHome corpus={facts} />;
  return (
    <Workspace>
      <ChatView chatId={null} initialDraft={q?.slice(0, 2000) ?? null} autoSend={send === "1"} corpus={facts} />
    </Workspace>
  );
}
