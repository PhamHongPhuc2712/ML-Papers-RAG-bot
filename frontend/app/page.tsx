import Link from "next/link";
import { cookies } from "next/headers";
import { ChatView } from "@/components/ChatView";
import { Workspace } from "@/components/Workspace";
import { SESSION_COOKIE, backendUrl } from "@/lib/server";

type Coverage = { as_of: string; totals: { papers: number; venue_years: number } };

async function coverage(): Promise<Coverage | null> {
  try {
    const response = await fetch(`${backendUrl()}/v1/corpus/coverage`, { next: { revalidate: 300 } });
    return response.ok ? ((await response.json()) as Coverage) : null;
  } catch {
    return null;
  }
}

/** The one entry point (RAG-only plan, U3): every request is a cited answer with its References. */
export default async function Home() {
  if ((await cookies()).get(SESSION_COOKIE)?.value) {
    return (
      <Workspace>
        <ChatView chatId={null} about={null} />
      </Workspace>
    );
  }
  const facts = await coverage();
  return (
    <main>
      <h1>Ask ML papers</h1>
      <p>
        Ask a question, or ask for papers on a topic. Each answer is written from the papers it finds, cites the
        passages it rests on, and lists those papers as its References.
      </p>
      {facts ? (
        <p className="muted">
          {facts.totals.papers.toLocaleString("en")} papers from {facts.totals.venue_years} venue-years, as of{" "}
          {new Date(facts.as_of).toLocaleDateString("en", { dateStyle: "medium" })}.
        </p>
      ) : null}
      <p>
        Every answer uses a paid model, so asking needs an account. <Link href="/login?next=%2F">Sign in to ask</Link>
      </p>
    </main>
  );
}
