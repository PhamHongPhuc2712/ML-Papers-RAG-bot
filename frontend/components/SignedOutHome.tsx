"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { corpusLine, withFilter, type AskFilter, type Corpus } from "@/lib/corpus";
import { Composer } from "./Composer";
import { Hero } from "./Hero";
import { Suggestions } from "./Suggestions";

/** Home before signing in (canvas board 1): asking leads to log in, and the question waits there. */
export function SignedOutHome({ corpus }: { corpus: Corpus | null }) {
  const router = useRouter();
  const [draft, setDraft] = useState("");
  const line = corpusLine(corpus);

  function ask(text: string, filter: AskFilter = { venue: null, year: null }) {
    const back = `/?q=${encodeURIComponent(withFilter(text, filter))}`;
    router.push(`/login?next=${encodeURIComponent(back)}`);
  }

  return (
    <main className="ask-home">
      <Hero />
      <Composer value={draft} onChange={setDraft} onSubmit={ask} placeholder="Ask a question, or ask for papers on a topic"
        venues={corpus?.venues} years={corpus?.years} />
      <Suggestions onPick={(question) => ask(question)} />
      {line ? <p className="corpus-line">{line}</p> : null}
    </main>
  );
}
