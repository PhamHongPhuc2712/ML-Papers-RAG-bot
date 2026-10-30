"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { paperLabel, withPaper, type PaperFacts } from "@/lib/papers";
import { Composer } from "./Composer";

/** "Ask about this paper" (canvas board 8). The paper is written into the question; signed in, the
 *  question is asked at once on a new chat, signed out it waits on the log-in page like the home's. */
export function PaperAsk({ paper, signedIn }: { paper: PaperFacts; signedIn: boolean }) {
  const router = useRouter();
  const [draft, setDraft] = useState("");

  function ask(text: string) {
    const question = encodeURIComponent(withPaper(text, paper));
    if (signedIn) router.push(`/?q=${question}&send=1`);
    else router.push(`/login?next=${encodeURIComponent(`/?q=${question}`)}`);
  }

  return (
    <Composer value={draft} onChange={setDraft} onSubmit={ask} label="Ask about this paper" placeholder="Ask about this paper"
      attachment={<>Asking about: {paperLabel(paper)}</>} />
  );
}
