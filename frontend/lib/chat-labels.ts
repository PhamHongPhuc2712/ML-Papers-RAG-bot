import type { ChatMessage } from "./api";

export const STAGES: Record<string, string> = {
  condensing: "Reading your follow-up…",
  searching: "Searching papers…",
  writing: "Writing an answer from the passages…",
};

export const WARNINGS: Record<string, string> = {
  condense_failed: "Your follow-up was searched as written.",
  answer_failed_validation: "The model's answer did not cite its sources correctly, so it was withheld.",
};

/** Board 11's wording for failures whose API message is generic; the code is the error frame's. */
export const ERRORS: Record<string, string> = {
  candidates_unavailable: "The answer could not be finished because the search timed out.",
  corpus_not_ready: "The paper index isn't ready yet.",
};

export const INSUFFICIENT = "The papers this search found don’t say enough to answer that.";

/** What to try after an insufficient answer; the papers it found stay in its References. */
export function insufficientHint(references: number): string {
  const kept = references ? ` The ${references} papers it found are still in References.` : "";
  return `Try naming a method or a paper, or ask a narrower question.${kept}`;
}

/** Evidence ids numbered in the order the answer first cites them. */
export function citationNumbers(message: ChatMessage): Map<string, number> {
  const numbers = new Map<string, number>();
  for (const claim of message.claims ?? []) {
    for (const id of claim.evidence_ids) if (!numbers.has(id)) numbers.set(id, numbers.size + 1);
  }
  return numbers;
}
