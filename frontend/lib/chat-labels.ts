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

export const INSUFFICIENT =
  "The indexed papers don't contain enough evidence to answer this. Try naming a paper or method, or ask more specifically.";

/** Evidence ids numbered in the order the answer first cites them. */
export function citationNumbers(message: ChatMessage): Map<string, number> {
  const numbers = new Map<string, number>();
  for (const claim of message.claims ?? []) {
    for (const id of claim.evidence_ids) if (!numbers.has(id)) numbers.set(id, numbers.size + 1);
  }
  return numbers;
}
