import type { ChatMessage, Reference, StoredEvidence } from "./api";
import { citationNumbers } from "./chat-labels";

/** The collapsed steps line above an answer (canvas board 6); null when the turn recorded no search. */
export function stepsSummary(message: ChatMessage): string | null {
  const query = message.standalone_query;
  if (!query) return null;
  const references = message.references ?? [];
  // An answer stored before References (2026-10-15) keeps only its query.
  if (!references.length) return `Searched for “${query}”`;
  const read = references.filter((item) => item.read).length;
  return `Searched for “${query}” and read ${read} of ${references.length} papers`;
}

/** Warnings that mean a search stage ran out of time or was missing: the answer may have missed papers. */
export function partialSearch(message: ChatMessage): boolean {
  return (message.warnings ?? []).some(
    (code) => /_(timeout|unavailable|failed)$/.test(code) && code !== "condense_failed",
  );
}

export type ReferenceFilter = { venue: string | null; year: number | null; citedOnly: boolean };

export function referenceFacets(references: Reference[]): { venues: string[]; years: number[] } {
  const venues = new Set<string>();
  const years = new Set<number>();
  for (const item of references) {
    if (item.venue) venues.add(item.venue);
    if (item.year != null) years.add(item.year);
  }
  return { venues: [...venues].sort(), years: [...years].sort((a, b) => b - a) };
}

export function filterReferences(references: Reference[], filter: ReferenceFilter, cited: Map<string, number[]>): Reference[] {
  return references.filter(
    (item) =>
      (filter.venue === null || item.venue === filter.venue) &&
      (filter.year === null || item.year === filter.year) &&
      (!filter.citedOnly || cited.has(item.paper_id)),
  );
}

/** Each paper's citation numbers, in the order the answer first cites the paper. */
export function citedByPaper(message: ChatMessage): Map<string, number[]> {
  const numbers = citationNumbers(message);
  const paperOf = new Map((message.evidence ?? []).map((item) => [item.id, item.paper_id]));
  const cited = new Map<string, number[]>();
  for (const [id, number] of numbers) {
    const paper = paperOf.get(id);
    if (paper) cited.set(paper, [...(cited.get(paper) ?? []), number]);
  }
  return cited;
}

/** One paper's passages in this answer: the cited ones in citation order, then the others it read. */
export function passagesOf(message: ChatMessage, paperId: string): { cited: StoredEvidence[]; read: StoredEvidence[] } {
  const numbers = citationNumbers(message);
  const mine = (message.evidence ?? []).filter((item) => item.paper_id === paperId);
  return {
    cited: mine.filter((item) => numbers.has(item.id)).sort((a, b) => (numbers.get(a.id) ?? 0) - (numbers.get(b.id) ?? 0)),
    read: mine.filter((item) => !numbers.has(item.id)),
  };
}

/** Where a passage sits: "Method, page 3", "Results, pages 3–4", or the section alone. */
export function where(item: StoredEvidence): string {
  if (item.page_start == null) return item.section;
  if (item.page_end == null || item.page_end === item.page_start) return `${item.section}, page ${item.page_start}`;
  return `${item.section}, pages ${item.page_start}–${item.page_end}`;
}

/** The answer as plain text for the clipboard: its claims with their numbers, then the cited papers. */
export function answerAsText(message: ChatMessage): string {
  const numbers = citationNumbers(message);
  const claims = (message.claims ?? []).map(
    (claim) => `${claim.text} ${claim.evidence_ids.map((id) => `[${numbers.get(id)}]`).join("")}`.trim(),
  );
  const evidence = new Map((message.evidence ?? []).map((item) => [item.id, item]));
  const sources = [...numbers].map(([id, n]) => {
    const item = evidence.get(id);
    return item ? `[${n}] ${item.title}${item.venue ? `, ${item.venue}` : ""}${item.year ? ` ${item.year}` : ""}` : `[${n}]`;
  });
  return [...claims, ...(sources.length ? ["", ...sources] : [])].join("\n");
}
