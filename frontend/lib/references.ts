import type { ChatMessage, Reference } from "./api";
import { citationNumbers } from "./chat-labels";

/** The answer's header: what the turn searched and how many papers it read (RAG-only plan, U3). */
export function searchLine(message: ChatMessage): string | null {
  const query = message.standalone_query;
  if (!query) return null;
  const references = message.references ?? [];
  // An answer stored before References (2026-10-15) keeps showing only a follow-up's query.
  if (!references.length) return message.ordinal > 1 ? `Searched for “${query}”` : null;
  const read = references.filter((item) => item.read).length;
  return `Searched for “${query}” · read ${read} paper${read === 1 ? "" : "s"}`;
}

export type ReferenceFilter = { venue: string | null; year: number | null };

export function referenceFacets(references: Reference[]): { venues: string[]; years: number[] } {
  const venues = new Set<string>();
  const years = new Set<number>();
  for (const item of references) {
    if (item.venue) venues.add(item.venue);
    if (item.year != null) years.add(item.year);
  }
  return { venues: [...venues].sort(), years: [...years].sort((a, b) => b - a) };
}

export function filterReferences(references: Reference[], filter: ReferenceFilter): Reference[] {
  return references.filter(
    (item) => (filter.venue === null || item.venue === filter.venue) && (filter.year === null || item.year === filter.year),
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
