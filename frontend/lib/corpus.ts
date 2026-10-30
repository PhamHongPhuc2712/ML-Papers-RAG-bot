export type Coverage = {
  as_of: string;
  venue_years: { venue: string; year: number; papers: number }[];
  totals: { papers: number; venue_years: number };
};

/** What the home page and the filter say about the corpus, from `GET /v1/corpus/coverage`. */
export type Corpus = { papers: number; venues: string[]; years: number[] };

export function summarizeCoverage(coverage: Coverage): Corpus {
  const venues = [...new Set(coverage.venue_years.map((v) => v.venue))].sort((a, b) => a.localeCompare(b));
  const years = [...new Set(coverage.venue_years.map((v) => v.year))].sort((a, b) => b - a);
  return { papers: coverage.totals.papers, venues, years };
}

/** "A, B and C": the venues in one sentence. */
export function listInWords(items: string[]): string {
  if (items.length <= 1) return items.join("");
  return `${items.slice(0, -1).join(", ")} and ${items.at(-1)}`;
}

/** The home page's line under the suggestions; null when the corpus is unknown. */
export function corpusLine(corpus: Corpus | null): string | null {
  if (!corpus || corpus.venues.length === 0) return null;
  const first = Math.min(...corpus.years);
  const last = Math.max(...corpus.years);
  const span = first === last ? `in ${first}` : `from ${first} to ${last}`;
  return `Searches the main-conference papers of ${listInWords(corpus.venues)} ${span}.`;
}

/** A filter chosen in the ask box, written into the question, since the search takes no filter yet. */
export type AskFilter = { venue: string | null; year: number | null };

export function withFilter(text: string, filter: AskFilter): string {
  if (!filter.venue && !filter.year) return text;
  const scope = [filter.venue ? `${filter.venue}` : null, filter.year ? `${filter.year}` : null].filter(Boolean).join(" ");
  return `${text}\n\nOnly use papers from ${scope}.`;
}

export function filterLabel(filter: AskFilter): string | null {
  const parts = [filter.venue, filter.year?.toString()].filter(Boolean);
  return parts.length ? parts.join(", ") : null;
}
