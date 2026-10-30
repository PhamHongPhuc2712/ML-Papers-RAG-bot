/** A paper as the paper page and its ask box describe it. */
export type PaperFacts = { title: string; authors: string[]; venue: string; year: number };

/** "Title (ACL 2024)": the paper named in a question and on the ask box's chip. */
export function paperLabel(paper: PaperFacts): string {
  return `${paper.title} (${paper.venue} ${paper.year})`;
}

/** The paper written into the question, since a turn has no way to attach one yet: the search
 *  then reads its title with the question. */
export function withPaper(text: string, paper: PaperFacts): string {
  return `${text}\n\nAbout the paper “${paperLabel(paper)}”.`;
}

/** A plain citation for the clipboard: "A, B and C. Title. ACL 2024." */
export function citationText(paper: PaperFacts, url?: string | null): string {
  const authors = paper.authors.length > 1
    ? `${paper.authors.slice(0, -1).join(", ")} and ${paper.authors.at(-1)}`
    : (paper.authors[0] ?? "");
  const parts = [authors, paper.title, `${paper.venue} ${paper.year}`].filter(Boolean).map((part) => part.replace(/\.$/, ""));
  return `${parts.join(". ")}.${url ? ` ${url}` : ""}`;
}
