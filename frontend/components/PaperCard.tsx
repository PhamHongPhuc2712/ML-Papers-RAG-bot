import Link from "next/link";
import type { SearchResponse } from "@/lib/api";

type Summary = SearchResponse["papers"][string];

export function authorLine(authors: string[]): string {
  return authors.length > 3 ? `${authors.slice(0, 3).join(", ")} et al.` : authors.join(", ");
}

export function PaperCard({ paperId, paper }: { paperId: string; paper: Summary }) {
  const abstract = paper.abstract ?? "";
  return (
    <article style={{ borderBottom: "1px solid var(--border)", padding: "12px 0" }}>
      <h3 style={{ margin: 0 }}>
        <Link href={`/papers/${paperId}`}>{paper.title}</Link>
      </h3>
      <p className="muted" style={{ margin: "4px 0" }}>
        {authorLine(paper.authors)} · {paper.venue} {paper.year}
        {paper.fulltext_indexed ? " · Full text indexed" : " · Abstract only"}
      </p>
      {abstract ? <p style={{ margin: 0 }}>{abstract.length > 320 ? `${abstract.slice(0, 317)}…` : abstract}</p> : null}
    </article>
  );
}
