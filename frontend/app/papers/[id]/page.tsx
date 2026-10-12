import Link from "next/link";
import { notFound } from "next/navigation";
import { PaperCard, authorLine } from "@/components/PaperCard";
import type { PaperDetail, SearchResponse } from "@/lib/api";
import { backendUrl } from "@/lib/server";

type Props = { params: Promise<{ id: string }> };

async function load<T>(path: string): Promise<T | null> {
  const response = await fetch(`${backendUrl()}${path}`, { cache: "no-store" });
  if (response.status === 404 || response.status === 422) return null;
  if (!response.ok) throw new Error(`paper request failed: ${response.status}`);
  return (await response.json()) as T;
}

export default async function PaperPage({ params }: Props) {
  const { id } = await params;
  const detail = await load<PaperDetail>(`/v1/papers/${encodeURIComponent(id)}`);
  if (!detail) notFound();
  const related = await load<SearchResponse>(`/v1/papers/${encodeURIComponent(detail.paper_id)}/related?limit=5`).catch(() => null);
  const paper = detail.paper;
  return (
    <main>
      <h1>{paper.title}</h1>
      <p className="muted">
        {authorLine(paper.authors)} · {paper.venue} {paper.year}
      </p>
      <p>
        {detail.source.pdf_url ? <a href={detail.source.pdf_url} rel="noopener noreferrer">PDF</a> : null}
        {detail.source.landing_url ? <> · <a href={detail.source.landing_url} rel="noopener noreferrer">Source page</a></> : null}
        {" · "}
        <Link href={`/chat?about=${detail.paper_id}`}>Ask about this paper</Link>
      </p>
      <h2>Abstract</h2>
      <p>{paper.abstract ?? "No abstract is available for this paper."}</p>
      {paper.fulltext_indexed ? null : <p className="banner warn">Only the abstract of this paper is indexed; answers about it can use only the abstract.</p>}
      <h2>Related papers</h2>
      {related && related.items.length ? (
        related.items.map((item) => {
          const summary = related.papers[item.paper_id];
          return summary ? <PaperCard key={item.paper_id} paperId={item.paper_id} paper={summary} /> : null;
        })
      ) : (
        <p className="muted">No related papers were found.</p>
      )}
    </main>
  );
}
