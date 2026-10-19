import { cookies } from "next/headers";
import Link from "next/link";
import { notFound } from "next/navigation";
import { CopyButton } from "@/components/CopyButton";
import { Icon } from "@/components/Icon";
import { PaperAsk } from "@/components/PaperAsk";
import type { PaperDetail, SearchResponse } from "@/lib/api";
import { citationText } from "@/lib/papers";
import { SESSION_COOKIE, backendUrl } from "@/lib/server";

type Props = { params: Promise<{ id: string }> };

async function load<T>(path: string): Promise<T | null> {
  const response = await fetch(`${backendUrl()}${path}`, { cache: "no-store" });
  if (response.status === 404 || response.status === 422) return null;
  if (!response.ok) throw new Error(`paper request failed: ${response.status}`);
  return (await response.json()) as T;
}

/** A paper (canvas board 8): its venue, authors and links, a box to ask about it, the abstract,
 *  whether its full text is indexed, and the papers nearest to it. */
export default async function PaperPage({ params }: Props) {
  const { id } = await params;
  const detail = await load<PaperDetail>(`/v1/papers/${encodeURIComponent(id)}`);
  if (!detail) notFound();
  const [related, signedIn] = await Promise.all([
    load<SearchResponse>(`/v1/papers/${encodeURIComponent(detail.paper_id)}/related?limit=5`).catch(() => null),
    cookies().then((jar) => Boolean(jar.get(SESSION_COOKIE)?.value)),
  ]);
  const paper = detail.paper;
  const facts = { title: paper.title, authors: paper.authors, venue: paper.venue, year: paper.year };
  return (
    <main className="page paper-page">
      <p className="paper-venue">{paper.venue} {paper.year}</p>
      <h1>{paper.title}</h1>
      {paper.authors.length ? <p className="paper-authors">{paper.authors.join(", ")}</p> : null}
      <p className="paper-actions">
        {detail.source.pdf_url ? (
          <a className="button-link primary" href={detail.source.pdf_url} rel="noopener noreferrer" target="_blank">
            PDF <Icon name="external" size={14} />
          </a>
        ) : null}
        {detail.source.landing_url ? (
          <a className="button-link" href={detail.source.landing_url} rel="noopener noreferrer" target="_blank">
            Source page <Icon name="external" size={14} />
          </a>
        ) : null}
        <CopyButton text={citationText(facts, detail.source.pdf_url ?? detail.source.landing_url)} label="Copy citation" />
      </p>
      <PaperAsk paper={facts} signedIn={signedIn} />
      <h2>Abstract</h2>
      <p className="paper-abstract">{paper.abstract ?? "No abstract is available for this paper."}</p>
      {paper.fulltext_indexed ? (
        <p className="muted paper-fulltext">Full text indexed: answers can quote any section of this paper.</p>
      ) : (
        <p className="banner warn">Only the abstract of this paper is indexed: answers about it can use only the abstract.</p>
      )}
      <h2>Related papers</h2>
      {related && related.items.length ? (
        <ul className="related">
          {related.items.map((item) => {
            const summary = related.papers[item.paper_id];
            return summary ? (
              <li key={item.paper_id}>
                <Link href={`/papers/${item.paper_id}`}>{summary.title}</Link>
                <p>{summary.venue} {summary.year}</p>
              </li>
            ) : null;
          })}
        </ul>
      ) : (
        <p className="muted">No related papers were found.</p>
      )}
    </main>
  );
}
