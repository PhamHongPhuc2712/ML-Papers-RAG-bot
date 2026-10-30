"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api, type ChatMessage, type PaperDetail } from "@/lib/api";
import { citationNumbers } from "@/lib/chat-labels";
import { citedByPaper, filterReferences, passagesOf, referenceFacets, where, type ReferenceFilter } from "@/lib/references";
import { Icon } from "./Icon";

export type PanelState = { messageId: string; paperId: string | null; evidenceId: string | null };

type Props = {
  message: ChatMessage;
  state: PanelState;
  onOpenPaper: (paperId: string) => void;
  onBack: () => void;
  onClose: () => void;
};

/** The side panel (canvas boards 6 and 7): an answer's References, or one paper and its passages. */
export function ReferencesPanel({ message, state, onOpenPaper, onBack, onClose }: Props) {
  useEffect(() => {
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", escape);
    return () => document.removeEventListener("keydown", escape);
  }, [onClose]);

  return (
    <aside className="panel" aria-label={state.paperId ? "Paper" : "References"}>
      <div className="panel-head">
        {state.paperId ? (
          <button type="button" className="text-button" onClick={onBack}><Icon name="back" size={18} /> All references</button>
        ) : (
          <h2>References</h2>
        )}
        <button type="button" className="icon-button" aria-label="Close panel" onClick={onClose}><Icon name="close" size={18} /></button>
      </div>
      {state.paperId ? (
        <PaperView key={`${state.paperId}-${state.evidenceId}`} message={message} paperId={state.paperId} evidenceId={state.evidenceId} />
      ) : (
        <ReferenceList message={message} onOpenPaper={onOpenPaper} />
      )}
    </aside>
  );
}

function ReferenceList({ message, onOpenPaper }: { message: ChatMessage; onOpenPaper: (paperId: string) => void }) {
  const references = message.references ?? [];
  const [filter, setFilter] = useState<ReferenceFilter>({ venue: null, year: null, citedOnly: false });
  const cited = citedByPaper(message);
  const facets = referenceFacets(references);
  const shown = filterReferences(references, filter, cited);
  return (
    <>
      <div className="panel-filters">
        <label htmlFor="ref-venue" className="visually-hidden">Venue</label>
        <select id="ref-venue" value={filter.venue ?? ""} onChange={(e) => setFilter({ ...filter, venue: e.target.value || null })}>
          <option value="">All venues</option>
          {facets.venues.map((venue) => <option key={venue} value={venue}>{venue}</option>)}
        </select>
        <label htmlFor="ref-year" className="visually-hidden">Year</label>
        <select id="ref-year" value={filter.year ?? ""} onChange={(e) => setFilter({ ...filter, year: e.target.value ? Number(e.target.value) : null })}>
          <option value="">All years</option>
          {facets.years.map((year) => <option key={year} value={year}>{year}</option>)}
        </select>
        <label className="check"><input type="checkbox" checked={filter.citedOnly}
          onChange={(e) => setFilter({ ...filter, citedOnly: e.target.checked })} /> Cited only</label>
      </div>
      <p className="panel-count" aria-live="polite">
        {shown.length === references.length ? `${references.length} papers, ranked by the search` : `Showing ${shown.length} of ${references.length} papers`}
      </p>
      <ol className="ref-list">
        {shown.map((item) => {
          const numbers = cited.get(item.paper_id);
          return (
            <li key={item.paper_id}>
              <span className="ref-rank">{item.rank}</span>
              <div>
                <button type="button" className="ref-title" onClick={() => onOpenPaper(item.paper_id)}>{item.title}</button>
                <p className="ref-meta">
                  {[item.venue, item.year].filter(Boolean).join(" ")}
                  {numbers ? <span className="tag cited">Cited {numbers.join(", ")}</span> : null}
                  <span className="tag">{item.read ? "Read" : "Ranked, not read"}</span>
                </p>
              </div>
            </li>
          );
        })}
      </ol>
    </>
  );
}

function PaperView({ message, paperId, evidenceId }: { message: ChatMessage; paperId: string; evidenceId: string | null }) {
  const [detail, setDetail] = useState<PaperDetail | null>(null);
  const numbers = citationNumbers(message);
  const reference = (message.references ?? []).find((item) => item.paper_id === paperId);
  const { cited, read } = passagesOf(message, paperId);
  const first = cited[0] ?? read[0];
  const title = reference?.title ?? first?.title ?? "Paper";
  const meta = [reference?.venue ?? first?.venue, reference?.year ?? first?.year].filter(Boolean).join(" ");

  useEffect(() => {
    let current = true;
    api.paper(paperId).then((paper) => { if (current) setDetail(paper); }, () => undefined);
    return () => { current = false; };
  }, [paperId]);

  useEffect(() => {
    if (evidenceId) document.getElementById(`passage-${evidenceId}`)?.scrollIntoView({ block: "nearest" });
  }, [evidenceId]);

  const authors = detail?.paper.authors ?? [];
  return (
    <div className="paper-view">
      <h2>{title}</h2>
      <p className="paper-meta">
        {authors.length ? `${authors.slice(0, 3).join(", ")}${authors.length > 3 ? ` and ${authors.length - 3} other${authors.length === 4 ? "" : "s"}` : ""}. ` : ""}{meta}
      </p>
      <p className="paper-links">
        <Link href={`/papers/${paperId}`}>Paper page</Link>
        {detail?.source.pdf_url ? <a href={detail.source.pdf_url} rel="noopener noreferrer" target="_blank">PDF <Icon name="external" size={14} /></a> : null}
      </p>
      {cited.length ? (
        <section aria-label="Cited passages" className="passages">
          {cited.map((item) => (
            <figure key={item.id} id={`passage-${item.id}`} className={item.id === evidenceId ? "passage selected" : "passage"}>
              <figcaption><span className="chip static">{numbers.get(item.id)}</span> {where(item)}{item.abstract_only ? ", abstract only" : ""}</figcaption>
              <blockquote>{item.id === evidenceId ? <mark>{item.text}</mark> : item.text}</blockquote>
            </figure>
          ))}
        </section>
      ) : (
        <p className="muted">The answer read this paper but cited none of its passages.</p>
      )}
      {read.length ? (
        <details className="also-read">
          <summary>{read.length} more passage{read.length === 1 ? "" : "s"} read from this paper</summary>
          {read.map((item) => (
            <figure key={item.id} className="passage">
              <figcaption>{where(item)}{item.abstract_only ? ", abstract only" : ""}</figcaption>
              <blockquote>{item.text}</blockquote>
            </figure>
          ))}
        </details>
      ) : null}
    </div>
  );
}
