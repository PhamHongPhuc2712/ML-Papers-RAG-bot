"use client";

import Link from "next/link";
import { useState } from "react";
import type { ChatMessage } from "@/lib/api";
import { citedByPaper, filterReferences, referenceFacets, type ReferenceFilter } from "@/lib/references";

/** The papers the turn's search ranked, the first of them read for the answer (RAG-only plan, U3). */
export function References({ message }: { message: ChatMessage }) {
  const references = message.references ?? [];
  const [filter, setFilter] = useState<ReferenceFilter>({ venue: null, year: null });
  if (!references.length) return null;
  const facets = referenceFacets(references);
  const shown = filterReferences(references, filter);
  const cited = citedByPaper(message);
  const venueId = `venue-${message.id}`;
  const yearId = `year-${message.id}`;
  return (
    <section aria-label="References" style={{ marginTop: 8 }}>
      <details>
        <summary>References ({references.length})</summary>
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap", margin: "8px 0" }}>
          <p style={{ margin: 0 }}>
            <label htmlFor={venueId}>Venue</label>{" "}
            <select id={venueId} value={filter.venue ?? ""} style={{ width: "auto" }}
              onChange={(e) => setFilter({ ...filter, venue: e.target.value || null })}>
              <option value="">All venues</option>
              {facets.venues.map((venue) => <option key={venue} value={venue}>{venue}</option>)}
            </select>
          </p>
          <p style={{ margin: 0 }}>
            <label htmlFor={yearId}>Year</label>{" "}
            <select id={yearId} value={filter.year ?? ""} style={{ width: "auto" }}
              onChange={(e) => setFilter({ ...filter, year: e.target.value ? Number(e.target.value) : null })}>
              <option value="">All years</option>
              {facets.years.map((year) => <option key={year} value={year}>{year}</option>)}
            </select>
          </p>
        </div>
        <p className="muted" aria-live="polite" style={{ margin: "0 0 8px" }}>
          {shown.length === references.length
            ? `${references.length} papers, ranked by the search.`
            : `Showing ${shown.length} of ${references.length} papers.`}
        </p>
        <ol style={{ paddingLeft: 24, margin: 0 }}>
          {shown.map((item) => {
            const numbers = cited.get(item.paper_id);
            const notes = [
              [item.venue, item.year].filter(Boolean).join(" "),
              item.read ? "read for this answer" : null,
              numbers ? `cited ${numbers.map((n) => `[${n}]`).join(" ")}` : null,
            ].filter(Boolean);
            return (
              <li key={item.paper_id} value={item.rank} style={{ marginBottom: 6 }}>
                <Link href={`/papers/${item.paper_id}`}>{item.title}</Link>{" "}
                <span className="muted">{notes.join(" · ")}</span>
              </li>
            );
          })}
        </ol>
      </details>
    </section>
  );
}
