"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState, type FormEvent } from "react";
import { ApiError, api, type SearchResponse } from "@/lib/api";
import { PaperCard } from "./PaperCard";

const FILTERS = { year_from: null, year_to: null, venues: [], fulltext_only: false };

type Results = { query: string; pages: SearchResponse[]; error: string | null; more: boolean };

function message(failure: unknown): string {
  return failure instanceof ApiError ? failure.body.message : "Search is not reachable. Try again.";
}

export function SearchView() {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();
  const query = params.get("q") ?? "";
  const [draft, setDraft] = useState(query);
  // The box follows the URL: a new ?q= (back, forward, a link) replaces what was typed.
  const [shown, setShown] = useState(query);
  if (shown !== query) {
    setShown(query);
    setDraft(query);
  }
  // Results remember the query they answer, so a stale response never shows under a new one
  // and "searching" is simply "no results for this query yet".
  const [results, setResults] = useState<Results | null>(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    if (!query) return;
    let current = true;
    api.search({ query, mode: "hybrid_rerank", filters: FILTERS, limit: 20, cursor: null }).then(
      (page) => {
        if (current) setResults({ query, pages: [page], error: null, more: false });
      },
      (failure: unknown) => {
        if (current) setResults({ query, pages: [], error: message(failure), more: false });
      },
    );
    return () => {
      current = false;
    };
  }, [query, attempt]);

  const mine = results?.query === query ? results : null;
  const pages = query ? (mine?.pages ?? []) : [];
  const error = query ? (mine?.error ?? null) : null;
  const busy = Boolean(query) && (mine === null || mine.more);

  function retry() {
    setResults(null);
    setAttempt((n) => n + 1);
  }

  async function more(cursor: string) {
    setResults((r) => (r ? { ...r, more: true } : r));
    try {
      const page = await api.search({ query, mode: "hybrid_rerank", filters: FILTERS, limit: 20, cursor });
      setResults((r) => (r?.query === query ? { ...r, pages: [...r.pages, page], more: false } : r));
    } catch (failure) {
      setResults((r) => (r?.query === query ? { ...r, error: message(failure), more: false } : r));
    }
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const text = draft.trim();
    if (text) router.push(`${pathname}?q=${encodeURIComponent(text)}`);
  }

  const last = pages.at(-1);
  const degraded = pages.some((page) => page.degraded);
  const items = pages.flatMap((page) => page.items.map((item) => ({ item, paper: page.papers[item.paper_id] })));

  return (
    <section aria-busy={busy}>
      <form role="search" onSubmit={submit} style={{ display: "flex", gap: 8 }}>
        <label htmlFor="q" className="visually-hidden">
          Search papers
        </label>
        <input id="q" value={draft} maxLength={2000} onChange={(e) => setDraft(e.target.value)} placeholder="e.g. contrastive learning for sentence embeddings" />
        <button className="primary" type="submit" disabled={busy}>
          Search
        </button>
      </form>
      <div role="status" aria-live="polite">
        {degraded ? (
          <p className="banner warn">These results may be incomplete: part of the search ran out of time ({last?.warnings.join(", ")}).</p>
        ) : null}
        {busy && !items.length ? <p className="muted">Searching…</p> : null}
      </div>
      {error ? (
        <p className="banner error" role="alert">
          {error} <button type="button" onClick={retry}>Retry</button>
        </p>
      ) : null}
      {query && !busy && !error && pages.length > 0 && items.length === 0 ? (
        <p>No papers matched. Try fewer or broader words.</p>
      ) : null}
      {items.map(({ item, paper }) => (paper ? <PaperCard key={item.paper_id} paperId={item.paper_id} paper={paper} /> : null))}
      {last?.next_cursor ? (
        <button type="button" disabled={busy} onClick={() => void more(last.next_cursor ?? "")}>
          More results
        </button>
      ) : null}
    </section>
  );
}
