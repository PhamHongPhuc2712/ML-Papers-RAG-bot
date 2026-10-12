import { Suspense } from "react";
import { backendUrl } from "@/lib/server";
import { SearchView } from "@/components/SearchView";

type Coverage = { as_of: string; totals: { papers: number; venue_years: number } };

async function coverage(): Promise<Coverage | null> {
  try {
    const response = await fetch(`${backendUrl()}/v1/corpus/coverage`, { next: { revalidate: 300 } });
    return response.ok ? ((await response.json()) as Coverage) : null;
  } catch {
    return null;
  }
}

export default async function SearchPage() {
  const facts = await coverage();
  return (
    <main>
      <h1>Search ML papers</h1>
      {facts ? (
        <p className="muted">
          {facts.totals.papers.toLocaleString("en")} papers from {facts.totals.venue_years} venue-years, as of{" "}
          {new Date(facts.as_of).toLocaleDateString("en", { dateStyle: "medium" })}.
        </p>
      ) : null}
      <Suspense fallback={null}>
        <SearchView />
      </Suspense>
    </main>
  );
}
