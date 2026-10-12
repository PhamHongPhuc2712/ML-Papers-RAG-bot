import Link from "next/link";
import type { ChatMessage, StoredEvidence } from "@/lib/api";
import { citationNumbers } from "@/lib/chat-labels";

function where(item: StoredEvidence): string {
  if (item.page_start == null) return item.section;
  const pages = item.page_end == null || item.page_end === item.page_start
    ? `p. ${item.page_start}`
    : `pp. ${item.page_start}–${item.page_end}`;
  return `${item.section} · ${pages}`;
}

export function EvidencePanel({ message, focused }: { message: ChatMessage; focused: string | null }) {
  const numbers = citationNumbers(message);
  const cited = (message.evidence ?? []).filter((item) => item.cited).sort((a, b) => (numbers.get(a.id) ?? 0) - (numbers.get(b.id) ?? 0));
  const read = (message.evidence ?? []).filter((item) => !item.cited);
  const render = (item: StoredEvidence) => (
    <li key={item.id} id={`source-${message.id}-${item.id}`} tabIndex={-1}
      style={{ marginBottom: 12, outline: focused === item.id ? "2px solid var(--accent)" : undefined, borderRadius: 4 }}>
      <strong>{numbers.has(item.id) ? `[${numbers.get(item.id)}] ` : ""}</strong>
      <Link href={`/papers/${item.paper_id}`}>{item.title}</Link>{" "}
      <span className="muted">{[item.venue, item.year].filter(Boolean).join(" ")} · {where(item)}{item.abstract_only ? " · abstract only" : ""}</span>
      <blockquote style={{ margin: "6px 0 0", maxHeight: 220, overflow: "auto", whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{item.text}</blockquote>
    </li>
  );
  return (
    <section aria-label="Sources" role="region">
      <h2 style={{ fontSize: "1rem" }}>Sources</h2>
      <ol style={{ listStyle: "none", padding: 0 }}>{cited.map(render)}</ol>
      {read.length ? (
        <details>
          <summary>Also read ({read.length})</summary>
          <ol style={{ listStyle: "none", padding: 0 }}>{read.map(render)}</ol>
        </details>
      ) : null}
    </section>
  );
}
