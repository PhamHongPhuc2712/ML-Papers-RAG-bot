import type { ChatMessage } from "@/lib/api";
import { INSUFFICIENT, WARNINGS, citationNumbers } from "@/lib/chat-labels";

type Props = { message: ChatMessage; onCite: (evidenceId: string) => void };

export function Answer({ message, onCite }: Props) {
  const numbers = citationNumbers(message);
  const notes = (message.warnings ?? []).map((code) => WARNINGS[code]).filter(Boolean);
  return (
    <article aria-label="Answer" style={{ background: "var(--surface)", borderRadius: "var(--radius)", padding: 12 }}>
      {message.standalone_query && message.ordinal > 1 ? (
        <p className="muted" style={{ marginTop: 0 }}>Searched for: {message.standalone_query}</p>
      ) : null}
      {message.status === "insufficient" ? <p>{INSUFFICIENT}</p> : null}
      {(message.claims ?? []).map((claim, index) => (
        <p key={index} style={{ margin: "0 0 8px" }}>
          {claim.text}{" "}
          {claim.evidence_ids.map((id) => (
            <button
              key={id}
              type="button"
              aria-label={`Source ${numbers.get(id)}`}
              onClick={() => onCite(id)}
              style={{ padding: "0 6px", fontSize: "0.85em" }}
            >
              [{numbers.get(id)}]
            </button>
          ))}
        </p>
      ))}
      {notes.map((note) => (
        <p key={note} className="muted">{note}</p>
      ))}
    </article>
  );
}
