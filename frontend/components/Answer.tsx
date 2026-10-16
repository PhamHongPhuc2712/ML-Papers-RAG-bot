import type { ChatMessage } from "@/lib/api";
import { INSUFFICIENT, WARNINGS, citationNumbers } from "@/lib/chat-labels";
import { searchLine } from "@/lib/references";

type Props = { message: ChatMessage; onCite: (evidenceId: string) => void };

export function Answer({ message, onCite }: Props) {
  const numbers = citationNumbers(message);
  const notes = (message.warnings ?? []).map((code) => WARNINGS[code]).filter(Boolean);
  const searched = searchLine(message);
  return (
    <article aria-label="Answer" style={{ background: "var(--surface)", borderRadius: "var(--radius)", padding: 12 }}>
      {searched ? <p className="muted" style={{ marginTop: 0 }}>{searched}</p> : null}
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
