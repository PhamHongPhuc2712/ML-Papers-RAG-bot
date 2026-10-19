"use client";

import { useState } from "react";
import type { ChatMessage } from "@/lib/api";
import { INSUFFICIENT, WARNINGS, citationNumbers, insufficientHint } from "@/lib/chat-labels";
import { answerAsText, partialSearch } from "@/lib/references";
import { Icon } from "./Icon";
import { DoneSteps } from "./StepsBox";

type Props = {
  message: ChatMessage;
  /** The evidence id whose claim is marked, when this answer's citation is open in the panel. */
  selected: string | null;
  onCite: (evidenceId: string) => void;
  onReferences: () => void;
};

/** One answer (canvas boards 6 and 7): steps, claims with numbered citations, and its actions. */
export function Answer({ message, selected, onCite, onReferences }: Props) {
  const [copied, setCopied] = useState(false);
  const numbers = citationNumbers(message);
  const notes = (message.warnings ?? []).map((code) => WARNINGS[code]).filter(Boolean);
  const references = message.references ?? [];

  async function copy() {
    await navigator.clipboard.writeText(answerAsText(message)).catch(() => undefined);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 2000);
  }

  return (
    <article aria-label="Answer" className="answer">
      <DoneSteps message={message} />
      {partialSearch(message) ? (
        <p className="banner warn note">Part of the search ran out of time, so this answer may have missed some papers. Asking again usually finds them.</p>
      ) : null}
      {message.status === "insufficient" ? (
        <div className="answer-insufficient">
          <p>{INSUFFICIENT}</p>
          <p className="muted">{insufficientHint(references.length)}</p>
        </div>
      ) : null}
      <div className="claims">
        {(message.claims ?? []).map((claim, index) => {
          const marked = selected !== null && claim.evidence_ids.includes(selected);
          return (
            <p key={index}>
              {marked ? <mark>{claim.text}</mark> : claim.text}{" "}
              {claim.evidence_ids.map((id) => (
                <button key={id} type="button" className={id === selected ? "chip selected" : "chip"}
                  aria-label={`Source ${numbers.get(id)}`} aria-pressed={id === selected} onClick={() => onCite(id)}>
                  {numbers.get(id)}
                </button>
              ))}
            </p>
          );
        })}
      </div>
      {notes.map((note) => <p key={note} className="muted">{note}</p>)}
      <div className="answer-actions">
        {references.length ? (
          <button type="button" onClick={onReferences}><Icon name="books" size={16} /> References ({references.length})</button>
        ) : null}
        {message.claims?.length ? (
          <button type="button" onClick={() => void copy()}><Icon name={copied ? "check" : "copy"} size={16} /> {copied ? "Copied" : "Copy"}</button>
        ) : null}
      </div>
    </article>
  );
}
