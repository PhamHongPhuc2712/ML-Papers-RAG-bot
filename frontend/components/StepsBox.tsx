import type { ChatMessage } from "@/lib/api";
import { stepsSummary } from "@/lib/references";
import { Icon } from "./Icon";

export type LiveSearch = { query: string; ranked: number; read: number } | null;

type Step = { label: string; state: "done" | "active" | "waiting" };

function liveSteps(stage: string, searched: LiveSearch): Step[] {
  const writing = stage === "writing";
  return [
    searched
      ? { label: `Searched for “${searched.query}”, ${searched.ranked} papers ranked`, state: "done" }
      : { label: stage === "condensing" ? "Reading your follow-up" : "Searching the papers", state: "active" },
    { label: searched ? `Reading the top ${searched.read} papers` : "Reading the top papers", state: writing ? "done" : searched ? "active" : "waiting" },
    { label: "Writing the answer", state: writing ? "active" : "waiting" },
  ];
}

/** What the turn is doing while it runs (canvas board 5): three steps that fill in from the stream. */
export function LiveSteps({ stage, searched }: { stage: string; searched: LiveSearch }) {
  return (
    <ol className="steps" aria-label="Progress">
      {liveSteps(stage, searched).map((step) => (
        <li key={step.label} className={`step ${step.state}`}>
          {step.state === "done" ? <Icon name="check" size={18} /> : <span className="step-dot" aria-hidden="true" />}
          {step.label}
        </li>
      ))}
    </ol>
  );
}

/** A finished answer's steps, collapsed to one line (canvas board 6). */
export function DoneSteps({ message }: { message: ChatMessage }) {
  const summary = stepsSummary(message);
  if (!summary) return null;
  return (
    <p className="steps-done"><Icon name="books" size={16} /> {summary}</p>
  );
}
