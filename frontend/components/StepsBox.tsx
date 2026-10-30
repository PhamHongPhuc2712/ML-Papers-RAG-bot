import type { ChatMessage } from "@/lib/api";
import { quotedList, stepsSummary } from "@/lib/references";
import { Icon } from "./Icon";

/** The search step once the agent has chosen its passages: every query, and the papers ranked. */
export type LiveSearch = { queries: string[]; ranked: number; read: number } | null;

type Step = { label: string; state: "done" | "active" | "waiting" };

function searchStep(stage: string, searching: string[], searched: LiveSearch): Step {
  if (searched) {
    return searched.queries.length
      ? { label: `Searched for ${quotedList(searched.queries)}, ${searched.ranked} papers ranked`, state: "done" }
      : { label: "Used passages cited earlier in this chat", state: "done" };
  }
  const current = searching[searching.length - 1];
  if (current) return { label: `Searching for “${current}”`, state: "active" };
  return { label: stage === "condensing" ? "Reading your follow-up" : "Searching the papers", state: "active" };
}

function liveSteps(stage: string, searching: string[], searched: LiveSearch): Step[] {
  const writing = stage === "writing";
  return [
    searchStep(stage, searching, searched),
    { label: searched ? `Reading the top ${searched.read} papers` : "Reading the top papers", state: writing ? "done" : searched ? "active" : "waiting" },
    { label: "Writing the answer", state: writing ? "active" : "waiting" },
  ];
}

/** What the turn is doing while it runs (canvas board 5): three steps that fill in from the stream. */
export function LiveSteps({ stage, searching, searched }: { stage: string; searching: string[]; searched: LiveSearch }) {
  return (
    <ol className="steps" aria-label="Progress">
      {liveSteps(stage, searching, searched).map((step) => (
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
