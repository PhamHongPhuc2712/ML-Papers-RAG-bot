"use client";

import { Icon } from "./Icon";

/** Questions about topics the corpus covers; picking one asks it (canvas boards 1 and 4). */
export const SUGGESTIONS = [
  "Find papers on contrastive learning of sentence embeddings",
  "How does LoRA compare with full fine-tuning on reasoning tasks?",
  "Which benchmarks test long-context retrieval in language models?",
];

export function Suggestions({ onPick }: { onPick: (question: string) => void }) {
  return (
    <ul className="suggestions" aria-label="Suggested questions">
      {SUGGESTIONS.map((question) => (
        <li key={question}>
          <button type="button" onClick={() => onPick(question)}>
            <span>{question}</span>
            <Icon name="arrowUp" size={18} />
          </button>
        </li>
      ))}
    </ul>
  );
}
