/** Helpful / Not helpful on an answer (sign-up plan, A5). */
export type Rating = "helpful" | "not_helpful";

/** Choosing the rating already given clears it; choosing the other one replaces it. */
export function nextRating(current: Rating | null, choice: Rating): Rating | null {
  return current === choice ? null : choice;
}
