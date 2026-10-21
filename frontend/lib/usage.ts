/** Answers a day (sign-up plan, A3): what the ask box and the limit banner say. */
export type Usage = { today: number; month: number; limit: number; resets_at: string };

/** Shown under the ask box once 5 or fewer answers are left today. */
export function answersLeft(usage: Usage | null | undefined): string | null {
  if (!usage) return null;
  const left = Math.max(0, usage.limit - usage.today);
  if (left > 5) return null;
  if (left === 0) return "No answers left today";
  return `${left} ${left === 1 ? "answer" : "answers"} left today`;
}

/** Board 11's "Daily limit reached": the limit, and when asking works again. */
export function limitReached(limit: number, resetsAt: string, now: Date = new Date()): string {
  const minutes = Math.max(1, Math.round((Date.parse(resetsAt) - now.getTime()) / 60_000));
  const wait = minutes >= 90 ? `about ${Math.round(minutes / 60)} hours` : minutes >= 60 ? "about an hour" : `about ${minutes} minutes`;
  return `You have used today's ${limit} answers. You can ask again after 00:00 UTC, in ${wait}.`;
}
