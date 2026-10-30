export type ChatSummary = { id: string; title: string; updated_at: string };
export type ChatGroup = { label: string; chats: ChatSummary[] };

const DAY = 24 * 60 * 60 * 1000;

/** Chats grouped by when they were last used, newest first: Today, Previous 7 days, Older. */
export function groupChats(chats: ChatSummary[], now: Date = new Date()): ChatGroup[] {
  const midnight = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const groups: ChatGroup[] = [
    { label: "Today", chats: [] },
    { label: "Previous 7 days", chats: [] },
    { label: "Older", chats: [] },
  ];
  const sorted = [...chats].sort((a, b) => b.updated_at.localeCompare(a.updated_at));
  for (const chat of sorted) {
    const at = new Date(chat.updated_at).getTime();
    const index = at >= midnight ? 0 : at >= midnight - 7 * DAY ? 1 : 2;
    groups[index]?.chats.push(chat);
  }
  return groups.filter((group) => group.chats.length > 0);
}

const MINUTE = 60 * 1000;
const HOUR = 60 * MINUTE;
const plural = (n: number, unit: string) => `${n} ${unit}${n === 1 ? "" : "s"} ago`;

/** How long ago a chat was last used, in words: "just now", "2 hours ago", "4 days ago", or its date. */
export function chatAge(updatedAt: string, now: Date = new Date()): string {
  const at = new Date(updatedAt);
  const elapsed = Math.max(0, now.getTime() - at.getTime());
  if (elapsed < MINUTE) return "just now";
  if (elapsed < HOUR) return plural(Math.floor(elapsed / MINUTE), "minute");
  if (elapsed < DAY) return plural(Math.floor(elapsed / HOUR), "hour");
  if (elapsed < 7 * DAY) return plural(Math.floor(elapsed / DAY), "day");
  return at.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
}

/** The chats whose title holds every word of the query, in any order and case. */
export function matchChats<T extends { title: string }>(chats: T[], query: string): T[] {
  const words = query.toLocaleLowerCase().split(/\s+/).filter(Boolean);
  if (words.length === 0) return chats;
  return chats.filter((chat) => {
    const title = chat.title.toLocaleLowerCase();
    return words.every((word) => title.includes(word));
  });
}
