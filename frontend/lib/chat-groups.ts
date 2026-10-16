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
