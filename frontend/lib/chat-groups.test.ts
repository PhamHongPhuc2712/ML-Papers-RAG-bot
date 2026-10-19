import { describe, expect, it } from "vitest";
import { chatAge, groupChats, matchChats } from "./chat-groups";

const NOW = new Date(2026, 9, 16, 15, 0, 0);
const at = (days: number, hour = 9) => new Date(2026, 9, 16 - days, hour).toISOString();

describe("chat groups", () => {
  it("groups by last use, newest first, and leaves out empty groups", () => {
    const groups = groupChats(
      [
        { id: "old", title: "old", updated_at: at(30) },
        { id: "today-early", title: "a", updated_at: at(0, 1) },
        { id: "week", title: "b", updated_at: at(3) },
        { id: "today-late", title: "c", updated_at: at(0, 14) },
      ],
      NOW,
    );
    expect(groups.map((g) => [g.label, g.chats.map((c) => c.id)])).toEqual([
      ["Today", ["today-late", "today-early"]],
      ["Previous 7 days", ["week"]],
      ["Older", ["old"]],
    ]);
  });
  it("puts yesterday late at night in the previous 7 days", () => {
    const groups = groupChats([{ id: "y", title: "y", updated_at: at(1, 23) }], NOW);
    expect(groups.map((g) => g.label)).toEqual(["Previous 7 days"]);
  });
  it("is empty for no chats", () => {
    expect(groupChats([], NOW)).toEqual([]);
  });
});

describe("chat age", () => {
  const ago = (ms: number) => new Date(NOW.getTime() - ms).toISOString();
  it("says how long ago in the largest whole unit under a week", () => {
    expect(chatAge(ago(20 * 1000), NOW)).toBe("just now");
    expect(chatAge(ago(60 * 1000), NOW)).toBe("1 minute ago");
    expect(chatAge(ago(2 * 3600 * 1000 + 5), NOW)).toBe("2 hours ago");
    expect(chatAge(ago(4 * 86400 * 1000), NOW)).toBe("4 days ago");
  });
  it("gives the date from a week on", () => {
    expect(chatAge(new Date(2026, 8, 30, 12).toISOString(), NOW)).toBe("30 Sept 2026");
  });
});

describe("chat search", () => {
  const chats = [{ title: "What loss does SimCSE use?" }, { title: "Graph transformers for molecules" }];
  it("keeps titles holding every word, in any order and case", () => {
    expect(matchChats(chats, "simcse LOSS")).toEqual([chats[0]]);
    expect(matchChats(chats, "graph simcse")).toEqual([]);
  });
  it("keeps every chat for an empty query", () => {
    expect(matchChats(chats, "  ")).toEqual(chats);
  });
});
