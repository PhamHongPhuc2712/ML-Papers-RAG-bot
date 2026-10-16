import { describe, expect, it } from "vitest";
import { groupChats } from "./chat-groups";

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
