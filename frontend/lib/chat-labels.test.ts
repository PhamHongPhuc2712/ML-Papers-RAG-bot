import { describe, expect, it } from "vitest";
import { citationNumbers } from "./chat-labels";
import type { ChatMessage } from "./api";

describe("citation numbers", () => {
  it("numbers ids by first citation, once each", () => {
    const message = {
      claims: [
        { text: "a", evidence_ids: ["E3", "E1"] },
        { text: "b", evidence_ids: ["E1", "E2"] },
      ],
    } as unknown as ChatMessage;
    expect([...citationNumbers(message)]).toEqual([["E3", 1], ["E1", 2], ["E2", 3]]);
  });
});
