import { describe, expect, it } from "vitest";
import { citationText, paperLabel, withPaper } from "./papers";

const PAPER = { title: "Contrastive sentence embeddings", authors: ["Ada Lovelace", "Alan Turing", "Grace Hopper"], venue: "ACL", year: 2024 };

describe("papers", () => {
  it("names the paper in the question after the question itself", () => {
    expect(paperLabel(PAPER)).toBe("Contrastive sentence embeddings (ACL 2024)");
    expect(withPaper("What loss does it use?", PAPER)).toBe(
      "What loss does it use?\n\nAbout the paper “Contrastive sentence embeddings (ACL 2024)”.",
    );
  });
  it("writes a citation with every author, and the link when there is one", () => {
    expect(citationText(PAPER)).toBe("Ada Lovelace, Alan Turing and Grace Hopper. Contrastive sentence embeddings. ACL 2024.");
    expect(citationText({ ...PAPER, authors: ["Ada Lovelace"], title: "Ends with a stop." }, "https://example.org/a.pdf")).toBe(
      "Ada Lovelace. Ends with a stop. ACL 2024. https://example.org/a.pdf",
    );
    expect(citationText({ ...PAPER, authors: [] })).toBe("Contrastive sentence embeddings. ACL 2024.");
  });
});
