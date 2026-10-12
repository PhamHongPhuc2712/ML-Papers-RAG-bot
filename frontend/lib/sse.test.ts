import { describe, expect, it } from "vitest";
import { parseBlock, readEvents } from "./sse";

function streamOf(chunks: Array<string | Uint8Array>): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(typeof chunk === "string" ? encoder.encode(chunk) : chunk);
      }
      controller.close();
    },
  });
}

async function collect(stream: ReadableStream<Uint8Array>) {
  const events = [];
  for await (const event of readEvents(stream)) events.push(event);
  return events;
}

describe("server-sent events", () => {
  it("parses one frame", () => {
    expect(parseBlock('event: status\ndata: {"stage":"searching"}')).toEqual({
      event: "status",
      data: { stage: "searching" },
    });
  });

  it("ignores a block without data", () => {
    expect(parseBlock(": keep-alive")).toBeNull();
  });

  it("joins frames split across chunks and splits frames sharing one", async () => {
    const events = await collect(
      streamOf(['event: chat\ndata: {"chat_id":"c', '1"}\n\nevent: status\n', 'data: {"stage":"writing"}\n\nevent: final\ndata: {}\n\n']),
    );
    expect(events.map((e) => e.event)).toEqual(["chat", "status", "final"]);
    expect(events[0]?.data).toEqual({ chat_id: "c1" });
  });

  it("decodes a character split across byte chunks", async () => {
    const bytes = new TextEncoder().encode('event: final\ndata: {"text":"é"}\n\n');
    const cut = bytes.indexOf(0xc3) + 1;
    const events = await collect(streamOf([bytes.slice(0, cut), bytes.slice(cut)]));
    expect(events[0]?.data).toEqual({ text: "é" });
  });
});
