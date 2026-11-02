// @vitest-environment jsdom

import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import type { ChatMessage, ToolWidget } from "../lib/events";
import { MessageBubble } from "./message-bubble";

afterEach(() => {
  cleanup();
});

const DATES = {
  type: "choices",
  id: "meeting-dates",
  prompt: "Which day suits you?",
  options: [{ id: "2026-11-03", label: "Tue, Nov 3" }],
};

const WIDGETS: ToolWidget[] = [
  { toolCallId: "event-1:0", metadata: { widget: DATES }, traceId: "trace-1" },
];

function reply(overrides: Partial<ChatMessage> = {}): ChatMessage {
  return {
    id: "event-2",
    role: "agent",
    text: "Which day suits you?",
    streaming: false,
    attachments: [],
    traceId: "trace-1",
    createdAt: "2026-10-30T10:00:00Z",
    ...overrides,
  };
}

function bubble(message: ChatMessage, streamStyle: "blur" | "instant" = "blur") {
  return (
    <MessageBubble message={message} widgets={WIDGETS} streamStyle={streamStyle} streamSpeed="fast" />
  );
}

const option = () => screen.queryByRole("button", { name: "Tue, Nov 3" });

describe("tool widgets on a reply", () => {
  it("renders tool widgets below the reply", () => {
    const { container } = render(bubble(reply()));

    const text = container.querySelector(".parlant-msg-bubble");
    const widgets = container.querySelector(".parlant-msg-widgets");

    expect(text?.compareDocumentPosition(widgets as Node)).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
  });

  it("holds tool widgets while the reply is streaming", () => {
    render(bubble(reply({ streaming: true, fresh: true })));

    expect(option()).toBeNull();
  });

  it("holds tool widgets until a fresh reply has finished revealing", async () => {
    render(bubble(reply({ fresh: true })));

    expect(option()).toBeNull();
    await waitFor(() => expect(option()).not.toBeNull());
  });

  it("shows tool widgets at once for a reply from history", () => {
    render(bubble(reply()));

    expect(option()).not.toBeNull();
  });

  it("shows tool widgets at once with the instant stream style", () => {
    render(bubble(reply({ fresh: true }), "instant"));

    expect(option()).not.toBeNull();
  });
});
