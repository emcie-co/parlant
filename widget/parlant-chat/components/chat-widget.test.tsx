// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ChatWidget } from "./chat-widget";

// Vitest runs without globals here, so RTL's automatic per-test cleanup isn't
// registered — unmount by hand or each render stacks onto the previous DOM.
afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

// jsdom has no layout: the transcript's stick-to-bottom needs these to exist.
beforeEach(() => {
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    },
  );
  Element.prototype.scrollTo = () => {};
});

const SERVER = "https://api.example";
const SESSION_KEY = "parlant:v1:session:https%3A%2F%2Fapi.example:agent:guest";

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

/** An event stream that opens and then stays quiet, like a session with nothing new. */
function idleStream(): Response {
  return new Response(new ReadableStream({ start() {} }), {
    headers: { "content-type": "text/event-stream" },
  });
}

/** A fake Parlant server: creates "fresh-session" and records every message sent to any session. */
function server() {
  const sent: { sessionId: string; body: Record<string, unknown> }[] = [];
  const fetchMock = vi.fn<typeof fetch>(async (input, init) => {
    const url = new URL(String(input));
    const method = init?.method ?? "GET";
    if (url.pathname === "/healthz") return json({ status: "healthy" });
    if (url.pathname === "/agents/agent")
      return json({ id: "agent", name: "Parlant" });
    if (method === "POST" && url.pathname === "/sessions")
      return json({ id: "fresh-session" }, 201);
    const events = /^\/sessions\/([^/]+)\/events$/.exec(url.pathname);
    if (events && method === "POST") {
      sent.push({ sessionId: events[1], body: JSON.parse(String(init?.body)) });
      return json({ offset: sent.length }, 201);
    }
    if (events) return idleStream();
    const session = /^\/sessions\/([^/]+)$/.exec(url.pathname);
    if (session) return json({ id: session[1], mode: "auto" });
    return json({}, 404);
  });
  vi.stubGlobal("fetch", fetchMock);
  return { sent };
}

describe("initial message", () => {
  it("sends the initial message once, as the first message of a new conversation", async () => {
    localStorage.setItem(SESSION_KEY, "earlier-session");
    const { sent } = server();

    render(
      <ChatWidget
        serverUrl={SERVER}
        agentId="agent"
        mode="inline"
        customerSound={false}
        agentSound={false}
        initialMessage="My email is dana@acme.com and I'd like to learn about Parlant for our company."
      />,
    );

    await waitFor(() => expect(sent).toHaveLength(1));
    expect(sent[0]).toEqual({
      sessionId: "fresh-session",
      body: {
        kind: "message",
        source: "customer",
        message:
          "My email is dana@acme.com and I'd like to learn about Parlant for our company.",
      },
    });
    expect(localStorage.getItem(SESSION_KEY)).toBe("fresh-session");
    // Give a duplicate send the chance to happen before asserting there was none.
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(sent).toHaveLength(1);
  });

  it("resumes the stored conversation and sends nothing without one", async () => {
    localStorage.setItem(SESSION_KEY, "earlier-session");
    const { sent } = server();

    render(
      <ChatWidget
        serverUrl={SERVER}
        agentId="agent"
        mode="inline"
        customerSound={false}
        agentSound={false}
      />,
    );

    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(sent).toHaveLength(0);
    expect(localStorage.getItem(SESSION_KEY)).toBe("earlier-session");
  });
});

describe("suggestions", () => {
  const chips = (c: HTMLElement) => c.querySelector(".parlant-suggestions");

  it("shows the suggestion chips in floating mode once the opened session is empty", async () => {
    server();

    const { container } = render(
      <ChatWidget
        serverUrl={SERVER}
        agentId="agent"
        mode="floating"
        customerSound={false}
        agentSound={false}
        suggestions={["Track my order", "Return an item"]}
      />,
    );

    // Floating mounts the panel on first open, so there is nothing to show until then.
    expect(chips(container)).toBeNull();

    fireEvent.click(screen.getByLabelText("Open chat"));

    await waitFor(() => expect(chips(container)).not.toBeNull());
    expect(screen.getByText("Track my order")).toBeDefined();
    expect(screen.getByText("Return an item")).toBeDefined();
  });

  it("sends a floating-mode suggestion when its chip is clicked and then hides the chips", async () => {
    const { sent } = server();

    const { container } = render(
      <ChatWidget
        serverUrl={SERVER}
        agentId="agent"
        mode="floating"
        customerSound={false}
        agentSound={false}
        suggestions={["Track my order"]}
      />,
    );

    fireEvent.click(screen.getByLabelText("Open chat"));
    await waitFor(() => expect(chips(container)).not.toBeNull());

    fireEvent.click(screen.getByText("Track my order"));

    await waitFor(() => expect(sent).toHaveLength(1));
    expect(sent[0].body).toEqual({
      kind: "message",
      source: "customer",
      message: "Track my order",
    });
    // The conversation is no longer empty, so the starters step aside.
    await waitFor(() => expect(chips(container)).toBeNull());
  });

  it("shows the suggestion chips in inline mode while the conversation is empty", async () => {
    server();

    const { container } = render(
      <ChatWidget
        serverUrl={SERVER}
        agentId="agent"
        mode="inline"
        customerSound={false}
        agentSound={false}
        suggestions={["Track my order"]}
      />,
    );

    await waitFor(() => expect(chips(container)).not.toBeNull());
  });

  it("does not show the suggestion chips in floating mode for a conversation that already has messages", async () => {
    localStorage.setItem(SESSION_KEY, "earlier-session");
    server();

    const { container } = render(
      <ChatWidget
        serverUrl={SERVER}
        agentId="agent"
        mode="floating"
        customerSound={false}
        agentSound={false}
        initialMessage="My order hasn't arrived."
        suggestions={["Track my order"]}
      />,
    );

    fireEvent.click(screen.getByLabelText("Open chat"));

    await waitFor(() => expect(screen.queryByText("My order hasn't arrived.")).not.toBeNull());
    expect(chips(container)).toBeNull();
  });
});
