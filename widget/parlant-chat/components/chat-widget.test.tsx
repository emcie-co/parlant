// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  V1_GUEST_KEY,
  fakeParlantServer,
  storedGuestSession,
} from "../testing/fake-parlant-server";
import { ChatWidget } from "./chat-widget";
import type { ParlantEvent } from "../lib/events";
import * as sounds from "../lib/sounds";
import { registerWidget } from "./tool-widget";

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

/** A conversation this browser had earlier with "agent", saved with its guest token. */
function rememberConversation(sessionId: string): void {
  localStorage.setItem(V1_GUEST_KEY, "guest-token");
  localStorage.setItem(SESSION_KEY, sessionId);
}

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

/** A fake Parlant server: creates "fresh-session", records every message sent to any session,
 *  and serves `history` as every session's existing events. */
function server(history: unknown[] = []) {
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
    // The live feed is a stream; a plain read of the history so far returns at once.
    if (events) return url.searchParams.get("sse") === "true" ? idleStream() : json(history);
    const session = /^\/sessions\/([^/]+)$/.exec(url.pathname);
    if (session) return json({ id: session[1], mode: "auto" });
    return json({}, 404);
  });
  vi.stubGlobal("fetch", fetchMock);
  return { sent };
}

describe("initial message", () => {
  it("sends the initial message once, as the first message of a new conversation", async () => {
    rememberConversation("earlier-session");
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
    expect(storedGuestSession(localStorage, "agent")).toBe("fresh-session");
    // Give a duplicate send the chance to happen before asserting there was none.
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(sent).toHaveLength(1);
  });

  it("resumes the stored conversation and sends nothing without one", async () => {
    rememberConversation("earlier-session");
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
    expect(storedGuestSession(localStorage, "agent")).toBe("earlier-session");
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
        openingSuggestions={["Track my order", "Return an item"]}
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
        openingSuggestions={["Track my order"]}
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
        openingSuggestions={["Track my order"]}
      />,
    );

    await waitFor(() => expect(chips(container)).not.toBeNull());
  });

  it("does not show the suggestion chips in floating mode for a conversation that already has messages", async () => {
    rememberConversation("earlier-session");
    server();

    const { container } = render(
      <ChatWidget
        serverUrl={SERVER}
        agentId="agent"
        mode="floating"
        customerSound={false}
        agentSound={false}
        initialMessage="My order hasn't arrived."
        openingSuggestions={["Track my order"]}
      />,
    );

    fireEvent.click(screen.getByLabelText("Open chat"));

    await waitFor(() => expect(screen.queryByText("My order hasn't arrived.")).not.toBeNull());
    expect(chips(container)).toBeNull();
  });
});

describe("persistent suggestions", () => {
  it("defaults to three chips, taking opening suggestions before persistent ones", async () => {
    server();
    const view = render(<ChatWidget serverUrl={SERVER} agentId="agent" mode="input"
      openingSuggestions={["First", "Second"]}
      persistentSuggestions={["Third", "Fourth"]} />);
    await waitFor(() => expect(screen.queryByRole("button", { name: "First" })).not.toBeNull());
    const labels = () => [...view.container.querySelectorAll(".parlant-cmp-suggestion")].map((e) => e.textContent);
    expect(labels()).toEqual(["First", "Second", "Third"]);
    view.rerender(<ChatWidget serverUrl={SERVER} agentId="agent" mode="input"
      openingSuggestions={["First", "Second"]}
      persistentSuggestions={["Third", "Fourth"]} maxVisibleSuggestions={2} />);
    expect(labels()).toEqual(["First", "Second"]);
    view.rerender(<ChatWidget serverUrl={SERVER} agentId="agent" mode="input"
      openingSuggestions={["First", "Second"]}
      persistentSuggestions={["Third", "Fourth"]} maxVisibleSuggestions={0} />);
    expect(labels()).toEqual([]);
  });

  it("marks only the persistent chips, so they can stand out", async () => {
    server();
    const view = render(<ChatWidget serverUrl={SERVER} agentId="agent" mode="input"
      openingSuggestions={["Track my order"]} persistentSuggestions={["Book a call"]} />);
    await waitFor(() => expect(screen.queryByRole("button", { name: "Book a call" })).not.toBeNull());
    const marked = [...view.container.querySelectorAll(".parlant-cmp-suggestion--persistent")].map((e) => e.textContent);
    expect(marked).toEqual(["Book a call"]);
  });

  it("fills a vacated persistent slot with the next unchosen suggestion", async () => {
    rememberConversation("earlier-session");
    server([message(0, "customer", "Hi"), message(1, "ai_agent", "Hello")]);
    render(<ChatWidget serverUrl={SERVER} agentId="agent" mode="inline"
      customerSound={false} agentSound={false} maxVisibleSuggestions={2}
      persistentSuggestions={["First", "Second", "Third"]} />);
    await waitFor(() => expect(screen.queryByText("Hello")).not.toBeNull());
    expect(screen.queryByRole("button", { name: "Third" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "First" }));
    expect(screen.queryByRole("button", { name: "First" })).toBeNull();
    expect(screen.getByRole("button", { name: "Second" })).toBeDefined();
    expect(screen.getByRole("button", { name: "Third" })).toBeDefined();
  });

  it("keeps persistent chips after a normal starter is sent, without duplicate chips", async () => {
    const { sent } = server();
    render(<ChatWidget serverUrl={SERVER} agentId="agent" mode="inline"
      customerSound={false} agentSound={false}
      openingSuggestions={["Track my order", "Talk to someone"]}
      persistentSuggestions={["Talk to someone"]} />);

    await waitFor(() => expect(screen.queryByText("Track my order")).not.toBeNull());
    expect(screen.getAllByRole("button", { name: "Talk to someone" })).toHaveLength(1);
    fireEvent.click(screen.getByText("Track my order"));
    await waitFor(() => expect(sent).toHaveLength(1));
    expect(screen.queryByRole("button", { name: "Track my order" })).toBeNull();
    expect(screen.getByRole("button", { name: "Talk to someone" })).toBeDefined();
  });

  it("sends a persistent suggestion in a resumed conversation", async () => {
    rememberConversation("earlier-session");
    const { sent } = server([message(0, "customer", "Hi"), message(1, "ai_agent", "Hello")]);
    render(<ChatWidget serverUrl={SERVER} agentId="agent" mode="inline"
      customerSound={false} agentSound={false}
      openingSuggestions={["Track my order"]} persistentSuggestions={["Book a free consultation"]} />);

    await waitFor(() => expect(screen.queryByText("Hello")).not.toBeNull());
    expect(screen.queryByRole("button", { name: "Track my order" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Book a free consultation" }));
    await waitFor(() => expect(sent).toHaveLength(1));
    expect(sent[0].body.message).toBe("Book a free consultation");
    expect(screen.queryByRole("button", { name: "Book a free consultation" })).toBeNull();
  });
});

describe("sidebar opening mode", () => {
  it("opens as a sidebar before expanding and widens the same column", async () => {
    rememberConversation("earlier-session");
    const { sent } = server();
    const { container } = render(<ChatWidget serverUrl={SERVER} agentId="agent"
      mode="input" openMode="sidebar" initialMessage="Hi"
      customerSound={false} agentSound={false} />);
    await waitFor(() => expect(container.querySelector(".parlant-sidebar")).not.toBeNull());
    await waitFor(() => expect(sent).toHaveLength(1));
    expect(container.querySelector(".parlant-sidebar--wide")).toBeNull();
    fireEvent.click(screen.getByLabelText("Expand chat"));
    expect(container.querySelector(".parlant-sidebar--wide")).not.toBeNull();
    fireEvent.click(screen.getByLabelText("Minimize chat"));
    expect(container.querySelector(".parlant-sidebar")).toBeNull();
    fireEvent.click(screen.getByLabelText("Reopen chat"));
    expect(container.querySelector(".parlant-sidebar")).not.toBeNull();
    expect(container.querySelector(".parlant-sidebar--wide")).toBeNull();
  });

  it("keeps the usual panel when openMode is omitted", async () => {
    server();
    const { container } = render(<ChatWidget serverUrl={SERVER} agentId="agent"
      mode="input" initialMessage="Hi" customerSound={false} agentSound={false} />);
    await waitFor(() => expect(screen.queryByLabelText("Expand chat")).not.toBeNull());
    expect(container.querySelector(".parlant-sidebar")).toBeNull();
    fireEvent.click(screen.getByLabelText("Expand chat"));
    expect(container.querySelector(".parlant-chat--expanded")).not.toBeNull();
  });
});

/** A message event as the server stores it. */
function message(offset: number, source: "customer" | "ai_agent", text: string, chunks?: (string | null)[]) {
  return {
    id: `event-${offset}`,
    offset,
    kind: "message",
    source,
    trace_id: `trace-${offset}`,
    creation_utc: "2026-10-24T10:00:00Z",
    data: chunks ? { chunks, participant: { id: source, display_name: source } } : { message: text },
    metadata: {},
  };
}

describe("resumed conversations", () => {
  const chips = (c: HTMLElement) => c.querySelector(".parlant-suggestions");
  const replies = (c: HTMLElement) => [...c.querySelectorAll(".parlant-stream")];
  const animating = (el: Element) => /parlant-fx-/.test(el.className);

  it("shows the suggestion chips for a stored session that is still empty", async () => {
    rememberConversation("empty-session");
    server([]);

    const { container } = render(
      <ChatWidget
        serverUrl={SERVER}
        agentId="agent"
        mode="input"
        customerSound={false}
        agentSound={false}
        openingSuggestions={["Track my order"]}
      />,
    );

    await waitFor(() => expect(chips(container)).not.toBeNull());
  });

  it("hides the suggestion chips for a stored session that has messages", async () => {
    rememberConversation("earlier-session");
    server([message(0, "customer", "Hi"), message(1, "ai_agent", "Hello, how can I help?")]);

    const { container } = render(
      <ChatWidget
        serverUrl={SERVER}
        agentId="agent"
        mode="inline"
        customerSound={false}
        agentSound={false}
        openingSuggestions={["Track my order"]}
      />,
    );

    await waitFor(() => expect(screen.queryByText("Hello, how can I help?")).not.toBeNull());
    expect(chips(container)).toBeNull();
  });

  it("shows finished replies at once, and animates only a reply still being written", async () => {
    rememberConversation("earlier-session");
    server([
      message(0, "customer", "Hi"),
      message(1, "ai_agent", "", ["An earlier reply.", null]),
      message(2, "customer", "And another thing"),
      message(3, "ai_agent", "", ["A reply in progress"]),
    ]);

    const { container } = render(
      <ChatWidget
        serverUrl={SERVER}
        agentId="agent"
        mode="inline"
        customerSound={false}
        agentSound={false}
      />,
    );

    await waitFor(() => expect(replies(container)).toHaveLength(2));
    const [finished, inProgress] = replies(container);
    expect(animating(finished)).toBe(false);
    expect(animating(inProgress)).toBe(true);
  });
});

describe("avatar", () => {
  it("shows the given image instead of the name's initials", async () => {
    server();

    const { container } = render(
      <ChatWidget
        serverUrl={SERVER}
        agentId="agent"
        mode="inline"
        customerSound={false}
        agentSound={false}
        avatarUrl="/logo-mark.svg"
      />,
    );

    await waitFor(() => expect(container.querySelector(".parlant-cw-avatar img")).not.toBeNull());
    expect(container.querySelector(".parlant-cw-avatar img")?.getAttribute("src")).toBe("/logo-mark.svg");
    expect(screen.queryByText("PA")).toBeNull();
    expect(container.querySelector(".parlant-cw-avatar--photo")).toBeNull();
  });

  it("fills a round frame with a photo", async () => {
    server();

    const { container } = render(
      <ChatWidget
        serverUrl={SERVER}
        agentId="agent"
        mode="inline"
        customerSound={false}
        agentSound={false}
        avatarUrl="/portrait.jpg"
        avatarStyle="photo"
      />,
    );

    await waitFor(() => expect(container.querySelector(".parlant-cw-avatar--photo img")).not.toBeNull());
    expect(container.querySelector(".parlant-cw-avatar--photo img")?.getAttribute("src")).toBe("/portrait.jpg");
  });

  it("puts the live dot on a photo's corner, and on nothing else", async () => {
    server();

    const { container, rerender } = render(
      <ChatWidget
        serverUrl={SERVER}
        agentId="agent"
        mode="inline"
        customerSound={false}
        agentSound={false}
        avatarUrl="/portrait.jpg"
        avatarStyle="photo"
        liveIndicator
      />,
    );

    await waitFor(() => expect(container.querySelector(".parlant-cw-avatar-frame .parlant-cw-live")).not.toBeNull());
    expect(container.querySelector(".parlant-cw-live")?.getAttribute("aria-label")).toBe("Live");

    rerender(
      <ChatWidget
        serverUrl={SERVER}
        agentId="agent"
        mode="inline"
        customerSound={false}
        agentSound={false}
        avatarUrl="/logo.svg"
        liveIndicator
      />,
    );

    expect(container.querySelector(".parlant-cw-live")).toBeNull();
  });

  it("turns the live dot red while the banner says the server can't be reached", async () => {
    server();
    const healthy = vi.mocked(fetch).getMockImplementation()!;
    let down = true;
    vi.mocked(fetch).mockImplementation(async (input, init) =>
      down && new URL(String(input)).pathname === "/healthz"
        ? json({ status: "down" }, 503)
        : healthy(input, init),
    );

    const { container } = render(
      <ChatWidget
        serverUrl={SERVER}
        agentId="agent"
        mode="input"
        customerSound={false}
        agentSound={false}
        avatarUrl="/portrait.jpg"
        avatarStyle="photo"
        liveIndicator
      />,
    );

    await screen.findByText("Can't reach the server. Reconnecting…", {}, { timeout: 6000 });
    const dot = () => container.querySelector(".parlant-cw-avatar-frame .parlant-cw-live");
    expect(dot()?.classList.contains("parlant-cw-live--offline")).toBe(true);
    expect(dot()?.getAttribute("aria-label")).toBe("Offline");

    // Dismissed to the launcher, the launcher's dot says the same.
    fireEvent.click(screen.getByLabelText("Dismiss chat"));
    expect(container.querySelector(".parlant-launcher .parlant-cw-live--offline")).not.toBeNull();
    down = false;
  }, 10000);
});

describe("dismissing the closed card", () => {
  it("shrinks an input-mode card to the launcher, which brings it back", async () => {
    server();
    const { container } = render(
      <ChatWidget serverUrl={SERVER} agentId="agent" mode="input" customerSound={false} agentSound={false} />,
    );
    const card = () => container.querySelector(".parlant-input") as HTMLElement;

    fireEvent.click(await screen.findByLabelText("Dismiss chat"));
    expect(card().hidden).toBe(true);

    fireEvent.click(screen.getByLabelText("Open chat"));
    expect(card().hidden).toBe(false);
    expect(container.querySelector(".parlant-launcher")).toBeNull();
  });

  it("isn't offered in bar mode, which has no header", async () => {
    server();
    const { container } = render(
      <ChatWidget serverUrl={SERVER} agentId="agent" mode="bar" customerSound={false} agentSound={false} />,
    );

    await waitFor(() => expect(container.querySelector(".parlant-bar")).not.toBeNull());
    expect(screen.queryByLabelText("Dismiss chat")).toBeNull();
  });
});

describe("header subtitle", () => {
  it("shows the subtitle under the title, and nothing when it isn't given", async () => {
    server();

    const { container, rerender } = render(
      <ChatWidget
        serverUrl={SERVER}
        agentId="agent"
        mode="inline"
        customerSound={false}
        agentSound={false}
        title="Shira"
        subtitle="Head of Product. Talk to my AI."
      />,
    );

    expect((await screen.findByText("Head of Product. Talk to my AI.")).className).toBe("parlant-cw-header-subtitle");
    expect(container.querySelector(".parlant-cw-header-text .parlant-cw-header-name")?.textContent).toBe("Shira");

    rerender(
      <ChatWidget
        serverUrl={SERVER}
        agentId="agent"
        mode="inline"
        customerSound={false}
        agentSound={false}
        title="Shira"
      />,
    );

    expect(container.querySelector(".parlant-cw-header-subtitle")).toBeNull();
    expect(container.querySelector(".parlant-cw-header-name")?.textContent).toBe("Shira");
  });
});

describe("keyboard inset", () => {
  /** A visual viewport the test can shrink, as a keyboard or a mobile browser's toolbar does. */
  function visualViewport() {
    const vv = Object.assign(new EventTarget(), {
      width: window.innerWidth,
      height: window.innerHeight,
      offsetTop: 0,
      offsetLeft: 0,
      scale: 1,
    });
    vi.stubGlobal("visualViewport", vv);
    return (height: number) => {
      vv.height = height;
      vv.dispatchEvent(new Event("resize"));
    };
  }

  const inset = () => document.documentElement.style.getPropertyValue("--parlant-kb-inset");

  it("test_that_a_shrinking_visual_viewport_without_a_focused_field_does_not_lift_the_bar", () => {
    server();
    const shrinkTo = visualViewport();
    render(<ChatWidget serverUrl={SERVER} agentId="agent" mode="bar" />);

    shrinkTo(window.innerHeight - 70);

    expect(inset()).toBe("0px");
  });

  it("test_that_a_focused_text_field_lifts_the_bar_above_the_keyboard", () => {
    server();
    const shrinkTo = visualViewport();
    render(<ChatWidget serverUrl={SERVER} agentId="agent" mode="bar" />);

    screen.getByRole("textbox").focus();
    shrinkTo(window.innerHeight - 300);

    expect(inset()).toBe("300px");
  });
});

describe("choices widget in the transcript", () => {
  const DATES = {
    type: "choices",
    id: "meeting-dates",
    prompt: "Which day suits you?",
    options: [
      { id: "2026-11-03", label: "Tue, Nov 3" },
      { id: "2026-11-04", label: "Wed, Nov 4", reply: "Wednesday, Nov 4 works" },
    ],
  };

  /** A tool event whose single call returns `widget` in its frontend-only metadata. */
  function tool(offset: number, traceId: string, widget: unknown) {
    return {
      id: `event-${offset}`,
      offset,
      kind: "tool",
      source: "system",
      trace_id: traceId,
      creation_utc: `2026-10-24T10:00:0${offset}Z`,
      data: { tool_calls: [{ tool_id: "built-in:list_dates", arguments: {}, result: { data: "", metadata: { widget } } }] },
      metadata: {},
    };
  }

  /** `message`, but in the given turn and at a distinct time. */
  function turn(offset: number, traceId: string, source: "customer" | "ai_agent", text: string) {
    return { ...message(offset, source, text), trace_id: traceId, creation_utc: `2026-10-24T10:00:0${offset}Z` };
  }

  const option = (label: string) => screen.getByRole("button", { name: label }) as HTMLButtonElement;

  it("sends the clicked option as the visitor's message, once, and then disables the options", async () => {
    rememberConversation("earlier-session");
    const { sent } = server([
      turn(0, "trace-a", "customer", "Can we meet?"),
      tool(1, "trace-b", DATES),
      turn(2, "trace-b", "ai_agent", "Sure, which day suits you?"),
    ]);

    render(
      <ChatWidget serverUrl={SERVER} agentId="agent" mode="inline" customerSound={false} agentSound={false} />,
    );

    await waitFor(() => expect(screen.queryByRole("group", { name: "Which day suits you?" })).not.toBeNull());
    expect(option("Wed, Nov 4").disabled).toBe(false);

    fireEvent.click(option("Wed, Nov 4"));
    fireEvent.click(option("Wed, Nov 4"));

    await waitFor(() => expect(sent).toHaveLength(1));
    expect(sent[0].body).toEqual({
      kind: "message",
      source: "customer",
      message: "Wednesday, Nov 4 works",
    });
    // The choice shows in the transcript as the visitor's own message.
    expect(screen.getByText("Wednesday, Nov 4 works")).toBeDefined();
    expect(option("Wed, Nov 4").getAttribute("aria-pressed")).toBe("true");
    expect(option("Tue, Nov 3").disabled).toBe(true);
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(sent).toHaveLength(1);
  });

  it("renders a past turn's options disabled, marking the one the next customer message picked", async () => {
    rememberConversation("earlier-session");
    server([
      turn(0, "trace-a", "customer", "Can we meet?"),
      tool(1, "trace-b", DATES),
      turn(2, "trace-b", "ai_agent", "Sure, which day suits you?"),
      turn(3, "trace-c", "customer", "Tue, Nov 3"),
      turn(4, "trace-c", "ai_agent", "Great, Tuesday it is."),
    ]);

    render(
      <ChatWidget serverUrl={SERVER} agentId="agent" mode="inline" customerSound={false} agentSound={false} />,
    );

    await waitFor(() => expect(screen.queryByText("Great, Tuesday it is.")).not.toBeNull());
    expect(option("Tue, Nov 3").disabled).toBe(true);
    expect(option("Wed, Nov 4").disabled).toBe(true);
    expect(option("Tue, Nov 3").getAttribute("aria-pressed")).toBe("true");
    expect(option("Wed, Nov 4").getAttribute("aria-pressed")).toBe("false");
  });

  it("holds a turn's tool widgets while the turn is open and it has no reply", async () => {
    rememberConversation("earlier-session");
    server([
      turn(0, "trace-a", "customer", "Can we meet?"),
      tool(1, "trace-b", DATES),
      { ...turn(2, "trace-b", "ai_agent", ""), kind: "status", data: { status: "processing" } },
    ]);

    render(
      <ChatWidget serverUrl={SERVER} agentId="agent" mode="inline" customerSound={false} agentSound={false} />,
    );

    await waitFor(() => expect(screen.queryByText("Can we meet?")).not.toBeNull());
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(screen.queryByRole("group", { name: "Which day suits you?" })).toBeNull();
  });

  it("shows the tool widgets of a turn that ended with no reply", async () => {
    rememberConversation("earlier-session");
    server([
      turn(0, "trace-a", "customer", "Can we meet?"),
      tool(1, "trace-b", DATES),
      { ...turn(2, "trace-b", "ai_agent", ""), kind: "status", data: { status: "ready" } },
    ]);

    render(
      <ChatWidget serverUrl={SERVER} agentId="agent" mode="inline" customerSound={false} agentSound={false} />,
    );

    await waitFor(() => expect(screen.queryByRole("group", { name: "Which day suits you?" })).not.toBeNull());
  });

  it("disables options once the visitor types a reply instead", async () => {
    rememberConversation("earlier-session");
    server([
      turn(0, "trace-a", "customer", "Can we meet?"),
      tool(1, "trace-b", DATES),
      turn(2, "trace-b", "ai_agent", "Sure, which day suits you?"),
    ]);

    render(
      <ChatWidget serverUrl={SERVER} agentId="agent" mode="inline" customerSound={false} agentSound={false} />,
    );

    await waitFor(() => expect(option("Tue, Nov 3").disabled).toBe(false));
    const field = screen.getByRole("textbox");
    fireEvent.change(field, { target: { value: "Is Friday possible?" } });
    fireEvent.keyDown(field, { key: "Enter" });

    await waitFor(() => expect(option("Tue, Nov 3").disabled).toBe(true));
    expect(option("Tue, Nov 3").getAttribute("aria-pressed")).toBe("false");
    expect(option("Wed, Nov 4").getAttribute("aria-pressed")).toBe("false");
  });
});

describe("guest identity", () => {
  it("test_that_two_widgets_for_different_agents_on_one_page_share_one_guest_identity", async () => {
    const server = fakeParlantServer();

    render(
      <>
        <ChatWidget
          serverUrl={SERVER}
          agentId="enterprise"
          mode="inline"
          customerSound={false}
          agentSound={false}
        />
        <ChatWidget
          serverUrl={SERVER}
          agentId="docs"
          mode="inline"
          customerSound={false}
          agentSound={false}
        />
      </>,
    );

    await waitFor(() => expect(server.sessionsCreated()).toBe(2));
    expect(server.guestsMinted()).toBe(1);
  });

  it("test_that_a_tokenless_saved_session_shows_the_notice_instead_of_reconnecting", async () => {
    const server = fakeParlantServer();
    const orphan = server.foreignSession();
    localStorage.setItem(SESSION_KEY, orphan);

    render(
      <ChatWidget
        serverUrl={SERVER}
        agentId="agent"
        mode="inline"
        customerSound={false}
        agentSound={false}
      />,
    );

    expect(
      await screen.findByText(
        "The previous conversation was no longer available, so a new one was started.",
      ),
    ).toBeTruthy();
    await new Promise((resolve) => setTimeout(resolve, 1200));
    expect(screen.queryByText(/Connection interrupted/)).toBeNull();
    expect(storedGuestSession(localStorage, "agent")).not.toBe(orphan);
  });
});

/** A status event the engine emitted while working on turn `trace`. */
function status(
  offset: number,
  trace: string,
  data: Record<string, unknown>,
  metadata: Record<string, unknown> = {},
) {
  return {
    id: `event-${offset}`,
    offset,
    kind: "status",
    source: "ai_agent",
    trace_id: trace,
    creation_utc: "2026-10-24T10:00:00Z",
    data,
    metadata,
  };
}

/** A customer or agent message belonging to turn `trace`. */
function turnMessage(offset: number, trace: string, source: "customer" | "ai_agent", text: string) {
  return { ...message(offset, source, text), trace_id: trace };
}

const INTERNAL_REASONING = { __parlant__: { internal_reasoning: true } };
const REASONING = "The customer is asking about order 123; I should check the warehouse first.";

describe("narration and reasoning", () => {
  const thinkingLine = (c: HTMLElement) => c.querySelector(".parlant-status .parlant-reasoning");
  const disclosureLabel = (c: HTMLElement) =>
    c.querySelector(".parlant-reasoning-disclosure--label");
  const disclosureToggle = (c: HTMLElement) =>
    c.querySelector<HTMLButtonElement>(".parlant-msg-reasoning-toggle");
  const disclosurePanel = (c: HTMLElement) => c.querySelector(".parlant-msg-reasoning-panel");

  function renderWidget(extra: Record<string, unknown> = {}) {
    return render(
      <ChatWidget
        serverUrl={SERVER}
        agentId="agent"
        mode="inline"
        customerSound={false}
        agentSound={false}
        thinkingDelay={0}
        streamStyle="instant"
        {...extra}
      />,
    );
  }

  it("test_that_a_tools_narration_is_the_thinking_line_while_the_agent_works", async () => {
    rememberConversation("working-session");
    server([
      turnMessage(0, "t1", "customer", "Where is my order?"),
      status(1, "t1", { status: "processing", message: "Running tools" }),
      status(2, "t1", { status: "processing", message: "Checking with the warehouse…" }),
    ]);

    const { container } = renderWidget();

    await waitFor(() =>
      expect(thinkingLine(container)?.textContent).toBe("Checking with the warehouse…"),
    );
  });

  it("test_that_a_redacted_reasoning_event_shows_as_the_thinking_line_without_a_special_case", async () => {
    rememberConversation("working-session");
    // What a caller without read_event_reasoning receives in place of the reasoning.
    server([
      turnMessage(0, "t1", "customer", "Where is my order?"),
      status(1, "t1", { status: "processing", message: "Thinking" }),
    ]);

    const { container } = renderWidget();

    await waitFor(() => expect(thinkingLine(container)?.textContent).toBe("Thinking"));
  });

  it("test_that_a_turn_without_internal_reasoning_discloses_the_redacted_steps_in_collapsible_mode", async () => {
    rememberConversation("done-session");
    server([
      turnMessage(0, "t1", "customer", "Where is my order?"),
      status(1, "t1", { status: "processing", message: "Thinking" }),
      status(2, "t1", { status: "processing", message: "Thinking" }), // the redacted reasoning
      status(3, "t1", { status: "processing", message: "Running tools" }),
      status(4, "t1", { status: "processing", message: "Checking with the warehouse…" }),
      status(5, "t1", { status: "processing", message: "Reviewing message" }),
      turnMessage(6, "t1", "ai_agent", "Your order ships tomorrow."),
      status(7, "t1", { status: "ready", data: { stage: "completed" } }),
    ]);

    const { container } = renderWidget({ reasoningMode: "collapsible" });

    await waitFor(() => expect(disclosureToggle(container)).not.toBeNull());
    expect(disclosureLabel(container)).toBeNull();

    fireEvent.click(disclosureToggle(container)!);

    const steps = [...container.querySelectorAll(".parlant-msg-reasoning-step")].map(
      (s) => s.textContent,
    );
    expect(steps).toEqual([
      "Thinking",
      "Running tools",
      "Checking with the warehouse…",
      "Reviewing message",
    ]);
  });

  it("test_that_internal_reasoning_is_disclosed_among_the_steps_in_collapsible_mode", async () => {
    rememberConversation("done-session");
    server([
      turnMessage(0, "t1", "customer", "Where is my order?"),
      status(1, "t1", { status: "processing", message: "Thinking" }),
      status(
        2,
        "t1",
        { status: "processing", message: REASONING, chunks: [REASONING, null] },
        INTERNAL_REASONING,
      ),
      status(3, "t1", { status: "processing", message: "Running tools" }),
      status(4, "t1", { status: "processing", message: "Checking with the warehouse…" }),
      status(5, "t1", { status: "processing", message: "Reviewing message" }),
      turnMessage(6, "t1", "ai_agent", "Your order ships tomorrow."),
      status(7, "t1", { status: "ready", data: { stage: "completed" } }),
    ]);

    const { container } = renderWidget({ reasoningMode: "collapsible" });

    await waitFor(() => expect(disclosureToggle(container)).not.toBeNull());
    fireEvent.click(disclosureToggle(container)!);

    const steps = [...container.querySelectorAll(".parlant-msg-reasoning-step")].map(
      (s) => s.textContent,
    );
    expect(steps).toEqual([
      "Thinking",
      REASONING,
      "Running tools",
      "Checking with the warehouse…",
      "Reviewing message",
    ]);
    expect(disclosurePanel(container)?.textContent).toContain(REASONING);
  });

  it("test_that_a_turn_with_no_status_text_shows_a_plain_thought_label_in_collapsible_mode", async () => {
    rememberConversation("done-session");
    server([
      turnMessage(0, "t1", "customer", "Hi"),
      status(1, "t1", { status: "processing", data: { stage: "started" } }),
      turnMessage(2, "t1", "ai_agent", "Hello! How can I help?"),
      status(3, "t1", { status: "ready", data: { stage: "completed" } }),
    ]);

    const { container } = renderWidget({ reasoningMode: "collapsible" });

    await waitFor(() => expect(replies(container)).toHaveLength(1));
    expect(disclosureToggle(container)).toBeNull();
    expect(disclosureLabel(container)).toBeNull();
  });

  const replies = (c: HTMLElement) => [...c.querySelectorAll(".parlant-stream")];
});

/** A fake Parlant server for a remembered session whose live feed the test drives: `push`
 *  delivers an event on the open stream, and `setLabels` changes the labels the session reads
 *  back with, as the engine or a tool stamps them during a turn. */
function liveServer(sessionId: string, history: unknown[] = [], labels: string[] = []) {
  let current = labels;
  let feed: ReadableStreamDefaultController<Uint8Array> | null = null;
  const sessionReads: number[] = [];
  const fetchMock = vi.fn<typeof fetch>(async (input, init) => {
    const url = new URL(String(input));
    const method = init?.method ?? "GET";
    if (url.pathname === "/healthz") return json({ status: "healthy" });
    if (url.pathname === "/agents/agent") return json({ id: "agent", name: "Parlant" });
    if (url.pathname === `/sessions/${sessionId}/events`) {
      if (method === "POST") return json({ offset: 100 }, 201);
      if (url.searchParams.get("sse") !== "true") return json(history);
      return new Response(
        new ReadableStream<Uint8Array>({ start: (controller) => void (feed = controller) }),
        { headers: { "content-type": "text/event-stream" } },
      );
    }
    if (url.pathname === `/sessions/${sessionId}`) {
      sessionReads.push(Date.now());
      return json({ id: sessionId, mode: "auto", labels: current });
    }
    return json({}, 404);
  });
  vi.stubGlobal("fetch", fetchMock);
  return {
    setLabels(next: string[]): void {
      current = next;
    },
    push(event: unknown): void {
      feed!.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(event)}\n\n`));
    },
    /** How many times the widget has read the session. */
    sessionReads: () => sessionReads.length,
    /** Resolves once the live feed is open. */
    opened: () => waitFor(() => expect(feed).not.toBeNull()),
  };
}

/** The status event that ends a turn. */
function ready(offset: number) {
  return {
    id: `event-${offset}`,
    offset,
    kind: "status",
    source: "ai_agent",
    trace_id: `trace-${offset}`,
    creation_utc: "2026-10-26T10:00:00Z",
    data: { status: "ready" },
    metadata: {},
  };
}

/** Wait until the widget treats the feed as live rather than as history it is loading. */
const settle = () => new Promise((resolve) => setTimeout(resolve, 400));

const links = (container: HTMLElement) =>
  [...container.querySelectorAll("a")].map((a) => a.getAttribute("href"));

describe("utm", () => {
  const UTM = { source: "site-chat", medium: "chat-widget" };

  it("test_that_agent_markdown_and_bare_url_links_carry_the_configured_utm_params", async () => {
    rememberConversation("earlier-session");
    server([
      message(0, "customer", "Where do I start?"),
      message(1, "ai_agent", "See [the guide](https://docs.example.com/start#install) or https://example.org/pricing?plan=pro"),
    ]);

    const { container } = render(
      <ChatWidget serverUrl={SERVER} agentId="agent" mode="inline" customerSound={false} agentSound={false} utm={UTM} />,
    );

    await waitFor(() => expect(screen.queryByText("the guide")).not.toBeNull());
    expect(screen.getByText("the guide").closest("a")!.getAttribute("href")).toBe(
      "https://docs.example.com/start?utm_source=site-chat&utm_medium=chat-widget#install",
    );
    expect(links(container)).toContain(
      "https://example.org/pricing?plan=pro&utm_source=site-chat&utm_medium=chat-widget",
    );
  });

  it("test_that_a_utm_key_the_agent_already_wrote_is_kept", async () => {
    rememberConversation("earlier-session");
    server([message(0, "ai_agent", "Read [this](https://example.org/?utm_source=newsletter)")]);

    render(
      <ChatWidget serverUrl={SERVER} agentId="agent" mode="inline" customerSound={false} agentSound={false} utm={UTM} />,
    );

    await waitFor(() => expect(screen.queryByText("this")).not.toBeNull());
    expect(screen.getByText("this").closest("a")!.getAttribute("href")).toBe(
      "https://example.org/?utm_source=newsletter&utm_medium=chat-widget",
    );
  });

  it("test_that_links_are_unchanged_without_a_utm_setting", async () => {
    rememberConversation("earlier-session");
    server([message(0, "ai_agent", "See [the guide](https://docs.example.com/start#install)")]);

    render(<ChatWidget serverUrl={SERVER} agentId="agent" mode="inline" customerSound={false} agentSound={false} />);

    await waitFor(() => expect(screen.queryByText("the guide")).not.toBeNull());
    expect(screen.getByText("the guide").closest("a")!.getAttribute("href")).toBe(
      "https://docs.example.com/start#install",
    );
  });

  it("test_that_external_utm_links_leave_links_to_the_embedding_pages_host_unchanged", async () => {
    const own = `${window.location.origin}/docs/start`;
    rememberConversation("earlier-session");
    server([message(0, "ai_agent", `Read [ours](${own}) and [theirs](https://docs.example.com/start)`)]);

    render(
      <ChatWidget
        serverUrl={SERVER}
        agentId="agent"
        mode="inline"
        customerSound={false}
        agentSound={false}
        utm={UTM}
        utmLinks="external"
      />,
    );

    await waitFor(() => expect(screen.queryByText("ours")).not.toBeNull());
    expect(screen.getByText("ours").closest("a")!.getAttribute("href")).toBe(own);
    expect(screen.getByText("theirs").closest("a")!.getAttribute("href")).toBe(
      "https://docs.example.com/start?utm_source=site-chat&utm_medium=chat-widget",
    );
  });

  it("test_that_all_utm_links_is_the_default_and_tags_links_to_the_embedding_pages_host", async () => {
    const own = `${window.location.origin}/docs/start`;
    rememberConversation("earlier-session");
    server([message(0, "ai_agent", `Read [ours](${own})`)]);

    render(
      <ChatWidget serverUrl={SERVER} agentId="agent" mode="inline" customerSound={false} agentSound={false} utm={UTM} />,
    );

    await waitFor(() => expect(screen.queryByText("ours")).not.toBeNull());
    expect(screen.getByText("ours").closest("a")!.getAttribute("href")).toBe(
      `${own}?utm_source=site-chat&utm_medium=chat-widget`,
    );
  });

  it("test_that_attachment_links_are_left_unchanged", async () => {
    const signed = "https://files.example.com/report.pdf?X-Amz-Signature=abc";
    rememberConversation("earlier-session");
    server([
      {
        ...message(0, "ai_agent", "Here is the report."),
        data: { message: "Here is the report.", attachments: [{ url: signed, filename: "report.pdf" }] },
      },
    ]);

    render(
      <ChatWidget serverUrl={SERVER} agentId="agent" mode="inline" customerSound={false} agentSound={false} utm={UTM} />,
    );

    await waitFor(() => expect(screen.queryByText("report.pdf")).not.toBeNull());
    expect(screen.getByText("report.pdf").closest("a")!.getAttribute("href")).toBe(signed);
  });

  it("test_that_the_powered_by_link_keeps_parlants_own_utm_params", async () => {
    rememberConversation("earlier-session");
    server([]);

    render(
      <ChatWidget serverUrl={SERVER} agentId="agent" mode="inline" customerSound={false} agentSound={false} utm={UTM} />,
    );

    const poweredBy = await screen.findByRole("link", { name: "Powered by Parlant" });
    const params = new URL(poweredBy.getAttribute("href")!).searchParams;
    expect(params.get("utm_source")).toBe(window.location.href);
    expect(params.get("utm_medium")).toBe("chat-widget");
  });

  it("test_that_a_custom_widget_can_tag_its_own_links_with_the_configured_utm_params", async () => {
    registerWidget("utm_test_card", (props, _widget, context) => (
      <a href={context.appendUtm(String(props.href))}>Open card</a>
    ));
    rememberConversation("earlier-session");
    server([
      {
        id: "event-0",
        offset: 0,
        kind: "tool",
        source: "system",
        trace_id: "trace-1",
        creation_utc: "2026-10-26T10:00:00Z",
        data: {
          tool_calls: [
            {
              tool_id: "built-in:card",
              arguments: {},
              result: { data: "", metadata: { widget: { type: "utm_test_card", href: "https://example.org/card" } } },
            },
          ],
        },
        metadata: {},
      },
      { ...message(1, "ai_agent", "Here it is."), trace_id: "trace-1" },
    ]);

    render(
      <ChatWidget serverUrl={SERVER} agentId="agent" mode="inline" customerSound={false} agentSound={false} utm={UTM} />,
    );

    await waitFor(() => expect(screen.queryByText("Open card")).not.toBeNull());
    expect(screen.getByText("Open card").closest("a")!.getAttribute("href")).toBe(
      "https://example.org/card?utm_source=site-chat&utm_medium=chat-widget",
    );
  });
});

describe("onLabel", () => {
  const SESSION = "labeled-session";

  function renderWidget(onLabel: (label: string, context: { sessionId: string; labels: string[] }) => void) {
    return render(
      <ChatWidget
        serverUrl={SERVER}
        agentId="agent"
        mode="inline"
        customerSound={false}
        agentSound={false}
        onLabel={onLabel}
      />,
    );
  }

  it("test_that_on_label_fires_once_for_a_label_a_turn_adds", async () => {
    rememberConversation(SESSION);
    const live = liveServer(SESSION, [], ["docs-failed"]);
    const onLabel = vi.fn();

    renderWidget(onLabel);
    await live.opened();
    await settle();
    live.setLabels(["docs-failed", "lead"]);
    live.push(ready(1));

    await waitFor(() => expect(onLabel).toHaveBeenCalledTimes(1));
    expect(onLabel).toHaveBeenCalledWith("lead", { sessionId: SESSION, labels: ["docs-failed", "lead"] });
  });

  it("test_that_on_label_does_not_fire_for_labels_a_resumed_session_already_has", async () => {
    rememberConversation(SESSION);
    const live = liveServer(SESSION, [message(0, "customer", "Please have someone reach out")], ["lead"]);
    const onLabel = vi.fn();

    renderWidget(onLabel);
    await live.opened();
    await settle();
    const readsBefore = live.sessionReads();
    live.push(ready(1));

    await waitFor(() => expect(live.sessionReads()).toBeGreaterThan(readsBefore));
    await settle();
    expect(onLabel).not.toHaveBeenCalled();
  });

  it("test_that_on_label_does_not_fire_again_for_a_label_already_reported", async () => {
    rememberConversation(SESSION);
    const live = liveServer(SESSION);
    const onLabel = vi.fn();

    renderWidget(onLabel);
    await live.opened();
    await settle();
    live.setLabels(["lead"]);
    live.push(ready(1));
    await waitFor(() => expect(onLabel).toHaveBeenCalledTimes(1));

    live.setLabels(["lead", "pricing"]);
    live.push(ready(2));
    await waitFor(() => expect(onLabel).toHaveBeenCalledTimes(2));
    const readsBefore = live.sessionReads();
    live.push(ready(3));
    await waitFor(() => expect(live.sessionReads()).toBeGreaterThan(readsBefore));
    await settle();

    expect(onLabel.mock.calls.map(([label]) => label)).toEqual(["lead", "pricing"]);
  });

  it("test_that_a_removed_label_does_not_fire_on_label", async () => {
    rememberConversation(SESSION);
    const live = liveServer(SESSION, [], ["lead", "docs-failed"]);
    const onLabel = vi.fn();

    renderWidget(onLabel);
    await live.opened();
    await settle();
    const readsBefore = live.sessionReads();
    live.setLabels(["lead"]);
    live.push(ready(1));

    await waitFor(() => expect(live.sessionReads()).toBeGreaterThan(readsBefore));
    await settle();
    expect(onLabel).not.toHaveBeenCalled();
  });

  it("test_that_a_throwing_on_label_callback_does_not_break_the_conversation", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    rememberConversation(SESSION);
    const live = liveServer(SESSION);
    const onLabel = vi.fn(() => {
      throw new Error("analytics blocked");
    });

    const { container } = renderWidget(onLabel);
    await live.opened();
    await settle();
    live.setLabels(["lead"]);
    live.push(ready(1));
    await waitFor(() => expect(onLabel).toHaveBeenCalledTimes(1));

    // A finished reply (terminated chunks), so the widget doesn't follow it token by token.
    live.push(message(2, "ai_agent", "", ["Our team will reach out soon.", null]));
    await waitFor(() => expect(container.textContent).toContain("Our team will reach out soon."));
  });
});


describe("message sound selection", () => {
  it.each(["input", "bar"] as const)("plays Halo once for the first %s send even when its echo arrives in the initial history", async (mode) => {
    const { sent } = server();
    const baseFetch = fetch;
    let finishHistory: ((response: Response) => void) | undefined;
    vi.stubGlobal("fetch", vi.fn<typeof fetch>(async (input, init) => {
      const url = new URL(String(input));
      if (url.pathname === "/sessions/fresh-session/events") {
        if ((init?.method ?? "GET") === "POST") {
          const response = await baseFetch(input, init);
          expect(finishHistory).toBeDefined();
          finishHistory!(json([message(0, "customer", "Hello Prysmic")]));
          return response;
        }
        if (url.searchParams.get("sse") !== "true") {
          return new Promise<Response>((resolve) => { finishHistory = resolve; });
        }
      }
      return baseFetch(input, init);
    }));
    const play = vi.spyOn(sounds, "playCustomerSound");
    render(<ChatWidget serverUrl={SERVER} agentId="agent" mode={mode}
      soundTheme="halo" agentSound={false} />);

    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Hello Prysmic" } });
    fireEvent.click(screen.getByLabelText("Send"));

    await waitFor(() => expect(sent).toHaveLength(1));
    await waitFor(() => expect(play).toHaveBeenCalledTimes(1));
    expect(play).toHaveBeenCalledWith("halo");
    await settle();
    expect(play).toHaveBeenCalledTimes(1);
  });

  it("keeps resumed customer and agent history silent", async () => {
    rememberConversation("earlier-session");
    server([message(0, "customer", "Earlier question"), message(1, "ai_agent", "Earlier answer")]);
    const customer = vi.spyOn(sounds, "playCustomerSound");
    const agent = vi.spyOn(sounds, "playAgentSound");
    render(<ChatWidget serverUrl={SERVER} agentId="agent" mode="inline" soundTheme="halo" />);
    await screen.findByText("Earlier answer");
    await settle();
    expect(customer).not.toHaveBeenCalled();
    expect(agent).not.toHaveBeenCalled();
  });

  it("uses only the latest sound theme once per live message after repeated theme changes", async () => {
    rememberConversation("sound-session");
    const live = liveServer("sound-session");
    const play = vi.spyOn(sounds, "playAgentSound");
    const config = { serverUrl: SERVER, agentId: "agent", mode: "inline" as const, customerSound: false };
    const view = render(<ChatWidget {...config} soundTheme="glass" />);
    await live.opened();
    await settle();
    for (const theme of ["halo", "anvil", "velvet", "drop"] as const) {
      view.rerender(<ChatWidget {...config} soundTheme={theme} />);
    }
    expect(play).not.toHaveBeenCalled();
    live.push(message(1, "ai_agent", "One reply"));
    await waitFor(() => expect(play).toHaveBeenCalledTimes(1));
    expect(play).toHaveBeenLastCalledWith("drop");
    view.rerender(<ChatWidget {...config} soundTheme="halo" />);
    expect(play).toHaveBeenCalledTimes(1);
    live.push(message(2, "ai_agent", "Next reply"));
    await waitFor(() => expect(play).toHaveBeenCalledTimes(2));
    expect(play).toHaveBeenLastCalledWith("halo");
  });
});

describe("signing in when the agent asks", () => {
  const authorizationRequired = (offset: number) => ({
    id: `event-${offset}`,
    offset,
    kind: "custom",
    source: "system",
    trace_id: `trace-${offset - 1}`,
    creation_utc: "2026-10-30T10:00:00Z",
    data: { type: "authorization_required", scopes: ["orders:read"], reason: "look up your orders" },
    metadata: {},
  });

  const askedToSignIn = () => [
    message(0, "customer", "Where is my order?"),
    authorizationRequired(1),
    message(2, "ai_agent", "Please sign in using the Sign in button to look up your orders."),
  ];

  /** The shared fake server, plus claim and authorize endpoints that record each call, and
   *  a set of tokens the server treats as expired. */
  function serverWithClaims(history: unknown[]) {
    const claims: { sessionId: string; authorization: string | null; body: unknown }[] = [];
    const authorizations: { sessionId: string; authorization: string | null }[] = [];
    const expired = new Set<string>();
    const inner = server(history);
    const base = vi.mocked(fetch);
    vi.stubGlobal(
      "fetch",
      vi.fn<typeof fetch>(async (input, init) => {
        const pathname = new URL(String(input)).pathname;
        // A reply still streaming: its own feed stays open with nothing new yet.
        if (/^\/sessions\/[^/]+\/events\/[^/]+$/.test(pathname)) return idleStream();
        const bearer = new Headers(init?.headers).get("authorization")?.replace(/^Bearer /, "");
        if (bearer && expired.has(bearer)) return json({ detail: "Token expired" }, 401);
        const allow = /^\/sessions\/([^/]+)\/authorize$/.exec(pathname);
        if (allow) {
          authorizations.push({
            sessionId: allow[1],
            authorization: new Headers(init?.headers).get("authorization"),
          });
          return json({ id: allow[1], customer_id: "dana" });
        }
        const claim = /^\/sessions\/([^/]+)\/claim$/.exec(pathname);
        if (claim) {
          claims.push({
            sessionId: claim[1],
            authorization: new Headers(init?.headers).get("authorization"),
            body: JSON.parse(String(init?.body)),
          });
          return json({ id: claim[1], customer_id: "dana" });
        }
        return base(input, init);
      }),
    );
    return { ...inner, claims, authorizations, expired };
  }

  it("shows a Sign in button under the agent's reply: outlined, with an icon", async () => {
    rememberConversation("earlier-session");
    serverWithClaims(askedToSignIn());

    render(
      <ChatWidget serverUrl={SERVER} agentId="agent" mode="inline"
        customerSound={false} agentSound={false}
        auth={{ type: "guest", signIn: async () => null }} />,
    );

    const button = await screen.findByRole("button", { name: "Sign in" });
    expect(button.textContent).toBe("Sign in");
    expect(button.classList.contains("parlant-msg-signin-button")).toBe(true);
    expect(button.querySelector("svg")).not.toBeNull();

    const reply = button.closest(".parlant-msg--agent");
    expect(reply?.textContent).toContain("Please sign in using the Sign in button to look up your orders.");
  });

  it("waits for the agent's reply to finish before offering to sign in", async () => {
    rememberConversation("earlier-session");
    serverWithClaims([
      message(0, "customer", "Where is my order?"),
      authorizationRequired(1),
      message(2, "ai_agent", "", ["Please sign in ", "using the Sign in button"]),
    ]);

    const rendered = render(
      <ChatWidget serverUrl={SERVER} agentId="agent" mode="inline"
        customerSound={false} agentSound={false}
        auth={{ type: "guest", signIn: async () => null }} />,
    );

    const { container } = rendered;
    await waitFor(() => expect(container.querySelector(".parlant-msg--agent")).not.toBeNull());
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(screen.queryByRole("button", { name: "Sign in" })).toBeNull();
  });

  it("shows who signed in, where they signed in", async () => {
    rememberConversation("earlier-session");
    serverWithClaims([
      ...askedToSignIn(),
      {
        id: "event-3",
        offset: 3,
        kind: "custom",
        source: "system",
        trace_id: "trace-3",
        creation_utc: "2026-10-30T10:01:00Z",
        data: { type: "customer.signed_in", customer_id: "dana", name: "Dana Cohen" },
        metadata: {},
      },
      message(4, "ai_agent", "Order 1234 has shipped."),
    ]);

    const { container } = render(
      <ChatWidget serverUrl={SERVER} agentId="agent" mode="inline"
        customerSound={false} agentSound={false}
        auth={{ type: "guest", signIn: async () => null }} />,
    );

    const line = await screen.findByText("Signed in as Dana Cohen");
    expect(screen.queryByRole("button", { name: "Sign in" })).toBeNull();

    // In order: the request to sign in, then the sign-in, then the agent carrying on.
    const transcript = container.querySelector(".parlant-mlist-content")!.textContent!;
    expect(transcript.indexOf("look up your orders")).toBeLessThan(transcript.indexOf("Signed in as Dana Cohen"));
    expect(transcript.indexOf("Signed in as Dana Cohen")).toBeLessThan(transcript.indexOf("Order 1234 has shipped."));
    expect(line.closest('[role="status"]')).not.toBeNull();
  });

  it("signs in with the requested scopes, claims the session and then hides the button", async () => {
    rememberConversation("earlier-session");
    const { claims } = serverWithClaims(askedToSignIn());
    const signIn = vi.fn(async () => ({ token: "customer-token", identityKey: "dana" }));

    render(
      <ChatWidget serverUrl={SERVER} agentId="agent" mode="inline"
        customerSound={false} agentSound={false}
        auth={{ type: "guest", signIn }} />,
    );

    fireEvent.click(await screen.findByRole("button", { name: "Sign in" }));

    await waitFor(() => expect(claims).toHaveLength(1));
    expect(signIn).toHaveBeenCalledWith({ scopes: ["orders:read"], reason: "look up your orders" });
    expect(claims[0]).toEqual({
      sessionId: "earlier-session",
      authorization: "Bearer customer-token",
      body: { guest_token: "guest-token" },
    });
    await waitFor(() => expect(screen.queryByRole("button", { name: "Sign in" })).toBeNull());
  });

  it("keeps the button and claims nothing when the customer cancels signing in", async () => {
    rememberConversation("earlier-session");
    const { claims } = serverWithClaims(askedToSignIn());
    const signIn = vi.fn(async () => null);

    render(
      <ChatWidget serverUrl={SERVER} agentId="agent" mode="inline"
        customerSound={false} agentSound={false}
        auth={{ type: "guest", signIn }} />,
    );

    fireEvent.click(await screen.findByRole("button", { name: "Sign in" }));

    await waitFor(() => expect(signIn).toHaveBeenCalledTimes(1));
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(claims).toHaveLength(0);
    expect(screen.getByRole("button", { name: "Sign in" })).toBeDefined();
  });

  it("offers a signed-in customer Sign in again when the agent needs more, and allows it", async () => {
    localStorage.setItem(`${SESSION_KEY.replace(/:guest$/, "")}:customer:dana`, "earlier-session");
    const { authorizations, claims } = serverWithClaims(askedToSignIn());
    const signIn = vi.fn(async () => ({ token: "more-token", identityKey: "dana" }));

    render(
      <ChatWidget serverUrl={SERVER} agentId="agent" mode="inline"
        customerSound={false} agentSound={false}
        auth={{ type: "customer", identityKey: "dana", getToken: async () => "read-token", signIn }} />,
    );

    fireEvent.click(await screen.findByRole("button", { name: "Sign in" }));

    await waitFor(() => expect(authorizations).toHaveLength(1));
    expect(signIn).toHaveBeenCalledWith({ scopes: ["orders:read"], reason: "look up your orders" });
    expect(authorizations[0]).toEqual({ sessionId: "earlier-session", authorization: "Bearer more-token" });
    expect(claims).toHaveLength(0);
    await waitFor(() => expect(screen.queryByRole("button", { name: "Sign in" })).toBeNull());
  });

  it("offers Sign in again when the token expires and the site gave no way to refresh it", async () => {
    rememberConversation("earlier-session");
    const { claims, expired } = serverWithClaims(askedToSignIn());
    const signIn = vi.fn(async () => ({ token: "short-token", identityKey: "dana" }));

    const { container } = render(
      <ChatWidget serverUrl={SERVER} agentId="agent" mode="inline"
        customerSound={false} agentSound={false}
        auth={{ type: "guest", signIn }} />,
    );

    fireEvent.click(await screen.findByRole("button", { name: "Sign in" }));
    await waitFor(() => expect(claims).toHaveLength(1));
    await waitFor(() => expect(screen.queryByRole("button", { name: "Sign in" })).toBeNull());

    expired.add("short-token");
    const input = container.querySelector("textarea")!;
    fireEvent.change(input, { target: { value: "And the tracking number?" } });
    fireEvent.keyDown(input, { key: "Enter" });

    expect(await screen.findByRole("button", { name: "Sign in" })).toBeDefined();
  });

  it("shows no Sign in button when the site offers no way to sign in", async () => {
    rememberConversation("earlier-session");
    serverWithClaims(askedToSignIn());

    render(
      <ChatWidget serverUrl={SERVER} agentId="agent" mode="inline"
        customerSound={false} agentSound={false} />,
    );

    await screen.findByText("Please sign in using the Sign in button to look up your orders.");
    expect(screen.queryByRole("button", { name: "Sign in" })).toBeNull();
  });
});

describe("launcher photo", () => {
  const photo = { avatarUrl: "/shira.jpg", avatarStyle: "photo" as const };

  it("shows the photo instead of the chat bubble when the card is dismissed", async () => {
    server();
    const { container } = render(<ChatWidget serverUrl={SERVER} agentId="agent" mode="input" {...photo} liveIndicator />);

    fireEvent.click(await screen.findByRole("button", { name: "Dismiss chat" }));

    const launcher = screen.getByRole("button", { name: "Open chat" });
    expect(launcher.querySelector("img")?.getAttribute("src")).toBe("/shira.jpg");
    expect(launcher.getAttribute("data-photo")).toBe("true");
    expect(launcher.querySelector(".parlant-cw-live")).not.toBeNull();
    expect(container.querySelector(".parlant-input")?.hasAttribute("hidden")).toBe(true);
  });

  it("shows the photo on the floating launcher", async () => {
    server();
    render(<ChatWidget serverUrl={SERVER} agentId="agent" mode="floating" {...photo} />);

    const launcher = await screen.findByRole("button", { name: "Open chat" });
    expect(launcher.querySelector("img")?.getAttribute("src")).toBe("/shira.jpg");
    expect(launcher.querySelector(".parlant-cw-live")).toBeNull();
  });

  it("keeps the chat bubble for a logo mark", async () => {
    server();
    render(<ChatWidget serverUrl={SERVER} agentId="agent" mode="floating" avatarUrl="/logo.svg" />);

    const launcher = await screen.findByRole("button", { name: "Open chat" });
    expect(launcher.querySelector("img")).toBeNull();
  });
});

describe("avatar", () => {
  const video = {
    type: "video",
    src: "/shira.mp4",
    poster: "/shira.jpg",
    live: true,
    launcher: { zoom: 1.6, x: "47%", y: "35%" },
  } as const;

  it("fits a logo in the header's avatar and keeps the launcher's chat bubble", async () => {
    server();
    const { container } = render(
      <ChatWidget serverUrl={SERVER} agentId="agent" mode="floating" avatar={{ type: "logo", src: "/logo.svg" }} />,
    );

    const launcher = await screen.findByRole("button", { name: "Open chat" });
    expect(launcher.querySelector("img")).toBeNull();
    fireEvent.click(launcher);
    await waitFor(() => expect(container.querySelector(".parlant-cw-avatar img")?.getAttribute("src")).toBe("/logo.svg"));
    expect(container.querySelector(".parlant-cw-avatar--photo")).toBeNull();
  });

  it("shows a live photo in the header and on the launcher", async () => {
    server();
    const { container } = render(
      <ChatWidget serverUrl={SERVER} agentId="agent" mode="floating" avatar={{ type: "photo", src: "/dana.jpg", live: true }} />,
    );

    const launcher = await screen.findByRole("button", { name: "Open chat" });
    expect(launcher.querySelector("img")?.getAttribute("src")).toBe("/dana.jpg");
    expect(launcher.querySelector(".parlant-cw-live")).not.toBeNull();
    fireEvent.click(launcher);
    await waitFor(() => expect(container.querySelector(".parlant-cw-avatar--photo img")?.getAttribute("src")).toBe("/dana.jpg"));
  });

  it("shows a video under the header in place of the avatar, with the subtitle beside the name", async () => {
    server();
    const { container } = render(
      <ChatWidget serverUrl={SERVER} agentId="agent" mode="input" subtitle="Head of Product" avatar={video} />,
    );

    await waitFor(() => expect(container.querySelector(".parlant-cw-video video")).not.toBeNull());
    const el = container.querySelector(".parlant-cw-video video") as HTMLVideoElement;
    expect(el.getAttribute("src")).toBe("/shira.mp4");
    expect(el.getAttribute("poster")).toBe("/shira.jpg");
    expect(container.querySelector(".parlant-cw-avatar")).toBeNull();
    expect(container.querySelector(".parlant-cw-header-text--inline")).not.toBeNull();
  });

  it("puts the live dot in the video's corner, only when it's live", async () => {
    server();
    const view = render(<ChatWidget serverUrl={SERVER} agentId="agent" mode="input" avatar={video} />);
    await waitFor(() => expect(view.container.querySelector(".parlant-cw-video .parlant-cw-live")).not.toBeNull());

    view.rerender(<ChatWidget serverUrl={SERVER} agentId="agent" mode="input" avatar={{ ...video, live: false }} />);
    expect(view.container.querySelector(".parlant-cw-video .parlant-cw-live")).toBeNull();
  });

  it("moves the video into the transcript once the conversation starts, so it scrolls away", async () => {
    server();
    const { container } = render(
      <ChatWidget serverUrl={SERVER} agentId="agent" mode="input" customerSound={false} agentSound={false} avatar={video} />,
    );

    await waitFor(() => expect(container.querySelector(".parlant-cw-video")).not.toBeNull());
    expect(container.querySelector(".parlant-mlist-content .parlant-cw-video")).toBeNull();

    const input = container.querySelector("textarea")!;
    fireEvent.change(input, { target: { value: "Hi" } });
    fireEvent.keyDown(input, { key: "Enter" });

    await waitFor(() => expect(container.querySelector(".parlant-mlist-content > .parlant-cw-video")).not.toBeNull());
    expect(container.querySelectorAll(".parlant-cw-video")).toHaveLength(1);
  });

  it("shows the video on the launcher, cropped onto the face, with its live dot", async () => {
    server();
    render(<ChatWidget serverUrl={SERVER} agentId="agent" mode="floating" avatar={video} />);

    const launcher = await screen.findByRole("button", { name: "Open chat" });
    expect(launcher.getAttribute("data-video")).toBe("true");
    expect(launcher.querySelector("video")?.getAttribute("src")).toBe("/shira.mp4");
    const crop = launcher.querySelector(".parlant-launcher-photo") as HTMLElement;
    expect(crop.style.getPropertyValue("--parlant-launcher-zoom")).toBe("1.6");
    expect(crop.style.getPropertyValue("--parlant-launcher-x")).toBe("47%");
    expect(crop.style.getPropertyValue("--parlant-launcher-y")).toBe("35%");
    expect(launcher.querySelector(".parlant-cw-live")).not.toBeNull();
  });

  it("prefers avatar over the deprecated avatarUrl", async () => {
    server();
    render(
      <ChatWidget serverUrl={SERVER} agentId="agent" mode="floating" avatarUrl="/old.jpg" avatarStyle="photo"
        avatar={{ type: "photo", src: "/new.jpg" }} />,
    );

    const launcher = await screen.findByRole("button", { name: "Open chat" });
    expect(launcher.querySelector("img")?.getAttribute("src")).toBe("/new.jpg");
  });
});

describe("default video avatar", () => {
  it("plays the widget's own video, with its poster and launcher crop, when a video names none", async () => {
    server();
    render(<ChatWidget serverUrl={SERVER} agentId="agent" mode="floating" avatar={{ type: "video", live: true }} />);

    const launcher = await screen.findByRole("button", { name: "Open chat" });
    const video = launcher.querySelector("video")!;
    expect(video.getAttribute("src")).toMatch(/assets\/agent\.mp4$/);
    expect(video.getAttribute("poster")).toMatch(/assets\/agent-poster\.jpg$/);
    const crop = launcher.querySelector(".parlant-launcher-photo") as HTMLElement;
    expect(crop.style.getPropertyValue("--parlant-launcher-zoom")).toBe("1.6");
    expect(launcher.querySelector(".parlant-cw-live")).not.toBeNull();
  });

  it("keeps a crop given for the default video", async () => {
    server();
    render(<ChatWidget serverUrl={SERVER} agentId="agent" mode="floating" avatar={{ type: "video", launcher: { zoom: 2 } }} />);

    const launcher = await screen.findByRole("button", { name: "Open chat" });
    const crop = launcher.querySelector(".parlant-launcher-photo") as HTMLElement;
    expect(crop.style.getPropertyValue("--parlant-launcher-zoom")).toBe("2");
  });
});

// --- WebMCP: the AI agent in the visitor's browser ---------------------------------------

type BrowserAgentTool = {
  name: string;
  description: string;
  inputSchema: { required?: string[] };
  execute: (
    input: unknown,
    options?: { signal?: AbortSignal },
  ) => Promise<{ content: { type: string; text: string }[] }>;
};

/** The browser's WebMCP broker: what a page registers, and a way to call it as the agent would. */
function browserAgent() {
  const tools = new Map<string, BrowserAgentTool>();
  Object.defineProperty(document, "modelContext", {
    configurable: true,
    value: {
      registerTool: (tool: BrowserAgentTool) => {
        if (tools.has(tool.name)) throw new Error(`A tool named ${tool.name} is registered`);
        tools.set(tool.name, tool);
      },
      unregisterTool: (name: string) => {
        tools.delete(name);
      },
    },
  });
  return tools;
}

/** A fake Parlant server whose agent answers every customer message, over the live stream. */
function talkingServer(reply: (question: string) => string) {
  const created: Record<string, unknown>[] = [];
  const sent: { sessionId: string; body: Record<string, unknown> }[] = [];
  const events = new Map<string, ParlantEvent[]>();
  const streams = new Map<string, Set<ReadableStreamDefaultController<Uint8Array>>>();
  const encoder = new TextEncoder();
  let sessions = 0;
  const push = (sessionId: string, event: Omit<ParlantEvent, "offset" | "id" | "creation_utc" | "trace_id">) => {
    const list = events.get(sessionId)!;
    const full: ParlantEvent = {
      ...event,
      id: `e-${sessionId}-${list.length}`,
      offset: list.length,
      creation_utc: new Date().toISOString(),
      trace_id: "t",
    };
    list.push(full);
    for (const controller of streams.get(sessionId) ?? []) {
      controller.enqueue(encoder.encode(`data: ${JSON.stringify(full)}\n\n`));
    }
  };
  const fetchMock = vi.fn<typeof fetch>(async (input, init) => {
    const url = new URL(String(input));
    const method = init?.method ?? "GET";
    if (url.pathname === "/healthz") return json({ status: "healthy" });
    if (url.pathname === "/agents/agent") return json({ id: "agent", name: "Parlant" });
    if (method === "POST" && url.pathname === "/sessions") {
      const id = `session-${++sessions}`;
      created.push(JSON.parse(String(init?.body)));
      events.set(id, []);
      return json({ id }, 201);
    }
    const eventsPath = /^\/sessions\/([^/]+)\/events$/.exec(url.pathname);
    if (eventsPath && method === "POST") {
      const sessionId = eventsPath[1];
      const body = JSON.parse(String(init?.body));
      sent.push({ sessionId, body });
      push(sessionId, { source: "customer", kind: "message", data: { message: body.message }, metadata: body.metadata });
      setTimeout(() => {
        push(sessionId, { source: "ai_agent", kind: "message", data: { message: reply(body.message) } });
        push(sessionId, { source: "ai_agent", kind: "status", data: { status: "ready" } });
      }, 20);
      return json({ offset: events.get(sessionId)!.length - 1 }, 201);
    }
    if (eventsPath) {
      const sessionId = eventsPath[1];
      const minOffset = Number(url.searchParams.get("min_offset") ?? 0);
      const backlog = (events.get(sessionId) ?? []).filter((e) => e.offset >= minOffset);
      if (url.searchParams.get("sse") !== "true") return json(backlog);
      return new Response(
        new ReadableStream<Uint8Array>({
          start(controller) {
            for (const event of backlog) controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
            if (!streams.has(sessionId)) streams.set(sessionId, new Set());
            streams.get(sessionId)!.add(controller);
          },
        }),
        { headers: { "content-type": "text/event-stream" } },
      );
    }
    const session = /^\/sessions\/([^/]+)$/.exec(url.pathname);
    if (session) return json({ id: session[1], mode: "auto" });
    return json({}, 404);
  });
  vi.stubGlobal("fetch", fetchMock);
  return { created, sent };
}

describe("WebMCP", () => {
  afterEach(() => {
    delete (document as { modelContext?: unknown }).modelContext;
  });

  it("registers ask_agent for the browser's agent, described with the agent's name once known, and removes it on unmount", async () => {
    const tools = browserAgent();
    talkingServer(() => "ok");

    const view = render(
      <ChatWidget serverUrl={SERVER} agentId="agent" mode="inline" customerSound={false} agentSound={false} />,
    );

    await waitFor(() => expect(tools.get("ask_agent")?.description).toContain("Ask Parlant, this site's own AI agent"));
    expect(tools.get("ask_agent")?.inputSchema.required).toEqual(["question"]);

    view.unmount();

    expect(tools.has("ask_agent")).toBe(false);
  });

  it("registers nothing in a browser without WebMCP, nor when the prop is off", async () => {
    talkingServer(() => "ok");
    render(<ChatWidget serverUrl={SERVER} agentId="agent" mode="floating" customerSound={false} agentSound={false} />);
    await new Promise((resolve) => setTimeout(resolve, 30));
    cleanup();

    const tools = browserAgent();
    render(
      <ChatWidget serverUrl={SERVER} agentId="agent" mode="floating" webmcp={false} customerSound={false} agentSound={false} />,
    );
    await new Promise((resolve) => setTimeout(resolve, 30));

    expect(tools.size).toBe(0);
  });

  it("starts the conversation as one an agent represents, shows the question in the chat, opens it, and answers after the turn", async () => {
    const tools = browserAgent();
    const { created, sent } = talkingServer((question) =>
      question.includes("warranty") ? "Yes, every chair has a 5-year warranty." : "Sure.",
    );
    render(<ChatWidget serverUrl={SERVER} agentId="agent" mode="input" customerSound={false} agentSound={false} />);
    await waitFor(() => expect(tools.has("ask_agent")).toBe(true));
    expect(created).toHaveLength(0); // input mode is lazy: no session until something is sent

    const answer = await tools.get("ask_agent")!.execute({ question: "Do chairs have a warranty?" });

    expect(answer).toEqual({ content: [{ type: "text", text: "Yes, every chair has a 5-year warranty." }] });
    expect(created).toEqual([{ agent_id: "agent", customer_represented_by_an_agent: true }]);
    expect(sent[0]?.body).toMatchObject({
      kind: "message",
      source: "customer",
      message: "Do chairs have a warranty?",
      metadata: { parlant_widget: { asked_by: "customer_agent" } },
    });
    expect(await screen.findByText("Do chairs have a warranty?")).toBeTruthy();
    await waitFor(() =>
      expect(document.body.textContent).toContain("Yes, every chair has a 5-year warranty."),
    );
    expect(document.querySelector(".parlant-input--open")).not.toBeNull();
    expect(document.querySelector('[data-asked-by="customer_agent"]')?.textContent).toContain(
      "Do chairs have a warranty?",
    );
  });

  it("continues the same conversation on a follow-up call", async () => {
    const tools = browserAgent();
    const { created, sent } = talkingServer(() => "Noted.");
    render(<ChatWidget serverUrl={SERVER} agentId="agent" mode="input" customerSound={false} agentSound={false} />);
    await waitFor(() => expect(tools.has("ask_agent")).toBe(true));

    await tools.get("ask_agent")!.execute({ question: "I'm Dana from Oslo." });
    await tools.get("ask_agent")!.execute({ question: "Where am I from?" });

    expect(created).toHaveLength(1);
    expect(sent.map((s) => s.sessionId)).toEqual(["session-1", "session-1"]);
  });

  it("registers the tool before the floating panel ever opens, and opens it when the agent's question lands", async () => {
    const tools = browserAgent();
    const { created } = talkingServer(() => "Hello.");
    render(<ChatWidget serverUrl={SERVER} agentId="agent" mode="floating" customerSound={false} agentSound={false} />);
    await waitFor(() => expect(tools.has("ask_agent")).toBe(true));
    expect(tools.get("ask_agent")?.description).toContain("Ask this site's own AI agent");
    expect(document.querySelector(".parlant-panel")).toBeNull();
    expect(created).toHaveLength(0); // no session for a visitor who never opened the chat

    const answer = await tools.get("ask_agent")!.execute({ question: "Hi" });

    expect(answer.content[0]?.text).toBe("Hello.");
    expect(created).toEqual([{ agent_id: "agent", customer_represented_by_an_agent: true }]);

    await waitFor(() =>
      expect(document.querySelector(".parlant-panel")?.getAttribute("aria-hidden")).toBe("false"),
    );
  });
});
