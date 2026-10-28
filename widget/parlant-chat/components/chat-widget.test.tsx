// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  V1_GUEST_KEY,
  fakeParlantServer,
  storedGuestSession,
} from "../testing/fake-parlant-server";
import { ChatWidget } from "./chat-widget";
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
