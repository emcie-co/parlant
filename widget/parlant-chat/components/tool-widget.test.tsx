// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ToolWidget } from "../lib/events";
import { ToolWidgetView, registerWidget, type WidgetContext } from "./tool-widget";

afterEach(() => {
  cleanup();
});

const DATES = {
  type: "choices",
  id: "meeting-dates",
  prompt: "Which day suits you?",
  options: [
    { id: "2026-11-03", label: "Tue, Nov 3" },
    { id: "2026-11-04", label: "Wed, Nov 4", reply: "Wednesday, Nov 4 works" },
  ],
};

function toolWidget(metadata: Record<string, unknown>, toolCallId = "event-1:0"): ToolWidget {
  return { toolCallId, metadata, traceId: "trace-1" };
}

function context(overrides: Partial<WidgetContext> = {}): WidgetContext {
  return { sendMessage: vi.fn(), isLatest: true, appendUtm: (href) => href, ...overrides };
}

const buttons = () => screen.queryAllByRole("button");

describe("choices widget", () => {
  it("renders each option as a button in a group named by the prompt", () => {
    render(<ToolWidgetView widget={toolWidget({ widget: DATES })} context={context()} />);

    const group = screen.getByRole("group", { name: "Which day suits you?" });
    expect(group.getAttribute("data-choices-id")).toBe("meeting-dates");
    expect(buttons().map((b) => b.textContent)).toEqual(["Tue, Nov 3", "Wed, Nov 4"]);
    for (const b of buttons()) {
      expect(b.tagName).toBe("BUTTON");
      expect((b as HTMLButtonElement).disabled).toBe(false);
      expect(b.getAttribute("aria-pressed")).toBe("false");
    }
  });

  it("renders from a `widgets` list too", () => {
    render(<ToolWidgetView widget={toolWidget({ widgets: [DATES] })} context={context()} />);
    expect(buttons()).toHaveLength(2);
  });

  it("sends the label when an option has no reply, exactly once", () => {
    const ctx = context();
    render(<ToolWidgetView widget={toolWidget({ widget: DATES })} context={ctx} />);

    const tue = screen.getByText("Tue, Nov 3");
    fireEvent.click(tue);
    fireEvent.click(tue);
    fireEvent.click(screen.getByText("Wed, Nov 4"));

    expect(ctx.sendMessage).toHaveBeenCalledTimes(1);
    expect(ctx.sendMessage).toHaveBeenCalledWith("Tue, Nov 3");
  });

  it("sends the reply instead of the label when an option has one", () => {
    const ctx = context();
    render(<ToolWidgetView widget={toolWidget({ widget: DATES })} context={ctx} />);

    fireEvent.click(screen.getByText("Wed, Nov 4"));

    expect(ctx.sendMessage).toHaveBeenCalledTimes(1);
    expect(ctx.sendMessage).toHaveBeenCalledWith("Wednesday, Nov 4 works");
  });

  it("marks the chosen option and disables the group after a choice", () => {
    render(<ToolWidgetView widget={toolWidget({ widget: DATES })} context={context()} />);

    fireEvent.click(screen.getByText("Wed, Nov 4"));

    expect(screen.getByText("Wed, Nov 4").getAttribute("aria-pressed")).toBe("true");
    expect(screen.getByText("Tue, Nov 3").getAttribute("aria-pressed")).toBe("false");
    for (const b of buttons()) expect((b as HTMLButtonElement).disabled).toBe(true);
  });

  it("is disabled once a later customer message exists, and marks the option it matches", () => {
    const ctx = context({ isLatest: false, nextCustomerMessage: "Wednesday, Nov 4 works" });
    render(<ToolWidgetView widget={toolWidget({ widget: DATES })} context={ctx} />);

    for (const b of buttons()) expect((b as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText("Wed, Nov 4").getAttribute("aria-pressed")).toBe("true");
    expect(screen.getByText("Tue, Nov 3").getAttribute("aria-pressed")).toBe("false");

    fireEvent.click(screen.getByText("Tue, Nov 3"));
    expect(ctx.sendMessage).not.toHaveBeenCalled();
  });

  it("marks nothing when the later customer message matches no option", () => {
    const ctx = context({ isLatest: false, nextCustomerMessage: "Actually, can we do Friday?" });
    render(<ToolWidgetView widget={toolWidget({ widget: DATES })} context={ctx} />);

    for (const b of buttons()) {
      expect((b as HTMLButtonElement).disabled).toBe(true);
      expect(b.getAttribute("aria-pressed")).toBe("false");
    }
  });

  it("is inert outside a transcript (no context given)", () => {
    render(<ToolWidgetView widget={toolWidget({ widget: DATES })} />);
    for (const b of buttons()) expect((b as HTMLButtonElement).disabled).toBe(true);
  });

  it.each([
    ["no options", { type: "choices", prompt: "Pick one" }],
    ["an empty options list", { type: "choices", options: [] }],
    ["options that aren't a list", { type: "choices", options: "Tue, Wed" }],
    ["an option without an id", { type: "choices", options: [{ label: "Tue" }] }],
    ["an option without a label", { type: "choices", options: [{ id: "tue" }] }],
    ["an option that isn't an object", { type: "choices", options: ["Tue"] }],
    ["a non-text reply", { type: "choices", options: [{ id: "tue", label: "Tue", reply: 3 }] }],
    [
      "duplicate option ids",
      { type: "choices", options: [{ id: "a", label: "A" }, { id: "a", label: "B" }] },
    ],
    [
      "more than eight options",
      {
        type: "choices",
        options: Array.from({ length: 9 }, (_, i) => ({ id: `o${i}`, label: `Option ${i}` })),
      },
    ],
  ])("renders nothing for %s", (_, bad) => {
    const { container } = render(
      <ToolWidgetView widget={toolWidget({ widget: bad })} context={context()} />,
    );
    expect(container.innerHTML).toBe("");
  });

  it("renders the valid widgets next to a malformed one", () => {
    render(
      <ToolWidgetView
        widget={toolWidget({ widgets: [{ type: "choices", options: [] }, DATES] })}
        context={context()}
      />,
    );
    expect(buttons()).toHaveLength(2);
  });
});

describe("custom renderers", () => {
  it("receive a context that can send a message", () => {
    const ctx = context();
    registerWidget("rating", (props, _widget, { sendMessage, isLatest }) => (
      <button type="button" disabled={!isLatest} onClick={() => sendMessage(`${props.stars} stars`)}>
        Rate
      </button>
    ));

    render(<ToolWidgetView widget={toolWidget({ widget: { type: "rating", stars: 5 } })} context={ctx} />);
    fireEvent.click(screen.getByText("Rate"));

    expect(ctx.sendMessage).toHaveBeenCalledWith("5 stars");
  });

  it("still work with the two-argument signature", () => {
    registerWidget("order_card", (props, widget) => (
      <div>
        Order {props.orderId} ({widget.toolCallId})
      </div>
    ));

    render(
      <ToolWidgetView widget={toolWidget({ widget: { type: "order_card", orderId: "5567" } })} />,
    );

    expect(screen.getByText("Order 5567 (event-1:0)")).toBeDefined();
  });

  it("can override the built-in choices widget", async () => {
    // A fresh module, so the override doesn't leak into the other tests' registry.
    vi.resetModules();
    const fresh = await import("./tool-widget");
    fresh.registerWidget("choices", (props) => <p>{props.options.length} custom options</p>);

    render(<fresh.ToolWidgetView widget={toolWidget({ widget: DATES })} context={context()} />);

    expect(screen.getByText("2 custom options")).toBeDefined();
    expect(buttons()).toHaveLength(0);
  });
});
