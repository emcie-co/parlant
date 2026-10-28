// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { Composer } from "./composer";

// Vitest runs without globals here, so RTL's automatic per-test cleanup isn't
// registered — unmount by hand or each render stacks onto the previous DOM.
afterEach(cleanup);

const noop = () => {};

describe("composer placeholder", () => {
  it("shows the built-in placeholder when none is given", () => {
    render(<Composer onSend={noop} onStop={noop} busy={false} />);

    expect(screen.getByPlaceholderText("Type a message…")).toBeDefined();
  });

  it("shows a custom placeholder instead", () => {
    render(
      <Composer onSend={noop} onStop={noop} busy={false} placeholder="Ask about your order…" />,
    );

    expect(screen.getByPlaceholderText("Ask about your order…")).toBeDefined();
    expect(screen.queryByPlaceholderText("Type a message…")).toBeNull();
  });
});

describe("composer bar mode", () => {
  const field = (c: HTMLElement) => c.querySelector(".parlant-cmp-field");
  const isBare = (c: HTMLElement) =>
    field(c)?.classList.contains("parlant-cmp-field--bare") ?? false;
  const chips = (c: HTMLElement) => c.querySelector(".parlant-suggestions");
  const input = (c: HTMLElement) => c.querySelector(".parlant-cmp-input") as HTMLTextAreaElement;

  it("drops the field container in bar mode, leaving just the input", () => {
    const { container } = render(<Composer onSend={noop} onStop={noop} busy={false} barMode />);

    expect(isBare(container)).toBe(true);
  });

  it("keeps the field container in bar mode during a connection issue", () => {
    const { container } = render(
      <Composer onSend={noop} onStop={noop} busy={false} barMode connectionIssue />,
    );

    expect(isBare(container)).toBe(false);
  });

  it("keeps the field container outside bar mode", () => {
    const { container } = render(<Composer onSend={noop} onStop={noop} busy={false} />);

    expect(isBare(container)).toBe(false);
  });

  it("does not show suggestion chips in bar mode until the input is focused", () => {
    const { container } = render(
      <Composer
        onSend={noop}
        onStop={noop}
        busy={false}
        barMode
        autoFocus={false}
        suggestions={["Track my order"]}
      />,
    );

    expect(chips(container)).toBeNull();

    fireEvent.focus(input(container));

    expect(chips(container)).not.toBeNull();
    expect(screen.getByText("Track my order")).toBeDefined();
  });

  it("sends a bar-mode suggestion when its floating chip is clicked", () => {
    const sent: string[] = [];
    const { container } = render(
      <Composer
        onSend={(t) => sent.push(t)}
        onStop={noop}
        busy={false}
        barMode
        autoFocus={false}
        suggestions={["Track my order"]}
      />,
    );

    fireEvent.focus(input(container));
    fireEvent.click(screen.getByText("Track my order"));

    expect(sent).toEqual(["Track my order"]);
  });
});

describe("composer suggestions (non-bar)", () => {
  it("renders suggestion chips inline right away", () => {
    const { container } = render(
      <Composer onSend={noop} onStop={noop} busy={false} suggestions={["Hi there"]} />,
    );

    expect(container.querySelector(".parlant-suggestions")).not.toBeNull();
  });
});
