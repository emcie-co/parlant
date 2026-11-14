// @vitest-environment jsdom
import { act, cleanup, render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useStickToBottom } from "./use-stick-to-bottom";

let resize: () => void = () => {};
let observed: Element[] = [];

function Transcript() {
  const { scrollRef, contentRef } = useStickToBottom();
  return (
    <div ref={scrollRef} data-testid="scroll">
      <div ref={contentRef} />
    </div>
  );
}

beforeEach(() => {
  observed = [];
  vi.stubGlobal("ResizeObserver", class {
    constructor(callback: () => void) { resize = callback; }
    observe(el: Element) { observed.push(el); }
    disconnect() {}
  });
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("sticking to the bottom", () => {
  it("re-pins when the view shrinks, not only when the messages grow", () => {
    const view = render(<Transcript />);
    const scroll = view.getByTestId("scroll");
    expect(observed).toContain(scroll);

    // Pinned at the bottom of 1000px of messages in a 400px view; then the reconnecting
    // banner takes 55px of the view, so the bottom 55px would hide under the composer.
    Object.defineProperty(scroll, "scrollHeight", { configurable: true, value: 1000 });
    scroll.scrollTop = 600;
    act(() => resize());
    expect(scroll.scrollTop).toBe(1000);
  });

  it("leaves the view alone when the reader has scrolled up", () => {
    const view = render(<Transcript />);
    const scroll = view.getByTestId("scroll");
    Object.defineProperty(scroll, "scrollHeight", { configurable: true, value: 1000 });
    Object.defineProperty(scroll, "clientHeight", { configurable: true, value: 400 });
    scroll.scrollTop = 100;
    act(() => { scroll.dispatchEvent(new Event("scroll")); });

    act(() => resize());
    expect(scroll.scrollTop).toBe(100);
  });

  it("stays unpinned while the reader scrolls up through the last lines", () => {
    const view = render(<Transcript />);
    const scroll = view.getByTestId("scroll");
    Object.defineProperty(scroll, "scrollHeight", { configurable: true, value: 1000 });
    Object.defineProperty(scroll, "clientHeight", { configurable: true, value: 400 });
    scroll.scrollTop = 600;
    act(() => { scroll.dispatchEvent(new Event("scroll")); });

    // 30px up from the bottom, within the near-bottom margin, but moving up.
    scroll.scrollTop = 570;
    act(() => { scroll.dispatchEvent(new Event("scroll")); });
    act(() => resize());
    expect(scroll.scrollTop).toBe(570);
  });

  it("unpins when the reader scrolls up, even where the view is within reach of the bottom", () => {
    const view = render(<Transcript />);
    const scroll = view.getByTestId("scroll");
    // A short transcript: at its top, 40px from the bottom, so it counts as pinned.
    Object.defineProperty(scroll, "scrollHeight", { configurable: true, value: 440 });
    Object.defineProperty(scroll, "clientHeight", { configurable: true, value: 400 });
    scroll.scrollTop = 0;
    act(() => { scroll.dispatchEvent(new Event("scroll")); });

    // Scrolling up at the top, then the content grows: the view stays at the top.
    act(() => { scroll.dispatchEvent(new WheelEvent("wheel", { deltaY: -40 })); });
    Object.defineProperty(scroll, "scrollHeight", { configurable: true, value: 600 });
    act(() => resize());
    expect(scroll.scrollTop).toBe(0);
  });
});
