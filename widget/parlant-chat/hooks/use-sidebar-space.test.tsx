// @vitest-environment jsdom
import { useRef } from "react";
import { act, cleanup, render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useSidebarSpace } from "./use-sidebar-space";

const resizes: (() => void)[] = [];
const frames = new Map<number, FrameRequestCallback>();
let nextFrame = 1;

function Sidebar({ active = true, width = 380 }) {
  const ref = useRef<HTMLDivElement>(null);
  useSidebarSpace(ref, active);
  return <div ref={ref} style={{ width }} />;
}
function flushFrames() {
  act(() => {
    const pending = [...frames.values()];
    frames.clear();
    pending.forEach((callback) => callback(0));
  });
}

beforeEach(() => {
  resizes.length = 0;
  document.body.style.marginRight = "0px";
  vi.stubGlobal("innerWidth", 1200);
  vi.spyOn(HTMLElement.prototype, "offsetWidth", "get").mockImplementation(function (this: HTMLElement) {
    return Number.parseFloat(this.style.width) || 0;
  });
  vi.stubGlobal("ResizeObserver", class {
    constructor(callback: () => void) { resizes.push(callback); }
    observe() {}
    disconnect() {}
  });
  vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
    const id = nextFrame++;
    frames.set(id, callback);
    return id;
  });
  vi.stubGlobal("cancelAnimationFrame", (id: number) => frames.delete(id));
});
afterEach(() => {
  cleanup();
  document.body.style.removeProperty("margin-right");
  frames.clear();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("page space beside a sidebar", () => {
  it("reserves space on open and restores the host margin on minimize and unmount", () => {
    document.body.style.setProperty("margin-right", "16px", "important");
    const view = render(<Sidebar active={false} />);
    expect(document.body.style.marginRight).toBe("16px");
    view.rerender(<Sidebar />);
    expect(document.body.style.marginRight).toBe("396px");
    view.rerender(<Sidebar active={false} />);
    expect(document.body.style.marginRight).toBe("16px");
    expect(document.body.style.getPropertyPriority("margin-right")).toBe("important");
    view.rerender(<Sidebar />);
    view.unmount();
    expect(document.body.style.marginRight).toBe("16px");
  });

  it("defers resize layout writes and coalesces notifications into one frame", () => {
    const view = render(<Sidebar />);
    const write = vi.spyOn(document.body.style, "setProperty");
    view.rerender(<Sidebar width={760} />);
    act(() => { resizes[0](); resizes[0](); });
    expect(write).not.toHaveBeenCalled();
    expect(frames.size).toBe(1);
    flushFrames();
    expect(document.body.style.marginRight).toBe("760px");
    write.mockClear();
    act(() => resizes[0]());
    flushFrames();
    expect(write).not.toHaveBeenCalled();
  });

  it("cancels queued writes when unmounted", () => {
    const view = render(<Sidebar />);
    act(() => resizes[0]());
    expect(frames.size).toBe(1);
    view.unmount();
    flushFrames();
    expect(document.body.style.marginRight).toBe("0px");
  });

  it("reserves no space on phones and follows a desktop resize", () => {
    vi.stubGlobal("innerWidth", 640);
    render(<Sidebar />);
    expect(document.body.style.marginRight).toBe("0px");
    vi.stubGlobal("innerWidth", 1200);
    act(() => window.dispatchEvent(new Event("resize")));
    flushFrames();
    expect(document.body.style.marginRight).toBe("380px");
  });

  it("keeps space reserved until the last of multiple sidebars closes", () => {
    const a = render(<Sidebar width={380} />);
    const b = render(<Sidebar width={760} />);
    expect(document.body.style.marginRight).toBe("760px");
    b.unmount();
    expect(document.body.style.marginRight).toBe("380px");
    a.unmount();
    expect(document.body.style.marginRight).toBe("0px");
  });
});
