// @vitest-environment jsdom
import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { usePersistentSuggestions } from "./use-persistent-suggestions";

afterEach(() => { cleanup(); localStorage.clear(); });

describe("chosen persistent suggestions", () => {
  it("keeps a first-send choice dismissed when the lazy session is created and resumed", () => {
    const view = renderHook(({ sessionId }) => usePersistentSuggestions("https://agent.example", "agent", sessionId), {
      initialProps: { sessionId: null as string | null },
    });
    act(() => view.result.current.choose("Book a free consultation"));
    view.rerender({ sessionId: "one" });
    expect(view.result.current.chosen.has("Book a free consultation")).toBe(true);
    view.unmount();
    const resumed = renderHook(() => usePersistentSuggestions("https://agent.example", "agent", "one"));
    expect(resumed.result.current.chosen.has("Book a free consultation")).toBe(true);
  });

  it("brings suggestions back for a new session without changing the earlier session", () => {
    const view = renderHook(({ sessionId }) => usePersistentSuggestions("https://agent.example", "agent", sessionId), {
      initialProps: { sessionId: "one" as string | null },
    });
    act(() => view.result.current.choose("How to get started?"));
    view.rerender({ sessionId: null });
    view.rerender({ sessionId: "two" });
    expect(view.result.current.chosen.size).toBe(0);
    view.rerender({ sessionId: "one" });
    expect(view.result.current.chosen.has("How to get started?")).toBe(true);
  });
});
