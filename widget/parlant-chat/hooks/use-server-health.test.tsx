// @vitest-environment jsdom

import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ParlantTransport } from "../lib/transport";
import { useServerHealth } from "./use-server-health";

let hidden = false;
let browserOnline = true;

beforeEach(() => {
  vi.useFakeTimers();
  vi.spyOn(console, "warn").mockImplementation(() => {});
  hidden = false;
  browserOnline = true;
  Object.defineProperty(document, "hidden", { configurable: true, get: () => hidden });
  Object.defineProperty(navigator, "onLine", { configurable: true, get: () => browserOnline });
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

function setHidden(value: boolean) {
  hidden = value;
  document.dispatchEvent(new Event("visibilitychange"));
}

function setBrowserOnline(value: boolean) {
  browserOnline = value;
  window.dispatchEvent(new Event(value ? "online" : "offline"));
}

/** A server whose /healthz answers in turn from `answers` (true = healthy, false = the
 *  connection drops), then healthy once they run out. Records when each check arrived. */
function server(answers: boolean[] = []) {
  const checks: number[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn<typeof fetch>(async () => {
      checks.push(Date.now());
      if (answers.shift() === false) throw new TypeError("Failed to fetch");
      return new Response('{"status":"healthy"}', { status: 200 });
    }),
  );
  return { checks };
}

let servers = 0;

function watchHealth() {
  // A fresh URL per test: the poll loop is shared per server URL.
  const transport = new ParlantTransport({
    serverUrl: `https://health-${++servers}.example`,
    agentId: "agent",
  });
  return renderHook(() => useServerHealth(transport));
}

describe("server health", () => {
  it("test_that_a_single_failed_health_check_does_not_report_the_server_offline", async () => {
    const { checks } = server([false]);
    const { result } = watchHealth();

    await act(() => vi.advanceTimersByTimeAsync(2500));

    expect(checks).toHaveLength(2);
    expect(checks[1] - checks[0]).toBe(2000);
    expect(result.current).toBe(true);
  });

  it("test_that_two_failed_health_checks_in_a_row_report_the_server_offline", async () => {
    server([false, false]);
    const { result } = watchHealth();

    await act(() => vi.advanceTimersByTimeAsync(2500));

    expect(result.current).toBe(false);
  });

  it("test_that_a_hidden_tab_skips_health_checks", async () => {
    const { checks } = server();
    watchHealth();
    await act(() => vi.advanceTimersByTimeAsync(0));
    expect(checks).toHaveLength(1);

    act(() => setHidden(true));
    await act(() => vi.advanceTimersByTimeAsync(120_000));

    expect(checks).toHaveLength(1);
  });

  it("test_that_showing_a_hidden_tab_checks_health_at_once", async () => {
    const { checks } = server();
    watchHealth();
    await act(() => vi.advanceTimersByTimeAsync(0));
    act(() => setHidden(true));
    await act(() => vi.advanceTimersByTimeAsync(60_000));

    act(() => setHidden(false));
    await act(() => vi.advanceTimersByTimeAsync(0));

    expect(checks).toHaveLength(2);
  });

  it("test_that_a_tab_shown_again_gets_five_seconds_before_a_failure_reports_the_server_offline", async () => {
    // A phone's network often takes a few seconds to come back when its screen turns on.
    const { checks } = server([true, false]);
    const { result } = watchHealth();
    await act(() => vi.advanceTimersByTimeAsync(0));
    act(() => setHidden(true));
    await act(() => vi.advanceTimersByTimeAsync(60_000));

    act(() => setHidden(false));
    await act(() => vi.advanceTimersByTimeAsync(4999));

    expect(checks).toHaveLength(2);
    expect(result.current).toBe(true);

    await act(() => vi.advanceTimersByTimeAsync(1));

    expect(checks).toHaveLength(3);
    expect(result.current).toBe(true);
  });

  it("test_that_a_server_still_down_after_the_grace_for_a_shown_tab_is_reported_offline", async () => {
    server([true, false, false]);
    const { result } = watchHealth();
    await act(() => vi.advanceTimersByTimeAsync(0));
    act(() => setHidden(true));
    await act(() => vi.advanceTimersByTimeAsync(60_000));

    act(() => setHidden(false));
    await act(() => vi.advanceTimersByTimeAsync(5000));

    expect(result.current).toBe(false);
  });

  it("test_that_health_is_not_checked_while_the_browser_is_offline", async () => {
    browserOnline = false;
    const { checks } = server([false, false, false]);
    const { result } = watchHealth();

    await act(() => vi.advanceTimersByTimeAsync(60_000));

    expect(checks).toHaveLength(0);
    expect(result.current).toBe(true);
  });

  it("test_that_a_failure_while_the_browser_goes_offline_does_not_report_the_server_offline", async () => {
    server([false, false, false]);
    const { result } = watchHealth();
    act(() => setBrowserOnline(false));

    await act(() => vi.advanceTimersByTimeAsync(2500));

    expect(result.current).toBe(true);
  });

  it("test_that_the_browser_coming_back_online_checks_health_at_once", async () => {
    browserOnline = false;
    const { checks } = server();
    watchHealth();
    await act(() => vi.advanceTimersByTimeAsync(60_000));

    act(() => setBrowserOnline(true));
    await act(() => vi.advanceTimersByTimeAsync(0));

    expect(checks).toHaveLength(1);
  });

  it("test_that_the_browser_coming_back_online_gets_five_seconds_before_a_failure_reports_the_server_offline", async () => {
    browserOnline = false;
    const { checks } = server([false]);
    const { result } = watchHealth();
    await act(() => vi.advanceTimersByTimeAsync(1000));

    act(() => setBrowserOnline(true));
    await act(() => vi.advanceTimersByTimeAsync(4999));

    expect(checks).toHaveLength(1);
    expect(result.current).toBe(true);

    await act(() => vi.advanceTimersByTimeAsync(1));

    expect(checks).toHaveLength(2);
    expect(result.current).toBe(true);
  });

  it("test_that_the_browser_reconnecting_during_the_wait_for_a_recheck_extends_the_wait_to_the_grace", async () => {
    const { checks } = server([false, false]);
    const { result } = watchHealth();
    await act(() => vi.advanceTimersByTimeAsync(500));

    act(() => setBrowserOnline(false));
    act(() => setBrowserOnline(true));
    await act(() => vi.advanceTimersByTimeAsync(4999));

    expect(checks).toHaveLength(1);
    expect(result.current).toBe(true);

    await act(() => vi.advanceTimersByTimeAsync(1));

    expect(checks).toHaveLength(2);
  });

  it("test_that_a_tab_shown_during_the_wait_for_a_recheck_extends_the_wait_to_the_grace", async () => {
    const { checks } = server([false, false]);
    const { result } = watchHealth();
    await act(() => vi.advanceTimersByTimeAsync(500));

    act(() => setHidden(true));
    act(() => setHidden(false));
    await act(() => vi.advanceTimersByTimeAsync(4999));

    expect(checks).toHaveLength(1);
    expect(result.current).toBe(true);

    await act(() => vi.advanceTimersByTimeAsync(1));

    expect(checks).toHaveLength(2);
  });

  it("test_that_checks_resume_when_the_browser_is_back_online_without_an_online_event", async () => {
    const { checks } = server([false, false]);
    const { result } = watchHealth();
    await act(() => vi.advanceTimersByTimeAsync(2500));
    expect(result.current).toBe(false);

    browserOnline = false;
    await act(() => vi.advanceTimersByTimeAsync(60_000));
    expect(checks).toHaveLength(2);

    browserOnline = true;
    await act(() => vi.advanceTimersByTimeAsync(5000));

    expect(checks).toHaveLength(3);
    expect(result.current).toBe(true);
  });
});
