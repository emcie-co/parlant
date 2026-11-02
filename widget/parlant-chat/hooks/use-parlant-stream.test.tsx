// @vitest-environment jsdom

import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ParlantTransport } from "../lib/transport";
import {
  SERVER,
  SharedStorage,
  V1_GUEST_KEY,
  fakeParlantServer,
} from "../testing/fake-parlant-server";
import { useParlantStream } from "./use-parlant-stream";

// Vitest runs without globals here, so unmount by hand: a hook left running would keep
// polling the next test's server.
afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("following a session's events", () => {
  it("test_that_a_local_send_echo_is_fresh_during_history_loading_but_earlier_messages_stay_silent", async () => {
    const server = fakeParlantServer();
    const storage = new SharedStorage();
    const transport = new ParlantTransport({ serverUrl: SERVER, agentId: "home", storage });
    const sessionId = await transport.ensureSession();
    const baseFetch = fetch;
    let finishHistory: ((response: Response) => void) | undefined;
    vi.stubGlobal("fetch", vi.fn<typeof fetch>(async (input, init) => {
      const url = new URL(String(input));
      if (url.pathname === `/sessions/${sessionId}/events` &&
          (init?.method ?? "GET") === "GET" && url.searchParams.get("sse") !== "true") {
        return new Promise<Response>((resolve) => { finishHistory = resolve; });
      }
      return baseFetch(input, init);
    }));
    const { result } = renderHook(() => useParlantStream(transport, sessionId));
    await waitFor(() => expect(finishHistory).toBeDefined());
    await act(() => result.current.send("New question"));
    const event = (offset: number, text: string) => ({
      id: `event-${offset}`, offset, kind: "message", source: "customer",
      trace_id: `trace-${offset}`, creation_utc: "2026-10-28T08:00:00Z",
      data: { message: text }, metadata: {},
    });
    await act(async () => {
      finishHistory!(new Response(JSON.stringify([event(0, "Old question"), event(1, "New question")]),
        { headers: { "content-type": "application/json" } }));
    });
    await waitFor(() => expect(result.current.hydrated).toBe(true));
    expect(result.current.messages.map(({ text, fresh }) => [text, fresh])).toEqual([
      ["Old question", false], ["New question", true],
    ]);
    expect(server.sessionsCreated()).toBe(1);
  });

  it("test_that_a_forbidden_stream_is_not_retried", async () => {
    const server = fakeParlantServer();
    const storage = new SharedStorage();
    storage.setItem(V1_GUEST_KEY, server.mintToken());
    const foreign = server.foreignSession();
    const transport = new ParlantTransport({
      serverUrl: SERVER,
      agentId: "home",
      storage,
    });
    const onSessionExpired = vi.fn();

    renderHook(() =>
      useParlantStream(transport, foreign, { onSessionExpired }),
    );

    await waitFor(() =>
      expect(onSessionExpired).toHaveBeenCalledWith("not-found"),
    );
    const requestsAtExpiry = server.requests.length;
    await new Promise((resolve) => setTimeout(resolve, 1500));
    expect(server.requests.length).toBe(requestsAtExpiry);
    expect(onSessionExpired).toHaveBeenCalledTimes(1);
  });

  it("test_that_server_errors_are_retried_with_backoff_up_to_30_seconds", async () => {
    vi.useFakeTimers();
    const server = fakeParlantServer();
    const storage = new SharedStorage();
    const transport = new ParlantTransport({
      serverUrl: SERVER,
      agentId: "home",
      storage,
    });
    const sessionPromise = transport.ensureSession();
    await vi.advanceTimersByTimeAsync(10);
    const sessionId = await sessionPromise;
    server.fail((r) => r.sse, 503, 100);

    renderHook(() => useParlantStream(transport, sessionId));
    await vi.advanceTimersByTimeAsync(2 * 60 * 1000);

    const attempts = server.requests.filter((r) => r.sse).map((r) => r.at);
    const gaps = attempts.slice(1, 8).map((at, i) => at - attempts[i]);
    expect(gaps).toEqual([1000, 2000, 4000, 8000, 16000, 30000, 30000]);
  });
});
