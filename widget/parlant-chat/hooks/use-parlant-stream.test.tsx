// @vitest-environment jsdom

import { cleanup, renderHook, waitFor } from "@testing-library/react";
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
