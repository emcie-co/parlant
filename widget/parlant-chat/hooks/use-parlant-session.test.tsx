// @vitest-environment jsdom

import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ParlantAuth } from "../lib/transport";
import { GuestIdentityExpiredError } from "../lib/transport";
import {
  SERVER,
  SharedLocks,
  SharedStorage,
  V1_GUEST_KEY,
  fakeParlantServer,
  storedGuestSession,
  storedGuestToken,
  v1SessionKey,
} from "../testing/fake-parlant-server";
import { useParlantSession } from "./use-parlant-session";
import { useParlantStream } from "./use-parlant-stream";

class MemoryStorage {
  readonly values = new Map<string, string>();

  getItem(key: string): string | null {
    return this.values.get(key) ?? null;
  }

  setItem(key: string, value: string): void {
    this.values.set(key, value);
  }

  removeItem(key: string): void {
    this.values.delete(key);
  }
}

function session(id: string, guestToken?: string): Response {
  return new Response(JSON.stringify({ id }), {
    status: 201,
    headers: {
      "content-type": "application/json",
      ...(guestToken ? { "x-parlant-guest-token": guestToken } : {}),
    },
  });
}

function authorization(call: unknown[]): string | null {
  const init = call[1] as RequestInit | undefined;
  return new Headers(init?.headers).get("authorization");
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("useParlantSession", () => {
  it("switches cleanly between guest accounts and signed-in customer accounts", async () => {
    const storage = new MemoryStorage();
    const fetchMock = vi.fn<typeof fetch>(async (_input, init) => {
      const auth = new Headers(init?.headers).get("authorization");
      if (auth === "Bearer customer-a-token")
        return session("customer-a-session");
      if (auth === "Bearer customer-b-token")
        return session("customer-b-session");
      return session("guest-session", "guest-token");
    });
    vi.stubGlobal("fetch", fetchMock);

    type Props = { auth: ParlantAuth };
    const { result, rerender } = renderHook(
      ({ auth }: Props) =>
        useParlantSession({
          serverUrl: "https://api.example",
          agentId: "agent",
          storage,
          auth,
        }),
      { initialProps: { auth: { type: "guest" } as ParlantAuth } },
    );

    await waitFor(() => expect(result.current.sessionId).toBe("guest-session"));
    rerender({
      auth: {
        type: "customer",
        identityKey: "customer-a",
        getToken: async () => "customer-a-token",
      },
    });
    await waitFor(() =>
      expect(result.current.sessionId).toBe("customer-a-session"),
    );
    rerender({
      auth: {
        type: "customer",
        identityKey: "customer-b",
        getToken: async () => "customer-b-token",
      },
    });
    await waitFor(() =>
      expect(result.current.sessionId).toBe("customer-b-session"),
    );

    const callsBeforeLogout = fetchMock.mock.calls.length;
    rerender({ auth: { type: "guest" } });
    await waitFor(() => expect(result.current.sessionId).toBe("guest-session"));
    expect(fetchMock).toHaveBeenCalledTimes(callsBeforeLogout);
  });

  it("uses the latest token callback without restarting the same customer session", async () => {
    const storage = new MemoryStorage();
    const firstGetToken = vi.fn(async () => "first-token");
    const latestGetToken = vi.fn(async () => "latest-token");
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(session("customer-session"))
      .mockResolvedValueOnce(new Response(null, { status: 401 }))
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ id: "agent", name: "Support" }), {
          headers: { "content-type": "application/json" },
        }),
      );
    vi.stubGlobal("fetch", fetchMock);

    type Props = { getToken: () => Promise<string> };
    const { result, rerender } = renderHook(
      ({ getToken }: Props) =>
        useParlantSession({
          serverUrl: "https://api.example",
          agentId: "agent",
          storage,
          auth: { type: "customer", identityKey: "customer", getToken },
        }),
      { initialProps: { getToken: firstGetToken } },
    );
    await waitFor(() =>
      expect(result.current.sessionId).toBe("customer-session"),
    );

    const transport = result.current.transport;
    rerender({ getToken: latestGetToken });
    expect(result.current.transport).toBe(transport);
    await expect(result.current.transport.fetchAgent()).resolves.toEqual({
      id: "agent",
      name: "Support",
    });

    expect(firstGetToken).toHaveBeenCalledTimes(1);
    expect(latestGetToken).toHaveBeenCalledTimes(1);
    expect(authorization(fetchMock.mock.calls[1])).toBe("Bearer first-token");
    expect(authorization(fetchMock.mock.calls[2])).toBe("Bearer latest-token");
  });

  it("starts a new anonymous conversation and explains when a guest identity expires", async () => {
    const storage = new MemoryStorage();
    const guestKey = "parlant:v1:guest:https%3A%2F%2Fapi.example";
    const sessionKey =
      "parlant:v1:session:https%3A%2F%2Fapi.example:agent:guest";
    storage.setItem(guestKey, "expired-token");
    storage.setItem(sessionKey, "expired-session");
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(new Response(null, { status: 401 }))
      .mockResolvedValueOnce(session("fresh-session", "fresh-token"));
    vi.stubGlobal("fetch", fetchMock);

    const { result } = renderHook(() =>
      useParlantSession({
        serverUrl: "https://api.example",
        agentId: "agent",
        storage,
      }),
    );
    await waitFor(() =>
      expect(result.current.sessionId).toBe("expired-session"),
    );

    await expect(
      result.current.transport.sendMessage("expired-session", "hello"),
    ).rejects.toBeInstanceOf(GuestIdentityExpiredError);
    act(() => result.current.recover("guest-expired"));

    await waitFor(() => expect(result.current.sessionId).toBe("fresh-session"));
    expect(result.current.notice).toBe(
      "Your session expired, so a new conversation was started.",
    );
    expect(authorization(fetchMock.mock.calls[0])).toBe("Bearer expired-token");
    expect(authorization(fetchMock.mock.calls[1])).toBeNull();
    expect(storedGuestToken(storage)).toBe("fresh-token");
  });

  it("starts a new conversation instead of resuming the stored one when asked for a fresh start", async () => {
    const storage = new MemoryStorage();
    const guestKey = "parlant:v1:guest:https%3A%2F%2Fapi.example";
    const sessionKey =
      "parlant:v1:session:https%3A%2F%2Fapi.example:agent:guest";
    storage.setItem(guestKey, "guest-token");
    storage.setItem(sessionKey, "earlier-session");
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(session("fresh-session"));
    vi.stubGlobal("fetch", fetchMock);

    const { result } = renderHook(() =>
      useParlantSession({
        serverUrl: "https://api.example",
        agentId: "agent",
        storage,
        fresh: true,
      }),
    );

    await waitFor(() => expect(result.current.sessionId).toBe("fresh-session"));
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(storedGuestSession(storage, "agent")).toBe("fresh-session");
    // The visitor keeps their guest identity; only the conversation is new.
    expect(authorization(fetchMock.mock.calls[0])).toBe("Bearer guest-token");
  });
});

describe("guest identity across agents and tabs", () => {
  /** One widget's session and live feed, wired the way ChatWidget wires them. */
  function useWidget(
    agentId: string,
    storage: SharedStorage,
    locks: SharedLocks,
  ) {
    const session = useParlantSession({
      serverUrl: SERVER,
      agentId,
      storage,
      locks,
    });
    useParlantStream(session.transport, session.sessionId, {
      onSessionExpired: session.recover,
    });
    return session;
  }

  it("test_that_a_forbidden_saved_session_starts_a_fresh_conversation_with_the_notice", async () => {
    const server = fakeParlantServer();
    const storage = new SharedStorage();
    storage.setItem(V1_GUEST_KEY, server.mintToken());
    const foreign = server.foreignSession();
    storage.setItem(v1SessionKey("enterprise"), foreign);

    const locks = new SharedLocks();

    const { result } = renderHook(() => useWidget("enterprise", storage, locks));

    await waitFor(() => expect([null, foreign]).not.toContain(result.current.sessionId));
    expect(result.current.notice).toBe(
      "The previous conversation was no longer available, so a new one was started.",
    );
    expect(result.current.error).toBeNull();
  });

  it("test_that_failed_session_creation_is_retried_with_backoff", async () => {
    const server = fakeParlantServer();
    server.fail((r) => r.method === "POST" && r.path === "/sessions", 503, 1);

    const storage = new SharedStorage();

    const { result } = renderHook(() =>
      useParlantSession({ serverUrl: SERVER, agentId: "home", storage }),
    );

    await waitFor(() => expect(result.current.error).not.toBeNull());
    await waitFor(() => expect(result.current.sessionId).not.toBeNull(), {
      timeout: 3000,
    });
    expect(result.current.error).toBeNull();
    expect(server.sessionsCreated()).toBe(1);
  });

  it("test_that_a_new_chat_in_another_tab_moves_this_tab_to_it", async () => {
    fakeParlantServer();
    const storage = new SharedStorage();
    const locks = new SharedLocks();
    const first = renderHook(() => useWidget("home", storage, locks));
    await waitFor(() => expect(first.result.current.sessionId).not.toBeNull());
    const second = renderHook(() => useWidget("home", storage, locks));
    await waitFor(() =>
      expect(second.result.current.sessionId).toBe(
        first.result.current.sessionId,
      ),
    );
    const earlier = first.result.current.sessionId;

    act(() => first.result.current.reset());

    await waitFor(() =>
      expect([null, earlier]).not.toContain(first.result.current.sessionId),
    );
    await waitFor(() =>
      expect(second.result.current.sessionId).toBe(
        first.result.current.sessionId,
      ),
    );
  });

  it("test_that_an_identity_change_in_another_tab_recovers_this_tab", async () => {
    const server = fakeParlantServer();
    const storage = new SharedStorage();
    const locks = new SharedLocks();
    const home = renderHook(() => useWidget("home", storage, locks));
    await waitFor(() => expect(home.result.current.sessionId).not.toBeNull());
    const homeSession = home.result.current.sessionId!;
    const docs = renderHook(() => useWidget("docs", storage, locks));
    await waitFor(() => expect(docs.result.current.sessionId).not.toBeNull());

    // The token expires; the docs tab finds out first and starts over as a new guest.
    server.expire(storedGuestToken(storage)!);
    await expect(
      docs.result.current.transport.sendMessage(
        docs.result.current.sessionId!,
        "hello",
      ),
    ).rejects.toBeInstanceOf(GuestIdentityExpiredError);
    act(() => docs.result.current.recover("guest-expired"));
    await waitFor(() => expect(server.guestsMinted()).toBe(2));

    await waitFor(() =>
      expect([null, homeSession]).not.toContain(home.result.current.sessionId),
    );
    expect(home.result.current.notice).toBe(
      "Your session expired, so a new conversation was started.",
    );
    expect(server.guestsMinted()).toBe(2);
  });
});
