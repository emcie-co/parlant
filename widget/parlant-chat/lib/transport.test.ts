import { afterEach, describe, expect, it, vi } from "vitest";

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
import {
  GuestIdentityExpiredError,
  ParlantTransport,
  SessionForbiddenError,
  UnauthorizedError,
} from "./transport";

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

function json(body: unknown, status = 200, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...headers },
  });
}

function authorization(call: unknown[]): string | null {
  const init = call[1] as RequestInit | undefined;
  return new Headers(init?.headers).get("authorization");
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("ParlantTransport auth", () => {
  it("captures a server-scoped guest token and sends no customer id", async () => {
    const storage = new MemoryStorage();
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(
        json(
          { id: "session-1" },
          201,
          { "x-parlant-guest-token": "guest-token" },
        ),
      )
      .mockResolvedValueOnce(json({ offset: 1 }, 201));
    vi.stubGlobal("fetch", fetchMock);

    const transport = new ParlantTransport({
      serverUrl: "https://api.example/",
      agentId: "support agent",
      storage,
    });

    expect(await transport.ensureSession()).toBe("session-1");
    expect(authorization(fetchMock.mock.calls[0])).toBeNull();
    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body))).toEqual({
      agent_id: "support agent",
    });

    await transport.sendMessage("session-1", "hello");
    expect(authorization(fetchMock.mock.calls[1])).toBe("Bearer guest-token");
    expect(storage.values).toEqual(
      new Map([
        [
          "parlant:v2:guest:https%3A%2F%2Fapi.example",
          JSON.stringify({ token: "guest-token", sessions: { "support agent": "session-1" } }),
        ],
      ]),
    );
  });

  it("isolates sessions by server, agent, and auth identity", async () => {
    const storage = new MemoryStorage();
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(json({ id: "guest-a" }, 201, { "x-parlant-guest-token": "one" }))
      .mockResolvedValueOnce(json({ id: "customer-a" }, 201))
      .mockResolvedValueOnce(json({ id: "customer-b" }, 201))
      .mockResolvedValueOnce(
        json({ id: "guest-other-server" }, 201, { "x-parlant-guest-token": "two" }),
      );
    vi.stubGlobal("fetch", fetchMock);

    const guest = new ParlantTransport({
      serverUrl: "https://one.example",
      agentId: "agent",
      storage,
    });
    const customerA = new ParlantTransport({
      serverUrl: "https://one.example",
      agentId: "agent",
      storage,
      auth: { type: "customer", identityKey: "customer-a", getToken: async () => "a" },
    });
    const customerB = new ParlantTransport({
      serverUrl: "https://one.example",
      agentId: "agent",
      storage,
      auth: { type: "customer", identityKey: "customer-b", getToken: async () => "b" },
    });
    const otherServer = new ParlantTransport({
      serverUrl: "https://two.example",
      agentId: "agent",
      storage,
    });

    await expect(guest.ensureSession()).resolves.toBe("guest-a");
    await expect(customerA.ensureSession()).resolves.toBe("customer-a");
    await expect(customerB.ensureSession()).resolves.toBe("customer-b");
    await expect(otherServer.ensureSession()).resolves.toBe("guest-other-server");

    const returningGuest = new ParlantTransport({
      serverUrl: "https://one.example",
      agentId: "agent",
      storage,
    });
    await expect(returningGuest.ensureSession()).resolves.toBe("guest-a");
    expect(fetchMock).toHaveBeenCalledTimes(4);
  });

  it.each([
    {
      operation: "session creation",
      response: () => json({ id: "session" }, 201),
      run: (transport: ParlantTransport) => transport.ensureSession(),
    },
    {
      operation: "agent fetch",
      response: () => json({ id: "agent", name: "Support" }),
      run: (transport: ParlantTransport) => transport.fetchAgent(),
    },
    {
      operation: "message send",
      response: () => json({ offset: 2 }, 201),
      run: (transport: ParlantTransport) => transport.sendMessage("session", "hello"),
    },
    {
      operation: "cancellation",
      response: () => new Response(null, { status: 204 }),
      run: (transport: ParlantTransport) => transport.cancel("session"),
    },
    {
      operation: "session event streaming",
      response: () =>
        new Response("", { status: 200, headers: { "content-type": "text/event-stream" } }),
      run: (transport: ParlantTransport) =>
        transport.streamEvents("session", 0, new AbortController().signal).next(),
    },
    {
      operation: "individual event streaming",
      response: () =>
        new Response("", { status: 200, headers: { "content-type": "text/event-stream" } }),
      run: (transport: ParlantTransport) =>
        transport.streamEvent("session", "event", new AbortController().signal).next(),
    },
  ])("refreshes and retries customer auth once for $operation", async ({ response, run }) => {
    const getToken = vi
      .fn<() => Promise<string>>()
      .mockResolvedValueOnce("stale-token")
      .mockResolvedValueOnce("fresh-token");
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(new Response(null, { status: 401 }))
      .mockResolvedValueOnce(response());
    vi.stubGlobal("fetch", fetchMock);

    const transport = new ParlantTransport({
      serverUrl: "https://api.example",
      agentId: "agent",
      storage: new MemoryStorage(),
      auth: { type: "customer", identityKey: "customer", getToken },
    });

    await run(transport);

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(authorization(fetchMock.mock.calls[0])).toBe("Bearer stale-token");
    expect(authorization(fetchMock.mock.calls[1])).toBe("Bearer fresh-token");
    expect(getToken).toHaveBeenCalledTimes(2);
  });

  it("stops after one customer refresh when authorization still fails", async () => {
    const getToken = vi
      .fn<() => Promise<string>>()
      .mockResolvedValueOnce("stale-token")
      .mockResolvedValueOnce("still-invalid");
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response(null, { status: 401 }));
    vi.stubGlobal("fetch", fetchMock);
    const transport = new ParlantTransport({
      serverUrl: "https://api.example",
      agentId: "agent",
      auth: { type: "customer", identityKey: "customer", getToken },
    });

    await expect(transport.fetchAgent()).rejects.toBeInstanceOf(UnauthorizedError);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(getToken).toHaveBeenCalledTimes(2);
  });

  it("single-flights concurrent customer token requests", async () => {
    let resolveToken: ((token: string) => void) | undefined;
    const getToken = vi.fn(
      () =>
        new Promise<string>((resolve) => {
          resolveToken = resolve;
        }),
    );
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(json({ id: "agent", name: "Support" }))
      .mockResolvedValueOnce(json({ offset: 1 }, 201));
    vi.stubGlobal("fetch", fetchMock);
    const transport = new ParlantTransport({
      serverUrl: "https://api.example",
      agentId: "agent",
      auth: { type: "customer", identityKey: "customer", getToken },
    });

    const requests = [transport.fetchAgent(), transport.sendMessage("session", "hello")];
    await vi.waitFor(() => expect(getToken).toHaveBeenCalledTimes(1));
    resolveToken?.("shared-token");
    await Promise.all(requests);

    expect(authorization(fetchMock.mock.calls[0])).toBe("Bearer shared-token");
    expect(authorization(fetchMock.mock.calls[1])).toBe("Bearer shared-token");
  });

  it("recovers anonymously when a stored guest token expires during bootstrap", async () => {
    const storage = new MemoryStorage();
    storage.setItem("parlant:v1:guest:https%3A%2F%2Fapi.example", "expired-token");
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(new Response(null, { status: 401 }))
      .mockResolvedValueOnce(
        json({ id: "fresh-session" }, 201, { "x-parlant-guest-token": "fresh-token" }),
      );
    vi.stubGlobal("fetch", fetchMock);
    const transport = new ParlantTransport({
      serverUrl: "https://api.example",
      agentId: "agent",
      storage,
    });

    await expect(transport.ensureSession()).resolves.toBe("fresh-session");
    expect(authorization(fetchMock.mock.calls[0])).toBe("Bearer expired-token");
    expect(authorization(fetchMock.mock.calls[1])).toBeNull();
    expect(storedGuestToken(storage)).toBe("fresh-token");
  });

  it("clears an expired guest identity and its session without retrying a message", async () => {
    const storage = new MemoryStorage();
    const guestKey = "parlant:v1:guest:https%3A%2F%2Fapi.example";
    const sessionKey = "parlant:v1:session:https%3A%2F%2Fapi.example:agent:guest";
    storage.setItem(guestKey, "expired-token");
    storage.setItem(sessionKey, "old-session");
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(new Response(null, { status: 401 }));
    vi.stubGlobal("fetch", fetchMock);
    const transport = new ParlantTransport({
      serverUrl: "https://api.example",
      agentId: "agent",
      storage,
    });

    await expect(transport.ensureSession()).resolves.toBe("old-session");
    await expect(transport.sendMessage("old-session", "hello")).rejects.toBeInstanceOf(
      GuestIdentityExpiredError,
    );
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(storedGuestToken(storage)).toBeNull();
    expect(storedGuestSession(storage, "agent")).toBeNull();
  });

  it("rejects empty customer identity keys and tokens", async () => {
    expect(
      () =>
        new ParlantTransport({
          serverUrl: "https://api.example",
          agentId: "agent",
          auth: { type: "customer", identityKey: " ", getToken: async () => "token" },
        }),
    ).toThrow("non-empty identityKey");

    const transport = new ParlantTransport({
      serverUrl: "https://api.example",
      agentId: "agent",
      auth: { type: "customer", identityKey: "customer", getToken: async () => " " },
    });
    await expect(transport.fetchAgent()).rejects.toThrow("empty customer token");
  });
});

describe("guest identity", () => {
  // One visitor, one guest identity per server: every agent and every tab of the site uses
  // it, and a conversation is resumed only while the identity that started it is still held.

  function tab(
    agentId: string,
    storage: SharedStorage,
    locks?: SharedLocks,
  ): ParlantTransport {
    return new ParlantTransport({ serverUrl: SERVER, agentId, storage, locks });
  }

  it("test_that_a_saved_session_without_a_guest_token_is_not_resumed", async () => {
    const server = fakeParlantServer();
    const storage = new SharedStorage();
    storage.setItem(v1SessionKey("enterprise"), server.foreignSession());

    const transport = tab("enterprise", storage);

    expect(transport.storedSessionId()).toBeNull();
    const sessionId = await transport.ensureSession();
    await expect(transport.listEvents(sessionId, 0)).resolves.toEqual([]);
  });

  it("test_that_fetch_session_returns_the_sessions_labels", async () => {
    const server = fakeParlantServer();
    const transport = tab("home", new SharedStorage());
    const sessionId = await transport.ensureSession();
    server.setLabels(sessionId, ["docs-failed", "lead"]);

    await expect(transport.fetchSession(sessionId)).resolves.toEqual({
      id: sessionId,
      mode: "auto",
      labels: ["docs-failed", "lead"],
    });
  });

  it("test_that_a_forbidden_saved_session_raises_session_forbidden", async () => {
    const server = fakeParlantServer();
    const storage = new SharedStorage();
    storage.setItem(V1_GUEST_KEY, server.mintToken());
    const foreign = server.foreignSession();
    const transport = tab("home", storage);

    await expect(transport.listEvents(foreign, 0)).rejects.toBeInstanceOf(
      SessionForbiddenError,
    );
    await expect(transport.fetchSession(foreign)).rejects.toBeInstanceOf(
      SessionForbiddenError,
    );
    await expect(
      transport.streamEvents(foreign, 0, new AbortController().signal).next(),
    ).rejects.toBeInstanceOf(SessionForbiddenError);
  });

  it("test_that_clearing_the_guest_identity_drops_every_agents_saved_session", async () => {
    const server = fakeParlantServer();
    const storage = new SharedStorage();
    const home = tab("home", storage);
    const homeSession = await home.ensureSession();
    const firstToken = storedGuestToken(storage)!;
    const docs = tab("docs", storage);
    await docs.ensureSession();

    // Seven days later the token has expired; the docs widget is the first to notice.
    server.expire(firstToken);
    docs.resetSession();
    await docs.ensureSession();

    expect(server.guestsMinted()).toBe(2);
    const reloadedHome = tab("home", storage);
    expect(reloadedHome.storedSessionId()).toBeNull();
    const newHomeSession = await reloadedHome.ensureSession();
    expect(newHomeSession).not.toBe(homeSession);
    await expect(reloadedHome.listEvents(newHomeSession, 0)).resolves.toEqual(
      [],
    );
  });

  it("test_that_each_request_uses_the_latest_stored_guest_token", async () => {
    const server = fakeParlantServer();
    const storage = new SharedStorage();
    const first = tab("home", storage);
    const sessionId = await first.ensureSession();
    const renewed = server.renew(storedGuestToken(storage)!);

    // Another tab's request is answered with the renewed token…
    await tab("docs", storage).ensureSession();
    // …and this tab's next request carries it.
    await first.listEvents(sessionId, 0);

    expect(lastAuthorization(server)).toBe(`Bearer ${renewed}`);
  });

  it("test_that_a_renewed_guest_token_from_any_response_is_stored", async () => {
    const server = fakeParlantServer();
    const storage = new SharedStorage();
    const transport = tab("home", storage);
    const sessionId = await transport.ensureSession();
    const renewed = server.renew(storedGuestToken(storage)!);

    await transport.listEvents(sessionId, 0);

    expect(storedGuestToken(storage)).toBe(renewed);
    expect(tab("home", storage).storedSessionId()).toBe(sessionId);
  });

  it("test_that_v1_storage_keeps_a_session_with_its_token_and_drops_a_tokenless_one", async () => {
    const server = fakeParlantServer();
    const withToken = new SharedStorage();
    const token = server.mintToken();
    const ownSession = server.sessionFor(token);
    withToken.setItem(V1_GUEST_KEY, token);
    withToken.setItem(v1SessionKey("home"), ownSession);

    const tokenless = new SharedStorage();
    tokenless.setItem(v1SessionKey("home"), ownSession);

    const migrated = tab("home", withToken);
    expect(migrated.storedSessionId()).toBe(ownSession);
    await expect(migrated.listEvents(ownSession, 0)).resolves.toEqual([]);
    expect(withToken.getItem(v1SessionKey("home"))).toBeNull();

    expect(tab("home", tokenless).storedSessionId()).toBeNull();
  });

  it("test_that_two_tabs_opened_at_once_by_a_new_visitor_mint_one_identity", async () => {
    const server = fakeParlantServer();
    const storage = new SharedStorage();
    const locks = new SharedLocks();
    const home = tab("home", storage, locks);
    const enterprise = tab("enterprise", storage, locks);

    const [homeSession, enterpriseSession] = await Promise.all([
      home.ensureSession(),
      enterprise.ensureSession(),
    ]);

    expect(server.guestsMinted()).toBe(1);
    await expect(home.listEvents(homeSession, 0)).resolves.toEqual([]);
    await expect(enterprise.listEvents(enterpriseSession, 0)).resolves.toEqual(
      [],
    );
  });

  it("test_that_two_tabs_of_one_agent_opened_at_once_share_one_session", async () => {
    const server = fakeParlantServer();
    const storage = new SharedStorage();
    const locks = new SharedLocks();

    const [first, second] = await Promise.all([
      tab("home", storage, locks).ensureSession(),
      tab("home", storage, locks).ensureSession(),
    ]);

    expect(first).toBe(second);
    expect(server.sessionsCreated()).toBe(1);
  });

  it("test_that_without_navigator_locks_widgets_on_one_page_still_mint_one_identity", async () => {
    const server = fakeParlantServer();
    const storage = new SharedStorage();
    // Browsers before Safari 15.4 have no Web Locks (Node and jsdom may have them).
    vi.stubGlobal("navigator", { userAgent: "no-web-locks" });

    await Promise.all([
      tab("home", storage).ensureSession(),
      tab("docs", storage).ensureSession(),
    ]);

    expect(server.guestsMinted()).toBe(1);
  });
});

function lastAuthorization(
  server: ReturnType<typeof fakeParlantServer>,
): string | null {
  return server.requests.at(-1)?.authorization ?? null;
}
