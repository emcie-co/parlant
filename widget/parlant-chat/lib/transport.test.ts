import { afterEach, describe, expect, it, vi } from "vitest";

import {
  GuestIdentityExpiredError,
  ParlantTransport,
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
        ["parlant:v1:guest:https%3A%2F%2Fapi.example", "guest-token"],
        [
          "parlant:v1:session:https%3A%2F%2Fapi.example:support%20agent:guest",
          "session-1",
        ],
      ]),
    );
  });

  it("isolates sessions by server, agent, and auth identity", async () => {
    const storage = new MemoryStorage();
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(json({ id: "guest-a" }, 201))
      .mockResolvedValueOnce(json({ id: "customer-a" }, 201))
      .mockResolvedValueOnce(json({ id: "customer-b" }, 201))
      .mockResolvedValueOnce(json({ id: "guest-other-server" }, 201));
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
    expect(storage.getItem("parlant:v1:guest:https%3A%2F%2Fapi.example")).toBe("fresh-token");
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
    expect(storage.getItem(guestKey)).toBeNull();
    expect(storage.getItem(sessionKey)).toBeNull();
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
