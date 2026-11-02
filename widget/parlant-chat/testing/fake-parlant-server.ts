// Test-only doubles for the widget's boundaries: a fake Parlant server behind `fetch`, a
// localStorage that tells other "tabs" about writes, and a Web Locks manager.
//
// The server keeps Parlant's guest rules: a request without credentials that creates a
// session mints a new guest and returns its token; a guest may only touch its own sessions
// (403 otherwise, and 403 without any credentials); an unknown or expired token is a 401.

import { vi } from "vitest";

export const SERVER = "https://api.example";

function base64url(value: string): string {
  return btoa(value).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

/** A JWT-shaped guest token naming `guest`. The signature is not checked by the widget. */
export function guestToken(guest: string, version = 1): string {
  return `${base64url('{"alg":"HS256"}')}.${base64url(
    JSON.stringify({ sub: guest, typ: "parlant:guest", v: version }),
  )}.signature`;
}

function json(
  body: unknown,
  status = 200,
  headers: Record<string, string> = {},
): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...headers },
  });
}

/** An event stream that opens and then stays quiet, like a session with nothing new. */
function idleStream(headers: Record<string, string> = {}): Response {
  return new Response(new ReadableStream({ start() {} }), {
    headers: { "content-type": "text/event-stream", ...headers },
  });
}

export interface FakeRequest {
  method: string;
  path: string;
  sse: boolean;
  authorization: string | null;
  at: number;
}

export function fakeParlantServer() {
  const validTokens = new Map<string, string>(); // token -> guest
  const sessionOwners = new Map<string, string>(); // session -> guest
  const sessionLabels = new Map<string, string[]>(); // session -> labels, as the server stamped them
  const requests: FakeRequest[] = [];
  const failures: {
    match: (request: FakeRequest) => boolean;
    status: number;
    times: number;
  }[] = [];
  const renewals = new Map<string, string>(); // old token -> renewed token
  let guests = 0;
  let sessions = 0;

  const mintGuest = (): { guest: string; token: string } => {
    const guest = `guest-${++guests}`;
    const token = guestToken(guest);
    validTokens.set(token, guest);
    return { guest, token };
  };

  const fetchMock = vi.fn<typeof fetch>(async (input, init) => {
    const sentAt = Date.now();
    // Answer asynchronously so concurrent callers interleave, as they would over a real
    // network, without spending (fake) time.
    for (let hop = 0; hop < 5; hop++) await Promise.resolve();

    const url = new URL(String(input));
    const method = init?.method ?? "GET";
    const authorization = new Headers(init?.headers).get("authorization");
    const request: FakeRequest = {
      method,
      path: url.pathname,
      sse: url.searchParams.get("sse") === "true",
      authorization,
      at: sentAt,
    };
    requests.push(request);

    const failure = failures.find((f) => f.times > 0 && f.match(request));
    if (failure) {
      failure.times -= 1;
      return json({ detail: "Unavailable" }, failure.status);
    }

    if (url.pathname === "/healthz") return json({ status: "healthy" });
    const agent = /^\/agents\/([^/]+)$/.exec(url.pathname);
    if (agent) return json({ id: agent[1], name: "Parlant" });

    const token = authorization?.replace(/^Bearer /, "") ?? null;
    const guest = token ? validTokens.get(token) : undefined;
    if (token && !guest) return json({ detail: "Unauthorized" }, 401);
    const renewed = token ? renewals.get(token) : undefined;
    const renewal: Record<string, string> = renewed
      ? { "x-parlant-guest-token": renewed }
      : {};

    if (method === "POST" && url.pathname === "/sessions") {
      const id = `session-${++sessions}`;
      if (guest) {
        sessionOwners.set(id, guest);
        return json({ id }, 201, renewal);
      }
      const minted = mintGuest();
      sessionOwners.set(id, minted.guest);
      return json({ id }, 201, { "x-parlant-guest-token": minted.token });
    }

    const match = /^\/sessions\/([^/]+)(\/events.*)?$/.exec(url.pathname);
    if (match) {
      const owner = sessionOwners.get(match[1]);
      if (!owner) return json({ detail: "Not found" }, 404);
      if (owner !== guest) return json({ detail: "Forbidden" }, 403);
      if (!match[2])
        return json(
          { id: match[1], mode: "auto", labels: sessionLabels.get(match[1]) ?? [] },
          200,
          renewal,
        );
      if (method === "POST") return json({ offset: 1 }, 201, renewal);
      return request.sse ? idleStream(renewal) : json([], 200, renewal);
    }

    return json({}, 404);
  });

  vi.stubGlobal("fetch", fetchMock);

  return {
    fetch: fetchMock,
    requests,
    /** How many guest identities the server has minted. */
    guestsMinted: () => guests,
    /** How many sessions the server has created. */
    sessionsCreated: () => sessions,
    /** Expire a guest's token, as seven days of silence would. */
    expire(tokenToExpire: string): void {
      validTokens.delete(tokenToExpire);
    },
    /** From now on, answer requests carrying `old` with a renewed token for the same guest. */
    renew(old: string): string {
      const guest = validTokens.get(old)!;
      const renewed = guestToken(guest, 2);
      validTokens.set(renewed, guest);
      renewals.set(old, renewed);
      return renewed;
    },
    /** Make the next `times` matching requests fail with `status`. */
    fail(
      match: (request: FakeRequest) => boolean,
      status: number,
      times: number,
    ): void {
      failures.push({ match, status, times });
    },
    /** Register a session owned by a guest who never visited from this browser. */
    foreignSession(): string {
      const { guest } = mintGuest();
      const id = `session-${++sessions}`;
      sessionOwners.set(id, guest);
      return id;
    },
    /** Give a session these labels, as the engine or a tool would stamp them. */
    setLabels(sessionId: string, labels: string[]): void {
      sessionLabels.set(sessionId, labels);
    },
    /** A token for a new guest, as if minted for this browser earlier. */
    mintToken(): string {
      return mintGuest().token;
    },
    /** Register a session owned by the guest holding `token`. */
    sessionFor(token: string): string {
      const id = `session-${++sessions}`;
      sessionOwners.set(id, validTokens.get(token)!);
      return id;
    },
  };
}

/** localStorage shared by every "tab" in a test. A write tells the page about it the way the
 *  browser tells *other* tabs: with a `storage` event, delivered later as its own task. */
export class SharedStorage {
  readonly values = new Map<string, string>();

  getItem(key: string): string | null {
    return this.values.get(key) ?? null;
  }

  setItem(key: string, value: string): void {
    const oldValue = this.getItem(key);
    this.values.set(key, value);
    this.announce(key, oldValue, value);
  }

  removeItem(key: string): void {
    const oldValue = this.getItem(key);
    this.values.delete(key);
    this.announce(key, oldValue, null);
  }

  private announce(key: string, oldValue: string | null, newValue: string | null): void {
    if (typeof window === "undefined" || oldValue === newValue) return;
    setTimeout(() => window.dispatchEvent(new StorageEvent("storage", { key, oldValue, newValue })));
  }
}

/** The Web Locks API as every tab of one site sees it: exclusive, queued per name. */
export class SharedLocks {
  private readonly tails = new Map<string, Promise<unknown>>();

  request<T>(name: string, callback: () => Promise<T>): Promise<T> {
    const previous = this.tails.get(name) ?? Promise.resolve();
    const run = previous.then(callback, callback);
    this.tails.set(
      name,
      run.catch(() => undefined),
    );
    return run;
  }
}

export const V1_GUEST_KEY = "parlant:v1:guest:https%3A%2F%2Fapi.example";

export function v1SessionKey(agentId: string): string {
  return `parlant:v1:session:https%3A%2F%2Fapi.example:${agentId}:guest`;
}

type ReadableStorage = Pick<Storage, "getItem">;

interface StoredIdentity {
  token: string;
  sessions: Record<string, string>;
}

function storedIdentity(
  storage: ReadableStorage,
  server = SERVER,
): StoredIdentity | null {
  const value = storage.getItem(
    `parlant:v2:guest:${encodeURIComponent(server)}`,
  );
  return value ? (JSON.parse(value) as StoredIdentity) : null;
}

/** The guest token the widget holds for `server`, in either storage format. */
export function storedGuestToken(
  storage: ReadableStorage,
  server = SERVER,
): string | null {
  return (
    storedIdentity(storage, server)?.token ??
    storage.getItem(`parlant:v1:guest:${encodeURIComponent(server)}`)
  );
}

/** The guest conversation the widget would resume with `agentId`, in either storage format. */
export function storedGuestSession(
  storage: ReadableStorage,
  agentId: string,
  server = SERVER,
): string | null {
  return (
    storedIdentity(storage, server)?.sessions[agentId] ??
    storage.getItem(
      `parlant:v1:session:${encodeURIComponent(server)}:${encodeURIComponent(agentId)}:guest`,
    )
  );
}
