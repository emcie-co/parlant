// Thin fetch + SSE client for one agent's customer sessions.
//
// This is the only file that knows Parlant's HTTP API. Swap it (e.g. for the
// official `parlant-client`) without touching the hooks or components.

import type { AttachmentRef, ParlantEvent, SessionMode } from "./events";

export type ParlantAuth =
  | { type: "guest" }
  | {
      type: "customer";
      /** Stable, app-owned key used only to isolate this customer's persisted session. */
      identityKey: string;
      /** Returns a short-lived Parlant customer token minted by your backend. */
      getToken: () => Promise<string>;
    };

export interface TransportOptions {
  /** Parlant server base URL, e.g. "https://parlant.your-app.com". */
  serverUrl: string;
  /** The agent to converse with. */
  agentId: string;
  /** Authentication mode. Defaults to an anonymous/guest conversation. */
  auth?: ParlantAuth;
  /** Override storage (e.g. for tests / SSR). Defaults to localStorage when available. */
  storage?: Pick<Storage, "getItem" | "setItem" | "removeItem">;
}

const GUEST_TOKEN_HEADER = "x-parlant-guest-token";

export class UnauthorizedError extends Error {
  constructor() {
    super("Unauthorized");
    this.name = "UnauthorizedError";
  }
}

/** A persisted guest credential expired and its conversation can no longer be resumed. */
export class GuestIdentityExpiredError extends UnauthorizedError {
  constructor() {
    super();
    this.message = "Guest identity expired";
    this.name = "GuestIdentityExpiredError";
  }
}

/**
 * The session no longer exists server-side (e.g. the server restarted).
 * Recover by starting a new one.
 */
export class SessionNotFoundError extends Error {
  constructor() {
    super("Session not found");
    this.name = "SessionNotFoundError";
  }
}

export class ParlantTransport {
  private readonly base: string;
  private readonly agentId: string;
  private readonly auth: ParlantAuth;
  private readonly storage?: Pick<Storage, "getItem" | "setItem" | "removeItem">;
  private guestToken: string | null;
  private cachedAuthToken: string | null = null;
  private authTokenRequest: Promise<string> | null = null;

  constructor(opts: TransportOptions) {
    this.base = opts.serverUrl.replace(/\/+$/, "");
    this.agentId = opts.agentId;
    this.auth = opts.auth ?? { type: "guest" };
    if (this.auth.type === "customer" && !this.auth.identityKey.trim()) {
      throw new Error("Customer auth requires a non-empty identityKey");
    }
    this.storage =
      opts.storage ?? (typeof localStorage !== "undefined" ? localStorage : undefined);
    this.guestToken =
      this.auth.type === "guest" ? (this.storage?.getItem(this.guestTokenKey()) ?? null) : null;
  }

  /** The normalized server base URL (trailing slashes stripped). Stable per server, so it
   *  keys the shared health poller across remounts. */
  get baseUrl(): string {
    return this.base;
  }

  private storagePart(value: string): string {
    return encodeURIComponent(value);
  }

  private guestTokenKey(): string {
    return `parlant:v1:guest:${this.storagePart(this.base)}`;
  }

  private sessionKey(): string {
    const scope =
      this.auth.type === "guest"
        ? "guest"
        : `customer:${this.storagePart(this.auth.identityKey)}`;
    return `parlant:v1:session:${this.storagePart(this.base)}:${this.storagePart(this.agentId)}:${scope}`;
  }

  private async customerToken(): Promise<string> {
    if (this.auth.type !== "customer") throw new Error("Customer token requested in guest mode");
    if (this.cachedAuthToken) return this.cachedAuthToken;

    if (!this.authTokenRequest) {
      this.authTokenRequest = this.auth.getToken().then((token) => {
        if (!token.trim()) throw new Error("getToken returned an empty customer token");
        this.cachedAuthToken = token;
        return token;
      });
    }

    try {
      return await this.authTokenRequest;
    } finally {
      this.authTokenRequest = null;
    }
  }

  private async authHeaders(): Promise<Record<string, string>> {
    if (this.auth.type === "customer") {
      return { Authorization: `Bearer ${await this.customerToken()}` };
    }
    if (this.guestToken) return { Authorization: `Bearer ${this.guestToken}` };
    return {};
  }

  private refreshCustomerAuth(): void {
    this.cachedAuthToken = null;
    this.authTokenRequest = null;
  }

  private clearGuestIdentity(): void {
    this.guestToken = null;
    this.storage?.removeItem(this.guestTokenKey());
    this.storage?.removeItem(this.sessionKey());
  }

  private async requestOnce(input: string, init: RequestInit = {}): Promise<Response> {
    const headers = new Headers(init.headers);
    for (const [name, value] of Object.entries(await this.authHeaders())) {
      headers.set(name, value);
    }
    return await fetch(input, { ...init, headers });
  }

  /** Attach auth consistently; customer credentials refresh once on any HTTP 401. */
  private async request(input: string, init: RequestInit = {}): Promise<Response> {
    const response = await this.requestOnce(input, init);
    if (response.status !== 401) return response;

    if (this.auth.type === "customer") {
      this.refreshCustomerAuth();
      const retried = await this.requestOnce(input, init);
      if (retried.status === 401) throw new UnauthorizedError();
      return retried;
    }

    if (this.guestToken) {
      this.clearGuestIdentity();
      throw new GuestIdentityExpiredError();
    }

    throw new UnauthorizedError();
  }

  private captureGuestToken(res: Response): void {
    if (this.auth.type !== "guest") return;
    const token = res.headers.get(GUEST_TOKEN_HEADER);
    if (token) {
      this.guestToken = token;
      this.storage?.setItem(this.guestTokenKey(), token);
    }
  }

  /** Return the cached session id, or create a fresh session. */
  async ensureSession(): Promise<string> {
    const cached = this.storage?.getItem(this.sessionKey());
    if (cached) return cached;

    const create = () =>
      this.request(`${this.base}/sessions?allow_greeting=false`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ agent_id: this.agentId }),
      });

    let res: Response;
    try {
      res = await create();
    } catch (error) {
      // An expired stored guest token cannot resume its old sessions, but session
      // bootstrap itself is safe to retry anonymously with a fresh guest identity.
      if (!(error instanceof GuestIdentityExpiredError)) throw error;
      res = await create();
    }
    if (!res.ok) throw new Error(`Failed to create session: ${res.status}`);
    this.captureGuestToken(res);
    const session = await res.json();
    this.storage?.setItem(this.sessionKey(), session.id);
    return session.id;
  }

  /** Forget the current session (e.g. a "New chat" button). */
  resetSession(): void {
    this.storage?.removeItem(this.sessionKey());
  }

  /** The stored (resumable) session id, if any — read without creating one. */
  storedSessionId(): string | null {
    return this.storage?.getItem(this.sessionKey()) ?? null;
  }

  /** Is the server reachable? Backs the offline banner (GET /healthz, no auth).
   *
   * `signal` lets the caller abort a check whose result is no longer wanted (the poller
   * unmounted or was superseded). Without it, a superseded check would linger as a zombie
   * request until its own timeout, so rapid remounts pile up simultaneous /healthz calls. */
  async checkHealth(timeoutMs = 5000, signal?: AbortSignal): Promise<boolean> {
    if (signal?.aborted) return false;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const onExternalAbort = () => controller.abort();
    signal?.addEventListener("abort", onExternalAbort, { once: true });
    const started =
      typeof performance !== "undefined" ? performance.now() : Date.now();
    try {
      const res = await fetch(`${this.base}/healthz`, { signal: controller.signal });
      // /healthz always answers 200 when the server is up, so a non-ok here is itself
      // worth seeing rather than silently treating as "offline".
      if (!res.ok) console.warn(`[parlant] /healthz -> HTTP ${res.status}`);
      return res.ok;
    } catch (e) {
      // A caller-initiated abort (unmount/supersede) is expected — don't log it as a failure.
      if (signal?.aborted) return false;
      // Show WHY, not just "offline". An AbortError at ~timeoutMs means the request
      // stalled (the connection or the server's event loop was tied up); a fast
      // "Failed to fetch" means the connection was refused/reset.
      const ms = Math.round(
        (typeof performance !== "undefined" ? performance.now() : Date.now()) - started,
      );
      const err = e as Error;
      console.warn(`[parlant] /healthz failed after ${ms}ms: ${err.name}: ${err.message}`);
      return false;
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onExternalAbort);
    }
  }

  /** Load the agent's public profile (used for the header name + avatar). */
  async fetchAgent(): Promise<{ id: string; name: string }> {
    const res = await this.request(`${this.base}/agents/${this.agentId}`);
    if (!res.ok) throw new Error(`Failed to load agent: ${res.status}`);
    const agent = await res.json();
    return { id: agent.id, name: agent.name };
  }

  /** Read the session's current mode ("auto" / "manual"). The event feed only signals a
   *  mid-turn flip to manual, so this is the authoritative read used to seed/refresh it. */
  async fetchSession(
    sessionId: string,
    signal?: AbortSignal,
  ): Promise<{ id: string; mode: SessionMode }> {
    const res = await this.request(
      `${this.base}/sessions/${sessionId}`,
      signal ? { signal } : undefined,
    );
    if (!res.ok) throw new Error(`Failed to load session: ${res.status}`);
    const session = await res.json();
    return { id: session.id, mode: session.mode === "manual" ? "manual" : "auto" };
  }

  async sendMessage(
    sessionId: string,
    message: string,
    attachments?: AttachmentRef[],
  ): Promise<{ offset: number }> {
    const body: Record<string, unknown> = { kind: "message", source: "customer", message };
    if (attachments?.length) body.attachments = attachments;

    const res = await this.request(`${this.base}/sessions/${sessionId}/events`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`Failed to send message: ${res.status}`);
    return await res.json();
  }

  /** Best-effort request to cancel this session's in-flight processing (the "Stop" button). */
  async cancel(sessionId: string): Promise<void> {
    await this.request(`${this.base}/sessions/${sessionId}/events`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ kind: "status", source: "customer", status: "cancelled" }),
    }).catch(() => undefined);
  }

  /** Stream session events (offset-ordered) from `minOffset`, long-polling for new ones.
   *  `onOpen` fires once the connection is established (before any data), so callers can
   *  clear a stale "reconnecting" state even when the reconnected stream is idle. */
  streamEvents(
    sessionId: string,
    minOffset: number,
    signal: AbortSignal,
    onOpen?: () => void,
  ): AsyncGenerator<ParlantEvent> {
    return this.sse(
      `${this.base}/sessions/${sessionId}/events?sse=true&min_offset=${minOffset}&wait_for_data=60`,
      signal,
      onOpen,
    );
  }

  /** Stream one event's updates — used to follow a message's chunks token-by-token. */
  streamEvent(
    sessionId: string,
    eventId: string,
    signal: AbortSignal,
  ): AsyncGenerator<ParlantEvent> {
    return this.sse(`${this.base}/sessions/${sessionId}/events/${eventId}?sse=true`, signal);
  }

  private async *sse(
    url: string,
    signal: AbortSignal,
    onOpen?: () => void,
  ): AsyncGenerator<ParlantEvent> {
    const res = await this.request(url, { signal });
    if (res.status === 404) throw new SessionNotFoundError();
    if (!res.ok || !res.body) throw new Error(`Stream failed: ${res.status}`);
    onOpen?.(); // connection established — the stream is healthy again

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) return;
      buffer += decoder.decode(value, { stream: true });
      let i: number;
      while ((i = buffer.indexOf("\n\n")) >= 0) {
        const block = buffer.slice(0, i);
        buffer = buffer.slice(i + 2);
        const line = block.split("\n").find((l) => l.startsWith("data:"));
        if (line) yield JSON.parse(line.slice(5).trim()) as ParlantEvent;
      }
    }
  }
}
