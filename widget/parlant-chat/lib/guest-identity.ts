// The visitor's guest identity on one Parlant server, shared by every agent and every tab.
//
// One storage record per server holds the guest token and, per agent, the conversation that
// identity started. Because the conversations live inside the identity, replacing or clearing
// the identity drops all of them in one write — no agent can be left holding a conversation
// that the current token isn't allowed to read (the server answers that with a 403).

export type StorageLike = Pick<Storage, "getItem" | "setItem" | "removeItem">;

/** The part of the Web Locks API the widget uses (`navigator.locks` satisfies it). */
export interface LockManagerLike {
  request<T>(name: string, callback: () => Promise<T>): Promise<T>;
}

interface IdentityRecord {
  token: string;
  sessions: Record<string, string>;
}

function part(value: string): string {
  return encodeURIComponent(value);
}

/** The guest a token names (its unverified `sub`), or null if it isn't a JWT. */
function guestOf(token: string): string | null {
  try {
    const payload = token.split(".")[1];
    if (!payload) return null;
    const json = atob(payload.replace(/-/g, "+").replace(/_/g, "/"));
    const sub = (JSON.parse(json) as { sub?: unknown }).sub;
    return typeof sub === "string" ? sub : null;
  } catch {
    return null;
  }
}

export class GuestIdentity {
  /** Agents whose saved conversation was dropped because no token could resume it. */
  private readonly lost = new Set<string>();

  constructor(
    private readonly base: string,
    private readonly storage?: StorageLike,
  ) {}

  private key(): string {
    return `parlant:v2:guest:${part(this.base)}`;
  }

  private v1TokenKey(): string {
    return `parlant:v1:guest:${part(this.base)}`;
  }

  private v1SessionKey(agentId: string): string {
    return `parlant:v1:session:${part(this.base)}:${part(agentId)}:guest`;
  }

  private read(): IdentityRecord | null {
    const value = this.storage?.getItem(this.key());
    if (value) {
      try {
        const record = JSON.parse(value) as IdentityRecord;
        if (typeof record.token === "string" && record.sessions) return record;
      } catch {
        /* unreadable — treat as no identity */
      }
    }

    // Widgets before v2 kept the token on its own; adopt it.
    const v1Token = this.storage?.getItem(this.v1TokenKey());
    if (!v1Token) return null;
    const record = { token: v1Token, sessions: {} };
    this.write(record);
    this.storage?.removeItem(this.v1TokenKey());
    return record;
  }

  private write(record: IdentityRecord | null): void {
    if (record) this.storage?.setItem(this.key(), JSON.stringify(record));
    else this.storage?.removeItem(this.key());
  }

  /** The guest token to send, if this browser has an identity on the server. */
  token(): string | null {
    return this.read()?.token ?? null;
  }

  /** Who the identity is: the guest it names, or the token itself when that can't be read. */
  who(): string | null {
    const token = this.token();
    return token ? (guestOf(token) ?? token) : null;
  }

  /** The conversation this identity has with `agentId`, if any. */
  session(agentId: string): string | null {
    const record = this.read();
    const saved = record?.sessions[agentId];
    if (saved) return saved;

    // A v1 conversation is kept only if there is a token to resume it with.
    const v1Session = this.storage?.getItem(this.v1SessionKey(agentId));
    if (!v1Session) return null;
    this.storage?.removeItem(this.v1SessionKey(agentId));
    if (!record) {
      this.lost.add(agentId);
      return null;
    }
    this.write({ ...record, sessions: { ...record.sessions, [agentId]: v1Session } });
    return v1Session;
  }

  /** Whether `agentId`'s saved conversation was dropped for lack of a token (reported once). */
  takeLost(agentId: string): boolean {
    return this.lost.delete(agentId);
  }

  setSession(agentId: string, sessionId: string): void {
    const record = this.read();
    if (!record) return;
    this.write({ ...record, sessions: { ...record.sessions, [agentId]: sessionId } });
  }

  clearSession(agentId: string): void {
    this.storage?.removeItem(this.v1SessionKey(agentId));
    const record = this.read();
    if (!record || !(agentId in record.sessions)) return;
    const sessions = { ...record.sessions };
    delete sessions[agentId];
    this.write({ ...record, sessions });
  }

  /** Store a token the server handed out. A renewal for the same guest keeps its
   *  conversations; a token for a different guest starts a new identity without them. */
  adopt(token: string): void {
    const record = this.read();
    if (record?.token === token) return;
    const same = record != null && guestOf(record.token) != null && guestOf(record.token) === guestOf(token);
    this.write({ token, sessions: same ? record.sessions : {} });
  }

  /** Forget the identity and every conversation it had. */
  clear(): void {
    this.write(null);
    this.storage?.removeItem(this.v1TokenKey());
  }

  /** Call `listener` when another tab changes this identity or its conversations. */
  subscribe(listener: () => void): () => void {
    if (typeof window === "undefined") return () => {};
    const keys = new Set([this.key(), this.v1TokenKey()]);
    const onStorage = (event: StorageEvent) => {
      if (event.key === null || keys.has(event.key)) listener();
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }
}

// Widgets on one page queue here when the browser has no Web Locks (Safari before 15.4).
// Separate tabs can't be coordinated that way; a conversation they split is caught by the
// server's 403 and restarted.
const pageLocks = new Map<string, Promise<unknown>>();

const pageLockManager: LockManagerLike = {
  request<T>(name: string, callback: () => Promise<T>): Promise<T> {
    const previous = pageLocks.get(name) ?? Promise.resolve();
    const run = previous.then(callback, callback);
    pageLocks.set(
      name,
      run.catch(() => undefined),
    );
    return run;
  },
};

/** The browser's Web Locks, shared by every tab of the site, or a page-wide queue without them. */
export function defaultLocks(): LockManagerLike {
  const locks = (globalThis.navigator as Navigator | undefined)?.locks;
  return locks
    ? { request: <T>(name: string, callback: () => Promise<T>) => locks.request(name, callback) as Promise<T> }
    : pageLockManager;
}
