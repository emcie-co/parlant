// Bootstraps (or resumes) a Parlant session for the widget and exposes the transport.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ParlantTransport, UnauthorizedError, type TransportOptions } from "../lib/transport";

const EXPIRED = "Your session expired, so a new conversation was started.";
const NO_LONGER_AVAILABLE =
  "The previous conversation was no longer available, so a new one was started.";

const RETRY_MIN_MS = 1000;
const RETRY_MAX_MS = 30_000;

export interface UseParlantSession {
  transport: ParlantTransport;
  sessionId: string | null;
  ready: boolean;
  error: Error | null;
  notice: string | null;
  /** Dismiss the recovery notice (e.g. once the customer sends in the new session). */
  clearNotice: () => void;
  /** Create the session now if there isn't one yet (used by lazy modes on expansion). */
  start: () => Promise<string>;
  /** Start a brand-new conversation (forgets the stored session). */
  reset: () => void;
  /** Recover from a session or guest identity that can no longer be resumed. */
  recover: (reason: "not-found" | "guest-expired") => void;
}

export function useParlantSession(
  options: TransportOptions & {
    autoStart?: boolean;
    /** Start a new conversation on mount instead of resuming the stored one. The guest
     *  identity is kept; only the conversation is new. */
    fresh?: boolean;
  },
): UseParlantSession {
  const autoStart = options.autoStart ?? true;
  // Forget the stored session once per mount, not on every re-bootstrap: a transport
  // rebuilt later (e.g. a new auth identity) resumes as usual.
  const freshPending = useRef(options.fresh ?? false);
  const auth = options.auth ?? { type: "guest" as const };
  const authRef = useRef(auth);
  authRef.current = auth;
  const identityKey = auth.type === "customer" ? auth.identityKey : null;

  // Rebuild on identity changes, but keep the latest token callback in a ref so
  // an inline getToken function does not restart the conversation every render.
  const transport = useMemo(
    () =>
      new ParlantTransport({
        serverUrl: options.serverUrl,
        agentId: options.agentId,
        storage: options.storage,
        locks: options.locks,
        auth:
          auth.type === "guest"
            ? { type: "guest" }
            : {
                type: "customer",
                identityKey: identityKey!,
                getToken: () => {
                  const currentAuth = authRef.current;
                  if (
                    currentAuth.type !== "customer" ||
                    currentAuth.identityKey !== identityKey
                  ) {
                    throw new Error("Customer auth identity changed while requesting a token");
                  }
                  return currentAuth.getToken();
                },
              },
      }),
    [options.serverUrl, options.agentId, options.storage, options.locks, auth.type, identityKey],
  );

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const transportRef = useRef(transport);
  const bootstrapGeneration = useRef(0);
  transportRef.current = transport;
  const sessionIdRef = useRef(sessionId);
  sessionIdRef.current = sessionId;
  // Who the current conversation belongs to, to tell another tab's new chat (same visitor)
  // from a replaced guest identity (the conversation is lost).
  const identityRef = useRef<string | null>(null);

  // Create (or resume) the session. Failures are retried with backoff (1s, doubling to 30s)
  // until one succeeds or a newer bootstrap supersedes this one.
  const bootstrap = useCallback((current: ParlantTransport, generation: number) => {
    const stale = () =>
      transportRef.current !== current || bootstrapGeneration.current !== generation;
    let backoff = RETRY_MIN_MS;
    const attempt = () => {
      if (stale()) return;
      current
        .ensureSession()
        .then((id) => {
          if (stale()) return;
          identityRef.current = current.identityId();
          if (current.takeLostSession()) setNotice(NO_LONGER_AVAILABLE);
          setError(null);
          setSessionId(id);
        })
        .catch((e) => {
          if (stale()) return;
          console.error("[parlant] failed to start session", e);
          setError(e as Error);
          // A customer the server refuses won't be accepted on a retry.
          if (e instanceof UnauthorizedError) return;
          setTimeout(attempt, backoff);
          backoff = Math.min(backoff * 2, RETRY_MAX_MS);
        });
    };
    attempt();
  }, []);

  useEffect(() => {
    const generation = ++bootstrapGeneration.current;
    setSessionId(null);
    setError(null);
    setNotice(null);
    if (freshPending.current) {
      freshPending.current = false;
      transport.resetSession();
    }

    // Lazy modes (autoStart=false) don't create a session on mount — they only bind to a
    // stored (resumable) one if it exists. The session is created later via start().
    if (!autoStart) {
      setSessionId(transport.storedSessionId());
      identityRef.current = transport.identityId();
      if (transport.takeLostSession()) setNotice(NO_LONGER_AVAILABLE);
    } else {
      bootstrap(transport, generation);
    }
    return () => {
      // Stop any retry still pending for this mount.
      if (bootstrapGeneration.current === generation) bootstrapGeneration.current++;
    };
  }, [transport, autoStart, bootstrap]);

  // Other tabs share the visitor's identity and conversations. Follow another tab's new chat
  // with this agent, and start over when the identity this conversation belongs to is gone.
  useEffect(
    () =>
      transport.subscribe(() => {
        const current = sessionIdRef.current;
        const stored = transport.storedSessionId();
        if (stored && stored !== current) {
          identityRef.current = transport.identityId();
          setSessionId(stored);
          return;
        }
        if (stored || !current) return;

        if (transport.identityId() !== identityRef.current) setNotice(EXPIRED);
        if (!autoStart) {
          setSessionId(null);
          return;
        }
        // Whatever the other tab is creating, ensureSession waits for it under the shared
        // lock and joins it rather than starting a second conversation.
        bootstrap(transport, ++bootstrapGeneration.current);
      }),
    [transport, autoStart, bootstrap],
  );

  // Create the session on demand (no-op if one already exists / is stored).
  const start = useCallback(async (): Promise<string> => {
    const generation = ++bootstrapGeneration.current;
    const id = await transport.ensureSession();
    if (transportRef.current === transport && bootstrapGeneration.current === generation) {
      identityRef.current = transport.identityId();
      setSessionId(id);
    }
    return id;
  }, [transport]);

  const startFresh = (recoveryNotice: string | null) => {
    const generation = ++bootstrapGeneration.current;
    setNotice(recoveryNotice);
    setError(null);
    setSessionId(null);
    transport.resetSession();
    bootstrap(transport, generation);
  };

  const reset = () => startFresh(null);
  const recover = (reason: "not-found" | "guest-expired") =>
    startFresh(reason === "guest-expired" ? EXPIRED : NO_LONGER_AVAILABLE);

  const clearNotice = useCallback(() => setNotice(null), []);

  return {
    transport,
    sessionId,
    ready: sessionId != null,
    error,
    notice,
    clearNotice,
    start,
    reset,
    recover,
  };
}
