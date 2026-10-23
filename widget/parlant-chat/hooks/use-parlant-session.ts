// Bootstraps (or resumes) a Parlant session for the widget and exposes the transport.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ParlantTransport, type TransportOptions } from "../lib/transport";

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
    [options.serverUrl, options.agentId, options.storage, auth.type, identityKey],
  );

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const transportRef = useRef(transport);
  const bootstrapGeneration = useRef(0);
  transportRef.current = transport;

  useEffect(() => {
    let cancelled = false;
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
      return () => {
        cancelled = true;
      };
    }

    transport
      .ensureSession()
      .then((id) => {
        if (
          !cancelled &&
          transportRef.current === transport &&
          bootstrapGeneration.current === generation
        ) {
          setSessionId(id);
        }
      })
      .catch((e) => {
        if (
          cancelled ||
          transportRef.current !== transport ||
          bootstrapGeneration.current !== generation
        ) {
          return;
        }
        console.error("[parlant] failed to start session", e);
        setError(e as Error);
      });
    return () => {
      cancelled = true;
    };
  }, [transport, autoStart]);

  // Create the session on demand (no-op if one already exists / is stored).
  const start = useCallback(async (): Promise<string> => {
    const generation = ++bootstrapGeneration.current;
    const id = await transport.ensureSession();
    if (transportRef.current === transport && bootstrapGeneration.current === generation) {
      setSessionId(id);
    }
    return id;
  }, [transport]);

  const startFresh = (recoveryNotice: string | null) => {
    const generation = ++bootstrapGeneration.current;
    const currentTransport = transport;
    transport.resetSession();
    setSessionId(null);
    setError(null);
    setNotice(recoveryNotice);
    transport
      .ensureSession()
      .then((id) => {
        if (
          transportRef.current === currentTransport &&
          bootstrapGeneration.current === generation
        ) {
          setSessionId(id);
        }
      })
      .catch((e) => {
        if (
          transportRef.current !== currentTransport ||
          bootstrapGeneration.current !== generation
        ) {
          return;
        }
        console.error("[parlant] failed to start session", e);
        setError(e as Error);
      });
  };

  const reset = () => startFresh(null);
  const recover = (reason: "not-found" | "guest-expired") =>
    startFresh(
      reason === "guest-expired"
        ? "Your session expired, so a new conversation was started."
        : "The previous conversation was no longer available, so a new one was started.",
    );

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
