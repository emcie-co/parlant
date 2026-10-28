// Subscribes to a session's event feed and exposes the live conversation:
// ordered messages (with token-by-token streaming), status, optional reasoning,
// tool-output widgets, plus `send` and `stop`.

import { useCallback, useEffect, useReducer, useRef, useState } from "react";

import {
  type AttachmentRef,
  type ChatMessage,
  type ParlantEvent,
  type ReasoningSummary,
  isInternalReasoning,
  type SessionMode,
  type SessionStatus,
  type ToolWidget,
  isChunksTerminated,
  isCompactionSummary,
  messageText,
} from "../lib/events";
import {
  GuestIdentityExpiredError,
  ParlantTransport,
  SessionForbiddenError,
  SessionNotFoundError,
  UnauthorizedError,
} from "../lib/transport";

export interface UseParlantStreamOptions {
  /** Surface the agent's narration as the thinking line (on by default): tool narrations, the
   *  engine's stage labels and, when the server lets this caller read it, the model's own
   *  reasoning. A caller without that permission sees "Thinking" in its place. */
  showReasoning?: boolean;
  /** Called when the session no longer exists server-side (start a fresh one). */
  onSessionExpired?: (reason: "not-found" | "guest-expired") => void;
  /** Current server reachability (from the health check). When it recovers to true, a stale
   *  stream error is dropped so the "reconnecting" banner clears. */
  online?: boolean;
  /** Called once for each label a turn adds to the session (see `ChatWidgetProps.onLabel`).
   *  Labels the session already has when it is first read are recorded without a call. */
  onLabel?: (label: string, context: { sessionId: string; labels: string[] }) => void;
}

/** What the UI should show: idle, "thinking" (working before the reply), or "typing" (reply streaming). */
export type Phase = "idle" | "thinking" | "typing";

export interface UseParlantStream {
  messages: ChatMessage[];
  /** True once the session's existing history has loaded. */
  hydrated: boolean;
  toolWidgets: ToolWidget[];
  status: SessionStatus;
  /** Whether a human operator has taken over ("manual") or the agent replies ("auto"). */
  sessionMode: SessionMode;
  /** Derived, UI-facing turn phase (robust to a missing/late `ready` status). */
  phase: Phase;
  /** Live reasoning for the turn in flight (drives the thinking line). */
  reasoning: string | null;
  reasoningEventId: string | null;
  reasoningStreaming: boolean;
  /** Finalized reasoning per turn (keyed by trace) — for the collapsible disclosure. */
  reasoningByTrace: Map<string, ReasoningSummary>;
  error: Error | null;
  send: (text: string, attachments?: AttachmentRef[]) => Promise<void>;
  stop: () => Promise<void>;
  /** Record that a live message has been fully revealed, so it renders at once from now on. */
  markRevealed: (messageId: string) => void;
}

interface Store {
  order: string[];
  messages: Map<string, ChatMessage>;
  toolWidgets: ToolWidget[];
  toolWidgetIds: Set<string>;
  status: SessionStatus;
  /** Session mode; seeded from a GET and refreshed on turn boundaries + a live flip event. */
  sessionMode: SessionMode;
  /** Live reasoning for the turn in flight (what the thinking line reveals). */
  reasoning: string | null;
  reasoningEventId: string | null;
  reasoningStreaming: boolean;
  /** Finalized reasoning per turn (keyed by trace) — backs the collapsible disclosure. */
  reasoningByTrace: Map<string, ReasoningSummary>;
  /** trace -> timestamp of the first reasoning chunk, for the "Thought for Xs" duration. */
  reasoningStart: Map<string, number>;
  nextOffset: number;
  /** Ids of optimistic customer messages awaiting their server echo, oldest first. */
  pendingQueue: string[];
  /** True once the initial backlog has loaded — messages after this are "fresh" (animate). */
  hydrated: boolean;
}

function freshStore(): Store {
  return {
    order: [],
    messages: new Map(),
    toolWidgets: [],
    toolWidgetIds: new Set(),
    status: "idle",
    sessionMode: "auto",
    reasoning: null,
    reasoningEventId: null,
    reasoningStreaming: false,
    reasoningByTrace: new Map(),
    reasoningStart: new Map(),
    nextOffset: 0,
    pendingQueue: [],
    hydrated: false,
  };
}

/** The narration text a status event currently carries: cumulative `message`, or joined chunks. */
function reasoningText(data: any): string {
  if (typeof data?.message === "string") return data.message;
  if (Array.isArray(data?.chunks)) return data.chunks.filter((c: unknown) => typeof c === "string").join("");
  return "";
}

/** Apply a status snapshot (from the list feed or a per-event follow) to the store: it becomes
 *  the live thinking line, and its step in the turn's log. `internal` marks the model's own
 *  reasoning (see `ReasoningStep.internal`). */
function applyReasoning(
  store: Store,
  traceId: string,
  eventId: string,
  data: any,
  internal: boolean,
): void {
  const text = reasoningText(data);
  if (!text) return;
  if (!store.reasoningStart.has(traceId)) store.reasoningStart.set(traceId, Date.now());
  store.reasoning = text;
  store.reasoningEventId = eventId;
  const hasChunks = Array.isArray(data?.chunks);
  store.reasoningStreaming = hasChunks ? !isChunksTerminated(data) : false;
  const start = store.reasoningStart.get(traceId)!;

  // A streamed event grows its own step in place; a new event appends one.
  const steps = [...(store.reasoningByTrace.get(traceId)?.steps ?? [])];
  const index = steps.findIndex((s) => s.eventId === eventId);
  const step = { eventId, text, internal };
  if (index >= 0) steps[index] = step;
  else steps.push(step);
  store.reasoningByTrace.set(traceId, { steps, durationMs: Date.now() - start });
}

function clearLiveReasoning(store: Store): void {
  store.reasoning = null;
  store.reasoningEventId = null;
  store.reasoningStreaming = false;
}

function applyEvent(store: Store, event: ParlantEvent, showReasoning: boolean): void {
  switch (event.kind) {
    case "message": {
      if (isCompactionSummary(event)) return;
      const role: ChatMessage["role"] | null =
        event.source === "customer"
          ? "customer"
          : event.source === "ai_agent" || event.source.startsWith("human_agent")
            ? "agent"
            : null;
      if (!role) return; // skip system and other non-conversational sources

      // Reconcile an optimistic customer bubble with its real echo (FIFO): drop the
      // muted placeholder so the confirmed message replaces it.
      if (role === "customer" && store.pendingQueue.length) {
        const pendingId = store.pendingQueue.shift()!;
        const idx = store.order.indexOf(pendingId);
        if (idx >= 0) store.order.splice(idx, 1);
        store.messages.delete(pendingId);
      }

      const streaming =
        event.source === "ai_agent" &&
        Array.isArray(event.data?.chunks) &&
        !isChunksTerminated(event.data);

      const existing = store.messages.get(event.id);
      if (!existing) store.order.push(event.id);
      store.messages.set(event.id, {
        id: event.id,
        role,
        text: messageText(event.data),
        streaming: existing ? existing.streaming || streaming : streaming,
        attachments: (event.data?.attachments as AttachmentRef[]) ?? [],
        traceId: event.trace_id,
        createdAt: event.creation_utc,
        // Fresh = really live: it arrived after the history loaded, or it is a reply the
        // agent is still writing. Sticky once set, so a live message that lands whole
        // still animates; finished history, including the last message, stays instant.
        fresh: existing ? existing.fresh : store.hydrated || streaming,
      });
      if (role === "agent") clearLiveReasoning(store); // a reply starting clears the live line
      break;
    }
    case "status": {
      const status = event.data?.status as string | undefined;
      if (status === "processing" || status === "acknowledged") store.status = "processing";
      else if (status === "typing") store.status = "typing";
      else if (status === "ready" || status === "cancelled") {
        store.status = "ready";
        clearLiveReasoning(store);
      } else if (status === "error") {
        store.status = "error";
        clearLiveReasoning(store);
      }
      // The engine tags a status event with `session_mode` when a turn flips to manual
      // (a human took over). Trust it only once live (post-hydration), so replaying an old
      // backlog event can't leave a stale "manual" — the seed/refresh GET is authoritative.
      const mode = event.data?.session_mode;
      if ((mode === "manual" || mode === "auto") && store.hydrated) store.sessionMode = mode;
      if (showReasoning)
        applyReasoning(
          store,
          event.trace_id,
          event.id,
          event.data,
          isInternalReasoning(event.metadata),
        );
      break;
    }
    case "tool": {
      const calls = (event.data?.tool_calls as any[]) ?? [];
      calls.forEach((call, i) => {
        const metadata = call?.result?.metadata;
        const id = `${event.id}:${i}`;
        if (metadata && Object.keys(metadata).length && !store.toolWidgetIds.has(id)) {
          store.toolWidgetIds.add(id);
          store.toolWidgets.push({
            toolCallId: id,
            metadata,
            traceId: event.trace_id,
            createdAt: event.creation_utc,
          });
        }
      });
      break;
    }
    // "custom" -> app-defined; ignored here
  }
}

const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

const RECONNECT_MIN_MS = 1000;
const RECONNECT_MAX_MS = 30_000;

export function useParlantStream(
  transport: ParlantTransport,
  sessionId: string | null,
  options: UseParlantStreamOptions = {},
): UseParlantStream {
  const showReasoning = options.showReasoning ?? true;
  const online = options.online ?? true;

  // Kept in a ref so the stream effect doesn't re-subscribe when the callback identity changes.
  const onExpired = useRef(options.onSessionExpired);
  onExpired.current = options.onSessionExpired;
  const onLabel = useRef(options.onLabel);
  onLabel.current = options.onLabel;

  const storeRef = useRef<Store>(freshStore());
  const pendingCounter = useRef(0);
  // Set by the subscribe effect; lets `send` (defined outside it) re-read the mode after a
  // message, so a session that's already manual is detected the moment the customer writes.
  const refreshModeRef = useRef<(() => void) | null>(null);
  const [, render] = useReducer((n: number) => n + 1, 0);
  const [error, setError] = useState<Error | null>(null);

  // When the server becomes reachable again (health recovered), drop any stale stream error so
  // the "reconnecting" banner clears — the consume loop below re-flags it if the stream is
  // genuinely still broken. Fixes a banner that lingered after a server restart even though
  // the connection had come back.
  useEffect(() => {
    if (online) setError((prev) => (prev ? null : prev));
  }, [online]);

  const commit = useCallback((mutate: (store: Store) => void) => {
    mutate(storeRef.current);
    render();
  }, []);

  // No re-render: the message is already on screen, and the flag only matters the next
  // time it mounts.
  const markRevealed = useCallback((messageId: string) => {
    const message = storeRef.current.messages.get(messageId);
    if (message?.fresh && !message.streaming) message.fresh = false;
  }, []);

  useEffect(() => {
    if (!sessionId) return;

    storeRef.current = freshStore();
    render();
    setError(null);

    const controller = new AbortController();
    const followed = new Set<string>();

    // Labels the session is known to have. The first successful read seeds it silently, so
    // reopening a conversation doesn't report labels it already had; every later read reports
    // the labels it adds. Removals are dropped from the set but never reported.
    let knownLabels: Set<string> | null = null;
    const reportLabels = (labels: string[]) => {
      if (knownLabels === null) {
        knownLabels = new Set(labels);
        return;
      }
      const added = labels.filter((label) => !knownLabels!.has(label));
      knownLabels = new Set(labels);
      for (const label of added) {
        try {
          onLabel.current?.(label, { sessionId, labels: [...labels] });
        } catch (e) {
          // The host page's callback must never break the conversation.
          console.error("parlant-chat: onLabel threw", e);
        }
      }
    };

    // Reads can overlap (a send and a turn's end close together); apply only the newest, so a
    // late, older answer can't make a label look removed and then added again.
    let reads = 0;
    let applied = 0;

    // Authoritatively read the session's mode and labels. The feed only signals a mid-turn
    // flip to manual: never a resumed-already-manual session, a revert to auto, or a label.
    // Seeded now and refreshed on each turn boundary and after a send.
    const refreshMode = () => {
      const read = ++reads;
      transport
        .fetchSession(sessionId, controller.signal)
        .then((s) => {
          if (controller.signal.aborted || read < applied) return;
          applied = read;
          commit((store) => void (store.sessionMode = s.mode));
          reportLabels(s.labels);
        })
        .catch(() => {
          /* transient — keep the last known mode and labels */
        });
    };
    refreshModeRef.current = refreshMode;
    refreshMode();

    // Mark the session "hydrated" once the initial backlog goes quiet (no events for a
    // beat). Messages that arrive after this are "fresh" and animate their reveal, even
    // if they land whole; the backlog loaded on resume stays instant.
    let hydrateTimer: ReturnType<typeof setTimeout> | undefined;
    const markHydratedSoon = () => {
      if (storeRef.current.hydrated) return;
      clearTimeout(hydrateTimer);
      hydrateTimer = setTimeout(() => commit((store) => void (store.hydrated = true)), 300);
    };
    // Note: the clock is started from the stream's onOpen (below), once the connection is
    // actually established — not here — so connect latency doesn't pre-hydrate the store.

    // Follow one agent message's chunks token-by-token until it terminates.
    const followMessage = async (eventId: string) => {
      if (followed.has(eventId)) return;
      followed.add(eventId);
      try {
        for await (const snap of transport.streamEvent(sessionId, eventId, controller.signal)) {
          commit((store) => {
            const msg = store.messages.get(eventId);
            if (msg) {
              msg.text = messageText(snap.data);
              msg.streaming = Array.isArray(snap.data?.chunks) && !isChunksTerminated(snap.data);
            }
          });
          if (isChunksTerminated(snap.data)) break;
        }
      } catch (error) {
        if (error instanceof GuestIdentityExpiredError) {
          controller.abort();
          onExpired.current?.("guest-expired");
          return;
        }
        if (error instanceof UnauthorizedError) {
          controller.abort();
          setError(error);
          return;
        }
        // per-event stream errors are non-fatal; the list stream remains authoritative
      } finally {
        commit((store) => {
          const msg = store.messages.get(eventId);
          if (msg) msg.streaming = false;
        });
      }
    };

    // Follow one reasoning status event's chunks as they stream, exactly like a message.
    const followStatus = async (eventId: string, traceId: string) => {
      if (followed.has(eventId)) return;
      followed.add(eventId);
      try {
        for await (const snap of transport.streamEvent(sessionId, eventId, controller.signal)) {
          commit((store) =>
            applyReasoning(store, traceId, eventId, snap.data, isInternalReasoning(snap.metadata)),
          );
          if (isChunksTerminated(snap.data)) break;
        }
      } catch (error) {
        if (error instanceof GuestIdentityExpiredError) {
          controller.abort();
          onExpired.current?.("guest-expired");
          return;
        }
        if (error instanceof UnauthorizedError) {
          controller.abort();
          setError(error);
          return;
        }
        // per-event stream errors are non-fatal; the list stream remains authoritative
      } finally {
        commit((store) => {
          if (store.reasoningEventId === eventId) store.reasoningStreaming = false;
        });
      }
    };

    // Load the conversation so far in one read, as history, before following it live. The
    // boundary is the events that already exist, not timing, so a slow connection can't
    // make finished messages animate. A reply still being written is followed (and
    // animates) from here. If the read fails or stalls, the stream below loads everything
    // and the quiet-timer marks the history instead.
    const loadHistory = async () => {
      const read = new AbortController();
      const abortRead = () => read.abort();
      controller.signal.addEventListener("abort", abortRead, { once: true });
      const timer = setTimeout(abortRead, 5000);
      try {
        const history = await transport.listEvents(sessionId, storeRef.current.nextOffset, read.signal);
        commit((store) => {
          for (const event of history) {
            applyEvent(store, event, showReasoning);
            if (event.offset != null) store.nextOffset = Math.max(store.nextOffset, event.offset + 1);
          }
          store.hydrated = true;
        });
        for (const event of history) {
          if (event.kind === "message" && event.source === "ai_agent" && !isChunksTerminated(event.data)) {
            void followMessage(event.id);
          }
        }
      } catch {
        /* fall back to loading the history from the stream */
      } finally {
        clearTimeout(timer);
        controller.signal.removeEventListener("abort", abortRead);
      }
    };

    // Consume the session's event feed, reconnecting on timeout/transient failure. Failures
    // back off (1s, doubling to 30s) so an outage isn't met with a request every second.
    const consume = async () => {
      await loadHistory();
      let backoff = RECONNECT_MIN_MS;
      while (!controller.signal.aborted) {
        try {
          for await (const event of transport.streamEvents(
            sessionId,
            storeRef.current.nextOffset,
            controller.signal,
            () => {
              // Connection opened (even if idle) — clear any stale "interrupted" banner, and
              // start the hydration clock HERE (not at subscribe) so slow connect latency
              // can't hydrate the store before the backlog arrives (which would make resumed
              // messages animate as if live).
              setError((prev) => (prev ? null : prev));
              backoff = RECONNECT_MIN_MS;
              markHydratedSoon();
            },
          )) {
            // A delivered event means the stream is healthy again — clear any stale error.
            setError((prev) => (prev ? null : prev));
            commit((store) => {
              applyEvent(store, event, showReasoning);
              if (event.offset != null) {
                store.nextOffset = Math.max(store.nextOffset, event.offset + 1);
              }
            });
            markHydratedSoon(); // reset the quiet timer as long as the backlog is flowing
            // A completed turn is when a dashboard-side mode change is most likely to matter;
            // re-read it (cheap) so a hand-off that produced no `session_mode` tag is caught.
            if (
              event.kind === "status" &&
              (event.data?.status === "ready" || event.data?.status === "error") &&
              storeRef.current.hydrated
            ) {
              refreshMode();
            }
            if (
              event.kind === "message" &&
              event.source === "ai_agent" &&
              !isChunksTerminated(event.data)
            ) {
              void followMessage(event.id);
            }
            if (
              event.kind === "status" &&
              showReasoning &&
              Array.isArray(event.data?.chunks) &&
              !isChunksTerminated(event.data)
            ) {
              void followStatus(event.id, event.trace_id);
            }
          }
          // Long-poll returned (with or without data) — the connection is healthy, so clear
          // any stale "connection interrupted" error even on an idle reconnect.
          setError((prev) => (prev ? null : prev));
          // Loop and reconnect from nextOffset.
        } catch (e) {
          if (controller.signal.aborted) return;
          if (e instanceof GuestIdentityExpiredError) {
            onExpired.current?.("guest-expired");
            return;
          }
          if (e instanceof UnauthorizedError) {
            setError(e);
            return;
          }
          if (e instanceof SessionNotFoundError || e instanceof SessionForbiddenError) {
            // The stored session is gone (server restarted / expired), or belongs to a guest
            // identity this browser no longer holds. Retrying can't help: start fresh — the
            // session hook will bootstrap a new one and this effect re-subscribes.
            onExpired.current?.("not-found");
            return;
          }
          console.error("[parlant] event stream error", e);
          setError(e as Error);
          await delay(backoff);
          backoff = Math.min(backoff * 2, RECONNECT_MAX_MS);
        }
      }
    };

    void consume();
    return () => {
      controller.abort();
      clearTimeout(hydrateTimer);
      refreshModeRef.current = null;
    };
  }, [transport, sessionId, showReasoning, commit]);

  const send = useCallback(
    async (text: string, attachments?: AttachmentRef[]) => {
      const trimmed = text.trim();
      // Allow an attachment-only message (empty text), which the composer permits.
      if (!sessionId || (!trimmed && !attachments?.length)) return;
      setError(null);

      // Optimistic: show the customer's message immediately (muted) and reconcile
      // it to the confirmed style when the feed echoes it back.
      const pendingId = `pending-${pendingCounter.current++}`;
      commit((store) => {
        store.order.push(pendingId);
        store.messages.set(pendingId, {
          id: pendingId,
          role: "customer",
          text: trimmed,
          streaming: false,
          attachments: attachments ?? [],
          traceId: "",
          createdAt: new Date().toISOString(),
          pending: true,
        });
        store.pendingQueue.push(pendingId);
      });

      try {
        await transport.sendMessage(sessionId, trimmed, attachments);
        // A session that's already manual answers a customer message with silence (the engine
        // short-circuits before emitting anything), so re-read the mode to surface the notice.
        refreshModeRef.current?.();
      } catch (e) {
        if (e instanceof GuestIdentityExpiredError) {
          onExpired.current?.("guest-expired");
          return;
        }
        if (e instanceof SessionForbiddenError) {
          onExpired.current?.("not-found");
          return;
        }
        console.error("[parlant] failed to send message", e);
        setError(e as Error);
        commit((store) => {
          const msg = store.messages.get(pendingId);
          if (msg) msg.failed = true;
          store.pendingQueue = store.pendingQueue.filter((id) => id !== pendingId);
        });
      }
    },
    [transport, sessionId, commit],
  );

  const stop = useCallback(async () => {
    if (sessionId) await transport.cancel(sessionId);
  }, [transport, sessionId]);

  const store = storeRef.current;
  const messages = store.order
    .map((id) => store.messages.get(id))
    .filter((m): m is ChatMessage => Boolean(m));

  // Derive the UI phase from status *and* the reply's streaming state:
  //  - "thinking": the agent is working but no reply text is visible yet.
  //  - "typing":   reply text is streaming (the text + caret is the indicator, so the
  //                status line hides; this state exists mainly to keep `busy` true).
  //  - "idle":     nothing in flight / the reply is done (robust to a late `ready`).
  const last = messages[messages.length - 1];
  const replyDone = last?.role === "agent" && !last.streaming && last.text.length > 0;
  const streamingText = last?.role === "agent" && last.streaming && last.text.length > 0;
  const working =
    ["acknowledged", "processing", "typing"].includes(store.status) ||
    (last?.role === "agent" && last.streaming);

  let phase: Phase = "idle";
  if (replyDone) phase = "idle";
  else if (streamingText) phase = "typing";
  else if (working) phase = "thinking";

  return {
    messages,
    hydrated: store.hydrated,
    toolWidgets: store.toolWidgets,
    status: store.status,
    sessionMode: store.sessionMode,
    phase,
    reasoning: store.reasoning,
    reasoningEventId: store.reasoningEventId,
    reasoningStreaming: store.reasoningStreaming,
    reasoningByTrace: store.reasoningByTrace,
    error,
    send,
    stop,
    markRevealed,
  };
}
