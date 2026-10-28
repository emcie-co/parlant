// Normalized event and message types for the Parlant chat widget.
//
// Parlant's `GET /sessions/{id}/events` feed returns one EventDTO per event. `data`
// is opaque passthrough JSON whose shape depends on `kind`. These types + helpers
// give the hooks a small, stable surface to work against.

export type EventKind = "message" | "status" | "tool" | "custom";

/**
 * How streamed agent text is revealed. See `StreamingText` in `message-bubble.tsx`
 * for what each looks like.
 */
export type StreamStyle = "instant" | "fade" | "rise" | "blur" | "letter" | "warm";

/** Reveal pace: characters per second, or a named preset. */
export type StreamSpeed = "slow" | "normal" | "fast" | number;

export type EventSource =
  | "customer"
  | "customer_ui"
  | "human_agent"
  | "human_agent_on_behalf_of_ai_agent"
  | "ai_agent"
  | "system";

/** A raw event as returned by the events feed. */
export interface ParlantEvent {
  id: string;
  source: EventSource;
  kind: EventKind;
  offset: number;
  creation_utc: string;
  trace_id: string;
  data: any;
  metadata?: Record<string, any>;
  deleted?: boolean;
}

/** A reference to a file that lives in your own storage (Parlant carries the reference, not bytes). */
export interface AttachmentRef {
  url: string;
  media_type?: string;
  caption?: string;
  filename?: string;
  metadata?: Record<string, unknown>;
}

/** A tool call as recorded on a `tool` event's data. */
export interface ToolCall {
  tool_id: string;
  arguments: Record<string, unknown>;
  rationale?: string;
  result?: { data?: unknown; metadata?: Record<string, unknown> };
}

/** A normalized chat message the UI renders. */
export interface ChatMessage {
  id: string;
  role: "customer" | "agent";
  text: string;
  streaming: boolean;
  attachments: AttachmentRef[];
  traceId: string;
  createdAt: string;
  /** A customer message shown optimistically, not yet echoed by the server (render it muted). */
  pending?: boolean;
  /** The optimistic send failed (render an error/retry affordance). */
  failed?: boolean;
  /** Arrived live (after initial hydration) — animate its reveal even if it lands whole.
   *  History loaded on resume is not fresh, so it appears at once. */
  fresh?: boolean;
}

export type SessionStatus = "idle" | "processing" | "typing" | "ready" | "error";

/** Who's driving the session: "auto" (the agent replies) or "manual" (a human operator has
 *  taken over — the engine stays silent for this session). */
export type SessionMode = "auto" | "manual";

/** A piece of tool output (from a tool result's frontend-only `metadata`), tied to a turn. */
export interface ToolWidget {
  toolCallId: string;
  metadata: Record<string, unknown>;
  traceId: string;
  /** When the tool event was created (server time), to tell which messages came after it. */
  createdAt?: string;
}

/** What happens to the reasoning shown while thinking, once the reply arrives. */
export type ReasoningMode = "transient" | "collapsible";

/** Finalized reasoning for one turn — the text plus how long the agent thought. */
/** One status text the agent showed while working on a turn, in order of arrival. */
export interface ReasoningStep {
  /** The status event this step came from; a streamed event updates its step in place. */
  eventId: string;
  text: string;
  /** The model's own reasoning (the server flags it `__parlant__.internal_reasoning`), shown
   *  as Markdown. Everything else — a tool's narration, "Running tools" — is a plain line. A
   *  caller the server doesn't let read reasoning never sees this flag: that step arrives as
   *  a bare "Thinking" label instead. */
  internal: boolean;
}

/** What the agent did during a turn, for the "Thought for Xs" disclosure. */
export interface ReasoningSummary {
  steps: ReasoningStep[];
  durationMs: number;
}

/** Whether a status event carries the model's own reasoning (see `ReasoningStep.internal`). */
export function isInternalReasoning(metadata: any): boolean {
  return metadata?.__parlant__?.internal_reasoning === true;
}

// --- helpers ---------------------------------------------------------------

/**
 * System summary messages emitted by compaction carry
 * `metadata.__parlant__.source === "compacter"`. The widget filters these so a
 * summary never renders as a chat bubble.
 */
export function isCompactionSummary(event: ParlantEvent): boolean {
  return (
    event.kind === "message" &&
    event.source === "system" &&
    event.metadata?.__parlant__?.source === "compacter"
  );
}

/** The text a message event currently shows: a block `message`, or the joined non-terminator chunks. */
export function messageText(data: any): string {
  const chunks = data?.chunks;
  if (Array.isArray(chunks)) return chunks.filter((c: unknown) => c != null).join("");
  return data?.message ?? "";
}

/** A streaming message is finished when its `chunks` list ends with a `null` terminator. */
export function isChunksTerminated(data: any): boolean {
  const chunks = data?.chunks;
  return Array.isArray(chunks) && chunks.length > 0 && chunks[chunks.length - 1] == null;
}
