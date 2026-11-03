// The scrollable transcript: renders messages (with their turn's tool widgets and, in
// collapsible mode, reasoning), the status line, and a "new messages" pill when the
// user has scrolled up.

import type { ReactNode } from "react";

import type {
  ChatMessage,
  ReasoningMode,
  ReasoningSummary,
  StreamSpeed,
  StreamStyle,
  ToolWidget,
} from "../lib/events";
import type { PendingSignIn, Phase, SignedInMarker } from "../hooks/use-parlant-stream";
import { useStickToBottom } from "../hooks/use-stick-to-bottom";
import { useLinkTagger } from "./link-tagger";
import { MessageBubble } from "./message-bubble";
import { StatusBar } from "./status-bar";
import { ToolWidgetView, type WidgetContext } from "./tool-widget";

export interface MessageListProps {
  messages: ChatMessage[];
  toolWidgets: ToolWidget[];
  phase: Phase;
  /** The current turn failed (session "error" status) — shown inline at the end. */
  errored: boolean;
  /** The text of that inline failure notice. */
  errorMessage: string;
  reasoning: string | null;
  reasoningEventId: string | null;
  reasoningStreaming: boolean;
  reasoningByTrace: Map<string, ReasoningSummary>;
  reasoningMode: ReasoningMode;
  /** Shown first in the transcript, scrolling away with it, such as the agent's video. */
  lead?: ReactNode;
  /** Muted intro shown above the first message. */
  preface?: string;
  streamStyle: StreamStyle;
  streamSpeed: StreamSpeed;
  /** Delay (ms) before the "thinking" dots fade in; see ChatWidget's `thinkingDelay`. */
  thinkingDelay: number;
  /** Called once a live reply's reveal has finished, so it won't animate again. */
  onRevealed?: (messageId: string) => void;
  /** Sends a message as the visitor — what interactive tool widgets (e.g. "choices") call. */
  onSend?: (text: string) => void;
  /** The agent's open request to sign in, offered under its finished reply when `onSignIn`
   *  is set (the site can sign customers in). */
  signInRequest?: PendingSignIn | null;
  onSignIn?: () => void;
  signingIn?: boolean;
  /** Where customers signed in, shown as lines in the transcript. */
  signIns?: SignedInMarker[];
}

/** The agent reply the Sign in button belongs under: the latest one after the request, once
 *  it has finished — the customer reads why before being asked to act. */
function signInAnchor(messages: ChatMessage[], request: PendingSignIn): string | null {
  const from = request.afterMessageId
    ? messages.findIndex((m) => m.id === request.afterMessageId) + 1
    : 0;
  for (let i = messages.length - 1; i >= from; i--) {
    const m = messages[i];
    if (m.role === "agent") return m.streaming ? null : m.id;
  }
  return null;
}

function SignInButton({ onClick, busy }: { onClick: () => void; busy: boolean }) {
  return (
    <div className="parlant-msg-signin">
      <button type="button" onClick={onClick} disabled={busy} className="parlant-msg-signin-button">
        <svg
          width="15"
          height="15"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden
        >
          <path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4" />
          <polyline points="10 17 15 12 10 7" />
          <line x1="15" x2="3" y1="12" y2="12" />
        </svg>
        <span>Sign in</span>
      </button>
    </div>
  );
}

function SignedInLine({ name }: { name: string }) {
  return (
    <div className="parlant-signed-in" role="status">
      <span className="parlant-signed-in-rule" />
      <span className="parlant-signed-in-label">
        <svg
          width="13"
          height="13"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden
        >
          <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
          <circle cx="9" cy="7" r="4" />
          <polyline points="16 11 18 13 22 9" />
        </svg>
        {name ? `Signed in as ${name}` : "Signed in"}
      </span>
      <span className="parlant-signed-in-rule" />
    </div>
  );
}

const noop = () => {};

/** The first customer message after a widget's turn: one placed after the agent reply the
 *  widget renders with (`anchor`), or — for a turn with no reply yet — one created after
 *  the tool event (server time; optimistic messages are covered by the widget's own state). */
function nextCustomerMessage(
  messages: ChatMessage[],
  anchor: number | null,
  widgets: ToolWidget[],
): ChatMessage | undefined {
  if (anchor !== null) return messages.find((m, i) => i > anchor && m.role === "customer");
  const since = Math.min(
    ...widgets.map((w) => (w.createdAt ? Date.parse(w.createdAt) : NaN)).filter((t) => !isNaN(t)),
  );
  if (!isFinite(since)) return undefined;
  return messages.find((m) => {
    if (m.role !== "customer" || m.pending) return false;
    const t = Date.parse(m.createdAt);
    return !isNaN(t) && t > since;
  });
}

export function MessageList({
  messages,
  toolWidgets,
  phase,
  errored,
  errorMessage,
  reasoning,
  reasoningEventId,
  reasoningStreaming,
  reasoningByTrace,
  reasoningMode,
  lead,
  preface,
  streamStyle,
  streamSpeed,
  thinkingDelay,
  onRevealed,
  onSend,
  signInRequest,
  onSignIn,
  signingIn = false,
  signIns = [],
}: MessageListProps) {
  const { scrollRef, contentRef, isPinned, scrollToBottom } = useStickToBottom();

  // Group tool widgets by trace so each turn's output renders under its agent reply.
  const widgetsByTrace = new Map<string, ToolWidget[]>();
  for (const w of toolWidgets) {
    const arr = widgetsByTrace.get(w.traceId) ?? [];
    arr.push(w);
    widgetsByTrace.set(w.traceId, arr);
  }
  const attached = new Set<string>();
  const sendMessage = onSend ?? noop;
  const appendUtm = useLinkTagger();
  const contextFor = (anchor: number | null, widgets: ToolWidget[]): WidgetContext => {
    const next = nextCustomerMessage(messages, anchor, widgets);
    return { sendMessage, isLatest: !next, nextCustomerMessage: next?.text, appendUtm };
  };

  const signInUnder = signInRequest && onSignIn ? signInAnchor(messages, signInRequest) : null;
  const signInsAfter = (messageId: string | null) =>
    signIns
      .filter((marker) => marker.afterMessageId === messageId)
      .map((marker) => <SignedInLine key={marker.id} name={marker.name} />);

  return (
    <div className="parlant-list">
      <div ref={scrollRef} className="parlant-hide-scrollbar parlant-mlist-scroll">
        <div ref={contentRef} className="parlant-mlist-content">
          {lead}
          {preface && (
            <p className="parlant-mlist-preface">
              {preface}
            </p>
          )}

          {signInsAfter(null)}

          {messages.map((m, i) => {
            let widgets: ToolWidget[] | undefined;
            let reasoningSummary: ReasoningSummary | undefined;
            if (m.role === "agent" && m.traceId && !attached.has(m.traceId)) {
              widgets = widgetsByTrace.get(m.traceId);
              reasoningSummary = reasoningByTrace.get(m.traceId);
              if (widgets || reasoningSummary) attached.add(m.traceId);
            }
            return [
              <MessageBubble
                key={m.id}
                message={m}
                widgets={widgets}
                widgetContext={widgets ? contextFor(i, widgets) : undefined}
                reasoning={reasoningSummary}
                reasoningMode={reasoningMode}
                streamStyle={streamStyle}
                streamSpeed={streamSpeed}
                onRevealed={onRevealed}
                footer={
                  signInUnder === m.id && onSignIn ? (
                    <SignInButton onClick={onSignIn} busy={signingIn} />
                  ) : undefined
                }
              />,
              ...signInsAfter(m.id),
            ];
          })}

          {/* Tool widgets whose turn ended with no agent message. While the turn is open, its
              reply may still be coming, and they wait for it. */}
          {[...widgetsByTrace.entries()]
            .filter(([trace]) => !attached.has(trace) && phase === "idle")
            .flatMap(([, ws]) => {
              const context = contextFor(null, ws);
              return ws.map((w) => (
                <div key={w.toolCallId} className="parlant-mlist-widget">
                  <ToolWidgetView widget={w} context={context} />
                </div>
              ));
            })}

          {/* A failed turn, inline where the reply would have been — not in the banner. */}
          {errored && (
            <div className="parlant-turn-error">
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden
                className="parlant-mlist-error-icon"
              >
                <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z" />
                <line x1="12" y1="9" x2="12" y2="13" />
                <line x1="12" y1="17" x2="12.01" y2="17" />
              </svg>
              <span>{errorMessage}</span>
            </div>
          )}

          <StatusBar
            phase={phase}
            reasoning={reasoning}
            reasoningEventId={reasoningEventId}
            reasoningStreaming={reasoningStreaming}
            streamStyle={streamStyle}
            streamSpeed={streamSpeed}
            thinkingDelay={thinkingDelay}
          />
        </div>
      </div>

      {!isPinned && (
        <button
          type="button"
          onClick={() => scrollToBottom("smooth")}
          className="parlant-jump"
        >
          ↓ New messages
        </button>
      )}
    </div>
  );
}
