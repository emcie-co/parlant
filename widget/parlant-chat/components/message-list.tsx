// The scrollable transcript: renders messages (with their turn's tool widgets and, in
// collapsible mode, reasoning), the status line, and a "new messages" pill when the
// user has scrolled up.

import type {
  ChatMessage,
  ReasoningMode,
  ReasoningSummary,
  StreamSpeed,
  StreamStyle,
  ToolWidget,
} from "../lib/events";
import type { Phase } from "../hooks/use-parlant-stream";
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
  preface,
  streamStyle,
  streamSpeed,
  thinkingDelay,
  onRevealed,
  onSend,
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

  return (
    <div className="parlant-list">
      <div ref={scrollRef} className="parlant-hide-scrollbar parlant-mlist-scroll">
        <div ref={contentRef} className="parlant-mlist-content">
          {preface && (
            <p className="parlant-mlist-preface">
              {preface}
            </p>
          )}

          {messages.map((m, i) => {
            let widgets: ToolWidget[] | undefined;
            let reasoningSummary: ReasoningSummary | undefined;
            if (m.role === "agent" && m.traceId && !attached.has(m.traceId)) {
              widgets = widgetsByTrace.get(m.traceId);
              reasoningSummary = reasoningByTrace.get(m.traceId);
              if (widgets || reasoningSummary) attached.add(m.traceId);
            }
            return (
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
              />
            );
          })}

          {/* Tool widgets whose turn has no agent message yet. */}
          {[...widgetsByTrace.entries()]
            .filter(([trace]) => !attached.has(trace))
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
