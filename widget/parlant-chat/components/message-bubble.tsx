// One chat message: streaming text (with a configurable reveal animation),
// optimistic/failed states, attachments, any tool-output widgets, and — in
// "collapsible" reasoning mode — a "Thought for Xs" disclosure for the turn.

import { useState } from "react";

import type {
  ChatMessage,
  ReasoningMode,
  ReasoningSummary,
  StreamStyle,
  StreamSpeed,
  ToolWidget,
} from "../lib/events";
import { cn } from "../lib/cn";
import { Attachment } from "./attachment";
import { StreamingText } from "./streaming-text";
import { ToolWidgetView, type WidgetContext } from "./tool-widget";

export interface MessageBubbleProps {
  message: ChatMessage;
  widgets?: ToolWidget[];
  /** Context for those widgets' renderers (send a message, whether the turn is still open). */
  widgetContext?: WidgetContext;
  /** Finalized reasoning for this turn — rendered as a disclosure in "collapsible" mode. */
  reasoning?: ReasoningSummary;
  reasoningMode?: ReasoningMode;
  streamStyle: StreamStyle;
  streamSpeed: StreamSpeed;
  /** Called once a live reply's reveal has finished, so it won't animate again. */
  onRevealed?: (messageId: string) => void;
}

export function MessageBubble({
  message,
  widgets,
  widgetContext,
  reasoning,
  reasoningMode,
  streamStyle,
  streamSpeed,
  onRevealed,
}: MessageBubbleProps) {
  const isCustomer = message.role === "customer";
  const showReasoning =
    message.role === "agent" &&
    reasoningMode === "collapsible" &&
    (reasoning?.steps.length ?? 0) > 0;

  return (
    <div className={cn("parlant-msg", isCustomer ? "parlant-msg--customer" : "parlant-msg--agent")}>
      {/* Tool-output widgets render before the reply text: the tool result event comes
          first in the turn, so its card should sit above the message it produced. */}
      {widgets && widgets.length > 0 && (
        <div className="parlant-msg-widgets">
          {widgets.map((w) => (
            <ToolWidgetView key={w.toolCallId} widget={w} context={widgetContext} />
          ))}
        </div>
      )}

      {/* Collapsible reasoning: a "Thought for Xs" toggle above the reply it produced. */}
      {showReasoning && reasoning && <ReasoningDisclosure summary={reasoning} />}

      {/* Skip the text bubble entirely for an attachment-only customer message (no empty
          bubble) — the attachment below stands on its own. Agent replies always render. */}
      {(!isCustomer || message.text.length > 0) && (
        <div
          className={cn(
            "parlant-msg-bubble",
            isCustomer ? "parlant-msg-bubble--customer" : "parlant-msg-bubble--agent",
            message.pending && "parlant-msg-bubble--pending",
            message.failed && "parlant-msg-bubble--failed",
          )}
        >
          {/* Only agent replies stream in; the customer's own text is shown at once. */}
          {isCustomer ? (
            message.text
          ) : (
            <StreamingText
              text={message.text}
              streaming={message.streaming}
              fresh={message.fresh ?? false}
              style={streamStyle}
              speed={streamSpeed}
              onRevealed={onRevealed && (() => onRevealed(message.id))}
            />
          )}
        </div>
      )}

      {message.attachments.length > 0 && (
        <div
          className={cn(
            "parlant-msg-attachments",
            isCustomer && "parlant-msg-attachments--customer",
            // Optimistic attachment: 50% until the server echoes the event carrying it.
            message.pending && "parlant-msg-attachments--pending",
          )}
        >
          {message.attachments.map((a, i) => (
            <Attachment key={i} attachment={a} />
          ))}
        </div>
      )}

      {message.failed && <span className="parlant-msg-failed">Failed to send</span>}
    </div>
  );
}

/** "Thought for Xs" — click to reveal what the agent did during this turn: each status it
 *  showed, in order. The model's own reasoning (when this caller may read it) renders as
 *  Markdown; a tool's narration or a stage label is a plain line. Adjacent repeats collapse,
 *  so a redacted reasoning step ("Thinking") doesn't echo the engine's own "Thinking" label. */
function ReasoningDisclosure({ summary }: { summary: ReasoningSummary }) {
  const [open, setOpen] = useState(false);
  const seconds = Math.max(1, Math.round(summary.durationMs / 1000));
  const steps = summary.steps.filter(
    (step, i, all) => i === 0 || step.text !== all[i - 1].text || step.internal !== all[i - 1].internal,
  );

  return (
    <div className="parlant-reasoning-disclosure parlant-reasoning-disclosure--group">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="parlant-msg-reasoning-toggle"
      >
        <svg
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden
          className={cn("parlant-msg-reasoning-chevron", open && "parlant-msg-reasoning-chevron--open")}
        >
          <path d="m9 18 6-6-6-6" />
        </svg>
        Thought for {seconds}s
      </button>
      {open && (
        <div className="parlant-msg-reasoning-panel">
          {steps.map((step) => (
            <div
              key={step.eventId}
              className={cn(
                "parlant-msg-reasoning-step",
                step.internal && "parlant-msg-reasoning-step--internal",
              )}
            >
              {step.internal ? (
                // Static Markdown (fresh=false → no reveal animation, STATIC_LEAF path).
                <StreamingText text={step.text} streaming={false} fresh={false} style="instant" speed="normal" />
              ) : (
                step.text
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
