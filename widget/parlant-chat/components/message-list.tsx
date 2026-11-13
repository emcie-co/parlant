// The scrollable transcript: renders messages (with their turn's tool widgets and, in
// collapsible mode, reasoning), the status line, and a "new messages" pill when the
// user has scrolled up.

import { type ReactNode, useEffect, useRef } from "react";
import { flushSync } from "react-dom";

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

/** How long a scroll takes to glide past a faded lead; matches the header's transition. */
const GLIDE_MS = 320;

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
  /** Called with true as soon as `lead` starts scrolling out of view, and false once it is
   *  back in its resting place (and on unmount), so the header can stand in for it. */
  onLeadScrolled?: (scrolled: boolean) => void;
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
  onLeadScrolled,
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
  const { scrollRef, contentRef, isPinned, scrollToBottom, holdRef } = useStickToBottom();
  const hasLead = Boolean(lead);

  const leadRef = useRef<HTMLDivElement>(null);

  // Reports whether the lead (such as the agent's video) has started to scroll away. It leads
  // the list, so it moves the moment the list is scrolled at all; measured on scroll, which
  // also fires when the list pins itself to a new message.
  //
  // The header stands in for the lead once it moves, so the lead fades and its space is empty.
  // A scroll that starts into that space glides on past it: down to just above the first
  // message (the content's gap stays as a margin), or up to the top. The glide is driven frame
  // by frame for GLIDE_MS, the same time the header takes to make room for the lead's stand-in
  // (see styles.css), so the two move together. Re-pinning to the bottom is held meanwhile.
  //
  // Docking makes the header taller in one step, which shortens this list from the top and
  // would push its content down (undocking, up). So the header's change is committed at once
  // (flushSync) and the scroll moves by the same amount in the same task, before anything is
  // painted: the content stays exactly where it was.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el || !hasLead || !onLeadScrolled) return;
    let scrolled = false;
    let last = el.scrollTop;
    let frame = 0;
    let gliding: { from: number; target: number; start: number } | null = null;
    const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
    const report = (next: boolean, keepInPlace = true) => {
      if (next === scrolled) return;
      scrolled = next;
      if (!keepInPlace) {
        onLeadScrolled(next);
        return;
      }
      const before = el.clientHeight;
      flushSync(() => onLeadScrolled(next));
      const shrunk = before - el.clientHeight;
      if (shrunk && el.scrollTop > 0) {
        el.scrollTop += shrunk;
        if (gliding) gliding.from += shrunk;
      }
    };
    const leadEnd = () => {
      const leadEl = leadRef.current;
      if (!leadEl) return 0;
      return el.scrollTop + leadEl.getBoundingClientRect().bottom - el.getBoundingClientRect().top;
    };
    const finish = () => {
      frame = 0;
      gliding = null;
      holdRef.current = false;
      last = el.scrollTop;
      el.dispatchEvent(new Event("scroll"));
    };
    const glide = (target: number) => {
      holdRef.current = true;
      report(target > 0);
      if (reduceMotion || typeof requestAnimationFrame === "undefined") {
        el.scrollTo({ top: target, behavior: "auto" });
        finish();
        return;
      }
      // The clock starts at the glide's first frame, so a slow first frame doesn't skip ahead.
      const glideState = { from: el.scrollTop, target, start: -1 };
      gliding = glideState;
      // Docking or undocking moved the scroll to keep the content in place; start from there.
      glideState.from = el.scrollTop;
      const step = (now: number) => {
        if (glideState.start < 0) glideState.start = now;
        const t = Math.min(1, (now - glideState.start) / GLIDE_MS);
        const eased = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
        const max = el.scrollHeight - el.clientHeight;
        el.scrollTop = Math.min(max, glideState.from + (glideState.target - glideState.from) * eased);
        if (t < 1) frame = requestAnimationFrame(step);
        else finish();
      };
      frame = requestAnimationFrame(step);
    };
    const onScroll = () => {
      if (frame) return;
      const top = el.scrollTop;
      const end = leadEnd();
      // At the bottom, this is the list pinning itself to a new message (or a short chat that
      // can't scroll further): that never glides.
      const atBottom = el.scrollHeight - el.clientHeight - top <= 1;
      if (top > 0 && top < end - 1 && top !== last && !atBottom) {
        glide(top > last ? end : 0);
        return;
      }
      report(top > 0);
      last = top;
    };
    // While gliding, the glide owns the scroll: a trackpad's momentum would otherwise nudge
    // the list between its frames, so it shakes. Momentum after the glide scrolls as usual.
    const holdInput = (event: Event) => {
      if (frame) event.preventDefault();
    };
    report(el.scrollTop > 0, false);
    el.addEventListener("scroll", onScroll, { passive: true });
    el.addEventListener("wheel", holdInput, { passive: false });
    el.addEventListener("touchmove", holdInput, { passive: false });
    return () => {
      el.removeEventListener("scroll", onScroll);
      el.removeEventListener("wheel", holdInput);
      el.removeEventListener("touchmove", holdInput);
      if (frame) cancelAnimationFrame(frame);
      holdRef.current = false;
      if (scrolled) onLeadScrolled(false);
    };
  }, [scrollRef, hasLead, onLeadScrolled, holdRef]);

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
          {lead && (
            <div ref={leadRef} className="parlant-mlist-lead">
              {lead}
            </div>
          )}
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
