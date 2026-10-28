// The line below the messages while the agent is thinking: a label with a light sweeping
// through it (see .parlant-status::after). The label is the agent's narration when reasoning is
// enabled and arriving ("Thinking", a tool's own narration), streamed in live with the
// widget's streamStyle/streamSpeed, multi-line, scroll-pinned to its latest text; otherwise
// a generic "Thinking…". Which texts arrive is the server's call: tool narrations and stage
// labels always do, the model's own reasoning only for callers allowed to read it.
// Connectivity/errors live in the header status bar.

import { useEffect, useRef, useState } from "react";

import type { StreamStyle, StreamSpeed } from "../lib/events";
import type { Phase } from "../hooks/use-parlant-stream";
import { StreamingText } from "./streaming-text";

export interface StatusBarProps {
  phase: Phase;
  reasoning: string | null;
  /** Identifies the current reasoning event, so a new round replays from empty. */
  reasoningEventId: string | null;
  reasoningStreaming: boolean;
  streamStyle: StreamStyle;
  streamSpeed: StreamSpeed;
  /** Wait this many ms in the "thinking" phase before fading the line in, so a fast reply
   *  never flashes the dots. 0 shows it immediately. */
  thinkingDelay: number;
  /** Shown while thinking when no narration has arrived (default "Thinking…"). */
  thinkingLabel?: string;
}

export function StatusBar({
  phase,
  reasoning,
  reasoningEventId,
  reasoningStreaming,
  streamStyle,
  streamSpeed,
  thinkingDelay,
  thinkingLabel = "Thinking…",
}: StatusBarProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  const [overflowing, setOverflowing] = useState(false);

  // Follow the reasoning as it grows: keep its box pinned to the bottom so the newest text
  // stays visible instead of scrolling off. When it overflows, fade the top edge (see CSS)
  // so older lines dissolve into the background instead of getting hard-clipped.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    setOverflowing(el.scrollHeight > el.clientHeight + 6);
  }, [reasoning]);

  // Hold the line back for `thinkingDelay` ms after entering "thinking": if the reply lands
  // first the phase leaves "thinking" and the timer is cleared, so the dots never flash.
  useEffect(() => {
    if (phase !== "thinking") {
      setVisible(false);
      return;
    }
    if (thinkingDelay <= 0) {
      setVisible(true);
      return;
    }
    const t = setTimeout(() => setVisible(true), thinkingDelay);
    return () => clearTimeout(t);
  }, [phase, thinkingDelay]);

  // Only the "thinking" phase gets a line, and only once the delay has elapsed. While the
  // reply streams the text itself is the indicator, so there's no redundant label.
  if (phase !== "thinking" || !visible) return null;

  return (
    <div
      className="parlant-status"
      style={{ animation: "parlant-fade 200ms ease both" }}
    >
      <div
        ref={scrollRef}
        className={`parlant-reasoning parlant-hide-scrollbar${overflowing ? " parlant-reasoning--faded" : ""}`}
      >
        {reasoning ? (
          // Keyed by the reasoning event so each new round animates from empty.
          <StreamingText
            key={reasoningEventId ?? "reasoning"}
            text={reasoning}
            streaming={reasoningStreaming}
            fresh
            style={streamStyle}
            speed={streamSpeed}
          />
        ) : (
          thinkingLabel
        )}
      </div>
    </div>
  );
}
