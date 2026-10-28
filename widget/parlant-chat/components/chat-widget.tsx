// The batteries-included widget: a floating input dock by default, or an
// inline surface that fills its container. Composed from the hooks and primitives —
// copy this file and rearrange it however you like.

import { type CSSProperties, type ReactNode, useEffect, useMemo, useRef, useState } from "react";

import "../styles.css";
import type { AttachmentRef, ReasoningMode, StreamSpeed, StreamStyle } from "../lib/events";
import { GuestIdentityExpiredError, type ParlantAuth } from "../lib/transport";
import { cn } from "../lib/cn";
import { appendUtm, type UtmLinks, type UtmParams } from "../lib/links";
import { useParlantSession } from "../hooks/use-parlant-session";
import { useParlantStream } from "../hooks/use-parlant-stream";
import { useServerHealth } from "../hooks/use-server-health";
import { useSidebarSpace } from "../hooks/use-sidebar-space";
import { usePersistentSuggestions } from "../hooks/use-persistent-suggestions";
import { playAgentSound, playCustomerSound, type SoundTheme } from "../lib/sounds";
import { Composer } from "./composer";
import { Launcher } from "./launcher";
import { LinkTaggerContext } from "./link-tagger";
import { MessageList } from "./message-list";

export interface ChatWidgetProps {
  /** Parlant server base URL. */
  serverUrl: string;
  /** The agent to converse with. */
  agentId: string;
  /** Floating launcher, embedded in your layout, a floating input box docked in
   *  the corner ("input", default), or a message bar centered at the bottom of the screen ("bar").
   *  Input and bar are the same surface in two positions: both start collapsed and grow
   *  into the full panel on the first message. */
  mode?: "floating" | "inline" | "input" | "bar";
  /** How the chat opens: the usual panel ("expanded", default), or a full-height
   * right sidebar that reserves page space. Expand widens the sidebar. */
  openMode?: "expanded" | "sidebar";
  /** Conversation starters shown as chips; clicking one sends it. They show only while the
   *  conversation hasn't begun: in input/bar modes while the card is still collapsed, and in
   *  floating/inline modes while the open transcript is empty. */
  openingSuggestions?: string[];
  /** Maximum visible chips, with opening suggestions before persistent ones (default 3). */
  maxVisibleSuggestions?: number;
  /** Chips that remain available until chosen, once per session. Clicking one sends it. */
  persistentSuggestions?: string[];
  /** Guest (default) or signed-in customer authentication. */
  auth?: ParlantAuth;
  /** Enable the attach button; you upload to your storage and return a reference. */
  onUpload?: (file: File) => Promise<AttachmentRef>;
  /** Show the attach-file button (only has effect together with `onUpload`). Default false. */
  showAttachments?: boolean;
  /** Restrict the attach picker to these file types (HTML input `accept`, e.g. "image/*"). */
  acceptFileTypes?: string;
  /** Show the agent's narration ("Thinking", a tool's own narration) as the thinking line,
   *  streamed live (default true). The model's own reasoning is part of it only when the
   *  server lets this caller read it; otherwise that step shows as "Thinking". */
  showReasoning?: boolean;
  /** When shown, what becomes of the narration once the reply arrives: "transient"
   *  (disappears, the default) or "collapsible" (folds into a "Thought for Xs" toggle that
   *  opens to every status the agent showed during the turn). */
  reasoningMode?: ReasoningMode;
  /** Inline notice shown at the end of the transcript when a turn fails (session
   *  "error" status). Defaults to a generic "please try again" line. */
  errorMessage?: string;
  /** Shown while the session is in "manual" mode — a human operator has taken over and the
   *  agent won't auto-reply. A string renders in the built-in banner; pass your own JSX to
   *  fully control the markup; pass `null` to show nothing. Defaults to a short line. The
   *  thinking indicator is suppressed in manual mode regardless (no reply is coming). */
  manualModeNotice?: ReactNode;
  /** Header title. Defaults to the agent's name. */
  title?: string;
  /** Image shown in the header's avatar instead of the name's initials, such as your
   *  logo mark (any image URL; a square image works best). */
  avatarUrl?: string;
  /** Muted intro shown above the conversation at the start of a session. */
  preface?: string;
  /** Small print under the input (e.g. a privacy notice). Accepts JSX so you can link. */
  disclaimer?: ReactNode;
  /** A nudge above the launcher before its first open — jumping text with an arrow at
   *  the bird. Defaults to "Talk to us"; pass "" or null to hide it. Floating mode only. */
  launcherHint?: string | null;
  /** Color of the launcher hint text + arrow (any CSS color). Default: follows the theme's
   *  foreground (dark on light themes, light on dark). */
  launcherHintColor?: string;
  /** Show a "New chat" button in the header that starts a fresh session. */
  showNewChat?: boolean;
  /** How streamed agent text is revealed (default "blur"). See `StreamStyle`. */
  streamStyle?: StreamStyle;
  /** Reveal pace: "slow" | "normal" | "fast", or characters/second (default "normal"). */
  streamSpeed?: StreamSpeed;
  /** Play a short blip when the customer's message is sent (default true). */
  customerSound?: boolean;
  /** Play a short blip when an agent reply arrives (default true). */
  agentSound?: boolean;
  /** Which pair of sounds to play (default "velvet"). See `SoundTheme`. */
  soundTheme?: SoundTheme;
  /** Text shown in the empty message input, in every mode (default "Type a message…"). */
  placeholder?: string;
  /** For input/bar/inline modes, focus the input on load (default true). Floating always
   *  focuses when opened, regardless of this. */
  autoFocus?: boolean;
  /** Milliseconds the agent must stay "thinking" before the typing dots fade in, so a fast
   *  reply never flashes them (default 1000). 0 shows them immediately. */
  thinkingDelay?: number;
  /** Color theme (default "monochrome"). Each is a palette of CSS-variable tokens defined in
   *  styles.css; add your own by defining `.parlant-theme-<name>` there. */
  theme?: ThemeName;
  /** CSS font-family for the whole widget (default "Inter", bundled). Pass any family or full
   *  stack; quote multi-word names (e.g. `'"IBM Plex Sans", sans-serif'`). */
  fontFamily?: string;
  /** Sent as the customer's first message, in a new conversation started for it (a stored
   *  conversation isn't resumed). Use it when the page has already asked the visitor something,
   *  e.g. a form that opens the chat. Sent once per mount: pass it only on the visit it's for. */
  initialMessage?: string;
  /** UTM parameters added to every http(s) link the agent writes (Markdown links and bare
   *  URLs), without the `utm_` prefix: `{ source: "site-chat", medium: "chat-widget" }`.
   *  A `utm_*` key the link already has is kept. Attachments and the "Powered by Parlant" link
   *  are never changed. Custom widgets can tag their own links with `context.appendUtm`. */
  utm?: UtmParams;
  /** Which links `utm` applies to: "all" (default) or "external", which leaves links to the
   *  embedding page's own host unchanged. */
  utmLinks?: UtmLinks;
  /** Called once for each label a turn adds to the session (by a policy or observation with
   *  `labels=`, or by code calling `session.labels.add`), with the session's id and its full
   *  current labels. Labels a reopened conversation already has are not reported, nor are
   *  removals. The widget checks after each turn ends, so the call comes then, not mid-reply.
   *  Use it to record a conversion, e.g. set a cookie once a session is labeled "lead". */
  onLabel?: (label: string, context: { sessionId: string; labels: string[] }) => void;
}

/** Built-in color themes; see `.parlant-theme-*` in styles.css. */
export type ThemeName = "monochrome" | "parlant" | "ocean" | "amethyst" | "sand";

export function ChatWidget(props: ChatWidgetProps) {
  const mode = props.mode ?? "input";
  const themeClass = `parlant-theme-${props.theme ?? "monochrome"}`;
  // Override the font only when asked; otherwise styles.css's Inter default stands.
  const rootStyle = props.fontFamily
    ? ({ "--parlant-font": props.fontFamily } as CSSProperties)
    : undefined;
  const [open, setOpen] = useState(mode === "inline");
  const [expanded, setExpanded] = useState(false);
  const [everOpened, setEverOpened] = useState(false);
  // Input mode: whether the panel is open (expanded). Starts closed — NOT derived from
  // message count — so a refresh with a stored session still opens collapsed.
  const [inputOpened, setInputOpened] = useState(false);
  const sidebar = props.openMode === "sidebar" && mode !== "inline";
  const sidebarOpen = sidebar && (mode === "floating" ? open : inputOpened);
  const sidebarRef = useRef<HTMLDivElement>(null);
  useSidebarSpace(sidebarRef, sidebarOpen);

  // Launcher nudge: defaults to "Talk to us"; "" or null hides it. Shown before the
  // first open only.
  const launcherHint = props.launcherHint === undefined ? "Talk to us" : props.launcherHint;
  // No explicit color → the hint follows the theme's foreground (dark on light, light on dark).
  const launcherHintColor = props.launcherHintColor;
  const showHint = Boolean(launcherHint) && !everOpened && !open;

  // Open/minimize animation. The panel mounts on first open and then STAYS mounted
  // (hidden) so its scroll position and session survive close→reopen. It animates by
  // scaling from the launcher corner (scale-0 ↔ scale-1, no opacity → no fade); once
  // fully closed it's `parked` (visibility:hidden) so it's out of the tab order.
  //  - `shown`    — target visible/hidden transform (grow vs. shrink).
  //  - `settling` — the open/close transform is running; during it the size stays
  //                 collapsed and the transition targets transform (not width/height),
  //                 so an expand never coincides with the open scale.
  //  - `parked`   — fully closed and hidden.
  const [shown, setShown] = useState(mode === "inline");
  const [settling, setSettling] = useState(false);
  const [parked, setParked] = useState(false);
  useEffect(() => {
    if (open) {
      setExpanded(false); // always re-open at the normal size
      setParked(false);
      setSettling(true);
      let inner = 0;
      const outer = requestAnimationFrame(() => {
        inner = requestAnimationFrame(() => setShown(true));
      });
      return () => {
        cancelAnimationFrame(outer);
        cancelAnimationFrame(inner);
      };
    }
    // Close — keep it mounted; just animate out (parked on transition end).
    setSettling(true);
    setShown(false);
  }, [open]);

  // Keep bottom-pinned UI above the on-screen keyboard: `position: fixed` anchors to the
  // layout viewport, but the keyboard shrinks the *visual* viewport, hiding the input behind
  // it. Track the gap and expose it as --parlant-kb-inset so the input box can offset by it.
  useEffect(() => {
    const vv = window.visualViewport;
    if (!vv) return;
    const root = document.documentElement.style;
    const update = () => {
      // Pinch-zoom also shrinks the visual viewport, but there the widgets should stay put
      // (you pan to them) — so only track the visual viewport when the page isn't zoomed.
      const zoomed = Math.abs(vv.scale - 1) > 0.01;
      if (zoomed) {
        root.setProperty("--parlant-kb-inset", "0px");
        root.setProperty("--parlant-vv-left", "0px");
        root.setProperty("--parlant-vv-top", "0px");
        root.setProperty("--parlant-vv-width", "100vw");
        root.setProperty("--parlant-vv-height", "100dvh");
      } else {
        // The exact visible band. --kb-inset is the gap below it (for bottom-pinned UI);
        // --vv-* let full-screen surfaces match the visible area precisely, so they clear
        // the keyboard and any emulated scrollbar without double-counting.
        // Only a focused text field raises the keyboard. Without one, the gap is the mobile
        // browser's toolbar sliding in mid-scroll (the layout viewport catches up only when
        // the touch ends), and the browser already keeps fixed elements clear of it, so
        // following it would drag the bar up during the scroll and drop it on release.
        root.setProperty(
          "--parlant-kb-inset",
          isEditable(document.activeElement)
            ? `${Math.max(0, window.innerHeight - vv.height - vv.offsetTop)}px`
            : "0px",
        );
        root.setProperty("--parlant-vv-left", `${vv.offsetLeft}px`);
        root.setProperty("--parlant-vv-top", `${vv.offsetTop}px`);
        root.setProperty("--parlant-vv-width", `${vv.width}px`);
        root.setProperty("--parlant-vv-height", `${vv.height}px`);
      }
    };
    // Focus moves before the keyboard does; on focusout, activeElement settles only after
    // the event, so read it a frame later.
    const updateAfterFocus = () => requestAnimationFrame(update);
    update();
    vv.addEventListener("resize", update);
    vv.addEventListener("scroll", update);
    document.addEventListener("focusin", updateAfterFocus);
    document.addEventListener("focusout", updateAfterFocus);
    return () => {
      vv.removeEventListener("resize", update);
      vv.removeEventListener("scroll", update);
      document.removeEventListener("focusin", updateAfterFocus);
      document.removeEventListener("focusout", updateAfterFocus);
    };
  }, []);

  // Collapse to normal size only while *opening* (so expand never coincides with the
  // open grow).
  const sizeExpanded = expanded && !(open && settling);

  if (mode === "input" || mode === "bar") {
    // A floating composer: docked in the corner ("input") or centered at the bottom of the
    // screen ("bar"). Both are bottom-anchored, so they grow upward from the composer into
    // the full panel when the first message is sent (ChatPanel switches its own height
    // based on whether there are messages). Only the positioning differs.
    const dock = mode === "bar" ? "parlant-bar" : "parlant-input";
    return (
      <div className={cn("parlant-widget", themeClass)} style={rootStyle}>
        <div
          ref={sidebarRef}
          className={cn(dock, sidebar && "parlant-sidebar-dock", inputOpened && `${dock}--open`, sidebarOpen && "parlant-sidebar", sidebarOpen && expanded && "parlant-sidebar--wide")}
        >
          <ChatPanel
            {...props}
            inputMode
            barMode={mode === "bar"}
            inputOpened={inputOpened}
            onOpen={() => setInputOpened(true)}
            onMinimize={() => { setInputOpened(false); setExpanded(false); }}
            expanded={expanded}
            onToggleExpand={() => setExpanded((e) => !e)}
          />
        </div>
      </div>
    );
  }

  if (mode === "inline") {
    return (
      <div className={cn("parlant-widget parlant-inline", themeClass)} style={rootStyle}>
        <ChatPanel {...props} />
      </div>
    );
  }

  return (
    <div className={cn("parlant-widget", themeClass)} style={rootStyle}>
      {showHint && (
        <div
          className="parlant-launcher-hint"
          style={launcherHintColor ? { color: launcherHintColor } : undefined}
        >
          <div className="parlant-cw-hint-bounce">
            <span className="parlant-cw-hint-text">{launcherHint}</span>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden className="parlant-cw-hint-arrow">
              <path d="M12 5v13M6 12l6 6 6-6" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
        </div>
      )}
      {!sidebarOpen && <Launcher
        open={open}
        onToggle={() => {
          setOpen((o) => !o);
          setEverOpened(true);
        }}
      />}
      {everOpened && (
        // Mounted once opened and kept mounted (hidden) thereafter, so scroll + session
        // survive close→reopen. The width tween moves the left edge; the scale animates
        // open/minimize from the corner.
        <div
          ref={sidebarRef}
          aria-hidden={!open}
          onTransitionEnd={(e) => {
            // Once the open/close transform finishes, hand size changes back to the
            // width/height transition (open), or park it out of view (close).
            if (e.target !== e.currentTarget || e.propertyName !== "transform") return;
            if (shown) setSettling(false);
            else setParked(true);
          }}
          className={cn(
            "parlant-panel",
            sizeExpanded ? "parlant-panel--lg" : "parlant-panel--sm",
            // Open: 300ms grow, opacity to full fast. Close: a touch slower (400ms), with
            // opacity on the scale's own curve at ~73% of its duration so it hits 0 before
            // the panel fully collapses (no lingering speck). Width/height animate for expand.
            shown ? "parlant-panel--shown" : "parlant-panel--hidden",
            parked && "parlant-panel--parked",
            sidebar && "parlant-sidebar",
            sidebar && expanded && "parlant-sidebar--wide",
          )}
        >
          <ChatPanel
            {...props}
            onClose={() => setOpen(false)}
            expanded={expanded}
            onToggleExpand={() => setExpanded((e) => !e)}
          />
        </div>
      )}
    </div>
  );
}

function ChatPanel({
  serverUrl,
  agentId,
  auth,
  onUpload,
  showReasoning = true,
  reasoningMode = "transient",
  errorMessage = "The agent ran into an error and couldn't finish this reply. Please try again.",
  manualModeNotice = "A team member has joined and will reply here.",
  title,
  avatarUrl,
  preface,
  disclaimer,
  showNewChat,
  showAttachments = false,
  acceptFileTypes,
  streamStyle = "blur",
  streamSpeed = "normal",
  customerSound = true,
  agentSound = true,
  soundTheme,
  placeholder,
  initialMessage,
  utm,
  utmLinks = "all",
  onLabel,
  autoFocus = true,
  thinkingDelay = 1000,
  openingSuggestions,
  maxVisibleSuggestions = 3,
  persistentSuggestions,
  inputMode,
  barMode,
  inputOpened,
  onOpen,
  onMinimize,
  onClose,
  expanded,
  onToggleExpand,
}: ChatWidgetProps & {
  inputMode?: boolean;
  /** Input mode docked bottom-center instead of in the corner — wider when collapsed. */
  barMode?: boolean;
  inputOpened?: boolean;
  onOpen?: () => void;
  onMinimize?: () => void;
  onClose?: () => void;
  expanded?: boolean;
  onToggleExpand?: () => void;
}) {
  const {
    transport,
    sessionId,
    error: sessionError,
    notice: sessionNotice,
    clearNotice,
    start,
    reset,
    recover,
    // Input mode is lazy: don't create a session on mount, only on the first send.
  } = useParlantSession({
    serverUrl,
    agentId,
    auth,
    autoStart: !inputMode,
    fresh: Boolean(initialMessage),
  });
  const online = useServerHealth(transport);
  const persistent = usePersistentSuggestions(serverUrl, agentId, sessionId);
  const {
    messages,
    hydrated,
    toolWidgets,
    phase,
    status,
    sessionMode,
    reasoning,
    reasoningEventId,
    reasoningStreaming,
    reasoningByTrace,
    error,
    send,
    stop,
    markRevealed,
  } = useParlantStream(transport, sessionId, {
    showReasoning,
    onSessionExpired: recover,
    online,
    onLabel,
  });
  // Recomputed only when the setting changes, so an inline `utm={{…}}` object doesn't re-render
  // every link on each render.
  const utmKey = JSON.stringify(utm ?? null);
  const tagLink = useMemo(
    () => (href: string) => appendUtm(href, utm, { links: utmLinks }),
    [utmKey, utmLinks],
  );
  const [confirmNewChat, setConfirmNewChat] = useState(false);

  // The header banner is for *connectivity* — a transient condition that resolves on its
  // own (server unreachable / stream reconnecting). A session "error" status is different:
  // the current turn failed, so it's shown inline at the end of the transcript (below),
  // right where the reply would have been — not up here as ambient noise.
  const statusBanner = !online
    ? "Can't reach the server. Reconnecting…"
    : error || sessionError
      ? "Connection interrupted. Reconnecting…"
      : null;

  // The agent's public profile drives the header name + avatar initials.
  const [agent, setAgent] = useState<{ id: string; name: string } | null>(null);
  useEffect(() => {
    let cancelled = false;
    transport
      .fetchAgent()
      .then((a) => !cancelled && setAgent(a))
      .catch((e) => {
        if (cancelled) return;
        if (e instanceof GuestIdentityExpiredError) {
          recover("guest-expired");
          return;
        }
        console.error("[parlant] failed to load agent profile", e);
      });
    return () => {
      cancelled = true;
    };
  }, [transport]);

  const name = title ?? agent?.name ?? "Assistant";
  // Manual mode: a human operator handles the session and the engine stays silent, so no reply
  // is coming — drop the "thinking" line and the busy/stop state so the widget doesn't spin.
  const manual = sessionMode === "manual";
  const effectivePhase = manual ? "idle" : phase;
  const busy = effectivePhase !== "idle";

  // Input mode: the open state (NOT message count) drives collapse, so a refresh always
  // starts collapsed. An "active session" — one that exists with messages — hides
  // suggestions and shows a reopen button; in the lazy model a stored session always has
  // messages, so its mere presence is that signal.
  const collapsed = Boolean(inputMode) && !inputOpened;
  const hasActiveSession = Boolean(inputMode) && Boolean(transport.storedSessionId());

  // Lazy send: in input mode with no session yet, create it on the first send and flush the
  // message once the session (and its event stream) is live.
  // An initial message waits here like any other first send, for the new session.
  const pendingSendRef = useRef<{ text: string; attachments?: AttachmentRef[] } | null>(
    initialMessage ? { text: initialMessage } : null,
  );
  // Starters are for a conversation that hasn't begun: in input/bar mode before any
  // session exists, and in every mode once a session's history has loaded and turned out
  // empty (a fresh session, or "New chat"). Waiting for the history keeps the chips from
  // flashing over a resumed conversation or the bootstrap spinner, and the pending send
  // from flashing them while an `initialMessage` is on its way.
  const conversationEmpty = messages.length === 0 && !pendingSendRef.current && !busy;
  const showSuggestions = sessionId
    ? hydrated && conversationEmpty
    : Boolean(inputMode) && collapsed && !pendingSendRef.current;

  const suggestionLimit = Number.isFinite(maxVisibleSuggestions) ? Math.max(0, Math.floor(maxVisibleSuggestions)) : 3;
  const visibleSuggestions = [...new Set([
    ...(showSuggestions ? openingSuggestions ?? [] : []),
    ...(persistentSuggestions ?? []),
  ])].filter((text) => !persistent.chosen.has(text)).slice(0, suggestionLimit);

  const handleSend = (text: string, attachments?: AttachmentRef[]) => {
    onOpen?.(); // input mode expands on send
    clearNotice(); // a message in the new session dismisses the "session expired" notice
    if (sessionId) {
      send(text, attachments);
    } else {
      pendingSendRef.current = { text, attachments };
      void start();
    }
  };
  // Lazy modes create no session on mount, so an initial message starts one (and opens).
  // Once, on mount: later sends go through handleSend.
  useEffect(() => {
    if (inputMode && pendingSendRef.current) {
      onOpen?.();
      void start();
    }
  }, []);
  useEffect(() => {
    if (sessionId && pendingSendRef.current) {
      const queued = pendingSendRef.current;
      pendingSendRef.current = null;
      send(queued.text, queued.attachments);
    }
  }, [sessionId, send]);

  // Blip once per newly-arrived message — only for *fresh* (live) ones, so loading a
  // history/backlog stays silent. Each id is remembered so a message never re-sounds.
  const soundedRef = useRef<Set<string>>(new Set());
  useEffect(() => {
    for (const m of messages) {
      if (soundedRef.current.has(m.id)) continue;
      soundedRef.current.add(m.id);
      if (!m.fresh) continue;
      if (m.role === "customer" && customerSound) playCustomerSound(soundTheme);
      else if (m.role === "agent" && agentSound) playAgentSound(soundTheme);
    }
  }, [messages, customerSound, agentSound, soundTheme]);

  return (
    <div
      className={cn(
        "parlant-chat",
        // Input mode sizes itself: collapsed starts ~20% narrower and auto-height; expanded
        // is the normal panel, or a larger one when the expand button is toggled. Other
        // modes fill their container.
        inputMode
          ? collapsed
            ? barMode
              ? "parlant-chat--bar"
              : "parlant-chat--collapsed"
            : expanded
              ? "parlant-chat--expanded"
              : "parlant-chat--normal"
          : "parlant-chat--fill",
      )}
    >
      {/* A collapsed bar is just the message box: no agent name, no controls. The header
        * comes back the moment it grows into the panel. */}
      {barMode && collapsed ? null : (
        <div className="parlant-cw-header">
          <div className="parlant-cw-header-id">
            <Avatar name={name} imageUrl={avatarUrl} />
            <span className="parlant-cw-header-name">{name}</span>
          </div>
          <div className="parlant-cw-header-actions">
            {showNewChat && !collapsed && (
              <Tooltip label="Start a new chat">
                <button
                  type="button"
                  onClick={() => setConfirmNewChat(true)}
                  aria-label="New chat"
                  className="parlant-cw-iconbtn"
                >
                  <NewChatIcon />
                </button>
              </Tooltip>
            )}
            {/* Size toggle — hidden while the input card is collapsed. */}
            {onToggleExpand && !collapsed && (
              <button
                type="button"
                onClick={onToggleExpand}
                aria-label={expanded ? "Shrink chat" : "Expand chat"}
                className="parlant-cw-iconbtn parlant-cw-size-toggle"
              >
                {expanded ? <ShrinkIcon /> : <ExpandIcon />}
              </button>
            )}
            {/* Input mode, collapsed with an existing conversation: reopen the panel. */}
            {inputMode && collapsed && hasActiveSession && onOpen && (
              <button
                type="button"
                onClick={onOpen}
                aria-label="Reopen chat"
                className="parlant-cw-iconbtn"
              >
                <ExpandIcon />
              </button>
            )}
            {/* Minimize: input mode collapses to the input card; floating closes to the launcher. */}
            {inputMode
              ? !collapsed &&
                onMinimize && (
                  <button
                    type="button"
                    onClick={onMinimize}
                    aria-label="Minimize chat"
                    className="parlant-cw-iconbtn"
                  >
                    <MinimizeIcon />
                  </button>
                )
              : onClose && (
                  <button
                    type="button"
                    onClick={onClose}
                    aria-label="Minimize chat"
                    className="parlant-cw-iconbtn"
                  >
                    <MinimizeIcon />
                  </button>
                )}
          </div>
        </div>
      )}

      {statusBanner && (
        <div className="parlant-cw-banner">
          <span className="parlant-cw-banner-dot" />
          {statusBanner}
        </div>
      )}

      {manual && !collapsed && manualModeNotice != null && (
        typeof manualModeNotice === "string" ? (
          <div role="status" className="parlant-cw-manual">
            <HandoffIcon />
            <span>{manualModeNotice}</span>
          </div>
        ) : (
          manualModeNotice
        )
      )}

      {sessionNotice && (
        <div role="status" className="parlant-cw-notice">
          {sessionNotice}
        </div>
      )}

      {/* Input mode hides the transcript until the first message expands the panel. */}
      {collapsed ? null : sessionId ? (
        // key = session id so scroll state (incl. the "new messages" pill) resets on a new chat.
        <LinkTaggerContext.Provider key={sessionId} value={tagLink}>
          <MessageList
            messages={messages}
            toolWidgets={toolWidgets}
            phase={effectivePhase}
            errored={status === "error"}
            errorMessage={errorMessage}
            reasoning={reasoning}
            reasoningEventId={reasoningEventId}
            reasoningStreaming={reasoningStreaming}
            reasoningByTrace={reasoningByTrace}
            reasoningMode={reasoningMode}
            preface={preface}
            streamStyle={streamStyle}
            streamSpeed={streamSpeed}
            thinkingDelay={thinkingDelay}
            onRevealed={markRevealed}
            onSend={handleSend}
          />
        </LinkTaggerContext.Provider>
      ) : (
        // Bootstrapping a session (initial load, or after "New chat" — which can take a moment).
        <div className="parlant-cw-bootstrap">
          <Spinner />
        </div>
      )}

      <Composer
        onSend={handleSend}
        onStop={stop}
        busy={busy}
        // Input mode has no session until the first send, so it stays enabled (online
        // permitting); other modes wait for the session to bootstrap.
        disabled={(!sessionId && !inputMode) || !online}
        onUpload={showAttachments ? onUpload : undefined}
        accept={acceptFileTypes}
        disclaimer={disclaimer}
        suggestions={visibleSuggestions}
        onSuggestionSelect={(text) => {
          if (persistentSuggestions?.includes(text)) persistent.choose(text);
        }}
        showPoweredBy={!collapsed}
        placeholder={placeholder}
        // Bar styling (bare input, typewriter placeholder, floating chips) is only for the
        // collapsed bar. Once it opens into the panel, the composer matches every other
        // mode — a normal bordered field with inline suggestions.
        barMode={barMode && collapsed}
        connectionIssue={Boolean(statusBanner)}
        // Floating mounts its composer on open, so focus-on-mount = focus-on-open (always).
        // Input/inline mount on load, so honor the autoFocus prop. A collapsed bar starts
        // unfocused so its typewriter attract animation plays until the user clicks in.
        autoFocus={barMode && collapsed ? false : onClose ? true : autoFocus}
      />

      {confirmNewChat && (
        <div className="parlant-cw-confirm">
          <div className="parlant-cw-dialog">
            <p className="parlant-cw-dialog-title">Start a new chat?</p>
            <p className="parlant-cw-dialog-body">
              This starts a fresh conversation.
            </p>
            <div className="parlant-cw-dialog-actions">
              <button
                type="button"
                onClick={() => setConfirmNewChat(false)}
                className="parlant-cw-btn-ghost"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => {
                  setConfirmNewChat(false);
                  reset();
                }}
                className="parlant-cw-btn-primary"
              >
                New chat
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/** Lightweight tooltip: fades in ~100ms after hover/focus, fades out the moment you leave. */
function Tooltip({ label, children }: { label: string; children: ReactNode }) {
  const [show, setShow] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const open = () => {
    timer.current = setTimeout(() => setShow(true), 100);
  };
  const close = () => {
    if (timer.current) clearTimeout(timer.current);
    setShow(false);
  };

  return (
    <span className="parlant-cw-tooltip" onMouseEnter={open} onMouseLeave={close} onFocus={open} onBlur={close}>
      {children}
      <span
        role="tooltip"
        className={cn(
          "parlant-cw-tooltip-label",
          show ? "parlant-cw-tooltip-label--show" : "parlant-cw-tooltip-label--hide",
        )}
      >
        {label}
      </span>
    </span>
  );
}

/** Whether focusing this element brings up an on-screen keyboard. */
function isEditable(el: Element | null): boolean {
  if (el instanceof HTMLTextAreaElement) return !el.readOnly;
  if (el instanceof HTMLInputElement) {
    const nonText = [
      "button", "checkbox", "color", "file", "hidden",
      "image", "radio", "range", "reset", "submit",
    ];
    return !el.readOnly && !nonText.includes(el.type);
  }
  return el instanceof HTMLElement && el.isContentEditable;
}

/** A round avatar: the given image, or else the uppercase first-two initials of the name. */
function Avatar({ name, imageUrl }: { name: string; imageUrl?: string }) {
  if (imageUrl) {
    return (
      <span className="parlant-cw-avatar parlant-cw-avatar--image">
        <img src={imageUrl} alt="" />
      </span>
    );
  }
  return (
    <span className="parlant-cw-avatar">
      {initials(name)}
    </span>
  );
}

function initials(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
  return (parts[0] ?? "").slice(0, 2).toUpperCase();
}

// maximize-2: two arrows on a diagonal, pointing away from each other (flipped).
function ExpandIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden className="parlant-cw-flip-x">
      <polyline points="15 3 21 3 21 9" />
      <polyline points="9 21 3 21 3 15" />
      <line x1="21" y1="3" x2="14" y2="10" />
      <line x1="3" y1="21" x2="10" y2="14" />
    </svg>
  );
}

// minimize-2: two arrows on a diagonal, pointing toward each other (flipped).
function ShrinkIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden className="parlant-cw-flip-x">
      <polyline points="4 14 10 14 10 20" />
      <polyline points="20 10 14 10 14 4" />
      <line x1="14" y1="10" x2="21" y2="3" />
      <line x1="3" y1="21" x2="10" y2="14" />
    </svg>
  );
}

// Shown while a session is being created (initial load / after "New chat").
function Spinner() {
  return (
    <svg
      className="parlant-cw-spinner"
      viewBox="0 0 24 24"
      fill="none"
      aria-label="Loading"
    >
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2.5" className="parlant-cw-spinner-track" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
    </svg>
  );
}

// message-circle-plus (Lucide) — start a fresh conversation ("New chat").
function NewChatIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M7.9 20A9 9 0 1 0 4 16.1L2 22Z" />
      <path d="M8 12h8" />
      <path d="M12 8v8" />
    </svg>
  );
}

// headset (Lucide) — a human operator is now handling the conversation.
function HandoffIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden className="parlant-cw-manual-icon">
      <path d="M3 11h3a1 1 0 0 1 1 1v5a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1z" />
      <path d="M18 11h3v6a1 1 0 0 1-1 1h-2a1 1 0 0 1-1-1v-5a1 1 0 0 1 1-1z" />
      <path d="M4 11a8 8 0 0 1 16 0" />
      <path d="M18 19a3 3 0 0 1-3 3h-2" />
    </svg>
  );
}

// A minus line — you minimize the widget back to the launcher; it never truly closes.
function MinimizeIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path d="M5 12h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}
