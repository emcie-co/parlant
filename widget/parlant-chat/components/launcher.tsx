import type { CSSProperties } from "react";

import type { ShownAvatar } from "./chat-widget";
import { LoopingVideo } from "./looping-video";

// The floating bubble that opens/closes the chat panel. Closed, it shows a round chat
// bubble, or the agent's video or photo when there is one; open, it becomes a down-chevron (no
// shaft). The two crossfade + scale into each other, keyed off the button's data-open
// (see styles.css).

export interface LauncherProps {
  open: boolean;
  onToggle: () => void;
  /** A photo or video avatar stands in for the chat bubble while closed, with its live dot.
   *  Initials and a logo don't fill it, so the bubble stays. */
  avatar?: ShownAvatar;
  /** The server can't be reached: the live dot turns red, like the panel's banner. */
  offline?: boolean;
}

/** The dot on a photo's corner: green while live, red while the connection is down. */
export function LiveDot({ offline = false }: { offline?: boolean }) {
  return (
    <span
      className={offline ? "parlant-cw-live parlant-cw-live--offline" : "parlant-cw-live"}
      role="img"
      aria-label={offline ? "Offline" : "Live"}
    />
  );
}

export function Launcher({ open, onToggle, avatar, offline = false }: LauncherProps) {
  const video = avatar?.type === "video" ? avatar : undefined;
  const photoUrl = avatar?.type === "photo" ? avatar.src : undefined;
  const portrait = Boolean(video || photoUrl);
  const live = (avatar?.type === "photo" || avatar?.type === "video") && Boolean(avatar.live);
  // The crop, as custom properties the video's transform reads (see styles.css).
  const crop = video?.launcher
    ? ({
        "--parlant-launcher-zoom": video.launcher.zoom,
        "--parlant-launcher-x": video.launcher.x,
        "--parlant-launcher-y": video.launcher.y,
      } as CSSProperties)
    : undefined;
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-label={open ? "Minimize chat" : "Open chat"}
      data-open={open ? "true" : "false"}
      data-photo={portrait ? "true" : undefined}
      data-video={video ? "true" : undefined}
      className="parlant-launcher"
    >
      {video ? (
        <span className="parlant-launcher-photo" style={crop}>
          <LoopingVideo src={video.src} poster={video.poster} className="parlant-launcher-video" />
        </span>
      ) : (
        photoUrl && <img src={photoUrl} alt="" className="parlant-launcher-photo" />
      )}
      {portrait && live && <LiveDot offline={offline} />}
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden
        className="parlant-face parlant-lch-icon"
      >
        <defs>
          {/* White = kept (solid bubble), black = knocked out (eyes + smile → transparent). */}
          <mask id="parlant-chatbot-mask">
            <path
              fill="white"
              stroke="none"
              d="M18 4a3 3 0 0 1 3 3v8a3 3 0 0 1 -3 3h-5l-5 3v-3h-2a3 3 0 0 1 -3 -3v-8a3 3 0 0 1 3 -3h12"
            />
            <g stroke="black" strokeWidth="2">
              <path d="M9.5 9h.01" />
              <path d="M14.5 9h.01" />
              <path d="M9.5 13a3.5 3.5 0 0 0 5 0" />
            </g>
          </mask>
        </defs>
        {/* Chatbot bubble (closed): solid on light themes, outline on dark themes (toggled in
            styles.css). Both share the .parlant-bubble crossfade. */}
        <g className="parlant-bubble">
          <rect
            className="parlant-lch-solid"
            width="24"
            height="24"
            fill="currentColor"
            stroke="none"
            mask="url(#parlant-chatbot-mask)"
          />
          <g className="parlant-lch-outline">
            <path d="M18 4a3 3 0 0 1 3 3v8a3 3 0 0 1 -3 3h-5l-5 3v-3h-2a3 3 0 0 1 -3 -3v-8a3 3 0 0 1 3 -3h12" />
            <path d="M9.5 9h.01" />
            <path d="M14.5 9h.01" />
            <path d="M9.5 13a3.5 3.5 0 0 0 5 0" />
          </g>
        </g>
        {/* Down chevron, no shaft (open) */}
        <path className="parlant-chevron" d="M6 10.5L12 16.5L18 10.5" />
      </svg>
    </button>
  );
}
