// The floating bubble that opens/closes the chat panel. Closed, it shows a round chat
// bubble; open, it becomes a down-chevron (no shaft). The two crossfade + scale into
// each other, keyed off the button's data-open (see styles.css).

export interface LauncherProps {
  open: boolean;
  onToggle: () => void;
}

export function Launcher({ open, onToggle }: LauncherProps) {
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-label={open ? "Minimize chat" : "Open chat"}
      data-open={open ? "true" : "false"}
      className="parlant-launcher"
    >
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
