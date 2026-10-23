// The input row: a single bordered field holding an optional attach button, the
// textarea (Enter sends, Shift+Enter newlines), and an inline Send/Stop icon.

import { type ReactNode, useEffect, useRef, useState } from "react";

import type { AttachmentRef } from "../lib/events";
import { cn } from "../lib/cn";

export interface ComposerProps {
  onSend: (text: string, attachments?: AttachmentRef[]) => void;
  onStop: () => void;
  busy: boolean;
  disabled?: boolean;
  onUpload?: (file: File) => Promise<AttachmentRef>;
  /** HTML input `accept` for the attach picker (e.g. "image/*"). */
  accept?: string;
  /** Small print rendered under the input. */
  disclaimer?: ReactNode;
  /** Conversation starters shown as chips above the input; clicking one sends it. The
   *  caller decides when they apply — see `suggestions` on ChatWidgetProps. */
  suggestions?: string[];
  /** Show the "Powered by Parlant" attribution (default true). */
  showPoweredBy?: boolean;
  /** Input placeholder (default "Type a message…"). */
  placeholder?: string;
  /** Focus the input when it becomes usable (default true). */
  autoFocus?: boolean;
  /** Bar mode: render just the input (no bordered container), rotate the
   *  suggestions through the placeholder while idle, and pop them up as floating
   *  chips above the input on focus. */
  barMode?: boolean;
  /** A connectivity issue is active. In bar mode this restores the bordered
   *  container so the bare input doesn't read as broken while disconnected. */
  connectionIssue?: boolean;
}

export function Composer({
  onSend,
  onStop,
  busy,
  disabled,
  onUpload,
  accept,
  disclaimer,
  suggestions,
  showPoweredBy = true,
  placeholder = "Type a message…",
  autoFocus = true,
  barMode,
  connectionIssue,
}: ComposerProps) {
  const [text, setText] = useState("");
  const [attachments, setAttachments] = useState<AttachmentRef[]>([]);
  const [uploading, setUploading] = useState(false);
  const [focused, setFocused] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Focus the input as soon as it's usable — i.e. when the panel opens and the
  // session is ready. `preventScroll` keeps an inline widget from yanking the page.
  useEffect(() => {
    if (!disabled && autoFocus) inputRef.current?.focus({ preventScroll: true });
  }, [disabled, autoFocus]);

  // Grow the textarea with its content (up to its max-height, then it scrolls).
  // Also recompute when the layout flips (bar collapsed <-> open, or the connection
  // banner toggles): those change the input's padding, so a height cached under the
  // old padding would otherwise carry over — e.g. opening the bar from a suggestion
  // (no text change) would leave the bare 28px height in the standard 20px field.
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }, [text, barMode, connectionIssue]);

  const canSend = Boolean(text.trim() || attachments.length);

  const hasSuggestions = Boolean(suggestions && suggestions.length);

  // Bar mode shows just the input — no bordered container — except while a
  // connectivity issue is active, when the box returns.
  const bare = Boolean(barMode) && !connectionIssue;

  // In bar mode the suggestion chips are not shown inline. They rotate through the
  // placeholder (typewriter) while the input is idle, and appear above the input as
  // floating chips once it's focused. Every other mode keeps the inline chips.
  const barSuggestions = Boolean(barMode) && hasSuggestions;
  const chipsVisible = hasSuggestions && (!barMode || (focused && !text));

  const prefersReducedMotion = usePrefersReducedMotion();
  const animatePlaceholder = barSuggestions && !focused && !text && !prefersReducedMotion;

  const [typed, setTyped] = useState("");
  const suggestionsKey = (suggestions ?? []).join("\u0000");
  useEffect(() => {
    if (!animatePlaceholder) return;

    // Cycle: type each suggestion out char by char, hold, erase it, then move on;
    // after the last suggestion, type the real placeholder, then loop.
    const phrases = [...(suggestions ?? []), placeholder];
    let phrase = 0;
    let char = 0;
    let erasing = false;
    let timer: ReturnType<typeof setTimeout>;
    const TYPE_MS = 55;
    const ERASE_MS = 25;
    const HOLD_MS = 1600;
    const GAP_MS = 400;

    const step = () => {
      const current = phrases[phrase] ?? "";
      if (!erasing) {
        char += 1;
        setTyped(current.slice(0, char));
        if (char >= current.length) {
          erasing = true;
          timer = setTimeout(step, HOLD_MS);
        } else {
          timer = setTimeout(step, TYPE_MS);
        }
      } else {
        char -= 1;
        setTyped(current.slice(0, Math.max(char, 0)));
        if (char <= 0) {
          erasing = false;
          phrase = (phrase + 1) % phrases.length;
          timer = setTimeout(step, GAP_MS);
        } else {
          timer = setTimeout(step, ERASE_MS);
        }
      }
    };

    setTyped("");
    timer = setTimeout(step, GAP_MS);
    return () => clearTimeout(timer);
  }, [animatePlaceholder, suggestionsKey, placeholder]);

  const shownPlaceholder = animatePlaceholder ? typed : placeholder;

  const submit = () => {
    if (!canSend) return;
    onSend(text.trim(), attachments.length ? attachments : undefined);
    setText("");
    setAttachments([]);
  };

  const handleFiles = async (files: FileList | null) => {
    if (!files || !onUpload) return;
    setUploading(true);
    try {
      const refs = await Promise.all(Array.from(files).map((f) => onUpload(f)));
      setAttachments((prev) => [...prev, ...refs]);
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  return (
    <div className="parlant-composer">
      {chipsVisible && suggestions && (
        <div className={cn("parlant-suggestions", barMode && "parlant-suggestions--floating")}>
          {suggestions.map((s, i) => (
            <button
              key={i}
              type="button"
              disabled={disabled}
              // Keep the input focused through the click so the chips (shown on
              // focus in bar mode) don't vanish before the send fires.
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => onSend(s)}
              className="parlant-cmp-suggestion"
            >
              <svg
                className="parlant-cmp-suggestion-icon"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden
              >
                <path d="M22 17a2 2 0 0 1-2 2H6.828a2 2 0 0 0-1.414.586l-2.202 2.202A.71.71 0 0 1 2 21.286V5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2z" />
                <path d="M7 11h10" />
                <path d="M7 15h6" />
                <path d="M7 7h8" />
              </svg>
              <span>{s}</span>
            </button>
          ))}
        </div>
      )}

      {attachments.length > 0 && (
        <div className="parlant-cmp-attachments">
          {attachments.map((a, i) => (
            <span key={i} className="parlant-cmp-attachment">
              <span aria-hidden>📎</span>
              <span className="parlant-cmp-attachment-name">{a.filename ?? a.caption ?? "file"}</span>
              <button
                type="button"
                aria-label="Remove attachment"
                onClick={() => setAttachments((prev) => prev.filter((_, j) => j !== i))}
              >
                ×
              </button>
            </span>
          ))}
        </div>
      )}

      <div className={cn("parlant-cmp-field", bare && "parlant-cmp-field--bare")}>
        {onUpload && (
          <>
            <button
              type="button"
              onClick={() => fileRef.current?.click()}
              disabled={disabled || uploading}
              aria-label="Attach file"
              className="parlant-cmp-attach"
            >
              <PaperclipIcon />
            </button>
            <input
              ref={fileRef}
              type="file"
              accept={accept}
              multiple
              hidden
              onChange={(e) => handleFiles(e.target.files)}
            />
          </>
        )}

        <textarea
          ref={inputRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          placeholder={shownPlaceholder}
          rows={1}
          disabled={disabled}
          // Keep the browser (and password managers) from popping autofill/suggestions.
          autoComplete="off"
          autoCorrect="off"
          spellCheck={false}
          data-1p-ignore
          data-lpignore="true"
          className="parlant-cmp-input"
        />

        {busy ? (
          <button
            type="button"
            onClick={onStop}
            aria-label="Stop"
            className="parlant-cmp-stop"
          >
            <StopIcon />
          </button>
        ) : (
          <button
            type="button"
            onClick={submit}
            disabled={disabled || !canSend}
            aria-label="Send"
            className="parlant-cmp-send"
          >
            <SendIcon />
          </button>
        )}
      </div>

      {disclaimer && (
        <p className="parlant-cmp-disclaimer">
          {disclaimer}
        </p>
      )}

      {/* Attribution under the disclaimer (whether or not one is set). Hidden only in the
          input-mode collapsed state, where it appears once the panel expands. */}
      {showPoweredBy && (
        <div
          className={cn(
            "parlant-cmp-poweredby",
            disclaimer ? "parlant-cmp-poweredby--with-disclaimer" : "parlant-cmp-poweredby--no-disclaimer",
          )}
        >
          <PoweredByParlant />
        </div>
      )}
    </div>
  );
}

// Tracks the user's reduced-motion preference so the typewriter placeholder can fall
// back to a plain static placeholder. Defaults to "no preference" (animate) and is
// safe when matchMedia is unavailable (e.g. jsdom in tests).
function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    if (!mq) return;
    setReduced(mq.matches);
    const onChange = () => setReduced(mq.matches);
    mq.addEventListener?.("change", onChange);
    return () => mq.removeEventListener?.("change", onChange);
  }, []);
  return reduced;
}

// Non-configurable "Powered by Parlant" attribution → parlant.io (UTM = the host page).
//
// Please don't remove this. Parlant is a free, open-source project and depends on the
// good will of its users to spread the word about it. Keeping this small credit visible
// is how new people find the project — thank you for leaving it in. 🙏
function PoweredByParlant() {
  const here = typeof window !== "undefined" ? window.location.href : "";
  const href = `https://www.parlant.io?utm_source=${encodeURIComponent(here)}&utm_medium=chat-widget`;
  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer noopener"
      aria-label="Powered by Parlant"
      className="parlant-cmp-poweredby-link"
    >
      Powered by
      <ParlantWordmark />
    </a>
  );
}

// The Parlant wordmark (bird + "parlant"), inlined and flattened to a single solid color so
// it inherits the theme's ink (--parlant-logo-ink: #222 on light themes, #fff on dark).
function ParlantWordmark() {
  return (
    <svg
      viewBox="0 0 164 38"
      fill="currentColor"
      aria-hidden
      className="parlant-cmp-poweredby-logo"
    >
      <path d="M9.84798 23.2861L6.69882 31.8965C1.87645 29.7059 -1.49209 23.7489 0.666377 17.6158C2.85265 11.4038 8.46412 10.0771 9.84798 9.90294C11.1734 9.73617 17.4809 8.98058 19.9913 12.9743C23.055 17.8481 19.0441 26.4301 9.84798 23.2861Z" />
      <path d="M10.9362 25.3876L6.35033 38L15.5188 32.9635C16.9972 32.9635 21.684 33.5411 25.3887 30.4736C29.5882 26.9966 30.8514 21.5527 28.7772 16.7713C26.7533 12.1057 22.3592 10.1532 19.3413 9.84852C20.4646 10.5438 23.8786 13.6344 22.4063 19.1286C22.4063 19.1286 21.9979 20.9382 20.429 22.7614C17.8703 25.7347 13.7375 26.0651 10.9362 25.3876Z" />
      <path d="M28.6846 3.45015C28.3354 2.93506 27.8115 0.359641 24.1444 0.0162495C21.2195 -0.257637 18.6653 2.94863 15.6217 8.24968H17.8467C22.7318 8.24968 25.6212 10.5181 26.2431 10.9774C26.2431 10.9774 26.4243 8.59872 26.9384 7.57084C27.8423 5.76328 30.0234 5.51049 31.4786 5.1671C29.0339 4.30863 29.0339 3.96523 28.6846 3.45015Z" />
      <path d="M59.5642 18.7758C59.5642 20.7587 59.1559 22.5783 58.3395 24.2346C57.523 25.8676 56.3565 27.1623 54.8402 28.1188C53.3239 29.0752 51.5509 29.5535 49.5213 29.5535C48.9615 29.5535 48.3549 29.5068 47.7017 29.4135C47.0485 29.3202 46.5237 29.2035 46.1271 29.0636L46.2321 32.0029V35.5372L49.8713 35.9221V37.6367H40.1084V35.9221L42.6278 35.5372V12.4421L40.0734 11.8123V9.92268L45.0423 9.29281H45.1123L45.8471 9.92268V11.9872C46.547 11.2174 47.4101 10.5642 48.4366 10.0277C49.4864 9.4911 50.6528 9.22283 51.9358 9.22283C53.3122 9.22283 54.5719 9.56109 55.715 10.2376C56.8814 10.9141 57.8146 11.9639 58.5144 13.3869C59.2143 14.81 59.5642 16.6062 59.5642 18.7758ZM50.5011 11.6723C49.6147 11.6723 48.7865 11.8589 48.0167 12.2322C47.2468 12.6054 46.652 13.0137 46.2321 13.4569V26.6491C46.3954 26.929 46.8036 27.1856 47.4568 27.4189C48.1333 27.6289 48.8682 27.7339 49.6613 27.7339C51.4576 27.7339 52.904 27.0223 54.0004 25.5993C55.1201 24.1763 55.68 22.0301 55.68 19.1607C55.68 16.6412 55.1901 14.7633 54.2103 13.5269C53.2539 12.2905 52.0175 11.6723 50.5011 11.6723ZM62.6061 23.9197C62.6061 21.7268 63.7025 20.1172 65.8954 19.0907C68.0882 18.0409 70.8643 17.4811 74.2236 17.4111V16.4313C74.2236 15.3349 74.1069 14.4717 73.8737 13.8418C73.6404 13.212 73.2321 12.7571 72.6489 12.4771C72.089 12.1739 71.2842 12.0222 70.2344 12.0222C69.0447 12.0222 67.9716 12.1972 67.0151 12.5471C66.082 12.8737 65.1489 13.282 64.2157 13.7719L63.3409 11.9522C63.6442 11.6956 64.2041 11.3574 65.0206 10.9375C65.837 10.5176 66.7818 10.1443 67.8549 9.8177C68.9281 9.4911 70.0012 9.32781 71.0743 9.32781C72.7306 9.32781 74.037 9.56109 74.9934 10.0277C75.9732 10.4709 76.6731 11.1941 77.093 12.1972C77.5129 13.2003 77.7228 14.5417 77.7228 16.2213V27.4539H79.7874V28.9936C79.3208 29.1102 78.7259 29.2269 78.0028 29.3435C77.3029 29.4602 76.6847 29.5185 76.1482 29.5185C75.495 29.5185 75.0517 29.4135 74.8185 29.2035C74.5852 29.0169 74.4685 28.5853 74.4685 27.9088V26.7541C73.7687 27.4539 72.9055 28.0954 71.8791 28.6787C70.876 29.2619 69.7329 29.5535 68.4498 29.5535C66.7702 29.5535 65.3705 29.0752 64.2507 28.1188C63.1543 27.139 62.6061 25.7393 62.6061 23.9197ZM69.9545 27.174C70.561 27.174 71.2609 26.9873 72.054 26.6141C72.8472 26.2408 73.5704 25.7976 74.2236 25.2844V19.2657C71.6808 19.2657 69.7562 19.6623 68.4498 20.4554C67.1668 21.2253 66.5252 22.24 66.5252 23.4998C66.5252 24.7595 66.8285 25.6926 67.435 26.2992C68.0649 26.8824 68.9047 27.174 69.9545 27.174ZM82.291 27.4539L84.9154 27.244V12.4421L82.396 11.8123V9.92268L87.2249 9.29281H87.2949L88.0647 9.92268V10.5525L87.9598 13.282H88.0647C88.298 12.8854 88.7179 12.3722 89.3245 11.7423C89.931 11.1124 90.7125 10.5409 91.669 10.0277C92.6254 9.4911 93.6635 9.22283 94.7833 9.22283C95.2499 9.22283 95.6348 9.28115 95.9381 9.39779V13.317C95.7748 13.2003 95.5065 13.0953 95.1332 13.002C94.7833 12.8854 94.3867 12.8271 93.9435 12.8271C92.0072 12.8271 90.211 13.4219 88.5546 14.6117V27.209L92.9987 27.4889V29.1685H82.291V27.4539ZM98.7027 3.06415L95.8683 2.67923V1.17456L101.502 0.509699H101.572L102.342 1.03459V27.174L105.316 27.4539V29.1685H95.8683V27.4539L98.7027 27.139V3.06415ZM107.801 23.9197C107.801 21.7268 108.897 20.1172 111.09 19.0907C113.283 18.0409 116.059 17.4811 119.419 17.4111V16.4313C119.419 15.3349 119.302 14.4717 119.069 13.8418C118.835 13.212 118.427 12.7571 117.844 12.4771C117.284 12.1739 116.479 12.0222 115.429 12.0222C114.24 12.0222 113.167 12.1972 112.21 12.5471C111.277 12.8737 110.344 13.282 109.411 13.7719L108.536 11.9522C108.839 11.6956 109.399 11.3574 110.216 10.9375C111.032 10.5176 111.977 10.1443 113.05 9.8177C114.123 9.4911 115.196 9.32781 116.269 9.32781C117.926 9.32781 119.232 9.56109 120.188 10.0277C121.168 10.4709 121.868 11.1941 122.288 12.1972C122.708 13.2003 122.918 14.5417 122.918 16.2213V27.4539H124.982V28.9936C124.516 29.1102 123.921 29.2269 123.198 29.3435C122.498 29.4602 121.88 29.5185 121.343 29.5185C120.69 29.5185 120.247 29.4135 120.013 29.2035C119.78 29.0169 119.664 28.5853 119.664 27.9088V26.7541C118.964 27.4539 118.101 28.0954 117.074 28.6787C116.071 29.2619 114.928 29.5535 113.645 29.5535C111.965 29.5535 110.565 29.0752 109.446 28.1188C108.349 27.139 107.801 25.7393 107.801 23.9197ZM115.149 27.174C115.756 27.174 116.456 26.9873 117.249 26.6141C118.042 26.2408 118.765 25.7976 119.419 25.2844V19.2657C116.876 19.2657 114.951 19.6623 113.645 20.4554C112.362 21.2253 111.72 22.24 111.72 23.4998C111.72 24.7595 112.023 25.6926 112.63 26.2992C113.26 26.8824 114.1 27.174 115.149 27.174ZM129.516 12.4421L126.996 11.8123V9.92268L132 9.29281H132.105L132.84 9.92268V11.3924L132.805 12.2322C133.645 11.4857 134.753 10.8092 136.129 10.2026C137.505 9.59608 138.824 9.29281 140.083 9.29281C141.553 9.29281 142.684 9.57275 143.478 10.1326C144.294 10.6925 144.866 11.5557 145.192 12.7221C145.519 13.8652 145.682 15.4165 145.682 17.3761V27.174L148.132 27.4189V29.1685H139.838V27.4539L142.008 27.174V17.3411C142.008 15.9647 141.915 14.8916 141.728 14.1218C141.541 13.3286 141.191 12.7454 140.678 12.3722C140.165 11.9756 139.418 11.7773 138.439 11.7773C137.599 11.7773 136.701 11.9872 135.744 12.4071C134.788 12.8271 133.936 13.317 133.19 13.8768V27.139L135.569 27.4539V29.1685H127.311V27.4539L129.516 27.139V12.4421ZM156.637 29.5185C155.307 29.5185 154.316 29.2152 153.663 28.6087C153.009 28.0021 152.683 26.964 152.683 25.4943V12.0572H150.163V10.4826C150.28 10.4592 150.607 10.3776 151.143 10.2376C151.68 10.0976 152.041 9.96933 152.228 9.85269C152.601 9.64274 152.893 9.22283 153.103 8.59297C153.289 8.14973 153.511 7.43821 153.768 6.45842C154.024 5.47863 154.176 4.89542 154.222 4.7088H156.287L156.357 9.74772H162.201V12.0572H156.357V22.7999C156.357 24.1296 156.392 25.0628 156.462 25.5993C156.555 26.1359 156.73 26.4858 156.987 26.6491C157.267 26.8124 157.733 26.894 158.387 26.894C159.016 26.894 159.705 26.8124 160.451 26.6491C161.221 26.4858 161.816 26.3108 162.236 26.1242L162.761 27.6989C162.201 28.1188 161.291 28.527 160.031 28.9236C158.795 29.3202 157.663 29.5185 156.637 29.5185Z" />
    </svg>
  );
}

function SendIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M12 19V5" />
      <path d="m5 12 7-7 7 7" />
    </svg>
  );
}

function StopIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
      <rect x="6" y="6" width="12" height="12" rx="2" />
    </svg>
  );
}

function PaperclipIcon() {
  return (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48" />
    </svg>
  );
}
