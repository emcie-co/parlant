// Keeps a scroll container pinned to the bottom while the user is already there,
// and reports when they've scrolled up (so the UI can show a "new messages" pill).
//
// Growth is tracked with a ResizeObserver on the content element, so it re-pins on
// both React updates AND the streaming reveal's imperative DOM appends.

import { type RefObject, useCallback, useEffect, useRef, useState } from "react";

const NEAR_BOTTOM_PX = 80;

export interface StickToBottom {
  /** Attach to the scrollable container. */
  scrollRef: RefObject<HTMLDivElement>;
  /** Attach to the growing content inside the container. */
  contentRef: RefObject<HTMLDivElement>;
  /** True when the view is at (or near) the bottom. */
  isPinned: boolean;
  /** Jump to the bottom; instant during streaming, smooth for a discrete new message. */
  scrollToBottom: (behavior?: ScrollBehavior) => void;
}

export function useStickToBottom(): StickToBottom {
  const scrollRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const pinnedRef = useRef(true);
  const [isPinned, setIsPinned] = useState(true);

  const setPinned = useCallback((value: boolean) => {
    pinnedRef.current = value;
    setIsPinned(value);
  }, []);

  const scrollToBottom = useCallback(
    (behavior: ScrollBehavior = "auto") => {
      const el = scrollRef.current;
      if (!el) return;
      el.scrollTo({ top: el.scrollHeight, behavior });
      setPinned(true);
    },
    [setPinned],
  );

  // Track whether the user is near the bottom.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const onScroll = () => {
      const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
      setPinned(distance <= NEAR_BOTTOM_PX);
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, [setPinned]);

  // As the content grows, stay pinned to the bottom — but only if we already were
  // (instant, to avoid per-token jank).
  useEffect(() => {
    const content = contentRef.current;
    const el = scrollRef.current;
    if (!content || !el) return;
    const observer = new ResizeObserver(() => {
      if (pinnedRef.current) el.scrollTop = el.scrollHeight;
    });
    observer.observe(content);
    return () => observer.disconnect();
  }, []);

  return { scrollRef, contentRef, isPinned, scrollToBottom };
}
