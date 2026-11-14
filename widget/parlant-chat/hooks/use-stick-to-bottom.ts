// Keeps a scroll container pinned to the bottom while the user is already there,
// and reports when they've scrolled up (so the UI can show a "new messages" pill).
//
// Growth is tracked with a ResizeObserver on the content element, so it re-pins on
// both React updates AND the streaming reveal's imperative DOM appends. The container
// is observed too: when something takes its space, such as the reconnecting banner or a
// chip coming back, the latest message would otherwise slip under the composer.

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
    // Near the bottom counts as pinned, except while moving up away from it: otherwise a
    // reader scrolling up through the last few lines is pinned again and pulled back down.
    let lastTop = el.scrollTop;
    const onScroll = () => {
      const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
      const movingUp = el.scrollTop < lastTop - 0.5;
      lastTop = el.scrollTop;
      setPinned(movingUp ? distance <= 1 : distance <= NEAR_BOTTOM_PX);
    };
    // Scrolling up is a request to read back, even where the view can't scroll (a short
    // transcript is within reach of the bottom at its top): unpin, so growth below doesn't
    // pull the view back down. Without this, a video avatar reopening its space at the top
    // of a short transcript was scrolled away and docked again, over and over.
    const unpin = () => setPinned(false);
    const onWheel = (event: WheelEvent) => {
      if (event.deltaY < 0) unpin();
    };
    let touchY = 0;
    const onTouchStart = (event: TouchEvent) => {
      touchY = event.touches[0]?.clientY ?? 0;
    };
    const onTouchMove = (event: TouchEvent) => {
      if ((event.touches[0]?.clientY ?? 0) > touchY + 4) unpin();
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "ArrowUp" || event.key === "PageUp" || event.key === "Home") unpin();
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    el.addEventListener("wheel", onWheel, { passive: true });
    el.addEventListener("touchstart", onTouchStart, { passive: true });
    el.addEventListener("touchmove", onTouchMove, { passive: true });
    el.addEventListener("keydown", onKey);
    return () => {
      el.removeEventListener("scroll", onScroll);
      el.removeEventListener("wheel", onWheel);
      el.removeEventListener("touchstart", onTouchStart);
      el.removeEventListener("touchmove", onTouchMove);
      el.removeEventListener("keydown", onKey);
    };
  }, [setPinned]);

  // As the content grows or the view shrinks, stay pinned to the bottom — but only if
  // we already were (instant, to avoid per-token jank).
  useEffect(() => {
    const content = contentRef.current;
    const el = scrollRef.current;
    if (!content || !el) return;
    const observer = new ResizeObserver(() => {
      if (pinnedRef.current) el.scrollTop = el.scrollHeight;
    });
    observer.observe(content);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return { scrollRef, contentRef, isPinned, scrollToBottom };
}
