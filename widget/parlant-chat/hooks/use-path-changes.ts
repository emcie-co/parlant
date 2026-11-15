// Reports the page's path as the widget mounts and whenever it changes without a reload: a
// single-page app's links, back and forward. Only the path counts, not its ?query or #hash.

import { useEffect, useRef } from "react";

const PATH_CHANGE = "parlant:pathchange";

/** Single-page apps move with history.pushState/replaceState, which fire no event of their
 *  own; announce them. Once per page, however many widgets (or copies of one) it has. */
function announceHistoryChanges(): void {
  const page = window as Window & { __parlantHistoryAnnounced?: boolean };
  if (page.__parlantHistoryAnnounced) return;
  page.__parlantHistoryAnnounced = true;
  for (const method of ["pushState", "replaceState"] as const) {
    const original = history[method];
    history[method] = function (this: History, ...args: Parameters<History["pushState"]>) {
      original.apply(this, args);
      // After the router that called it is done, not in the middle of its update.
      queueMicrotask(() => window.dispatchEvent(new Event(PATH_CHANGE)));
    };
  }
}

export function usePathChanges(onPath: ((path: string) => void) | undefined): void {
  const callback = useRef(onPath);
  callback.current = onPath;
  const enabled = onPath !== undefined;

  useEffect(() => {
    if (!enabled || typeof window === "undefined") return;
    announceHistoryChanges();
    let last: string | null = null;
    const check = () => {
      const path = window.location.pathname;
      if (path === last) return;
      last = path;
      callback.current?.(path);
    };
    check();
    window.addEventListener(PATH_CHANGE, check);
    window.addEventListener("popstate", check);
    return () => {
      window.removeEventListener(PATH_CHANGE, check);
      window.removeEventListener("popstate", check);
    };
  }, [enabled]);
}
