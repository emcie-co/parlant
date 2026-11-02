import { type RefObject, useEffect } from "react";

interface PageSpace {
  widths: Map<symbol, number>;
  margin: string;
  priority: string;
  base: number;
}

// A page may mount more than one widget. Reserve the widest right-hand column,
// and restore the host's original inline style when the last sidebar closes.
const pages = new WeakMap<HTMLElement, PageSpace>();

export function useSidebarSpace(ref: RefObject<HTMLDivElement>, active: boolean): void {
  useEffect(() => {
    const element = ref.current;
    if (!active || !element) return;

    const body = document.body;
    let page = pages.get(body);
    if (!page) {
      page = {
        widths: new Map(),
        margin: body.style.getPropertyValue("margin-right"),
        priority: body.style.getPropertyPriority("margin-right"),
        base: Number.parseFloat(getComputedStyle(body).marginRight) || 0,
      };
      pages.set(body, page);
    }
    const token = Symbol();
    const restore = () => {
      if (page.margin) body.style.setProperty("margin-right", page.margin, page.priority);
      else body.style.removeProperty("margin-right");
    };
    const apply = () => {
      const width = Math.max(0, ...page.widths.values());
      if (width) {
        const margin = `${page.base + width}px`;
        if (body.style.getPropertyValue("margin-right") !== margin) {
          body.style.setProperty("margin-right", margin);
        }
      }
      else restore();
    };
    const update = () => {
      // Phones keep the existing full-screen chat behavior; a reserved column
      // would leave no usable page beside it.
      page.widths.set(token, window.innerWidth > 768 ? element.offsetWidth : 0);
      apply();
    };
    update();
    // ResizeObserver runs during layout. Changing the page margin there can
    // resize other observed elements before their notifications are delivered.
    // Coalesce writes into the next frame instead of feeding that layout cycle.
    let frame = 0;
    const schedule = () => {
      if (frame) return;
      frame = requestAnimationFrame(() => { frame = 0; update(); });
    };
    const observer = new ResizeObserver(schedule);
    observer.observe(element);
    window.addEventListener("resize", schedule);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", schedule);
      cancelAnimationFrame(frame);
      page.widths.delete(token);
      apply();
      if (!page.widths.size) pages.delete(body);
    };
  }, [ref, active]);
}
