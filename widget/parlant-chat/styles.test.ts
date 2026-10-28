// @vitest-environment jsdom
import { createElement } from "react";
import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import styles from "./styles.css?raw";
import { StreamingText } from "./components/streaming-text";

let stylesheet: HTMLStyleElement;
beforeEach(() => {
  stylesheet = document.createElement("style");
  stylesheet.textContent = styles;
  document.head.appendChild(stylesheet);
});
afterEach(() => {
  cleanup();
  stylesheet.remove();
  vi.unstubAllGlobals();
});

/** The declarations of the first rule whose selector is exactly `selector`. */
function rule(selector: string): string {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = new RegExp(`^${escaped}\\s*\\{([^}]*)\\}`, "m").exec(styles);
  if (!match) throw new Error(`No rule for ${selector} in styles.css`);
  return match[1];
}

// jsdom does no layout, so this pins the rule rather than measuring the bubble.
describe("message bubbles", () => {
  it("break a long word like a URL instead of overflowing", () => {
    // The default "rise" and the "blur" stream styles render each word as an
    // inline-block, which `overflow-wrap: break-word` can't break inside.
    expect(rule(".parlant-widget .parlant-msg-bubble")).toMatch(
      /overflow-wrap:\s*anywhere;/,
    );
  });
});

describe("form controls", () => {
  it.each(["textarea", "input"])(
    "test_that_%s_elements_show_no_border_or_padding_from_a_page_without_a_css_reset",
    (tag) => {
      // Stands in for the browser's own control styling, which a page without
      // a CSS reset (no Tailwind preflight) leaves in place.
      const host = document.createElement("style");
      host.textContent = `${tag} { border: 2px solid red; padding: 4px; background-color: white; }`;
      document.head.prepend(host);
      // jsdom cascades in insertion order, not DOM order; re-insert the widget's
      // sheet so it comes after the page's, as it would in a browser.
      document.head.appendChild(stylesheet);
      try {
        render(
          createElement(
            "div",
            { className: "parlant-widget" },
            createElement(tag, { "data-testid": "control" }),
          ),
        );
        const computed = getComputedStyle(screen.getByTestId("control"));
        expect(computed.borderTopWidth).toBe("0px");
        expect(computed.paddingTop).toBe("0px");
        expect(computed.backgroundColor).toBe("rgba(0, 0, 0, 0)");
      } finally {
        host.remove();
      }
    },
  );
});

describe("opened chat", () => {
  it.each([
    "parlant-input parlant-input--open",
    "parlant-bar parlant-bar--open",
    "parlant-panel parlant-panel--shown",
  ])(
    "test_that_an_opened_chat_sits_above_everything_the_page_layers (%s)",
    (className) => {
      // On a phone the opened chat fills the screen; a page's sticky header at
      // any z-index must not cover its close button.
      render(
        createElement(
          "div",
          { className: "parlant-widget" },
          createElement("div", { className, "data-testid": "surface" }),
        ),
      );
      expect(getComputedStyle(screen.getByTestId("surface")).zIndex).toBe(
        "2147483647",
      );
    },
  );
});

describe("Markdown links", () => {
  it.each([false, true])(
    "test_that_markdown_links_are_bold_and_underlined_despite_the_widget_reset (fresh=%s)",
    (fresh) => {
      const frames: FrameRequestCallback[] = [];
      vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) =>
        frames.push(callback),
      );
      vi.stubGlobal("cancelAnimationFrame", () => {});
      render(
        createElement(
          "div",
          { className: "parlant-widget" },
          createElement(StreamingText, {
            text: "[Quickstart](https://parlant.io/docs/quickstart/installation) and https://parlant.io/docs and [Contact](mailto:hello@parlant.io)",
            streaming: false,
            fresh,
            style: fresh ? "rise" : "instant",
            speed: "normal",
          }),
        ),
      );
      if (fresh) {
        act(() => frames.shift()?.(1));
        act(() => frames.shift()?.(10001));
      }
      const links = screen.getAllByRole("link");
      expect(links).toHaveLength(3);
      for (const link of links) {
        const computed = getComputedStyle(link);
        expect(computed.fontWeight).toBe("700");
        expect(computed.textDecoration).toContain("underline");
        // Rise/blur tokens are inline-blocks: the anchor's underline does not
        // propagate into them, so newly revealed link text needs its own line.
        for (const token of link.querySelectorAll(".parlant-tok")) {
          expect(getComputedStyle(token).textDecoration).toContain("underline");
        }
      }
    },
  );

  it("test_that_non_message_links_keep_their_existing_style", () => {
    render(
      createElement(
        "div",
        { className: "parlant-widget" },
        createElement(
          "a",
          { href: "https://parlant.io" },
          "Powered by Parlant",
        ),
      ),
    );
    const computed = getComputedStyle(screen.getByRole("link"));
    expect(computed.textDecoration).toBe("none");
    expect(computed.fontWeight).not.toBe("700");
  });
});
