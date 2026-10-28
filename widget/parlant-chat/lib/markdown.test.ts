import { describe, expect, it } from "vitest";

import {
  completeIncompleteMarkdown,
  parseInline,
  parseMarkdown,
  type Inline,
} from "./markdown";

const link = (href: string): Inline => ({
  type: "link",
  href,
  children: [{ type: "text", value: href }],
});

describe("bare URLs", () => {
  it("turns a bare https URL into a link to itself", () => {
    expect(
      parseInline(
        "compare here: https://parlant.io/docs/comparisons/langgraph",
      ),
    ).toEqual([
      { type: "text", value: "compare here: " },
      link("https://parlant.io/docs/comparisons/langgraph"),
    ]);
  });

  it("links plain http too", () => {
    expect(parseInline("http://localhost:8800/agents")).toEqual([
      link("http://localhost:8800/agents"),
    ]);
  });

  it.each([".", ",", ")", "?", "!", ":", ";"])(
    "leaves trailing sentence punctuation %j out of the link",
    (mark) => {
      expect(parseInline(`see https://x.io/a${mark} next`)).toEqual([
        { type: "text", value: "see " },
        link("https://x.io/a"),
        { type: "text", value: `${mark} next` },
      ]);
    },
  );

  it("keeps a URL inside parentheses without its closing parenthesis", () => {
    expect(parseInline("(see https://x.io/a)")).toEqual([
      { type: "text", value: "(see " },
      link("https://x.io/a"),
      { type: "text", value: ")" },
    ]);
  });

  it("keeps underscores, query strings and fragments in the link", () => {
    const url = "https://x.io/a_b_c/d?q=1&r=two#section_3";
    expect(parseInline(url)).toEqual([link(url)]);
  });

  it("does not link a URL that is already a markdown link's text or target", () => {
    expect(parseInline("[the docs](https://x.io/a)")).toEqual([
      {
        type: "link",
        href: "https://x.io/a",
        children: [{ type: "text", value: "the docs" }],
      },
    ]);
    expect(parseInline("[https://x.io/a](https://x.io/a)")).toEqual([
      {
        type: "link",
        href: "https://x.io/a",
        children: [{ type: "text", value: "https://x.io/a" }],
      },
    ]);
  });

  it("does not link a URL in inline code", () => {
    expect(parseInline("run `curl https://x.io/a`")).toEqual([
      { type: "text", value: "run " },
      { type: "code", value: "curl https://x.io/a" },
    ]);
  });

  it("does not link other schemes or a scheme glued to a word", () => {
    expect(
      parseInline("javascript:alert(1) and ftp://x.io and xhttps://x.io"),
    ).toEqual([
      {
        type: "text",
        value: "javascript:alert(1) and ftp://x.io and xhttps://x.io",
      },
    ]);
  });

  it("renders a URL that is still streaming in as a link to what has arrived", () => {
    const frame = completeIncompleteMarkdown(
      "more at https://parlant.io/docs/compa",
    );
    expect(parseMarkdown(frame)).toEqual([
      {
        type: "paragraph",
        children: [
          { type: "text", value: "more at " },
          link("https://parlant.io/docs/compa"),
        ],
      },
    ]);
  });

  it("does not treat a lone underscore in a streaming URL as open emphasis", () => {
    const frame = completeIncompleteMarkdown("see https://x.io/a_b");
    expect(parseMarkdown(frame)).toEqual([
      {
        type: "paragraph",
        children: [{ type: "text", value: "see " }, link("https://x.io/a_b")],
      },
    ]);
  });
});
