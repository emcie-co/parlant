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

describe("tables", () => {
  const text = (value: string): Inline[] => (value ? [{ type: "text", value }] : []);

  it("parses a table with a header, a delimiter row and body rows", () => {
    expect(
      parseMarkdown("| Plan | Price |\n| --- | ---: |\n| Starter | Free |\n| Basic | $79 |"),
    ).toEqual([
      {
        type: "table",
        align: [null, "right"],
        header: [text("Plan"), text("Price")],
        rows: [
          [text("Starter"), text("Free")],
          [text("Basic"), text("$79")],
        ],
      },
    ]);
  });

  it("reads each column's alignment from its delimiter cell", () => {
    const [table] = parseMarkdown("a | b | c | d\n:-- | :-: | --: | ---\n1 | 2 | 3 | 4");
    expect(table).toMatchObject({ align: ["left", "center", "right", null] });
  });

  it("accepts rows without outer pipes", () => {
    const [table] = parseMarkdown("Plan | Price\n--- | ---\nBasic | $79");
    expect(table).toMatchObject({ header: [text("Plan"), text("Price")], rows: [[text("Basic"), text("$79")]] });
  });

  it("parses inline Markdown in cells, and keeps escaped pipes and pipes in code in their cell", () => {
    const [table] = parseMarkdown("| Name | Note |\n|---|---|\n| **Plus** | `a|b` or a \\| b |");
    expect(table).toMatchObject({
      rows: [
        [
          [{ type: "strong", children: text("Plus") }],
          [{ type: "code", value: "a|b" }, { type: "text", value: " or a | b" }],
        ],
      ],
    });
  });

  it("pads short rows and drops extra cells, so every row has the header's columns", () => {
    const [table] = parseMarkdown("| a | b |\n|---|---|\n| 1 |\n| 1 | 2 | 3 |");
    expect(table).toMatchObject({ rows: [[text("1"), []], [text("1"), text("2")]] });
  });

  it("ends at a blank line, and keeps the text around it as paragraphs", () => {
    const blocks = parseMarkdown("Our plans:\n| Plan | Price |\n|---|---|\n| Basic | $79 |\n\nAsk me anything.");
    expect(blocks.map((b) => b.type)).toEqual(["paragraph", "table", "paragraph"]);
  });

  it("ends at a line without a pipe", () => {
    const blocks = parseMarkdown("| Plan | Price |\n|---|---|\n| Basic | $79 |\nThat's per month.");
    expect(blocks.map((b) => b.type)).toEqual(["table", "paragraph"]);
  });

  it("shows a streaming table as a table once its delimiter row starts", () => {
    expect(parseMarkdown("| Plan | Price |\n| --")).toEqual([
      { type: "table", align: [null, null], header: [text("Plan"), text("Price")], rows: [] },
    ]);
    const [table] = parseMarkdown("| Plan | Price |\n|---|---|\n| Basic | $7");
    expect(table).toMatchObject({ rows: [[text("Basic"), text("$7")]] });
  });

  it("leaves a pipe in a sentence, and a lone dashed line, as text", () => {
    expect(parseMarkdown("Pick one | or the other").map((b) => b.type)).toEqual(["paragraph"]);
    expect(parseMarkdown("Title\n---").map((b) => b.type)).toEqual(["paragraph"]);
  });
});
