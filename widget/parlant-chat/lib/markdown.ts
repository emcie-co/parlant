// A tiny, zero-dependency Markdown parser scoped to what chat agents actually emit:
// bold, italic, inline code, links, headings, bullet/ordered lists, blockquotes, and
// fenced code blocks. It exists so the widget can render Markdown *and* still reveal it
// token-by-token through our own animation engine (see streaming-text.tsx) — something a
// React-Markdown/Streamdown tree can't do, because it owns its DOM.
//
// Two streaming-specific helpers make it robust mid-stream, mirroring Streamdown's core
// trick: `completeIncompleteMarkdown` closes unterminated syntax so a half-typed `**bold`
// renders as bold immediately instead of flashing literal asterisks; the parser itself is
// cheap enough to re-run every frame.

export type Inline =
  | { type: "text"; value: string }
  | { type: "strong"; children: Inline[] }
  | { type: "em"; children: Inline[] }
  | { type: "code"; value: string }
  | { type: "link"; href: string; children: Inline[] }
  | { type: "br" };

export type Block =
  | { type: "paragraph"; children: Inline[] }
  | { type: "heading"; level: number; children: Inline[] }
  | { type: "code"; value: string }
  | { type: "list"; ordered: boolean; items: Inline[][] }
  | { type: "blockquote"; children: Block[] };

const HEADING = /^(#{1,6})\s+(.*)$/;
const LIST_ITEM = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/;
const BLOCKQUOTE = /^>\s?(.*)$/;
const FENCE = /^[ \t]*```/;

/** Parse a Markdown string into a block AST. Safe to call every frame while streaming. */
export function parseMarkdown(src: string): Block[] {
  const lines = src.replace(/\r\n?/g, "\n").split("\n");
  const blocks: Block[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (line.trim() === "") {
      i++;
      continue;
    }

    // Fenced code block: everything until the closing fence (or end of input).
    if (FENCE.test(line)) {
      const body: string[] = [];
      i++;
      while (i < lines.length && !FENCE.test(lines[i])) body.push(lines[i++]);
      if (i < lines.length) i++; // consume the closing fence
      blocks.push({ type: "code", value: body.join("\n") });
      continue;
    }

    const heading = HEADING.exec(line);
    if (heading) {
      blocks.push({
        type: "heading",
        level: heading[1].length,
        children: parseInline(heading[2]),
      });
      i++;
      continue;
    }

    if (LIST_ITEM.test(line)) {
      const items: Inline[][] = [];
      const ordered = /^\s*\d+[.)]/.test(line);
      while (i < lines.length) {
        const m = LIST_ITEM.exec(lines[i]);
        if (!m) break;
        items.push(parseInline(m[3]));
        i++;
      }
      blocks.push({ type: "list", ordered, items });
      continue;
    }

    if (BLOCKQUOTE.test(line)) {
      const inner: string[] = [];
      while (i < lines.length && BLOCKQUOTE.test(lines[i])) {
        inner.push(BLOCKQUOTE.exec(lines[i])![1]);
        i++;
      }
      blocks.push({ type: "blockquote", children: parseMarkdown(inner.join("\n")) });
      continue;
    }

    // Paragraph: consecutive non-blank lines that aren't another block type.
    const para: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() !== "" &&
      !FENCE.test(lines[i]) &&
      !HEADING.test(lines[i]) &&
      !LIST_ITEM.test(lines[i]) &&
      !BLOCKQUOTE.test(lines[i])
    ) {
      para.push(lines[i]);
      i++;
    }
    blocks.push({ type: "paragraph", children: parseInlineMultiline(para) });
  }

  return blocks;
}

/** Inline parse across paragraph lines, turning single newlines into hard breaks. */
function parseInlineMultiline(lines: string[]): Inline[] {
  const out: Inline[] = [];
  lines.forEach((line, idx) => {
    if (idx > 0) out.push({ type: "br" });
    out.push(...parseInline(line));
  });
  return out;
}

// A bare http(s) URL: starts where a word doesn't, runs to whitespace or a quote/angle.
const BARE_URL = /^https?:\/\/[^\s<>"'`]+/i;
// Sentence punctuation that ends a sentence rather than the URL.
const URL_TRAILING = /[.,;:!?'"\]]$/;

/** The bare URL starting at `i`, without trailing sentence punctuation, if one does. */
function bareUrlAt(text: string, i: number): string | null {
  if (i > 0 && /[A-Za-z0-9]/.test(text[i - 1])) return null;
  const m = BARE_URL.exec(text.slice(i));
  if (!m) return null;
  let url = m[0];
  for (;;) {
    if (URL_TRAILING.test(url)) url = url.slice(0, -1);
    // A closing parenthesis belongs to the URL only if the URL opened one.
    else if (url.endsWith(")") && url.split("(").length < url.split(")").length) url = url.slice(0, -1);
    else break;
  }
  return /^https?:\/\/./i.test(url) ? url : null;
}

/** Parse inline Markdown (bold / italic / code / links, and bare URLs) into nodes. */
export function parseInline(text: string, linkBareUrls = true): Inline[] {
  const nodes: Inline[] = [];
  let buf = "";
  let i = 0;
  const flush = () => {
    if (buf) {
      nodes.push({ type: "text", value: buf });
      buf = "";
    }
  };

  while (i < text.length) {
    const c = text[i];

    // Backslash escape: the next char is literal.
    if (c === "\\" && i + 1 < text.length) {
      buf += text[i + 1];
      i += 2;
      continue;
    }

    // Inline code — highest precedence; nothing formats inside it.
    if (c === "`") {
      const end = text.indexOf("`", i + 1);
      if (end !== -1) {
        flush();
        nodes.push({ type: "code", value: text.slice(i + 1, end) });
        i = end + 1;
        continue;
      }
    }

    // Bare URL (before emphasis, so a URL's underscores stay in it).
    if (linkBareUrls && (c === "h" || c === "H")) {
      const url = bareUrlAt(text, i);
      if (url) {
        flush();
        nodes.push({ type: "link", href: url, children: [{ type: "text", value: url }] });
        i += url.length;
        continue;
      }
    }

    // Strong: ** or __ (checked before single-char emphasis).
    if ((c === "*" || c === "_") && text[i + 1] === c) {
      const marker = c + c;
      const end = text.indexOf(marker, i + 2);
      if (end !== -1) {
        flush();
        nodes.push({ type: "strong", children: parseInline(text.slice(i + 2, end), linkBareUrls) });
        i = end + 2;
        continue;
      }
    }

    // Emphasis: * or _
    if (c === "*" || c === "_") {
      const end = text.indexOf(c, i + 1);
      if (end !== -1 && end !== i + 1) {
        flush();
        nodes.push({ type: "em", children: parseInline(text.slice(i + 1, end), linkBareUrls) });
        i = end + 1;
        continue;
      }
    }

    // Link: [text](href)
    if (c === "[") {
      const m = /^\[([^\]]*)\]\(([^)\s]+)\)/.exec(text.slice(i));
      if (m) {
        flush();
        // A link's text is never linked again.
        nodes.push({ type: "link", href: m[2], children: parseInline(m[1], false) });
        i += m[0].length;
        continue;
      }
    }

    buf += c;
    i++;
  }

  flush();
  return nodes;
}

/**
 * Close unterminated Markdown so a mid-stream frame renders cleanly rather than flashing
 * literal syntax. Mirrors Streamdown's key streaming trick, scoped to our subset:
 *  - an open fenced code block is closed (its body renders as code meanwhile);
 *  - unterminated inline emphasis/strong and inline code get their closers appended.
 */
export function completeIncompleteMarkdown(src: string): string {
  const s = src.replace(/\r\n?/g, "\n");

  // Inside an unclosed fence, everything trailing is code — just close the fence.
  const fenceCount = (s.match(/^[ \t]*```/gm) || []).length;
  if (fenceCount % 2 === 1) return `${s}\n\`\`\``;

  // Track open inline markers (outside inline code) and append the needed closers.
  const open: string[] = [];
  let inCode = false;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (c === "\\") {
      i++;
      continue;
    }
    if (c === "`") {
      inCode = !inCode;
      continue;
    }
    if (inCode) continue;

    // A URL's underscores aren't emphasis markers.
    const url = (c === "h" || c === "H") && bareUrlAt(s, i);
    if (url) {
      i += url.length - 1;
      continue;
    }

    if ((c === "*" || c === "_") && s[i + 1] === c) {
      const marker = c + c;
      if (open[open.length - 1] === marker) open.pop();
      else open.push(marker);
      i++;
      continue;
    }
    if (c === "*" || c === "_") {
      if (open[open.length - 1] === c) open.pop();
      else open.push(c);
    }
  }

  let suffix = "";
  if (inCode) suffix += "`";
  for (let k = open.length - 1; k >= 0; k--) suffix += open[k];
  return s + suffix;
}

/**
 * Split a run of text into reveal units, in the order the animation shows them:
 *  - "letter" style → one unit per character;
 *  - otherwise → one unit per word plus its trailing whitespace (so inline-block token
 *    transforms don't swallow spaces).
 */
export function splitUnits(text: string, letter: boolean): string[] {
  if (letter) return Array.from(text);
  const units: string[] = [];
  let i = 0;
  while (i < text.length) {
    let j = i;
    while (j < text.length && !/\s/.test(text[j])) j++;
    while (j < text.length && /\s/.test(text[j])) j++;
    units.push(text.slice(i, j > i ? j : i + 1));
    i = j > i ? j : i + 1;
  }
  return units;
}
