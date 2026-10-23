// The streaming reveal engine, shared by agent replies (message-bubble) and the
// reasoning line (status-bar). It renders Markdown *and* reveals it token-by-token under
// the chosen `streamStyle` / `streamSpeed`.
//
// The trick a React-Markdown/Streamdown tree can't do: because we own the renderer (see
// lib/markdown.ts), we render the Markdown structure ourselves and, at each text leaf,
// emit our animated `.parlant-tok` spans. So a **bold list item** streams in word-by-word
// with the rise/blur/fade animation intact. New tokens mount as the reveal advances (each
// animates once); already-shown tokens are preserved by key and never re-animate.

import { createElement, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import type { StreamStyle, StreamSpeed } from "../lib/events";
import {
  completeIncompleteMarkdown,
  parseMarkdown,
  splitUnits,
  type Block,
  type Inline,
} from "../lib/markdown";

const SPEED_CPS: Record<string, number> = { slow: 120, normal: 240, fast: 480 };
const cps = (speed: StreamSpeed): number =>
  typeof speed === "number" ? speed : (SPEED_CPS[speed] ?? SPEED_CPS.normal);

/**
 * Reveal-animated Markdown for agent text (replies and reasoning alike).
 *
 * Styles (`StreamStyle`): "instant" (no animation), "fade", "rise" (default), "blur",
 * "letter", "warm". `speed` is characters/second ("slow" | "normal" | "fast" or a number).
 *
 * A message that mounts already complete (history, block mode) renders at once; only a
 * fresh live message animates. Remount (via `key`) to replay from empty for a new turn.
 */
export function StreamingText({
  text,
  streaming,
  fresh,
  style,
  speed,
}: {
  text: string;
  streaming: boolean;
  fresh: boolean;
  style: StreamStyle;
  speed: StreamSpeed;
}) {
  const letter = style === "letter";
  // Decide once: animate a message that arrived live; history / "instant" render at once.
  const animate = useRef<boolean | null>(null);
  if (animate.current === null) animate.current = fresh && style !== "instant";
  const animating = animate.current;

  // Re-parse only when the text grows (a new chunk), not on every reveal tick.
  const blocks = useMemo(() => parseMarkdown(completeIncompleteMarkdown(text)), [text]);
  const unitLens = useMemo(() => flattenUnitLens(blocks, letter), [blocks, letter]);

  // How many reveal units are currently shown. Static (history/instant) shows everything.
  const [revealed, setRevealed] = useState(animating ? 0 : Number.MAX_SAFE_INTEGER);
  const revealedRef = useRef(revealed);
  revealedRef.current = revealed;
  const speedRef = useRef(speed);
  speedRef.current = speed;
  const streamingRef = useRef(streaming);
  streamingRef.current = streaming;
  const lensRef = useRef(unitLens);
  lensRef.current = unitLens;

  // A single rAF loop for the message's life: reveal buffered units at `speed`, idle while
  // caught up but still streaming, and stop once the stream has ended. Reading from refs
  // (not deps) keeps one stable loop — robust to re-renders and StrictMode.
  useEffect(() => {
    if (!animating) return;
    let cancelled = false;
    let last = 0;
    let budget = 0;
    let id = requestAnimationFrame(function loop(ts) {
      if (cancelled) return;
      const lens = lensRef.current;
      let r = revealedRef.current;
      if (r < lens.length) {
        if (!last) last = ts;
        budget += ((ts - last) / 1000) * cps(speedRef.current);
        last = ts;
        let advanced = false;
        while (r < lens.length && budget >= lens[r]) {
          budget -= lens[r];
          r++;
          advanced = true;
        }
        if (advanced) {
          revealedRef.current = r;
          setRevealed(r);
        }
        id = requestAnimationFrame(loop);
      } else if (streamingRef.current) {
        last = 0; // caught up, but more may still arrive
        id = requestAnimationFrame(loop);
      }
      // else: caught up and the stream ended — stop looping.
    });
    return () => {
      cancelled = true;
      cancelAnimationFrame(id);
    };
  }, [animating]);

  const leaf: Leaf = animating
    ? revealLeaf({ count: 0, limit: revealed, letter })
    : STATIC_LEAF;

  return (
    <div className={`parlant-stream parlant-md ${animating ? `parlant-fx-${style}` : ""}`}>
      {renderBlocks(blocks, leaf, "b")}
    </div>
  );
}

// --- rendering -------------------------------------------------------------

/** How text leaves render: plain (static) or as reveal-gated animated tokens. */
interface Leaf {
  text(value: string, keyBase: string): ReactNode[];
  inlineCode(value: string, key: string): ReactNode | null;
  fencedCode(value: string): ReactNode | null;
  done(): boolean;
}

const STATIC_LEAF: Leaf = {
  text: (value) => [value],
  inlineCode: (value, key) => (
    <code key={key} className="parlant-md-code">
      {value}
    </code>
  ),
  fencedCode: (value) => value,
  done: () => false,
};

function revealLeaf(ctx: { count: number; limit: number; letter: boolean }): Leaf {
  const emit = (): boolean => {
    if (ctx.count >= ctx.limit) return false;
    ctx.count += 1;
    return true;
  };
  return {
    text(value) {
      const out: ReactNode[] = [];
      for (const unit of splitUnits(value, ctx.letter)) {
        if (!emit()) break;
        const i = ctx.count; // unique, monotonic global index → stable key, no re-animation
        if (ctx.letter) {
          out.push(
            <span key={i} className="parlant-tok">
              {unit}
            </span>,
          );
        } else {
          const word = unit.replace(/\s+$/, "");
          const space = unit.slice(word.length);
          if (word)
            out.push(
              <span key={i} className="parlant-tok">
                {word}
              </span>,
            );
          if (space) out.push(space);
        }
      }
      return out;
    },
    inlineCode(value, key) {
      if (!emit()) return null;
      return (
        <code key={key} className="parlant-md-code parlant-tok">
          {value}
        </code>
      );
    },
    fencedCode(value) {
      if (!emit()) return null;
      return value;
    },
    done: () => ctx.count >= ctx.limit,
  };
}

function renderBlocks(blocks: Block[], leaf: Leaf, keyBase: string): ReactNode[] {
  const out: ReactNode[] = [];
  blocks.forEach((block, i) => {
    if (leaf.done()) return;
    const key = `${keyBase}.${i}`;
    switch (block.type) {
      case "paragraph": {
        const kids = renderInline(block.children, leaf, key);
        if (kids.length) out.push(<p key={key}>{kids}</p>);
        break;
      }
      case "heading": {
        const kids = renderInline(block.children, leaf, key);
        if (kids.length) out.push(createElement(`h${block.level}`, { key }, kids));
        break;
      }
      case "code": {
        const body = leaf.fencedCode(block.value);
        if (body != null)
          out.push(
            <pre key={key} className="parlant-md-pre">
              <code>{body}</code>
            </pre>,
          );
        break;
      }
      case "list": {
        const lis: ReactNode[] = [];
        block.items.forEach((item, j) => {
          if (leaf.done()) return;
          const kids = renderInline(item, leaf, `${key}.${j}`);
          if (kids.length) lis.push(<li key={j}>{kids}</li>);
        });
        if (lis.length) {
          out.push(
            block.ordered ? (
              <ol key={key} className="parlant-md-ol">
                {lis}
              </ol>
            ) : (
              <ul key={key} className="parlant-md-ul">
                {lis}
              </ul>
            ),
          );
        }
        break;
      }
      case "blockquote": {
        const kids = renderBlocks(block.children, leaf, key);
        if (kids.length)
          out.push(
            <blockquote key={key} className="parlant-md-quote">
              {kids}
            </blockquote>,
          );
        break;
      }
    }
  });
  return out;
}

function renderInline(nodes: Inline[], leaf: Leaf, keyBase: string): ReactNode[] {
  const out: ReactNode[] = [];
  nodes.forEach((node, i) => {
    if (leaf.done()) return;
    const key = `${keyBase}.${i}`;
    switch (node.type) {
      case "text":
        out.push(...leaf.text(node.value, key));
        break;
      case "strong": {
        const kids = renderInline(node.children, leaf, key);
        if (kids.length) out.push(<strong key={key}>{kids}</strong>);
        break;
      }
      case "em": {
        const kids = renderInline(node.children, leaf, key);
        if (kids.length) out.push(<em key={key}>{kids}</em>);
        break;
      }
      case "code": {
        const el = leaf.inlineCode(node.value, key);
        if (el) out.push(el);
        break;
      }
      case "link": {
        const kids = renderInline(node.children, leaf, key);
        if (kids.length) out.push(renderLink(node.href, kids, key));
        break;
      }
      case "br":
        out.push(<br key={key} />);
        break;
    }
  });
  return out;
}

function renderLink(href: string, children: ReactNode[], key: string): ReactNode {
  // Only allow safe schemes; anything else renders as plain text (no javascript: etc.).
  const safe = /^(https?:|mailto:)/i.test(href);
  if (!safe) return <span key={key}>{children}</span>;
  return (
    <a key={key} href={href} target="_blank" rel="noreferrer noopener" className="parlant-md-link">
      {children}
    </a>
  );
}

/** Reveal-unit char lengths, in render order — drives the reveal clock's char budget. */
function flattenUnitLens(blocks: Block[], letter: boolean): number[] {
  const lens: number[] = [];
  const inline = (nodes: Inline[]) => {
    for (const n of nodes) {
      switch (n.type) {
        case "text":
          for (const u of splitUnits(n.value, letter)) lens.push(u.length);
          break;
        case "strong":
        case "em":
        case "link":
          inline(n.children);
          break;
        case "code":
          lens.push(n.value.length);
          break;
        case "br":
          break;
      }
    }
  };
  const walk = (bs: Block[]) => {
    for (const b of bs) {
      switch (b.type) {
        case "paragraph":
        case "heading":
          inline(b.children);
          break;
        case "code":
          lens.push(b.value.length);
          break;
        case "list":
          for (const item of b.items) inline(item);
          break;
        case "blockquote":
          walk(b.children);
          break;
      }
    }
  };
  walk(blocks);
  return lens;
}
