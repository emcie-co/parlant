// Renders a tool result's frontend-only `metadata` as UI, via an extensible registry.
//
// The convention (see docs/rich-content.md) is that a tool returns, in
// ToolResult.metadata, one of:
//   - `widget:  { type, ...props }`
//   - `widgets: [{ type, ...props }, ...]`
//   - `attachments: [AttachmentRef, ...]`
// Register a renderer per `type`; unknown types fall back to a JSON preview.
// "choices" is built in (a row of buttons that answers for the visitor) and can be
// overridden like any other type.

import type { ReactNode } from "react";

import type { AttachmentRef, ToolWidget } from "../lib/events";
import { Attachment } from "./attachment";
import { ChoicesWidget, parseChoices } from "./choices-widget";

export type WidgetProps = Record<string, any>;

/** What a renderer can do beyond drawing its props. */
export interface WidgetContext {
  /** Send `text` as the visitor's own chat message, exactly as if they had typed it. */
  sendMessage: (text: string) => void;
  /** No customer message has come after this widget's turn yet, so it is still the open
   *  question. History widgets the visitor has already moved past are not latest. */
  isLatest: boolean;
  /** The text of the first customer message after this widget's turn, if there is one —
   *  e.g. to show which option an earlier answer picked. */
  nextCustomerMessage?: string;
  /** `href` with the widget's `utm` parameters added, under the same rules as the agent's own
   *  links (see `ChatWidgetProps.utm`). Returns `href` unchanged when no `utm` is set. */
  appendUtm: (href: string) => string;
}

export type WidgetRenderer = (
  props: WidgetProps,
  widget: ToolWidget,
  context: WidgetContext,
) => ReactNode;

const registry = new Map<string, WidgetRenderer>();

/** Register a renderer for a widget `type` returned in a tool's metadata. Registering a
 *  built-in type (such as "choices") replaces the built-in renderer. */
export function registerWidget(type: string, renderer: WidgetRenderer): void {
  registry.set(type, renderer);
}

// Built-in: a "pick one" question. Malformed metadata renders nothing.
registry.set("choices", (props, widget, context) => {
  const choices = parseChoices(props);
  if (!choices) return null;
  return <ChoicesWidget choices={choices} toolCallId={widget.toolCallId} context={context} />;
});

/** Outside a chat transcript (e.g. a standalone ToolWidgetView) nothing can be sent. */
const INERT_CONTEXT: WidgetContext = {
  sendMessage: () => {},
  isLatest: false,
  appendUtm: (href) => href,
};

function renderOne(
  w: WidgetProps,
  widget: ToolWidget,
  context: WidgetContext,
  key: string,
): ReactNode {
  if (!w || typeof w.type !== "string") return null;
  const renderer = registry.get(w.type);
  if (renderer) {
    const content = renderer(w, widget, context);
    // A renderer that declines (null/undefined/false) leaves no empty wrapper behind.
    if (content == null || content === false) return null;
    return <div key={key}>{content}</div>;
  }
  return (
    <pre
      key={key}
      className="parlant-tool-widget"
    >
      {JSON.stringify(w, null, 2)}
    </pre>
  );
}

export function ToolWidgetView({
  widget,
  context = INERT_CONTEXT,
}: {
  widget: ToolWidget;
  context?: WidgetContext;
}): JSX.Element | null {
  const meta = widget.metadata as {
    widget?: WidgetProps;
    widgets?: WidgetProps[];
    attachments?: AttachmentRef[];
  };

  const out: ReactNode[] = [];
  if (meta?.widget) out.push(renderOne(meta.widget, widget, context, `${widget.toolCallId}:w`));
  if (Array.isArray(meta?.widgets))
    meta.widgets.forEach((w, i) =>
      out.push(renderOne(w, widget, context, `${widget.toolCallId}:w${i}`)),
    );
  if (Array.isArray(meta?.attachments))
    meta.attachments.forEach((a, i) =>
      out.push(<Attachment key={`${widget.toolCallId}:a${i}`} attachment={a} />),
    );

  const rendered = out.filter((node) => node != null);
  if (rendered.length === 0) return null;
  return <div className="parlant-tool-widgets">{rendered}</div>;
}
