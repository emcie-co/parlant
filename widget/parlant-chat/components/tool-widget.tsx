// Renders a tool result's frontend-only `metadata` as UI, via an extensible registry.
//
// The convention (see docs/rich-content.md) is that a tool returns, in
// ToolResult.metadata, one of:
//   - `widget:  { type, ...props }`
//   - `widgets: [{ type, ...props }, ...]`
//   - `attachments: [AttachmentRef, ...]`
// Register a renderer per `type`; unknown types fall back to a JSON preview.

import type { ReactNode } from "react";

import type { AttachmentRef, ToolWidget } from "../lib/events";
import { Attachment } from "./attachment";

type WidgetProps = Record<string, any>;
type WidgetRenderer = (props: WidgetProps, widget: ToolWidget) => ReactNode;

const registry = new Map<string, WidgetRenderer>();

/** Register a renderer for a widget `type` returned in a tool's metadata. */
export function registerWidget(type: string, renderer: WidgetRenderer): void {
  registry.set(type, renderer);
}

function renderOne(w: WidgetProps, widget: ToolWidget, key: string): ReactNode {
  if (!w || typeof w.type !== "string") return null;
  const renderer = registry.get(w.type);
  if (renderer) return <div key={key}>{renderer(w, widget)}</div>;
  return (
    <pre
      key={key}
      className="parlant-tool-widget"
    >
      {JSON.stringify(w, null, 2)}
    </pre>
  );
}

export function ToolWidgetView({ widget }: { widget: ToolWidget }): JSX.Element | null {
  const meta = widget.metadata as {
    widget?: WidgetProps;
    widgets?: WidgetProps[];
    attachments?: AttachmentRef[];
  };

  const out: ReactNode[] = [];
  if (meta?.widget) out.push(renderOne(meta.widget, widget, `${widget.toolCallId}:w`));
  meta?.widgets?.forEach((w, i) => out.push(renderOne(w, widget, `${widget.toolCallId}:w${i}`)));
  meta?.attachments?.forEach((a, i) => out.push(<Attachment key={`${widget.toolCallId}:a${i}`} attachment={a} />));

  if (out.length === 0) return null;
  return <div className="parlant-tool-widgets">{out}</div>;
}
