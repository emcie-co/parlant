// Example: teach the widget how to render a custom tool output as UI.
//
// Agent Tom's `describe_attached_image` tool returns, in its ToolResult metadata, a
// `widget: { type: "picture_description", ... }` payload. The widget core knows nothing
// about that type — the APP registers a renderer for it here. This is the whole rich-
// content contract: tools emit typed JSON, the frontend decides how it looks.
//
// Import this file once (for its side effect) before the widget mounts — see main.tsx.

import { registerWidget } from "../../parlant-chat/components/tool-widget";

interface PictureDescription {
  type: "picture_description";
  filename?: string;
  imageUrl?: string;
  description?: string;
}

registerWidget("picture_description", (props) => {
  const { filename, imageUrl, description } = props as PictureDescription;
  return (
    <figure className="parlant-picture-card overflow-hidden rounded-xl border border-[#eeeecc] bg-[#ffffee]">
      {imageUrl && (
        <img
          src={imageUrl}
          alt={filename ?? "attached image"}
          className="max-h-56 w-full object-cover"
        />
      )}
      <figcaption className="flex flex-col gap-1 p-3">
        <span className="inline-flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
          <svg
            width="13"
            height="13"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden
          >
            <rect x="3" y="3" width="18" height="18" rx="2" />
            <circle cx="9" cy="9" r="2" />
            <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
          </svg>
          Picture description
        </span>
        {filename && <span className="truncate text-[10px] text-muted-foreground/80">{filename}</span>}
        {description && <p className="text-xs leading-relaxed text-foreground">{description}</p>}
      </figcaption>
    </figure>
  );
});
