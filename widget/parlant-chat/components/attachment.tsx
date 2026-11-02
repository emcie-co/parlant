// Renders a file reference (inbound or from a tool result). Parlant carries the
// reference; the bytes live in your storage, so we just link/preview the URL.

import type { AttachmentRef } from "../lib/events";

export function Attachment({ attachment }: { attachment: AttachmentRef }) {
  const { url, media_type, filename, caption } = attachment;
  const label = filename ?? caption ?? "file";

  if (media_type?.startsWith("image/")) {
    return (
      <a href={url} target="_blank" rel="noreferrer" className="parlant-attachment parlant-att-image">
        <img src={url} alt={caption ?? label} className="parlant-att-media" />
      </a>
    );
  }

  return (
    <a
      href={url}
      target="_blank"
      rel="noreferrer"
      className="parlant-attachment parlant-att-file"
    >
      <span aria-hidden>📎</span>
      <span className="parlant-att-name">{label}</span>
    </a>
  );
}
