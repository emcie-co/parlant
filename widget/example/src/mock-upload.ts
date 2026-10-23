import type { AttachmentRef } from "../../parlant-chat/lib/events";

// Demo-only "upload": reads the file into a self-contained data: URL. That keeps the
// bytes inside the reference, so the bundled sample agent can actually fetch the image
// (a blob: URL would only work in this browser). In production, upload to your own
// storage and return that URL instead — Parlant carries the reference, not the bytes.
export async function mockUpload(file: File): Promise<AttachmentRef> {
  const url = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
  return { url, media_type: file.type, filename: file.name };
}
