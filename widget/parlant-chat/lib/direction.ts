// The direction a message reads in, from its first letter: right to left for Hebrew,
// Arabic and the other RTL scripts. The bubble sets it explicitly rather than relying on
// dir="auto", because a streamed reply mounts empty and browsers don't reliably
// re-resolve "auto" as its words arrive.
const RTL = /[\p{Script=Hebrew}\p{Script=Arabic}\p{Script=Syriac}\p{Script=Thaana}\p{Script=Nko}]/u;
const LETTER = /\p{L}/u;

export function textDirection(text: string): "rtl" | "ltr" | "auto" {
  const first = LETTER.exec(text)?.[0];
  if (!first) return "auto";
  return RTL.test(first) ? "rtl" : "ltr";
}
