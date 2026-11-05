// The direction a message reads in: right to left when most of its words are in Hebrew,
// Arabic or another RTL script, so a Hebrew reply that opens with an English name ("Parlant
// הוא…") still reads right to left. A tie goes by the first word. The bubble sets it
// explicitly rather than relying on dir="auto", because a streamed reply mounts empty and
// browsers don't reliably re-resolve "auto" as its words arrive; "auto" also goes by the
// first letter alone.
const RTL = /[\p{Script=Hebrew}\p{Script=Arabic}\p{Script=Syriac}\p{Script=Thaana}\p{Script=Nko}]/u;
// A word, by its first letter: Markdown, digits and punctuation around it don't count.
const WORD = /\p{L}[\p{L}\p{M}'’\-]*/gu;

export function textDirection(text: string): "rtl" | "ltr" | "auto" {
  let rtl = 0;
  let ltr = 0;
  let first: "rtl" | "ltr" | null = null;
  for (const [word] of text.matchAll(WORD)) {
    const direction = RTL.test(word[0]) ? "rtl" : "ltr";
    first ??= direction;
    if (direction === "rtl") rtl++;
    else ltr++;
  }
  if (!first) return "auto";
  if (rtl === ltr) return first;
  return rtl > ltr ? "rtl" : "ltr";
}
