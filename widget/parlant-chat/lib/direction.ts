// The direction a message reads in: right to left when most of its words are in Hebrew,
// Arabic or another RTL script, so a Hebrew reply that opens with an English name ("Parlant
// הוא…") still reads right to left. A tie reads right to left too: English text rarely holds
// as many Hebrew words as English ones, while a short Hebrew reply often names a product. The
// bubble sets it explicitly rather than relying on dir="auto", because a streamed reply
// mounts empty and browsers don't reliably re-resolve "auto" as its words arrive; "auto"
// also goes by the first letter alone.
const RTL = /[\p{Script=Hebrew}\p{Script=Arabic}\p{Script=Syriac}\p{Script=Thaana}\p{Script=Nko}]/u;
const LETTER = /\p{L}/u;
// Code is always left to right and says nothing about the language around it, including a
// fence or span still streaming in.
const CODE = /```[\s\S]*?(?:```|$)|`[^`\n]*`/g;

/** Words are split by spaces, so a URL, a domain or "ה-Dashboard" counts once, by its first
 *  letter. Markdown, digits and punctuation around a word don't count. */
export function textDirection(text: string): "rtl" | "ltr" | "auto" {
  let rtl = 0;
  let ltr = 0;
  for (const [word] of text.replace(CODE, " ").matchAll(/\S+/g)) {
    const letter = LETTER.exec(word);
    if (!letter) continue;
    if (RTL.test(letter[0])) rtl++;
    else ltr++;
  }
  if (!rtl && !ltr) return "auto";
  return rtl >= ltr ? "rtl" : "ltr";
}
