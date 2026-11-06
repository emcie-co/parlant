import { describe, expect, it } from "vitest";

import { textDirection } from "./direction";

describe("textDirection", () => {
  it("reads Hebrew and Arabic right to left", () => {
    expect(textDirection("היי! איזה סוג של סוכן?")).toBe("rtl");
    expect(textDirection("مرحبا")).toBe("rtl");
  });

  it("reads a Hebrew reply that opens with English words right to left", () => {
    expect(textDirection("Parlant הוא פריימוורק לבניית סוכנים")).toBe("rtl");
    expect(textDirection("**Parlant Dashboard** מאפשר לך לנהל את הסוכן שלך")).toBe("rtl");
  });

  it("reads an English reply with a Hebrew word left to right", () => {
    expect(textDirection("In Hebrew, hello is שלום.")).toBe("ltr");
  });

  it("reads a tie right to left, past punctuation, digits and Markdown", () => {
    expect(textDirection("**1. מה** plan")).toBe("rtl");
    expect(textDirection("> Parlant הוא")).toBe("rtl");
    expect(textDirection("Parlant Dashboard Pro זה התוכנית שלכם")).toBe("rtl");
  });

  it("counts a URL, a domain or a prefixed word once", () => {
    expect(textDirection("אפשר לקרוא על זה כאן https://www.parlant.io/docs/pricing")).toBe("rtl");
    expect(textDirection("כל הפרטים ב-parlant.io")).toBe("rtl");
    expect(textDirection("See https://www.parlant.io/he/דף")).toBe("ltr");
  });

  it("leaves code out of the count, even while it streams in", () => {
    expect(textDirection("הריצו `pip install parlant` ואז")).toBe("rtl");
    expect(textDirection("הריצו את זה:\n```bash\npip install parlant --upgrade")).toBe("rtl");
  });

  it("leaves text without letters to the browser", () => {
    expect(textDirection("")).toBe("auto");
    expect(textDirection("123 :)")).toBe("auto");
  });
});
