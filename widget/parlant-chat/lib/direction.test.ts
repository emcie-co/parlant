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

  it("goes by the first word on a tie, past punctuation, digits and Markdown", () => {
    expect(textDirection("**1. מה** plan")).toBe("rtl");
    expect(textDirection("> Parlant הוא")).toBe("ltr");
  });

  it("leaves text without letters to the browser", () => {
    expect(textDirection("")).toBe("auto");
    expect(textDirection("123 :)")).toBe("auto");
  });
});
