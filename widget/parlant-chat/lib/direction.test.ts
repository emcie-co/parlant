import { describe, expect, it } from "vitest";

import { textDirection } from "./direction";

describe("textDirection", () => {
  it("reads Hebrew and Arabic right to left", () => {
    expect(textDirection("היי! איזה סוג של סוכן?")).toBe("rtl");
    expect(textDirection("مرحبا")).toBe("rtl");
  });

  it("goes by the first letter, past punctuation, digits and Markdown", () => {
    expect(textDirection("**1. מה** the plan?")).toBe("rtl");
    expect(textDirection("> Parlant הוא")).toBe("ltr");
  });

  it("leaves text without letters to the browser", () => {
    expect(textDirection("")).toBe("auto");
    expect(textDirection("123 :)")).toBe("auto");
  });
});
