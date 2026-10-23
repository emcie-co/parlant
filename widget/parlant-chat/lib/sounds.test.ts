import { describe, expect, it } from "vitest";

import {
  DEFAULT_SOUND_THEME,
  SOUND_THEMES,
  SOUND_THEME_NAMES,
  type SoundTheme,
  type SoundTone,
} from "./sounds";

// The offering, pinned. Adding or dropping a theme changes what every embedder can
// choose from, so it should have to change this list too.
const SHIPPED: SoundTheme[] = [
  "glass",
  "drop",
  "marimba",
  "thock",
  "tabla",
  "snap",
  "chip",
  "bloop",
  "anvil",
  "chalice",
  "shard",
  "halo",
];

describe("sound themes", () => {
  it("offers exactly the twelve themes we ship, in order", () => {
    expect(SOUND_THEME_NAMES).toEqual(SHIPPED);
    expect(Object.keys(SOUND_THEMES)).toEqual(SHIPPED);
  });

  it("defaults to glass", () => {
    expect(DEFAULT_SOUND_THEME).toBe("glass");
    expect(SOUND_THEMES[DEFAULT_SOUND_THEME]).toBeDefined();
  });

  it("gives every theme a label and both directions", () => {
    for (const name of SOUND_THEME_NAMES) {
      const theme = SOUND_THEMES[name];
      expect(theme.label, `${name} label`).toBeTruthy();
      expect(theme.customer.length, `${name} customer`).toBeGreaterThan(0);
      expect(theme.agent.length, `${name} agent`).toBeGreaterThan(0);
    }
  });

  it("keeps every tone within audible, gentle bounds", () => {
    // A sound the widget plays unprompted must never be startling: the gain ceiling
    // keeps it a blip rather than a beep, and the pitch range keeps it a tone. On a
    // noise burst `freq` is a filter cutoff rather than a pitch, so it may sit far
    // higher without ever being a shrill note.
    for (const name of SOUND_THEME_NAMES) {
      const tones: SoundTone[] = [...SOUND_THEMES[name].customer, ...SOUND_THEMES[name].agent];
      for (const tone of tones) {
        const ceiling = tone.type === "noise" ? 12000 : 4000;
        expect(tone.freq, `${name} freq`).toBeGreaterThan(100);
        expect(tone.freq, `${name} freq`).toBeLessThanOrEqual(ceiling);
        expect(tone.peakGain, `${name} gain`).toBeGreaterThan(0);
        expect(tone.peakGain, `${name} gain`).toBeLessThanOrEqual(0.08);
        expect(tone.durationMs, `${name} duration`).toBeGreaterThan(0);
        expect(tone.durationMs, `${name} duration`).toBeLessThanOrEqual(500);
        if (tone.to !== undefined) {
          expect(tone.to, `${name} sweep target`).toBeGreaterThan(100);
          expect(tone.to, `${name} sweep target`).toBeLessThanOrEqual(4000);
        }
      }
    }
  });

  it("sweeps tabla and bloop in opposite directions per side", () => {
    // Both themes say which way the message went with the glide itself: down when it
    // leaves, up when one arrives. A sweep that lost its direction would still play,
    // and would silently stop carrying that meaning.
    for (const name of ["tabla", "bloop"] as const) {
      const [out] = SOUND_THEMES[name].customer;
      const [into] = SOUND_THEMES[name].agent;
      expect(out.to, `${name} out sweep`).toBeLessThan(out.freq);
      expect(into.to, `${name} in sweep`).toBeGreaterThan(into.freq);
    }
  });

  it("builds snap and shard on noise, which is what makes them electric", () => {
    for (const name of ["snap", "shard"] as const) {
      const tones: SoundTone[] = [...SOUND_THEMES[name].customer, ...SOUND_THEMES[name].agent];
      expect(
        tones.some((t) => t.type === "noise"),
        name,
      ).toBe(true);
    }
  });
});
