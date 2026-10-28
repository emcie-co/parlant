// @vitest-environment jsdom
import { describe, expect, it, vi } from "vitest";

import {
  DEFAULT_SOUND_THEME,
  SOUND_THEMES,
  SOUND_THEME_NAMES,
  type SoundTheme,
  type SoundTone,
  playAgentSound,
  playCustomerSound,
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
  "velvet",
];

describe("sound themes", () => {
  it("offers all thirteen themes we ship, in order", () => {
    expect(SOUND_THEME_NAMES).toEqual(SHIPPED);
    expect(Object.keys(SOUND_THEMES)).toEqual(SHIPPED);
  });

  it("test_that_the_widget_defaults_to_velvet", () => {
    expect(DEFAULT_SOUND_THEME).toBe("velvet");
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
        expect(tone.freq, `${name} freq`).toBeGreaterThan(name === "velvet" ? 20 : 100);
        expect(tone.freq, `${name} freq`).toBeLessThanOrEqual(ceiling);
        expect(tone.peakGain, `${name} gain`).toBeGreaterThan(0);
        expect(tone.peakGain, `${name} gain`).toBeLessThanOrEqual(name === "velvet" ? .25 : .08);
        expect(tone.durationMs, `${name} duration`).toBeGreaterThan(0);
        expect(tone.durationMs, `${name} duration`).toBeLessThanOrEqual(500);
        if (tone.to !== undefined) {
          expect(tone.to, `${name} sweep target`).toBeGreaterThan(name === "velvet" ? 20 : 100);
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


describe("Velvet playback", () => {
  it("test_that_sent_uses_the_soft_sent_cue_and_every_reply_uses_the_soft_first_incoming_cue_with_bass_and_echo", () => {
    const gains: { linearRampToValueAtTime: ReturnType<typeof vi.fn> }[] = [];
    const oscillators: { frequency: { setValueAtTime: ReturnType<typeof vi.fn> }; start: ReturnType<typeof vi.fn> }[] = [];
    class Audio {
      currentTime = 1;
      state = "running";
      destination = {};
      createGain() {
        const gain = { setValueAtTime: vi.fn(), linearRampToValueAtTime: vi.fn(), exponentialRampToValueAtTime: vi.fn() };
        gains.push(gain);
        return { gain, connect: vi.fn() };
      }
      createOscillator() {
        const osc = { frequency: { setValueAtTime: vi.fn(), exponentialRampToValueAtTime: vi.fn() }, type: "sine", connect: vi.fn(), start: vi.fn(), stop: vi.fn() };
        oscillators.push(osc);
        return osc;
      }
    }
    vi.stubGlobal("AudioContext", Audio);
    try {
      playCustomerSound();
      const sentCount = oscillators.length;
      expect(sentCount).toBe(7);
      expect(oscillators[0].frequency.setValueAtTime).toHaveBeenCalledWith(65, 1);
      expect(oscillators[6].frequency.setValueAtTime).toHaveBeenCalledWith(1305 * .78, 1);
      expect(gains[0].linearRampToValueAtTime).toHaveBeenCalledWith(.221, 1.018);
      expect(oscillators[3].start).toHaveBeenCalledWith(1.13);
      expect(oscillators[4].start).toHaveBeenCalledWith(1.26);
      expect(oscillators[5].start.mock.calls[0][0]).toBeCloseTo(1.39);
      for (let reply = 0; reply < 2; reply++) {
        const offset = oscillators.length;
        playAgentSound();
        expect(oscillators.length - offset).toBe(9);
        expect(oscillators[offset].frequency.setValueAtTime).toHaveBeenCalledWith(90, 1);
        expect(oscillators[offset + 6].frequency.setValueAtTime).toHaveBeenCalledWith(1960 * .78, 1);
        expect(oscillators[offset + 7].start).toHaveBeenCalledWith(1.0216);
        expect(oscillators[offset + 8].frequency.setValueAtTime).toHaveBeenCalledWith(3145 * .78, 1.027);
      }
    } finally { vi.unstubAllGlobals(); }
  });
});
