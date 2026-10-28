// Tiny message blips, synthesized with the Web Audio API — no audio files, no
// dependencies. A single AudioContext is created lazily on first use (browsers only allow
// it after a user gesture, which sending/opening the chat provides).
//
// A theme is just a pair of short note sequences: one for the customer's outgoing message,
// one for the agent's reply. Add your own by extending SOUND_THEMES.

/** One note: a shaped burst of either an oscillator or filtered noise. */
export interface SoundTone {
  /** Pitch in Hz for an oscillator; the high-pass cutoff for a noise burst. */
  freq: number;
  /** How long the note rings, including its decay. */
  durationMs: number;
  /** Loudest point of the envelope — keep these small; the sound plays unprompted. */
  peakGain: number;
  /** Oscillator shape, or "noise" for a high-passed white-noise burst (default "sine").
   *  Noise is what makes a tick sound electric rather than pitched — an oscillator that
   *  short and that high just reads as a chirp. */
  type?: OscillatorType | "noise";
  /** Glide to this pitch over the note's length. Oscillators only — a sweep is what
   *  lets a theme say which way the message went: down on the way out, up on the way in. */
  to?: number;
  /** Start offset within the sound, for two-note themes (default 0). */
  delayMs?: number;
  /** Optional attack length in milliseconds; useful for a gentler bass swell. */
  attackMs?: number;
}

export interface SoundThemeDefinition {
  /** Human-readable name, for pickers. */
  label: string;
  /** Played when the customer's own message goes out. */
  customer: SoundTone[];
  /** Played when an agent reply arrives. */
  agent: SoundTone[];
}

/** Low, rounded body and progressively quieter reflections under Velvet's Soft tone. */
function velvetBody(pitch: number): SoundTone[] {
  return [
    { freq: pitch, to: pitch * .62, durationMs: 340, peakGain: 0.221, attackMs: 18 },
    { freq: pitch * .65, to: pitch * .55, durationMs: 380, peakGain: 0.0715, attackMs: 25 },
    { freq: pitch * 2.1, to: pitch * 1.6, durationMs: 120, peakGain: 0.02925, attackMs: 14 },
    { freq: pitch * .92, to: pitch * .6, durationMs: 240, peakGain: 0.065, attackMs: 22, delayMs: 130 },
    { freq: pitch * .84, to: pitch * .56, durationMs: 210, peakGain: 0.0325, attackMs: 26, delayMs: 260 },
    { freq: pitch * .78, to: pitch * .52, durationMs: 180, peakGain: 0.01625, attackMs: 28, delayMs: 390 },
  ];
}

/** Built-in sound themes, selected with the widget's `soundTheme` prop. */
export const SOUND_THEMES = {
  // High, thin sines with a faint overtone — a struck wine glass.
  glass: {
    label: "Glass",
    customer: [
      { freq: 1318, durationMs: 260, peakGain: 0.03 },
      { freq: 1976, durationMs: 200, peakGain: 0.012 },
    ],
    agent: [
      { freq: 1568, durationMs: 300, peakGain: 0.032 },
      { freq: 2349, durationMs: 220, peakGain: 0.012 },
    ],
  },
  // A single round sine, low and unhurried.
  drop: {
    label: "Drop",
    customer: [{ freq: 330, durationMs: 140, peakGain: 0.045 }],
    agent: [{ freq: 440, durationMs: 190, peakGain: 0.05 }],
  },
  // Wooden: a fundamental with its octave stacked on top and a fast decay, which is what
  // makes a struck bar sound struck.
  marimba: {
    label: "Marimba",
    customer: [
      { freq: 392, durationMs: 180, peakGain: 0.045 },
      { freq: 784, durationMs: 140, peakGain: 0.02 },
    ],
    agent: [
      { freq: 587, durationMs: 200, peakGain: 0.05 },
      { freq: 1175, durationMs: 150, peakGain: 0.022 },
    ],
  },
  // Deep and soft, like a keyboard switch — closer to felt than heard.
  thock: {
    label: "Thock",
    customer: [
      { freq: 196, durationMs: 120, peakGain: 0.055 },
      { freq: 392, durationMs: 80, peakGain: 0.015 },
    ],
    agent: [
      { freq: 262, durationMs: 140, peakGain: 0.055 },
      { freq: 523, durationMs: 90, peakGain: 0.015 },
    ],
  },
  // A drum-like pitch bend: down as the message leaves, up as one arrives.
  tabla: {
    label: "Tabla",
    customer: [{ freq: 400, durationMs: 90, peakGain: 0.05, to: 180 }],
    agent: [{ freq: 300, durationMs: 110, peakGain: 0.05, to: 520 }],
  },
  // Fingers snapping: a noise tick with a pitched ping inside the reply.
  snap: {
    label: "Snap",
    customer: [{ freq: 6000, durationMs: 22, peakGain: 0.06, type: "noise" }],
    agent: [
      { freq: 7500, durationMs: 20, peakGain: 0.06, type: "noise" },
      { freq: 1568, durationMs: 60, peakGain: 0.02, delayMs: 18 },
    ],
  },
  // Bright, plastic triangle pair — high and unmistakably digital.
  chip: {
    label: "Chip",
    customer: [
      { freq: 2093, durationMs: 40, peakGain: 0.03, type: "triangle" },
      { freq: 1568, durationMs: 50, peakGain: 0.03, type: "triangle", delayMs: 35 },
    ],
    agent: [
      { freq: 1568, durationMs: 40, peakGain: 0.03, type: "triangle" },
      { freq: 2093, durationMs: 60, peakGain: 0.03, type: "triangle", delayMs: 35 },
    ],
  },
  // A fat sine glide, falling out and rising in.
  bloop: {
    label: "Bloop",
    customer: [{ freq: 700, durationMs: 120, peakGain: 0.05, to: 300 }],
    agent: [{ freq: 300, durationMs: 140, peakGain: 0.05, to: 700 }],
  },
  // Struck metal: a short square with a ringing tail above it.
  anvil: {
    label: "Anvil",
    customer: [
      { freq: 1318, durationMs: 60, peakGain: 0.028, type: "square" },
      { freq: 1760, durationMs: 200, peakGain: 0.018, type: "square", delayMs: 20 },
    ],
    agent: [
      { freq: 1568, durationMs: 60, peakGain: 0.03, type: "square" },
      { freq: 2093, durationMs: 220, peakGain: 0.02, type: "square", delayMs: 20 },
    ],
  },
  // A slow bloom of triangle notes; the reply adds a third. The most ceremonial of the set.
  chalice: {
    label: "Chalice",
    customer: [
      { freq: 588, durationMs: 380, peakGain: 0.03, type: "triangle" },
      { freq: 392, durationMs: 420, peakGain: 0.018, type: "triangle", delayMs: 60 },
    ],
    agent: [
      { freq: 392, durationMs: 400, peakGain: 0.035, type: "triangle" },
      { freq: 588, durationMs: 420, peakGain: 0.02, type: "triangle", delayMs: 40 },
      { freq: 784, durationMs: 380, peakGain: 0.012, type: "triangle", delayMs: 80 },
    ],
  },
  // Electric: a high-passed noise tick with a glassy ping riding on it.
  shard: {
    label: "Shard",
    customer: [
      { freq: 7000, durationMs: 26, peakGain: 0.045, type: "noise" },
      { freq: 2637, durationMs: 120, peakGain: 0.018, delayMs: 10 },
    ],
    agent: [
      { freq: 8000, durationMs: 28, peakGain: 0.045, type: "noise" },
      { freq: 3136, durationMs: 140, peakGain: 0.02, delayMs: 10 },
    ],
  },
  // Stacked high sines arriving in sequence — a soft shimmer rather than an event.
  halo: {
    label: "Halo",
    customer: [
      { freq: 2093, durationMs: 300, peakGain: 0.02 },
      { freq: 1568, durationMs: 340, peakGain: 0.022, delayMs: 60 },
    ],
    agent: [
      { freq: 1046, durationMs: 340, peakGain: 0.03 },
      { freq: 1568, durationMs: 380, peakGain: 0.02, delayMs: 60 },
      { freq: 2093, durationMs: 300, peakGain: 0.012, delayMs: 120 },
    ],
  },
  // Soft, compact chime over a deeper, gently swelling thud and fading echoes.
  // Outgoing uses the sent cue; every reply uses the fuller first-incoming cue.
  velvet: {
    label: "Velvet",
    customer: [
      ...velvetBody(65),
      { freq: 1305 * .78, to: 1295 * .78, durationMs: 59.4, peakGain: 0.156, attackMs: 2 },
    ],
    agent: [
      ...velvetBody(90),
      { freq: 1960 * .78, durationMs: 48.6, peakGain: 0.156, attackMs: 2 },
      { freq: 1960 * .78, durationMs: 46.8, peakGain: 0.04992, attackMs: 1.44, delayMs: 21.6 },
      { freq: 3145 * .78, to: 3090 * .78, durationMs: 63, peakGain: 0.039, attackMs: 1.8, delayMs: 27 },
    ],
  },
} satisfies Record<string, SoundThemeDefinition>;

export type SoundTheme = keyof typeof SOUND_THEMES;

/** Every built-in theme, in picker order. */
export const SOUND_THEME_NAMES = Object.keys(SOUND_THEMES) as SoundTheme[];

export const DEFAULT_SOUND_THEME: SoundTheme = "velvet";

let ctx: AudioContext | null = null;

function context(): AudioContext | null {
  if (typeof window === "undefined") return null;
  const AC = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!AC) return null;
  if (!ctx) ctx = new AC();
  return ctx;
}

// Half a second of white noise, generated once and reused by every noise burst.
let noise: AudioBuffer | null = null;

function noiseBuffer(ac: AudioContext): AudioBuffer {
  if (!noise) {
    noise = ac.createBuffer(1, Math.floor(ac.sampleRate * 0.5), ac.sampleRate);
    const samples = noise.getChannelData(0);
    for (let i = 0; i < samples.length; i++) samples[i] = Math.random() * 2 - 1;
  }
  return noise;
}

/** One note: quick attack, exponential decay, so it reads as a soft chime rather than a beep. */
function tone(ac: AudioContext, startAt: number, spec: SoundTone): void {
  const gain = ac.createGain();

  const seconds = spec.durationMs / 1000;
  const end = startAt + seconds;
  // Short percussive notes need a shorter attack, or the ramp swallows the note.
  const attack = spec.attackMs === undefined
    ? Math.min(0.012, seconds * 0.25)
    : Math.min(spec.attackMs / 1000, seconds * .9);

  gain.gain.setValueAtTime(0, startAt);
  gain.gain.linearRampToValueAtTime(spec.peakGain, startAt + attack);
  gain.gain.exponentialRampToValueAtTime(0.0001, end);
  gain.connect(ac.destination);

  if (spec.type === "noise") {
    // White noise through a steep high-pass: what's left is all treble, which is how a
    // hi-hat (and every "electric" UI tick) is made.
    const source = ac.createBufferSource();
    source.buffer = noiseBuffer(ac);
    const filter = ac.createBiquadFilter();
    filter.type = "highpass";
    filter.frequency.value = spec.freq;
    filter.Q.value = 0.7;

    source.connect(filter).connect(gain);
    source.start(startAt);
    source.stop(end);
    return;
  }

  const osc = ac.createOscillator();
  osc.type = spec.type ?? "sine";
  osc.frequency.setValueAtTime(spec.freq, startAt);
  if (spec.to !== undefined) osc.frequency.exponentialRampToValueAtTime(spec.to, end);
  osc.connect(gain);
  osc.start(startAt);
  osc.stop(end);
}

function play(tones: SoundTone[]): void {
  const ac = context();
  if (!ac) return;
  if (ac.state === "suspended") void ac.resume();

  const now = ac.currentTime;
  for (const spec of tones) tone(ac, now + (spec.delayMs ?? 0) / 1000, spec);
}

function themeOrDefault(theme: SoundTheme | undefined): SoundThemeDefinition {
  return SOUND_THEMES[theme ?? DEFAULT_SOUND_THEME] ?? SOUND_THEMES[DEFAULT_SOUND_THEME];
}

/** Played when the customer's own message goes out. */
export function playCustomerSound(theme?: SoundTheme): void {
  play(themeOrDefault(theme).customer);
}

/** Played when an agent reply arrives. */
export function playAgentSound(theme?: SoundTheme): void {
  play(themeOrDefault(theme).agent);
}
