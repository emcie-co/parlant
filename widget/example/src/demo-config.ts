// The set of ChatWidget options the playground lets you tweak live. Shared by both
// example wrappers (guest + authenticated) so the Configure panel drives them uniformly.
// serverUrl/agentId/auth are handled separately (they define identity and remount).

import type { ReasoningMode, StreamSpeed, StreamStyle } from "../../parlant-chat/lib/events";
import type { SoundTheme, ThemeName } from "../../parlant-chat";

export interface WidgetDemoConfig {
  mode: "floating" | "inline" | "input" | "bar";
  /** Color theme (maps to the widget's `theme` prop). */
  theme: ThemeName;
  /** CSS font-family for the widget; empty falls back to the bundled Inter. */
  fontFamily: string;
  streamStyle: StreamStyle;
  streamSpeed: StreamSpeed;
  showAttachments: boolean;
  /** undefined = accept any file; a list restricts the picker. */
  acceptFileTypes?: string;
  showReasoning: boolean;
  reasoningMode: ReasoningMode;
  /** Inline notice when a turn fails; empty falls back to the widget's default. */
  errorMessage: string;
  customerSound: boolean;
  agentSound: boolean;
  /** Which pair of blips the widget plays. */
  soundTheme: SoundTheme;
  /** Delay (ms) before the "thinking" dots fade in. */
  thinkingDelay: number;
  /** Input-mode conversation starters (chips). */
  suggestions: string[];
  showNewChat: boolean;
  showDisclaimer: boolean;
  launcherHint: string;
  preface: string;
  /** Header title; empty falls back to the agent's name. */
  title: string;
  /** Message input placeholder; empty falls back to the widget's default. */
  placeholder: string;
}
