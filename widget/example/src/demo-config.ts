// The set of ChatWidget options the playground lets you tweak live. Shared by both
// example wrappers (guest + authenticated) so the Configure panel drives them uniformly.
// serverUrl/agentId/auth are handled separately (they define identity and remount).

import type { ReasoningMode, StreamSpeed, StreamStyle } from "../../parlant-chat/lib/events";
import type { AgentAvatar, SoundTheme, ThemeName } from "../../parlant-chat";

export interface WidgetDemoConfig {
  mode: "floating" | "inline" | "input" | "bar";
  openMode: "expanded" | "sidebar";
  persistentSuggestions: string[];
  maxVisibleSuggestions: number;
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
  /** Conversation starters (chips). */
  openingSuggestions: string[];
  showNewChat: boolean;
  showDisclaimer: boolean;
  launcherHint: string;
  preface: string;
  /** Header title; empty falls back to the agent's name. */
  title: string;
  /** Muted line under the header title; empty hides it. */
  subtitle: string;
  /** Header avatar image URL, or a video's MP4; empty shows the agent's initials, or for a
   *  video, the widget's own. */
  avatarUrl: string;
  /** How the avatar fills its frame: a logo, a photo, or a looping video. */
  avatarStyle: "mark" | "photo" | "video";
  /** A live dot on the avatar's corner (photo avatars only). */
  liveIndicator: boolean;
  /** Message input placeholder; empty falls back to the widget's default. */
  placeholder: string;
}

/** The widget's `avatar` for the playground's image, style and live-dot controls. */
export function demoAvatar(c: Pick<WidgetDemoConfig, "avatarUrl" | "avatarStyle" | "liveIndicator">): AgentAvatar | undefined {
  const src = c.avatarUrl.trim();
  if (c.avatarStyle === "video") return src ? { type: "video", src, live: c.liveIndicator } : { type: "video", live: c.liveIndicator };
  if (!src) return undefined;
  return c.avatarStyle === "photo" ? { type: "photo", src, live: c.liveIndicator } : { type: "logo", src };
}
