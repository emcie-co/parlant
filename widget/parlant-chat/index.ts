// Public API of the Parlant chat widget. Copy this whole folder into your app
// (e.g. src/parlant-chat/) and import from it:
//
//   import { ChatWidget } from "./parlant-chat";
//
// No Tailwind, no build config — styles.css is imported by the widget itself.

export { ChatWidget } from "./components/chat-widget";
export type { ChatWidgetProps, ThemeName } from "./components/chat-widget";
export { registerWidget } from "./components/tool-widget";
export { ParlantTransport } from "./lib/transport";
export {
  SOUND_THEMES,
  SOUND_THEME_NAMES,
  DEFAULT_SOUND_THEME,
  playAgentSound,
  playCustomerSound,
} from "./lib/sounds";
export type { SoundTheme, SoundThemeDefinition, SoundTone } from "./lib/sounds";
export type {
  AttachmentRef,
  ReasoningMode,
  StreamSpeed,
  StreamStyle,
} from "./lib/events";
