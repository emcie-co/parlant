/**
 * Build-level UI configuration.
 *
 * MINIMAL_UI is the default build: a chat-only surface with a top nav bar
 * (logo, dashboard link, new-session CTA), no session-list sidebar, and no
 * log viewer or log websocket. Build the full sandbox UI with:
 *
 *     VITE_MINIMAL_UI=false npm run build
 */
export const MINIMAL_UI: boolean = import.meta.env.VITE_MINIMAL_UI !== 'false';
