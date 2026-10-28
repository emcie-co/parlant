// Example 2: signed-in customer.
//
// This playground is intentionally a manual harness: paste a customer identity key
// and a Parlant customer JWT into the UI. In a real app, identityKey comes from your
// signed-in user and getToken calls your backend (see the README).

import { ChatWidget } from "../../parlant-chat/components/chat-widget";
import { mockUpload } from "./mock-upload";
import type { WidgetDemoConfig } from "./demo-config";

export function AuthenticatedExample({
  serverUrl,
  agentId,
  identityKey,
  token,
  mode,
  openMode,
  persistentSuggestions,
  maxVisibleSuggestions,
  theme,
  fontFamily,
  streamStyle,
  streamSpeed,
  showAttachments,
  acceptFileTypes,
  showReasoning,
  reasoningMode,
  errorMessage,
  customerSound,
  agentSound,
  soundTheme,
  thinkingDelay,
  openingSuggestions,
  showNewChat,
  showDisclaimer,
  launcherHint,
  preface,
  title,
  placeholder,
}: { serverUrl: string; agentId: string; identityKey: string; token: string } & WidgetDemoConfig) {
  return (
    <ChatWidget
      serverUrl={serverUrl}
      agentId={agentId}
      mode={mode}
      openMode={openMode}
      persistentSuggestions={persistentSuggestions}
      maxVisibleSuggestions={maxVisibleSuggestions}
      theme={theme}
      fontFamily={fontFamily.trim() && fontFamily.trim() !== "Inter" ? fontFamily.trim() : undefined}
      openingSuggestions={openingSuggestions}
      title={title || undefined}
      placeholder={placeholder || undefined}
      auth={{ type: "customer", identityKey, getToken: async () => token }}
      onUpload={mockUpload}
      showAttachments={showAttachments}
      acceptFileTypes={acceptFileTypes}
      showReasoning={showReasoning}
      reasoningMode={reasoningMode}
      errorMessage={errorMessage || undefined}
      customerSound={customerSound}
      agentSound={agentSound}
      soundTheme={soundTheme}
      thinkingDelay={thinkingDelay}
      showNewChat={showNewChat}
      launcherHint={launcherHint}
      streamStyle={streamStyle}
      streamSpeed={streamSpeed}
      preface={preface}
      disclaimer={
        showDisclaimer ? (
          <>
            By chatting with us, you agree to our{" "}
            <a href="about:blank" className="underline">
              Privacy Policy
            </a>
          </>
        ) : undefined
      }
    />
  );
}
