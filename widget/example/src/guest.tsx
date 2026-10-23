// Example 1: anonymous guest session. No auth configured — the widget uses a
// guest token from the server (stored in localStorage to resume the thread).

import { ChatWidget } from "../../parlant-chat/components/chat-widget";
import { mockUpload } from "./mock-upload";
import type { WidgetDemoConfig } from "./demo-config";

export function GuestExample({
  serverUrl,
  agentId,
  mode,
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
  suggestions,
  showNewChat,
  showDisclaimer,
  launcherHint,
  preface,
  title,
  placeholder,
}: { serverUrl: string; agentId: string } & WidgetDemoConfig) {
  return (
    <ChatWidget
      serverUrl={serverUrl}
      agentId={agentId}
      auth={{ type: "guest" }}
      mode={mode}
      theme={theme}
      fontFamily={fontFamily.trim() && fontFamily.trim() !== "Inter" ? fontFamily.trim() : undefined}
      suggestions={suggestions}
      title={title || undefined}
      placeholder={placeholder || undefined}
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
