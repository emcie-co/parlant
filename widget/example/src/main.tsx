import { type ReactNode, StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";

import "../../parlant-chat/styles.css";
import "./index.css";
import type { ReasoningMode, StreamSpeed, StreamStyle } from "../../parlant-chat/lib/events";
import type { SoundTheme, ThemeName } from "../../parlant-chat";
import { SOUND_THEMES, SOUND_THEME_NAMES, playAgentSound } from "../../parlant-chat";
import { ParlantTransport } from "../../parlant-chat/lib/transport";
import type { WidgetDemoConfig } from "./demo-config";
import { AuthenticatedExample } from "./authenticated";
import { GuestExample } from "./guest";
import { playgroundSteps } from "./steps";
// Side-effect import: registers the "picture_description" renderer Agent Tom emits.
import "./widgets";

type Auth = "guest" | "authenticated";
type Mode = "floating" | "inline" | "input" | "bar";
type Health = "checking" | "online" | "offline";

const STREAM_STYLES: { value: StreamStyle; label: string }[] = [
  { value: "instant", label: "Instant" },
  { value: "fade", label: "Soft fade" },
  { value: "rise", label: "Fade & rise" },
  { value: "blur", label: "Focus pull" },
  { value: "letter", label: "Letter fade" },
  { value: "warm", label: "Warm-up" },
];

const THEMES: { value: ThemeName; label: string }[] = [
  { value: "monochrome", label: "Monochrome (light/dark)" },
  { value: "parlant", label: "Parlant green" },
  { value: "ocean", label: "Ocean (dark)" },
  { value: "amethyst", label: "Amethyst (dark)" },
  { value: "sand", label: "Sand (light)" },
];

const SOUND_THEME_OPTIONS: { value: SoundTheme; label: string }[] = SOUND_THEME_NAMES.map(
  (value) => ({ value, label: SOUND_THEMES[value].label }),
);

const inputCls =
  "w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-400 outline-none transition focus:border-emerald-500/50 focus:ring-2 focus:ring-emerald-500/20";

// Honest page-level reachability: poll the applied server's /healthz (same check the
// widget uses for its offline banner) so the status here can't claim "Connected" for a
// server that isn't actually up. Quick recheck while down, lazy while healthy.
function usePageHealth(url: string, agentId: string, enabled: boolean): Health {
  const [status, setStatus] = useState<Health>("checking");

  useEffect(() => {
    if (!enabled) return;
    setStatus("checking");

    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const transport = new ParlantTransport({ serverUrl: url, agentId });

    const tick = async () => {
      const ok = await transport.checkHealth();
      if (cancelled) return;
      setStatus(ok ? "online" : "offline");
      timer = setTimeout(tick, ok ? 20000 : 4000);
    };
    tick();

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [url, agentId, enabled]);

  return status;
}

function App() {
  const [auth, setAuth] = useState<Auth>("guest");
  const [mode, setMode] = useState<Mode>("input");
  const [theme, setTheme] = useState<ThemeName>("parlant");
  const [fontFamily, setFontFamily] = useState("Inter");
  const [streamStyle, setStreamStyle] = useState<StreamStyle>("rise");
  const [streamSpeed, setStreamSpeed] = useState<StreamSpeed>("normal");
  const [showAttachments, setShowAttachments] = useState(true);
  const [imagesOnly, setImagesOnly] = useState(true);
  const [showReasoning, setShowReasoning] = useState(false);
  const [reasoningMode, setReasoningMode] = useState<ReasoningMode>("transient");
  const [errorMessage, setErrorMessage] = useState(
    "The agent ran into an error and couldn't finish this reply. Please try again.",
  );
  const [customerSound, setCustomerSound] = useState(true);
  const [agentSound, setAgentSound] = useState(true);
  const [soundTheme, setSoundTheme] = useState<SoundTheme>("glass");
  const [placeholder, setPlaceholder] = useState("");
  const [thinkingDelay, setThinkingDelay] = useState(1000);
  const [suggestions, setSuggestions] = useState<string[]>([
    "What can you do?",
    "Describe a photo I send you",
    "Give me a fun fact",
  ]);
  const [showNewChat, setShowNewChat] = useState(true);
  const [showDisclaimer, setShowDisclaimer] = useState(true);
  const [launcherHint, setLauncherHint] = useState("Try me");
  const [title, setTitle] = useState("");
  const [preface, setPreface] = useState(
    "Hi there! Try attaching a photo and asking me to describe what's in it.",
  );

  // Playground-only convenience: the widget itself never decodes tokens (identityKey is an
// explicit prop), but this harness has no sign-in system, so it borrows the pasted
// token's `sub` claim as the identity key.
function subFromJwt(token: string): string {
  try {
    const payload = token.split(".")[1];
    const json = JSON.parse(atob(payload.replace(/-/g, "+").replace(/_/g, "/")));
    return typeof json.sub === "string" ? json.sub : "";
  } catch {
    return "";
  }
}

// Connection is set live in the UI. Defaults match the bundled sample agent
  // ("Agent Tom" on :8810); env vars override them.
  const envUrl = (import.meta.env.VITE_PARLANT_URL as string | undefined) ?? "http://localhost:8810";
  const envAgent = (import.meta.env.VITE_AGENT_ID as string | undefined) ?? "tom";
  const [draftUrl, setDraftUrl] = useState(envUrl);
  const [draftAgent, setDraftAgent] = useState(envAgent);
  const [draftToken, setDraftToken] = useState("");
  const [applied, setApplied] = useState({
    url: envUrl,
    agent: envAgent,
    identityKey: "",
    token: "",
  });

  // Set while you are editing an already-satisfied connection, which keeps the
  // Connect form open until you apply it.
  const [editingConnection, setEditingConnection] = useState(false);

  const connect = () => {
    setEditingConnection(false);
    setApplied({
      url: draftUrl.trim(),
      agent: draftAgent.trim(),
      // In a real app you pass identityKey: user.id from your auth system (see the
      // README). The playground has no sign-in, so it reads the pasted token's `sub`.
      identityKey: subFromJwt(draftToken.trim()),
      token: draftToken.trim(),
    });
  };

  // Remounting the widget (via widgetKey) after clearing its localStorage forgets the
  // stored session + guest token — handy for testing the fresh / no-active-session flow.
  // The auth mode decides which sample server you run (guest :8810, signed-in :8811),
  // so switching it also swaps the draft URL — but only between those two defaults,
  // never over a URL the user typed themselves.
  const GUEST_URL = envUrl;
  const SIGNED_IN_URL = envUrl.replace(":8810", ":8811");
  const switchAuth = (next: Auth) => {
    setAuth(next);
    if (next === "authenticated" && draftUrl.trim() === GUEST_URL) setDraftUrl(SIGNED_IN_URL);
    if (next === "guest" && draftUrl.trim() === SIGNED_IN_URL) setDraftUrl(GUEST_URL);
  };

  const [resetNonce, setResetNonce] = useState(0);
  const [confirmReset, setConfirmReset] = useState(false);
  const clearSession = () => {
    try {
      // The widget's storage keys are hashed (base + agent + scope), so just clear every
      // parlant key — this is a test reset.
      for (const key of Object.keys(localStorage)) {
        if (key.startsWith("parlant:")) localStorage.removeItem(key);
      }
    } catch {
      /* ignore storage errors */
    }
    setResetNonce((n) => n + 1);
    setConfirmReset(false);
  };

  const dirty =
    draftUrl.trim() !== applied.url ||
    draftAgent.trim() !== applied.agent ||
    (auth === "authenticated" && draftToken.trim() !== applied.token);
  const configured = Boolean(
    applied.url &&
    applied.agent &&
    (auth === "guest" || (applied.identityKey && applied.token)),
  );
  // /healthz needs no credentials, so poll as soon as we know where the server
  // is. Gating this on `configured` would report a running signed-in agent as
  // offline until a token was pasted — and tell you to start what is already up.
  const health = usePageHealth(
    applied.url,
    applied.agent,
    Boolean(applied.url && applied.agent),
  );
  // Section 1 tells you how to start the agent — unless it is already answering,
  // which is the case when the Parlant CLI launched it alongside this page.
  const agentRunning = health === "online";

  // Connecting is only a step while it still needs something from you: a
  // server that answers, and in signed-in mode a token. When it doesn't, the
  // section collapses to a status line you can expand to point elsewhere, and
  // the remaining steps renumber. See steps.test.ts for the state matrix.
  const step = playgroundSteps({
    agentRunning,
    dirty,
    auth,
    hasToken: Boolean(applied.token),
    editing: editingConnection,
  });
  const connectIsStep = step.connectIsStep;

  // Every live-tweakable widget option, gathered in one place and spread into whichever
  // example wrapper is active.
  const demo: WidgetDemoConfig = {
    mode,
    theme,
    fontFamily,
    streamStyle,
    streamSpeed,
    showAttachments,
    acceptFileTypes: imagesOnly ? "image/png,image/jpeg,image/webp,image/gif" : undefined,
    showReasoning,
    reasoningMode,
    errorMessage,
    customerSound,
    agentSound,
    soundTheme,
    thinkingDelay,
    suggestions: suggestions.filter((s) => s.trim()),
    showNewChat,
    showDisclaimer,
    launcherHint,
    preface,
    title,
    placeholder,
  };

  // A copy-pasteable <ChatWidget/> reflecting the current knobs — only props that differ
  // from the widget's own defaults, so it stays minimal.
  const snippet = buildWidgetSnippet(auth, applied, demo);

  // Remount the widget when identity or display mode changes (auth/mode/connection);
  // every other option in `demo` is a live prop that updates in place.
  const widgetKey = `${auth}|${mode}|${applied.url}|${applied.agent}|${applied.identityKey}|${applied.token}|${resetNonce}`;
  const widget = !configured ? null : auth === "guest" ? (
    <GuestExample key={widgetKey} serverUrl={applied.url} agentId={applied.agent} {...demo} />
  ) : (
    <AuthenticatedExample
      key={widgetKey}
      serverUrl={applied.url}
      agentId={applied.agent}
      identityKey={applied.identityKey}
      token={applied.token}
      {...demo}
    />
  );

  // Flip the playground's own skin to match the chosen theme, so a light theme sits on a
  // light page and a dark theme on a dark page (makes the theme easy to judge).
  const lightPage = theme === "monochrome" || theme === "parlant" || theme === "sand";

  return (
    <div className={`page-bg min-h-screen overflow-x-hidden${lightPage ? " page-light" : ""}`}>
      {confirmReset && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
          onClick={() => setConfirmReset(false)}
        >
          <div
            role="dialog"
            aria-modal="true"
            aria-labelledby="reset-title"
            className="w-full max-w-md rounded-2xl border border-white/10 bg-zinc-900 p-6 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 id="reset-title" className="text-lg font-semibold text-zinc-50">
              Reset session?
            </h2>
            <p className="mt-2 text-sm leading-relaxed text-zinc-300">
              This clears the stored conversation and guest token from this browser's{" "}
              <code className="font-mono text-emerald-300/80">localStorage</code> and remounts the
              widget with a fresh, empty session. The current thread will be forgotten and can't be
              recovered.
            </p>
            <div className="mt-6 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setConfirmReset(false)}
                className="rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-sm text-zinc-300 transition-colors hover:bg-white/10 hover:text-zinc-100"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={clearSession}
                className="rounded-lg bg-red-600 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-red-500"
              >
                Reset session
              </button>
            </div>
          </div>
        </div>
      )}
      <div className="mx-auto max-w-6xl px-6 py-10">
        {/* Brand */}
        <header className="flex flex-wrap items-center gap-y-2 gap-2.5">
          <img
            src="https://www.parlant.io/logo/logo-full-white.svg"
            alt="Parlant"
            className="h-7 w-auto"
          />
          <span className="ml-1 rounded-full border border-white/10 bg-white/5 px-2.5 py-0.5 text-xs text-zinc-300">
            Widget playground
          </span>
          <span className="ml-auto text-xs text-zinc-400">
            see <code className="font-mono text-emerald-300/80">widget/README.md</code>
          </span>
        </header>

        {/* Hero */}
        <section className="mt-12 max-w-4xl">
          <h1 className="text-balance text-4xl font-bold tracking-tight text-zinc-50 sm:text-5xl">
            Meet the Parlant chat widget.
          </h1>
          <p className="mt-4 text-lg leading-relaxed text-zinc-300">
            This page runs the <b className="text-zinc-200">real, shadcn-style copy-paste widget</b> against a
            live Parlant agent — not a mockup. Start the bundled sample agent, hit Connect, and you
            have a working chat in under two minutes. Then walk the feature tour and lift the source
            straight into your app.
          </p>
          <p className="mt-4 text-sm text-zinc-400">
            The path below is <b className="text-zinc-300">Run Parlant → Connect widget → Configure widget → Try the widget</b>. It
            takes about 3 minutes.
          </p>
        </section>

        {/* 1 · Run the sample agent — the thing to do FIRST. The auth choice lives here
            because it decides WHICH server you run (and whether you need a token). */}
        <section className="mt-12 overflow-hidden rounded-2xl border border-emerald-500/25 bg-emerald-500/[0.06] p-6">
          <SectionHead
            n={step.run}
            title="Run the sample agent"
            sub={
              agentRunning
                ? "Already running — the playground is pointed at it."
                : "Do this first — the playground is already pointed at it."
            }
          />
          <div className="mt-5 flex max-w-2xl flex-col gap-5">
            <p className="text-sm leading-relaxed text-zinc-300">
              <b className="text-zinc-100">Agent Tom</b> is a tiny sample agent that chats normally
              and, whenever you attach a photo, describes what&apos;s in it using an OpenAI vision
              model. Pick how the widget should authenticate — it decides which server you run:
            </p>

            <div className="max-w-xs">
              <Segmented
                value={auth}
                onChange={switchAuth}
                options={[
                  { value: "guest", label: "Guest" },
                  { value: "authenticated", label: "Signed in" },
                ]}
              />
            </div>

            {agentRunning ? (
              <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-4 py-3">
                <p className="text-sm leading-relaxed text-zinc-200">
                  <b className="text-zinc-100">Agent Tom is already running</b> on{" "}
                  <code className="font-mono text-emerald-300/90">{applied.url}</code>.
                  {auth === "guest"
                    ? " The widget below is already talking to it."
                    : " Paste your customer token under Connect below and you're set."}
                </p>
              </div>
            ) : (
              <>
                <div>
                  <p className="text-sm leading-relaxed text-zinc-300">
                    The quickest way is the Parlant CLI, which runs this playground and the sample
                    agent together{auth === "guest" ? "" : " (and prints a customer token for you)"}:
                  </p>
                  <CodeBlock
                    className="mt-2"
                    preClassName="overflow-x-auto whitespace-nowrap rounded-lg border border-white/10 bg-black/40 px-3 py-2.5 pr-12 font-mono text-xs text-emerald-300/90"
                    code={
                      auth === "guest"
                        ? "export OPENAI_API_KEY=sk-…  &&  parlant widget playground"
                        : "export OPENAI_API_KEY=sk-…  &&  parlant widget playground --signed-in"
                    }
                  />
                  <p className="mt-2 text-xs leading-relaxed text-zinc-400">
                    Tom uses the OpenAI key for both chat and vision.
                    {auth === "guest"
                      ? " He serves :8810 with no auth, so the widget gets a guest identity automatically."
                      : " In signed-in mode he serves :8811 with customer-JWT verification and auto-provisioning."}
                  </p>
                </div>

                <div>
                  <p className="text-sm leading-relaxed text-zinc-300">
                    Working inside a Parlant checkout? Start the agent yourself, in another
                    terminal:
                  </p>
                  <CodeBlock
                    className="mt-2"
                    preClassName="overflow-x-auto whitespace-nowrap rounded-lg border border-white/10 bg-black/40 px-3 py-2.5 pr-12 font-mono text-xs text-emerald-300/90"
                    code={
                      auth === "guest"
                        ? "uv run widget/sample-agent/agent.py"
                        : "uv run widget/sample-agent/signed_in_agent.py"
                    }
                  />
                </div>
              </>
            )}

            {auth === "authenticated" && (
              <div>
                <p className="text-sm leading-relaxed text-zinc-300">
                  Signed-in mode needs a <b className="text-zinc-100">customer token</b>. The CLI
                  prints one when it starts; from a checkout, mint one yourself (add your own id and
                  name if you like):
                </p>
                <CodeBlock
                  className="mt-2"
                  preClassName="overflow-x-auto whitespace-nowrap rounded-lg border border-white/10 bg-black/40 px-3 py-2.5 pr-12 font-mono text-xs text-emerald-300/90"
                  code="uv run widget/sample-agent/mint_token.py"
                />
                <p className="mt-2 text-sm leading-relaxed text-zinc-300">
                  Paste it into the highlighted <b className="text-zinc-100">Customer token</b> field
                  under <b className="text-zinc-100">Connect</b> below, then click Connect. Tokens
                  expire after an hour — get a fresh one when yours does.
                </p>
              </div>
            )}

            <p className="text-xs leading-relaxed text-zinc-400">
              No OpenAI key? You don't have to use this vision example. You can still point <b className="text-zinc-300">Connect</b> at any Parlant
              server and the widget will chat fine. Just note the attachment demo is Tom-specific —
              describing an image depends on <i>that</i> agent&apos;s tool/config, so on another
              agent, attached files won&apos;t do anything unless it&apos;s set up to handle them.
            </p>
          </div>
        </section>

        {/* Setup: connect + configure — stacked, not side-by-side, so the floating widget
            doesn't overlay the panel you're tweaking. */}
        <div className="mt-6 flex flex-col gap-5">
          {/* 2 · Connect — a step only while it still needs something from you. */}
          {!connectIsStep ? (
            <section className="rounded-2xl border border-white/10 bg-white/[0.03] px-5 py-3.5 backdrop-blur">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <p className="text-sm text-zinc-300">
                  <span className="mr-2 inline-block h-2 w-2 rounded-full bg-emerald-400 align-middle" />
                  Connected to <code className="font-mono text-emerald-300/90">{applied.url}</code>{" "}
                  as <code className="font-mono text-emerald-300/90">{applied.agent}</code>
                  {auth === "authenticated" ? " (signed in)" : " (guest)"}
                </p>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setEditingConnection(true)}
                    className="rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-zinc-300 transition-colors hover:bg-white/10 hover:text-zinc-100"
                  >
                    Change server
                  </button>
                  <button
                    type="button"
                    onClick={() => setConfirmReset(true)}
                    title="Clear the stored session + guest token for this agent, then remount"
                    className="rounded-lg bg-red-600 px-3 py-1.5 text-xs font-medium text-white transition-colors hover:bg-red-500"
                  >
                    Reset session
                  </button>
                </div>
              </div>
            </section>
          ) : (
          <section className="rounded-2xl border border-white/10 bg-white/[0.03] p-6 backdrop-blur">
            <div className="flex items-start justify-between gap-3">
              <SectionHead
                n={step.connect}
                title="Connect the widget"
                sub="Where the widget talks to."
              />
              <button
                type="button"
                onClick={() => setConfirmReset(true)}
                title="Clear the stored session + guest token for this agent, then remount"
                className="shrink-0 rounded-lg bg-red-600 px-3 py-1 text-xs font-medium text-white transition-colors hover:bg-red-500"
              >
                Reset session
              </button>
            </div>
            <form
              className="mt-5 flex flex-col gap-4"
              onSubmit={(e) => {
                e.preventDefault();
                connect();
              }}
            >
              <Field label="Server URL" hint="Your Parlant server, reachable from this browser. Note that the default Parlant port (unless configured otherwise, as in Tom's example) is :8800, not :8810.">
                <input
                  className={inputCls}
                  value={draftUrl}
                  onChange={(e) => setDraftUrl(e.target.value)}
                  placeholder="http://localhost:8810"
                  autoComplete="off"
                  spellCheck={false}
                />
              </Field>
              <Field label="Agent ID" hint="This is the `id=` argument you passed to `create_agent()` on the server.">
                <input
                  className={inputCls}
                  value={draftAgent}
                  onChange={(e) => setDraftAgent(e.target.value)}
                  placeholder="tom"
                  autoComplete="off"
                  spellCheck={false}
                />
              </Field>
              {auth === "authenticated" && (
                <>
                  <Field
                    label="Customer token"
                    hint="A Parlant customer JWT — mint one with `uv run widget/sample-agent/mint_token.py` (against the signed-in server: `uv run widget/sample-agent/signed_in_agent.py`, port 8811). Its `sub` claim doubles as the identity key here; a real app passes identityKey: user.id (see the README)."
                  >
                    <input
                      className={`${inputCls} token-field-highlight${draftToken.trim() ? " token-filled" : ""}`}
                      value={draftToken}
                      onChange={(e) => setDraftToken(e.target.value)}
                      placeholder="eyJhbGciOiJIUzI1Ni…"
                      autoComplete="off"
                      spellCheck={false}
                    />
                  </Field>
                </>
              )}
              <div className="flex items-center gap-3 pt-1">
                <button
                  type="submit"
                  className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-emerald-500"
                >
                  {health === "online" && !dirty ? "Reconnect" : "Connect"}
                </button>
                <ConnectionStatus dirty={dirty} configured={configured} health={health} />
              </div>
            </form>
          </section>
          )}

          {/* 3 · Configure */}
          <section className="rounded-2xl border border-white/10 bg-white/[0.03] p-6 backdrop-blur">
            <SectionHead
              n={step.configure}
              title="Configure the widget"
              sub="Each maps straight to a widget prop."
            />
            <div className="mt-5 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
              <Control label="Display" hint={`mode="${mode}"`}>
                <Segmented
                  value={mode}
                  onChange={setMode}
                  options={[
                    { value: "floating", label: "Floating" },
                    { value: "inline", label: "Inline" },
                    { value: "input", label: "Input" },
                    { value: "bar", label: "Bar" },
                  ]}
                />
              </Control>
              <Control label="Theme" hint={`theme="${theme}"`}>
                <select
                  value={theme}
                  onChange={(e) => setTheme(e.target.value as ThemeName)}
                  className={inputCls}
                >
                  {THEMES.map((t) => (
                    <option key={t.value} value={t.value} className="bg-zinc-900">
                      {t.label}
                    </option>
                  ))}
                </select>
              </Control>
              <Control label="Font family" hint={`fontFamily="${fontFamily || "Inter"}"`}>
                <input
                  className={inputCls}
                  value={fontFamily}
                  onChange={(e) => setFontFamily(e.target.value)}
                  placeholder="Inter"
                  autoComplete="off"
                  spellCheck={false}
                />
              </Control>
              <Control label="Stream style" hint={`streamStyle="${streamStyle}"`}>
                <select
                  value={streamStyle}
                  onChange={(e) => setStreamStyle(e.target.value as StreamStyle)}
                  className={inputCls}
                >
                  {STREAM_STYLES.map((s) => (
                    <option key={s.value} value={s.value} className="bg-zinc-900">
                      {s.label}
                    </option>
                  ))}
                </select>
              </Control>
              <Control label="Stream speed" hint={`streamSpeed="${streamSpeed}"`}>
                <Segmented
                  value={streamSpeed as string}
                  onChange={(v) => setStreamSpeed(v as StreamSpeed)}
                  options={[
                    { value: "slow", label: "Slow" },
                    { value: "normal", label: "Normal" },
                    { value: "fast", label: "Fast" },
                  ]}
                />
              </Control>
              <Control label="Attachments" hint={`showAttachments={${showAttachments}}`}>
                <Segmented
                  value={showAttachments ? "on" : "off"}
                  onChange={(v) => setShowAttachments(v === "on")}
                  options={[
                    { value: "on", label: "On" },
                    { value: "off", label: "Off" },
                  ]}
                />
              </Control>
              <Control
                label="Accept file types"
                hint={imagesOnly ? 'acceptFileTypes="image/*"' : "acceptFileTypes={undefined}"}
              >
                <Segmented
                  value={imagesOnly ? "images" : "any"}
                  onChange={(v) => setImagesOnly(v === "images")}
                  options={[
                    { value: "images", label: "Images" },
                    { value: "any", label: "Any file" },
                  ]}
                />
              </Control>
              <Control label="Reasoning" hint={`showReasoning={${showReasoning}}`}>
                <Segmented
                  value={showReasoning ? "on" : "off"}
                  onChange={(v) => setShowReasoning(v === "on")}
                  options={[
                    { value: "on", label: "On" },
                    { value: "off", label: "Off" },
                  ]}
                />
              </Control>
              <Control
                label="Reasoning display"
                hint={`reasoningMode="${reasoningMode}"${showReasoning ? "" : " — turn Reasoning on"}`}
              >
                <Segmented
                  value={reasoningMode}
                  onChange={setReasoningMode}
                  options={[
                    { value: "transient", label: "Transient" },
                    { value: "collapsible", label: "Collapsible" },
                  ]}
                />
              </Control>
              <Control label="New-chat button" hint={`showNewChat={${showNewChat}}`}>
                <Segmented
                  value={showNewChat ? "on" : "off"}
                  onChange={(v) => setShowNewChat(v === "on")}
                  options={[
                    { value: "on", label: "On" },
                    { value: "off", label: "Off" },
                  ]}
                />
              </Control>
              <Control label="Customer sound" hint={`customerSound={${customerSound}}`}>
                <Segmented
                  value={customerSound ? "on" : "off"}
                  onChange={(v) => setCustomerSound(v === "on")}
                  options={[
                    { value: "on", label: "On" },
                    { value: "off", label: "Off" },
                  ]}
                />
              </Control>
              <Control label="Agent sound" hint={`agentSound={${agentSound}}`}>
                <Segmented
                  value={agentSound ? "on" : "off"}
                  onChange={(v) => setAgentSound(v === "on")}
                  options={[
                    { value: "on", label: "On" },
                    { value: "off", label: "Off" },
                  ]}
                />
              </Control>
              <Control
                label="Sound theme"
                hint={`soundTheme="${soundTheme}" — press Play to hear the reply blip.`}
              >
                <div className="flex gap-2">
                  <select
                    value={soundTheme}
                    onChange={(e) => setSoundTheme(e.target.value as SoundTheme)}
                    className={inputCls}
                  >
                    {SOUND_THEME_OPTIONS.map((t) => (
                      <option key={t.value} value={t.value} className="bg-zinc-900">
                        {t.label}
                      </option>
                    ))}
                  </select>
                  <button
                    type="button"
                    onClick={() => playAgentSound(soundTheme)}
                    aria-label={`Play the ${SOUND_THEMES[soundTheme].label} reply sound`}
                    className="shrink-0 rounded-lg border border-white/10 bg-white/5 px-3 text-sm text-zinc-200 transition hover:border-emerald-500/50 hover:text-white"
                  >
                    ▸ Play
                  </button>
                </div>
              </Control>
              <Control
                label="Input placeholder"
                hint={'placeholder — empty uses "Type a message…"'}
              >
                <input
                  className={inputCls}
                  value={placeholder}
                  placeholder="Type a message…"
                  onChange={(e) => setPlaceholder(e.target.value)}
                />
              </Control>
              <Control
                label="Thinking delay"
                hint={`thinkingDelay={${thinkingDelay}} — ms before the dots fade in; 0 shows them instantly.`}
              >
                <input
                  type="number"
                  min={0}
                  step={100}
                  className={inputCls}
                  value={thinkingDelay}
                  onChange={(e) => setThinkingDelay(Math.max(0, Number(e.target.value) || 0))}
                />
              </Control>
              <div className="sm:col-span-2 lg:col-span-3">
                <Control
                  label="Suggestions"
                  hint="suggestions — starter chips (click to send), shown before the conversation begins."
                >
                  <div className="flex flex-col gap-1.5">
                    {suggestions.map((s, i) => (
                      <div key={i} className="flex items-center gap-1.5">
                        <input
                          className={inputCls}
                          value={s}
                          onChange={(e) =>
                            setSuggestions((prev) => prev.map((v, j) => (j === i ? e.target.value : v)))
                          }
                          placeholder="Conversation starter…"
                          autoComplete="off"
                          spellCheck={false}
                        />
                        <button
                          type="button"
                          onClick={() => setSuggestions((prev) => prev.filter((_, j) => j !== i))}
                          aria-label="Remove suggestion"
                          className="shrink-0 rounded-lg border border-white/10 px-2.5 py-2 text-zinc-400 transition-colors hover:border-white/20 hover:text-zinc-100"
                        >
                          ×
                        </button>
                      </div>
                    ))}
                    <button
                      type="button"
                      onClick={() => setSuggestions((prev) => [...prev, ""])}
                      className="self-start rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-zinc-300 transition-colors hover:bg-white/10"
                    >
                      + Add suggestion
                    </button>
                  </div>
                </Control>
              </div>
              <Control label="Disclaimer" hint={showDisclaimer ? "disclaimer={<…>}" : "disclaimer={undefined}"}>
                <Segmented
                  value={showDisclaimer ? "on" : "off"}
                  onChange={(v) => setShowDisclaimer(v === "on")}
                  options={[
                    { value: "on", label: "On" },
                    { value: "off", label: "Off" },
                  ]}
                />
              </Control>
              <Control label="Header title" hint="title — empty falls back to the agent's name.">
                <input
                  className={inputCls}
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="Agent Tom"
                  autoComplete="off"
                  spellCheck={false}
                />
              </Control>
              <Control
                label="Launcher hint"
                hint="launcherHint — the floating-launcher nudge; empty hides it."
              >
                <input
                  className={inputCls}
                  value={launcherHint}
                  onChange={(e) => setLauncherHint(e.target.value)}
                  placeholder="Try me"
                  autoComplete="off"
                  spellCheck={false}
                />
              </Control>
              <Control label="Preface" hint="preface — muted intro shown above the first message.">
                <input
                  className={inputCls}
                  value={preface}
                  onChange={(e) => setPreface(e.target.value)}
                  placeholder="Muted intro line…"
                  autoComplete="off"
                  spellCheck={false}
                />
              </Control>
              <Control label="Error message" hint="errorMessage — shown inline when a turn fails.">
                <input
                  className={inputCls}
                  value={errorMessage}
                  onChange={(e) => setErrorMessage(e.target.value)}
                  placeholder="Shown when the agent errors…"
                  autoComplete="off"
                  spellCheck={false}
                />
              </Control>
            </div>
            <p className="mt-4 text-xs text-zinc-400">
              Send a message to feel the stream style. Everything here updates live; switching{" "}
              <b className="text-zinc-300">Authentication</b> or <b className="text-zinc-300">Display</b>{" "}
              (or reconnecting) reloads the conversation scoped to that server, agent, and identity.
            </p>
          </section>

          {/* Copy the folder, then paste the configured JSX. */}
          <section className="rounded-2xl border border-white/10 bg-white/[0.03] p-6 backdrop-blur">
            <SectionHead
              n={step.add}
              title="Add the widget to your app"
              sub="Copy one folder, then paste the JSX. No npm install, no Tailwind, no build config."
            />
            <div className="mt-5 space-y-5">
              <div>
                <p className="mb-2 text-sm text-zinc-300">
                  <b className="text-zinc-100">1.</b> Copy the single{" "}
                  <code className="font-mono text-emerald-300/80">parlant-chat/</code> folder into your
                  app — it&apos;s self-contained (styles included).
                </p>
                <CodeBlock code={"cp -r parlant-chat ~/your-app/src/parlant-chat"} />
              </div>
              <div>
                <p className="mb-2 text-sm text-zinc-300">
                  <b className="text-zinc-100">2.</b> Render it with your configuration.{" "}
                  <span className="text-zinc-400">
                    {"("}
                    <code className="font-mono text-emerald-300/80">handleUpload</code> /{" "}
                    <code className="font-mono text-emerald-300/80">getToken</code> are your own
                    functions — only needed if you enable attachments / customer auth.{")"}
                  </span>
                </p>
                <CodeBlock code={snippet} />
              </div>
            </div>
          </section>
        </div>

        {/* 4 · Try it */}
        <section className="mt-10">
          <SectionHead n={step.tryIt} title="Try it" sub="The live widget." />
          <div className="mt-5">
            {!configured ? (
              <div className="rounded-2xl border border-dashed border-white/15 bg-white/[0.02] p-10 text-center text-sm text-zinc-400">
                Enter your server URL and agent ID above, then <b className="text-zinc-300">Connect</b>.
              </div>
            ) : mode === "inline" ? (
              <div className="mx-auto h-[560px] w-full max-w-md overflow-hidden rounded-2xl border border-white/10 shadow-2xl shadow-black/40">
                {widget}
              </div>
            ) : (
              <>
                <div className="flex items-center gap-2.5 rounded-2xl border border-dashed border-white/15 bg-white/[0.02] p-4 text-sm text-zinc-300">
                  <span className="text-lg text-emerald-400" aria-hidden>
                    {mode === "bar" ? "↓" : "↘"}
                  </span>
                  {mode === "input"
                    ? "The input box is docked in the bottom-right corner — type or pick a suggestion, and it expands into the full chat."
                    : mode === "bar"
                      ? "The message bar sits at the bottom center of the screen — type or pick a suggestion, and it grows into the full chat."
                      : "The launcher is docked in the bottom-right corner — click the chat button to open it, and your suggestions greet you above the input."}
                </div>
                {widget}
              </>
            )}
          </div>
        </section>

        {/* Guided tour of the features */}
        <section className="mt-16">
          <h2 className="text-xs font-semibold uppercase tracking-[0.14em] text-emerald-400/90">
            Take it for a spin
          </h2>
          <p className="mt-3 max-w-2xl text-sm leading-relaxed text-zinc-300">
            With Agent Tom running and connected, here&apos;s the guided path through everything the
            widget does — top to bottom, no setup between steps.
          </p>
          <ol className="mt-6 grid gap-x-10 gap-y-4 sm:grid-cols-2">
            <TourItem n={1} title="Say hi and watch it stream">
              Send a message; the reply animates in using the <b className="text-zinc-300">Stream
                style</b> you picked. Switch styles and speed in <b className="text-zinc-300">Configure</b>{" "}
              — they update live, mid-conversation.
            </TourItem>
            <TourItem n={2} title="See Markdown render live">
              Ask Tom to <i>“reply with a bulleted list and a code block.”</i> It renders as
              <b className="text-zinc-300"> Markdown</b> — and the reveal animation streams{" "}
              <i>through</i> the formatting, word-by-word.
            </TourItem>
            <TourItem n={3} title="Watch it think">
              Before the reply, a three-dot <b className="text-zinc-300">thinking indicator</b>{" "}
              appears. If the agent emits a status message, it shows right beside the dots.
            </TourItem>
            <TourItem n={4} title="Attach a photo">
              Hit the paperclip, pick an image, and ask <i>“what&apos;s in this?”</i> Tom calls an
              OpenAI vision tool that returns a <b className="text-zinc-300">picture-description
                card</b> — a custom UI widget (thumbnail + caption) the app renders from the tool&apos;s
              output. Images only, via <code className="font-mono text-emerald-300/80">acceptFileTypes</code>.
            </TourItem>
            <TourItem n={5} title="Interrupt it">
              Ask for something long, then hit <b className="text-zinc-300">Stop</b> to cancel the
              turn mid-stream.
            </TourItem>
            <TourItem n={6} title="Make room">
              <b className="text-zinc-300">Expand</b> grows the panel; minimizing tucks it back to
              the launcher without losing the conversation or your scroll position.
            </TourItem>
            <TourItem n={7} title="Start over">
              <b className="text-zinc-300">New chat</b> spins up a fresh session (with a confirm),
              and a pill lets you jump back to the latest message.
            </TourItem>
            <TourItem n={8} title="Come back later">
              Refresh the page — the thread <b className="text-zinc-300">resumes</b>. The session id
              and guest token live in <code className="font-mono text-emerald-300/80">localStorage</code>,
              scoped by server, agent, and auth identity.
            </TourItem>
            <TourItem n={9} title="Pull the plug">
              Stop the agent process. The widget flips to an <b className="text-zinc-300">offline
                banner</b> and blocks input; restart Tom and it recovers on its own (and the status
              above turns red, then green).
            </TourItem>
            <TourItem n={10} title="Switch identities">
              Flip <b className="text-zinc-300">Authentication</b> to Signed in to see the
              customer-token flow. A dev server ignores the token — see “Signed-in setup” below.
            </TourItem>
          </ol>
        </section>

        {/* Install in your app */}
        <section className="mt-16">
          <h2 className="text-xs font-semibold uppercase tracking-[0.14em] text-emerald-400/90">
            Install in your app
          </h2>
          <p className="mt-3 max-w-2xl text-sm leading-relaxed text-zinc-300">
            It&apos;s shadcn-style — you copy the source and own it, no npm package and{" "}
            <b className="text-zinc-200">no Tailwind</b>. The full version is in{" "}
            <code className="font-mono text-emerald-300/80">widget/README.md</code>.
          </p>
          <div className="mt-6 grid gap-5 lg:grid-cols-2">
            <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-5 backdrop-blur">
              <h3 className="text-sm font-semibold text-zinc-100">1 · Copy one folder</h3>
              <p className="mt-1.5 text-sm text-zinc-400">
                The whole widget is the single{" "}
                <code className="font-mono text-emerald-300/80">parlant-chat/</code> directory — copy it
                into your app. The rest of <code className="font-mono text-emerald-300/80">widget/</code>{" "}
                is just this demo harness.
              </p>
              <CodeBlock
                className="mt-3"
                preClassName="overflow-x-auto rounded-lg border border-white/10 bg-black/40 p-3 pr-12 font-mono text-xs leading-relaxed text-zinc-300"
                code={`parlant widget init src/parlant-chat`}
              />
              <p className="mt-3 text-xs text-zinc-500">
                The widget ships inside the pip package, so this is a plain file copy — no node
                toolchain, no download. From a checkout,{" "}
                <code className="font-mono text-emerald-300/80">cp -r parlant-chat …</code> does the
                same thing.
              </p>
              <p className="mt-2 text-xs text-zinc-500">
                Self-contained: Inter font, animations, theme tokens and all component CSS live in{" "}
                <code className="font-mono text-emerald-300/80">parlant-chat/styles.css</code>.
              </p>
            </div>

            <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-5 backdrop-blur">
              <h3 className="text-sm font-semibold text-zinc-100">2 · Render it</h3>
              <p className="mt-1.5 text-sm text-zinc-400">
                Import from the folder&apos;s barrel.{" "}
                <code className="font-mono text-emerald-300/80">styles.css</code> is imported by the
                widget itself, so there&apos;s no separate CSS import.
              </p>
              <CodeBlock
                className="mt-3"
                preClassName="overflow-x-auto rounded-lg border border-white/10 bg-black/40 p-3 pr-12 font-mono text-xs leading-relaxed text-zinc-300"
                code={`import { ChatWidget } from "./parlant-chat";

<ChatWidget
  serverUrl="https://parlant.your-app.com"
  agentId="AGENT_ID"
/>`}
              />
            </div>

            <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-5 backdrop-blur">
              <h3 className="text-sm font-semibold text-zinc-100">3 · Theme it (optional)</h3>
              <p className="mt-1.5 text-sm text-zinc-400">
                Ships light + dark defaults via CSS variables on{" "}
                <code className="font-mono text-emerald-300/80">.parlant-widget</code> — works with zero
                config. To match your brand, redefine the tokens:
              </p>
              <CodeBlock
                className="mt-3"
                preClassName="overflow-x-auto rounded-lg border border-white/10 bg-black/40 p-3 pr-12 font-mono text-xs leading-relaxed text-zinc-300"
                code={`.parlant-widget {
  --primary: 240 6% 10%;
  --background: 0 0% 100%;
  /* --foreground, --muted, --border, … */
}`}
              />
              <p className="mt-3 text-xs text-zinc-500">
                Using shadcn/ui? Point these at your own <code className="font-mono text-emerald-300/80">--background</code>/… to inherit your theme.
              </p>
            </div>

            <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-5 backdrop-blur">
              <h3 className="text-sm font-semibold text-zinc-100">4 · Wire your integration points</h3>
              <ul className="mt-3 list-disc space-y-1.5 pl-4 text-sm text-zinc-400">
                <li>
                  <code className="font-mono text-emerald-300/80">auth</code> — signed-in
                  customers provide an identity key and a backend token callback.
                </li>
                <li>
                  <code className="font-mono text-emerald-300/80">onUpload</code> — attachments:
                  upload to your storage, return an <code className="font-mono text-emerald-300/80">AttachmentRef</code>.
                </li>
                <li>Reskin by changing the CSS token values — the widget hard-codes no colors.</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Reference */}
        <section className="mt-16">
          <h2 className="text-xs font-semibold uppercase tracking-[0.14em] text-emerald-400/90">
            Reference
          </h2>
          <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <Card title="Guest vs. signed in">
              <p>
                <b className="text-zinc-200">Guest</b> needs no auth — the server issues a guest
                token automatically. <b className="text-zinc-200">Signed in</b> sends a customer
                token via <code className="font-mono text-emerald-300/80">auth.getToken()</code>; in
                production that function calls <b className="text-zinc-200">your backend</b>, which
                mints the token from your server-side Parlant credentials.
              </p>
            </Card>

            <Card title="Signed-in setup">
              <p>To bind sessions to a real customer (not just a dev server):</p>
              <ol className="mt-2 list-decimal space-y-1 pl-4">
                <li>Run the server with <code className="font-mono text-emerald-300/80">AuthConfig</code> (admin key + <code className="font-mono text-emerald-300/80">customer_jwt_secret</code>).</li>
                <li>Make the customer known — <code className="font-mono text-emerald-300/80">create_customer(id=…)</code> or <code className="font-mono text-emerald-300/80">auto_provision_customers=True</code>.</li>
                <li>Mint a JWT (<code className="font-mono text-emerald-300/80">sub</code> = CustomerId) and paste it above — here its <code className="font-mono text-emerald-300/80">sub</code> doubles as the identity key.</li>
              </ol>
              <p className="mt-2 text-zinc-400">
                A dev-mode server ignores the token entirely — the Signed-in tab works but isn't real
                auth until you do the above. To see it working for real, run{" "}
                <code className="font-mono text-emerald-300/80">parlant widget playground --signed-in</code>
                , which does all three steps for you.
              </p>
            </Card>

            <Card title="Sessions persist">
              <p>
                The session id (and guest token) live in{" "}
                <code className="font-mono text-emerald-300/80">localStorage</code>, scoped by server,
                agent, and auth identity — refresh and the same thread resumes. Customer bearer
                tokens remain memory-only. “New chat” starts a fresh conversation.
              </p>
            </Card>

            <Card title="Markdown">
              <p>
                Replies and streamed reasoning render as <b className="text-zinc-200">Markdown</b> —
                bold, lists, links, headings, blockquotes, and fenced code — and the reveal
                animation runs <i>through</i> it, word-by-word. It&apos;s a tiny zero-dependency
                renderer (no react-markdown), so the widget stays copy-paste and dependency-free.
                Ask Tom to “reply with a bulleted list and a code block” to see it.
              </p>
            </Card>

            <Card title="Rich content">
              <p>
                Tool results can render as custom UI. A tool returns a typed{" "}
                <code className="font-mono text-emerald-300/80">widget</code> in its metadata; the
                app registers a renderer per <code className="font-mono text-emerald-300/80">type</code>.
                Tom&apos;s image tool does exactly this — the card under a described photo is a{" "}
                <code className="font-mono text-emerald-300/80">picture_description</code> widget
                (see <code className="font-mono text-emerald-300/80">example/src/widgets.tsx</code>):
              </p>
              <CodeBlock
                className="mt-2"
                preClassName="overflow-x-auto rounded-lg border border-white/10 bg-black/40 p-3 pr-12 font-mono text-xs leading-relaxed text-zinc-300"
                code={`registerWidget("picture_description",
  (w) => <PictureCard {...w} />);`}
              />
            </Card>

          </div>
        </section>

        <footer className="mt-14 border-t border-white/10 pt-6 text-xs text-zinc-400">
          The page is dark on purpose; the chat widget keeps its own theme so you see it as your
          users would.
        </footer>
      </div>
    </div>
  );
}

function ConnectionStatus({
  dirty,
  configured,
  health,
}: {
  dirty: boolean;
  configured: boolean;
  health: Health;
}) {
  if (dirty) return <span className="text-xs text-amber-400/90">Unapplied changes</span>;
  if (!configured) return <span className="text-xs text-zinc-400">Enter a server URL and agent ID.</span>;
  if (health === "checking")
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-zinc-300">
        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-zinc-400" /> Checking…
      </span>
    );
  if (health === "online")
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-emerald-400">
        <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" /> Connected
      </span>
    );
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-rose-400">
      <span className="h-1.5 w-1.5 rounded-full bg-rose-400" /> Can&apos;t reach it — is the agent running?
    </span>
  );
}

// Build a minimal, copy-pasteable <ChatWidget/> from the live config: identity always, then
// only the props that differ from the widget's own defaults.
function buildWidgetSnippet(
  auth: Auth,
  applied: { url: string; agent: string; identityKey: string },
  c: WidgetDemoConfig,
): string {
  const q = (s: string) => s.replace(/"/g, '\\"');
  const lines: string[] = [`serverUrl="${q(applied.url)}"`, `agentId="${q(applied.agent)}"`];

  if (auth === "guest") lines.push(`auth={{ type: "guest" }}`);
  else lines.push(`auth={{ type: "customer", identityKey: "${q(applied.identityKey)}", getToken }}`);

  if (c.mode !== "input") lines.push(`mode="${c.mode}"`);
  if (c.theme !== "monochrome") lines.push(`theme="${c.theme}"`);
  if (c.fontFamily.trim() && c.fontFamily.trim() !== "Inter")
    lines.push(`fontFamily="${q(c.fontFamily.trim())}"`);
  if (c.title.trim()) lines.push(`title="${q(c.title)}"`);
  if (c.streamStyle !== "rise") lines.push(`streamStyle="${c.streamStyle}"`);
  if (c.streamSpeed !== "normal") lines.push(`streamSpeed="${c.streamSpeed}"`);
  if (c.showReasoning) {
    lines.push(`showReasoning={true}`);
    if (c.reasoningMode !== "transient") lines.push(`reasoningMode="${c.reasoningMode}"`);
  }
  if (c.showAttachments) {
    lines.push(`showAttachments={true}`);
    lines.push(`onUpload={handleUpload}`);
    if (c.acceptFileTypes) lines.push(`acceptFileTypes="${c.acceptFileTypes}"`);
  }
  if (!c.customerSound) lines.push(`customerSound={false}`);
  if (!c.agentSound) lines.push(`agentSound={false}`);
  if (c.soundTheme !== "glass") lines.push(`soundTheme="${c.soundTheme}"`);
  if (c.placeholder.trim()) lines.push(`placeholder="${q(c.placeholder)}"`);
  if (c.thinkingDelay !== 1000) lines.push(`thinkingDelay={${c.thinkingDelay}}`);
  if (c.showNewChat) lines.push(`showNewChat={true}`);
  if (c.suggestions.length)
    lines.push(`suggestions={[${c.suggestions.map((s) => `"${q(s)}"`).join(", ")}]}`);
  if (c.mode === "floating" && c.launcherHint.trim() && c.launcherHint !== "Talk to us")
    lines.push(`launcherHint="${q(c.launcherHint)}"`);
  if (c.preface.trim()) lines.push(`preface="${q(c.preface)}"`);
  if (c.showDisclaimer)
    lines.push(`disclaimer={<>By chatting, you agree to our <a href="/privacy">Privacy Policy</a>.</>}`);

  return `import { ChatWidget } from "./parlant-chat";\n\n<ChatWidget\n${lines
    .map((l) => `  ${l}`)
    .join("\n")}\n/>`;
}

function CodeBlock({
  code,
  className,
  preClassName,
}: {
  code: string;
  className?: string;
  preClassName?: string;
}) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable */
    }
  };
  return (
    <div className={`relative${className ? ` ${className}` : ""}`}>
      <button
        type="button"
        onClick={copy}
        aria-label={copied ? "Copied" : "Copy"}
        title={copied ? "Copied" : "Copy"}
        className="absolute right-2 top-2 flex h-7 w-7 items-center justify-center rounded-md text-white/80 transition-colors hover:bg-white/15 hover:text-white"
      >
        {copied ? (
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="M20 6 9 17l-5-5" />
          </svg>
        ) : (
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
          </svg>
        )}
      </button>
      <pre
        className={
          preClassName ??
          "overflow-x-auto rounded-xl border border-white/10 bg-black/40 p-4 pr-12 font-mono text-xs leading-relaxed text-zinc-200"
        }
      >
        <code>{code}</code>
      </pre>
    </div>
  );
}

function SectionHead({ n, title, sub }: { n: number; title: string; sub: string }) {
  return (
    <div className="flex items-center gap-3">
      <span className="grid h-7 w-7 place-items-center rounded-lg border border-emerald-500/30 bg-emerald-500/10 text-sm font-bold tabular-nums text-emerald-400">
        {n}
      </span>
      <div>
        <h2 className="text-base font-semibold leading-tight text-zinc-100">{title}</h2>
        <p className="text-xs text-zinc-400">{sub}</p>
      </div>
    </div>
  );
}

function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-sm font-medium text-zinc-300">{label}</span>
      {children}
      {hint && <span className="mt-1.5 block text-xs text-zinc-400">{hint}</span>}
    </label>
  );
}

function Control({ label, hint, children }: { label: string; hint: string; children: ReactNode }) {
  return (
    <div>
      <div className="mb-1.5 text-sm font-medium text-zinc-300">{label}</div>
      {children}
      <div className="mt-1.5 text-xs text-zinc-400">{hint}</div>
    </div>
  );
}

function Segmented<T extends string>({
  value,
  onChange,
  options,
}: {
  value: T;
  onChange: (v: T) => void;
  options: { value: T; label: string }[];
}) {
  return (
    <div className="inline-flex rounded-lg border border-white/10 bg-white/5 p-1">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          onClick={() => onChange(o.value)}
          className={
            "rounded-md px-3 py-1 text-sm transition-colors " +
            (value === o.value
              ? "bg-emerald-600 font-medium text-white"
              : "text-zinc-300 hover:text-zinc-100")
          }
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

function TourItem({ n, title, children }: { n: number; title: string; children: ReactNode }) {
  return (
    <li className="flex gap-3.5">
      <span className="mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded-full border border-emerald-500/30 bg-emerald-500/10 text-xs font-bold tabular-nums text-emerald-400">
        {n}
      </span>
      <div className="text-sm leading-relaxed text-zinc-300">
        <b className="text-zinc-100">{title}.</b> {children}
      </div>
    </li>
  );
}

function Card({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-5 backdrop-blur">
      <h3 className="text-sm font-semibold text-zinc-100">{title}</h3>
      <div className="mt-2 text-sm leading-relaxed text-zinc-300">{children}</div>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
