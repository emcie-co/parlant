// Which sections the playground shows as numbered steps, given how it was
// started. Extracted from main.tsx so the state matrix can be tested: the
// numbering has to stay contiguous whether you arrived via
// `parlant widget playground`, `--signed-in`, `--no-agent`, or your own server.

export type PlaygroundAuth = "guest" | "authenticated";

export interface ConnectionState {
  /** The configured server is answering /healthz. */
  agentRunning: boolean;
  /** The form has edits that haven't been applied yet. */
  dirty: boolean;
  auth: PlaygroundAuth;
  /** A customer token has been applied (signed-in mode only). */
  hasToken: boolean;
  /** The user asked to change an already-satisfied connection. */
  editing: boolean;
}

export interface PlaygroundSteps {
  /** Connect is a step only while it still needs something from you. */
  connectIsStep: boolean;
  run: number;
  connect: number;
  configure: number;
  add: number;
  tryIt: number;
}

export function playgroundSteps(state: ConnectionState): PlaygroundSteps {
  const satisfied =
    state.agentRunning && !state.dirty && (state.auth === "guest" || state.hasToken);
  const connectIsStep = !satisfied || state.editing;

  return {
    connectIsStep,
    run: 1,
    connect: 2,
    configure: connectIsStep ? 3 : 2,
    add: connectIsStep ? 4 : 3,
    tryIt: connectIsStep ? 5 : 4,
  };
}
