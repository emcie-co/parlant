import { describe, expect, it } from "vitest";
import { playgroundSteps, type ConnectionState } from "./steps";

const base: ConnectionState = {
  agentRunning: false,
  dirty: false,
  auth: "guest",
  hasToken: false,
  editing: false,
};

describe("playgroundSteps", () => {
  it("asks you to connect when no server is answering", () => {
    const steps = playgroundSteps(base);

    expect(steps.connectIsStep).toBe(true);
    expect([steps.run, steps.connect, steps.configure, steps.add, steps.tryIt]).toEqual([
      1, 2, 3, 4, 5,
    ]);
  });

  it("drops the connect step once the CLI's guest agent is answering", () => {
    const steps = playgroundSteps({ ...base, agentRunning: true });

    expect(steps.connectIsStep).toBe(false);
    // Numbering stays contiguous with the connect step gone.
    expect([steps.run, steps.configure, steps.add, steps.tryIt]).toEqual([1, 2, 3, 4]);
  });

  it("keeps the connect step in signed-in mode until a token is applied", () => {
    const running = { ...base, agentRunning: true, auth: "authenticated" as const };

    expect(playgroundSteps(running).connectIsStep).toBe(true);
    expect(playgroundSteps({ ...running, hasToken: true }).connectIsStep).toBe(false);
  });

  it("reopens the connect step while you edit an applied connection", () => {
    const connected = { ...base, agentRunning: true };

    expect(playgroundSteps({ ...connected, editing: true }).connectIsStep).toBe(true);
    expect(playgroundSteps({ ...connected, dirty: true }).connectIsStep).toBe(true);
  });
});
