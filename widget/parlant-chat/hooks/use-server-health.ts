// Polls the server's /healthz so the widget can show an offline banner and block
// input while the backend is unreachable. Checks quickly when down (to recover
// fast) and lazily when healthy.
//
// The poll loop is a module-level singleton keyed by server URL: every widget instance and
// every StrictMode/HMR remount shares ONE loop and one in-flight request. Without this,
// rapid remounts (React StrictMode, Vite hot updates) each start their own loop and the
// /healthz requests stack up into a flurry.

import { useEffect, useState } from "react";

import type { ParlantTransport } from "../lib/transport";

interface HealthLoop {
  online: boolean;
  subscribers: Set<(online: boolean) => void>;
  stop: () => void;
}

const loops = new Map<string, HealthLoop>();

function getLoop(transport: ParlantTransport, healthyMs: number, downMs: number): HealthLoop {
  const key = transport.baseUrl;
  const existing = loops.get(key);
  if (existing) return existing;

  const abort = new AbortController();
  let timer: ReturnType<typeof setTimeout>;
  const loop: HealthLoop = {
    online: true,
    subscribers: new Set(),
    stop: () => {
      abort.abort();
      clearTimeout(timer);
      loops.delete(key);
    },
  };

  const tick = async () => {
    const ok = await transport.checkHealth(5000, abort.signal);
    if (abort.signal.aborted) return;
    loop.online = ok;
    for (const notify of loop.subscribers) notify(ok);
    timer = setTimeout(tick, ok ? healthyMs : downMs);
  };
  tick();

  loops.set(key, loop);
  return loop;
}

export function useServerHealth(
  transport: ParlantTransport,
  { healthyMs = 20000, downMs = 5000 }: { healthyMs?: number; downMs?: number } = {},
): boolean {
  const [online, setOnline] = useState(true);

  useEffect(() => {
    const loop = getLoop(transport, healthyMs, downMs);
    setOnline(loop.online);
    loop.subscribers.add(setOnline);
    return () => {
      loop.subscribers.delete(setOnline);
      // Stop the shared loop once nothing is listening, so it isn't left polling forever.
      if (loop.subscribers.size === 0) loop.stop();
    };
  }, [transport, healthyMs, downMs]);

  return online;
}

// Stop every loop when this module is hot-replaced, so a stale poller from the previous
// module version can't keep firing after a Vite hot update (the classic HMR leak). Inert
// outside Vite dev.
const hot = (import.meta as { hot?: { dispose: (cb: () => void) => void } }).hot;
if (hot) {
  hot.dispose(() => {
    for (const loop of loops.values()) loop.stop();
    loops.clear();
  });
}
