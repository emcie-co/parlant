// Polls the server's /healthz so the widget can show an offline banner and block
// input while the backend is unreachable. Checks quickly when down (to recover
// fast) and lazily when healthy, confirms a failure before going offline, and pauses
// while the tab is hidden or the browser itself is offline.
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

// How long after a failed check to check again before reporting the server offline.
const RECHECK_MS = 2000;

// How long a tab that was just shown, or a browser that just came back online, has before a
// failed check can report the server offline. A phone's network often takes a few seconds
// to return when its screen turns on, and a failure then says nothing about the server.
const RESUME_GRACE_MS = 5000;

// The browser knows it has no network (airplane mode, no signal): a failed check would
// blame the server for the device's connection.
function browserOffline(): boolean {
  return typeof navigator !== "undefined" && navigator.onLine === false;
}

function getLoop(transport: ParlantTransport, healthyMs: number, downMs: number): HealthLoop {
  const key = transport.baseUrl;
  const existing = loops.get(key);
  if (existing) return existing;

  const abort = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  let recheckTimer: ReturnType<typeof setTimeout> | undefined;
  let checking = false;
  let graceUntil = 0;
  const loop: HealthLoop = {
    online: true,
    subscribers: new Set(),
    stop: () => {
      abort.abort();
      clearTimeout(timer);
      clearTimeout(recheckTimer);
      document.removeEventListener("visibilitychange", onVisibilityChange);
      window.removeEventListener("online", onOnline);
      loops.delete(key);
    },
  };

  const pause = (ms: number) =>
    new Promise<void>((resolve) => {
      recheckTimer = setTimeout(resolve, ms);
    });

  const tick = async () => {
    // A hidden tab is not checked: browsers throttle or freeze its timers and requests,
    // so a check from there says more about the tab than the server. Polling resumes,
    // with an immediate check, when the tab is shown again.
    if (document.hidden || checking) return;
    // While the browser is offline nothing is checked. The `online` event checks at once; the
    // timer covers a browser that comes back without firing it.
    if (browserOffline()) {
      timer = setTimeout(tick, downMs);
      return;
    }
    checking = true;
    let ok = await transport.checkHealth(5000, abort.signal);
    // One failed check is often a network blip, or a phone waking up with no connection
    // yet. Going offline takes a second failure a moment later.
    // Right after the tab is shown or the browser reconnects, the recheck waits out the grace.
    if (!ok && loop.online && !abort.signal.aborted && !document.hidden && !browserOffline()) {
      await pause(Math.max(RECHECK_MS, graceUntil - Date.now()));
      // The tab may have been shown, or the browser reconnected, during the wait.
      while (!abort.signal.aborted && graceUntil > Date.now()) {
        await pause(graceUntil - Date.now());
      }
      if (!abort.signal.aborted && !document.hidden && !browserOffline()) {
        ok = await transport.checkHealth(5000, abort.signal);
      }
    }
    checking = false;
    if (abort.signal.aborted || document.hidden) return;
    // A failure while the browser went offline isn't counted.
    if (!ok && browserOffline()) {
      timer = setTimeout(tick, downMs);
      return;
    }
    loop.online = ok;
    for (const notify of loop.subscribers) notify(ok);
    timer = setTimeout(tick, ok ? healthyMs : downMs);
  };

  function onVisibilityChange() {
    if (!document.hidden) graceUntil = Date.now() + RESUME_GRACE_MS;
    // A check already under way finishes on its own and decides what to schedule.
    if (checking) return;
    clearTimeout(timer);
    if (!document.hidden) tick();
  }

  function onOnline() {
    graceUntil = Date.now() + RESUME_GRACE_MS;
    if (checking) return;
    clearTimeout(timer);
    tick();
  }

  document.addEventListener("visibilitychange", onVisibilityChange);
  window.addEventListener("online", onOnline);
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
