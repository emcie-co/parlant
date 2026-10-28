import { useEffect, useRef, useState } from "react";

/** A chosen chip stays dismissed when this browser resumes the same session. */
export function usePersistentSuggestions(serverUrl: string, agentId: string, sessionId: string | null) {
  const key = sessionId ? `parlant:v1:persistent-suggestions:${encodeURIComponent(serverUrl)}:${encodeURIComponent(agentId)}:${sessionId}` : null;
  const bound = useRef<string | null>(null);
  const pending = useRef(new Set<string>());
  const [chosen, setChosen] = useState(new Set<string>());

  const save = (storageKey: string, values: Set<string>) => {
    try { localStorage.setItem(storageKey, JSON.stringify([...values])); } catch { /* Storage may be disabled. */ }
  };
  useEffect(() => {
    if (!key) { bound.current = null; return; }
    let stored: string[] = [];
    try {
      const value: unknown = JSON.parse(localStorage.getItem(key) ?? "[]");
      if (Array.isArray(value)) stored = value.filter((item): item is string => typeof item === "string");
    } catch { /* Missing or invalid storage starts with all chips available. */ }
    const next = new Set([...stored, ...pending.current]);
    pending.current.clear();
    bound.current = key;
    setChosen(next);
    if (next.size) save(key, next);
  }, [key]);

  const choose = (text: string) => {
    const next = new Set(chosen).add(text);
    setChosen(next);
    if (key && bound.current === key) save(key, next);
    else pending.current.add(text); // First chip can create the lazy session.
  };
  return { chosen, choose };
}
