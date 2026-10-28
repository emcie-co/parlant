# Hot Reload

Editing your agent script doesn't have to require a manual restart. When hot
reload is on — and it is on by default — the server watches your script's
directory and restarts itself the moment you save a Python file:

```
Hot reload: change detected (/home/dev/my_agent.py); restarting the server
Hot reload: restarting...
```

Save your file, switch to the chat UI, refresh, and talk to the updated
agent. There is nothing else to do.

## What it does

`p.Server(hot_reload=True)` (the default) watches the **directory containing your entry script**, recursively, for changes to Python files. On the first
change it:

1. Logs which file triggered the reload.
2. Shuts the server down through the exact same path as Ctrl-C — in-flight
   work drains, background tasks stop, ports are released.
3. Replaces the process in place with a fresh run of your original command
   line (`uv run my_agent.py`, `python my_agent.py --whatever` — argv is
   preserved verbatim). The process and PID stay the same; the server inside
   it starts completely fresh.

Your script then re-executes from the top: agents and policies are recreated
exactly as on any restart. Because rule indexing is cached on disk
(`indexing_cache.json` in Parlant's home directory, `./parlant-data` unless
you set `PARLANT_HOME`), re-evaluating unchanged policies is fast. The reboot
cost is dominated by plain server startup.

## What it deliberately does not do

- **No in-process patching.** The whole script re-runs. This is the point:
  every kind of edit — renamed agents, deleted policies, changed tools,
  different imports — is handled correctly, because nothing
  is diffed or patched.
- **No crash-loop supervision.** If your edit makes the script crash during
  startup, the process exits — there is nothing left running to watch for
  your fix. Fix the file and start it once manually.

## When it turns itself off

- **No file-backed entry script**: in a REPL, notebook, or embedded
  interpreter there is nothing meaningful to watch (and re-exec would
  restart the host), so the watcher turns itself off.
- **Explicitly**: `p.Server(hot_reload=False)`.

In production nothing special is required: if the container's files never
change, the only cost is an idle file watcher. Set `hot_reload=False` to
remove even that.

## Notes

- The watch root is your *script's* directory — if you run a script that
  lives inside a larger repository, edits anywhere in that tree's Python
  files trigger a reload, including edits to a locally-checked-out Parlant.
- Only `.py` changes trigger reloads (via `watchfiles`' Python filter);
  editing data files, prompts-on-disk, or `.env` does not.
- One reload per process life: after the first change the process is already
  shutting down, and the new process starts its own watcher.
