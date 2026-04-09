import atexit
import json
import os
import time
from threading import Lock

_PROFILE_FILE = os.environ.get("MONGO_PROFILE_FILE", "mongo_profile.jsonl")

_lock = Lock()
_entries: list[dict[str, object]] = []
_start_time = time.time()


def record(layer: str, collection: str, op: str, start: float) -> None:
    elapsed_ms = (time.perf_counter() - start) * 1000
    wall_time = time.time() - _start_time
    entry = {
        "t": round(wall_time, 3),
        "layer": layer,
        "collection": collection,
        "op": op,
        "ms": round(elapsed_ms, 1),
    }
    with _lock:
        _entries.append(entry)


def flush() -> None:
    with _lock:
        if not _entries:
            return
        with open(_PROFILE_FILE, "a") as f:
            for entry in _entries:
                f.write(json.dumps(entry) + "\n")
        _entries.clear()


atexit.register(flush)
