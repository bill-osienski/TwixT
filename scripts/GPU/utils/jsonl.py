from __future__ import annotations

import fcntl
import json
from pathlib import Path
from typing import Any, Dict, Iterator


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    """Append a JSON object to a JSONL file with file locking.

    Uses fcntl.flock() to prevent race conditions when multiple
    processes write to the same file in parallel.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Open in append mode and acquire exclusive lock
    with path.open("a", encoding="utf-8") as f:
        # LOCK_EX = exclusive lock, blocks until acquired
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(json.dumps(obj, separators=(",", ":"), sort_keys=True))
            f.write("\n")
            f.flush()  # Ensure data is written before releasing lock
        finally:
            # LOCK_UN = unlock
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Iterate over JSON objects in a JSONL file with shared lock.

    Uses a shared lock to allow concurrent reads but block writes
    while reading.
    """
    if not path.exists():
        return iter(())

    with path.open("r", encoding="utf-8") as f:
        # LOCK_SH = shared lock (allows concurrent reads)
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        try:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
