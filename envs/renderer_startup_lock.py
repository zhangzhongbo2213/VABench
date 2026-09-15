"""Cross-process guard for SAPIEN renderer creation on shared GPU hosts."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import time
from typing import Iterator


RENDERER_STARTUP_LOCK_ENV = "ACTIVE_SPATIAL_RENDERER_STARTUP_LOCK"


@contextmanager
def renderer_startup_lock() -> Iterator[None]:
    """Serialize only the renderer constructor when a shared lock is configured."""

    lock_value = os.environ.get(RENDERER_STARTUP_LOCK_ENV, "").strip()
    if not lock_value:
        yield
        return
    lock_path = Path(lock_value).expanduser().resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        waited_seconds = time.monotonic() - started
        print(
            "renderer_startup_lock_acquired: "
            f"waited_seconds={waited_seconds:.3f} path={lock_path}",
            flush=True,
        )
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
