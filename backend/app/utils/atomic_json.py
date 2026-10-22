"""
Crash-safe JSON file writes.
"""

import contextlib
import json
import os
import tempfile
from typing import Any


def write_json_atomic(path: str, data: Any) -> None:
    """Write JSON via a temp file + rename, so an interrupted write never leaves a truncated file."""
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # mkstemp creates 0600; keep the permissions a plain open() would have given.
        os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_path)
        raise
