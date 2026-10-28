"""Small filesystem helpers: atomic writes, hashing, write guards."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def _json_default(o: Any) -> Any:
    if hasattr(o, "item"):  # numpy scalars
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


@contextmanager
def atomic_path(target: Path) -> Iterator[Path]:
    """Yield a temp path next to ``target``; rename it over ``target`` on success."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        yield tmp
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def write_text_atomic(target: Path, text: str) -> None:
    with atomic_path(target) as tmp:
        tmp.write_text(text)


def write_json_atomic(target: Path, obj: Any) -> None:
    write_text_atomic(target, json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True
