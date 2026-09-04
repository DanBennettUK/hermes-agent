"""Bounded local index of recently-sent outbound message text."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from typing import Iterator, Optional

_MAX_ENTRIES = 1000
_MAX_TEXT_CHARS = 2000
_CACHE_LOCK = threading.RLock()
_CACHE = {}


def _store_path() -> str:
    from hermes_constants import get_hermes_home

    return os.path.join(str(get_hermes_home()), "state", "sent_text_index.json")


def _key(chat_id, message_id) -> str:
    return f"{chat_id}:{message_id}"


def _fingerprint(path: str):
    try:
        stat = os.stat(path)
        return stat.st_mtime_ns, stat.st_size
    except OSError:
        return None


def _read(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


@contextmanager
def _exclusive_file_lock(path: str) -> Iterator[None]:
    """Lock the complete read-modify-write cycle across processes."""
    lock_path = f"{path}.lock"
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            os.chmod(lock_path, 0o600)
        except OSError:
            pass
        if os.name == "nt":  # pragma: no cover - exercised on Windows CI
            import msvcrt

            if os.fstat(fd).st_size == 0:
                os.write(fd, b"0")
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _write(path: str, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(
        dir=os.path.dirname(path), prefix=".sent-text-", suffix=".tmp"
    )
    try:
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        fd = -1
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    except BaseException:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def record(chat_id, message_id, text: Optional[str]) -> None:
    """Persist text for a same-chat reply target; no-op on any failure."""
    if not text or not message_id or not chat_id:
        return
    path = _store_path()
    try:
        os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
        with _CACHE_LOCK, _exclusive_file_lock(path):
            # Refresh after locking so concurrent subprocess sends are retained.
            data = _read(path)
            data[_key(chat_id, message_id)] = {
                "t": str(text)[:_MAX_TEXT_CHARS],
                "ts": int(time.time()),
            }
            if len(data) > _MAX_ENTRIES:
                oldest = sorted(data.items(), key=lambda item: item[1].get("ts", 0))[
                    : len(data) - _MAX_ENTRIES
                ]
                for key, _ in oldest:
                    data.pop(key, None)
            _write(path, data)
            _CACHE[path] = (_fingerprint(path), data)
    except Exception:
        return


def lookup(chat_id, message_id) -> Optional[str]:
    """Return text for the exact same-chat key, or None on a miss."""
    if not message_id or not chat_id:
        return None
    path = _store_path()
    try:
        with _CACHE_LOCK:
            fingerprint = _fingerprint(path)
            cached = _CACHE.get(path)
            if cached is None or cached[0] != fingerprint:
                data = _read(path)
                _CACHE[path] = (fingerprint, data)
            else:
                data = cached[1]
            entry = data.get(_key(chat_id, message_id))
            if isinstance(entry, dict):
                return entry.get("t") or None
    except Exception:
        return None
    return None
