"""Tests for tui_gateway JSON-RPC protocol plumbing."""

import io
import json
import sys
import threading
import time
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_original_stdout = sys.stdout


@pytest.fixture(autouse=True)
def _restore_stdout():
    yield
    sys.stdout = _original_stdout


@pytest.fixture()
def server():
    with patch.dict("sys.modules", {
        "hermes_constants": MagicMock(get_hermes_home=MagicMock(return_value="/tmp/hermes_test")),
        "hermes_cli.env_loader": MagicMock(),
        "hermes_cli.banner": MagicMock(),
        "hermes_state": MagicMock(),
    }):
        import importlib
        mod = importlib.import_module("tui_gateway.server")
        yield mod
        # Reset module-level session state without re-importing. importlib.reload
        # would re-register the module's atexit hooks (ThreadPoolExecutor
        # shutdown, _shutdown_sessions); the duplicates race the stderr
        # buffer at interpreter shutdown and surface as Fatal Python error:
        # _enter_buffered_busy. Clearing the per-session dicts gives the
        # next test a clean slate; _methods is NOT cleared because it's
        # populated at module import time and re-registration only happens
        # via reload (which we don't do).
        mod._sessions.clear()
        mod._pending.clear()
        mod._answers.clear()


@pytest.fixture()
def capture(server):
    """Redirect server's real stdout to a StringIO and return (server, buf)."""
    buf = io.StringIO()
    server._real_stdout = buf
    return server, buf


# ── JSON-RPC envelope ────────────────────────────────────────────────


def test_unknown_method(server):
    resp = server.handle_request({"id": "1", "method": "bogus"})
    assert resp["error"]["code"] == -32601


def test_ok_envelope(server):
    assert server._ok("r1", {"x": 1}) == {
        "jsonrpc": "2.0", "id": "r1", "result": {"x": 1},
    }


def test_err_envelope(server):
    assert server._err("r2", 4001, "nope") == {
        "jsonrpc": "2.0", "id": "r2", "error": {"code": 4001, "message": "nope"},
    }


# ── write_json ───────────────────────────────────────────────────────


def test_write_json(capture):
    server, buf = capture
    assert server.write_json({"test": True})
    assert json.loads(buf.getvalue()) == {"test": True}


def test_write_json_broken_pipe(server):
    class _Broken:
        def write(self, _): raise BrokenPipeError
        def flush(self): raise BrokenPipeError

    server._real_stdout = _Broken()
    assert server.write_json({"x": 1}) is False


def test_write_json_closed_stream_returns_false(server):
    """ValueError ('I/O on closed file') used to bubble up; treat as gone."""

    class _Closed:
        def write(self, _): raise ValueError("I/O operation on closed file")
        def flush(self): raise ValueError("I/O operation on closed file")

    server._real_stdout = _Closed()
    assert server.write_json({"x": 1}) is False


def test_write_json_unicode_encode_error_re_raises(server):
    """A non-UTF-8 stdout encoding raises UnicodeEncodeError (a ValueError
    subclass).  It must NOT be swallowed as 'peer gone' — that would let
    `entry.py` exit cleanly via the False path and hide the real config
    bug.  We re-raise so the existing crash-log infrastructure records it."""

    class _AsciiOnly:
        def write(self, line):
            line.encode("ascii")  # raises UnicodeEncodeError on non-ascii
        def flush(self): pass

    server._real_stdout = _AsciiOnly()
    with pytest.raises(UnicodeEncodeError):
        server.write_json({"msg": "héllo"})


def test_write_json_unrelated_value_error_re_raises(server):
    """Only ValueError('...closed file...') means peer gone.  Other
    ValueErrors are programming errors and must surface."""

    class _BadValue:
        def write(self, _): raise ValueError("something else entirely")
        def flush(self): pass

    server._real_stdout = _BadValue()
    with pytest.raises(ValueError, match="something else entirely"):
        server.write_json({"x": 1})


def test_write_json_non_serializable_payload_re_raises(server):
    """Non-JSON-safe payloads are programming errors — they must NOT be
    silently dropped via the False path (which would trigger a clean exit
    in entry.py and mask the real bug)."""
    import io

    server._real_stdout = io.StringIO()
    with pytest.raises(TypeError):
        server.write_json({"obj": object()})


def test_write_json_peer_gone_oserror_on_flush_returns_false(server):
    """A flush that raises a peer-gone OSError (EPIPE) must not strand
    the lock or crash; it returns False so the dispatcher exits cleanly."""
    import errno

    written = []

    class _FlushPeerGone:
        def write(self, line): written.append(line)
        def flush(self): raise OSError(errno.EPIPE, "broken pipe")

    server._real_stdout = _FlushPeerGone()
    assert server.write_json({"x": 1}) is False
    assert written and json.loads(written[0]) == {"x": 1}


def test_write_json_non_peer_gone_oserror_re_raises(server):
    """Host I/O failures (ENOSPC, EACCES, EIO …) are NOT peer-gone — they
    must re-raise so the crash log records them instead of looking like
    a clean disconnect via the False path."""
    import errno

    class _DiskFull:
        def write(self, _): raise OSError(errno.ENOSPC, "no space left")
        def flush(self): pass

    server._real_stdout = _DiskFull()
    with pytest.raises(OSError, match="no space"):
        server.write_json({"x": 1})


def test_write_json_skips_flush_when_disable_flush_true(monkeypatch):
    """`StdioTransport` skips flush when `_DISABLE_FLUSH` is true.

    Tests the runtime *behaviour* via direct module-attr patch.  The env
    var → module constant wiring is covered by the dedicated env test
    below; reloading server.py here would re-register atexit hooks and
    recreate the worker pool.
    """
    import importlib

    transport_mod = importlib.import_module("tui_gateway.transport")
    monkeypatch.setattr(transport_mod, "_DISABLE_FLUSH", True)

    flushed = {"count": 0}
    written = []

    class _Stream:
        def write(self, line): written.append(line)
        def flush(self): flushed["count"] += 1

    stream = _Stream()
    transport = transport_mod.StdioTransport(lambda: stream, threading.Lock())

    assert transport.write({"x": 1}) is True
    assert flushed["count"] == 0


def test_disable_flush_env_var_actually_wires_to_module_constant(monkeypatch):
    """End-to-end: setting `HERMES_TUI_GATEWAY_NO_FLUSH=1` and importing
    `tui_gateway.transport` fresh actually flips `_DISABLE_FLUSH` true.

    Reloads only the transport module — server.py is untouched so its
    atexit hooks/worker pool stay intact."""
    import importlib

    monkeypatch.setenv("HERMES_TUI_GATEWAY_NO_FLUSH", "1")
    transport_mod = importlib.reload(importlib.import_module("tui_gateway.transport"))

    try:
        assert transport_mod._DISABLE_FLUSH is True
    finally:
        # Restore the env-disabled state so other tests see the default.
        monkeypatch.delenv("HERMES_TUI_GATEWAY_NO_FLUSH", raising=False)
        importlib.reload(transport_mod)


# ── _emit ────────────────────────────────────────────────────────────


def test_emit_with_payload(capture):
    server, buf = capture
    server._emit("test.event", "s1", {"key": "val"})
    msg = json.loads(buf.getvalue())

    assert msg["method"] == "event"
    assert msg["params"]["type"] == "test.event"
    assert msg["params"]["session_id"] == "s1"
    assert msg["params"]["payload"]["key"] == "val"


def test_emit_without_payload(capture):
    server, buf = capture
    server._emit("ping", "s2")

    assert "payload" not in json.loads(buf.getvalue())["params"]


# ── Blocking prompt round-trip ───────────────────────────────────────


def test_block_and_respond(capture):
    server, _ = capture
    result = [None]

    threading.Thread(
        target=lambda: result.__setitem__(0, server._block("test.prompt", "s1", {"q": "?"}, timeout=5)),
    ).start()

    for _ in range(100):
        if server._pending:
            break
        threading.Event().wait(0.01)

    rid = next(iter(server._pending))
    server._answers[rid] = "my_answer"
    # _pending values are (sid, Event) tuples — unpack to set the Event
    _, ev = server._pending[rid]
    ev.set()

    threading.Event().wait(0.1)
    assert result[0] == "my_answer"


def test_clear_pending(server):
    ev = threading.Event()
    # _pending values are (sid, Event) tuples
    server._pending["r1"] = ("sid-x", ev)
    server._clear_pending()

    assert ev.is_set()
    assert server._answers["r1"] == ""


# ── Session lookup ───────────────────────────────────────────────────


def test_sess_missing(server):
    _, err = server._sess({"session_id": "nope"}, "r1")
    assert err["error"]["code"] == 4001


def test_sess_found(server):
    server._sessions["abc"] = {"agent": MagicMock()}
    s, err = server._sess({"session_id": "abc"}, "r1")

    assert s is not None
    assert err is None


# ── session.resume payload ────────────────────────────────────────────


def test_session_resume_returns_hydrated_messages(server, monkeypatch):
    class _DB:
        def get_session(self, _sid):
            return {"id": "20260409_010101_abc123"}

        def get_session_by_title(self, _title):
            return None

        def reopen_session(self, _sid):
            return None

        def get_messages_as_conversation(self, _sid, include_ancestors=False):
            return [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "yo", "reasoning": "thoughts"},
                {"role": "tool", "content": "searched"},
                {"role": "assistant", "content": "   "},
                {"role": "assistant", "content": None},
                {"role": "narrator", "content": "skip"},
            ]

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(server, "_make_agent", lambda sid, key, session_id=None, session_db=None, **_kwargs: object())
    monkeypatch.setattr(server, "_init_session", lambda sid, key, agent, history, cols=80, **_kwargs: None)
    monkeypatch.setattr(server, "_session_info", lambda _agent, _session=None: {"model": "test/model"})

    resp = server.handle_request(
        {
            "id": "r1",
            "method": "session.resume",
            # eager_build: exercise the synchronous build path (this test
            # monkeypatches _make_agent/_init_session/_session_info).
            "params": {"session_id": "20260409_010101_abc123", "cols": 100, "eager_build": True},
        }
    )

    assert "error" not in resp
    assert resp["result"]["message_count"] == 3
    assert resp["result"]["messages"] == [
        {"role": "user", "text": "hello"},
        {"role": "assistant", "text": "yo", "reasoning": "thoughts"},
        {"role": "tool", "name": "tool", "context": ""},
    ]


def test_session_resume_defaults_to_deferred_build(server, monkeypatch):
    """A normal cold resume (no ``eager_build``) must return the full display
    transcript immediately and register an upgradable live session WITHOUT
    building the agent on the response path — that eager build is the
    multi-second switch latency. Deferred is the default; ``eager_build: true``
    opts back into the synchronous path."""

    target = "20260409_010101_abc123"

    class _DB:
        def get_session(self, _sid):
            return {
                "id": target,
                "model": "vendor/cool-model",
                "model_config": {"provider": "vendor"},
            }

        def get_session_by_title(self, _title):
            return None

        def resolve_resume_session_id(self, sid):
            return sid

        def reopen_session(self, _sid):
            return None

        def get_messages_as_conversation(self, _sid, include_ancestors=False):
            return [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "yo"},
            ]

    builds: list = []

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    # The response path must never call _make_agent; route the deferred timer
    # through a recorder so a 50ms fire can't build (or crash) under the test.
    monkeypatch.setattr(
        server, "_make_agent", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no eager build"))
    )
    monkeypatch.setattr(server, "_start_agent_build", lambda sid, session: builds.append(sid))
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)

    resp = server.handle_request(
        {
            "id": "r1",
            "method": "session.resume",
            "params": {"session_id": target, "cols": 100},
        }
    )

    assert "error" not in resp
    result = resp["result"]
    assert result["resumed"] == target
    assert result["session_key"] == target
    assert result["message_count"] == 2
    assert result["messages"] == [
        {"role": "user", "text": "hello"},
        {"role": "assistant", "text": "yo"},
    ]
    # Lazy info contract (same shape session.create returns), with the session's
    # persisted model/provider restored rather than the global default.
    assert result["info"]["lazy"] is True
    assert result["info"]["model"] == "vendor/cool-model"
    assert result["info"]["provider"] == "vendor"
    assert result["info"]["desktop_contract"] == server.DESKTOP_BACKEND_CONTRACT

    sid = result["session_id"]
    session = server._sessions[sid]
    # Registered but not built: agent is None and the resume key is carried so a
    # later prompt.submit / _sess() upgrade continues THIS stored conversation.
    assert session["agent"] is None
    assert session["resume_session_id"] == target
    assert not session["agent_ready"].is_set()
    # Not a watch spectator: a normal deferred resume is a real session.
    assert not session.get("lazy")
    # The persisted runtime identity is stashed for the deferred build so it
    # can't drop the provider ("No LLM provider configured").
    assert session["resume_runtime_overrides"]["model_override"]["model"] == "vendor/cool-model"
    assert server._find_live_session_by_key(target) == (sid, session)


def test_enforce_session_cap_evicts_oldest_detached_only(server, monkeypatch):
    """The LRU cap frees the least-recently-active DETACHED sessions when over
    the limit, and never a live-transport / running / mid-build one."""

    monkeypatch.setattr(server, "_load_cfg", lambda: {"max_live_sessions": 2})
    evicted: list[str] = []
    monkeypatch.setattr(
        server, "_close_session_by_id", lambda sid, end_reason=None: evicted.append(sid)
    )

    def _ready() -> threading.Event:
        ev = threading.Event()
        ev.set()
        return ev

    detached = server._detached_ws_transport
    live = object()  # no _closed attr -> live transport, never evictable

    server._sessions.clear()
    server._sessions.update(
        {
            "old_detached": {"transport": detached, "last_active": 100.0, "agent_ready": _ready()},
            "new_detached": {"transport": detached, "last_active": 300.0, "agent_ready": _ready()},
            "running_detached": {
                "transport": detached,
                "last_active": 50.0,
                "running": True,
                "agent_ready": _ready(),
            },
            "focused_live": {"transport": live, "last_active": 200.0, "agent_ready": _ready()},
        }
    )

    server._enforce_session_cap()

    # 4 sessions, cap 2 -> evict 2. Only detached+idle+built are eligible, oldest
    # first; the running one and the live-transport one are exempt.
    assert evicted == ["old_detached", "new_detached"]


def test_enforce_session_cap_disabled_is_noop(server, monkeypatch):
    monkeypatch.setattr(server, "_load_cfg", lambda: {"max_live_sessions": 0})
    evicted: list[str] = []
    monkeypatch.setattr(
        server, "_close_session_by_id", lambda sid, end_reason=None: evicted.append(sid)
    )
    server._sessions.clear()
    server._sessions.update(
        {
            f"s{i}": {"transport": server._detached_ws_transport, "last_active": float(i)}
            for i in range(5)
        }
    )

    server._enforce_session_cap()

    assert evicted == []


def test_session_resume_handles_multimodal_list_content(server, monkeypatch):
    """A user message persisted with list-shaped multimodal content used to
    crash session resume with ``'list' object has no attribute 'strip'``."""

    multimodal_user = {
        "role": "user",
        "content": [
            {"type": "text", "text": "describe this"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,AAAA"},
            },
        ],
    }
    text_only_assistant = {"role": "assistant", "content": "ok"}

    class _DB:
        def get_session(self, _sid):
            return {"id": "20260502_000000_listcontent"}

        def get_session_by_title(self, _title):
            return None

        def reopen_session(self, _sid):
            return None

        def get_messages_as_conversation(self, _sid, include_ancestors=False):
            return [multimodal_user, text_only_assistant]

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(server, "_make_agent", lambda sid, key, session_id=None, session_db=None, **_kwargs: object())
    monkeypatch.setattr(server, "_init_session", lambda sid, key, agent, history, cols=80, **_kwargs: None)
    monkeypatch.setattr(server, "_session_info", lambda _agent, _session=None: {"model": "test/model"})

    resp = server.handle_request(
        {
            "id": "r1",
            "method": "session.resume",
            "params": {"session_id": "20260502_000000_listcontent", "cols": 100, "eager_build": True},
        }
    )

    assert "error" not in resp
    assert resp["result"]["message_count"] == 2
    # The image_url part is preserved as a raw data URL inside the text so
    # the desktop renderer (which extracts embedded images) sees the same
    # content the optimistic local cache returns. Otherwise the inline
    # image flashes during initial cache hydration and then vanishes when
    # the resume payload overwrites it with cleaned text.
    assert resp["result"]["messages"] == [
        {
            "role": "user",
            "text": "describe this\ndata:image/png;base64,AAAA",
        },
        {"role": "assistant", "text": "ok"},
    ]


def test_session_resume_lazy_registers_watch_session_without_agent(server, monkeypatch):
    """``lazy: true`` (subagent watch windows) must register the live session
    — keyed for the child mirror, on this transport — WITHOUT building an
    agent. The eager build is what made opening a subagent window contend
    with the already-running parent turn."""

    target = "20260612_000000_child99"

    class _DB:
        def get_session(self, _sid):
            return {"id": target}

        def get_session_by_title(self, _title):
            return None

        def reopen_session(self, _sid):
            return None

        def get_messages_as_conversation(self, _sid, include_ancestors=False):
            return [
                {"role": "user", "content": "delegated goal"},
            ]

    def _boom(*_args, **_kwargs):
        raise AssertionError("lazy resume must not build an agent")

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(server, "_make_agent", _boom)

    resp = server.handle_request(
        {
            "id": "r1",
            "method": "session.resume",
            "params": {"session_id": target, "cols": 100, "lazy": True},
        }
    )

    assert "error" not in resp
    result = resp["result"]
    assert result["resumed"] == target
    assert result["session_key"] == target
    assert result["info"]["lazy"] is True
    assert result["info"]["desktop_contract"] == server.DESKTOP_BACKEND_CONTRACT
    assert result["messages"] == [{"role": "user", "text": "delegated goal"}]

    sid = result["session_id"]
    session = server._sessions[sid]
    assert session["agent"] is None
    # The child mirror finds the watch window by stored key.
    assert server._find_live_session_by_key(target) == (sid, session)
    # A later prompt.submit upgrade must continue THIS stored conversation.
    assert session["resume_session_id"] == target
    # No build started: the idle reaper must still be able to evict it, and
    # the live status must not report a never-ending "starting".
    assert not session["agent_ready"].is_set()
    assert server._session_live_status(sid, session) != "starting"
    session["transport"] = server._detached_ws_transport
    far_future = time.time() + 999999
    assert server._session_is_evictable(sid, session, far_future)

    # Resuming again (window refresh) reuses the same live session.
    resp2 = server.handle_request(
        {
            "id": "r2",
            "method": "session.resume",
            "params": {"session_id": target, "cols": 100, "lazy": True},
        }
    )
    assert "error" not in resp2
    assert resp2["result"]["session_id"] == sid
    assert len(server._sessions) == 1


def test_session_resume_lazy_reports_running_for_inflight_child(server, monkeypatch):
    """A watch window attaching to a child mid-delegation must learn the run is
    live from the resume response itself — the child can sit silent inside a
    long tool call, so waiting for the next stream event leaves the window
    looking dead."""

    target = "20260612_000000_child42"

    class _DB:
        def get_session(self, _sid):
            return {"id": target}

        def get_session_by_title(self, _title):
            return None

        def reopen_session(self, _sid):
            return None

        def get_messages_as_conversation(self, _sid, include_ancestors=False):
            return [{"role": "user", "content": "delegated goal"}]

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(
        server, "_make_agent", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no build"))
    )
    server._active_child_runs[target] = time.time()
    try:
        resp = server.handle_request(
            {
                "id": "r1",
                "method": "session.resume",
                "params": {"session_id": target, "cols": 100, "lazy": True},
            }
        )
    finally:
        server._active_child_runs.pop(target, None)

    assert "error" not in resp
    assert resp["result"]["running"] is True
    assert resp["result"]["status"] == "streaming"


def test_session_resume_lazy_tolerates_missing_row_for_active_child(server, monkeypatch):
    """Race regression: a watch window opens on a freshly-spawned subagent and
    resumes BEFORE the child's first run_conversation() flushes its DB row.

    The child relays ``subagent.start`` (carrying child_session_id, which opens
    the window) before ``_ensure_db_session`` writes the row, so
    ``db.get_session(target)`` is momentarily empty. On slower hosts (WSL2) the
    window's lazy resume consistently lands in this gap. It used to hard-fail
    "session not found"; the frontend then 404'd on its REST messages fallback
    and the watch window spun forever. Since the child is provably live
    (``_child_run_active``), the lazy resume must instead register the live
    session with empty history so the mirror can stream the turn.
    """

    target = "20260616_131212_racey"

    class _DB:
        def get_session(self, _sid):
            # Row not flushed yet — the whole point of the race.
            return None

        def get_session_by_title(self, _title):
            return None

        def reopen_session(self, _sid):
            return None

        def get_messages_as_conversation(self, _sid, include_ancestors=False):
            # No rows for an unwritten session.
            return []

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(
        server, "_make_agent", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no build"))
    )
    # Child is live in the relay registry even though its row isn't written.
    server._active_child_runs[target] = time.time()
    try:
        resp = server.handle_request(
            {
                "id": "r1",
                "method": "session.resume",
                "params": {"session_id": target, "cols": 100, "lazy": True},
            }
        )
    finally:
        server._active_child_runs.pop(target, None)

    # The resume must succeed (no "session not found") and register a live,
    # agent-less watch session the mirror can find by stored key.
    assert "error" not in resp
    result = resp["result"]
    assert result["resumed"] == target
    assert result["session_key"] == target
    assert result["info"]["lazy"] is True
    assert result["messages"] == []
    # Live for the mirror; reported running so the window shows a busy state.
    assert result["running"] is True
    assert result["status"] == "streaming"
    sid = result["session_id"]
    assert server._find_live_session_by_key(target) == (sid, server._sessions[sid])
    assert server._sessions[sid]["agent"] is None


def test_session_resume_missing_row_non_lazy_still_errors(server, monkeypatch):
    """The missing-row tolerance is scoped to lazy resumes of an ACTIVE child.
    A normal (non-lazy) resume of a genuinely unknown id must still fail fast
    with "session not found" rather than silently registering an empty session.
    """

    target = "20260616_000000_ghost"

    class _DB:
        def get_session(self, _sid):
            return None

        def get_session_by_title(self, _title):
            return None

    monkeypatch.setattr(server, "_get_db", lambda: _DB())

    # Non-lazy resume, no active child → hard error.
    resp = server.handle_request(
        {
            "id": "r1",
            "method": "session.resume",
            "params": {"session_id": target, "cols": 100},
        }
    )
    assert "error" in resp
    assert "session not found" in resp["error"]["message"].lower()

    # Lazy resume but the child is NOT live → still an error (no live mirror to
    # justify an empty session; this would just be a dead, sessionless window).
    resp2 = server.handle_request(
        {
            "id": "r2",
            "method": "session.resume",
            "params": {"session_id": target, "cols": 100, "lazy": True},
        }
    )
    assert "error" in resp2
    assert "session not found" in resp2["error"]["message"].lower()


def test_session_resume_reuses_existing_live_session(server, monkeypatch):
    """Repeated resume must not allocate duplicate live agents."""

    target = "20260409_010101_abc123"
    created_sids: list[str] = []
    closed_sids: list[str] = []
    first_agent_started = threading.Event()
    agent_can_finish = threading.Event()

    class _DB:
        def get_session(self, _sid):
            return {"id": target}

        def get_session_by_title(self, _title):
            return None

        def reopen_session(self, _sid):
            return None

        def get_messages_as_conversation(self, _sid, include_ancestors=False):
            return [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "yo"},
            ]

    class _Worker:
        def close(self):
            pass

    class _Agent:
        def __init__(self, sid, session_id):
            self.sid = sid
            self.model = "test/model"
            self.session_id = session_id

        def close(self):
            closed_sids.append(self.sid)

    def make_agent(sid, key, session_id=None, session_db=None, **_kwargs):
        created_sids.append(sid)
        first_agent_started.set()
        assert agent_can_finish.wait(timeout=1)
        return _Agent(sid, session_id or key)

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(server, "_make_agent", make_agent)
    monkeypatch.setattr(server, "_SlashWorker", lambda _key, _model: _Worker())
    monkeypatch.setattr(
        server,
        "_start_notification_poller",
        lambda _sid, _session: threading.Event(),
    )
    monkeypatch.setattr(server, "_notify_session_boundary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_session_info",
        lambda _agent, _session=None: {"model": "test/model"},
    )

    fake_approval = types.SimpleNamespace(
        load_permanent_allowlist=lambda: None,
        register_gateway_notify=lambda *_args, **_kwargs: None,
    )

    with patch.dict(sys.modules, {"tools.approval": fake_approval}):
        first_holder = {}

        def resume_first():
            first_holder["resp"] = server.handle_request(
                {
                    "id": "first",
                    "method": "session.resume",
                    # eager_build: this test drives the synchronous build race +
                    # double-checked locking that only the eager path exercises.
                    "params": {"session_id": target, "cols": 100, "eager_build": True},
                }
            )

        first_thread = threading.Thread(target=resume_first)
        first_thread.start()
        assert first_agent_started.wait(timeout=1)

        second_holder = {}

        def resume_second():
            second_holder["resp"] = server.handle_request(
                {
                    "id": "second",
                    "method": "session.resume",
                    "params": {"session_id": target, "cols": 120, "eager_build": True},
                }
            )

        second_thread = threading.Thread(target=resume_second)
        second_thread.start()
        agent_can_finish.set()

        first_thread.join(timeout=1)
        second_thread.join(timeout=1)
        assert not first_thread.is_alive()
        assert not second_thread.is_alive()
        first = first_holder["resp"]
        second = second_holder["resp"]

    assert "error" not in first
    assert "error" not in second
    # Both resumes resolve to the SAME single live session — the core invariant.
    assert second["result"]["session_id"] == first["result"]["session_id"]
    assert len(server._sessions) == 1
    assert [s.get("session_key") for s in server._sessions.values()].count(target) == 1
    winner = first["result"]["session_id"]
    # The agent build happens outside the resume lock, so a racing resume may
    # build a redundant agent; double-checked locking keeps only one live
    # session and closes any loser's agent (no worker/poller is wired for it).
    assert winner in created_sids
    survivors = [sid for sid in created_sids if sid not in closed_sids]
    assert survivors == [winner]
    assert all(sid == winner for sid in server._sessions)


def test_session_resume_reuses_live_agent_after_compression_rotation(server, monkeypatch):
    """Resume must match the live agent's current session_id, not stale session_key."""

    target = "20260409_020202_child"
    stale_parent = "20260409_010101_parent"
    sid = "live-rotated"
    server._sessions[sid] = {
        "agent": types.SimpleNamespace(model="test/model", session_id=target),
        "created_at": 123.0,
        "display_history_prefix": [],
        "history": [{"role": "assistant", "content": "live child"}],
        "history_lock": threading.RLock(),
        "last_active": 123.0,
        "running": False,
        "session_key": stale_parent,
        "transport": server._stdio_transport,
    }

    class _DB:
        def get_session(self, _sid):
            return {"id": target}

        def get_session_by_title(self, _title):
            return None

        def resolve_resume_session_id(self, _target):
            return target

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_session_info",
        lambda _agent, _session=None: {"model": "test/model"},
    )

    result = server.handle_request(
        {
            "id": "r1",
            "method": "session.resume",
            "params": {"session_id": target, "cols": 100},
        }
    )

    assert "error" not in result
    assert result["result"]["session_id"] == sid
    assert result["result"]["session_key"] == target
    assert len(server._sessions) == 1


def test_sync_session_key_after_compress_reanchors_active_session_lease(
    server, monkeypatch, tmp_path
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli.active_sessions import (
        active_session_registry_snapshot,
        try_acquire_active_session,
    )

    lease, message = try_acquire_active_session(
        session_id="session-old",
        surface="tui",
        config={"max_concurrent_sessions": 1},
        metadata={"live_session_id": "ui-1"},
    )
    assert message is None
    assert lease is not None

    session = {
        "active_session_lease": lease,
        "agent": types.SimpleNamespace(session_id="session-new"),
        "session_key": "session-old",
    }
    fake_approval = types.SimpleNamespace(
        disable_session_yolo=lambda *_args, **_kwargs: None,
        enable_session_yolo=lambda *_args, **_kwargs: None,
        is_session_yolo_enabled=lambda *_args, **_kwargs: False,
        register_gateway_notify=lambda *_args, **_kwargs: None,
        unregister_gateway_notify=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *_args, **_kwargs: None)

    with patch.dict(sys.modules, {"tools.approval": fake_approval}):
        server._sync_session_key_after_compress("ui-1", session)

    snapshot = active_session_registry_snapshot()
    assert session["session_key"] == "session-new"
    assert lease.session_id == "session-new"
    assert [entry["session_id"] for entry in snapshot] == ["session-new"]
    lease.release()


def test_session_resume_live_payload_uses_current_history_with_ancestors(server, monkeypatch):
    """Live resume should not reuse a stale ancestor-inclusive snapshot."""

    target = "20260409_010101_child"
    ancestor_history = [{"role": "user", "content": "ancestor"}]
    current_history = [
        {"role": "user", "content": "current"},
        {"role": "assistant", "content": "current reply"},
    ]

    class _DB:
        def get_session(self, _sid):
            return {"id": target}

        def get_session_by_title(self, _title):
            return None

        def reopen_session(self, _sid):
            return None

        def get_messages_as_conversation(self, _sid, include_ancestors=False):
            if include_ancestors:
                return ancestor_history + current_history
            return list(current_history)

    class _Worker:
        def close(self):
            pass

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(
        server,
        "_make_agent",
        lambda _sid, key, session_id=None, session_db=None, **_kwargs: types.SimpleNamespace(
            model="test/model", session_id=session_id or key
        ),
    )
    monkeypatch.setattr(server, "_SlashWorker", lambda _key, _model: _Worker())
    monkeypatch.setattr(
        server,
        "_start_notification_poller",
        lambda _sid, _session: threading.Event(),
    )
    monkeypatch.setattr(server, "_notify_session_boundary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_session_info",
        lambda _agent, _session=None: {"model": "test/model"},
    )

    fake_approval = types.SimpleNamespace(
        load_permanent_allowlist=lambda: None,
        register_gateway_notify=lambda *_args, **_kwargs: None,
    )

    with patch.dict(sys.modules, {"tools.approval": fake_approval}):
        first = server.handle_request(
            {
                "id": "first",
                "method": "session.resume",
                "params": {"session_id": target, "cols": 100},
            }
        )

        assert "error" not in first
        sid = first["result"]["session_id"]
        assert first["result"]["messages"] == [
            {"role": "user", "text": "ancestor"},
            {"role": "user", "text": "current"},
            {"role": "assistant", "text": "current reply"},
        ]

        with server._sessions[sid]["history_lock"]:
            server._sessions[sid]["history"] = current_history + [
                {"role": "user", "content": "new live turn"},
                {"role": "assistant", "content": "new live reply"},
            ]

        second = server.handle_request(
            {
                "id": "second",
                "method": "session.resume",
                "params": {"session_id": target, "cols": 120},
            }
        )

    assert "error" not in second
    assert second["result"]["session_id"] == sid
    assert second["result"]["messages"] == [
        {"role": "user", "text": "ancestor"},
        {"role": "user", "text": "current"},
        {"role": "assistant", "text": "current reply"},
        {"role": "user", "text": "new live turn"},
        {"role": "assistant", "text": "new live reply"},
    ]


def test_session_activate_rebinds_orphaned_ws_session_to_current_transport(server, monkeypatch):
    """Reconnect + activate must reattach a parked live session before orphan reap."""

    class _Transport:
        authenticated_principal = ("dashboard-token", "local-session")
        allow_profile_override = True

        def write(self, _obj):
            return True

    sid = "runtime01"
    old_transport = server._stdio_transport
    new_transport = _Transport()
    server._sessions[sid] = {
        "agent": types.SimpleNamespace(model="test/model"),
        "created_at": 123.0,
        "history": [],
        "history_lock": threading.RLock(),
        "last_active": 123.0,
        "running": False,
        "session_key": "20260409_010101_abc123",
        "transport": old_transport,
    }
    monkeypatch.setattr(server, "current_transport", lambda: new_transport)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(
        server,
        "_session_info",
        lambda _agent, _session=None: {"model": "test/model"},
    )

    resp = server.handle_request(
        {"id": "activate", "method": "session.activate", "params": {"session_id": sid}}
    )

    assert "error" not in resp
    assert resp["result"]["session_id"] == sid
    assert server._sessions[sid]["transport"] is new_transport
    assert not server._ws_session_is_orphaned(server._sessions[sid])


class _PrincipalTransport:
    def __init__(self, subject: str):
        self.authenticated_principal = ("stub", subject)
        self.authorized_profile = "default"
        self.allow_profile_override = False

    def write(self, _obj):
        return True


def _ticket_ws_transport(subject: str = "alice", *, operator: bool = False):
    """Build the real transport class stamped by the WS ticket boundary."""
    from tui_gateway.ws import WSTransport

    return WSTransport(
        MagicMock(),
        MagicMock(),
        peer="ticket-test",
        authenticated_principal=(
            ("dashboard-token", subject) if operator else ("stub", subject)
        ),
        authorized_profile="default",
        allow_profile_override=operator,
    )


def test_authenticated_session_create_binds_principal_and_rejects_cross_profile(
    server, monkeypatch, tmp_path
):
    transport = _PrincipalTransport("alice")
    monkeypatch.setattr(server, "current_transport", lambda: transport)
    monkeypatch.setattr(server, "_default_session_cwd", lambda: str(tmp_path))
    monkeypatch.setattr(server, "_completion_cwd", lambda _params=None: str(tmp_path))
    monkeypatch.setattr(
        server, "_claim_active_session_slot", lambda *_args, **_kwargs: (None, None)
    )
    monkeypatch.setattr(server, "_schedule_agent_build", lambda _sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)

    denied = server.handle_request(
        {
            "id": "deny",
            "method": "session.create",
            "params": {"profile": "work", "source": "desktop"},
        }
    )
    assert denied["error"] == {
        "code": 4033,
        "message": "requested profile is not authorized for this authenticated transport",
    }
    assert server._sessions == {}

    allowed = server.handle_request(
        {"id": "allow", "method": "session.create", "params": {"source": "desktop"}}
    )
    sid = allowed["result"]["session_id"]
    session = server._sessions[sid]
    assert session["authenticated_principal"] == ("stub", "alice")
    assert session["profile"] == "default"
    assert session["transport"] is transport

    # Desktop may echo the workspace it learned from the server.  A matching
    # hint remains compatible, while the handler still derives the actual cwd
    # from server state rather than probing the request value.
    matching = server.handle_request(
        {
            "id": "matching-cwd",
            "method": "session.create",
            "params": {"cwd": str(tmp_path), "source": "desktop"},
        }
    )
    matching_session = server._sessions[matching["result"]["session_id"]]
    assert matching_session["cwd"] == str(tmp_path)
    assert matching_session["explicit_cwd"] is True


def test_public_session_create_rejects_ungranted_cwd_before_filesystem_probe(
    server, monkeypatch, tmp_path
):
    trusted = tmp_path / "trusted"
    foreign = tmp_path / "foreign"
    trusted.mkdir()
    foreign.mkdir()
    monkeypatch.setattr(server, "_default_session_cwd", lambda: str(trusted))
    monkeypatch.setattr(
        server,
        "_completion_cwd",
        lambda _params=None: pytest.fail("denied create reached cwd resolution"),
    )
    monkeypatch.setattr(
        server,
        "_claim_active_session_slot",
        lambda *_args, **_kwargs: pytest.fail("denied create claimed a session slot"),
    )

    denied = _handle_with_transport(
        server,
        _PrincipalTransport("alice"),
        {
            "id": "foreign-create",
            "method": "session.create",
            "params": {"cwd": str(foreign), "source": "desktop"},
        },
    )

    assert denied["error"] == {
        "code": 4033,
        "message": "public cwd override is not authorized; use the server-selected workspace",
    }
    assert server._sessions == {}


def test_public_pet_mutation_rejects_other_existing_profile(
    server, monkeypatch, tmp_path
):
    """A verified public principal cannot turn params.profile into authority."""
    from hermes_cli import profiles as profiles_mod

    work_home = tmp_path / "profiles" / "work"
    work_home.mkdir(parents=True)
    config_path = work_home / "config.yaml"
    config_path.write_text("display:\n  pet:\n    slug: untouched\n", encoding="utf-8")

    monkeypatch.setattr(server, "current_transport", lambda: _PrincipalTransport("alice"))
    monkeypatch.setattr(server, "_current_profile_name", lambda: "default")
    monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda name: str(name))
    monkeypatch.setattr(profiles_mod, "validate_profile_name", lambda _name: None)
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda name: name == "work")
    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda _name: work_home)

    denied = server.handle_request(
        {
            "id": "pet-deny",
            "method": "pet.select",
            "params": {"profile": "work", "slug": "hijacked"},
        }
    )

    assert denied["error"] == {
        "code": 4033,
        "message": "public authenticated transport is not authorized for RPC method: pet.select",
    }
    assert config_path.read_text(encoding="utf-8") == (
        "display:\n  pet:\n    slug: untouched\n"
    )


def _owned_live_session(server, owner: str, *, running: bool = False) -> dict:
    return {
        "agent": types.SimpleNamespace(model="test/model", interrupt=lambda: None),
        "authenticated_principal": ("stub", owner),
        "cols": 80,
        "created_at": 123.0,
        "history": [{"role": "user", "content": "hello"}],
        "history_lock": threading.RLock(),
        "last_active": 123.0,
        "profile": "default",
        "profile_home": None,
        "running": running,
        "session_key": "20260713_010101_owned",
        "source": "desktop",
        "transport": _PrincipalTransport(owner),
    }


def test_public_tools_configure_requires_owned_live_session(server, monkeypatch):
    """Only a verified live session may select the profile config to mutate."""
    transport = _PrincipalTransport("alice")
    monkeypatch.setattr(server, "current_transport", lambda: transport)

    denied_missing = server.handle_request(
        {
            "id": "missing",
            "method": "tools.configure",
            "params": {"action": "disable", "names": ["web"]},
        }
    )
    denied_unknown = server.handle_request(
        {
            "id": "unknown",
            "method": "tools.configure",
            "params": {
                "session_id": "invented",
                "action": "disable",
                "names": ["web"],
            },
        }
    )

    foreign_sid = "foreign-tools"
    server._sessions[foreign_sid] = _owned_live_session(server, "bob")
    denied_foreign = server.handle_request(
        {
            "id": "foreign",
            "method": "tools.configure",
            "params": {
                "session_id": foreign_sid,
                "action": "disable",
                "names": ["web"],
            },
        }
    )

    assert denied_missing["error"]["code"] == 4033
    assert denied_unknown["error"]["code"] == 4033
    assert denied_foreign["error"] == {
        "code": 4033,
        "message": "session belongs to a different authenticated principal",
    }


def _handle_with_transport(server, transport, request):
    token = server.bind_transport(transport)
    try:
        return server.handle_request(request)
    finally:
        server.reset_transport(token)


@pytest.mark.parametrize(
    ("method", "params"),
    [
        ("shell.exec", {"command": "printf should-not-run"}),
        ("cli.exec", {"argv": ["status"]}),
        ("model.save_key", {"provider": "openai", "key": "secret"}),
        ("model.disconnect", {"provider": "openai"}),
        ("process.stop", {}),
        ("cron.manage", {"action": "add", "name": "bad", "schedule": "* * * * *"}),
        ("plugins.manage", {"action": "toggle", "name": "demo", "enable": True}),
        ("learning.delete", {"id": "memory:1"}),
        ("learning.edit", {"id": "memory:1", "content": "changed"}),
        ("projects.list", {}),
        ("projects.tree", {"preview_limit": 3}),
        ("projects.project_sessions", {"project_id": "private-project"}),
    ],
    ids=(
        "shell",
        "cli",
        "credential-write",
        "credential-delete",
        "process-kill-all",
        "cron-mutation",
        "plugin-mutation",
        "learning-delete",
        "learning-edit",
        "projects-list",
        "projects-tree",
        "project-sessions",
    ),
)
def test_public_ticket_denies_sessionless_operator_rpc(server, method, params):
    response = _handle_with_transport(
        server,
        _PrincipalTransport("alice"),
        {"id": method, "method": method, "params": params},
    )

    assert response["error"] == {
        "code": 4033,
        "message": f"public authenticated transport is not authorized for RPC method: {method}",
    }


@pytest.mark.parametrize(
    "method",
    ["projects.list", "projects.tree", "projects.project_sessions"],
)
def test_operator_transport_keeps_project_reads(server, monkeypatch, method):
    called = []

    def project_read(rid, params):
        called.append(dict(params))
        return server._ok(rid, {"operator": True})

    monkeypatch.setitem(server._methods, method, project_read)
    response = _handle_with_transport(
        server,
        types.SimpleNamespace(
            authenticated_principal=("dashboard-token", "local"),
            authorized_profile="default",
            allow_profile_override=True,
        ),
        {"id": method, "method": method, "params": {"probe": method}},
    )

    assert response["result"] == {"operator": True}
    assert called == [{"probe": method}]


def test_public_ticket_denies_global_yolo_even_with_owned_live_session(
    server, monkeypatch
):
    sid = "owned-yolo"
    server._sessions[sid] = _owned_live_session(server, "alice")

    class _NoRowsDB:
        def session_exists(self, _target):
            return False

    monkeypatch.setattr(server, "_get_db", lambda: _NoRowsDB())
    response = _handle_with_transport(
        server,
        _PrincipalTransport("alice"),
        {
            "id": "global-yolo",
            "method": "config.set",
            "params": {
                "key": "yolo",
                "scope": "global",
                "session_id": sid,
                "value": "1",
            },
        },
    )

    assert response["error"] == {
        "code": 4033,
        "message": "public config.set is limited to session-scoped model, fast, reasoning, and yolo changes",
    }


def test_public_ticket_model_and_fast_changes_do_not_persist_profile_config(
    server, monkeypatch
):
    sid = "owned-config"
    session = _owned_live_session(server, "alice")
    session["agent"].service_tier = "priority"
    server._sessions[sid] = session

    class _NoRowsDB:
        def session_exists(self, _target):
            return False

    monkeypatch.setattr(server, "_get_db", lambda: _NoRowsDB())
    captured = {}

    def apply_model_switch(*_args, **kwargs):
        captured["persist_override"] = kwargs.get("persist_override")
        return {"value": "safe/model", "warning": ""}

    monkeypatch.setattr(server, "_apply_model_switch", apply_model_switch)
    monkeypatch.setattr(
        server,
        "_write_config_key",
        lambda *_args, **_kwargs: pytest.fail("public session change wrote profile config"),
    )
    monkeypatch.setattr(server, "_persist_live_session_runtime", lambda _session: None)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_session_info",
        lambda _agent, _session=None: {"model": "safe/model"},
    )

    transport = _PrincipalTransport("alice")
    model = _handle_with_transport(
        server,
        transport,
        {
            "id": "session-model",
            "method": "config.set",
            "params": {
                "key": "model",
                "session_id": sid,
                "value": "safe/model --provider stub",
            },
        },
    )
    fast = _handle_with_transport(
        server,
        transport,
        {
            "id": "session-fast",
            "method": "config.set",
            "params": {"key": "fast", "session_id": sid, "value": "normal"},
        },
    )

    assert model["result"]["value"] == "safe/model"
    assert captured["persist_override"] is False
    assert fast["result"]["value"] == "normal"
    assert session["agent"].service_tier is None
    assert session["create_service_tier_override"] == ""


@pytest.mark.parametrize("method", ["command.dispatch", "slash.exec"])
def test_public_ticket_denies_alias_and_slash_execution_with_owned_session(
    server, monkeypatch, method
):
    sid = "owned-command"
    server._sessions[sid] = _owned_live_session(server, "alice")
    monkeypatch.setattr(
        server,
        "_get_db",
        lambda: pytest.fail("operator-only dispatch must stop before session access"),
    )

    response = _handle_with_transport(
        server,
        _PrincipalTransport("alice"),
        {
            "id": method,
            "method": method,
            "params": {
                "session_id": sid,
                "name": "dangerous-alias",
                "command": "/dangerous-alias",
            },
        },
    )

    assert response["error"] == {
        "code": 4033,
        "message": f"public authenticated transport is not authorized for RPC method: {method}",
    }


def test_public_desktop_command_runs_fixed_builtin_for_owned_session(
    server, monkeypatch
):
    sid = "owned-desktop-command"
    server._sessions[sid] = _owned_live_session(server, "alice")

    class _NoRowsDB:
        def session_exists(self, _target):
            return False

    monkeypatch.setattr(server, "_get_db", lambda: _NoRowsDB())
    monkeypatch.setattr(
        server,
        "_dispatch_dynamic_command",
        lambda *_args, **_kwargs: pytest.fail("fixed Desktop command used dynamic dispatch"),
    )

    response = _handle_with_transport(
        server,
        _PrincipalTransport("alice"),
        {
            "id": "queue",
            "method": "desktop.command",
            "params": {
                "command": "queue continue the owned chat",
                "session_id": sid,
            },
        },
    )

    assert response["result"] == {
        "type": "send",
        "message": "continue the owned chat",
    }


@pytest.mark.parametrize(
    "command",
    [
        "private-quick-command --token secret",
        "private-alias",
        "plugin-command",
        "customer-secret-skill",
        "debug",
        "rollback 1",
        "stop",
        "tools disable terminal",
    ],
)
def test_public_desktop_command_rejects_dynamic_and_operator_forms(
    server, monkeypatch, command
):
    sid = "owned-denied-desktop-command"
    server._sessions[sid] = _owned_live_session(server, "alice")

    class _NoRowsDB:
        def session_exists(self, _target):
            return False

    monkeypatch.setattr(server, "_get_db", lambda: _NoRowsDB())
    monkeypatch.setattr(
        server,
        "_dispatch_dynamic_command",
        lambda *_args, **_kwargs: pytest.fail("denied Desktop command used dynamic dispatch"),
    )

    response = _handle_with_transport(
        server,
        _PrincipalTransport("alice"),
        {
            "id": command,
            "method": "desktop.command",
            "params": {"command": command, "session_id": sid},
        },
    )

    assert response["error"] == {
        "code": 4033,
        "message": f"public desktop command is not authorized: {command.split(maxsplit=1)[0]}",
    }


@pytest.mark.parametrize("command", ["tools", "debug", "rollback", "stop"])
def test_operator_desktop_command_retains_legacy_fallback_signal(
    server, monkeypatch, command
):
    sid = "operator-desktop-command"
    server._sessions[sid] = _owned_live_session(server, "local")
    monkeypatch.setattr(
        server,
        "_dispatch_dynamic_command",
        lambda *_args, **_kwargs: pytest.fail(
            "desktop.command capability probe used dynamic dispatch"
        ),
    )
    operator = types.SimpleNamespace(
        authenticated_principal=("dashboard-token", "local"),
        authorized_profile="default",
        allow_profile_override=True,
    )

    response = _handle_with_transport(
        server,
        operator,
        {
            "id": command,
            "method": "desktop.command",
            "params": {"command": command, "session_id": sid},
        },
    )

    assert response["error"] == {
        "code": 4018,
        "message": f"desktop command is not allowed: {command}",
    }


def test_public_desktop_command_requires_principal_owned_live_session(
    server, monkeypatch
):
    sid = "foreign-desktop-command"
    server._sessions[sid] = _owned_live_session(server, "bob")
    monkeypatch.setattr(
        server,
        "_get_db",
        lambda: pytest.fail("foreign command must stop before session DB access"),
    )

    response = _handle_with_transport(
        server,
        _PrincipalTransport("alice"),
        {
            "id": "foreign",
            "method": "desktop.command",
            "params": {"command": "usage", "session_id": sid},
        },
    )

    assert response["error"] == {
        "code": 4033,
        "message": "session belongs to a different authenticated principal",
    }


def test_public_ticket_denies_new_unclassified_registered_method(server):
    called = False

    def future_mutator(rid, _params):
        nonlocal called
        called = True
        return server._ok(rid, {"mutated": True})

    server._methods["future.mutate"] = future_mutator
    try:
        response = _handle_with_transport(
            server,
            _PrincipalTransport("alice"),
            {"id": "future", "method": "future.mutate", "params": {}},
        )
    finally:
        server._methods.pop("future.mutate", None)

    assert response["error"] == {
        "code": 4033,
        "message": "public authenticated transport is not authorized for RPC method: future.mutate",
    }
    assert called is False


def test_public_allowlists_are_registered_and_disjoint(server):
    classes = (
        server._PUBLIC_READ_ONLY_METHODS,
        server._PUBLIC_OPTIONAL_LIVE_SESSION_READ_METHODS,
        server._PUBLIC_SESSION_BOOTSTRAP_METHODS,
        server._PUBLIC_LIVE_SESSION_METHODS,
        server._PUBLIC_PENDING_REPLY_METHODS,
    )
    assert set().union(*classes) <= set(server._methods)
    for index, methods in enumerate(classes):
        for other in classes[index + 1 :]:
            assert methods.isdisjoint(other)


def test_public_ticket_does_not_expose_profile_home(server, monkeypatch):
    monkeypatch.setattr(server, "_current_profile_name", lambda: "default")
    response = _handle_with_transport(
        server,
        _PrincipalTransport("alice"),
        {"id": "profile", "method": "config.get", "params": {"key": "profile"}},
    )

    assert response["error"] == {
        "code": 4033,
        "message": "public config.get is limited to project status",
    }
    assert "home" not in json.dumps(response)


def test_public_commands_catalog_omits_configured_commands_and_aliases(
    server, monkeypatch
):
    secret_command = "curl -H 'Authorization: Bearer catalog-secret' private"
    secret_target = "/shell deploy --token alias-secret"
    monkeypatch.setattr(server, "_current_profile_name", lambda: "default")
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {
            "quick_commands": {
                "private-deploy": {"type": "exec", "command": secret_command},
                "private-alias": {"type": "alias", "target": secret_target},
            }
        },
    )

    public = _handle_with_transport(
        server,
        _PrincipalTransport("alice"),
        {"id": "public-catalog", "method": "commands.catalog", "params": {}},
    )
    serialized_public = json.dumps(public, sort_keys=True)
    assert "/help" in dict(public["result"]["pairs"])
    assert "User commands" not in {
        category["name"] for category in public["result"]["categories"]
    }
    for private_value in (
        "private-deploy",
        "private-alias",
        secret_command,
        secret_target,
        "catalog-secret",
        "alias-secret",
    ):
        assert private_value not in serialized_public

    operator = _handle_with_transport(
        server,
        types.SimpleNamespace(
            authenticated_principal=("dashboard-token", "local"),
            authorized_profile="default",
            allow_profile_override=True,
        ),
        {"id": "operator-catalog", "method": "commands.catalog", "params": {}},
    )
    operator_pairs = dict(operator["result"]["pairs"])
    assert secret_command in operator_pairs["/private-deploy"]
    assert secret_target in operator_pairs["/private-alias"]


def test_public_command_metadata_omits_profile_skills_and_bundles(
    server, monkeypatch
):
    import importlib

    skill_bundles = importlib.import_module("agent.skill_bundles")
    skill_commands = importlib.import_module("agent.skill_commands")

    calls = {"bundles": 0, "get_skills": 0, "scan_skills": 0}
    secret_path = "/profiles/default/skills/customer-secret/SKILL.md"
    secret_skill = {
        "/customer-secret": {
            "description": "Operate customer codename ORCHID",
            "name": "customer-secret",
            "skill_md_path": secret_path,
        }
    }
    secret_bundle = {
        "/merger-secret": {
            "description": "Private merger bundle BLUEBIRD",
            "name": "merger-secret",
            "path": "/profiles/default/skill-bundles/merger.yaml",
            "skills": ["customer-secret"],
        }
    }

    def scan_skills():
        calls["scan_skills"] += 1
        return secret_skill

    def get_skills():
        calls["get_skills"] += 1
        return secret_skill

    def get_bundles():
        calls["bundles"] += 1
        return secret_bundle

    monkeypatch.setattr(server, "_current_profile_name", lambda: "default")
    monkeypatch.setattr(skill_commands, "scan_skill_commands", scan_skills)
    monkeypatch.setattr(skill_commands, "get_skill_commands", get_skills)
    monkeypatch.setattr(skill_bundles, "get_skill_bundles", get_bundles)

    public_transport = _PrincipalTransport("alice")
    catalog = _handle_with_transport(
        server,
        public_transport,
        {"id": "catalog", "method": "commands.catalog", "params": {}},
    )
    completion = _handle_with_transport(
        server,
        public_transport,
        {
            "id": "completion",
            "method": "complete.slash",
            "params": {"text": "/customer"},
        },
    )

    public_serialized = json.dumps([catalog, completion], sort_keys=True)
    for private_value in (
        "customer-secret",
        "ORCHID",
        "merger-secret",
        "BLUEBIRD",
        secret_path,
    ):
        assert private_value not in public_serialized
    assert calls == {"bundles": 0, "get_skills": 0, "scan_skills": 0}

    operator = types.SimpleNamespace(
        authenticated_principal=("dashboard-token", "local"),
        authorized_profile="default",
        allow_profile_override=True,
    )
    monkeypatch.setattr(server, "current_transport", lambda: operator)
    monkeypatch.setattr(
        server, "_transport_allows_profile_override", lambda _transport=None: True
    )
    operator_catalog = server.handle_request(
        {"id": "operator-catalog", "method": "commands.catalog", "params": {}}
    )
    operator_completion = server.handle_request(
        {
            "id": "operator-completion",
            "method": "complete.slash",
            "params": {"text": "/merger"},
        },
    )
    assert "/customer-secret" in dict(operator_catalog["result"]["pairs"])
    assert any(
        item["display"] == "/merger-secret"
        for item in operator_completion["result"]["items"]
    )
    assert calls == {"bundles": 1, "get_skills": 1, "scan_skills": 1}


def test_public_project_status_never_falls_back_to_profile_config(
    server, monkeypatch, tmp_path
):
    configured = tmp_path / "configured-private-workspace"
    configured.mkdir()
    owned = tmp_path / "owned-workspace"
    owned.mkdir()
    config_reads = []

    def load_cfg():
        config_reads.append(True)
        return {"terminal": {"cwd": str(configured)}}

    monkeypatch.setattr(server, "_current_profile_name", lambda: "default")
    monkeypatch.setattr(server, "_load_cfg", load_cfg)
    monkeypatch.setattr(
        server,
        "_git_branch_for_cwd",
        lambda cwd: "owned-branch" if cwd == str(owned) else "configured-secret",
    )

    denied = _handle_with_transport(
        server,
        _PrincipalTransport("alice"),
        {"id": "no-scope", "method": "config.get", "params": {"key": "project"}},
    )
    assert denied["error"] == {
        "code": 4033,
        "message": "a valid owned live session is required for this public RPC",
    }
    assert str(configured) not in json.dumps(denied)
    assert config_reads == []

    sid = "owned-project"
    session = _owned_live_session(server, "alice")
    session["cwd"] = str(owned)
    server._sessions[sid] = session

    class _NoRowsDB:
        def session_exists(self, _target):
            return False

    monkeypatch.setattr(server, "_get_db", lambda: _NoRowsDB())
    denied_override = _handle_with_transport(
        server,
        _PrincipalTransport("alice"),
        {
            "id": "foreign-project",
            "method": "config.get",
            "params": {
                "cwd": str(configured),
                "key": "project",
                "session_id": sid,
            },
        },
    )
    assert denied_override["error"] == {
        "code": 4033,
        "message": "public cwd override is not authorized; use the owned session workspace",
    }
    assert config_reads == []

    allowed = _handle_with_transport(
        server,
        _PrincipalTransport("alice"),
        {
            "id": "owned",
            "method": "config.get",
            "params": {"key": "project", "session_id": sid},
        },
    )
    assert allowed["result"] == {"cwd": str(owned), "branch": "owned-branch"}
    assert config_reads == []

    operator = _handle_with_transport(
        server,
        types.SimpleNamespace(
            authenticated_principal=("dashboard-token", "local"),
            authorized_profile="default",
            allow_profile_override=True,
        ),
        {"id": "operator", "method": "config.get", "params": {"key": "project"}},
    )
    assert operator["result"] == {
        "cwd": str(configured),
        "branch": "configured-secret",
    }
    assert config_reads


def test_public_path_completion_rejects_client_override_without_probing_it(
    server, monkeypatch, tmp_path
):
    owned = tmp_path / "owned"
    foreign = tmp_path / "foreign"
    owned.mkdir()
    foreign.mkdir()
    (owned / "owned.txt").write_text("owned", encoding="utf-8")
    (foreign / "foreign-secret.txt").write_text("secret", encoding="utf-8")

    sid = "owned-completion"
    session = _owned_live_session(server, "alice")
    session["cwd"] = str(owned)
    server._sessions[sid] = session

    class _NoRowsDB:
        def session_exists(self, _target):
            return False

    monkeypatch.setattr(server, "_get_db", lambda: _NoRowsDB())
    monkeypatch.setattr(
        server,
        "_list_repo_files",
        lambda _root: pytest.fail("denied completion walked a filesystem"),
    )
    response = _handle_with_transport(
        server,
        _PrincipalTransport("alice"),
        {
            "id": "completion",
            "method": "complete.path",
            "params": {
                "cwd": str(foreign),
                "session_id": sid,
                "word": "@file:",
            },
        },
    )

    assert response["error"] == {
        "code": 4033,
        "message": "public cwd override is not authorized; use the owned session workspace",
    }


def test_public_path_completion_accepts_owned_session_workspace(
    server, monkeypatch, tmp_path
):
    owned = tmp_path / "owned"
    owned.mkdir()
    (owned / "owned.txt").write_text("owned", encoding="utf-8")
    sid = "owned-completion"
    session = _owned_live_session(server, "alice")
    session["cwd"] = str(owned)
    server._sessions[sid] = session

    class _NoRowsDB:
        def session_exists(self, _target):
            return False

    monkeypatch.setattr(server, "_get_db", lambda: _NoRowsDB())
    response = _handle_with_transport(
        server,
        _PrincipalTransport("alice"),
        {
            "id": "completion",
            "method": "complete.path",
            "params": {"cwd": str(owned), "session_id": sid, "word": "@file:"},
        },
    )

    assert "owned.txt" in json.dumps(response, sort_keys=True)


@pytest.mark.parametrize(
    "word",
    ["/etc/", "~/", "../../", "escape/"],
    ids=("absolute-etc", "home", "traversal", "symlink"),
)
def test_ticket_path_completion_rejects_every_workspace_escape_before_listing(
    server, monkeypatch, tmp_path, word
):
    owned = tmp_path / "owned"
    child = owned / "child"
    outside = tmp_path / "outside"
    child.mkdir(parents=True)
    outside.mkdir()
    (outside / "foreign-secret.txt").write_text("secret", encoding="utf-8")
    (owned / "escape").symlink_to(outside, target_is_directory=True)

    sid = "ticket-completion-escape"
    session = _owned_live_session(server, "alice")
    session["cwd"] = str(owned)
    server._sessions[sid] = session

    class _NoRowsDB:
        def session_exists(self, _target):
            return False

    monkeypatch.setattr(server, "_get_db", lambda: _NoRowsDB())
    real_isdir = server.os.path.isdir
    real_listdir = server.os.listdir

    def guarded_isdir(path):
        candidate = Path(path).resolve()
        assert candidate == owned.resolve() or candidate.is_relative_to(owned.resolve())
        return real_isdir(path)

    def guarded_listdir(path):
        candidate = Path(path).resolve()
        assert candidate == owned.resolve() or candidate.is_relative_to(owned.resolve())
        return real_listdir(path)

    monkeypatch.setattr(server.os.path, "isdir", guarded_isdir)
    monkeypatch.setattr(server.os, "listdir", guarded_listdir)

    response = _handle_with_transport(
        server,
        _ticket_ws_transport(),
        {
            "id": word,
            "method": "complete.path",
            "params": {"session_id": sid, "word": word},
        },
    )

    assert response["error"] == {
        "code": 4033,
        "message": "public path completion is confined to the owned session workspace",
    }


def test_ticket_path_completion_lists_valid_child_and_skips_outside_symlink(
    server, monkeypatch, tmp_path
):
    owned = tmp_path / "owned"
    child = owned / "child"
    outside = tmp_path / "outside"
    child.mkdir(parents=True)
    outside.mkdir()
    (child / "owned.txt").write_text("owned", encoding="utf-8")
    (outside / "foreign-secret.txt").write_text("secret", encoding="utf-8")
    (child / "outside-link").symlink_to(outside, target_is_directory=True)

    sid = "ticket-completion-valid"
    session = _owned_live_session(server, "alice")
    session["cwd"] = str(owned)
    server._sessions[sid] = session

    class _NoRowsDB:
        def session_exists(self, _target):
            return False

    monkeypatch.setattr(server, "_get_db", lambda: _NoRowsDB())
    response = _handle_with_transport(
        server,
        _ticket_ws_transport(),
        {
            "id": "valid-child",
            "method": "complete.path",
            "params": {"session_id": sid, "word": "@file:child/"},
        },
    )

    serialized = json.dumps(response, sort_keys=True)
    assert "owned.txt" in serialized
    assert "outside-link" not in serialized
    assert "foreign-secret" not in serialized


@pytest.mark.parametrize(
    ("method", "field"),
    [
        ("image.attach", "path"),
        ("pdf.attach", "path"),
        ("file.attach", "path"),
        ("input.detect_drop", "text"),
    ],
)
def test_ticket_raw_attachment_paths_are_denied_before_gateway_resolution(
    server, monkeypatch, tmp_path, method, field
):
    owned = tmp_path / "owned"
    outside = tmp_path / "outside"
    owned.mkdir()
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("gateway secret", encoding="utf-8")
    symlink = owned / "secret-link"
    symlink.symlink_to(secret)

    sid = "ticket-raw-attachment"
    session = _owned_live_session(server, "alice")
    session["cwd"] = str(owned)
    server._sessions[sid] = session

    class _NoRowsDB:
        def session_exists(self, _target):
            return False

    monkeypatch.setattr(server, "_get_db", lambda: _NoRowsDB())
    monkeypatch.setattr(
        server,
        "_resolve_gateway_attachment_path",
        lambda _raw: pytest.fail("public raw path reached gateway resolution"),
    )
    fake_cli = types.ModuleType("cli")
    fake_cli._detect_file_drop = lambda _raw: pytest.fail(
        "public raw path reached drop detection"
    )
    fake_cli._resolve_attachment_path = lambda _raw: pytest.fail(
        "public raw path reached attachment resolution"
    )
    fake_cli._split_path_input = lambda _raw: pytest.fail(
        "public raw path reached path parsing"
    )
    monkeypatch.setitem(sys.modules, "cli", fake_cli)

    paths = ["/etc/passwd", str(secret), "../../outside/secret.txt", str(symlink)]
    for raw_path in paths:
        response = _handle_with_transport(
            server,
            _ticket_ws_transport(),
            {
                "id": f"{method}:{raw_path}",
                "method": method,
                "params": {"session_id": sid, field: raw_path},
            },
        )
        assert response["error"]["code"] == 4033

    assert session.get("attached_images", []) == []
    assert not (owned / ".hermes" / "desktop-attachments").exists()


def test_ticket_file_upload_uses_bytes_and_ignores_existing_gateway_path(
    server, monkeypatch, tmp_path
):
    owned = tmp_path / "owned"
    outside = tmp_path / "outside"
    owned.mkdir()
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("gateway secret", encoding="utf-8")

    sid = "ticket-byte-attachment"
    session = _owned_live_session(server, "alice")
    session["cwd"] = str(owned)
    server._sessions[sid] = session

    class _NoRowsDB:
        def session_exists(self, _target):
            return False

    monkeypatch.setattr(server, "_get_db", lambda: _NoRowsDB())
    monkeypatch.setattr(
        server,
        "_resolve_gateway_attachment_path",
        lambda _raw: pytest.fail("public byte upload probed its path hint"),
    )

    response = _handle_with_transport(
        server,
        _ticket_ws_transport(),
        {
            "id": "byte-upload",
            "method": "file.attach",
            "params": {
                "data_url": "data:text/plain;base64,dXBsb2FkZWQgYnkgYWxpY2U=",
                "name": "note.txt",
                "path": str(secret),
                "session_id": sid,
            },
        },
    )

    staged = owned / ".hermes" / "desktop-attachments" / "note.txt"
    assert response["result"]["uploaded"] is True
    assert response["result"]["ref_text"] == (
        "@file:.hermes/desktop-attachments/note.txt"
    )
    assert staged.read_text(encoding="utf-8") == "uploaded by alice"
    assert secret.read_text(encoding="utf-8") == "gateway secret"


def test_ticket_image_byte_upload_remains_available_for_owned_desktop_session(
    server, monkeypatch, tmp_path
):
    sid = "ticket-image-bytes"
    session = _owned_live_session(server, "alice")
    session["cwd"] = str(tmp_path / "owned")
    Path(session["cwd"]).mkdir()
    server._sessions[sid] = session

    class _NoRowsDB:
        def session_exists(self, _target):
            return False

    monkeypatch.setattr(server, "_get_db", lambda: _NoRowsDB())
    monkeypatch.setattr(server, "_hermes_home", tmp_path / "profile-home")
    monkeypatch.setattr(server, "_allowed_image_extensions", lambda: frozenset({".png"}))

    response = _handle_with_transport(
        server,
        _ticket_ws_transport(),
        {
            "id": "image-bytes",
            "method": "image.attach_bytes",
            "params": {
                "content_base64": "iVBORw0KGgoA=",
                "filename": "desktop.png",
                "session_id": sid,
            },
        },
    )

    written = Path(response["result"]["path"])
    assert response["result"]["attached"] is True
    assert written.parent == tmp_path / "profile-home" / "images"
    assert written.read_bytes().startswith(b"\x89PNG")
    assert session["attached_images"] == [str(written)]


def test_ticket_file_upload_rejects_attachment_directory_symlink_escape(
    server, monkeypatch, tmp_path
):
    owned = tmp_path / "owned"
    outside = tmp_path / "outside"
    (owned / ".hermes").mkdir(parents=True)
    outside.mkdir()
    (owned / ".hermes" / "desktop-attachments").symlink_to(
        outside, target_is_directory=True
    )

    sid = "ticket-byte-staging-symlink"
    session = _owned_live_session(server, "alice")
    session["cwd"] = str(owned)
    server._sessions[sid] = session

    class _NoRowsDB:
        def session_exists(self, _target):
            return False

    monkeypatch.setattr(server, "_get_db", lambda: _NoRowsDB())
    response = _handle_with_transport(
        server,
        _ticket_ws_transport(),
        {
            "id": "staging-symlink",
            "method": "file.attach",
            "params": {
                "data_url": "data:text/plain;base64,c2FmZSBieXRlcw==",
                "name": "note.txt",
                "session_id": sid,
            },
        },
    )

    assert response["error"] == {
        "code": 4033,
        "message": (
            "public attachment staging directory escapes the owned session workspace"
        ),
    }
    assert list(outside.iterdir()) == []


def test_operator_transport_retains_gateway_path_attachment_flow(
    server, monkeypatch, tmp_path
):
    owned = tmp_path / "owned"
    owned.mkdir()
    source = owned / "operator.txt"
    source.write_text("operator path", encoding="utf-8")
    sid = "operator-path-attachment"
    session = _owned_live_session(server, "alice")
    session["cwd"] = str(owned)
    server._sessions[sid] = session
    monkeypatch.setattr(server, "_resolve_gateway_attachment_path", lambda _raw: source)

    response = _handle_with_transport(
        server,
        _ticket_ws_transport("local", operator=True),
        {
            "id": "operator-path",
            "method": "file.attach",
            "params": {"path": str(source), "session_id": sid},
        },
    )

    assert response["result"]["attached"] is True
    assert response["result"]["uploaded"] is False
    assert response["result"]["path"] == str(source)
    assert response["result"]["ref_text"] == "@file:operator.txt"


def test_public_cwd_set_and_preview_override_are_denied_but_operator_cwd_set_works(
    server, monkeypatch, tmp_path
):
    owned = tmp_path / "owned"
    foreign = tmp_path / "foreign"
    owned.mkdir()
    foreign.mkdir()
    sid = "cwd-boundary"
    session = _owned_live_session(server, "alice")
    session["cwd"] = str(owned)
    server._sessions[sid] = session

    class _NoRowsDB:
        def session_exists(self, _target):
            return False

        def update_session_cwd(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(server, "_get_db", lambda: _NoRowsDB())

    public = _PrincipalTransport("alice")
    denied_set = _handle_with_transport(
        server,
        public,
        {
            "id": "cwd-set",
            "method": "session.cwd.set",
            "params": {"cwd": str(foreign), "session_id": sid},
        },
    )
    denied_preview = _handle_with_transport(
        server,
        public,
        {
            "id": "preview",
            "method": "preview.restart",
            "params": {
                "cwd": str(foreign),
                "session_id": sid,
                "url": "http://127.0.0.1:5173",
            },
        },
    )

    for response in (denied_set, denied_preview):
        assert response["error"] == {
            "code": 4033,
            "message": "public cwd override is not authorized; use the owned session workspace",
        }
    assert session["cwd"] == str(owned)

    operator = types.SimpleNamespace(
        authenticated_principal=("dashboard-token", "local"),
        authorized_profile="default",
        allow_profile_override=True,
    )
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    monkeypatch.setattr(server, "_persist_session_git_meta", lambda *_args: None)
    operator_set = _handle_with_transport(
        server,
        operator,
        {
            "id": "operator-cwd-set",
            "method": "session.cwd.set",
            "params": {"cwd": str(foreign), "session_id": sid},
        },
    )
    assert operator_set["result"]["cwd"] == str(foreign)
    assert session["cwd"] == str(foreign)


def test_public_usage_and_readiness_do_not_perform_credentialed_probes(
    server, monkeypatch
):
    from agent import account_usage

    main_mod = types.ModuleType("hermes_cli.main")
    main_mod._has_any_provider_configured = lambda: pytest.fail(
        "public readiness invoked the operator credential check"
    )
    runtime_provider = types.ModuleType("hermes_cli.runtime_provider")
    runtime_provider.resolve_runtime_provider = lambda **_kwargs: pytest.fail(
        "public readiness probed provider runtime"
    )
    monkeypatch.setitem(sys.modules, "hermes_cli.main", main_mod)
    monkeypatch.setitem(
        sys.modules, "hermes_cli.runtime_provider", runtime_provider
    )

    sid = "owned-usage"
    server._sessions[sid] = _owned_live_session(server, "alice")

    class _NoRowsDB:
        def session_exists(self, _target):
            return False

    monkeypatch.setattr(server, "_get_db", lambda: _NoRowsDB())
    monkeypatch.setattr(
        account_usage,
        "nous_credits_lines",
        lambda: pytest.fail("public usage performed a credentialed portal probe"),
    )
    monkeypatch.setattr(server, "_public_provider_configuration_present", lambda: True)
    transport = _PrincipalTransport("alice")
    usage = _handle_with_transport(
        server,
        transport,
        {
            "id": "usage",
            "method": "session.usage",
            "params": {"session_id": sid},
        },
    )
    readiness = _handle_with_transport(
        server,
        transport,
        {"id": "ready", "method": "setup.runtime_check", "params": {}},
    )
    status = _handle_with_transport(
        server,
        transport,
        {"id": "status", "method": "setup.status", "params": {}},
    )

    assert "credits_lines" not in usage["result"]
    assert readiness["result"] == {"ok": True}
    assert status["result"] == {"provider_configured": True}


def test_public_ticket_owned_action_uses_server_session_profile(server, monkeypatch):
    sid = "owned-resize"
    owned = _owned_live_session(server, "alice")
    server._sessions[sid] = owned

    class _NoRowsDB:
        def session_exists(self, _target):
            return False

    monkeypatch.setattr(server, "_get_db", lambda: _NoRowsDB())
    allowed = _handle_with_transport(
        server,
        _PrincipalTransport("alice"),
        {
            "id": "resize",
            "method": "terminal.resize",
            # A contradictory client profile is ignored for authorization; the
            # server-owned live session remains bound to default.
            "params": {"session_id": sid, "profile": "work", "cols": 132},
        },
    )
    assert allowed["result"] == {"cols": 132}

    owned["profile"] = "work"
    denied = _handle_with_transport(
        server,
        _PrincipalTransport("alice"),
        {
            "id": "wrong-profile",
            "method": "terminal.resize",
            "params": {"session_id": sid, "profile": "default", "cols": 80},
        },
    )
    assert denied["error"] == {
        "code": 4033,
        "message": "requested profile is not authorized for this authenticated transport",
    }
    assert owned["cols"] == 132


def test_public_pending_reply_resolves_server_owned_session(server, monkeypatch):
    sid = "owned-prompt"
    server._sessions[sid] = _owned_live_session(server, "alice")
    event = threading.Event()
    server._pending["request-1"] = (sid, event)

    class _NoRowsDB:
        def session_exists(self, _target):
            return False

    monkeypatch.setattr(server, "_get_db", lambda: _NoRowsDB())
    denied = _handle_with_transport(
        server,
        _PrincipalTransport("bob"),
        {
            "id": "foreign-answer",
            "method": "clarify.respond",
            "params": {"request_id": "request-1", "answer": "steal"},
        },
    )
    assert denied["error"]["code"] == 4033
    assert not event.is_set()

    allowed = _handle_with_transport(
        server,
        _PrincipalTransport("alice"),
        {
            "id": "owned-answer",
            "method": "clarify.respond",
            "params": {"request_id": "request-1", "answer": "yes"},
        },
    )
    assert allowed["result"] == {"status": "ok"}
    assert server._answers["request-1"] == "yes"
    assert event.is_set()


@pytest.mark.parametrize(
    "transport",
    [
        types.SimpleNamespace(
            authenticated_principal=("dashboard-token", "local-session"),
            allow_profile_override=True,
        ),
        types.SimpleNamespace(
            authenticated_principal=("stub", "server-internal"),
            allow_profile_override=True,
        ),
        None,
    ],
    ids=("dashboard-token", "internal", "stdio"),
)
def test_operator_transports_keep_sessionless_shell_exec(server, transport):
    request = {
        "id": "operator-shell",
        "method": "shell.exec",
        "params": {"command": "printf operator-ok"},
    }
    response = (
        server.handle_request(request)
        if transport is None
        else _handle_with_transport(server, transport, request)
    )

    assert response["result"]["code"] == 0
    assert response["result"]["stdout"] == "operator-ok"


@pytest.mark.parametrize(
    "transport",
    [
        types.SimpleNamespace(
            authenticated_principal=("dashboard-token", "local"),
            allow_profile_override=True,
        ),
        types.SimpleNamespace(
            authenticated_principal=("internal", "server"),
            allow_profile_override=True,
        ),
        None,
    ],
    ids=("dashboard-token", "internal", "stdio"),
)
def test_operator_tools_configure_remains_sessionless(
    server, monkeypatch, transport
):
    import importlib

    config_mod = importlib.import_module("hermes_cli.config")
    tools_config_mod = importlib.import_module("hermes_cli.tools_config")

    saved = []
    monkeypatch.setattr(server, "current_transport", lambda: transport)
    monkeypatch.setattr(config_mod, "load_config", lambda: {})
    monkeypatch.setattr(config_mod, "save_config", lambda cfg: saved.append(dict(cfg)))
    monkeypatch.setattr(
        tools_config_mod, "CONFIGURABLE_TOOLSETS", [("web", "Web", "")]
    )
    monkeypatch.setattr(
        tools_config_mod,
        "_apply_toolset_change",
        lambda cfg, _platform, targets, action: cfg.update(
            {"changed": (list(targets), action)}
        ),
    )
    monkeypatch.setattr(tools_config_mod, "_get_plugin_toolset_keys", lambda: set())
    monkeypatch.setattr(tools_config_mod, "_get_platform_tools", lambda *_a, **_k: [])

    response = server.handle_request(
        {
            "id": "operator-tools",
            "method": "tools.configure",
            "params": {"action": "disable", "names": ["web"]},
        }
    )

    assert "error" not in response, response
    assert response["result"]["reset"] is False
    assert saved == [{"changed": (["web"], "disable")}]


def test_session_active_list_filters_public_owner_and_authorized_profile(
    server, monkeypatch
):
    alice = _owned_live_session(server, "alice")
    alice["session_key"] = "alice-key"
    alice["pending_title"] = "Alice default title"
    bob = _owned_live_session(server, "bob")
    bob["session_key"] = "bob-key"
    other_profile = _owned_live_session(server, "alice")
    other_profile["profile"] = "work"
    other_profile["session_key"] = "work-private-key"
    other_profile["pending_title"] = "Work private title"
    other_profile["history"] = [
        {"role": "user", "content": "work private preview"}
    ]
    local = _owned_live_session(server, "alice")
    local["session_key"] = "local-key"
    local["authenticated_principal"] = None
    finalized = _owned_live_session(server, "alice")
    finalized["session_key"] = "dead-key"
    finalized["_finalized"] = True
    server._sessions.update(
        {
            "alice": alice,
            "bob": bob,
            "work-private-id": other_profile,
            "local": local,
            "dead": finalized,
        }
    )
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "current_transport", lambda: _PrincipalTransport("alice"))

    response = server.handle_request(
        {"id": "active-public", "method": "session.active_list", "params": {}}
    )

    serialized = json.dumps(response, sort_keys=True)
    assert [row["id"] for row in response["result"]["sessions"]] == ["alice"]
    assert response["result"]["sessions"][0]["session_key"] == "alice-key"
    assert response["result"]["sessions"][0]["title"] == "Alice default title"
    for private_value in (
        "work-private-id",
        "work-private-key",
        "Work private title",
        "work private preview",
    ):
        assert private_value not in serialized


def test_public_model_options_rejects_same_owner_session_from_other_profile(
    server, monkeypatch
):
    default_sid = "alice-default"
    work_sid = "alice-work"
    server._sessions[default_sid] = _owned_live_session(server, "alice")
    server._sessions[work_sid] = _owned_live_session(server, "alice")
    server._sessions[work_sid]["profile"] = "work"
    server._sessions[work_sid]["agent"].model = "work/private-model"

    class _NoRowsDB:
        def session_exists(self, _target):
            return False

    monkeypatch.setattr(server, "_get_db", lambda: _NoRowsDB())
    monkeypatch.setattr(server, "_current_profile_name", lambda: "default")
    calls = []

    def model_options(rid, params):
        calls.append(dict(params))
        return server._ok(rid, {"model": "safe/default-model", "providers": []})

    monkeypatch.setitem(server._methods, "model.options", model_options)
    transport = _PrincipalTransport("alice")

    denied = _handle_with_transport(
        server,
        transport,
        {
            "id": "work-model-options",
            "method": "model.options",
            "params": {"session_id": work_sid},
        },
    )
    assert denied["error"] == {
        "code": 4033,
        "message": "requested profile is not authorized for this authenticated transport",
    }
    assert "work/private-model" not in json.dumps(denied)
    assert calls == []

    allowed = _handle_with_transport(
        server,
        transport,
        {
            "id": "default-model-options",
            "method": "model.options",
            "params": {"session_id": default_sid},
        },
    )
    assert allowed["result"]["model"] == "safe/default-model"
    assert calls == [{"session_id": default_sid}]


def test_public_model_options_is_sanitized_and_cannot_probe_or_refresh(
    server, monkeypatch
):
    from hermes_cli import inventory

    calls = []
    secret_endpoint = "https://user:catalog-secret@private.example/v1"
    ctx = inventory.ConfigContext(
        current_provider="custom:private",
        current_model="private/chat-model",
        current_base_url=secret_endpoint,
        user_providers={},
        custom_providers=[],
    )

    monkeypatch.setattr(server, "_current_profile_name", lambda: "default")
    monkeypatch.setattr(inventory, "load_picker_context", lambda: ctx)

    def build_payload(_ctx, **kwargs):
        calls.append(dict(kwargs))
        return {
            "model": "private/chat-model",
            "provider": "custom:private",
            "providers": [
                {
                    "api_url": secret_endpoint,
                    "auth_type": "api_key",
                    "authenticated": True,
                    "capabilities": {
                        "private/chat-model": {"fast": False, "reasoning": True}
                    },
                    "extra_headers": {"Authorization": "Bearer header-secret"},
                    "is_current": True,
                    "is_user_defined": True,
                    "key_env": "PRIVATE_PROVIDER_TOKEN",
                    "models": ["private/chat-model"],
                    "name": "Private provider",
                    "slug": "custom:private",
                    "source": "profile-config",
                    "total_models": 1,
                    "warning": "credential secret-warning",
                }
            ],
        }

    monkeypatch.setattr(inventory, "build_models_payload", build_payload)
    transport = _PrincipalTransport("alice")
    public = _handle_with_transport(
        server,
        transport,
        {"id": "models", "method": "model.options", "params": {}},
    )

    assert public["result"] == {
        "model": "private/chat-model",
        "provider": "custom:private",
        "providers": [
            {
                "capabilities": {
                    "private/chat-model": {"fast": False, "reasoning": True}
                },
                "is_current": True,
                "is_user_defined": True,
                "models": ["private/chat-model"],
                "name": "Private provider",
                "slug": "custom:private",
                "total_models": 1,
            }
        ],
    }
    serialized = json.dumps(public, sort_keys=True)
    for secret in (
        secret_endpoint,
        "catalog-secret",
        "header-secret",
        "PRIVATE_PROVIDER_TOKEN",
        "secret-warning",
        "profile-config",
    ):
        assert secret not in serialized
    assert calls[-1]["pricing"] is False
    assert calls[-1]["refresh"] is False
    assert calls[-1]["probe_custom_providers"] is False
    assert calls[-1]["probe_current_custom_provider"] is False

    denied_refresh = _handle_with_transport(
        server,
        transport,
        {
            "id": "refresh",
            "method": "model.options",
            "params": {"refresh": True},
        },
    )
    assert denied_refresh["error"] == {
        "code": 4033,
        "message": "public model.options does not permit provider refresh or probing",
    }
    assert len(calls) == 1

    operator = _handle_with_transport(
        server,
        types.SimpleNamespace(
            authenticated_principal=("dashboard-token", "local"),
            authorized_profile="default",
            allow_profile_override=True,
        ),
        {
            "id": "operator-refresh",
            "method": "model.options",
            "params": {"refresh": True},
        },
    )
    assert operator["result"]["providers"][0]["api_url"] == secret_endpoint
    assert calls[-1]["pricing"] is True
    assert calls[-1]["refresh"] is True
    assert calls[-1]["probe_custom_providers"] is True
    assert calls[-1]["probe_current_custom_provider"] is False


@pytest.mark.parametrize(
    "transport",
    [
        types.SimpleNamespace(
            authenticated_principal=("dashboard-token", "local"),
            allow_profile_override=True,
        ),
        types.SimpleNamespace(
            authenticated_principal=("internal", "server"),
            allow_profile_override=True,
        ),
        None,
    ],
    ids=("dashboard-token", "internal", "stdio"),
)
def test_session_active_list_operator_enumerates_all_live_sessions(
    server, monkeypatch, transport
):
    alice = _owned_live_session(server, "alice")
    alice["session_key"] = "alice-key"
    bob = _owned_live_session(server, "bob")
    bob["session_key"] = "bob-key"
    local = _owned_live_session(server, "alice")
    local["session_key"] = "local-key"
    local["authenticated_principal"] = None
    finalized = _owned_live_session(server, "alice")
    finalized["session_key"] = "dead-key"
    finalized["_finalized"] = True
    server._sessions.update(
        {"alice": alice, "bob": bob, "local": local, "dead": finalized}
    )
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "current_transport", lambda: transport)

    response = server.handle_request(
        {"id": "active-operator", "method": "session.active_list", "params": {}}
    )

    assert [row["id"] for row in response["result"]["sessions"]] == [
        "alice",
        "bob",
        "local",
    ]


def test_live_session_reattach_rejects_other_principal_and_accepts_owner(
    server, monkeypatch
):
    sid = "owned01"
    session = _owned_live_session(server, "alice")
    original_transport = session["transport"]
    server._sessions[sid] = session

    class _NoRowsDB:
        def session_exists(self, _target):
            return False

    monkeypatch.setattr(server, "_get_db", lambda: _NoRowsDB())
    monkeypatch.setattr(
        server,
        "_session_info",
        lambda _agent, _session=None: {"model": "test/model"},
    )
    monkeypatch.setattr(server, "current_transport", lambda: _PrincipalTransport("bob"))
    denied = server.handle_request(
        {"id": "deny", "method": "session.activate", "params": {"session_id": sid}}
    )

    assert denied["error"] == {
        "code": 4033,
        "message": "session belongs to a different authenticated principal",
    }
    assert session["transport"] is original_transport

    owner_transport = _PrincipalTransport("alice")
    monkeypatch.setattr(server, "current_transport", lambda: owner_transport)
    allowed = server.handle_request(
        {"id": "allow", "method": "session.activate", "params": {"session_id": sid}}
    )

    assert allowed["result"]["session_id"] == sid
    assert session["transport"] is owner_transport


@pytest.mark.parametrize(
    ("principal", "allow_profile_override"),
    [
        (("dashboard-token", "local-session"), True),
        (("stub", "server-operator"), True),
        (None, False),
    ],
    ids=("dashboard-token", "server-internal", "stdio"),
)
def test_live_session_public_denial_preserves_local_operator_override(
    server, monkeypatch, principal, allow_profile_override
):
    """Public tickets stay owner-bound; trusted local transports stay operators."""
    sid = "operator01"
    session = _owned_live_session(server, "alice")
    original_transport = session["transport"]
    server._sessions[sid] = session

    monkeypatch.setattr(
        server,
        "_get_db",
        lambda: pytest.fail("principal mismatch/override must stop before DB access"),
    )
    monkeypatch.setattr(
        server,
        "_session_info",
        lambda _agent, _session=None: {"model": "test/model"},
    )

    monkeypatch.setattr(server, "current_transport", lambda: _PrincipalTransport("bob"))
    denied = server.handle_request(
        {"id": "deny", "method": "session.activate", "params": {"session_id": sid}}
    )

    assert denied["error"] == {
        "code": 4033,
        "message": "session belongs to a different authenticated principal",
    }
    assert session["transport"] is original_transport

    class _OperatorTransport:
        def __init__(self):
            self.authenticated_principal = principal
            self.allow_profile_override = allow_profile_override

        def write(self, _obj):
            return True

    operator = None if principal is None else _OperatorTransport()
    monkeypatch.setattr(server, "current_transport", lambda: operator)
    allowed = server.handle_request(
        {"id": "allow", "method": "session.activate", "params": {"session_id": sid}}
    )

    assert allowed["result"]["session_id"] == sid
    assert session["transport"] is (operator or server._stdio_transport)


def test_live_session_resume_rejects_other_principal_and_accepts_owner(
    server, monkeypatch
):
    sid = "owned02"
    session = _owned_live_session(server, "alice")
    original_transport = session["transport"]
    server._sessions[sid] = session

    class _DB:
        def get_session(self, _target, authenticated_owner=None):
            if authenticated_owner != ("stub", "alice"):
                return None
            return {"id": session["session_key"]}

        def get_session_by_title(self, _target, authenticated_owner=None):
            return None

        def resolve_resume_session_id(self, target):
            return target

        def session_exists(self, _target):
            return True

        def session_owned_by(self, _target, owner):
            return owner == ("stub", "alice")

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(
        server,
        "_session_info",
        lambda _agent, _session=None: {"model": "test/model"},
    )
    monkeypatch.setattr(server, "current_transport", lambda: _PrincipalTransport("bob"))
    denied = server.handle_request(
        {
            "id": "deny",
            "method": "session.resume",
            "params": {"session_id": session["session_key"]},
        }
    )

    assert denied["error"]["code"] == 4033
    assert session["transport"] is original_transport

    owner_transport = _PrincipalTransport("alice")
    monkeypatch.setattr(server, "current_transport", lambda: owner_transport)
    allowed = server.handle_request(
        {
            "id": "allow",
            "method": "session.resume",
            "params": {"session_id": session["session_key"]},
        }
    )

    assert allowed["result"]["session_id"] == sid
    assert session["transport"] is owner_transport


def test_live_session_submit_rejects_other_principal_and_accepts_owner(
    server, monkeypatch
):
    sid = "owned03"
    session = _owned_live_session(server, "alice", running=True)
    original_transport = session["transport"]
    server._sessions[sid] = session

    monkeypatch.setattr(server, "current_transport", lambda: _PrincipalTransport("bob"))
    denied = server.handle_request(
        {
            "id": "deny",
            "method": "prompt.submit",
            "params": {"session_id": sid, "text": "steal it"},
        }
    )

    assert denied["error"]["code"] == 4033
    assert session["transport"] is original_transport
    assert not session.get("queued_prompt")

    owner_transport = _PrincipalTransport("alice")
    monkeypatch.setattr(server, "current_transport", lambda: owner_transport)
    allowed = server.handle_request(
        {
            "id": "allow",
            "method": "prompt.submit",
            "params": {"session_id": sid, "text": "next turn"},
        }
    )

    assert allowed["result"]["status"] == "queued"
    assert session["queued_prompt"]["text"] == "next turn"
    assert session["transport"] is owner_transport


def test_live_session_branch_rejects_other_principal_before_parent_access(
    server, monkeypatch
):
    sid = "owned04"
    session = _owned_live_session(server, "alice")
    server._sessions[sid] = session
    monkeypatch.setattr(server, "current_transport", lambda: _PrincipalTransport("bob"))
    monkeypatch.setattr(
        server,
        "_get_db",
        lambda: pytest.fail("cross-principal branch must stop before DB access"),
    )

    denied = server.handle_request(
        {"id": "deny", "method": "session.branch", "params": {"session_id": sid}}
    )

    assert denied["error"] == {
        "code": 4033,
        "message": "session belongs to a different authenticated principal",
    }


def test_session_branch_persists_branched_from_marker(server, monkeypatch):
    """TUI /branch must persist a _branched_from marker so the branch stays
    visible in /resume and /sessions.

    Regression for issue #20856: the TUI branch leaves the parent live (it
    never ends it with end_reason='branched'), so list_sessions_rich's legacy
    heuristic never surfaces it — the stable model_config marker is the only
    thing that keeps a TUI branch visible.
    """
    create_calls = []
    init_calls = []

    class _DB:
        def session_exists(self, _key):
            return True

        def session_owned_by(self, _key, owner):
            return owner == ("stub", "alice")

        def get_session_title(self, _key, **_kwargs):
            return "parent-title"

        def get_next_title_in_lineage(self, base, **_kwargs):
            return f"{base} 2"

        def create_session(self, new_key, **kwargs):
            create_calls.append((new_key, kwargs))
            return new_key

        def append_message(self, **_kwargs):
            return None

        def set_session_title(self, _key, _title, **_kwargs):
            return None

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(server, "_resolve_model", lambda: "test/model")
    monkeypatch.setattr(server, "_new_session_key", lambda: "20260101_000001_child0")
    monkeypatch.setattr(
        server,
        "_make_agent",
        lambda _sid, key, session_id=None, session_db=None, **_kwargs: types.SimpleNamespace(
            model="test/model", session_id=session_id or key
        ),
    )
    monkeypatch.setattr(
        server,
        "_init_session",
        lambda *_args, **kwargs: init_calls.append(kwargs),
    )
    monkeypatch.setattr(server, "_set_session_context", lambda *_a, **_k: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_session_cwd", lambda _s: "/tmp/branch-cwd")

    parent_sid = "parent01"
    parent_key = "20260101_000000_parent"
    owner_transport = _PrincipalTransport("alice")
    monkeypatch.setattr(server, "current_transport", lambda: owner_transport)
    server._sessions[parent_sid] = {
        "authenticated_principal": ("stub", "alice"),
        "session_key": parent_key,
        "history": [{"role": "user", "content": "hello"}],
        "history_lock": threading.Lock(),
        "cols": 80,
        "profile": "default",
        "profile_home": None,
        "transport": owner_transport,
    }

    resp = server.handle_request(
        {"id": "b1", "method": "session.branch", "params": {"session_id": parent_sid}}
    )

    assert "error" not in resp, resp
    assert len(create_calls) == 1
    new_key, kwargs = create_calls[0]
    assert new_key == "20260101_000001_child0"
    assert kwargs["parent_session_id"] == parent_key
    assert kwargs["authenticated_owner"] == ("stub", "alice")
    # The marker — without it the branch is invisible in /resume and /sessions.
    assert kwargs["model_config"] == {"_branched_from": parent_key}
    assert init_calls[0]["authenticated_principal"] == ("stub", "alice")
    assert init_calls[0]["profile"] == "default"
    assert init_calls[0]["transport"] is owner_transport


def test_make_agent_accepts_list_system_prompt(server, monkeypatch):
    captured = {}

    class _Agent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.model = kwargs.get("model", "")

    monkeypatch.setitem(sys.modules, "run_agent", types.SimpleNamespace(AIAgent=_Agent))
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        types.SimpleNamespace(
            resolve_runtime_provider=lambda **_kwargs: {
                "provider": "test",
                "base_url": None,
                "api_key": None,
                "api_mode": None,
            }
        ),
    )
    monkeypatch.setattr(server, "_load_cfg", lambda: {"agent": {"system_prompt": ["one", "two"]}})
    monkeypatch.setattr(server, "_resolve_startup_runtime", lambda: ("test/model", "test"))
    monkeypatch.setattr(server, "_get_db", lambda: None)

    server._make_agent("sid", "session-key", session_id="session-key")

    assert captured["ephemeral_system_prompt"] == "one\ntwo"


# ── Config I/O ───────────────────────────────────────────────────────


def test_config_load_missing(server, tmp_path):
    server._hermes_home = tmp_path
    assert server._load_cfg() == {}


def test_config_roundtrip(server, tmp_path):
    server._hermes_home = tmp_path
    server._save_cfg({"model": "test/model"})
    assert server._load_cfg()["model"] == "test/model"


# ── _cli_exec_blocked ────────────────────────────────────────────────


@pytest.mark.parametrize("argv", [
    [],
    ["setup"],
    ["gateway"],
    ["sessions", "browse"],
    ["config", "edit"],
])
def test_cli_exec_blocked(server, argv):
    assert server._cli_exec_blocked(argv) is not None


@pytest.mark.parametrize("argv", [
    ["version"],
    ["sessions", "list"],
])
def test_cli_exec_allowed(server, argv):
    assert server._cli_exec_blocked(argv) is None


# ── slash.exec skill command interception ────────────────────────────


def test_slash_exec_rejects_skill_commands(server):
    """slash.exec must reject skill commands so the TUI falls through to command.dispatch."""
    # Register a mock session
    sid = "test-session"
    server._sessions[sid] = {"session_key": sid, "agent": None}

    # Mock scan_skill_commands to return a known skill
    fake_skills = {"/hermes-agent-dev": {"name": "hermes-agent-dev", "description": "Dev workflow"}}

    with patch("agent.skill_commands.get_skill_commands", return_value=fake_skills):
        resp = server.handle_request({
            "id": "r1",
            "method": "slash.exec",
            "params": {"command": "hermes-agent-dev", "session_id": sid},
        })

    # Should return an error so the TUI's .catch() fires command.dispatch
    assert "error" in resp
    assert resp["error"]["code"] == 4018
    assert "skill command" in resp["error"]["message"]


def test_slash_exec_handles_plugin_commands_in_live_gateway(server):
    """Plugin slash commands return normal slash.exec output without using the worker."""
    sid = "test-session"

    class Worker:
        def __init__(self):
            self.calls = []

        def run(self, cmd):
            self.calls.append(cmd)
            return f"worker:{cmd}"

    worker = Worker()
    server._sessions[sid] = {"session_key": sid, "agent": None, "slash_worker": worker}

    with patch(
        "hermes_cli.plugins.get_plugin_command_handler",
        lambda name: (lambda arg: f"plugin:{arg}") if name == "plugin-cmd" else None,
    ):
        resp = server.handle_request({
            "id": "r-plugin-slash",
            "method": "slash.exec",
            "params": {"command": "plugin-cmd hello", "session_id": sid},
        })

    assert "error" not in resp
    assert resp["result"] == {"output": "plugin:hello"}
    assert worker.calls == []


def test_slash_exec_plugin_lookup_failure_falls_back_to_worker(server):
    """Plugin discovery failures must not break ordinary slash-worker commands."""
    sid = "test-session"

    class Worker:
        def __init__(self):
            self.calls = []

        def run(self, cmd):
            self.calls.append(cmd)
            return f"worker:{cmd}"

    worker = Worker()
    server._sessions[sid] = {"session_key": sid, "agent": None, "slash_worker": worker}

    with patch(
        "hermes_cli.plugins.get_plugin_command_handler",
        side_effect=RuntimeError("discovery boom"),
    ):
        resp = server.handle_request({
            "id": "r-plugin-lookup-failure",
            "method": "slash.exec",
            "params": {"command": "help", "session_id": sid},
        })

    assert "error" not in resp
    assert resp["result"] == {"output": "worker:help"}
    assert worker.calls == ["help"]


def test_slash_exec_plugin_handler_error_returns_output(server):
    """Plugin handler failures return slash output so the TUI does not redispatch."""
    sid = "test-session"

    class Worker:
        def __init__(self):
            self.calls = []

        def run(self, cmd):
            self.calls.append(cmd)
            return f"worker:{cmd}"

    def handler(arg):
        raise RuntimeError(f"handler boom: {arg}")

    worker = Worker()
    server._sessions[sid] = {"session_key": sid, "agent": None, "slash_worker": worker}

    with patch(
        "hermes_cli.plugins.get_plugin_command_handler",
        lambda name: handler if name == "plugin-cmd" else None,
    ):
        resp = server.handle_request({
            "id": "r-plugin-handler-error",
            "method": "slash.exec",
            "params": {"command": "plugin-cmd hello", "session_id": sid},
        })

    assert "error" not in resp
    assert resp["result"] == {"output": "Plugin command error: handler boom: hello"}
    assert worker.calls == []


@pytest.mark.parametrize("cmd", ["retry", "queue hello", "q hello", "steer fix the test", "plan", "learn create a skill from https://example.com/docs"])
def test_slash_exec_routes_pending_input_commands_to_dispatch(server, cmd):
    """slash.exec must route _pending_input commands to command.dispatch
    internally instead of returning the old 4018 "use command.dispatch"
    fallback error (#48848). Some TUI clients failed that client-side
    fallback, dropping the input and surfacing "empty command".

    The contract is that slash.exec produces exactly the response
    command.dispatch would for the same command — no fragile retry hop.
    """
    base, _, arg = cmd.partition(" ")

    def fresh_session():
        return {"session_key": "test-session", "agent": None}

    sid = "test-session"

    # Response from the (new) internal routing in slash.exec.
    server._sessions[sid] = fresh_session()
    routed = server.handle_request({
        "id": "r1",
        "method": "slash.exec",
        "params": {"command": cmd, "session_id": sid},
    })

    # Response from calling command.dispatch directly with the parsed parts.
    server._sessions[sid] = fresh_session()
    direct = server.handle_request({
        "id": "r1",
        "method": "command.dispatch",
        "params": {"name": base, "arg": arg, "session_id": sid},
    })

    # slash.exec must no longer emit the old client-fallback rejection.
    if "error" in routed:
        assert "pending-input command" not in routed["error"]["message"]

    # Internal routing must yield the same payload as command.dispatch.
    assert routed.get("result") == direct.get("result")
    assert routed.get("error") == direct.get("error")


def test_command_dispatch_queue_sends_message(server):
    """command.dispatch /queue returns {type: 'send', message: ...} for the TUI."""
    sid = "test-session"
    server._sessions[sid] = {"session_key": sid}

    resp = server.handle_request({
        "id": "r1",
        "method": "command.dispatch",
        "params": {"name": "queue", "arg": "tell me about quantum computing", "session_id": sid},
    })

    assert "error" not in resp
    result = resp["result"]
    assert result["type"] == "send"
    assert result["message"] == "tell me about quantum computing"


def test_command_dispatch_queue_requires_arg(server):
    """command.dispatch /queue without an argument returns an error."""
    sid = "test-session"
    server._sessions[sid] = {"session_key": sid}

    resp = server.handle_request({
        "id": "r2",
        "method": "command.dispatch",
        "params": {"name": "queue", "arg": "", "session_id": sid},
    })

    assert "error" in resp
    assert resp["error"]["code"] == 4004


def test_command_dispatch_learn_sends_built_prompt(server):
    """command.dispatch /learn returns {type: 'send', message: <built prompt>}
    so the TUI fires a real agent turn (#51829). The CLI handler queues onto
    _pending_input — a queue the TUI slash worker has no reader for — so the
    prompt was silently dropped after the ack. Routing through command.dispatch
    injects the standards-guided prompt as a normal turn instead.
    """
    from agent.learn_prompt import build_learn_prompt

    sid = "test-session"
    server._sessions[sid] = {"session_key": sid}

    arg = "create a skill from https://example.com/docs"
    resp = server.handle_request({
        "id": "r-learn",
        "method": "command.dispatch",
        "params": {"name": "learn", "arg": arg, "session_id": sid},
    })

    assert "error" not in resp
    result = resp["result"]
    assert result["type"] == "send"
    assert result["message"] == build_learn_prompt(arg)


def test_pending_input_commands_includes_learn(server):
    """Guard: _PENDING_INPUT_COMMANDS must list 'learn' — without it slash.exec
    routes /learn to the slash worker, which only prints the ack and drops the
    prompt onto the dead _pending_input queue (#51829)."""
    assert "learn" in server._PENDING_INPUT_COMMANDS


def test_skills_manage_search_uses_tools_hub_sources(server):
    result = type("Result", (), {
        "description": "Build better terminal demos",
        "name": "showroom",
    })()
    auth = MagicMock(return_value="auth")
    router = MagicMock(return_value=["source"])
    search = MagicMock(return_value=[result])
    fake_hub = types.SimpleNamespace(
        GitHubAuth=auth,
        create_source_router=router,
        unified_search=search,
    )

    with patch.dict(sys.modules, {"tools.skills_hub": fake_hub}):
        resp = server.handle_request({
            "id": "skills-search",
            "method": "skills.manage",
            "params": {"action": "search", "query": "showroom"},
        })

    assert "error" not in resp
    assert resp["result"] == {
        "results": [{"description": "Build better terminal demos", "name": "showroom"}]
    }
    auth.assert_called_once_with()
    router.assert_called_once_with("auth")
    search.assert_called_once_with("showroom", ["source"], source_filter="all", limit=20)


def test_command_dispatch_steer_fallback_sends_message(server):
    """command.dispatch /steer with no active agent falls back to send."""
    sid = "test-session"
    server._sessions[sid] = {"session_key": sid, "agent": None}

    resp = server.handle_request({
        "id": "r3",
        "method": "command.dispatch",
        "params": {"name": "steer", "arg": "focus on testing", "session_id": sid},
    })

    assert "error" not in resp
    result = resp["result"]
    assert result["type"] == "send"
    assert result["message"] == "focus on testing"


def test_command_dispatch_retry_finds_last_user_message(server):
    """command.dispatch /retry walks session['history'] to find the last user message."""
    sid = "test-session"
    history = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
    ]
    server._sessions[sid] = {
        "session_key": sid,
        "agent": None,
        "history": history,
        "history_lock": threading.Lock(),
        "history_version": 0,
    }

    resp = server.handle_request({
        "id": "r4",
        "method": "command.dispatch",
        "params": {"name": "retry", "session_id": sid},
    })

    assert "error" not in resp
    result = resp["result"]
    assert result["type"] == "send"
    assert result["message"] == "second question"
    # Verify history was truncated: everything from last user message onward removed
    assert len(server._sessions[sid]["history"]) == 2
    assert server._sessions[sid]["history"][-1]["role"] == "assistant"
    assert server._sessions[sid]["history_version"] == 1


def test_command_dispatch_retry_empty_history(server):
    """command.dispatch /retry with empty history returns error."""
    sid = "test-session"
    server._sessions[sid] = {
        "session_key": sid,
        "agent": None,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
    }

    resp = server.handle_request({
        "id": "r5",
        "method": "command.dispatch",
        "params": {"name": "retry", "session_id": sid},
    })

    assert "error" in resp
    assert resp["error"]["code"] == 4018


def test_command_dispatch_retry_handles_multipart_content(server):
    """command.dispatch /retry extracts text from multipart content lists."""
    sid = "test-session"
    history = [
        {"role": "user", "content": [
            {"type": "text", "text": "analyze this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]},
        {"role": "assistant", "content": "I see the image."},
    ]
    server._sessions[sid] = {
        "session_key": sid,
        "agent": None,
        "history": history,
        "history_lock": threading.Lock(),
        "history_version": 0,
    }

    resp = server.handle_request({
        "id": "r6",
        "method": "command.dispatch",
        "params": {"name": "retry", "session_id": sid},
    })

    assert "error" not in resp
    result = resp["result"]
    assert result["type"] == "send"
    assert result["message"] == "analyze this"


def test_command_dispatch_returns_skill_payload(server):
    """command.dispatch returns structured skill payload for the TUI to send()."""
    sid = "test-session"
    server._sessions[sid] = {"session_key": sid}

    fake_skills = {"/hermes-agent-dev": {"name": "hermes-agent-dev", "description": "Dev workflow"}}
    fake_msg = "Loaded skill content here"

    with patch("agent.skill_commands.scan_skill_commands", return_value=fake_skills), \
         patch("agent.skill_commands.build_skill_invocation_message", return_value=fake_msg):
        resp = server.handle_request({
            "id": "r2",
            "method": "command.dispatch",
            "params": {"name": "hermes-agent-dev", "session_id": sid},
        })

    assert "error" not in resp
    result = resp["result"]
    assert result["type"] == "skill"
    assert result["message"] == fake_msg
    assert result["name"] == "hermes-agent-dev"


def test_command_dispatch_awaits_async_plugin_handler(server):
    async def _handler(arg):
        return f"async:{arg}"

    with patch(
        "hermes_cli.plugins.get_plugin_command_handler",
        lambda name: _handler if name == "async-cmd" else None,
    ):
        resp = server.handle_request({
            "id": "r-plugin",
            "method": "command.dispatch",
            "params": {"name": "async-cmd", "arg": "hello"},
        })

    assert "error" not in resp
    assert resp["result"] == {"type": "plugin", "output": "async:hello"}


# ── dispatch(): pool routing for long handlers (#12546) ──────────────


def test_dispatch_runs_short_handlers_inline(server):
    """Non-long handlers return their response synchronously from dispatch()."""
    server._methods["fast.ping"] = lambda rid, params: server._ok(rid, {"pong": True})

    resp = server.dispatch({"id": "r1", "method": "fast.ping", "params": {}})

    assert resp == {"jsonrpc": "2.0", "id": "r1", "result": {"pong": True}}


def test_dispatch_offloads_long_handlers_and_emits_via_stdout(capture):
    """Long handlers run on the pool and write their response via write_json."""
    server, buf = capture
    server._methods["slash.exec"] = lambda rid, params: server._ok(rid, {"output": "hi"})

    resp = server.dispatch({"id": "r2", "method": "slash.exec", "params": {}})
    assert resp is None

    for _ in range(50):
        if buf.getvalue():
            break
        time.sleep(0.01)

    written = json.loads(buf.getvalue())
    assert written == {"jsonrpc": "2.0", "id": "r2", "result": {"output": "hi"}}


def test_dispatch_long_handler_does_not_block_fast_handler(server):
    """A slow long handler must not prevent a concurrent fast handler from completing."""
    released = threading.Event()
    server._methods["slash.exec"] = lambda rid, params: (released.wait(timeout=5), server._ok(rid, {"done": True}))[1]
    server._methods["fast.ping"] = lambda rid, params: server._ok(rid, {"pong": True})

    t0 = time.monotonic()
    assert server.dispatch({"id": "slow", "method": "slash.exec", "params": {}}) is None

    fast_resp = server.dispatch({"id": "fast", "method": "fast.ping", "params": {}})
    fast_elapsed = time.monotonic() - t0

    assert fast_resp["result"] == {"pong": True}
    assert fast_elapsed < 0.5, f"fast handler blocked for {fast_elapsed:.2f}s behind slow handler"

    released.set()


def test_dispatch_session_compress_does_not_block_fast_handler(server):
    """Manual TUI compaction can take minutes, so it must not block the RPC loop."""
    released = threading.Event()

    def slow_compress(rid, params):
        released.wait(timeout=5)
        return server._ok(rid, {"done": True})

    server._methods["session.compress"] = slow_compress
    server._methods["fast.ping"] = lambda rid, params: server._ok(rid, {"pong": True})

    t0 = time.monotonic()
    assert server.dispatch({"id": "slow", "method": "session.compress", "params": {}}) is None

    fast_resp = server.dispatch({"id": "fast", "method": "fast.ping", "params": {}})
    fast_elapsed = time.monotonic() - t0

    assert fast_resp["result"] == {"pong": True}
    assert fast_elapsed < 0.5, f"fast handler blocked for {fast_elapsed:.2f}s behind session.compress"

    released.set()


def test_dispatch_long_handler_exception_produces_error_response(capture):
    """An exception inside a pool-dispatched handler still yields a JSON-RPC error."""
    server, buf = capture

    def boom(rid, params):
        raise RuntimeError("kaboom")

    server._methods["slash.exec"] = boom

    server.dispatch({"id": "r3", "method": "slash.exec", "params": {}})

    for _ in range(50):
        if buf.getvalue():
            break
        time.sleep(0.01)

    written = json.loads(buf.getvalue())
    assert written["id"] == "r3"
    assert written["error"]["code"] == -32000
    assert "kaboom" in written["error"]["message"]


def test_dispatch_unknown_long_method_still_goes_inline(server):
    """Method name not in _LONG_HANDLERS takes the sync path even if handler is slow."""
    server._methods["some.method"] = lambda rid, params: server._ok(rid, {"ok": True})

    resp = server.dispatch({"id": "r4", "method": "some.method", "params": {}})

    assert resp["result"] == {"ok": True}


@pytest.mark.parametrize("completion_method", ["complete.path", "complete.slash"])
def test_completion_handlers_are_pool_routed(completion_method, server):
    """complete.path/complete.slash must run on the pool, never the reader thread.

    Regression for #21123: completion ran inline, so a slow git ls-files /
    skill-scan blocked prompt.submit and froze the TUI for the 120s RPC timeout.
    """
    assert completion_method in server._LONG_HANDLERS


@pytest.mark.parametrize("completion_method", ["complete.path", "complete.slash"])
def test_slow_completion_does_not_block_fast_handler(completion_method, server):
    """A slow completion RPC must not block a concurrent fast handler (#21123)."""
    released = threading.Event()

    def slow_completion(rid, params):
        released.wait(timeout=5)
        return server._ok(rid, {"items": []})

    server._methods[completion_method] = slow_completion
    server._methods["fast.ping"] = lambda rid, params: server._ok(rid, {"pong": True})

    t0 = time.monotonic()
    assert server.dispatch({"id": "slow", "method": completion_method, "params": {}}) is None

    fast_resp = server.dispatch({"id": "fast", "method": "fast.ping", "params": {}})
    fast_elapsed = time.monotonic() - t0

    assert fast_resp["result"] == {"pong": True}
    assert fast_elapsed < 0.5, f"fast handler blocked for {fast_elapsed:.2f}s behind {completion_method}"

    released.set()
