"""US-008: one-shot migration of legacy sessions into canonical events.jsonl.

Table-driven fixtures cover both legacy sources (events.md, events.compact.jsonl)
plus legacy trace.md run entries; asserts idempotency, error routing, and view
parity with the pre-migration data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from splinter.memory.session import Session
from splinter.obs import events
from splinter.obs.trace import Trace


@pytest.fixture
def sess(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Session:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    s = Session("ses_legacy")
    s._ensure_dir()
    return s


_LEGACY_EVENTS_MD = (
    "=== run start · cascade · 21:06:23 ===\n"
    "[21:06:23] session ses_legacy · strategy cascade · 2 task(s)\n"
    "[21:06:24] localizing against the codebase…\n"
)

_LEGACY_TRACE_MD = (
    "# Trace\n\n## Runs\n\n"
    "- iter 0: opus (T0) tokens={'input': 100, 'output': 50} cost=$0.0955 0.0s\n"
    "- task 1 iter 1: sonnet (T1) tokens={'input': 40, 'output': 70} "
    "cost=$0.1096 1.2s @ 2026-07-05T00:10:00+00:00\n"
)


def _errors(session: Session) -> list[dict[str, object]]:
    p = session.dir / "errors.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def test_migrates_log_and_run_from_legacy(sess: Session) -> None:
    (sess.dir / "events.md").write_text(_LEGACY_EVENTS_MD)
    (sess.dir / "trace.md").write_text(_LEGACY_TRACE_MD)

    migrated = events.migrate_legacy(sess)

    assert migrated is True
    assert events.events_path(sess).exists()
    logs = events.load_log_events(sess)
    runs = events.load_run_entries(sess)
    assert len(logs) == 3
    assert len(runs) == 2
    # baseline schema + default task id stamped on migrated records
    assert all(ev.payload["schema_version"] == events.SCHEMA_VERSION for ev in logs)
    assert runs[0].task == 0 and runs[1].task == 1


def test_idempotent_no_duplicate(sess: Session) -> None:
    (sess.dir / "events.md").write_text(_LEGACY_EVENTS_MD)

    assert events.migrate_legacy(sess) is True
    first = events.events_path(sess).read_text()
    # second call is a no-op: events.jsonl already exists
    assert events.migrate_legacy(sess) is False
    assert events.events_path(sess).read_text() == first


def test_malformed_compact_routed_to_errors_not_fatal(sess: Session) -> None:
    (sess.dir / "events.compact.jsonl").write_text(
        '{"ts": "t1", "stage": "s", "message": "good one"}\n'
        "{bad json line\n"
        '{"ts": "t2", "stage": "s", "message": "good two"}\n'
    )

    migrated = events.migrate_legacy(sess)  # must not raise

    assert migrated is True
    logs = events.load_log_events(sess)
    assert [ev.payload["message"] for ev in logs] == ["good one", "good two"]
    errs = _errors(sess)
    assert any(e["source"] == "events.migrate_legacy" and e["op"] == "decode" for e in errs)


def test_view_parity_log_render_matches_legacy(sess: Session) -> None:
    (sess.dir / "events.md").write_text(_LEGACY_EVENTS_MD)
    events.migrate_legacy(sess)

    # render_events_md rebuilds the log verbatim from the migrated raw lines.
    assert sess.render_events_md() == _LEGACY_EVENTS_MD


def test_view_parity_cost_matches_legacy_trace(sess: Session) -> None:
    (sess.dir / "trace.md").write_text(_LEGACY_TRACE_MD)
    events.migrate_legacy(sess)

    trace = Trace.from_jsonl(sess)
    assert trace.total_cost == pytest.approx(0.0955 + 0.1096)
    assert trace.cost_by_model["opus"] == pytest.approx(0.0955)
    assert trace.cost_by_model["sonnet"] == pytest.approx(0.1096)


def test_no_legacy_files_no_migration(sess: Session) -> None:
    assert events.migrate_legacy(sess) is False
    assert not events.events_path(sess).exists()


def test_existing_events_jsonl_not_overwritten(sess: Session) -> None:
    events.append_event(sess, events.Event(type="log", ts="t", payload={"message": "live"}))
    (sess.dir / "events.md").write_text(_LEGACY_EVENTS_MD)

    assert events.migrate_legacy(sess) is False  # canonical file present → skip
    logs = events.load_log_events(sess)
    assert [ev.payload.get("message") for ev in logs] == ["live"]


def test_migration_triggered_lazily_on_read(sess: Session) -> None:
    """A load call migrates transparently — no explicit migrate_legacy needed."""
    (sess.dir / "events.md").write_text(_LEGACY_EVENTS_MD)

    logs = events.load_log_events(sess)  # triggers migration via _read_records

    assert len(logs) == 3
    assert events.events_path(sess).exists()
