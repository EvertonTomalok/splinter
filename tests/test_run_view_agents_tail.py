"""The run screen takes only the bytes appended to attempts.jsonl since its last tick."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any
from unittest.mock import patch

import pytest

import splinter.agents.progress as progress
from splinter.agents.progress import ATTEMPTS, STORIES, AttemptsTail, read_attempts
from splinter.run_view import RunViewBuilder
from splinter.run_view_agents import agents_groups, agents_rows
from tests.fixtures_run_view import FIXTURE_LADDER, build_session


def _start(story: int, n: int, brief: str = "q") -> str:
    record = {
        "event": "start", "story": story, "n": n, "brief": brief, "engine": "mcode", "tier": 0,
    }
    return json.dumps(record) + "\n"


def _end(story: int, n: int, verdict: str = "PASS") -> str:
    return json.dumps({"event": "end", "story": story, "n": n, "verdict": verdict}) + "\n"


def _append(path: Path, text: str) -> None:
    with path.open("a") as handle:
        handle.write(text)


def _parsed(tail: AttemptsTail) -> tuple[int, list[str]]:
    with patch("splinter.agents.progress.json.loads", wraps=json.loads) as loads:
        attempts = tail.read()
    return loads.call_count, [a.verdict for a in attempts]


@pytest.fixture
def log(tmp_path: Path) -> Path:
    return tmp_path / "attempts.jsonl"


def test_a_tick_parses_only_the_appended_lines(log: Path) -> None:
    log.write_text(_start(1, 1) + _end(1, 1, "RETRY") + _start(1, 2))
    tail = AttemptsTail(log)
    assert _parsed(tail) == (3, ["RETRY", ""])
    _append(log, _end(1, 2) + _start(2, 1))
    assert _parsed(tail) == (2, ["RETRY", "PASS", ""])


def test_a_tick_takes_only_the_new_bytes_off_the_disk(log: Path) -> None:
    log.write_text(_start(1, 1) * 200)
    tail = AttemptsTail(log)
    tail.read()
    added = _end(1, 1)
    _append(log, added)
    taken: list[int] = []
    real_open = Path.open

    def spy_open(path: Path, *args: Any, **kwargs: Any) -> IO[Any]:
        handle = real_open(path, *args, **kwargs)
        real_read = handle.read

        def read(*read_args: Any) -> Any:
            data = real_read(*read_args)
            taken.append(len(data))
            return data

        return _Handle(handle, read)

    with patch.object(Path, "open", spy_open):
        tail.read()
    assert sum(taken) <= len(added) + progress._MARK_BYTES


class _Handle:
    def __init__(self, handle: Any, read: Any) -> None:
        self._handle = handle
        self.read = read
        self.seek = handle.seek

    def __enter__(self) -> _Handle:
        return self

    def __exit__(self, *exc: object) -> None:
        self._handle.close()


def test_a_tick_with_no_growth_parses_nothing(log: Path) -> None:
    log.write_text(_start(1, 1))
    tail = AttemptsTail(log)
    tail.read()
    assert _parsed(tail) == (0, [""])


def test_a_shrunk_file_is_parsed_from_zero(log: Path) -> None:
    log.write_text(_start(1, 1) + _end(1, 1) + _start(1, 2) + _end(1, 2))
    tail = AttemptsTail(log)
    tail.read()
    log.write_text(_start(3, 1))
    assert _parsed(tail) == (1, [""])


def test_a_rewritten_longer_file_is_parsed_from_zero(log: Path) -> None:
    log.write_text(_start(1, 1, "first"))
    tail = AttemptsTail(log)
    tail.read()
    log.write_text(_start(2, 1, "other") + _start(2, 2, "other"))
    assert [a.story for a in tail.read()] == [2, 2]


def test_a_half_written_last_line_waits_for_its_newline(log: Path) -> None:
    line = _end(1, 1)
    log.write_text(_start(1, 1) + line[:10])
    tail = AttemptsTail(log)
    assert [a.verdict for a in tail.read()] == [""]
    log.write_text(_start(1, 1) + line)
    assert _parsed(tail) == (1, ["PASS"])


def test_a_multibyte_character_split_across_ticks_survives(log: Path) -> None:
    record = {"event": "start", "story": 1, "n": 1, "brief": "ação", "engine": "m", "tier": 0}
    raw = (json.dumps(record, ensure_ascii=False) + "\n").encode()
    cut = raw.index("ç".encode()) + 1
    log.write_bytes(raw[:cut])
    tail = AttemptsTail(log)
    assert tail.read() == []
    log.write_bytes(raw)
    assert [a.brief for a in tail.read()] == ["ação"]


def test_a_missing_file_has_no_attempts(log: Path) -> None:
    assert AttemptsTail(log).read() == []


def test_same_result_as_a_full_parse_at_every_prefix(log: Path) -> None:
    lines = [_start(1, 1), _end(1, 1, "RETRY"), _start(1, 2), _start(2, 1), _end(2, 1), "junk\n"]
    tail = AttemptsTail(log)
    text = ""
    for line in lines:
        text += line
        log.write_text(text)
        assert tail.read() == read_attempts(text)


def test_views_take_text_or_folded_attempts() -> None:
    stories = json.dumps(
        {"stories": [{"number": 1, "id": "US-1", "title": "t"}], "briefs": {"q": 1}}
    )
    text = _start(1, 1)
    attempts = read_attempts(text)
    assert agents_groups(stories, attempts) == agents_groups(stories, text)
    assert agents_rows(stories, attempts, "") == agents_rows(stories, text, "")


def test_builder_folds_each_appended_line_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = build_session(tmp_path, monkeypatch)
    (session.dir / "agents-run").mkdir(parents=True, exist_ok=True)
    (session.dir / STORIES).write_text(
        json.dumps({"stories": [{"number": 1, "id": "US-1", "title": "t"}], "briefs": {"q": 1}})
    )
    log = session.dir / ATTEMPTS
    log.write_text(_start(1, 1))
    now = datetime(2026, 11, 8, 1, tzinfo=timezone.utc)
    builder = RunViewBuilder()
    builder.build(session, "RUNNING", now, FIXTURE_LADDER)
    _append(log, _end(1, 1, "RETRY"))
    with patch.object(progress, "_fold", wraps=progress._fold) as fold:
        builder.build(session, "RUNNING", now, FIXTURE_LADDER)
    assert fold.call_count == 1
    assert builder._groups[0][2] == [(1, "T0", "RETRY")]


def test_builder_follows_the_session_it_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = RunViewBuilder()
    first = tmp_path / "one.jsonl"
    second = tmp_path / "two.jsonl"
    first.write_text(_start(1, 1))
    second.write_text(_start(2, 1))
    assert [a.story for a in builder._attempts_at(first)] == [1]
    assert [a.story for a in builder._attempts_at(second)] == [2]
