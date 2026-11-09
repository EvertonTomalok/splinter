from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

from splinter.procreg import bind_log_story, current_log_story, log_story_scope
from splinter.run_view import story_running
from splinter.tui import _TextualLogHandler
from splinter.tui_run_focus import FocusColumn
from splinter.tui_run_logs import StoryLog
from splinter.tui_run_views import _queued_note


def _story(number: int, state: str) -> SimpleNamespace:
    return SimpleNamespace(number=number, state=state)


def _view(*stories: SimpleNamespace, iterations: dict[int, tuple[Any, ...]] | None = None) -> Any:
    live = next((story.number for story in stories if story.state == "live"), None)
    return SimpleNamespace(
        stories=stories, live_story=live, stage="merge", iterations_by_story=iterations or {}
    )


def test_each_story_shows_only_its_own_lines_and_the_pipeline() -> None:
    logs = StoryLog()
    logs.add(None, "pipeline")
    logs.add(2, "from 2")
    logs.add(20, "from 20")

    assert logs.visible(2) == ["pipeline", "from 2"]
    assert logs.visible(20) == ["pipeline", "from 20"]


def test_no_selected_story_shows_every_line() -> None:
    logs = StoryLog()
    logs.add(2, "from 2")
    logs.add(20, "from 20")

    assert logs.visible(None) == ["from 2", "from 20"]


def test_story_log_drops_the_oldest_lines_past_its_limit() -> None:
    logs = StoryLog(limit=3, trim=2)
    for index in range(4):
        logs.add(None, str(index))

    assert logs.visible(None) == ["2", "3"]


def test_log_story_scope_marks_lines_and_ends() -> None:
    assert current_log_story() is None
    with log_story_scope(7):
        assert current_log_story() == 7
    assert current_log_story() is None


def test_bind_log_story_holds_in_its_own_context() -> None:
    import contextvars

    def worker() -> int | None:
        bind_log_story(9)
        return current_log_story()

    assert contextvars.copy_context().run(worker) == 9
    assert current_log_story() is None


def test_handler_forwards_the_story_of_the_emitting_thread() -> None:
    calls: list[tuple[Any, ...]] = []
    app = SimpleNamespace(
        write_log=object(), call_from_thread=lambda fn, *args: calls.append(args)
    )
    handler = _TextualLogHandler(app)
    record = logging.LogRecord("splinter", logging.INFO, "f", 1, "edit a.py", None, None)

    with log_story_scope(20):
        handler.emit(record)

    assert calls == [("edit a.py", logging.INFO, 20)]


def test_every_live_story_counts_as_running() -> None:
    view = _view(_story(2, "live"), _story(20, "live"), _story(3, "queued"))

    assert story_running(view, 2) and story_running(view, 20)
    assert not story_running(view, 3)
    assert not story_running(view, None)


def _note(view: Any, story: int) -> str:
    return _queued_note(SimpleNamespace(view=view, story=story)).plain  # type: ignore[arg-type]


def test_a_second_running_story_is_not_called_not_started() -> None:
    view = _view(_story(2, "live"), _story(20, "live"))

    assert "Not started" not in _note(view, 20)


def test_a_queued_story_still_waits_for_its_turn() -> None:
    view = _view(_story(2, "live"), _story(3, "queued"))

    assert "Not started" in _note(view, 3)


def test_a_running_story_never_sits_on_merge() -> None:
    view = _view(_story(2, "live"), _story(20, "live"))

    assert FocusColumn._stage_index(view, view.stories[1], True) == 1


def test_a_parallel_story_with_a_finished_run_is_on_eval() -> None:
    open_run = (SimpleNamespace(verdict=""),)
    view = _view(_story(2, "live"), _story(20, "live"), iterations={20: open_run})

    assert FocusColumn._stage_index(view, view.stories[1], True) == 3


def test_a_parallel_retry_goes_back_to_run() -> None:
    judged = (SimpleNamespace(verdict="RETRY"),)
    view = _view(_story(2, "live"), _story(20, "live"), iterations={20: judged})

    assert FocusColumn._stage_index(view, view.stories[1], True) == 1


def test_a_single_live_story_keeps_the_stage_of_the_run() -> None:
    view = _view(_story(2, "live"))
    view.stage = "gate"

    assert FocusColumn._stage_index(view, view.stories[0], True) == 2
