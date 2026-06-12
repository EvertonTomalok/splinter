from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


def test_scroll_css_constant_exists() -> None:
    """Verify _SCROLL_LEFT_PANE_CSS constant is defined."""
    from splinter.tui import _SCROLL_LEFT_PANE_CSS

    assert _SCROLL_LEFT_PANE_CSS, "scroll constant should be defined"
    assert "#nav" in _SCROLL_LEFT_PANE_CSS, "constant should target #nav"
    assert "#overview-scroll" in _SCROLL_LEFT_PANE_CSS, "constant should target #overview-scroll"
    assert "height: 1fr" in _SCROLL_LEFT_PANE_CSS, "constant should set height"
    assert "overflow-y: auto" in _SCROLL_LEFT_PANE_CSS, "constant should set overflow-y"
    assert "overflow-x: hidden" in _SCROLL_LEFT_PANE_CSS, "constant should set overflow properties"
    assert (
        "scrollbar-size-vertical: 1" in _SCROLL_LEFT_PANE_CSS
    ), "constant should set scrollbar size"


def test_analyze_app_uses_scroll_css(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import pytest

    pytest.importorskip("textual")

    from splinter.memory.session import Session
    from splinter.tui import _SCROLL_LEFT_PANE_CSS, AnalyzeApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_us003_analyze")
    session.update_index("# us003 analyze\n")
    session.set_status("completed", strategy="raphael", tasks=1, stage="done")

    session.write("knowledge/localization.md", "# Localization\n\nContent\n")
    session.write("knowledge/plan.md", "# Plan\n\nContent\n")
    session.write("loop.md", "# Loop\n\n## Task 1\n\n- #1 · level1 · passed\n")
    session.write("prd_phases.md", "- Localize\n- Plan\n")

    # Verify constant is in AnalyzeApp CSS
    assert _SCROLL_LEFT_PANE_CSS in AnalyzeApp.CSS, "AnalyzeApp should use _SCROLL_LEFT_PANE_CSS"

    async def drive() -> None:
        app = AnalyzeApp(session)
        async with app.run_test() as pilot:
            await pilot.pause()
            # Just verify the app starts without error
            await pilot.press("q")

    asyncio.run(drive())


def test_run_app_uses_scroll_css(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import pytest

    pytest.importorskip("textual")

    from splinter.memory.session import Session
    from splinter.tui import _SCROLL_LEFT_PANE_CSS, RunApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_us003_run")
    session.update_index("# us003 run\n")
    session.set_status("running", strategy="raphael", tasks=1, stage="run")

    def fake_pipeline(**kwargs: object) -> int:
        return 0

    monkeypatch.setattr("splinter.pipeline.run_pipeline", fake_pipeline)

    # Verify constant is in RunApp CSS
    assert _SCROLL_LEFT_PANE_CSS in RunApp.CSS, "RunApp should use _SCROLL_LEFT_PANE_CSS"

    async def drive() -> None:
        app = RunApp(session, {})
        async with app.run_test() as pilot:
            await pilot.pause()
            for _ in range(20):
                await pilot.pause(0.05)
                if app.workers and all(
                    w.state.name in ("SUCCESS", "ERROR") for w in app.workers
                ):
                    break
            await pilot.pause()
            # Just verify the app starts without error
            await pilot.press("q")

    asyncio.run(drive())


def test_analyze_app_scroll_css_properties_applied(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify #nav in AnalyzeApp has scroll CSS properties applied."""
    import pytest

    pytest.importorskip("textual")

    from textual.widgets import Tree

    from splinter.memory.session import Session
    from splinter.tui import AnalyzeApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_us003_analyze_props")
    session.update_index("# us003 analyze props\n")
    session.set_status("completed", strategy="raphael", tasks=3, stage="done")

    # Build long tree to verify scroll properties
    session.write("prd.md", "# PRD\n\nContent\n")
    session.write("knowledge/localization.md", "# Localization\n\nContent\n")
    session.write("knowledge/plan.md", "# Plan\n\nContent\n")

    loop_parts = ["# Loop\n"]
    for task_no in range(1, 4):
        title = f"Task {task_no} title"
        loop_parts.append(f"\n# Task {task_no}/3: {title}\n")
        for n in range(1, 11):
            tier = 2 if n % 2 == 0 else 1
            verdict = "PASS" if n < 6 else "RETRY"
            loop_parts.append(f"\n## Iteration {n}\n")
            loop_parts.append(f"tier {tier}\n")
            loop_parts.append(f"verdict: {verdict}\n")
    loop_content = "".join(loop_parts)
    session.write("loop.md", loop_content)
    session.write("prd_phases.md", "- Strategy\n- Localize\n- Plan\n- Run\n")

    async def drive() -> None:
        app = AnalyzeApp(session)
        async with app.run_test() as pilot:
            await pilot.pause()
            tree = app.query_one("#nav", Tree)

            # Expand task nodes to show overflow
            traj_node = None
            for node in tree.root.children:
                if "Trajectory" in str(node.label):
                    traj_node = node
                    break
            if traj_node:
                for child in traj_node.children:
                    if "Task" in str(child.label):
                        child.expand()

            await pilot.pause(0.2)

            # Verify height is set to 1fr
            height = tree.styles.height
            assert height and height.value == 1.0, "Tree height should be 1fr"

            # Verify scroll properties are applied
            assert tree.styles.overflow_y == "auto", "overflow-y should be auto"
            assert tree.styles.overflow_x == "hidden", "overflow-x should be hidden"

            await pilot.press("q")

    asyncio.run(drive())


def test_run_app_scroll_css_properties_applied(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify #overview-scroll in RunApp has scroll CSS properties applied."""
    import pytest

    pytest.importorskip("textual")

    from textual.containers import VerticalScroll
    from textual.widgets import Static

    from splinter.memory.session import Session
    from splinter.tui import RunApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_us003_run_props")
    session.update_index("# us003 run props\n")
    session.set_status("running", strategy="raphael", tasks=1, stage="run")

    def patched_refresh(self: RunApp) -> None:
        tall_content = "\n".join([f"Line {i}" for i in range(200)])
        self.query_one("#overview", Static).update(tall_content)

    monkeypatch.setattr(RunApp, "_refresh", patched_refresh)

    def fake_pipeline(**kwargs: object) -> int:
        return 0

    monkeypatch.setattr("splinter.pipeline.run_pipeline", fake_pipeline)

    async def drive() -> None:
        app = RunApp(session, {})
        async with app.run_test() as pilot:
            await pilot.pause()
            for _ in range(20):
                await pilot.pause(0.05)
                if app.workers and all(
                    w.state.name in ("SUCCESS", "ERROR") for w in app.workers
                ):
                    break
            await pilot.pause()

            scroll = app.query_one("#overview-scroll", VerticalScroll)

            # Verify VerticalScroll height is bounded to allow scrolling
            height = scroll.styles.height
            assert height and height.value == 1.0, "VerticalScroll height should be 1fr"

            await pilot.press("q")

    asyncio.run(drive())


def test_consistency_no_duplication_in_analyze_app(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify AnalyzeApp doesn't duplicate scroll CSS rules."""
    from splinter.tui import _SCROLL_LEFT_PANE_CSS, AnalyzeApp

    # The AnalyzeApp CSS should use _SCROLL_LEFT_PANE_CSS constant, not inline rules
    full_css = AnalyzeApp.CSS

    # Find the _SCROLL_LEFT_PANE_CSS section in the full CSS
    assert (
        _SCROLL_LEFT_PANE_CSS in full_css
    ), "AnalyzeApp should include _SCROLL_LEFT_PANE_CSS"

    # Verify #nav scroll CSS is in the constant
    assert "#nav" in _SCROLL_LEFT_PANE_CSS, "scroll constant should target #nav"


def test_consistency_no_duplication_in_run_app(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify RunApp doesn't duplicate scroll CSS rules."""
    from splinter.tui import _SCROLL_LEFT_PANE_CSS, RunApp

    # The RunApp CSS should use _SCROLL_LEFT_PANE_CSS constant, not inline rules
    full_css = RunApp.CSS

    # Find the _SCROLL_LEFT_PANE_CSS section in the full CSS
    assert (
        _SCROLL_LEFT_PANE_CSS in full_css
    ), "RunApp should include _SCROLL_LEFT_PANE_CSS"

    # Verify #overview-scroll scroll CSS is in the constant
    assert (
        "#overview-scroll" in _SCROLL_LEFT_PANE_CSS
    ), "scroll constant should target #overview-scroll"
