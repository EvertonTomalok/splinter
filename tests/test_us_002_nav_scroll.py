from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


def test_long_tree_shows_scrollbar(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    import pytest

    pytest.importorskip("textual")

    from textual.widgets import Tree

    from splinter.memory.session import Session
    from splinter.tui import AnalyzeApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_us002_long")
    session.update_index("# us002 long\n")
    session.set_status("completed", strategy="raphael", tasks=3, stage="done")

    # Populate session with prd, loop, and prd_phases to build a large tree
    session.write("prd.md", "# PRD\n\nLong PRD content for testing.\n")
    session.write("knowledge/localization.md", "# Localization\n\nContent\n")
    session.write("knowledge/plan.md", "# Plan\n\nContent\n")

    # Build a loop.md with 3 tasks, each with 10 iterations
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

    # Add prd_phases
    session.write(
        "prd_phases.md",
        "- Strategy\n- Localize\n- Plan\n- Run\n",
    )

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
            max_scroll = tree.max_scroll_y
            assert max_scroll > 0, f"scrollbar not present — max_scroll_y = {max_scroll}"
            await pilot.press("q")

    asyncio.run(drive())


def test_short_tree_no_scrollbar(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    import pytest

    pytest.importorskip("textual")

    from textual.widgets import Tree

    from splinter.memory.session import Session
    from splinter.tui import AnalyzeApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_us002_short")
    session.update_index("# us002 short\n")
    session.set_status("completed", strategy="raphael", tasks=1, stage="done")

    # Minimal session data
    session.write("knowledge/localization.md", "# Localization\n\nMinimal\n")
    session.write("knowledge/plan.md", "# Plan\n\nMinimal\n")
    session.write("loop.md", "# Loop\n\n## Task 1\n\n- #1 · level1 · passed\n")
    session.write("prd_phases.md", "- Localize\n- Plan\n")

    async def drive() -> None:
        app = AnalyzeApp(session)
        async with app.run_test() as pilot:
            await pilot.pause()
            tree = app.query_one("#nav", Tree)
            max_scroll = tree.max_scroll_y
            assert max_scroll == 0, (
                f"scrollbar present but should not be — max_scroll_y = {max_scroll}"
            )
            await pilot.press("q")

    asyncio.run(drive())


def test_nav_scroll_reaches_top_and_bottom(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import pytest

    pytest.importorskip("textual")

    from textual.widgets import Tree

    from splinter.memory.session import Session
    from splinter.tui import AnalyzeApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_us002_topbot")
    session.update_index("# us002 topbot\n")
    session.set_status("completed", strategy="raphael", tasks=3, stage="done")

    # Build long tree (3 tasks x 10 iters = ~44 tree nodes >> 28 lines viewport)
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
            max_scroll = tree.max_scroll_y

            assert tree.scroll_offset.y == 0, "should start at top"

            await pilot.press("end")
            await pilot.pause()
            assert tree.scroll_offset.y == max_scroll, (
                f"end key should reach bottom — {tree.scroll_offset.y} != {max_scroll}"
            )

            await pilot.press("home")
            await pilot.pause()
            assert tree.scroll_offset.y == 0, "home key should return to top"

            await pilot.press("q")

    asyncio.run(drive())


def test_nav_pagedown_pageup_scroll(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    import pytest

    pytest.importorskip("textual")

    from textual.widgets import Tree

    from splinter.memory.session import Session
    from splinter.tui import AnalyzeApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_us002_pageupdown")
    session.update_index("# us002 pageupdown\n")
    session.set_status("completed", strategy="raphael", tasks=3, stage="done")

    # Build long tree (3 tasks x 10 iters = ~44 tree nodes >> 28 lines viewport)
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

            start_offset = tree.scroll_offset.y
            assert start_offset == 0, "should start at top"

            # Press pagedown to scroll
            for _ in range(5):
                await pilot.press("pagedown")
                await pilot.pause(0.05)

            after_pagedown = tree.scroll_offset.y
            assert after_pagedown > start_offset, (
                f"pagedown should scroll — {after_pagedown} <= {start_offset}"
            )

            # Press pageup to scroll back
            for _ in range(3):
                await pilot.press("pageup")
                await pilot.pause(0.05)

            after_pageup = tree.scroll_offset.y
            assert after_pageup < after_pagedown, (
                f"pageup should scroll back — {after_pageup} >= {after_pagedown}"
            )

            await pilot.press("q")

    asyncio.run(drive())


def test_analyze_keybindings_not_shadowed(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import pytest

    pytest.importorskip("textual")

    from splinter.memory.session import Session
    from splinter.tui import AnalyzeApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_us002_keybind")
    session.update_index("# us002 keybind\n")
    session.set_status("completed", strategy="raphael", tasks=1, stage="done")

    # Minimal session
    session.write("knowledge/localization.md", "# Localization\n\nContent\n")
    session.write("knowledge/plan.md", "# Plan\n\nContent\n")
    session.write("loop.md", "# Loop\n\n## Task 1\n\n- #1 · level1 · passed\n")
    session.write("prd_phases.md", "- Localize\n- Plan\n")

    actions_called: list[str] = []

    def patched_reload(self: AnalyzeApp) -> None:
        actions_called.append("reload")

    def patched_toggle_auto(self: AnalyzeApp) -> None:
        actions_called.append("toggle_auto")

    monkeypatch.setattr(AnalyzeApp, "action_reload", patched_reload)
    monkeypatch.setattr(AnalyzeApp, "action_toggle_auto", patched_toggle_auto)

    async def drive() -> None:
        app = AnalyzeApp(session)
        async with app.run_test() as pilot:
            await pilot.pause()

            await pilot.press("r")
            await pilot.pause()

            await pilot.press("R")
            await pilot.pause()

            await pilot.press("q")

    asyncio.run(drive())

    assert "reload" in actions_called, f"r key did not trigger reload — got: {actions_called}"
    assert "toggle_auto" in actions_called, (
        f"R key did not trigger toggle_auto — got: {actions_called}"
    )


def test_detail_pane_unchanged(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    import pytest

    pytest.importorskip("textual")

    from textual.containers import VerticalScroll
    from textual.widgets import Markdown

    from splinter.memory.session import Session
    from splinter.tui import AnalyzeApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_us002_detail")
    session.update_index("# us002 detail\n")
    session.set_status("completed", strategy="raphael", tasks=1, stage="done")

    # Minimal session
    session.write("knowledge/localization.md", "# Localization\n\nContent\n")
    session.write("knowledge/plan.md", "# Plan\n\nContent\n")
    session.write("loop.md", "# Loop\n\n## Task 1\n\n- #1 · level1 · passed\n")
    session.write("prd_phases.md", "- Localize\n- Plan\n")

    async def drive() -> None:
        app = AnalyzeApp(session)
        async with app.run_test() as pilot:
            await pilot.pause()

            # Find the detail markdown parent container (VerticalScroll)
            detail = app.query_one("#detail", Markdown)
            parent = detail.parent
            while parent is not None:
                if isinstance(parent, VerticalScroll):
                    assert isinstance(parent, VerticalScroll), (
                        "detail pane should be inside VerticalScroll"
                    )
                    break
                parent = parent.parent
            else:
                assert False, "detail pane not wrapped in VerticalScroll"

            await pilot.press("q")

    asyncio.run(drive())
