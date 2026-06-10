"""US-005: Both PRD-accept flows converge on run_with_tui with unified termination."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from splinter.memory.session import Session, new_session_id
from splinter.tui import _prd_run_kwargs, run_prd_interactive


@pytest.fixture
def session() -> Session:
    return Session(new_session_id())


@pytest.fixture
def run_kwargs() -> dict:
    return {
        "strategy": None,
        "prd_path": None,
        "task_path": None,
        "effort": None,
        "budget": None,
        "max_iterations": 5,
        "cowabunga": False,
    }


class TestPrdRunKwargs:
    """Test _prd_run_kwargs helper builds canonical kwargs."""

    def test_prd_run_kwargs_shape(self, session: Session, run_kwargs: dict) -> None:
        session.write("prd.md", "---\nstrategy: cascade\n---\n# Test")
        prd_path = str(session.dir / "prd.md")
        final = _prd_run_kwargs(prd_path, session, run_kwargs)
        assert final["strategy"] == "cascade"
        assert final["prd_path"] == prd_path
        assert final["task_path"] is None

    def test_prd_run_kwargs_preserves_other_keys(self, session: Session, run_kwargs: dict) -> None:
        session.write("prd.md", "---\nstrategy: direct\n---\n# Test")
        run_kwargs["effort"] = "high"
        run_kwargs["budget"] = 10.0
        prd_path = str(session.dir / "prd.md")
        final = _prd_run_kwargs(prd_path, session, run_kwargs)
        assert final["effort"] == "high"
        assert final["budget"] == 10.0
        assert final["strategy"] == "direct"

    def test_prd_run_kwargs_defaults_strategy(self, session: Session, run_kwargs: dict) -> None:
        session.write("prd.md", "# Test")
        prd_path = str(session.dir / "prd.md")
        final = _prd_run_kwargs(prd_path, session, run_kwargs)
        assert final["strategy"] == "cascade"


class TestFlowConvergence:
    """Test both flows converge on run_with_tui."""

    def test_prd_interactive_calls_run_with_tui_on_accept(
        self, session: Session, run_kwargs: dict
    ) -> None:
        with patch("splinter.tui.PrdSessionApp") as MockApp:
            app_instance = MagicMock()
            app_instance.run.return_value = 0
            MockApp.return_value = app_instance
            with patch("splinter.tui.Session", return_value=session):
                with patch("splinter.tui.run_with_tui") as mock_run:
                    mock_run.return_value = 42
                    session.set_status("refining", phase="run")
                    session.write("prd.md", "---\nstrategy: cascade\n---\n# Test")
                    rc = run_prd_interactive(run_kwargs)
                    assert rc == 42
                    assert mock_run.called

    def test_prd_interactive_returns_0_on_abort(self, session: Session, run_kwargs: dict) -> None:
        with patch("splinter.tui.PrdSessionApp") as MockApp:
            app_instance = MagicMock()
            app_instance.run.return_value = None
            MockApp.return_value = app_instance
            with patch("splinter.tui.Session", return_value=session):
                rc = run_prd_interactive(run_kwargs)
                assert rc == 0

    def test_prd_interactive_passes_session_to_run_with_tui(
        self, session: Session, run_kwargs: dict
    ) -> None:
        with patch("splinter.tui.PrdSessionApp") as MockApp:
            app_instance = MagicMock()
            app_instance.run.return_value = 0
            MockApp.return_value = app_instance
            with patch("splinter.tui.Session", return_value=session):
                with patch("splinter.tui.run_with_tui") as mock_run:
                    mock_run.return_value = 0
                    session.set_status("refining", phase="run")
                    session.write("prd.md", "---\nstrategy: cascade\n---\n# Test")
                    run_prd_interactive(run_kwargs)
                    call_args = mock_run.call_args
                    assert call_args[1]["session"] == session

    def test_prd_interactive_passes_final_prd_path_to_run_with_tui(
        self, session: Session, run_kwargs: dict
    ) -> None:
        with patch("splinter.tui.PrdSessionApp") as MockApp:
            app_instance = MagicMock()
            app_instance.run.return_value = 0
            MockApp.return_value = app_instance
            with patch("splinter.tui.Session", return_value=session):
                with patch("splinter.tui.run_with_tui") as mock_run:
                    mock_run.return_value = 0
                    session.set_status("refining", phase="run")
                    session.write("prd.md", "---\nstrategy: cascade\n---\n# Test")
                    run_prd_interactive(run_kwargs)
                    call_args = mock_run.call_args
                    assert "prd_path" in call_args[0][0]
                    assert str(session.dir / "prd.md") == call_args[0][0]["prd_path"]
