from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from splinter.memory.session import Session, new_session_id
from splinter.tui import PrdSessionApp


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


class TestOnReview:
    """Unit tests for _on_review text routing."""

    def test_on_review_accept_triggers_begin_run(self, session: Session, run_kwargs: dict) -> None:
        app = PrdSessionApp(session, run_kwargs)
        app.final_prd = "# Test PRD"
        app.strategy = "cascade"
        app.phase = "review"

        with patch.object(app, "_begin_run") as mock_begin:
            app._on_review("accept")
            mock_begin.assert_called_once()

    def test_on_review_run_triggers_begin_run(self, session: Session, run_kwargs: dict) -> None:
        app = PrdSessionApp(session, run_kwargs)
        app.final_prd = "# Test PRD"
        app.strategy = "cascade"
        app.phase = "review"

        with patch.object(app, "_begin_run") as mock_begin:
            app._on_review("run")
            mock_begin.assert_called_once()

    def test_on_review_yes_triggers_begin_run(self, session: Session, run_kwargs: dict) -> None:
        app = PrdSessionApp(session, run_kwargs)
        app.final_prd = "# Test PRD"
        app.strategy = "cascade"
        app.phase = "review"

        with patch.object(app, "_begin_run") as mock_begin:
            app._on_review("yes")
            mock_begin.assert_called_once()

    def test_on_review_go_triggers_begin_run(self, session: Session, run_kwargs: dict) -> None:
        app = PrdSessionApp(session, run_kwargs)
        app.final_prd = "# Test PRD"
        app.strategy = "cascade"
        app.phase = "review"

        with patch.object(app, "_begin_run") as mock_begin:
            app._on_review("go")
            mock_begin.assert_called_once()

    def test_on_review_y_triggers_begin_run(self, session: Session, run_kwargs: dict) -> None:
        app = PrdSessionApp(session, run_kwargs)
        app.final_prd = "# Test PRD"
        app.strategy = "cascade"
        app.phase = "review"

        with patch.object(app, "_begin_run") as mock_begin:
            app._on_review("y")
            mock_begin.assert_called_once()

    def test_on_review_other_text_triggers_revise(self, session: Session, run_kwargs: dict) -> None:
        app = PrdSessionApp(session, run_kwargs)
        app.final_prd = "# Test PRD"
        app.strategy = "cascade"
        app.phase = "review"
        app._busy = False

        with patch.object(app, "_set_busy") as mock_busy:
            with patch.object(app, "_spawn") as mock_spawn:
                app._on_review("add a new feature")
                mock_busy.assert_called_with(True, "applying your changes…")
                assert mock_spawn.called


class TestOnEdit:
    """Unit tests for _on_edit functionality."""

    def test_on_edit_preserves_final_prd(self, session: Session, run_kwargs: dict) -> None:
        app = PrdSessionApp(session, run_kwargs)
        original_prd = "# Original PRD\nSome content"
        app.final_prd = original_prd
        app.phase = "review"

        with patch.object(app, "_say"):
            with patch.object(app, "_set_busy"):
                with patch.object(app, "query_one"):
                    app._on_edit()

        assert app.final_prd == original_prd

    def test_on_edit_stays_in_review_phase(self, session: Session, run_kwargs: dict) -> None:
        app = PrdSessionApp(session, run_kwargs)
        app.final_prd = "# PRD"
        app.phase = "something_else"

        with patch.object(app, "_say"):
            with patch.object(app, "_set_busy"):
                with patch.object(app, "query_one"):
                    app._on_edit()

        assert app.phase == "review"

    def test_on_edit_calls_set_busy(self, session: Session, run_kwargs: dict) -> None:
        app = PrdSessionApp(session, run_kwargs)
        app.final_prd = "# PRD"

        with patch.object(app, "_say"):
            with patch.object(app, "_set_busy") as mock_busy:
                with patch.object(app, "query_one"):
                    app._on_edit()

                mock_busy.assert_called_once_with(False, "describe changes / accept / cowabunga")


class TestLiveDraftDispatch:
    """Ensure chat/review workers receive the current left-pane draft."""

    def test_on_chat_refine_uses_live_draft(self, session: Session, run_kwargs: dict) -> None:
        app = PrdSessionApp(session, run_kwargs)
        app.phase = "chat"
        app._busy = False

        with patch.object(app, "_read_draft", return_value="# live draft") as mock_read:
            with patch.object(app, "_set_busy") as mock_busy:
                with patch.object(app, "_spawn") as mock_spawn:
                    app._on_chat("1A")
                    mock_read.assert_called_once()
                    mock_busy.assert_called_once_with(True, "incorporating your answers…")
                    assert mock_spawn.call_args.kwargs == {
                        "answers": "1A",
                        "draft": "# live draft",
                    }

    def test_on_chat_finalize_uses_live_draft(self, session: Session, run_kwargs: dict) -> None:
        app = PrdSessionApp(session, run_kwargs)
        app.phase = "chat"
        app._busy = False

        with patch.object(app, "_read_draft", return_value="# live draft") as mock_read:
            with patch.object(app, "_set_busy") as mock_busy:
                with patch.object(app, "_spawn") as mock_spawn:
                    app._on_chat("fulfilled")
                    mock_read.assert_called_once()
                    mock_busy.assert_called_once_with(True, "finalizing the PRD…")
                    assert mock_spawn.call_args.kwargs == {
                        "autodecide": False,
                        "draft": "# live draft",
                    }

    def test_on_review_revise_uses_live_draft(self, session: Session, run_kwargs: dict) -> None:
        app = PrdSessionApp(session, run_kwargs)
        app.phase = "review"
        app._busy = False

        with patch.object(app, "_read_draft", return_value="# live draft") as mock_read:
            with patch.object(app, "_set_busy") as mock_busy:
                with patch.object(app, "_spawn") as mock_spawn:
                    app._on_review("add acceptance criteria")
                    mock_read.assert_called_once()
                    mock_busy.assert_called_once_with(True, "applying your changes…")
                    assert mock_spawn.call_args.kwargs == {
                        "instructions": "add acceptance criteria",
                        "draft": "# live draft",
                    }


class TestSourcePrdPersistence:
    """Persist updated PRD back to the original prd_path when provided."""

    def test_set_preview_writes_to_source_prd_path(
        self,
        session: Session,
        run_kwargs: dict,
    ) -> None:
        src = session.dir / "source-prd.md"
        src.write_text("# old")
        run_kwargs["prd_path"] = str(src)

        app = PrdSessionApp(session, run_kwargs)
        app._set_preview("# new")

        assert src.read_text().strip() == "# new"
        assert session.read("prd.md").strip() == "# new"

    def test_begin_run_writes_final_prd_to_source_prd_path(
        self,
        session: Session,
        run_kwargs: dict,
    ) -> None:
        src = session.dir / "source-prd.md"
        src.write_text("# old")
        run_kwargs["prd_path"] = str(src)

        app = PrdSessionApp(session, run_kwargs)
        app.strategy = "cascade"
        app._source_prd_path = str(src)

        with patch.object(app, "_read_draft", return_value="# final from draft"):
            with patch.object(app, "_say"):
                with patch.object(app, "_save_state"):
                    with patch.object(app, "exit"):
                        app._begin_run()

        assert src.read_text().strip() == "# final from draft"
        assert session.read("prd.md").strip() == "# final from draft"


class TestAcceptButton:
    """Unit tests for Accept/Edit button dispatch."""

    def test_accept_button_calls_submit(self, session: Session, run_kwargs: dict) -> None:
        app = PrdSessionApp(session, run_kwargs)
        app.phase = "review"
        app.final_prd = "# Test PRD"

        with patch.object(app, "_submit") as mock_submit:
            app.on_button_pressed(MagicMock(button=MagicMock(id="accept")))
            mock_submit.assert_called_once_with("accept")

    def test_edit_button_calls_on_edit(self, session: Session, run_kwargs: dict) -> None:
        app = PrdSessionApp(session, run_kwargs)
        app.phase = "review"
        app.final_prd = "# Test PRD"

        with patch.object(app, "_on_edit") as mock_edit:
            app.on_button_pressed(MagicMock(button=MagicMock(id="edit")))
            mock_edit.assert_called_once()

    def test_submit_accept_routes_to_on_review(self, session: Session, run_kwargs: dict) -> None:
        app = PrdSessionApp(session, run_kwargs)
        app.phase = "review"
        app.final_prd = "# Test PRD"
        app._busy = False

        with patch.object(app, "query_one"):
            with patch.object(app, "_on_review") as mock_review:
                app._submit("accept")
                mock_review.assert_called_once_with("accept")

    def test_revise_changes_routes_to_revise_worker(
        self, session: Session, run_kwargs: dict
    ) -> None:
        app = PrdSessionApp(session, run_kwargs)
        app.phase = "review"
        app.final_prd = "# Test PRD"
        app._busy = False

        with patch.object(app, "query_one"):
            with patch.object(app, "_set_busy"):
                with patch.object(app, "_spawn") as mock_spawn:
                    app._submit("add new feature")
                    mock_spawn.assert_called_once()


class TestTrustedPrd:
    """Tests for US-004: load non-empty PRD as-is, skip generation."""

    def test_accept_button_in_trust_phase_calls_accept_trusted(
        self, session: Session, run_kwargs: dict
    ) -> None:
        app = PrdSessionApp(session, run_kwargs)
        app.phase = "trust"
        app.trusted = True

        with patch.object(app, "_accept_trusted") as mock_accept:
            app.on_button_pressed(MagicMock(button=MagicMock(id="accept")))
            mock_accept.assert_called_once()

    def test_accept_button_in_review_phase_still_uses_submit(
        self, session: Session, run_kwargs: dict
    ) -> None:
        app = PrdSessionApp(session, run_kwargs)
        app.phase = "review"
        app.final_prd = "# Test PRD"

        with patch.object(app, "_submit") as mock_submit:
            app.on_button_pressed(MagicMock(button=MagicMock(id="accept")))
            mock_submit.assert_called_once_with("accept")

    def test_submit_routes_trust_to_on_trust(self, session: Session, run_kwargs: dict) -> None:
        app = PrdSessionApp(session, run_kwargs)
        app.phase = "trust"
        app.trusted = True
        app._busy = False

        with patch.object(app, "query_one"):
            with patch.object(app, "_on_trust") as mock_on_trust:
                app._submit("cowabunga")
                mock_on_trust.assert_called_once_with("cowabunga")

    def test_on_trust_cowabunga_triggers_begin_run(
        self, session: Session, run_kwargs: dict
    ) -> None:
        app = PrdSessionApp(session, run_kwargs)
        app.phase = "trust"
        app.trusted = True
        app._initial_prd = "# Initial PRD\n\n### US-001: Test"
        app._desc = "test"

        with patch.object(app, "_read_trusted_draft", return_value=app._initial_prd):
            with patch.object(app, "_begin_run") as mock_begin:
                app._on_trust("cowabunga")
                mock_begin.assert_called_once_with(autopick=True)

    def test_on_trust_non_cowabunga_is_noop(self, session: Session, run_kwargs: dict) -> None:
        app = PrdSessionApp(session, run_kwargs)
        app.phase = "trust"
        app.trusted = True

        with patch.object(app, "_begin_run") as mock_begin:
            app._on_trust("some random text")
            mock_begin.assert_not_called()

    def test_accept_trusted_sets_final_prd(self, session: Session, run_kwargs: dict) -> None:
        app = PrdSessionApp(session, run_kwargs)
        app.phase = "trust"
        app.trusted = True
        app._initial_prd = "# Test PRD"
        app._desc = "test"

        edited_prd = "# Test PRD\n\n### US-001: Test Story"
        with patch.object(app, "_read_trusted_draft", return_value=edited_prd):
            with patch.object(app, "_to_strategy_phase"):
                with patch.object(app, "_set_preview"):
                    with patch.object(app, "_show_stories"):
                        app._accept_trusted()
                        assert "# Test PRD" in app.final_prd
                        assert "### US-001" in app.final_prd

    def test_accept_trusted_proceeds_to_strategy_phase(
        self, session: Session, run_kwargs: dict
    ) -> None:
        app = PrdSessionApp(session, run_kwargs)
        app.phase = "trust"
        app.trusted = True
        app._initial_prd = "# Test PRD"
        app._desc = "test"

        with patch.object(app, "_read_trusted_draft", return_value=app._initial_prd):
            with patch.object(app, "_to_strategy_phase") as mock_to_strategy:
                with patch.object(app, "_set_preview"):
                    with patch.object(app, "_show_stories"):
                        app._accept_trusted()
                        mock_to_strategy.assert_called_once()

    def test_read_trusted_draft_returns_textarea_text(
        self, session: Session, run_kwargs: dict
    ) -> None:
        app = PrdSessionApp(session, run_kwargs)
        draft_text = "# Edited Draft\n\n### US-001: Story"

        mock_textarea = MagicMock()
        mock_textarea.text = draft_text
        with patch.object(app, "query_one", return_value=mock_textarea):
            result = app._read_trusted_draft()
            assert result == draft_text

    def test_read_trusted_draft_fallback_on_error(self, session: Session, run_kwargs: dict) -> None:
        app = PrdSessionApp(session, run_kwargs)
        app._initial_prd = "# Fallback PRD"

        with patch.object(app, "query_one", side_effect=Exception("widget not found")):
            result = app._read_trusted_draft()
            assert result == app._initial_prd

    def test_render_actions_trust_phase_has_accept_and_cowabunga(
        self, session: Session, run_kwargs: dict
    ) -> None:
        app = PrdSessionApp(session, run_kwargs)
        app.phase = "trust"

        buttons_created = []

        def capture_button(label, id, variant):
            buttons_created.append({"label": label, "id": id, "variant": variant})
            return MagicMock()

        with patch("splinter.tui.Button", side_effect=capture_button):
            with patch.object(app, "query_one"):
                with patch.object(app, "run_worker"):
                    app._render_actions("trust")

        button_ids = [b["id"] for b in buttons_created]
        assert "accept" in button_ids
        assert "cowabunga" in button_ids
