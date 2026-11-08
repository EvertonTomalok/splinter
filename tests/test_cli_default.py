from __future__ import annotations

import sys
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from splinter.cli import app, main


@pytest.mark.parametrize("argv", [[], ["prd"]])
@pytest.mark.parametrize("exit_code", [0, 7])
def test_default_command_matches_prd(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], exit_code: int
) -> None:
    run_prd = Mock(return_value=exit_code)
    monkeypatch.setenv("SPLINTER_NO_TUI", "1")
    monkeypatch.setattr("splinter.prd.run_prd", run_prd)

    assert main(argv) == exit_code
    run_prd.assert_called_once_with(description="", strategy=None, no_ground=False)


@pytest.mark.parametrize("argv", [["--help"], ["prd", "--help"], ["setup"]])
def test_explicit_command_does_not_start_default_prd(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    run_prd = Mock()
    monkeypatch.setattr("splinter.prd.run_prd", run_prd)
    monkeypatch.setattr("splinter.tui.run_prd_interactive", run_prd)
    monkeypatch.setattr("splinter.setup.run_setup", lambda: 0)

    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 0
    run_prd.assert_not_called()


def test_console_without_arguments_opens_prd_tui(monkeypatch: pytest.MonkeyPatch) -> None:
    run_prd_interactive = Mock(return_value=0)
    monkeypatch.setattr(sys, "argv", ["splinter"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.delenv("SPLINTER_NO_TUI", raising=False)
    monkeypatch.setattr("splinter.tui.run_prd_interactive", run_prd_interactive)

    assert main() == 0
    run_prd_interactive.assert_called_once()
    assert run_prd_interactive.call_args.args[0]["description"] == ""
