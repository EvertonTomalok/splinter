from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from splinter.providers import dispatch

PROVIDER_CASES = [
    (
        "claude",
        "opus",
        "sess-1",
        "hello",
        [
            "claude",
            "-p",
            "--model",
            "opus",
            "--dangerously-skip-permissions",
            "--resume",
            "sess-1",
            "--",
            "hello",
        ],
    ),
    (
        "opencode",
        "opencode-go/big",
        "sess-2",
        "hi there",
        [
            "opencode",
            "run",
            "-m",
            "opencode-go/big",
            "--dangerously-skip-permissions",
            "-s",
            "sess-2",
            "--",
            "hi there",
        ],
    ),
    (
        "codex",
        "codex/gpt-5.4",
        "sess-3",
        "keep going",
        [
            "codex",
            "exec",
            "resume",
            "--json",
            "--skip-git-repo-check",
            "-m",
            "gpt-5.4",
            "--dangerously-bypass-approvals-and-sandbox",
            "--",
            "sess-3",
            "keep going",
        ],
    ),
    (
        "cursor",
        "cursor/gpt-5",
        "sess-4",
        "continue",
        [
            "agent",
            "-p",
            "--trust",
            "--resume",
            "sess-4",
            "--model",
            "gpt-5",
            "--",
            "continue",
        ],
    ),
    (
        "cursor",
        "cursor/auto",
        "sess-5",
        "continue",
        [
            "agent",
            "-p",
            "--trust",
            "--resume",
            "sess-5",
            "--",
            "continue",
        ],
    ),
]


@pytest.mark.parametrize(("provider", "model", "session_id", "message", "expected"), PROVIDER_CASES)
def test_argv_per_provider(
    provider: str, model: str, session_id: str, message: str, expected: list[str]
) -> None:
    with patch("splinter.providers.dispatch.subprocess.Popen") as mock_popen:
        mock_popen.return_value = MagicMock()
        dispatch.fire_message_into_session(
            provider=provider, model=model, session_id=session_id, message=message
        )
    assert mock_popen.call_args.args[0] == expected


def test_detached_flags() -> None:
    with patch("splinter.providers.dispatch.subprocess.Popen") as mock_popen:
        mock_popen.return_value = MagicMock()
        dispatch.fire_message_into_session(
            provider="claude", model="opus", session_id="sess-1", message="hi"
        )
    assert mock_popen.call_args.kwargs["start_new_session"] is True
    assert mock_popen.call_args.kwargs["stdout"] == subprocess.DEVNULL
    assert mock_popen.call_args.kwargs["stderr"] == subprocess.DEVNULL


def test_no_blocking_wait_on_caller() -> None:
    mock_proc = MagicMock()
    with patch("splinter.providers.dispatch.subprocess.Popen", return_value=mock_proc):
        result = dispatch.fire_message_into_session(
            provider="claude", model="opus", session_id="sess-1", message="hi"
        )
    assert result is None
    mock_proc.communicate.assert_not_called()


def test_not_registered_in_procreg_active() -> None:
    import splinter.procreg as procreg

    before = len(procreg._active)
    with patch("splinter.providers.dispatch.subprocess.Popen") as mock_popen:
        mock_popen.return_value = MagicMock()
        dispatch.fire_message_into_session(
            provider="claude", model="opus", session_id="sess-1", message="hi"
        )
    assert len(procreg._active) == before


def test_spawn_failure_swallowed(caplog: pytest.LogCaptureFixture) -> None:
    with patch("splinter.providers.dispatch.subprocess.Popen", side_effect=OSError("boom")):
        with caplog.at_level("WARNING", logger="splinter.providers.dispatch"):
            result = dispatch.fire_message_into_session(
                provider="claude", model="opus", session_id="sess-1", message="hi"
            )
    assert result is None
    assert any("failed to fire message into session" in rec.message for rec in caplog.records)


def test_unknown_provider_swallowed(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="splinter.providers.dispatch"):
        result = dispatch.fire_message_into_session(
            provider="nope", model="m", session_id="sess-1", message="hi"
        )
    assert result is None
    assert any("failed to fire message into session" in rec.message for rec in caplog.records)


def test_reap_best_effort_on_wait_failure() -> None:
    mock_proc = MagicMock()
    mock_proc.wait.side_effect = OSError("wait failed")
    with patch("splinter.providers.dispatch.subprocess.Popen", return_value=mock_proc):
        result = dispatch.fire_message_into_session(
            provider="claude", model="opus", session_id="sess-1", message="hi"
        )
    assert result is None
    dispatch._reap(mock_proc)
