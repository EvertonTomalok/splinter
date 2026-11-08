"""Tests for engine run failure classification (splinter.engines.failures)."""

from __future__ import annotations

import dataclasses

import pytest

from splinter.engines.failures import (
    LIMIT_TAIL_BYTES,
    EngineFailure,
    FailureKind,
    classify,
    find_limit,
    find_result,
)


def _pad(n: int) -> str:
    return "x" * n


class TestFailureKind:
    def test_str_values(self) -> None:
        assert FailureKind.STALL == "stall"
        assert FailureKind.LIMIT == "limit"
        assert FailureKind.CRASH == "crash"
        assert FailureKind.EMPTY == "empty"
        assert FailureKind.MISSING == "missing"

    def test_engine_failure_defaults(self) -> None:
        failure = EngineFailure(FailureKind.CRASH)
        assert failure.detail == ""
        assert failure.exit_code is None

    def test_engine_failure_is_frozen(self) -> None:
        failure = EngineFailure(FailureKind.CRASH, detail="boom", exit_code=3)
        with pytest.raises(dataclasses.FrozenInstanceError):
            failure.detail = "changed"


class TestFindLimit:
    def test_match_inside_tail_window_counts(self) -> None:
        stdout = _pad(10_000) + "\nrate limit hit\n"
        assert find_limit(stdout, "") == "rate limit hit"

    def test_match_7000_bytes_back_is_ignored(self) -> None:
        stdout = "rate limit hit\n" + _pad(7000)
        assert find_limit(stdout, "") is None

    def test_match_just_inside_window_counts(self) -> None:
        stdout = _pad(LIMIT_TAIL_BYTES - len("\nquota exceeded")) + "\nquota exceeded"
        assert find_limit(stdout, "") == "quota exceeded"

    def test_match_just_outside_window_is_ignored(self) -> None:
        stdout = "quota exceeded\n" + _pad(LIMIT_TAIL_BYTES)
        assert find_limit(stdout, "") is None

    def test_stderr_fallback(self) -> None:
        result = find_limit("all good here", "error: usage limit reached")
        assert result == "error: usage limit reached"

    def test_stderr_tail_window_also_applies(self) -> None:
        stderr = "quota exceeded\n" + _pad(7000)
        assert find_limit("", stderr) is None

    def test_stdout_preferred_over_stderr(self) -> None:
        result = find_limit("stdout: quota exceeded", "stderr: quota exceeded")
        assert result == "stdout: quota exceeded"

    def test_no_match_returns_none(self) -> None:
        assert find_limit("plain output", "plain errors") is None

    def test_empty_streams_return_none(self) -> None:
        assert find_limit("", "") is None

    def test_returns_matching_line_not_bare_pattern(self) -> None:
        stdout = "wrap: the engine said quota exceeded badly"
        assert find_limit(stdout, "") == stdout

    def test_result_is_a_single_stripped_line(self) -> None:
        stdout = "before\nError: 429 too many requests, retry later\nafter"
        assert find_limit(stdout, "") == "Error: 429 too many requests, retry later"

    def test_long_line_truncated_to_about_120_chars(self) -> None:
        stdout = "A" * 200 + " quota exceeded " + "B" * 200
        result = find_limit(stdout, "")
        assert result is not None
        assert len(result) <= 120
        assert "\n" not in result

    def test_case_insensitive(self) -> None:
        assert find_limit("Usage LIMIT Reached", "") is not None


class TestFindLimitPhrases:
    @pytest.mark.parametrize(
        "phrase",
        [
            "usage limit",
            "rate limit",
            "rate-limit",
            "limit reached",
            "credit balance",
            "quota exceeded",
            "insufficient credits",
            "too many requests",
            '"status":429',
            "returned 429 after",
        ],
    )
    def test_phrase_in_stdout_tail_matches(self, phrase: str) -> None:
        assert find_limit(f"engine said: {phrase}, giving up", "") is not None

    def test_benign_output_does_not_match(self) -> None:
        assert find_limit("AGENT_RESULT: PASS q1\nall good", "") is None


class TestFindResult:
    def test_single_match(self) -> None:
        assert find_result("noise\nAGENT_RESULT: PASS q1\ndone") == ("PASS", "q1")

    def test_last_match_wins(self) -> None:
        text = "AGENT_RESULT: RETRY q1\nAGENT_RESULT: PASS q1"
        assert find_result(text) == ("PASS", "q1")

    def test_no_match_returns_none(self) -> None:
        assert find_result("no contract line here") is None

    def test_empty_text_returns_none(self) -> None:
        assert find_result("") is None

    def test_id_with_digits_and_hyphens(self) -> None:
        assert find_result("AGENT_RESULT: FAIL pb3a-9") == ("FAIL", "pb3a-9")

    def test_lowercase_status_not_matched(self) -> None:
        assert find_result("AGENT_RESULT: pass q1") is None

    def test_trailing_words_after_id_ignored(self) -> None:
        assert find_result("AGENT_RESULT: PASS q1 (accepted)") == ("PASS", "q1")


class TestClassifyHealthy:
    def test_exit_zero_with_result_returns_none(self) -> None:
        failure = classify(
            returncode=0, stdout="AGENT_RESULT: PASS q1", stderr="", has_result=True
        )
        assert failure is None

    def test_exit_none_with_result_returns_none(self) -> None:
        failure = classify(returncode=None, stdout="", stderr="", has_result=True)
        assert failure is None


class TestClassifyPrecedence:
    def test_missing_beats_stall_limit_crash_empty(self) -> None:
        failure = classify(
            returncode=127,
            stdout="usage limit reached",
            stderr="",
            has_result=False,
            stalled=True,
            missing=True,
        )
        assert failure == EngineFailure(
            FailureKind.MISSING,
            detail="engine binary not found (exit 127)",
            exit_code=127,
        )

    def test_stall_beats_limit_crash_empty(self) -> None:
        failure = classify(
            returncode=1, stdout="quota exceeded", stderr="", has_result=False, stalled=True
        )
        assert failure is not None
        assert failure.kind is FailureKind.STALL

    def test_limit_beats_crash_empty(self) -> None:
        failure = classify(
            returncode=1, stdout="error: rate limit reached", stderr="", has_result=False
        )
        assert failure is not None
        assert failure.kind is FailureKind.LIMIT

    def test_limit_from_stderr_beats_crash(self) -> None:
        failure = classify(
            returncode=1, stdout="", stderr="fatal: insufficient credits", has_result=True
        )
        assert failure is not None
        assert failure.kind is FailureKind.LIMIT

    def test_crash_beats_empty(self) -> None:
        failure = classify(returncode=2, stdout="", stderr="traceback", has_result=False)
        assert failure is not None
        assert failure.kind is FailureKind.CRASH


class TestClassifyKinds:
    def test_exit_127_auto_detected_as_missing(self) -> None:
        failure = classify(returncode=127, stdout="command not found", stderr="", has_result=True)
        assert failure == EngineFailure(
            FailureKind.MISSING,
            detail="engine binary not found (exit 127)",
            exit_code=127,
        )

    def test_missing_flag_without_exit_code(self) -> None:
        failure = classify(returncode=None, stdout="", stderr="", has_result=False, missing=True)
        assert failure == EngineFailure(
            FailureKind.MISSING,
            detail="engine binary not found",
            exit_code=None,
        )

    def test_stall_detail_mentions_watchdog(self) -> None:
        failure = classify(
            returncode=None, stdout="", stderr="", has_result=False, stalled=True
        )
        assert failure is not None
        assert failure.kind is FailureKind.STALL
        assert "idle watchdog" in failure.detail
        assert failure.exit_code is None

    def test_stall_detail_embeds_exit_code(self) -> None:
        failure = classify(
            returncode=9, stdout="", stderr="", has_result=True, stalled=True
        )
        assert failure is not None
        assert failure.detail == "idle watchdog tripped (exit 9)"
        assert failure.exit_code == 9

    def test_limit_detail_is_matched_line(self) -> None:
        failure = classify(
            returncode=0,
            stdout="x\nError: 429 too many requests",
            stderr="",
            has_result=True,
        )
        assert failure is not None
        assert failure.kind is FailureKind.LIMIT
        assert failure.detail == "Error: 429 too many requests"
        assert failure.exit_code == 0

    def test_limit_even_on_healthy_exit(self) -> None:
        failure = classify(
            returncode=0,
            stdout="AGENT_RESULT: PASS q1\nquota exceeded",
            stderr="",
            has_result=True,
        )
        assert failure is not None
        assert failure.kind is FailureKind.LIMIT

    def test_crash_detail_embeds_exit_code(self) -> None:
        failure = classify(returncode=137, stdout="", stderr="killed", has_result=True)
        assert failure is not None
        assert failure.kind is FailureKind.CRASH
        assert "137" in failure.detail
        assert failure.exit_code == 137

    def test_empty_when_only_signal_is_missing_result(self) -> None:
        failure = classify(returncode=0, stdout="some text", stderr="", has_result=False)
        assert failure is not None
        assert failure.kind is FailureKind.EMPTY
        assert failure.detail == "no AGENT_RESULT line"

    def test_empty_with_none_returncode(self) -> None:
        failure = classify(returncode=None, stdout="", stderr="", has_result=False)
        assert failure is not None
        assert failure.kind is FailureKind.EMPTY
        assert failure.exit_code is None


class TestLimitFalsePositives:
    @pytest.mark.parametrize(
        "text",
        [
            "insufficient test coverage on greet()",
            "the quota module now exports QuotaError",
            "AGENT_RESULT: DONE q1\ninsufficient context in the brief",
        ],
    )
    def test_ordinary_prose_is_not_a_limit(self, text: str) -> None:
        assert find_limit(text, "") is None
        assert classify(returncode=0, stdout=text, stderr="", has_result=True) is None

    @pytest.mark.parametrize(
        "text",
        [
            "Error: insufficient credits",
            "insufficient balance to run this request",
            "quota exceeded for model x",
            "You have exceeded your quota",
            "quota reached, try later",
        ],
    )
    def test_real_limit_lines_still_match(self, text: str) -> None:
        assert find_limit(text, "") is not None
