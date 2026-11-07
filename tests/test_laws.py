"""Tests for the workhorse contract, AGENT_RESULT protocol and brief document."""

from __future__ import annotations

from splinter.agents.laws import (
    REPORT_PATH,
    RESULT_LINE_PATTERN,
    AgentResult,
    brief_document,
    parse_agent_result,
    workhorse_contract,
)


class TestParseAgentResult:
    def test_match(self) -> None:
        result = parse_agent_result("noise\nAGENT_RESULT: DONE q1\nbye")
        assert result == AgentResult(status="DONE", ident="q1")

    def test_last_match_wins(self) -> None:
        text = "AGENT_RESULT: DONE q1\nmiddle\nAGENT_RESULT: BLOCKED pb3a"
        assert parse_agent_result(text) == AgentResult(status="BLOCKED", ident="pb3a")

    def test_no_match_returns_none(self) -> None:
        assert parse_agent_result("AGENT RESULT: done q1") is None
        assert parse_agent_result("") is None

    def test_blocked_status_with_dashed_ident(self) -> None:
        assert parse_agent_result("AGENT_RESULT: BLOCKED q-12a") == AgentResult(
            status="BLOCKED", ident="q-12a"
        )

    def test_pattern_requires_uppercase_status(self) -> None:
        assert not RESULT_LINE_PATTERN.findall("AGENT_RESULT: done q1")


class TestWorkhorseContract:
    def test_contains_the_seven_contract_items(self) -> None:
        text = workhorse_contract()
        for marker in (
            "1. Read the brief, then only the files it names",
            "2. Edit only the files the brief lists",
            "3. Follow the project rules below",
            "4. Run no build, test, lint, generator",
            "5. Workhorse mode: execute directly",
            "6. Finish in minutes",
            "7. Last two actions, always",
        ):
            assert marker in text

    def test_protocol_lines_are_exact(self) -> None:
        text = workhorse_contract()
        assert "`AGENT_RESULT: DONE <id>` or `AGENT_RESULT: BLOCKED <id>`." in text
        assert "BLOCKED by the run guard" in text
        assert REPORT_PATH in text

    def test_with_laws_keeps_laws_section_and_plan_reference(self) -> None:
        laws = "## Law one\n- no mocks of what you own"
        text = workhorse_contract(laws=laws, plan_ref="/plans/slug")
        assert "## The laws of this project (read them, they bind every change)" in text
        assert laws in text
        assert "- The plan this run implements: /plans/slug/README.md" in text
        assert "the brief is the authority for the task" in text

    def test_without_laws_drops_header_keeps_plan_reference(self) -> None:
        text = workhorse_contract(plan_ref="/plans/x")
        assert "## The laws of this project" not in text
        assert "- The plan this run implements: /plans/x/README.md" in text

    def test_ends_with_plan_line_and_newline(self) -> None:
        assert workhorse_contract().endswith("the plan for decisions).\n")


class TestBriefDocument:
    def test_title_uses_goal_first_line(self) -> None:
        doc = brief_document(ident="q1", goal="Add the guard\nMore context")
        assert doc.splitlines()[0] == "# Brief q1 — Add the guard"
        assert "## Goal" in doc
        assert "More context" in doc

    def test_title_without_goal(self) -> None:
        doc = brief_document(ident="pb3a")
        assert doc.splitlines()[0] == "# Brief pb3a"

    def test_context_line_matches_template(self) -> None:
        doc = brief_document(
            ident="q1",
            goal="g",
            worktree="/wt/slug-q1",
            branch="splinter/slug",
            run_dir="/run",
        )
        line = "Worktree: `/wt/slug-q1` (branch `splinter/slug-q1`). Run folder: `/run`."
        assert line in doc

    def test_context_line_omitted_when_empty(self) -> None:
        doc = brief_document(ident="q1")
        assert "Worktree:" not in doc
        assert "Run folder:" not in doc

    def test_files_bullets_and_static_rule(self) -> None:
        doc = brief_document(ident="q1", files=["splinter/vcs/guard.py", "tests/x.py"])
        assert "## Files you may touch" in doc
        assert "- `splinter/vcs/guard.py`" in doc
        assert "- `tests/x.py`" in doc
        assert "Everything else stays as it is." in doc

    def test_sketch_section(self) -> None:
        doc = brief_document(ident="q1", sketch="Follow splinter/vcs/worktree.py")
        assert "## Change sketch" in doc
        assert "Follow splinter/vcs/worktree.py" in doc

    def test_acceptance_checkboxes(self) -> None:
        doc = brief_document(ident="q1", acceptance=["gate passes", "eval returns PASS"])
        assert "## Acceptance" in doc
        assert "- [ ] gate passes" in doc
        assert "- [ ] eval returns PASS" in doc

    def test_report_instruction_mentions_path_and_result_line(self) -> None:
        doc = brief_document(ident="q1")
        assert "## Report" in doc
        assert REPORT_PATH in doc
        assert f"Write `{REPORT_PATH}` in your worktree" in doc
        assert "`AGENT_RESULT: DONE q1` (or `BLOCKED q1`)." in doc

    def test_report_instruction_parses_back(self) -> None:
        doc = brief_document(ident="q1")
        assert parse_agent_result(doc) == AgentResult(status="DONE", ident="q1")
