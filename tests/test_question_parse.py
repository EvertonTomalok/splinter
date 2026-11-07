"""Tests for the pure question parser behind the question form."""

from __future__ import annotations

from splinter.tui_questions import (
    Option,
    Question,
    extract_open_questions,
    parse_questions,
)


def test_parses_numbered_question_with_one_option_per_line() -> None:
    text = "1. Which database should we use?\nA) Postgres\nB) SQLite\n"
    assert parse_questions(text) == [
        Question(
            number=1,
            text="Which database should we use?",
            options=(Option("A", "Postgres"), Option("B", "SQLite")),
        )
    ]


def test_parses_parenthesised_question_header() -> None:
    text = "1) Which database should we use?\nA) Postgres\nB) SQLite"
    questions = parse_questions(text)
    assert [q.number for q in questions] == [1]
    assert questions[0].options[0] == Option("A", "Postgres")


def test_parses_bold_question_header() -> None:
    text = "**1.** Which database should we use?\nA) Postgres\nB) SQLite"
    questions = parse_questions(text)
    assert questions[0].number == 1
    assert questions[0].text == "Which database should we use?"


def test_parses_options_on_the_same_line() -> None:
    text = "1. Which database? A) Postgres  B) SQLite"
    questions = parse_questions(text)
    assert questions[0].text == "Which database?"
    assert questions[0].options == (Option("A", "Postgres"), Option("B", "SQLite"))


def test_parses_bulleted_and_parenthesised_options() -> None:
    text = "1. Which database?\n- A: Postgres\n(B) SQLite"
    questions = parse_questions(text)
    assert questions[0].options == (Option("A", "Postgres"), Option("B", "SQLite"))


def test_parses_question_text_spanning_several_lines() -> None:
    text = (
        "1. Which database should we use?\n"
        "   It must support transactions and local development.\n"
        "A) Postgres\n"
        "B) SQLite\n"
    )
    questions = parse_questions(text)
    assert questions[0].text == (
        "Which database should we use? It must support transactions and local development."
    )
    assert len(questions[0].options) == 2


def test_parses_several_questions() -> None:
    text = (
        "1. First question?\nA) Yes\nB) No\n"
        "2. Second question?\nA) One\nB) Two\n"
        "3. Third question?\nA) Red\nB) Green\n"
    )
    questions = parse_questions(text)
    assert [q.number for q in questions] == [1, 2, 3]
    assert questions[2].options[1] == Option("B", "Green")


def test_text_without_questions_parses_to_nothing() -> None:
    assert parse_questions("The PRD looks complete. Let me know if you want changes.") == []


def test_garbage_parses_to_nothing() -> None:
    assert parse_questions("asdf 1234 !!! ??? ***") == []


def test_question_without_options_is_dropped() -> None:
    assert parse_questions("1. What is the scope of this feature?") == []


def test_extract_open_questions_reads_the_open_questions_section() -> None:
    text = (
        "## Working Draft\n"
        "1. Should we ship it?\n"
        "A) Yes\n"
        "B) No\n"
        "\n"
        "## Open Questions\n"
        "1. Which rollout strategy?\n"
        "A) Blue-green\n"
        "B) Canary\n"
    )
    questions = extract_open_questions(text)
    assert len(questions) == 1
    assert questions[0].text == "Which rollout strategy?"
    assert questions[0].options == (Option("A", "Blue-green"), Option("B", "Canary"))


def test_extract_open_questions_without_section_parses_whole_text() -> None:
    text = "1. Which rollout strategy?\nA) Blue-green\nB) Canary"
    questions = extract_open_questions(text)
    assert len(questions) == 1
    assert questions[0].number == 1
