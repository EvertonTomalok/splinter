"""Tests for the --effort fix, the interactive PRD helpers, and cowabunga/ASK_USER."""

from __future__ import annotations

from pathlib import Path

import pytest

from splinter import prd_session
from splinter.agents.runner import RunResult, Task
from splinter.enums import Decision
from splinter.memory.session import Session
from splinter.providers.claude_cli import _normalize_effort
from splinter.strategies.base import EvalVerdict
from splinter.strategies.direct import DirectStrategy
from splinter.tui import _fm_block, _set_fm_strategy

# --- the original crash: --effort minimal -----------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("minimal", "low"),   # the bug: claude CLI rejects 'minimal'
        ("auto", None),       # auto means "don't pass --effort at all"
        ("low", "low"),
        ("high", "high"),
        ("max", "max"),
        ("medium", "medium"),
        ("xhigh", "xhigh"),
        ("bogus", None),      # unknown → omit rather than crash the subprocess
        (None, None),
    ],
)
def test_normalize_effort(raw: str | None, expected: str | None) -> None:
    assert _normalize_effort(raw) == expected


def test_prompt_with_leading_dashes_is_not_parsed_as_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A PRD prompt starting with '---' must ride behind '--', not crash the CLI."""
    from types import SimpleNamespace

    from splinter.providers import claude_cli

    captured: dict[str, list[str]] = {}

    def fake_subprocess(cmd: list[str], timeout: int = 0) -> object:
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout='{"result": "ok"}', stderr="")

    monkeypatch.setattr(claude_cli, "run_subprocess", fake_subprocess)
    claude_cli.run("---\nname: prd\n---\nbody", "opus", output_format="json")

    cmd = captured["cmd"]
    # The prompt is the final arg, immediately preceded by the '--' terminator.
    assert cmd[-1].startswith("---\nname: prd")
    assert cmd[-2] == "--"


def test_configured_timeout_default_and_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from splinter import configure

    # No config file → 1 hour default.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(configure.Path, "home", lambda: tmp_path / "nohome")
    assert configure.configured_timeout() == 3600

    # Written override is honoured.
    (tmp_path / ".splinter").mkdir()
    (tmp_path / ".splinter" / "config.yaml").write_text("defaults:\n  timeout: 7200\n")
    assert configure.configured_timeout() == 7200


def test_claude_run_uses_configured_timeout_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from splinter.providers import claude_cli

    seen: dict[str, int] = {}

    def fake_subprocess(cmd: list[str], timeout: int = 0) -> object:
        seen["timeout"] = timeout
        return SimpleNamespace(returncode=0, stdout='{"result": "ok"}', stderr="")

    monkeypatch.setattr(claude_cli, "run_subprocess", fake_subprocess)
    monkeypatch.setattr("splinter.configure.configured_timeout", lambda: 4242)
    claude_cli.run("hi", "sonnet")
    assert seen["timeout"] == 4242


def test_run_text_routes_by_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """opencode-go/* → opencode CLI; claude aliases → claude -p."""
    from types import SimpleNamespace

    from splinter.providers import dispatch

    calls: list[str] = []

    def fake_claude(prompt: str, model: str, **kw: object) -> object:
        calls.append(f"claude:{model}")
        return SimpleNamespace(text="c")

    def fake_opencode(prompt: str, model: str, **kw: object) -> object:
        calls.append(f"opencode:{model}")
        return SimpleNamespace(text="o")

    monkeypatch.setattr(dispatch.claude_cli, "run", fake_claude)
    monkeypatch.setattr(dispatch.opencode, "run", fake_opencode)

    dispatch.run_text("p", "sonnet")
    dispatch.run_text("p", "opus")
    dispatch.run_text("p", "opencode-go/deepseek-v4-flash")
    assert calls == ["claude:sonnet", "claude:opus", "opencode:opencode-go/deepseek-v4-flash"]


def test_per_step_timeouts_resolve_into_ladder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Per-step `timeouts` override individual steps; the rest fall back to default."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("splinter.configure.Path", __import__("pathlib").Path)
    (tmp_path / ".splinter").mkdir()
    (tmp_path / ".splinter" / "config.yaml").write_text(
        "defaults:\n  timeout: 3600\n"
        "timeouts:\n"
        "  planner: 1800\n"
        "  eval: 0\n"            # invalid → ignored, keeps default
        "  tiers: [null, null, null, null, 5400]\n"
    )
    from splinter.models.roster import load_ladder

    ladder = load_ladder()
    assert ladder.planner_timeout == 1800     # overridden
    assert ladder.eval_timeout == 3600        # 0 ignored → global default
    assert ladder.tier_timeout(4) == 5400     # per-tier override
    assert ladder.tier_timeout(0) == 3600     # default fallback


# --- prd_session pure helpers ------------------------------------------------


@pytest.mark.parametrize("word", ["cowabunga", "COWABUNGA", "  Cowabunga "])
def test_is_cowabunga(word: str) -> None:
    assert prd_session.is_cowabunga(word)


def test_is_cowabunga_rejects_other() -> None:
    assert not prd_session.is_cowabunga("cowabunga dude")
    assert not prd_session.is_cowabunga("yes")


@pytest.mark.parametrize("word", ["fulfilled", "done", "READY", " go "])
def test_is_done(word: str) -> None:
    assert prd_session.is_done(word)


def test_ensure_frontmatter_adds_when_missing() -> None:
    out = prd_session.ensure_frontmatter(
        "# Feature\nbody", description="My Cool Thing", strategy="raphael"
    )
    assert out.startswith("---\n")
    assert "feature: my-cool-thing" in out
    assert "strategy: raphael" in out


def test_ensure_frontmatter_keeps_existing() -> None:
    src = "---\nfeature: x\nstrategy: direct\n---\n\nbody"
    out = prd_session.ensure_frontmatter(src, description="x", strategy="direct")
    # No second frontmatter block injected.
    assert out.count("---") == 2


def test_log_phase_appends_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_prd")
    prd_session.log_phase(session, "clarify")
    prd_session.log_phase(session, "finalize", "3 stories")
    body = session.read(prd_session.PRD_PHASE_FILE)
    assert body.splitlines() == ["- clarify", "- finalize · 3 stories"]


def test_user_story_titles() -> None:
    prd = "### US-001: First\nblah\n### US-002: Second thing\nblah"
    assert prd_session.user_story_titles(prd) == ["US-001: First", "US-002: Second thing"]


# --- resume ------------------------------------------------------------------


def test_resume_preamble_only_when_conversation_lost() -> None:
    # Live conversation (resume id present) → no re-seed.
    assert prd_session._resume_preamble("draft", resume="sess_abc") == ""
    # No draft → nothing to seed.
    assert prd_session._resume_preamble("", resume="") == ""
    assert prd_session._resume_preamble(None, resume="") == ""
    # Lost conversation + a draft → re-seed with the draft.
    out = prd_session._resume_preamble("### US-001: A", resume="")
    assert "RESUMING" in out
    assert "### US-001: A" in out


def test_refine_reseeds_draft_without_resume_id(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def fake_ask(prompt: str, *, resume: str | None) -> prd_session.Turn:
        captured["prompt"] = prompt
        captured["resume"] = resume or ""
        return prd_session.Turn(text="ok", session_id="new")

    monkeypatch.setattr(prd_session, "_ask", fake_ask)
    prd_session.refine("1A", resume="", prd_text="### US-001: Draft")
    assert "### US-001: Draft" in captured["prompt"]

    # With a live resume id the draft is NOT re-injected (server keeps context).
    prd_session.refine("1A", resume="sess_x", prd_text="### US-001: Draft")
    assert "### US-001: Draft" not in captured["prompt"]


def test_resume_no_resumable_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from splinter.tui import resume_session

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    assert resume_session(None) == 1
    assert "no resumable" in capsys.readouterr().out


def test_resume_rejects_unknown_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from splinter.tui import resume_session

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    assert resume_session("ses_does_not_exist") == 1
    assert "no such session" in capsys.readouterr().out


def test_resume_rejects_completed_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from splinter.tui import resume_session

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_done")
    session.set_status("completed", source="prd")
    assert resume_session("ses_done") == 1
    assert "not resumable" in capsys.readouterr().out


# --- tui frontmatter helpers -------------------------------------------------


def test_fm_block_parses() -> None:
    fm, body = _fm_block("---\nstrategy: direct\nfeature: x\n---\n\nhello")
    assert fm == {"strategy": "direct", "feature": "x"}
    assert "hello" in body


def test_set_fm_strategy_overrides() -> None:
    src = "---\nfeature: x\nstrategy: direct\n---\n\nbody"
    out = _set_fm_strategy(src, "raphael")
    fm, _ = _fm_block(out)
    assert fm["strategy"] == "raphael"
    assert fm["feature"] == "x"


# --- the eval loop: ASK_USER + JUMP_PREMIUM + cowabunga ----------------------


def _drive_loop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    verdicts: list[EvalVerdict],
    *,
    cowabunga: bool,
) -> tuple[Session, list[int]]:
    """Run DirectStrategy with scripted eval verdicts; return (session, tiers seen)."""
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    tiers_seen: list[int] = []

    def fake_run_task(task: Task, plan: str, tier: int, ladder: object, **kw: object) -> RunResult:
        tiers_seen.append(tier)
        return RunResult(
            text="did the thing", model="stub", tier=tier,
            tokens={"in": 1, "out": 1}, cost=0.0, raw={}, opencode_session=None,
        )

    queue = list(verdicts)

    def fake_evaluate(*a: object, **k: object) -> EvalVerdict:
        return queue.pop(0)

    # Gate: raising → stages treats it as "no gate configured" → pass.
    def boom_gate() -> object:
        raise RuntimeError("no gate")

    monkeypatch.setattr("splinter.strategies.stages.run_task", fake_run_task)
    monkeypatch.setattr("splinter.strategies.stages.run_gate", boom_gate)
    monkeypatch.setattr("splinter.strategies.stages._evaluate", fake_evaluate)
    monkeypatch.setattr(
        "splinter.strategies.direct._make_plan", lambda *a, **k: "the plan"
    )

    from splinter.models.roster import load_ladder

    session = Session("ses_test_loop")
    DirectStrategy().execute(
        [Task(description="t", acceptance="a", effort="normal")],
        session,
        load_ladder(),
        max_iterations=6,
        cowabunga=cowabunga,
    )
    return session, tiers_seen


def _v(decision: str) -> EvalVerdict:
    return EvalVerdict(decision=decision, reason="r", corrections="c", raw="")


def test_ask_user_stops_without_cowabunga(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session, tiers = _drive_loop(
        monkeypatch, tmp_path, [_v(Decision.ASK_USER)], cowabunga=False
    )
    # Stopped after the very first iteration, handed to the human.
    assert tiers == [1]
    assert "ASK_USER" in session.read("loop.md")


def test_ask_user_continues_with_cowabunga(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session, tiers = _drive_loop(
        monkeypatch, tmp_path, [_v(Decision.ASK_USER), _v(Decision.PASS)], cowabunga=True
    )
    # cowabunga = no asking: it retried and then passed.
    assert len(tiers) == 2


def test_jump_premium_skips_to_premium_tier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session, tiers = _drive_loop(
        monkeypatch, tmp_path, [_v(Decision.JUMP_PREMIUM), _v(Decision.PASS)], cowabunga=False
    )
    # Started at tier 1 (normal), jumped straight to premium tier 3.
    assert tiers[0] == 1
    assert tiers[1] == 3
