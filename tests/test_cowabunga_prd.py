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
        ("minimal", "low"),  # the bug: claude CLI rejects 'minimal'
        ("auto", None),  # auto means "don't pass --effort at all"
        ("low", "low"),
        ("high", "high"),
        ("max", "max"),
        ("medium", "medium"),
        ("xhigh", "xhigh"),
        ("bogus", None),  # unknown → omit rather than crash the subprocess
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

    def fake_subprocess(
        cmd: list[str],
        timeout: int = 0,
        cwd: object = None,
        on_line: object = None,
    ) -> object:
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout='{"result": "ok"}', stderr="")

    monkeypatch.setattr(claude_cli, "run_subprocess", fake_subprocess)
    claude_cli.run("---\nname: prd\n---\nbody", "opus", output_format="json")

    cmd = captured["cmd"]
    # The prompt is the final arg, immediately preceded by the '--' terminator.
    assert cmd[-1].startswith("---\nname: prd")
    assert cmd[-2] == "--"
    assert "stream-json" in cmd


def test_claude_json_runs_use_stream_json_for_live_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from splinter.providers import claude_cli

    captured: dict[str, object] = {}

    def fake_subprocess(
        cmd: list[str],
        timeout: int = 0,
        cwd: object = None,
        on_line: object = None,
    ) -> object:
        captured["cmd"] = cmd
        captured["on_line"] = on_line
        stdout = (
            '{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Read",'
            '"input":{"file_path":"foo.py"}}]}}\n'
            '{"type":"result","result":"ok","usage":{"input_tokens":1,"output_tokens":2},'
            '"session_id":"sid1","is_error":false}\n'
        )
        if on_line is not None:
            for line in stdout.splitlines():
                on_line(line)
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(claude_cli, "run_subprocess", fake_subprocess)
    result = claude_cli.run("hi", "sonnet")

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert "stream-json" in cmd
    assert "--verbose" in cmd
    assert captured["on_line"] is claude_cli._stream_claude_event
    assert result.text == "ok"
    assert result.raw.get("_session_id") == "sid1"


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
    configure.invalidate_config_cache()
    assert configure.configured_timeout() == 7200


def test_claude_run_uses_configured_timeout_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from splinter.providers import claude_cli

    seen: dict[str, int] = {}

    def fake_subprocess(
        cmd: list[str],
        timeout: int = 0,
        cwd: object = None,
        on_line: object = None,
    ) -> object:
        seen["timeout"] = timeout
        return SimpleNamespace(returncode=0, stdout='{"result": "ok"}', stderr="")

    monkeypatch.setattr(claude_cli, "run_subprocess", fake_subprocess)
    monkeypatch.setattr("splinter.configure.configured_timeout", lambda: 4242)
    claude_cli.run("hi", "sonnet")
    assert seen["timeout"] == 4242


def test_run_text_routes_by_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """opencode-go/* → opencode provider; claude aliases → claude provider."""
    from splinter.providers import dispatch
    from splinter.providers.base import ProviderResponse

    calls: list[str] = []

    class _FakeProvider:
        def __init__(self, label: str) -> None:
            self._label = label

        def run(self, prompt: str, model: str, **kw: object) -> ProviderResponse:
            calls.append(f"{self._label}:{model}")
            return ProviderResponse(text="ok", tokens={}, cost=0.0)

    _providers = {
        "claude": _FakeProvider("claude"),
        "opencode": _FakeProvider("opencode"),
    }
    monkeypatch.setattr(dispatch, "get_provider", lambda name: _providers[name])

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
        "  eval: 0\n"  # invalid → ignored, keeps default
        "  tiers: [null, null, null, null, 5400]\n"
    )
    from splinter.models.roster import load_ladder

    ladder = load_ladder()
    assert ladder.planner_timeout == 1800  # overridden
    assert ladder.eval_timeout == 3600  # 0 ignored → global default
    assert ladder.tier_timeout(4) == 5400  # per-tier override
    assert ladder.tier_timeout(0) == 3600  # default fallback


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


def test_log_phase_appends_lines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    def fake_ask(prompt: str, *, resume: str | None, session: object = None) -> prd_session.Turn:
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


def test_resume_completed_session_opens_analyze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import splinter.tui as tui
    from splinter.tui import resume_session

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    # Don't launch the real TUI — just record that it was opened.
    opened: list[str] = []
    monkeypatch.setattr(tui.AnalyzeApp, "run", lambda self: opened.append("run"))

    session = Session("ses_done")
    session.set_status("completed", source="prd")
    assert resume_session("ses_done") == 0
    assert opened == ["run"]
    out = capsys.readouterr().out
    assert "finished" in out
    assert "Opening analyze" in out


def test_resume_without_id_prefers_run_checkpoint_over_refining(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import splinter.tui as tui
    from splinter.tui import resume_session

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))

    refining = Session("ses_refining")
    refining.write("prd.md", "---\nstrategy: cascade\n---\n# Stub")
    refining.set_status("refining", source="prd", phase="chat")

    interrupted = Session("ses_interrupted")
    interrupted.write("prd.md", "---\nstrategy: cascade\n---\n### US-001: Story")
    interrupted.write("checkpoint.json", '{"completed":["US-001"]}')
    interrupted.set_status(
        "running",
        source=str(interrupted.dir / "prd.md"),
        strategy="cascade",
        pid=999999,
    )

    picked: list[str] = []
    monkeypatch.setattr(tui, "_resume_run", lambda s, st, reset=False: picked.append(s.id) or 0)
    monkeypatch.setattr(tui, "_resume_prd", lambda s, st: 0)

    assert resume_session(None) == 0
    assert picked == ["ses_interrupted"]


def test_resume_without_id_uses_refining_when_no_run_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import splinter.tui as tui
    from splinter.tui import resume_session

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))

    refining = Session("ses_refining")
    refining.write("prd.md", "---\nstrategy: cascade\n---\n### US-001: Story")
    refining.set_status("refining", source="prd", phase="review")

    picked: list[str] = []
    monkeypatch.setattr(tui, "_resume_prd", lambda s, st: picked.append(s.id) or 0)
    monkeypatch.setattr(tui, "_resume_run", lambda s, st, reset=False: 0)

    assert resume_session(None) == 0
    assert picked == ["ses_refining"]


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
# Pure unit tests of DirectStrategy's retry/escalate policy. Every model call is
# mocked — run_task, the gate, the evaluator's judge(), and the planner — so the
# loop is exercised without spawning a single claude/opencode subprocess.


def _drive_loop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    verdicts: list[EvalVerdict],
    *,
    cowabunga: bool,
    task: Task | None = None,
    session: Session | None = None,
    prd: str | None = None,
) -> tuple[Session, list[int]]:
    """Run DirectStrategy with scripted eval verdicts; return (session, tiers seen)."""
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    tiers_seen: list[int] = []

    def fake_run_task(t: Task, plan: str, tier: int, ladder: object, **kw: object) -> RunResult:
        tiers_seen.append(tier)
        return RunResult(
            text="did the thing",
            model="stub",
            tier=tier,
            tokens={"in": 1, "out": 1},
            cost=0.0,
            raw={},
            opencode_session=None,
        )

    queue = list(verdicts)

    # EvalStage builds `Evaluator(ctx.ladder)` and calls `.judge()`, which would
    # otherwise hit run_text → a real CLI. Stub judge() on the class so it returns
    # the next scripted verdict instead. next_action() (pure policy) stays real.
    def fake_judge(self: object, *a: object, **k: object) -> EvalVerdict:
        return queue.pop(0)

    # Gate: raising → stages treats it as "no gate configured" → pass.
    def boom_gate() -> object:
        raise RuntimeError("no gate")

    monkeypatch.setattr("splinter.strategies.stages.run_task", fake_run_task)
    monkeypatch.setattr("splinter.strategies.stages.run_gate", boom_gate)
    monkeypatch.setattr("splinter.strategies.stages.Evaluator.judge", fake_judge)
    monkeypatch.setattr("splinter.strategies.direct._make_plan", lambda *a, **k: "the plan")

    from splinter.models.roster import load_ladder

    run_session = session or Session("ses_test_loop")
    if prd:
        run_session.write("prd.md", prd)
    run_task = task or Task(description="t", acceptance="a", effort="normal")
    DirectStrategy().execute(
        [run_task],
        run_session,
        load_ladder(),
        max_iterations=6,
        cowabunga=cowabunga,
    )
    return run_session, tiers_seen


def _v(decision: str) -> EvalVerdict:
    return EvalVerdict(decision=decision, reason="r", corrections="c", raw="")


def test_ask_user_stops_without_cowabunga(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from splinter.strategies.base import AskUserPause

    with pytest.raises(AskUserPause):
        _drive_loop(monkeypatch, tmp_path, [_v(Decision.ASK_USER)], cowabunga=False)
    session = Session("ses_test_loop")
    assert "ASK_USER" in session.read("loop.md")
    assert session.read("run_checkpoint.json").strip()


def test_ask_user_stops_with_cowabunga(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    session, tiers = _drive_loop(
        monkeypatch, tmp_path, [_v(Decision.ASK_USER), _v(Decision.PASS)], cowabunga=True
    )
    # cowabunga = no human gate: ASK_USER doesn't pause for input — the task just
    # stops after the first iteration (mirrors Evaluator.next_action's stop=True).
    assert tiers == [1]
    # It stopped without flagging a human handoff (no ASK_USER section written).
    assert "needs human input" not in session.read("loop.md")


def test_jump_premium_skips_to_premium_tier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session, tiers = _drive_loop(
        monkeypatch, tmp_path, [_v(Decision.JUMP_PREMIUM), _v(Decision.PASS)], cowabunga=False
    )
    # Started at tier 1 (normal), jumped straight to premium tier 3.
    assert tiers[0] == 1
    assert tiers[1] == 3


def test_max_tier_without_pass_pauses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from splinter.strategies.base import AskUserPause

    with pytest.raises(AskUserPause):
        _drive_loop(
            monkeypatch,
            tmp_path,
            [_v(Decision.ESCALATE)] * 5,
            cowabunga=True,
        )


def test_jump_premium_stuck_pauses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from splinter.strategies.base import AskUserPause

    with pytest.raises(AskUserPause):
        _drive_loop(
            monkeypatch,
            tmp_path,
            [_v(Decision.JUMP_PREMIUM), _v(Decision.JUMP_PREMIUM)],
            cowabunga=False,
        )


def test_loop_exhausted_without_pass_pauses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from splinter.strategies.base import AskUserPause

    with pytest.raises(AskUserPause):
        _drive_loop(
            monkeypatch,
            tmp_path,
            [_v(Decision.RETRY)] * 6,
            cowabunga=False,
            task=Task(description="US-001: First", acceptance="a", effort="normal"),
            prd="""---
strategy: direct
---

### US-001: First
**Acceptance Criteria:**
- [ ] does A
""",
        )


def test_pause_blocks_next_prd_task(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from splinter.strategies.base import AskUserPause

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    ran: list[str] = []

    def fake_run_task(task: Task, plan: str, tier: int, ladder: object, **kw: object) -> RunResult:
        ran.append(task.description)
        return RunResult(
            text="did the thing",
            model="stub",
            tier=tier,
            tokens={"in": 1, "out": 1},
            cost=0.0,
            raw={},
            opencode_session=None,
        )

    def boom_gate() -> object:
        raise RuntimeError("no gate")

    monkeypatch.setattr("splinter.strategies.stages.run_task", fake_run_task)
    monkeypatch.setattr("splinter.strategies.stages.run_gate", boom_gate)
    monkeypatch.setattr(
        "splinter.strategies.stages.Evaluator.judge",
        lambda self, *a, **k: _v(Decision.ASK_USER),
    )
    monkeypatch.setattr("splinter.strategies.direct._make_plan", lambda *a, **k: "the plan")

    from splinter.models.roster import load_ladder

    session = Session("ses_pause_multi")
    session.write("prd.md", _PRD)
    tasks = [
        Task(description="US-001: First", acceptance="a", effort="normal"),
        Task(description="US-002: Second", acceptance="b", effort="normal"),
    ]
    with pytest.raises(AskUserPause):
        DirectStrategy().execute(tasks, session, load_ladder(), cowabunga=False)
    assert len(ran) == 1
    assert ran[0].startswith("US-001")


# --- PRD as source of truth: checkbox progress + multi-task resume -----------

_PRD = """---
strategy: direct
---

### US-001: First
**Acceptance Criteria:**
- [ ] does A
- [ ] does B

### US-002: Second
**Acceptance Criteria:**
- [ ] does C

### US-003: Third
**Acceptance Criteria:**
- [ ] does D
"""


def test_mark_story_done_ticks_only_its_block() -> None:
    out = prd_session.mark_story_done(_PRD, "US-002")
    assert "### US-002: Second\n**Acceptance Criteria:**\n- [x] does C" in out
    # Other stories untouched.
    assert "- [ ] does A" in out
    assert "- [ ] does D" in out


def test_completed_story_ids() -> None:
    assert prd_session.completed_story_ids(_PRD) == set()
    done = prd_session.mark_story_done(prd_session.mark_story_done(_PRD, "US-001"), "US-002")
    assert prd_session.completed_story_ids(done) == {"US-001", "US-002"}


def test_story_id() -> None:
    assert prd_session.story_id("US-003: CLI overrides") == "US-003"
    assert prd_session.story_id("no id here") is None


def _drive_tasks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tasks: list[Task],
    *,
    resume: bool,
    prd: str,
) -> tuple[Session, list[str]]:
    """Run DirectStrategy over multiple tasks (every model call mocked, PASS each)."""
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    ran: list[str] = []

    def fake_run_task(task: Task, plan: str, tier: int, ladder: object, **kw: object) -> RunResult:
        ran.append(task.description)
        return RunResult(
            text="done",
            model="stub",
            tier=tier,
            tokens={"in": 1, "out": 1},
            cost=0.0,
            raw={},
            opencode_session=None,
        )

    def boom_gate() -> object:
        raise RuntimeError("no gate")

    monkeypatch.setattr("splinter.strategies.stages.run_task", fake_run_task)
    monkeypatch.setattr("splinter.strategies.stages.run_gate", boom_gate)

    def fake_judge(self: object, *a: object, **k: object) -> EvalVerdict:
        return EvalVerdict(decision=Decision.PASS, reason="ok", corrections="", raw="")

    monkeypatch.setattr("splinter.strategies.stages.Evaluator.judge", fake_judge)
    monkeypatch.setattr("splinter.strategies.direct._make_plan", lambda *a, **k: "plan")

    from splinter.models.roster import load_ladder

    session = Session("ses_multi")
    session.write("prd.md", prd)
    DirectStrategy().execute(tasks, session, load_ladder(), max_iterations=3, resume=resume)
    return session, ran


def test_single_shot_pass_checks_off_all_stories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Raphael is single-shot: the pipeline merges every story into one task, so a
    single holistic PASS ticks all stories at once (task_index ends at 1)."""
    from splinter.pipeline import _merge_stories_into_task

    stories = [
        Task(description="US-001: First", acceptance="does A\ndoes B"),
        Task(description="US-002: Second", acceptance="does C"),
        Task(description="US-003: Third", acceptance="does D"),
    ]
    merged = _merge_stories_into_task(_PRD, stories)
    session, ran = _drive_tasks(monkeypatch, tmp_path, [merged], resume=False, prd=_PRD)
    assert len(ran) == 1
    assert prd_session.completed_story_ids(session.read("prd.md")) == {"US-001", "US-002", "US-003"}
    assert session.read_status().get("task_index") == 1


def test_single_shot_runs_one_task_on_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Single-shot has no per-story skipping — resume drives the one merged task,
    regardless of which stories were previously ticked."""
    from splinter.pipeline import _merge_stories_into_task

    stories = [
        Task(description="US-001: First", acceptance="does A\ndoes B"),
        Task(description="US-002: Second", acceptance="does C"),
        Task(description="US-003: Third", acceptance="does D"),
    ]
    merged = _merge_stories_into_task(_PRD, stories)
    prd = prd_session.mark_story_done(_PRD, "US-001")
    session, ran = _drive_tasks(monkeypatch, tmp_path, [merged], resume=True, prd=prd)
    assert len(ran) == 1


# --- PRD version snapshots -----------------------------------------------------


def test_save_prd_version_numbers_and_dedupes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from splinter.memory.session import Session

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_prd_ver")
    v1 = "# PRD v1\n\n### US-001: A"
    v2 = "# PRD v2\n\n### US-001: A\n### US-002: B"
    prd_session.save_prd_version(session, v1, label="generate", detail="1 stories")
    n2 = prd_session.save_prd_version(session, v2, label="refine")
    assert n2 == 1
    assert prd_session.save_prd_version(session, v2, label="refine") is None
    versions = prd_session.list_prd_versions(session)
    assert [v.num for v in versions] == [0, 1]
    assert versions[0].label == "generate"
    assert versions[1].label == "refine"


def test_should_accept_prd_update_rejects_wipe(tmp_path: Path) -> None:
    stories = "\n".join(f"### US-{i:03d}: Story {i}\n**Description:** x\n" for i in range(1, 26))
    big = f"---\nfeature: x\n---\n\n{stories}"
    assert prd_session.should_accept_prd_update(big, "### PresentmentAmount") is False
    assert prd_session.should_accept_prd_update(big, big + "\n\nmore detail") is True


def test_render_prd_version_compare_shows_previous_and_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from splinter.memory.session import Session

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_prd_cmp")
    prd_session.save_prd_version(session, "# before\n\n### US-001: A", label="generate")
    prd_session.save_prd_version(session, "# after\n\n### US-001: A\n### US-002: B", label="refine")
    versions = prd_session.list_prd_versions(session)
    md = prd_session.render_prd_version_compare(session, versions[1])
    assert "## Previous" in md
    assert "# before" in md
    assert "## Current" in md
    assert "# after" in md
    assert "Stories: 1 → 2" in md


def test_version_for_phase_maps_occurrence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from splinter.memory.session import Session

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_prd_map")
    prd_session.save_prd_version(session, "v0", label="generate")
    prd_session.save_prd_version(session, "v1", label="refine")
    prd_session.save_prd_version(session, "v2", label="refine")
    versions = prd_session.list_prd_versions(session)
    assert prd_session.version_for_phase(versions, "refine", 0).num == 1
    assert prd_session.version_for_phase(versions, "refine", 1).num == 2


def test_extract_working_draft_prefers_section_over_small_fence() -> None:
    text = (
        "## Working Draft\n\n"
        "```markdown\n### PresentmentAmount\n```\n\n"
        "---\nfeature: x\n---\n\n### US-001: Full PRD\n\n"
        "## Open Questions\nNone"
    )
    out = prd_session.extract_working_draft(text)
    assert "### US-001: Full PRD" in out
    assert "---" in out


# --- gate short-circuits eval on failure; eval session continuity ---------------------


def test_gate_failure_skips_eval_forces_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Gate fail must short-circuit eval and force a RETRY. Eval runs only after gate passes."""
    from splinter.agents.gate import GateResult

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    judge_calls: list[int] = []
    corrections_seen: list[str] = []

    def fake_run_task(
        task: Task, plan: str, tier: int, ladder: object, *, corrections: str = "", **kw: object
    ) -> RunResult:
        corrections_seen.append(corrections)
        return RunResult(
            text="x", model="stub", tier=tier, tokens={"in": 1}, cost=0.0, raw={},
            opencode_session=None,
        )

    gate_calls = [0]

    def fake_run_gate(**k: object) -> GateResult:
        gate_calls[0] += 1
        # fail first two calls, pass on the third
        if gate_calls[0] < 3:
            return GateResult(passed=False, checks=[("pytest", False, "BOOM output")])
        return GateResult(passed=True, checks=[("pytest", True, "")])

    def fake_judge_pass(self: object, *a: object, **k: object) -> EvalVerdict:
        judge_calls.append(1)
        return EvalVerdict(decision=Decision.PASS, reason="ok", corrections="", raw="")

    monkeypatch.setattr("splinter.strategies.stages.run_task", fake_run_task)
    monkeypatch.setattr("splinter.strategies.stages.run_gate", fake_run_gate)
    monkeypatch.setattr("splinter.strategies.stages.Evaluator.judge", fake_judge_pass)
    monkeypatch.setattr("splinter.strategies.direct._make_plan", lambda *a, **k: "plan")

    from splinter.models.roster import load_ladder

    session = Session("ses_gate_shortcircuit")
    DirectStrategy().execute(
        [Task(description="t", acceptance="a")], session, load_ladder(), max_iterations=5
    )
    # Eval only called once (after gate finally passes on iter 3).
    assert judge_calls == [1]
    assert any("BOOM output" in c for c in corrections_seen)
    for c in corrections_seen:
        if "BOOM output" in c:
            assert c.count("BOOM output") == 1
    assert "gate: FAIL" in session.read("loop.md")
    assert "gate: PASS" in session.read("loop.md")


def test_eval_session_continues_same_runner_resets_on_escalate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same runner → eval keeps its session; on escalate (runner change) the eval
    session resets. Gate passes so eval runs normally each iteration."""
    from splinter.agents.gate import GateResult

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    eval_sessions_in: list[str | None] = []

    def fake_run_task(
        task: Task, plan: str, tier: int, ladder: object, *, corrections: str = "", **kw: object
    ) -> RunResult:
        return RunResult(
            text="x",
            model="stub",
            tier=tier,
            tokens={"in": 1},
            cost=0.0,
            raw={},
            opencode_session=f"oc{tier}",
        )

    verdicts = iter(
        [
            EvalVerdict(Decision.RETRY, "fix", corrections="do X", eval_session="ev1"),
            EvalVerdict(Decision.ESCALATE, "cant", corrections="harder", eval_session="ev2"),
            EvalVerdict(Decision.PASS, "ok", eval_session="ev3"),
        ]
    )

    def fake_judge(
        self: object, *a: object, session: str | None = None, **k: object
    ) -> EvalVerdict:
        eval_sessions_in.append(session)
        return next(verdicts)

    monkeypatch.setattr("splinter.strategies.stages.run_task", fake_run_task)
    monkeypatch.setattr(
        "splinter.strategies.stages.run_gate",
        lambda **k: GateResult(passed=True, checks=[("pytest", True, "")]),
    )
    monkeypatch.setattr("splinter.strategies.stages.Evaluator.judge", fake_judge)
    monkeypatch.setattr("splinter.strategies.direct._make_plan", lambda *a, **k: "plan")

    from splinter.models.roster import load_ladder

    session = Session("ses_evalsess")
    DirectStrategy().execute(
        [Task(description="t", acceptance="a", effort="normal")],
        session,
        load_ladder(),
        max_iterations=3,
    )
    # iter1 starts with no eval session; iter2 (same tier RETRY) continues "ev1";
    # iter2's ESCALATE resets it, so iter3 (new tier) starts fresh (None).
    assert eval_sessions_in == [None, "ev1", None]
