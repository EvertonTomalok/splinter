"""Content dependencies for each run-screen view."""

from __future__ import annotations

from splinter.tui_run_live import FRESH_S
from splinter.tui_run_views import ViewContext, relevant_paths


def view_content_key(ctx: ViewContext, name: str) -> tuple[object, ...]:
    view = ctx.view
    rows = view.iterations_by_story.get(ctx.story or 0, ())
    loop = ctx.session.dir / "loop.md"
    paths = relevant_paths(ctx, name)
    files = ctx.files.mtime_key(tuple(path for path in paths if path != loop))
    story_text = ctx.files.story_body(loop, ctx.story or 0) if loop in paths else ""
    common = (ctx.session.dir, ctx.story, ctx.width, ctx.multi, files, story_text)
    if name == "act":
        live = ctx.story is not None and ctx.story == view.live_story
        actions = tuple(action for action in ctx.live if live and action.story == ctx.story)
        latest = {action.iteration: action for action in actions}
        fresh = tuple(ctx.now - action.at < FRESH_S for action in latest.values())
        opened = frozenset(item for item in ctx.opened if item[0] == ctx.story)
        return (*common, rows, live, opened, actions, fresh)
    if name in ("gate", "eval"):
        return (*common, rows)
    if name == "spend":
        from splinter.tui_run_views_more import more_state

        state = more_state()
        return (
            *common, view.stories, view.spend, view.ladder, view.total_cost,
            ctx.spend_by, ctx.budget, state.spend_by, state.budget,
        )
    if name == "tier":
        from splinter.engines import runtime
        from splinter.tui_run_views_more import more_state

        breakers = tuple(
            (model, runtime.model_state(model)) for rung in view.ladder for model in rung.models
        )
        return (
            *common, view.stories, view.ladder, view.iterations_by_story,
            ctx.tier, more_state().tier, breakers,
        )
    if name == "diff":
        from splinter.tui_run_views_more import more_state

        state = more_state()
        row = next((story for story in view.stories if story.number == ctx.story), None)
        return (*common, row, ctx.diff_text, state.diff, state.diff_cwd)
    return common
