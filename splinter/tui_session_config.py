from __future__ import annotations

from typing import Any

from textual.app import App, SuspendNotSupported


async def edit_session_config(app: App[Any], *, busy: bool) -> bool:
    if busy:
        app.notify("Configure is available after the current model call finishes.")
        return False

    from splinter.tui_configure import ConfigureApp

    try:
        with app.suspend():
            saved = await ConfigureApp().run_async()
    except SuspendNotSupported:
        app.notify("Open splinter configure in another terminal.")
        return False
    if saved:
        app.notify("Configuration saved. The next call uses the updated settings.")
    return bool(saved)
