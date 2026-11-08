"""Full-screen trace of the run: the outline, the selected text, the live log.

Same frame as the run screen: crumbs on top, the block word and one sentence, the
body, the clickable keys at the bottom. The left tree is
:func:`splinter.run_trace.build_trace` (needs you, now, done, queued, events; each
story opens into its plan, files and iterations, each iteration into its review,
gate, actions and output). The right pane names the path of the selected line and
shows its full text in a read-only editor, where it can be selected and copied. The
live log keeps only the rows it has, at most six, above the keys.
"""

from __future__ import annotations

from collections.abc import Iterable

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import RichLog, TextArea, Tree
from textual.widgets.tree import TreeNode

from splinter.run_trace import TraceNode
from splinter.tui_blockword import BlockWord
from splinter.tui_run_chrome import key_markup
from splinter.tui_run_palette import BG, DIM, INK, LINE, LINE2, MUTED, Ink

#: Rows the live log may take; it shrinks to the rows it has.
LOG_ROWS = 6
_KEYS = (
    ("enter", "open", "screen.open_node"),
    ("e", "expand all", "screen.expand_all"),
    ("c", "collapse", "screen.collapse_all"),
    ("r", "reload", "screen.reload"),
    ("esc", "back to the run", "screen.close"),
)


def _plain(label: object) -> str:
    return label.plain if isinstance(label, Text) else Text.from_markup(str(label)).plain


def _fill(parent: TreeNode[str], node: TraceNode) -> None:
    for child in node.children:
        if child.children:
            branch = parent.add(child.label, data=child.detail, expand=child.expanded)
            _fill(branch, child)
        else:
            parent.add_leaf(child.label, data=child.detail)


class TraceTree(Tree[str]):
    """The outline, with the activity view's thin arrows."""

    ICON_NODE = "▸ "
    ICON_NODE_EXPANDED = "▾ "


class TraceScreen(Screen[None]):
    """The run as a navigable tree, the selected node's text, and the live log."""

    BINDINGS = [
        Binding("escape", "close", "Back"),
        Binding("L", "close", "Back", show=False),
        Binding("r", "reload", "Reload"),
        Binding("e", "expand_all", "Expand all"),
        Binding("c", "collapse_all", "Collapse all"),
    ]

    DEFAULT_CSS = f"""
    TraceScreen {{ background: {BG}; }}
    TraceScreen #trace-top {{ height: 2; padding: 0 3; border-bottom: solid {LINE}; }}
    TraceScreen #trace-crumbs {{ width: 1fr; height: 1; }}
    TraceScreen #trace-stats {{ width: auto; height: 1; }}
    TraceScreen #trace-word {{ margin: 1 3 0 3; }}
    TraceScreen #trace-sentence {{ height: 1; margin: 1 3 1 3; }}
    TraceScreen #trace-body {{ height: 1fr; padding: 0 3; border-top: solid {LINE}; }}
    TraceScreen #trace-tree {{
        width: 42%;
        height: 1fr;
        padding: 1 1 0 0;
        background: {BG};
        border-right: solid {LINE};
        scrollbar-size-vertical: 1;
        scrollbar-size-horizontal: 0;
        scrollbar-background: {BG};
        scrollbar-color: {LINE2};
    }}
    TraceScreen #trace-tree:focus {{ background-tint: {BG} 0%; }}
    TraceScreen #trace-tree > .tree--cursor {{ background: #1c2409; color: {INK}; }}
    TraceScreen #trace-tree:focus > .tree--cursor {{
        background: #1c2409;
        color: {INK};
        text-style: bold;
    }}
    TraceScreen #trace-tree > .tree--highlight-line {{ background: #121210; }}
    TraceScreen #trace-tree > .tree--guides {{ color: {LINE2}; }}
    TraceScreen #trace-tree > .tree--guides-selected {{ color: {DIM}; }}
    TraceScreen #trace-pane {{ width: 1fr; height: 1fr; padding: 1 0 0 2; }}
    TraceScreen #trace-path {{ height: 1; margin-bottom: 1; }}
    TraceScreen #trace-detail {{
        height: 1fr;
        border: none;
        padding: 0;
        background: {BG};
        scrollbar-size-vertical: 1;
        scrollbar-background: {BG};
        scrollbar-color: {LINE2};
    }}
    TraceScreen #trace-detail:focus {{ border: none; background-tint: {BG} 0%; }}
    TraceScreen #trace-log {{
        height: 1;
        padding: 0 3;
        background: {BG};
        border-top: solid {LINE};
        scrollbar-size-vertical: 0;
    }}
    TraceScreen #trace-keys {{ height: 2; padding: 0 3; border-top: solid {LINE}; }}
    TraceScreen #trace-keys Ink {{ width: auto; height: 1; margin-right: 2; }}
    """

    def __init__(
        self,
        root: TraceNode,
        history: Iterable[str],
        crumbs: str = "",
        stats: str = "",
    ) -> None:
        super().__init__()
        self._root = root
        self._history = list(history)
        self._crumbs = crumbs
        self._stats = stats
        self._log_rows = 0

    def compose(self) -> ComposeResult:
        with Horizontal(id="trace-top"):
            yield Ink(self._crumbs, id="trace-crumbs")
            yield Ink(self._stats, id="trace-stats")
        yield BlockWord(word="TRACE", id="trace-word")
        yield Ink(
            f"[{INK}]What happened in this run, story by story.[/] "
            f"[{MUTED}]Pick a line to read it whole.[/]",
            id="trace-sentence",
        )
        with Horizontal(id="trace-body"):
            tree = TraceTree(self._root.label, data=self._root.detail, id="trace-tree")
            tree.show_root = False
            tree.guide_depth = 3
            yield tree
            with Vertical(id="trace-pane"):
                yield Ink(id="trace-path")
                yield TextArea(
                    "", id="trace-detail", read_only=True, soft_wrap=True, language="markdown"
                )
        yield RichLog(id="trace-log", markup=True, wrap=False, highlight=False)
        with Horizontal(id="trace-keys"):
            for key, label, action in _KEYS:
                yield Ink(key_markup(key, label, action))

    def on_mount(self) -> None:
        self._load(self._root)
        for line in self._history[-LOG_ROWS:]:
            self.write(line)
        if not self._history:
            self.query_one("#trace-log", RichLog).display = False
        self.query_one("#trace-tree", Tree).focus()

    def _load(self, root: TraceNode) -> None:
        tree: Tree[str] = self.query_one("#trace-tree", Tree)
        tree.clear()
        tree.root.set_label(root.label)
        tree.root.data = root.detail
        _fill(tree.root, root)
        tree.root.expand()
        first = tree.root.children[0] if tree.root.children else None
        if first is not None:
            self._show(first)

    def write(self, line: str) -> None:
        """Append one formatted line to the live log while the screen is open."""
        log = self.query_one("#trace-log", RichLog)
        log.display = True
        log.write(line)
        self._log_rows = min(LOG_ROWS, self._log_rows + 1)
        log.styles.height = self._log_rows + 1

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted[str]) -> None:
        self._show(event.node)

    def on_tree_node_selected(self, event: Tree.NodeSelected[str]) -> None:
        self._show(event.node)

    def _show(self, node: TreeNode[str]) -> None:
        path: list[str] = []
        walk: TreeNode[str] | None = node
        while walk is not None and walk.parent is not None:
            path.append(_plain(walk.label).split(" · ")[0].strip())
            walk = walk.parent
        crumbs = f" [{DIM}]›[/] ".join(f"[{MUTED}]{part}[/]" for part in reversed(path))
        self.query_one("#trace-path", Ink).update(crumbs)
        text = node.data or "\n".join(_plain(child.label) for child in node.children)
        self.query_one("#trace-detail", TextArea).load_text(text or "nothing recorded yet")

    def action_open_node(self) -> None:
        tree: Tree[str] = self.query_one("#trace-tree", Tree)
        if tree.cursor_node is not None:
            tree.cursor_node.toggle()
            self._show(tree.cursor_node)

    def action_close(self) -> None:
        self.dismiss(None)

    def action_reload(self) -> None:
        build = getattr(self.app, "build_trace_root", None)
        if callable(build):
            self._root = build()
            self._load(self._root)

    def action_expand_all(self) -> None:
        self.query_one("#trace-tree", Tree).root.expand_all()

    def action_collapse_all(self) -> None:
        tree: Tree[str] = self.query_one("#trace-tree", Tree)
        tree.root.collapse_all()
        tree.root.expand()
