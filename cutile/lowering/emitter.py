"""CodeEmitter -- indentation-aware Python source code builder.

A lightweight helper used by the stencil-to-cuTile lowering pass to
assemble syntactically correct Python source strings.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


class CodeEmitter:
    """Builds Python source code with proper indentation.

    Usage::

        e = CodeEmitter()
        e.line("import cuda.tile as ct")
        e.blank()
        e.line("@ct.kernel")
        e.line("def my_kernel(x, y, TILE: ConstInt):")
        with e.indent():
            e.line("pid = ct.bid(0)")
        print(e.render())
    """

    def __init__(self, indent_str: str = "    ") -> None:
        self._lines: list[str] = []
        self._indent_str = indent_str
        self._level: int = 0

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #

    def line(self, text: str = "") -> None:
        """Append a line at the current indentation level."""
        if text:
            self._lines.append(self._indent_str * self._level + text)
        else:
            self._lines.append("")

    def blank(self) -> None:
        """Append a blank line."""
        self._lines.append("")

    @contextmanager
    def indent(self) -> Iterator[None]:
        """Context manager that increases indentation by one level."""
        self._level += 1
        try:
            yield
        finally:
            self._level -= 1

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def render(self) -> str:
        """Return the accumulated source as a single string."""
        return "\n".join(self._lines) + "\n"
