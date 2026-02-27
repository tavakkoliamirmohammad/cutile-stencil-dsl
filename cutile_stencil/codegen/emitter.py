"""CodeEmitter — indented code builder for generating Python source."""

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

    def __init__(self, indent_str: str = "    "):
        self._lines: list[str] = []
        self._indent_str = indent_str
        self._level = 0

    def line(self, text: str = "") -> None:
        if text:
            self._lines.append(self._indent_str * self._level + text)
        else:
            self._lines.append("")

    def blank(self) -> None:
        self._lines.append("")

    @contextmanager
    def indent(self) -> Iterator[None]:
        self._level += 1
        try:
            yield
        finally:
            self._level -= 1

    def render(self) -> str:
        return "\n".join(self._lines) + "\n"
