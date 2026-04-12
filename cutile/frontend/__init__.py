"""cuTile Stencil DSL frontend: Python-to-IR conversion."""

from cutile.frontend.decorator import stencil
from cutile.frontend.parser import parse_stencil

__all__ = ["stencil", "parse_stencil"]
