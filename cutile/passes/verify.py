"""
Verification pass: validate IR well-formedness after optimization passes.

Checks cuTile-specific constraints on attributes attached by earlier passes
and raises ``ValueError`` with a diagnostic message if any check fails.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

from xdsl.context import Context
from xdsl.dialects.builtin import FileLineColLoc, ModuleOp
from xdsl.dialects.stencil import ApplyOp
from xdsl.passes import ModulePass


def _format_diagnostic(op, message: str) -> str:
    """Prefix *message* with source location info from *op* if available."""
    loc = getattr(op, "location", None)
    if isinstance(loc, FileLineColLoc):
        filename = loc.filename.data
        line = loc.line.data
        return f"{filename}:{line} -- {message}"
    return message


@dataclass(frozen=True)
class VerifyPass(ModulePass):
    """Validate IR well-formedness for cuTile lowering.

    Checks:
    1. ``tile_sizes`` values are powers of 2 (cuTile constraint).
    2. ``halo_widths`` exists when ``tile_sizes`` is set.
    3. ``temporal_steps`` >= 1 if present.
    4. ``num_gpus`` >= 1 if decomposition attributes are set.
    """

    name: ClassVar[str] = "verify-ir"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        for child in op.walk():
            if isinstance(child, ApplyOp):
                self._verify_apply(child)

    # ------------------------------------------------------------------

    @staticmethod
    def _is_power_of_2(n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0

    def _verify_apply(self, apply_op: ApplyOp) -> None:
        attrs = apply_op.attributes

        # --- tile_sizes must be powers of 2 ---
        tile_attr = attrs.get("tile_sizes")
        if tile_attr is not None:
            for idx, ts in enumerate(tile_attr):
                if not self._is_power_of_2(ts.data):
                    raise ValueError(
                        _format_diagnostic(
                            apply_op,
                            f"verify-ir: tile_sizes[{idx}] = {ts.data} "
                            f"is not a power of 2 (cuTile constraint)",
                        )
                    )

            # --- halo_widths must exist when tile_sizes is set ---
            halo_attr = attrs.get("halo_widths")
            if halo_attr is None:
                raise ValueError(
                    _format_diagnostic(
                        apply_op,
                        "verify-ir: tile_sizes is set but halo_widths is missing "
                        "(AnalysisPass must run before TilingPass)",
                    )
                )

        # --- temporal_steps >= 1 ---
        temporal_attr = attrs.get("temporal_steps")
        if temporal_attr is not None:
            if temporal_attr.data < 1:
                raise ValueError(
                    _format_diagnostic(
                        apply_op,
                        f"verify-ir: temporal_steps = {temporal_attr.data} "
                        f"must be >= 1",
                    )
                )

        # --- num_gpus >= 1 if decomposition is set ---
        num_gpus_attr = attrs.get("num_gpus")
        if num_gpus_attr is not None:
            if num_gpus_attr.data < 1:
                raise ValueError(
                    _format_diagnostic(
                        apply_op,
                        f"verify-ir: num_gpus = {num_gpus_attr.data} "
                        f"must be >= 1",
                    )
                )
