"""
Halo exchange pass: annotate ``stencil.ApplyOp`` for inter-GPU halo exchange.

Finds each ``stencil.ApplyOp`` that has been marked for multi-GPU execution
(``num_gpus > 1``) and attaches halo-exchange metadata so the lowering can
generate the appropriate NCCL / P2P communication calls.

Attached attributes (on each qualifying ``stencil.ApplyOp``):
    ``halo_exchange_enabled``  -- ``IntAttr(1)``.
    ``halo_exchange_axis``     -- ``IntAttr``: axis for halo exchange.
    ``halo_exchange_width``    -- ``IntAttr``: halo width on that axis.
    ``overlap_compute_comm``   -- ``IntAttr``: 1 if compute overlaps comm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from xdsl.context import Context
from xdsl.dialects.builtin import IntAttr, ModuleOp
from xdsl.dialects.stencil import ApplyOp
from xdsl.passes import ModulePass


@dataclass(frozen=True)
class HaloExchangePass(ModulePass):
    """Annotate decomposed stencil ops with halo-exchange metadata.

    Parameters
    ----------
    overlap : bool
        Whether to overlap interior computation with halo exchange
        communication.
    """

    name: ClassVar[str] = "halo-exchange"

    overlap: bool = True

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        for child in op.walk():
            if isinstance(child, ApplyOp):
                self._process_apply(child)

    # ------------------------------------------------------------------

    def _process_apply(self, apply_op: ApplyOp) -> None:
        # Only annotate ops that have been decomposed for multi-GPU
        num_gpus_attr = apply_op.attributes.get("num_gpus")
        if num_gpus_attr is None or num_gpus_attr.data <= 1:
            return

        # Determine the split axis (set by DecompositionPass)
        split_axis_attr = apply_op.attributes.get("split_axis")
        split_axis: int = split_axis_attr.data if split_axis_attr is not None else 0

        # Determine halo width on the split axis
        halo_attr = apply_op.attributes.get("halo_widths")
        if halo_attr is not None and split_axis < len(halo_attr):
            halo_width: int = halo_attr.data[split_axis].data
        else:
            # Fallback: use the stencil order if available
            order_attr = apply_op.attributes.get("order")
            halo_width = order_attr.data if order_attr is not None else 1

        # Attach halo exchange attributes
        apply_op.attributes["halo_exchange_enabled"] = IntAttr(1)
        apply_op.attributes["halo_exchange_axis"] = IntAttr(split_axis)
        apply_op.attributes["halo_exchange_width"] = IntAttr(halo_width)
        apply_op.attributes["overlap_compute_comm"] = IntAttr(
            1 if self.overlap else 0
        )
