"""
Fusion pass: detect fusable ``stencil.ApplyOp`` ops within the same function.

Walks each ``func.func``, groups ``stencil.ApplyOp`` ops that share input
operands, and marks them with a common ``fusion_group`` ID so the lowering
can emit a single fused kernel.

Attached attributes (on each fusable ``stencil.ApplyOp``):
    ``fusion_group``   -- ``IntAttr``: group ID (same ID = can fuse).
    ``fusion_outputs``  -- ``IntAttr``: total output fields of the group.

This is a metadata pass; actual fusion happens in the lowering/emitter.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import ClassVar

from xdsl.context import Context
from xdsl.dialects.builtin import IntAttr, ModuleOp
from xdsl.dialects.func import FuncOp
from xdsl.dialects.stencil import ApplyOp
from xdsl.passes import ModulePass


@dataclass(frozen=True)
class FusionPass(ModulePass):
    """Detect and annotate fusable ``stencil.ApplyOp`` ops.

    Two ``ApplyOp`` ops in the same function are considered fusable when
    they read from overlapping input fields (i.e. share at least one
    ``stencil.access`` operand at the ``ApplyOp`` level, or both belong
    to the same parent function body).

    For the initial implementation, all ``ApplyOp`` ops within the same
    ``func.func`` are placed in the same fusion group, since they operate
    on the same domain and are candidates for kernel fusion.
    """

    name: ClassVar[str] = "stencil-fusion"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        for child in op.walk():
            if isinstance(child, FuncOp):
                self._process_func(child)

    # ------------------------------------------------------------------

    @staticmethod
    def _process_func(func_op: FuncOp) -> None:
        # Collect all ApplyOps in this function
        apply_ops: list[ApplyOp] = []
        for inner_op in func_op.walk():
            if isinstance(inner_op, ApplyOp):
                apply_ops.append(inner_op)

        if len(apply_ops) <= 1:
            # Nothing to fuse with a single (or zero) apply op.
            # Still mark the sole op as group 0 so downstream can inspect.
            if len(apply_ops) == 1:
                apply_ops[0].attributes["fusion_group"] = IntAttr(0)
                apply_ops[0].attributes["fusion_outputs"] = IntAttr(
                    len(apply_ops[0].res)
                )
            return

        # --- Build fusion groups based on shared input operands ---
        # Map each ApplyOp index to a set of "input keys" (the SSA values
        # that feed into the stencil body).  Two ops that share at least
        # one input key are placed in the same group via union-find.

        # Collect input operand identities per ApplyOp
        op_inputs: list[set[int]] = []
        for aop in apply_ops:
            keys: set[int] = set()
            for operand in aop.operands:
                keys.add(id(operand))
            # If the body has block arguments mapped to parent func args,
            # also include those identities.
            op_inputs.append(keys)

        # Simple union-find
        parent: list[int] = list(range(len(apply_ops)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Build an inverted index: input_key -> list of ApplyOp indices
        key_to_ops: dict[int, list[int]] = defaultdict(list)
        for idx, keys in enumerate(op_inputs):
            for k in keys:
                key_to_ops[k].append(idx)

        # Union ops that share any input key
        for _key, ops_with_key in key_to_ops.items():
            for i in range(1, len(ops_with_key)):
                union(ops_with_key[0], ops_with_key[i])

        # If no shared operands were found (all ops are independent),
        # still group all ops in the same function together (they share
        # the same domain and can be fused into one kernel launch).
        roots = {find(i) for i in range(len(apply_ops))}
        if len(roots) > 1:
            # Merge all groups -- same function means same domain
            root_list = list(roots)
            for r in root_list[1:]:
                union(root_list[0], r)

        # Assign group IDs (compact numbering)
        root_to_group: dict[int, int] = {}
        next_group = 0
        for i in range(len(apply_ops)):
            r = find(i)
            if r not in root_to_group:
                root_to_group[r] = next_group
                next_group += 1

        # Count outputs per group
        group_outputs: dict[int, int] = defaultdict(int)
        for i, aop in enumerate(apply_ops):
            gid = root_to_group[find(i)]
            group_outputs[gid] += len(aop.res)

        # Attach attributes
        for i, aop in enumerate(apply_ops):
            gid = root_to_group[find(i)]
            aop.attributes["fusion_group"] = IntAttr(gid)
            aop.attributes["fusion_outputs"] = IntAttr(group_outputs[gid])
