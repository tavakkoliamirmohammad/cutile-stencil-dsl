"""Comprehensive unit tests for all 5 cuTile xDSL dialects.

Tests cover op construction, attribute values, op names, custom printers,
boundary attributes, stencil spec attributes, and dialect registrations.
"""

from __future__ import annotations

import io

import pytest
from xdsl.dialects.builtin import (
    ArrayAttr,
    FloatAttr,
    IntAttr,
    NoneAttr,
    StringAttr,
    f32,
    f64,
)
from xdsl.ir import Block, Region
from xdsl.printer import Printer
from xdsl.utils.test_value import create_ssa_value

# ===================================================================
# Dialect 1: cutile_stencil
# ===================================================================
from cutile.dialects.cutile_stencil.dialect import (
    AccessOp,
    BoundaryAttr,
    ComputeOp,
    CutileStencilDialect,
    FuncOp,
    StencilSpecAttr,
    YieldOp,
)

# ===================================================================
# Dialect 2: cutile_target
# ===================================================================
from cutile.dialects.cutile_target.dialect import (
    AllocOp,
    AsyncOp,
    BidOp,
    BoundaryApplyOp,
    CutileTargetDialect,
    ForEachGpuOp,
    ForLoopOp,
    HostProgramOp,
    KernelOp,
    LayoutCastOp,
    LaunchOp,
    LoadOp,
    ReturnOp as TargetReturnOp,
    SliceOp,
    StoreOp,
    SwapBuffersOp,
    SyncOp,
)

# ===================================================================
# Dialect 3: comm
# ===================================================================
from cutile.dialects.comm.dialect import (
    BarrierOp,
    CommDialect,
    ExchangeHalosOp,
    RecvHaloOp,
    SendHaloOp,
)

# ===================================================================
# Dialect 4: timestep
# ===================================================================
from cutile.dialects.timestep.dialect import (
    CombineOp as TSCombineOp,
    ReturnOp as TSReturnOp,
    RKLoopOp,
    StageOp,
    TimestepDialect,
)

# ===================================================================
# Dialect 5: layout
# ===================================================================
from cutile.dialects.layout.dialect import (
    AddressOp,
    CastOp as LayoutCastDialectOp,
    LayoutAttr,
    LayoutDialect,
)


# ===================================================================
# Helper fixtures
# ===================================================================


@pytest.fixture
def f64_val():
    """Create an SSA value of type f64."""
    return create_ssa_value(f64)


@pytest.fixture
def f32_val():
    """Create an SSA value of type f32."""
    return create_ssa_value(f32)


@pytest.fixture
def empty_region():
    """Create an empty region with a single empty block."""
    return Region(Block())


# ===================================================================
# Tests: BoundaryAttr
# ===================================================================


class TestBoundaryAttr:
    """Tests for cutile_stencil.boundary attribute."""

    def test_dirichlet_with_value(self):
        attr = BoundaryAttr("dirichlet", 0.0)
        assert attr.bc_type_str == "dirichlet"
        assert attr.has_value is True
        assert isinstance(attr.value, FloatAttr)
        assert float(attr.value.value.data) == 0.0

    def test_neumann_without_value(self):
        attr = BoundaryAttr("neumann")
        assert attr.bc_type_str == "neumann"
        assert attr.has_value is False
        assert isinstance(attr.value, IntAttr)

    def test_periodic(self):
        attr = BoundaryAttr("periodic")
        assert attr.bc_type_str == "periodic"
        assert attr.has_value is False

    def test_reflecting(self):
        attr = BoundaryAttr("reflecting")
        assert attr.bc_type_str == "reflecting"
        assert attr.has_value is False

    def test_dirichlet_nonzero_value(self):
        attr = BoundaryAttr("dirichlet", 1.5)
        assert attr.has_value is True
        assert float(attr.value.value.data) == 1.5

    def test_dirichlet_with_float_attr(self):
        fa = FloatAttr(2.5, f64)
        attr = BoundaryAttr("dirichlet", fa)
        assert attr.has_value is True
        assert float(attr.value.value.data) == 2.5

    def test_string_attr_bc_type(self):
        attr = BoundaryAttr(StringAttr("periodic"))
        assert attr.bc_type_str == "periodic"

    def test_attr_name(self):
        attr = BoundaryAttr("dirichlet")
        assert attr.name == "cutile_stencil.boundary"

    def test_verify_valid_types(self):
        for bc in ("dirichlet", "neumann", "periodic", "reflecting"):
            attr = BoundaryAttr(bc)
            attr.verify()  # Should not raise

    def test_verify_invalid_type_raises(self):
        from xdsl.utils.exceptions import VerifyException

        # xDSL verifies at construction time, so the constructor itself raises
        with pytest.raises(VerifyException, match="Invalid boundary type"):
            BoundaryAttr("absorbing")

    def test_none_value_gives_sentinel(self):
        attr = BoundaryAttr("neumann", None)
        assert attr.has_value is False

    def test_int_value_converted_to_float(self):
        attr = BoundaryAttr("dirichlet", 3)
        assert attr.has_value is True
        assert float(attr.value.value.data) == 3.0


# ===================================================================
# Tests: StencilSpecAttr
# ===================================================================


class TestStencilSpecAttr:
    """Tests for cutile_stencil.spec attribute."""

    def test_basic_construction(self):
        spec = StencilSpecAttr(ndim=2, order=2)
        assert spec.name == "cutile_stencil.spec"
        assert spec.ndim.data == 2
        assert spec.order.data == 2
        assert spec.dtype.data == "float64"
        assert spec.has_boundary is False

    def test_construction_with_int_attr(self):
        spec = StencilSpecAttr(ndim=IntAttr(3), order=IntAttr(4))
        assert spec.ndim.data == 3
        assert spec.order.data == 4

    def test_float32_dtype(self):
        spec = StencilSpecAttr(ndim=1, order=1, dtype="float32")
        assert spec.dtype.data == "float32"

    def test_string_attr_dtype(self):
        spec = StencilSpecAttr(ndim=2, order=2, dtype=StringAttr("float16"))
        assert spec.dtype.data == "float16"

    def test_with_boundary(self):
        bc = BoundaryAttr("dirichlet", 0.0)
        spec = StencilSpecAttr(ndim=2, order=2, boundary=bc)
        assert spec.has_boundary is True
        assert isinstance(spec.boundary, BoundaryAttr)
        assert spec.boundary.bc_type_str == "dirichlet"

    def test_with_array_boundary(self):
        bc_x = BoundaryAttr("periodic")
        bc_y = BoundaryAttr("dirichlet", 0.0)
        arr = ArrayAttr([bc_x, bc_y])
        spec = StencilSpecAttr(ndim=2, order=2, boundary=arr)
        assert spec.has_boundary is True
        assert isinstance(spec.boundary, ArrayAttr)

    def test_without_boundary(self):
        spec = StencilSpecAttr(ndim=2, order=2, boundary=None)
        assert spec.has_boundary is False
        assert isinstance(spec.boundary, IntAttr)

    def test_1d_spec(self):
        spec = StencilSpecAttr(ndim=1, order=6)
        assert spec.ndim.data == 1
        assert spec.order.data == 6

    def test_3d_spec(self):
        spec = StencilSpecAttr(ndim=3, order=2)
        assert spec.ndim.data == 3


# ===================================================================
# Tests: FuncOp
# ===================================================================


class TestFuncOp:
    """Tests for cutile_stencil.func operation."""

    def test_basic_construction(self, empty_region):
        op = FuncOp("heat_2d", ndim=2, order=2, body=empty_region)
        assert op.name == "cutile_stencil.func"
        assert op.func_name.data == "heat_2d"
        assert op.ndim.data == 2
        assert op.order.data == 2

    def test_with_dtype(self, empty_region):
        op = FuncOp("stencil", ndim=2, order=4, body=empty_region, dtype="float32")
        assert op.dtype.data == "float32"

    def test_with_boundary(self, empty_region):
        bc = BoundaryAttr("periodic")
        op = FuncOp("wave", ndim=3, order=2, body=empty_region, boundary=bc)
        assert op.boundary.bc_type_str == "periodic"

    def test_with_block_body(self):
        block = Block()
        op = FuncOp("f", ndim=1, order=1, body=block)
        assert op.name == "cutile_stencil.func"
        assert len(op.body.blocks) == 1

    def test_with_string_attr_name(self, empty_region):
        op = FuncOp(StringAttr("my_stencil"), ndim=2, order=2, body=empty_region)
        assert op.func_name.data == "my_stencil"

    def test_with_int_attr_ndim_order(self, empty_region):
        op = FuncOp("s", ndim=IntAttr(3), order=IntAttr(6), body=empty_region)
        assert op.ndim.data == 3
        assert op.order.data == 6

    def test_with_constants(self, empty_region):
        from xdsl.dialects.builtin import DictionaryAttr

        consts = DictionaryAttr({"alpha": FloatAttr(0.5, f64)})
        op = FuncOp("heat", ndim=2, order=2, body=empty_region, constants=consts)
        assert "alpha" in op.constants.data

    def test_no_dtype_is_none(self, empty_region):
        op = FuncOp("s", ndim=1, order=1, body=empty_region)
        assert op.dtype is None

    def test_no_boundary_is_none(self, empty_region):
        op = FuncOp("s", ndim=1, order=1, body=empty_region)
        assert op.boundary is None

    def test_with_input_output_types(self, empty_region, f64_val):
        op = FuncOp(
            "s",
            ndim=2,
            order=2,
            body=empty_region,
            input_types=[f64],
            output_types=[f64],
            operands=[f64_val],
        )
        assert len(op.inputs) == 1
        assert len(op.outputs) == 1


# ===================================================================
# Tests: AccessOp
# ===================================================================


class TestAccessOp:
    """Tests for cutile_stencil.access operation."""

    def test_basic_construction(self, f64_val):
        op = AccessOp(f64_val, offset=[-1, 0], result_type=f64)
        assert op.name == "cutile_stencil.access"

    def test_offset_values(self, f64_val):
        op = AccessOp(f64_val, offset=[-1, 0], result_type=f64)
        offsets = [int(idx.data) for idx in op.offset.parameters[0].data]
        assert offsets == [-1, 0]

    def test_center_access(self, f64_val):
        op = AccessOp(f64_val, offset=[0, 0], result_type=f64)
        offsets = [int(idx.data) for idx in op.offset.parameters[0].data]
        assert offsets == [0, 0]

    def test_3d_offset(self, f64_val):
        op = AccessOp(f64_val, offset=[1, -1, 0], result_type=f64)
        offsets = [int(idx.data) for idx in op.offset.parameters[0].data]
        assert offsets == [1, -1, 0]

    def test_1d_offset(self, f64_val):
        op = AccessOp(f64_val, offset=[2], result_type=f64)
        offsets = [int(idx.data) for idx in op.offset.parameters[0].data]
        assert offsets == [2]

    def test_with_index_names(self, f64_val):
        op = AccessOp(f64_val, offset=[-1, 0], result_type=f64, index_names=["i", "j"])
        assert op.index_names is not None
        names = [a.data for a in op.index_names]
        assert names == ["i", "j"]

    def test_without_index_names(self, f64_val):
        op = AccessOp(f64_val, offset=[0, 0], result_type=f64)
        assert op.index_names is None

    def test_with_array_attr_index_names(self, f64_val):
        arr = ArrayAttr([StringAttr("x"), StringAttr("y")])
        op = AccessOp(f64_val, offset=[0, 0], result_type=f64, index_names=arr)
        names = [a.data for a in op.index_names]
        assert names == ["x", "y"]

    def test_f32_result_type(self, f32_val):
        op = AccessOp(f32_val, offset=[0, 0], result_type=f32)
        assert op.res.type == f32

    def test_f64_result_type(self, f64_val):
        op = AccessOp(f64_val, offset=[0, 0], result_type=f64)
        assert op.res.type == f64

    def test_custom_printer_no_index_names(self, f64_val):
        op = AccessOp(f64_val, offset=[-1, 0], result_type=f64)
        buf = io.StringIO()
        printer = Printer(stream=buf)
        op.print(printer)
        text = buf.getvalue()
        assert "[-1, 0]" in text
        assert "{" not in text  # no index names section

    def test_custom_printer_with_index_names(self, f64_val):
        op = AccessOp(f64_val, offset=[-1, 0], result_type=f64, index_names=["i", "j"])
        buf = io.StringIO()
        printer = Printer(stream=buf)
        op.print(printer)
        text = buf.getvalue()
        assert "[-1, 0]" in text
        assert '"i"' in text
        assert '"j"' in text

    def test_custom_printer_3d(self, f64_val):
        op = AccessOp(f64_val, offset=[1, -1, 2], result_type=f64)
        buf = io.StringIO()
        printer = Printer(stream=buf)
        op.print(printer)
        text = buf.getvalue()
        assert "[1, -1, 2]" in text

    def test_index_attr_passthrough(self, f64_val):
        from xdsl.dialects.stencil import IndexAttr

        idx = IndexAttr.from_indices(-2, 3)
        op = AccessOp(f64_val, offset=idx, result_type=f64)
        offsets = [int(i.data) for i in op.offset.parameters[0].data]
        assert offsets == [-2, 3]


# ===================================================================
# Tests: ComputeOp
# ===================================================================


class TestComputeOp:
    """Tests for cutile_stencil.compute operation."""

    def test_basic_construction(self, empty_region):
        op = ComputeOp(empty_region, result_type=f64)
        assert op.name == "cutile_stencil.compute"
        assert op.res.type == f64

    def test_with_block(self):
        block = Block()
        op = ComputeOp(block, result_type=f32)
        assert op.name == "cutile_stencil.compute"
        assert op.res.type == f32
        assert len(op.body.blocks) == 1

    def test_result_type_f64(self, empty_region):
        op = ComputeOp(empty_region, result_type=f64)
        assert op.res.type == f64

    def test_result_type_f32(self, empty_region):
        op = ComputeOp(empty_region, result_type=f32)
        assert op.res.type == f32


# ===================================================================
# Tests: YieldOp
# ===================================================================


class TestYieldOp:
    """Tests for cutile_stencil.yield operation."""

    def test_basic_construction(self, f64_val):
        op = YieldOp(f64_val)
        assert op.name == "cutile_stencil.yield"

    def test_is_terminator(self):
        from xdsl.traits import IsTerminator

        assert any(isinstance(t, IsTerminator) for t in YieldOp.traits)

    def test_value_operand(self, f64_val):
        op = YieldOp(f64_val)
        assert op.value == f64_val


# ===================================================================
# Tests: CutileStencilDialect registration
# ===================================================================


class TestCutileStencilDialect:
    """Tests for dialect registration."""

    def test_dialect_name(self):
        assert CutileStencilDialect.name == "cutile_stencil"

    def test_dialect_operations(self):
        op_types = CutileStencilDialect.operations
        op_names = {op.name for op in op_types}
        assert "cutile_stencil.func" in op_names
        assert "cutile_stencil.access" in op_names
        assert "cutile_stencil.compute" in op_names
        assert "cutile_stencil.yield" in op_names

    def test_dialect_attributes(self):
        attr_types = CutileStencilDialect.attributes
        attr_names = {a.name for a in attr_types}
        assert "cutile_stencil.boundary" in attr_names
        assert "cutile_stencil.spec" in attr_names


# ===================================================================
# Tests: cutile_target dialect — Device ops
# ===================================================================


class TestKernelOp:
    """Tests for cutile.kernel operation."""

    def test_basic_construction(self, empty_region, f64_val):
        op = KernelOp(
            operands=[
                [f64_val],
            ],
            regions=[empty_region],
            properties={
                "kernel_name": StringAttr("heat_kernel"),
                "tile_shape": ArrayAttr([IntAttr(128), IntAttr(128)]),
                "halo": ArrayAttr([IntAttr(1), IntAttr(1)]),
            },
        )
        assert op.name == "cutile.kernel"
        assert op.kernel_name.data == "heat_kernel"

    def test_tile_shape(self, empty_region, f64_val):
        op = KernelOp(
            operands=[[f64_val]],
            regions=[empty_region],
            properties={
                "kernel_name": StringAttr("k"),
                "tile_shape": ArrayAttr([IntAttr(64), IntAttr(64)]),
                "halo": ArrayAttr([IntAttr(2)]),
            },
        )
        tile = [int(x.data) for x in op.tile_shape.data]
        assert tile == [64, 64]


class TestSliceOp:
    """Tests for cutile.slice operation."""

    def test_basic_construction(self, f64_val):
        op = SliceOp(
            operands=[f64_val],
            result_types=[f64],
            properties={
                "axis": IntAttr(0),
                "start": StringAttr("bid*T"),
                "stop": StringAttr("bid*T+T+2*H"),
            },
        )
        assert op.name == "cutile.slice"
        assert op.axis.data == 0
        assert op.start.data == "bid*T"
        assert op.stop.data == "bid*T+T+2*H"


class TestLoadOp:
    """Tests for cutile.load operation."""

    def test_basic_construction(self, f64_val):
        op = LoadOp(operands=[f64_val], result_types=[f64])
        assert op.name == "cutile.load"

    def test_result_type(self, f64_val):
        op = LoadOp(operands=[f64_val], result_types=[f64])
        assert op.result.type == f64


class TestStoreOp:
    """Tests for cutile.store operation."""

    def test_basic_construction(self, f64_val):
        val = create_ssa_value(f64)
        op = StoreOp(operands=[f64_val, val])
        assert op.name == "cutile.store"


class TestBidOp:
    """Tests for cutile.bid operation."""

    def test_axis_0(self):
        op = BidOp(result_types=[f64], properties={"axis": IntAttr(0)})
        assert op.name == "cutile.bid"
        assert op.axis.data == 0

    def test_axis_1(self):
        op = BidOp(result_types=[f64], properties={"axis": IntAttr(1)})
        assert op.axis.data == 1

    def test_axis_2(self):
        op = BidOp(result_types=[f64], properties={"axis": IntAttr(2)})
        assert op.axis.data == 2


class TestTargetReturnOp:
    """Tests for cutile.return operation."""

    def test_basic_construction(self):
        op = TargetReturnOp()
        assert op.name == "cutile.return"

    def test_is_terminator(self):
        from xdsl.traits import IsTerminator

        assert any(isinstance(t, IsTerminator) for t in TargetReturnOp.traits)


# ===================================================================
# Tests: cutile_target dialect — Host ops
# ===================================================================


class TestHostProgramOp:
    """Tests for cutile.host_program operation."""

    def test_basic_construction(self, empty_region):
        op = HostProgramOp(
            regions=[empty_region],
            properties={"program_name": StringAttr("main_launcher")},
        )
        assert op.name == "cutile.host_program"
        assert op.program_name.data == "main_launcher"


class TestAllocOp:
    """Tests for cutile.alloc operation."""

    def test_basic_construction(self):
        op = AllocOp(
            result_types=[f64],
            properties={
                "var_name": StringAttr("buf_u"),
                "dtype": StringAttr("float64"),
            },
        )
        assert op.name == "cutile.alloc"
        assert op.var_name.data == "buf_u"
        assert op.dtype.data == "float64"

    def test_float32_dtype(self):
        op = AllocOp(
            result_types=[f32],
            properties={
                "var_name": StringAttr("buf"),
                "dtype": StringAttr("float32"),
            },
        )
        assert op.dtype.data == "float32"


class TestLaunchOp:
    """Tests for cutile.launch operation."""

    def test_basic_construction(self, f64_val):
        op = LaunchOp(
            operands=[[f64_val]],
            properties={
                "kernel_name": StringAttr("heat_kernel"),
                "grid_expr": StringAttr("(N // T, N // T)"),
            },
        )
        assert op.name == "cutile.launch"
        assert op.kernel_name.data == "heat_kernel"
        assert op.grid_expr.data == "(N // T, N // T)"

    def test_no_args(self):
        op = LaunchOp(
            operands=[[]],
            properties={
                "kernel_name": StringAttr("simple_kernel"),
                "grid_expr": StringAttr("(1,)"),
            },
        )
        assert len(op.args) == 0


class TestSyncOp:
    """Tests for cutile.sync operation."""

    def test_basic_construction(self):
        op = SyncOp()
        assert op.name == "cutile.sync"


class TestSwapBuffersOp:
    """Tests for cutile.swap_buffers operation."""

    def test_basic_construction(self, f64_val):
        val_b = create_ssa_value(f64)
        op = SwapBuffersOp(operands=[f64_val, val_b])
        assert op.name == "cutile.swap_buffers"


class TestForLoopOp:
    """Tests for cutile.for_loop operation."""

    def test_basic_construction(self, empty_region):
        op = ForLoopOp(
            regions=[empty_region],
            properties={"count": IntAttr(100)},
        )
        assert op.name == "cutile.for_loop"
        assert op.count.data == 100


class TestForEachGpuOp:
    """Tests for cutile.for_each_gpu operation."""

    def test_basic_construction(self, empty_region):
        op = ForEachGpuOp(
            regions=[empty_region],
            properties={"num_gpus": IntAttr(4)},
        )
        assert op.name == "cutile.for_each_gpu"
        assert op.num_gpus.data == 4


class TestAsyncOp:
    """Tests for cutile.async operation."""

    def test_basic_construction(self, empty_region):
        op = AsyncOp(regions=[empty_region])
        assert op.name == "cutile.async"


class TestBoundaryApplyOp:
    """Tests for cutile.boundary_apply operation."""

    def test_basic_construction(self, f64_val):
        op = BoundaryApplyOp(
            operands=[f64_val],
            properties={"bc_type": StringAttr("dirichlet")},
        )
        assert op.name == "cutile.boundary_apply"
        assert op.bc_type.data == "dirichlet"

    def test_periodic_bc(self, f64_val):
        op = BoundaryApplyOp(
            operands=[f64_val],
            properties={"bc_type": StringAttr("periodic")},
        )
        assert op.bc_type.data == "periodic"


class TestLayoutCastOp:
    """Tests for cutile.layout_cast operation."""

    def test_basic_construction(self, f64_val):
        op = LayoutCastOp(
            operands=[f64_val],
            result_types=[f64],
            properties={
                "from_layout": StringAttr("row_major"),
                "to_layout": StringAttr("bricked"),
            },
        )
        assert op.name == "cutile.layout_cast"
        assert op.from_layout.data == "row_major"
        assert op.to_layout.data == "bricked"


# ===================================================================
# Tests: CutileTargetDialect registration
# ===================================================================


class TestCutileTargetDialect:
    """Tests for cutile_target dialect registration."""

    def test_dialect_name(self):
        assert CutileTargetDialect.name == "cutile"

    def test_dialect_has_all_ops(self):
        op_types = CutileTargetDialect.operations
        op_names = {op.name for op in op_types}
        expected = {
            "cutile.kernel",
            "cutile.slice",
            "cutile.load",
            "cutile.store",
            "cutile.bid",
            "cutile.return",
            "cutile.host_program",
            "cutile.alloc",
            "cutile.launch",
            "cutile.sync",
            "cutile.swap_buffers",
            "cutile.for_loop",
            "cutile.for_each_gpu",
            "cutile.async",
            "cutile.boundary_apply",
            "cutile.layout_cast",
        }
        assert expected == op_names

    def test_dialect_has_no_attributes(self):
        assert len(list(CutileTargetDialect.attributes)) == 0


# ===================================================================
# Tests: comm dialect
# ===================================================================


class TestSendHaloOp:
    """Tests for comm.send_halo operation."""

    def test_basic_construction(self, f64_val):
        op = SendHaloOp(
            operands=[f64_val],
            properties={
                "dst_gpu": IntAttr(1),
                "axis": IntAttr(0),
                "side": StringAttr("left"),
                "halo_width": IntAttr(2),
            },
        )
        assert op.name == "comm.send_halo"
        assert op.dst_gpu.data == 1
        assert op.axis.data == 0
        assert op.side.data == "left"
        assert op.halo_width.data == 2

    def test_right_side(self, f64_val):
        op = SendHaloOp(
            operands=[f64_val],
            properties={
                "dst_gpu": IntAttr(3),
                "axis": IntAttr(1),
                "side": StringAttr("right"),
                "halo_width": IntAttr(1),
            },
        )
        assert op.side.data == "right"
        assert op.dst_gpu.data == 3
        assert op.axis.data == 1


class TestRecvHaloOp:
    """Tests for comm.recv_halo operation."""

    def test_basic_construction(self, f64_val):
        op = RecvHaloOp(
            operands=[f64_val],
            properties={
                "src_gpu": IntAttr(0),
                "axis": IntAttr(0),
                "side": StringAttr("left"),
                "halo_width": IntAttr(2),
            },
        )
        assert op.name == "comm.recv_halo"
        assert op.src_gpu.data == 0
        assert op.axis.data == 0
        assert op.side.data == "left"
        assert op.halo_width.data == 2

    def test_right_side_axis1(self, f64_val):
        op = RecvHaloOp(
            operands=[f64_val],
            properties={
                "src_gpu": IntAttr(2),
                "axis": IntAttr(1),
                "side": StringAttr("right"),
                "halo_width": IntAttr(3),
            },
        )
        assert op.side.data == "right"
        assert op.axis.data == 1
        assert op.halo_width.data == 3


class TestExchangeHalosOp:
    """Tests for comm.exchange_halos operation."""

    def test_basic_construction(self, f64_val):
        val2 = create_ssa_value(f64)
        op = ExchangeHalosOp(
            operands=[[f64_val, val2]],
            properties={
                "halo_width": IntAttr(1),
                "axis": IntAttr(0),
            },
        )
        assert op.name == "comm.exchange_halos"
        assert op.halo_width.data == 1
        assert op.axis.data == 0
        assert len(op.partition_buffers) == 2

    def test_four_partitions(self):
        vals = [create_ssa_value(f64) for _ in range(4)]
        op = ExchangeHalosOp(
            operands=[vals],
            properties={
                "halo_width": IntAttr(2),
                "axis": IntAttr(1),
            },
        )
        assert len(op.partition_buffers) == 4


class TestBarrierOp:
    """Tests for comm.barrier operation."""

    def test_basic_construction(self):
        op = BarrierOp()
        assert op.name == "comm.barrier"


class TestCommDialect:
    """Tests for comm dialect registration."""

    def test_dialect_name(self):
        assert CommDialect.name == "comm"

    def test_dialect_has_all_ops(self):
        op_types = CommDialect.operations
        op_names = {op.name for op in op_types}
        expected = {
            "comm.send_halo",
            "comm.recv_halo",
            "comm.exchange_halos",
            "comm.barrier",
        }
        assert expected == op_names

    def test_dialect_has_no_attributes(self):
        assert len(list(CommDialect.attributes)) == 0


# ===================================================================
# Tests: timestep dialect
# ===================================================================


class TestStageOp:
    """Tests for timestep.stage operation."""

    def test_basic_construction(self, f64_val):
        deriv = create_ssa_value(f64)
        op = StageOp(
            operands=[f64_val, deriv],
            result_types=[f64],
            properties={
                "coeff": FloatAttr(0.5, f64),
                "dt_name": StringAttr("dt"),
            },
        )
        assert op.name == "timestep.stage"
        assert float(op.coeff.value.data) == 0.5
        assert op.dt_name.data == "dt"

    def test_rk4_coefficients(self, f64_val):
        """Test with RK4-style coefficient."""
        deriv = create_ssa_value(f64)
        for coeff in [0.0, 0.5, 0.5, 1.0]:
            op = StageOp(
                operands=[f64_val, deriv],
                result_types=[f64],
                properties={
                    "coeff": FloatAttr(coeff, f64),
                    "dt_name": StringAttr("dt"),
                },
            )
            assert float(op.coeff.value.data) == coeff


class TestTSCombineOp:
    """Tests for timestep.combine operation."""

    def test_basic_construction(self, f64_val):
        k1 = create_ssa_value(f64)
        k2 = create_ssa_value(f64)
        weights = ArrayAttr([FloatAttr(0.5, f64), FloatAttr(0.5, f64)])
        op = TSCombineOp(
            operands=[f64_val, [k1, k2]],
            result_types=[f64],
            properties={
                "weights": weights,
                "dt_name": StringAttr("dt"),
            },
        )
        assert op.name == "timestep.combine"
        assert len(op.stage_derivatives) == 2
        assert op.dt_name.data == "dt"

    def test_rk4_weights(self, f64_val):
        """Test with standard RK4 weights (1/6, 1/3, 1/3, 1/6)."""
        ks = [create_ssa_value(f64) for _ in range(4)]
        w = [1 / 6, 1 / 3, 1 / 3, 1 / 6]
        weights = ArrayAttr([FloatAttr(v, f64) for v in w])
        op = TSCombineOp(
            operands=[f64_val, ks],
            result_types=[f64],
            properties={
                "weights": weights,
                "dt_name": StringAttr("dt"),
            },
        )
        assert len(op.stage_derivatives) == 4
        result_weights = [float(x.value.data) for x in op.weights.data]
        assert len(result_weights) == 4
        assert abs(sum(result_weights) - 1.0) < 1e-10


class TestRKLoopOp:
    """Tests for timestep.rk_loop operation."""

    def test_basic_construction(self, empty_region, f64_val):
        op = RKLoopOp(
            operands=[[f64_val]],
            regions=[empty_region],
            properties={
                "method": StringAttr("RK4"),
                "num_stages": IntAttr(4),
            },
        )
        assert op.name == "timestep.rk_loop"
        assert op.method.data == "RK4"
        assert op.num_stages.data == 4

    def test_euler_method(self, empty_region, f64_val):
        op = RKLoopOp(
            operands=[[f64_val]],
            regions=[empty_region],
            properties={
                "method": StringAttr("Euler"),
                "num_stages": IntAttr(1),
            },
        )
        assert op.method.data == "Euler"
        assert op.num_stages.data == 1

    def test_rk2_method(self, empty_region, f64_val):
        op = RKLoopOp(
            operands=[[f64_val]],
            regions=[empty_region],
            properties={
                "method": StringAttr("RK2"),
                "num_stages": IntAttr(2),
            },
        )
        assert op.method.data == "RK2"
        assert op.num_stages.data == 2

    def test_multiple_inputs(self, empty_region):
        vals = [create_ssa_value(f64) for _ in range(3)]
        op = RKLoopOp(
            operands=[vals],
            regions=[empty_region],
            properties={
                "method": StringAttr("RK4"),
                "num_stages": IntAttr(4),
            },
        )
        assert len(op.inputs) == 3


class TestTSReturnOp:
    """Tests for timestep.return operation."""

    def test_basic_construction(self, f64_val):
        op = TSReturnOp(operands=[[f64_val]])
        assert op.name == "timestep.return"

    def test_is_terminator(self):
        from xdsl.traits import IsTerminator

        assert any(isinstance(t, IsTerminator) for t in TSReturnOp.traits)

    def test_no_values(self):
        op = TSReturnOp(operands=[[]])
        assert len(op.values) == 0

    def test_multiple_values(self):
        vals = [create_ssa_value(f64) for _ in range(3)]
        op = TSReturnOp(operands=[vals])
        assert len(op.values) == 3


class TestTimestepDialect:
    """Tests for timestep dialect registration."""

    def test_dialect_name(self):
        assert TimestepDialect.name == "timestep"

    def test_dialect_has_all_ops(self):
        op_types = TimestepDialect.operations
        op_names = {op.name for op in op_types}
        expected = {
            "timestep.stage",
            "timestep.combine",
            "timestep.rk_loop",
            "timestep.return",
        }
        assert expected == op_names

    def test_dialect_has_no_attributes(self):
        assert len(list(TimestepDialect.attributes)) == 0


# ===================================================================
# Tests: layout dialect
# ===================================================================


class TestLayoutAttr:
    """Tests for layout.type attribute."""

    def test_row_major(self):
        attr = LayoutAttr(StringAttr("row_major"), NoneAttr())
        assert attr.kind.data == "row_major"
        assert isinstance(attr.brick_size, NoneAttr)

    def test_bricked(self):
        attr = LayoutAttr(StringAttr("bricked"), IntAttr(8))
        assert attr.kind.data == "bricked"
        assert attr.brick_size.data == 8

    def test_attr_name(self):
        attr = LayoutAttr(StringAttr("row_major"), NoneAttr())
        assert attr.name == "layout.type"

    def test_bricked_various_sizes(self):
        for size in [4, 8, 16, 32]:
            attr = LayoutAttr(StringAttr("bricked"), IntAttr(size))
            assert attr.brick_size.data == size


class TestLayoutCastDialectOp:
    """Tests for layout.cast operation."""

    def test_basic_construction(self, f64_val):
        from_layout = LayoutAttr(StringAttr("row_major"), NoneAttr())
        to_layout = LayoutAttr(StringAttr("bricked"), IntAttr(8))
        op = LayoutCastDialectOp(
            operands=[f64_val],
            result_types=[f64],
            properties={
                "from_layout": from_layout,
                "to_layout": to_layout,
            },
        )
        assert op.name == "layout.cast"
        assert op.from_layout.kind.data == "row_major"
        assert op.to_layout.kind.data == "bricked"

    def test_bricked_to_row_major(self, f64_val):
        from_layout = LayoutAttr(StringAttr("bricked"), IntAttr(16))
        to_layout = LayoutAttr(StringAttr("row_major"), NoneAttr())
        op = LayoutCastDialectOp(
            operands=[f64_val],
            result_types=[f64],
            properties={
                "from_layout": from_layout,
                "to_layout": to_layout,
            },
        )
        assert op.from_layout.brick_size.data == 16
        assert isinstance(op.to_layout.brick_size, NoneAttr)


class TestAddressOp:
    """Tests for layout.address operation."""

    def test_basic_construction(self, f64_val):
        layout = LayoutAttr(StringAttr("row_major"), NoneAttr())
        idx_val = create_ssa_value(f64)
        op = AddressOp(
            operands=[[f64_val, idx_val]],
            result_types=[f64],
            properties={"layout": layout},
        )
        assert op.name == "layout.address"
        assert op.layout.kind.data == "row_major"
        assert len(op.indices) == 2

    def test_bricked_layout_3d(self):
        layout = LayoutAttr(StringAttr("bricked"), IntAttr(8))
        idx_vals = [create_ssa_value(f64) for _ in range(3)]
        op = AddressOp(
            operands=[idx_vals],
            result_types=[f64],
            properties={"layout": layout},
        )
        assert op.layout.brick_size.data == 8
        assert len(op.indices) == 3


class TestLayoutDialect:
    """Tests for layout dialect registration."""

    def test_dialect_name(self):
        assert LayoutDialect.name == "layout"

    def test_dialect_has_all_ops(self):
        op_types = LayoutDialect.operations
        op_names = {op.name for op in op_types}
        expected = {"layout.cast", "layout.address"}
        assert expected == op_names

    def test_dialect_has_layout_attr(self):
        attr_types = LayoutDialect.attributes
        attr_names = {a.name for a in attr_types}
        assert "layout.type" in attr_names
