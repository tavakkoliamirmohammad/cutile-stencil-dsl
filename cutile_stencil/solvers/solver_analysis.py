"""Solver-level analysis: per-kernel profiling and CG iteration analysis.

Extends the stencil pipeline's roofline analysis to full CG iterations,
profiling each kernel (dot, axpy, operator) and identifying the
bottleneck across the entire solve step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from cutile_stencil.dsl.types import HardwareSpec, StencilSpec


_DTYPE_BYTES = {"float64": 8, "float32": 4, "float16": 2}


@dataclass
class SolverKernelProfile:
    """Performance profile for a single solver kernel."""
    name: str
    flops_per_element: int
    loads_per_element: int
    stores_per_element: int
    bytes_per_element: int
    arithmetic_intensity: float
    bound: str  # "memory" or "compute"
    peak_gpoints_s: float
    recommended_tile: int = 256


@dataclass
class SolverAnalysisResult:
    """Full CG iteration analysis."""
    kernel_profiles: List[SolverKernelProfile]
    total_flops_per_element: int
    total_bytes_per_element: int
    overall_arithmetic_intensity: float
    bottleneck: str  # name of the slowest kernel
    bottleneck_bound: str  # "memory" or "compute"


def _dtype_bytes(dtype: str) -> int:
    return _DTYPE_BYTES.get(dtype, 8)


def _profile_kernel(
    name: str,
    flops: int,
    loads: int,
    stores: int,
    hw: HardwareSpec,
    dtype: str = "float64",
) -> SolverKernelProfile:
    """Compute roofline profile for one kernel."""
    db = _dtype_bytes(dtype)
    bytes_per_elem = (loads + stores) * db
    ai = flops / bytes_per_elem if bytes_per_elem > 0 else 0.0

    mem_ceiling = hw.peak_bandwidth_gbs * ai
    bound = "memory" if mem_ceiling < hw.peak_gflops else "compute"
    effective = min(mem_ceiling, hw.peak_gflops)
    peak_gpts = effective / max(flops, 1)

    return SolverKernelProfile(
        name=name,
        flops_per_element=flops,
        loads_per_element=loads,
        stores_per_element=stores,
        bytes_per_element=bytes_per_elem,
        arithmetic_intensity=ai,
        bound=bound,
        peak_gpoints_s=peak_gpts,
    )


def analyze_dot(
    N: int,
    hw: HardwareSpec | None = None,
    dtype: str = "float64",
) -> SolverKernelProfile:
    """Analyze the dot product kernel: result = sum(a * b).

    2 loads (a, b), 0 stores (atomic scalar), 2 FLOPs/element (mul + add).
    """
    hw = hw or HardwareSpec()
    return _profile_kernel("dot", flops=2, loads=2, stores=0, hw=hw, dtype=dtype)


def analyze_axpy(
    N: int,
    hw: HardwareSpec | None = None,
    dtype: str = "float64",
) -> SolverKernelProfile:
    """Analyze the axpy kernel: y = alpha * x + y.

    2 loads (x, y), 1 store (y), 2 FLOPs/element (mul + add).
    """
    hw = hw or HardwareSpec()
    return _profile_kernel("axpy", flops=2, loads=2, stores=1, hw=hw, dtype=dtype)


def analyze_dia_spmv(
    N: int,
    num_diags: int,
    hw: HardwareSpec | None = None,
    dtype: str = "float64",
) -> SolverKernelProfile:
    """Analyze the DIA SpMV kernel.

    Per element: num_diags loads for diag_vals, num_diags gathers for x,
    1 store for y.  FLOPs: num_diags muls + (num_diags - 1) adds.
    """
    hw = hw or HardwareSpec()
    flops = 2 * num_diags - 1
    loads = 2 * num_diags  # diag_vals + x gathers
    stores = 1
    return _profile_kernel("dia_spmv", flops=flops, loads=loads, stores=stores, hw=hw, dtype=dtype)


def analyze_stencil_operator(
    spec: StencilSpec,
    domain: Tuple[int, ...],
    hw: HardwareSpec | None = None,
) -> SolverKernelProfile:
    """Analyze a stencil operator by delegating to the pipeline's roofline.

    Returns a SolverKernelProfile wrapping the pipeline's RooflineResult.
    """
    from cutile_stencil.dsl.pipeline import analyze as pipeline_analyze

    hw = hw or HardwareSpec()
    ar = pipeline_analyze(spec, domain, hw)
    rf = ar.roofline

    db = _dtype_bytes(spec.dtype)
    loads = rf.bytes_per_point // db - 1  # total bytes includes 1 store
    stores = 1

    return SolverKernelProfile(
        name=f"stencil_{spec.name}",
        flops_per_element=rf.flops_per_point,
        loads_per_element=loads,
        stores_per_element=stores,
        bytes_per_element=rf.bytes_per_point,
        arithmetic_intensity=rf.arithmetic_intensity,
        bound=rf.bound,
        peak_gpoints_s=rf.peak_gpoints_s,
    )


def analyze_cg_iteration(
    operator_profile: SolverKernelProfile,
    N: int,
    hw: HardwareSpec | None = None,
    dtype: str = "float64",
) -> SolverAnalysisResult:
    """Analyze a full CG iteration.

    One CG step consists of:
    - 1 operator application (SpMV or stencil)
    - 3 dot products (r^T r initial, p^T Ap, r^T r new)
    - 2 axpy operations (x += alpha*p, r -= alpha*Ap)

    Parameters
    ----------
    operator_profile : SolverKernelProfile
        Profile of the operator kernel (from analyze_dia_spmv or analyze_stencil_operator).
    N : int
        Problem size (number of unknowns).
    hw : HardwareSpec, optional
        Hardware specification.
    dtype : str
        Data type string.

    Returns
    -------
    SolverAnalysisResult
        Complete analysis of one CG iteration.
    """
    hw = hw or HardwareSpec()
    dot_prof = analyze_dot(N, hw, dtype)
    axpy_prof = analyze_axpy(N, hw, dtype)

    profiles = [
        operator_profile,
        SolverKernelProfile(**{**dot_prof.__dict__, "name": "dot_pAp"}),
        SolverKernelProfile(**{**dot_prof.__dict__, "name": "dot_rr_new"}),
        SolverKernelProfile(**{**dot_prof.__dict__, "name": "dot_rr_init"}),
        SolverKernelProfile(**{**axpy_prof.__dict__, "name": "axpy_x"}),
        SolverKernelProfile(**{**axpy_prof.__dict__, "name": "axpy_r"}),
    ]

    total_flops = (
        operator_profile.flops_per_element
        + 3 * dot_prof.flops_per_element
        + 2 * axpy_prof.flops_per_element
    )
    total_bytes = (
        operator_profile.bytes_per_element
        + 3 * dot_prof.bytes_per_element
        + 2 * axpy_prof.bytes_per_element
    )
    overall_ai = total_flops / total_bytes if total_bytes > 0 else 0.0

    # Bottleneck = kernel with lowest peak throughput
    bottleneck_prof = min(profiles, key=lambda p: p.peak_gpoints_s)

    return SolverAnalysisResult(
        kernel_profiles=profiles,
        total_flops_per_element=total_flops,
        total_bytes_per_element=total_bytes,
        overall_arithmetic_intensity=overall_ai,
        bottleneck=bottleneck_prof.name,
        bottleneck_bound=bottleneck_prof.bound,
    )


def print_solver_analysis(result: SolverAnalysisResult) -> None:
    """Pretty-print a solver analysis result."""
    print("CG Iteration Analysis")
    print("=" * 60)
    print(f"{'Kernel':<20} {'FLOPs/pt':>8} {'B/pt':>6} {'AI':>8} {'Bound':>8} {'Gpts/s':>10}")
    print("-" * 60)
    for p in result.kernel_profiles:
        print(f"{p.name:<20} {p.flops_per_element:>8} {p.bytes_per_element:>6} "
              f"{p.arithmetic_intensity:>8.3f} {p.bound:>8} {p.peak_gpoints_s:>10.2f}")
    print("-" * 60)
    print(f"{'Total':<20} {result.total_flops_per_element:>8} {result.total_bytes_per_element:>6} "
          f"{result.overall_arithmetic_intensity:>8.3f}")
    print(f"\nBottleneck: {result.bottleneck} ({result.bottleneck_bound}-bound)")
