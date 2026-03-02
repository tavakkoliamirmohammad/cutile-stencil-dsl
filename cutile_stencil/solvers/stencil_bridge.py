"""Bridge between DIA sparse matrices and stencil specifications.

Key insight: a DIA matrix with offsets ``[-1, 0, 1]`` and coefficients
``[1, -2, 1]`` is exactly the stencil ``u[i-1] - 2*u[i] + u[i+1]``.
This module converts DIA operators into ``StencilSpec`` objects so that
the full stencil pipeline (analysis, codegen, tiling) can be applied to
iterative solvers.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from cutile_stencil.dsl.types import OffsetAccess, StencilSpec


def dia_to_stencil_spec(
    name: str,
    ndim: int,
    offsets_nd: Sequence[Tuple[int, ...]],
    coefficients: Sequence[float],
    dtype: str = "float64",
) -> StencilSpec:
    """Convert DIA-style offsets and coefficients into a StencilSpec.

    Parameters
    ----------
    name : str
        Name for the generated stencil.
    ndim : int
        Number of spatial dimensions.
    offsets_nd : sequence of int-tuples
        Each entry is an nD offset tuple, e.g. ``(0,)``, ``(-1,)``, ``(1,)``.
    coefficients : sequence of float
        Weight for each offset.  Must have same length as *offsets_nd*.
    dtype : str
        Data type string (``"float64"`` or ``"float32"``).

    Returns
    -------
    StencilSpec
        A specification with ``accesses``, ``halo_widths``, and a synthetic
        ``update_fn`` that reproduces the linear combination.
    """
    if len(offsets_nd) != len(coefficients):
        raise ValueError("offsets_nd and coefficients must have the same length")
    if any(len(o) != ndim for o in offsets_nd):
        raise ValueError(f"Every offset must have {ndim} elements")

    # Build OffsetAccess list
    accesses: List[OffsetAccess] = []
    for off in offsets_nd:
        accesses.append(OffsetAccess(array_name="u", offsets=tuple(off)))

    # Compute halo widths from max absolute offsets per dimension
    halo_widths = [0] * ndim
    for off in offsets_nd:
        for d, o in enumerate(off):
            halo_widths[d] = max(halo_widths[d], abs(o))
    halo_widths = tuple(halo_widths)

    # Compute stencil order = 2 * max absolute offset
    max_abs = max(abs(o) for off in offsets_nd for o in off) if offsets_nd else 0
    order = 2 * max_abs

    # Build synthetic update function via exec
    idx_params = ", ".join(f"i{d}" for d in range(ndim))
    terms = []
    for coeff, off in zip(coefficients, offsets_nd):
        subscript = ", ".join(
            f"i{d}" if o == 0 else f"i{d} + {o}" if o > 0 else f"i{d} - {abs(o)}"
            for d, o in enumerate(off)
        )
        if ndim == 1:
            access = f"u[{subscript}]"
        else:
            access = f"u[{subscript}]"
        if coeff == 1.0:
            terms.append(access)
        elif coeff == -1.0:
            terms.append(f"-{access}")
        else:
            terms.append(f"{coeff} * {access}")
    if not terms:
        body = "0.0"
    else:
        body = " + ".join(terms)
        # Clean up double signs: "+ -" -> "- "
        body = body.replace("+ -", "- ")

    fn_source = f"def {name}(u, {idx_params}):\n    return {body}\n"
    ns = {}
    exec(fn_source, ns)  # noqa: S102
    update_fn = ns[name]
    update_fn._source = fn_source

    spec = StencilSpec(
        name=name,
        ndim=ndim,
        order=order,
        inputs=("u",),
        output="result",
        update_fn=update_fn,
        accesses=accesses,
        dtype=dtype,
        halo_widths=halo_widths,
    )
    return spec


def laplacian_stencil_spec(ndim: int, dtype: str = "float64") -> StencilSpec:
    """Convenience constructor for the standard Laplacian stencil.

    Parameters
    ----------
    ndim : {1, 2, 3}
        Spatial dimension.
    dtype : str
        Data type (``"float64"`` or ``"float32"``).

    Returns
    -------
    StencilSpec
        The Laplacian stencil specification.  For 1D this is
        ``u[i-1] - 2*u[i] + u[i+1]``; for 2D the 5-point stencil; etc.
    """
    if ndim == 1:
        # Matches laplacian_1d_dia: offsets [-1,0,1], data [-1,2,-1]
        offsets_nd = [(-1,), (0,), (1,)]
        coeffs = [-1.0, 2.0, -1.0]
    elif ndim == 2:
        # Matches laplacian_2d_dia: 5-point stencil with center=4
        offsets_nd = [(0, -1), (-1, 0), (0, 0), (1, 0), (0, 1)]
        coeffs = [-1.0, -1.0, 4.0, -1.0, -1.0]
    elif ndim == 3:
        # Matches laplacian_3d_dia: 7-point stencil with center=6
        offsets_nd = [
            (0, 0, -1), (0, -1, 0), (-1, 0, 0),
            (0, 0, 0),
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
        ]
        coeffs = [-1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0]
    else:
        raise ValueError(f"Unsupported ndim={ndim}; expected 1, 2, or 3")

    return dia_to_stencil_spec(
        name=f"laplacian_{ndim}d",
        ndim=ndim,
        offsets_nd=offsets_nd,
        coefficients=coeffs,
        dtype=dtype,
    )


def stencil_spmv_kernel(
    spec: StencilSpec,
    domain: Tuple[int, ...],
    hw=None,
) -> str:
    """Compile a stencil spec and return the generated kernel source.

    This is the bridge function: given a stencil spec (possibly created
    from ``dia_to_stencil_spec``), run the full compile pipeline and
    return the generated cuTile kernel code.

    Parameters
    ----------
    spec : StencilSpec
        Stencil specification (with accesses already set).
    domain : tuple of int
        Domain size per dimension.
    hw : HardwareSpec, optional
        Hardware parameters.

    Returns
    -------
    str
        Generated cuTile kernel source code.
    """
    from cutile_stencil.dsl.pipeline import compile as stencil_compile

    result = stencil_compile(spec, domain, hw)
    return result.code
