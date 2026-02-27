# cuTile Stencil DSL

High-order stencil compilation and tile-based iterative solvers via [cuTile](https://github.com/NVIDIA/cutile). PhD research framework implementing:

- **Direction 8**: High-order stencil compilation via tile abstractions
- **Direction 9**: Tile-based iterative solvers for sparse linear systems

## Architecture

The framework has two layers:

1. **Pure Python** — DSL, analysis, reference implementations, and tests. Runs on any machine with Python 3.10+ and NumPy.
2. **Generated cuTile code** — Syntactically correct Python files (using `cuda.tile`) ready to run when cuTile is available on an NVIDIA GPU.

```
cutile_stencil/
├── dsl/            # @stencil decorator, types, registry
├── analysis/       # Footprint extraction, tiling, temporal blocking, roofline model
├── codegen/        # cuTile kernel code generation (1D/2D/3D stencils, solvers)
├── solvers/        # Sparse formats (DIA/BSR), CG drivers, persistent CG, mixed-precision
└── reference/      # NumPy reference: stencil executor, SpMV, CG solver
```

## Setup

```bash
# Clone and create virtual environment
git clone <repo-url> && cd cutile-stencil-dsl
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy pytest
```

## Running Tests

```bash
python -m pytest tests/ -v
```

This runs 53 tests covering:

| Test file | What it tests |
|---|---|
| `test_dsl.py` | `@stencil` decorator, footprint extraction (1D/2D/3D), registry |
| `test_tiling.py` | Tile config selection, temporal blocking, roofline analysis |
| `test_codegen.py` | All generated code passes `ast.parse()`, contains expected cuTile API calls |
| `test_reference.py` | NumPy stencil correctness (heat equation, 2D/3D Laplacian) |
| `test_solvers.py` | DIA/BSR SpMV, CG convergence (1D/2D Poisson), mixed-precision CG |
| `test_examples.py` | Smoke tests — runs every example end-to-end (DSL → analysis → codegen → NumPy validation) |

## Running Examples

Each example demonstrates the full pipeline: define stencil → run analysis → generate cuTile kernel → validate with NumPy.

```bash
# 1D heat equation (explicit Euler, Gaussian initial condition)
python examples/heat_1d.py

# 2D acoustic wave equation (4th-order stencil, leapfrog integration)
python examples/wave_2d.py

# 3D 7-point Laplacian stencil
python examples/laplacian_3d.py

# Poisson equation solved with Conjugate Gradient (1D and 2D)
python examples/poisson_cg.py

# Mixed-precision CG: FP64 outer refinement + FP32 inner solve
python examples/mixed_precision_cg.py
```

Generated cuTile kernels are written to `examples/generated/`.

## Quick Start

Define a stencil, analyze it, and generate a cuTile kernel:

```python
from cutile_stencil.dsl.decorator import stencil
from cutile_stencil.dsl.types import HardwareSpec
from cutile_stencil.analysis.footprint import extract_footprint, compute_halo
from cutile_stencil.analysis.tiling import compute_tile_config
from cutile_stencil.analysis.roofline import roofline_analysis
from cutile_stencil.codegen.stencil_codegen import StencilCodeGenerator

@stencil(ndim=1, order=2, dtype="float64")
def heat_1d(u, i):
    return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

spec = heat_1d._stencil_spec
accesses = extract_footprint(spec)
spec.halo_widths = compute_halo(accesses, spec.ndim)

hw = HardwareSpec(peak_bandwidth_gbs=1000.0, shared_mem_bytes=49152,
                  sm_count=108, dtype_bytes=8)
tile_cfg = compute_tile_config(spec, domain=(1024,), hw=hw)
roof = roofline_analysis(spec, hw)

print(f"Arithmetic intensity: {roof.arithmetic_intensity:.3f}, bound: {roof.bound}")

codegen = StencilCodeGenerator(spec, tile_cfg)
codegen.emit_to_file("heat_1d_kernel.py")
```

To run the NumPy reference simulation (no GPU needed):

```python
import numpy as np
from cutile_stencil.reference.stencil_ref import time_march

u0 = np.exp(-((np.linspace(0, 1, 128) - 0.5) ** 2) / 0.01)
results = time_march(u0, spec, steps=100)
```

## Solver Example

Solve Poisson's equation with CG:

```python
import numpy as np
from cutile_stencil.solvers.formats import laplacian_1d_dia
from cutile_stencil.reference.spmv_ref import dia_spmv
from cutile_stencil.reference.cg_ref import cg_solve

N = 100
A = laplacian_1d_dia(N)
b = np.ones(N)

x, iters, residuals = cg_solve(lambda v: dia_spmv(A, v), b, tol=1e-10)
print(f"Converged in {iters} iterations, final residual: {residuals[-1]:.2e}")
```
