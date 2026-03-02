from cutile_stencil.solvers.formats import DIAMatrix, BSRMatrix, laplacian_1d_dia, laplacian_2d_dia, laplacian_3d_dia
from cutile_stencil.solvers.kernels import generate_dia_spmv, generate_axpy, generate_dot, generate_norm2
from cutile_stencil.solvers.stencil_bridge import dia_to_stencil_spec, laplacian_stencil_spec
from cutile_stencil.solvers.stencil_cg import generate_stencil_cg, compile_stencil_cg
from cutile_stencil.solvers.solver_analysis import analyze_cg_iteration, SolverAnalysisResult
from cutile_stencil.solvers.solver_compile import (
    SolverCompileResult,
    CGCompileResult,
    MixedPrecisionCGCompileResult,
    PersistentCGCompileResult,
    StencilCGCompileResult,
)
from cutile_stencil.solvers.cg import compile_cg
from cutile_stencil.solvers.mixed_precision import compile_mixed_precision_cg
from cutile_stencil.solvers.persistent_cg import compile_persistent_cg
