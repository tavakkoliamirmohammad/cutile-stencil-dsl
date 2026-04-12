"""Variable-coefficient diffusion stencil example.

Demonstrates multi-input stencils where D[i,j] is a spatially-varying
diffusion coefficient:
    du/dt = D[i,j] * laplacian(u[i,j])
         = D[i,j] * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*u[i,j])
"""

from cutile_stencil import stencil, compile


@stencil(ndim=2, order=2)
def variable_diffusion(D, u, i, j):
    """Variable-coefficient 2D diffusion: D[i,j] * Laplacian(u)."""
    lap = u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j]
    return D[i, j] * lap


def main():
    print("=" * 60)
    print("Variable-Coefficient Diffusion Stencil")
    print("=" * 60)

    result = compile(variable_diffusion, domain=(256, 256))
    result.print_summary()

    print(f"\nInputs: {result.spec.inputs}")
    assert result.spec.inputs == ("D", "u"), f"Expected ('D', 'u'), got {result.spec.inputs}"
    print(f"Number of accesses: {len(result.spec.accesses)}")
    print(f"Generated code length: {len(result.code)} chars")

    # Verify multi-input: code should reference both D and u
    assert "D" in result.code, "Generated code should reference D array"
    assert "variable_diffusion_kernel(D, u" in result.code or "variable_diffusion_kernel(D," in result.code, \
        "Kernel should accept both D and u as parameters"

    print("\nDone! Variable-coefficient stencil compiled successfully.")


if __name__ == "__main__":
    main()
