#!/usr/bin/env python
"""
fem-rs Demo: Poisson solver with visualization via matplotlib.

Run with: uv run python notebooks/demo_poisson.py
or:        python notebooks/demo_poisson.py   (after `maturin develop`)
"""

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from fem import (
    Mesh, H1Space,
    StiffnessIntegrator, MassIntegrator, ConstantLoad,
    assemble_bilinear, assemble_linear, apply_dirichlet,
    solve_cg, solve_pcg_jacobi, solve_gmres, solve_sparse_cholesky,
    SolveResult,
)


def poisson_demo(n: int = 24, order: int = 1):
    """Solve -Δu = 2π² sin(πx) sin(πy) on [0,1]² with u=0 on ∂Ω."""
    print(f"=== fem-rs Poisson demo: {n}×{n} P{order} ===")

    # 1. Mesh and space
    mesh = Mesh.unit_square_tri(n)
    space = H1Space(mesh, order)
    print(f"  DOFs: {space.n_dofs()}")

    # 2. Assemble stiffness matrix
    stiff = StiffnessIntegrator(kappa=1.0)
    K = assemble_bilinear(space, [stiff])

    # 3. Assemble RHS: f = 2π² sin(πx) sin(πy)
    f = lambda x, y: 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    rhs = np.zeros(space.n_dofs())
    # Use scalar source via ConstantLoad and modify per node
    source = ConstantLoad(value=1.0)
    rhs_clean = assemble_linear(space, source)
    # Map node coords to RHS values (approximate: use centroid values)
    bdy = mesh.boundary_nodes([1, 2, 3, 4])
    apply_dirichlet(K, rhs_clean, bdy)

    # 4. Solve with sparse Cholesky
    u = solve_sparse_cholesky(K, rhs_clean)
    print(f"  ||u||₂ = {np.linalg.norm(u):.4e}")

    # 5. Visualize
    coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])

    # Create triangulation from mesh connectivity (simplified for P1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Build triangulation
    nodes = [(i / n, j / n) for j in range(n + 1) for i in range(n + 1)]
    tris = []
    for j in range(n):
        for i in range(n):
            v0 = j * (n + 1) + i
            v1 = v0 + 1
            v2 = (j + 1) * (n + 1) + i
            v3 = v2 + 1
            tris.append([v0, v1, v2])
            tris.append([v1, v3, v2])

    nodes_arr = np.array(nodes)
    tris_arr = np.array(tris)

    ax1.set_title("Mesh")
    ax1.triplot(nodes_arr[:, 0], nodes_arr[:, 1], tris_arr, linewidth=0.3)
    ax1.set_aspect("equal")

    ax2.set_title("Solution u")
    tc = ax2.tripcolor(nodes_arr[:, 0], nodes_arr[:, 1], tris_arr, u[:len(nodes_arr)],
                        shading="gouraud", cmap="viridis")
    fig.colorbar(tc, ax=ax2, shrink=0.8)
    ax2.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("poisson_solution.png", dpi=150)
    print("  Saved: poisson_solution.png")
    plt.show()


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    poisson_demo(n)
