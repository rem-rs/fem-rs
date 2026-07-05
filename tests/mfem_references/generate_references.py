#!/usr/bin/env python3
"""
Generate MFEM reference values for cross-validation with fem-rs.

Runs MFEM's ex1 (Poisson) and ex2 (Elasticity) and extracts numerical
results for comparison with fem-rs.

Requirements:
    - MFEM Python package installed: pip install mfem
    - Python 3.8+ with numpy

Usage:
    python generate_references.py > references.json
"""

import json
import sys
import numpy as np
from math import pi, sin, cos


def solve_poisson_ex1(n_subdiv=8, order=1):
    """
    MFEM ex1: Poisson equation with manufactured solution.

    Problem:
        -Laplacian(u) = 2*pi^2*sin(pi*x)*sin(pi*y)  in [0,1]^2
        u = 0  on boundary

    Exact solution: u = sin(pi*x)*sin(pi*y)
    """
    from mfem.ser import (
        Mesh, H1_FECollection, FiniteElementSpace,
        BilinearForm, LinearForm,
        DiffusionIntegrator, DomainLFIntegrator,
        GridFunction, GSSmoother, CGSolver,
        ConstantCoefficient, FunctionCoefficient,
        intArray, Element,
    )

    # Create mesh: n_subdiv x n_subdiv triangular mesh on [0,1]^2
    mesh = Mesh.MakeCartesian2D(n_subdiv, n_subdiv, Element.TRIANGLE, True, 1.0, 1.0)

    # H1 finite element space
    fec = H1_FECollection(order, 2)
    fespace = FiniteElementSpace(mesh, fec)
    n_dofs = fespace.GetNDofs()

    # Bilinear form: a(u,v) = int grad(u) . grad(v)
    a = BilinearForm(fespace)
    a.AddDomainIntegrator(DiffusionIntegrator())
    a.Assemble()

    # Linear form: f(v) = int 2*pi^2*sin(pi*x)*sin(pi*y) * v
    def rhs_exact(vec):
        x, y = vec[0], vec[1]
        return 2.0 * pi**2 * sin(pi * x) * sin(pi * y)

    f = LinearForm(fespace)
    f.AddDomainIntegrator(DomainLFIntegrator(FunctionCoefficient(rhs_exact)))
    f.Assemble()

    # Solve with CG + Jacobi
    u = GridFunction(fespace)
    u.Assign(0.0)
    A = a.FormLinearSystem([])  # no essential BCs for this test setup

    # Actually, MFEM ex1 applies Dirichlet BC. Let's do it properly.
    # For the manufactured solution, we need to set u=0 on boundary.
    ess_tdof_list = intArray()
    fespace.GetBoundaryTrueDofs(ess_tdof_list)
    a.FormEssentialBC(ess_tdof_list, u, f)

    A = a.FormLinearSystem(ess_tdof_list, u, f)
    B = a.rhs
    X = a.solution

    solver = CGSolver()
    prec = GSSmoother()
    solver.SetPreconditioner(prec)
    solver.SetOperator(A)
    solver.SetRelTol(1e-12)
    solver.SetAbsTol(1e-24)
    solver.SetMaxIter(2000)
    solver.SetVerbose(0)
    solver.Mult(B, X)

    a.RecoverFEMSolution(X, f, u)

    # Compute L2 error
    def exact_solution(vec):
        x, y = vec[0], vec[1]
        return sin(pi * x) * sin(pi * y)

    l2_error = u.ComputeLpError(2.0, FunctionCoefficient(exact_solution))

    return {
        "example": "ex1_poisson",
        "mesh": f"{n_subdiv}x{n_subdiv}",
        "order": order,
        "n_dofs": n_dofs,
        "l2_error": l2_error,
        "converged": solver.GetConverged(),
        "iterations": solver.GetNumIterations(),
    }


def solve_elasticity_ex2(n_subdiv=8, order=1):
    """
    MFEM ex2: Linear elasticity with gravity.

    Problem:
        -div(sigma(u)) = f  in [0,1]^2
        u = 0  on left wall (x=0)
        sigma.n = 0  elsewhere

    Material: E=1, nu=0.3 (plane strain)
    Body force: f = (0, -1) (gravity)
    """
    from mfem.ser import (
        Mesh, H1_FECollection, VectorFiniteElementSpace,
        BilinearForm, LinearForm,
        ElasticityIntegrator, VectorDomainLFIntegrator,
        GridFunction, GSSmoother, CGSolver,
        VectorConstantCoefficient, VectorFunctionCoefficient,
        intArray, Element,
    )

    # Create mesh
    mesh = Mesh.MakeCartesian2D(n_subdiv, n_subdiv, Element.TRIANGLE, True, 1.0, 1.0)

    # Vector H1 space (2 components)
    fec = H1_FECollection(order, 2)
    fespace = VectorFiniteElementSpace(mesh, fec, 2)
    n_dofs = fespace.GetNDofs()

    # Material: E=1, nu=0.3 (plane strain)
    E_mod = 1.0
    nu = 0.3
    lambda_val = E_mod * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu_val = E_mod / (2.0 * (1.0 + nu))

    # Bilinear form: elasticity
    a = BilinearForm(fespace)
    a.AddDomainIntegrator(ElasticityIntegrator(lambda_val, mu_val))
    a.Assemble()

    # Linear form: body force (0, -1)
    def gravity(vec):
        return [0.0, -1.0]

    f = LinearForm(fespace)
    f.AddDomainIntegrator(VectorDomainLFIntegrator(VectorFunctionCoefficient(2, gravity)))
    f.Assemble()

    # Apply Dirichlet BC on left wall (x=0)
    u = GridFunction(fespace)
    u.Assign(0.0)

    ess_tdof_list = intArray()
    fespace.GetBoundaryTrueDofs(ess_tdof_list)
    a.FormEssentialBC(ess_tdof_list, u, f)

    A = a.FormLinearSystem(ess_tdof_list, u, f)
    B = a.rhs
    X = a.solution

    # Solve
    solver = CGSolver()
    prec = GSSmoother()
    solver.SetPreconditioner(prec)
    solver.SetOperator(A)
    solver.SetRelTol(1e-12)
    solver.SetAbsTol(1e-24)
    solver.SetMaxIter(5000)
    solver.SetVerbose(0)
    solver.Mult(B, X)

    a.RecoverFEMSolution(X, f, u)

    # Extract component norms
    ux = u.GetVector(0)
    uy = u.GetVector(1)
    ux_norm = ux.Norm()
    uy_norm = uy.Norm()

    return {
        "example": "ex2_elasticity",
        "mesh": f"{n_subdiv}x{n_subdiv}",
        "order": order,
        "n_dofs": n_dofs,
        "ux_norm": ux_norm,
        "uy_norm": uy_norm,
        "converged": solver.GetConverged(),
        "iterations": solver.GetNumIterations(),
    }


def main():
    results = {}

    # Ex1: Poisson
    print("Running MFEM ex1 (Poisson)...", file=sys.stderr)
    for n in [8, 16]:
        for order in [1, 2]:
            key = f"ex1_poisson_{n}x{n}_p{order}"
            try:
                ex1 = solve_poisson_ex1(n, order)
                results[key] = ex1
                print(f"  {key}: L2={ex1['l2_error']:.15e}, dofs={ex1['n_dofs']}", file=sys.stderr)
            except Exception as e:
                print(f"  {key}: FAILED - {e}", file=sys.stderr)

    # Ex2: Elasticity
    print("Running MFEM ex2 (Elasticity)...", file=sys.stderr)
    for n in [8, 16]:
        for order in [1, 2]:
            key = f"ex2_elasticity_{n}x{n}_p{order}"
            try:
                ex2 = solve_elasticity_ex2(n, order)
                results[key] = ex2
                print(f"  {key}: ||ux||={ex2['ux_norm']:.15e}, ||uy||={ex2['uy_norm']:.15e}", file=sys.stderr)
            except Exception as e:
                print(f"  {key}: FAILED - {e}", file=sys.stderr)

    # Output JSON
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
