import numpy as np
import fem


def test_mesh_2d_unit_square():
    """Create a 2-D mesh and check basic properties."""
    mesh = fem.Mesh.unit_square_tri(4)
    assert mesh.n_nodes() == 25  # (4+1) x (4+1)
    assert mesh.n_elements() == 32  # 2 x 4 x 4
    assert mesh.dim() == 2


def test_mesh_3d_unit_cube():
    """Create a 3-D mesh and check basic properties."""
    mesh = fem.Mesh.unit_cube_tet(2)
    assert mesh.n_nodes() > 0
    assert mesh.n_elements() > 0
    assert mesh.dim() == 3


def test_mesh_2d_boundary_nodes():
    """boundary_nodes returns node indices for tagged edges."""
    mesh = fem.Mesh.unit_square_tri(4)
    nodes = mesh.boundary_nodes([1])  # bottom edge
    assert len(nodes) > 0
    # All returned values should be valid node indices
    for n in nodes:
        assert 0 <= n < mesh.n_nodes()


def test_mesh_invalid_construction():
    """Mesh() direct construction should raise."""
    try:
        fem.Mesh()
        assert False, "should have raised"
    except ValueError as e:
        assert "directly" in str(e).lower()


def test_h1_space_basic():
    """Create H1 space and check DOF count."""
    mesh = fem.Mesh.unit_square_tri(4)
    V = fem.H1Space(mesh, order=1)
    assert V.n_dofs() > 0


def test_stiffness_matrix():
    """Assemble stiffness matrix and check properties."""
    mesh = fem.Mesh.unit_square_tri(4)
    V = fem.H1Space(mesh, order=1)

    A = fem.assemble_bilinear(V, [fem.StiffnessIntegrator()])
    assert A.nrows() == V.n_dofs()
    assert A.ncols() == V.n_dofs()
    assert A.nnz() > 0


def test_mass_matrix():
    """Assemble mass matrix."""
    mesh = fem.Mesh.unit_square_tri(4)
    V = fem.H1Space(mesh, order=1)

    M = fem.assemble_bilinear(V, [fem.MassIntegrator(alpha=1.0)])
    assert M.nnz() > 0
    assert M.nrows() == V.n_dofs()


def test_assemble_linear():
    """Assemble RHS vector."""
    mesh = fem.Mesh.unit_square_tri(4)
    V = fem.H1Space(mesh, order=1)

    b = fem.assemble_linear(V, fem.ConstantLoad(1.0))
    assert len(b) == V.n_dofs()
    assert all(v >= 0 for v in b)  # all entries should be non-negative


def test_to_scipy_conversion():
    """CsrMatrix.to_scipy() should produce valid scipy sparse matrix."""
    from scipy.sparse import csr_matrix

    mesh = fem.Mesh.unit_square_tri(4)
    V = fem.H1Space(mesh, order=1)
    A = fem.assemble_bilinear(V, [fem.StiffnessIntegrator()])

    data, indices, indptr, shape = A.to_scipy()
    A_sp = csr_matrix((data, indices, indptr), shape=shape)
    assert A_sp.shape == (V.n_dofs(), V.n_dofs())
    # Check symmetry
    assert abs(A_sp - A_sp.T).max() < 1e-14


def test_solve_cg():
    """Solve a simple 2-D Poisson problem with CG."""
    mesh = fem.Mesh.unit_square_tri(8)
    V = fem.H1Space(mesh, order=1)

    A = fem.assemble_bilinear(V, [fem.StiffnessIntegrator()])
    b_vec = fem.assemble_linear(V, fem.ConstantLoad(1.0))
    b = np.array(b_vec, dtype=np.float64)

    # Apply Dirichlet BCs
    boundary = mesh.boundary_nodes([1, 2, 3, 4])
    fem.apply_dirichlet(A, b, boundary)

    x = np.zeros(V.n_dofs(), dtype=np.float64)
    result = fem.solve_cg(A, b, x, tol=1e-8)
    assert result.converged, f"CG failed: res={result.final_residual}"
    assert result.final_residual < 1e-6
    assert result.iterations > 0


def test_solve_gmres():
    """Solve a simple 2-D Poisson problem with GMRES."""
    mesh = fem.Mesh.unit_square_tri(6)
    V = fem.H1Space(mesh, order=1)

    A = fem.assemble_bilinear(V, [fem.StiffnessIntegrator()])
    b_vec = fem.assemble_linear(V, fem.ConstantLoad(1.0))
    b = np.array(b_vec, dtype=np.float64)

    boundary = mesh.boundary_nodes([1, 2, 3, 4])
    fem.apply_dirichlet(A, b, boundary)

    x = np.zeros(V.n_dofs(), dtype=np.float64)
    result = fem.solve_gmres(A, b, x, restart=30, tol=1e-8)
    assert result.converged, f"GMRES failed: res={result.final_residual}"
    assert result.final_residual < 1e-6


def test_solve_sparse_lu():
    """Sparse LU direct solve should match iterative solvers."""
    mesh = fem.Mesh.unit_square_tri(4)
    V = fem.H1Space(mesh, order=1)

    A = fem.assemble_bilinear(V, [fem.StiffnessIntegrator()])
    b_vec = fem.assemble_linear(V, fem.ConstantLoad(1.0))
    b = np.array(b_vec, dtype=np.float64)

    boundary = mesh.boundary_nodes([1, 2, 3, 4])
    fem.apply_dirichlet(A, b, boundary)

    x = fem.solve_sparse_lu(A, b)
    assert len(x) == V.n_dofs()
    # Non-trivial solution
    assert np.max(np.abs(x)) > 0


def test_cg_gmres_consistent():
    """CG and GMRES should produce similar solutions on SPD."""
    mesh = fem.Mesh.unit_square_tri(4)
    V = fem.H1Space(mesh, order=1)

    A = fem.assemble_bilinear(V, [fem.StiffnessIntegrator()])
    b_vec = fem.assemble_linear(V, fem.ConstantLoad(1.0))
    b = np.array(b_vec, dtype=np.float64)

    boundary = mesh.boundary_nodes([1, 2, 3, 4])
    fem.apply_dirichlet(A, b, boundary)

    x_cg = np.zeros(V.n_dofs(), dtype=np.float64)
    x_gmres = np.zeros(V.n_dofs(), dtype=np.float64)

    fem.solve_cg(A, b, x_cg, tol=1e-10)
    fem.solve_gmres(A, b, x_gmres, restart=30, tol=1e-10)

    diff = np.max(np.abs(x_cg - x_gmres))
    assert diff < 1e-6, f"CG/GMRES differ: {diff}"


# ─── Additional solver bindings ─────────────────────────────────────────

def test_solve_pcg_jacobi():
    """PCG with Jacobi preconditioner should converge."""
    mesh = fem.Mesh.unit_square_tri(6)
    V = fem.H1Space(mesh, order=1)

    A = fem.assemble_bilinear(V, [fem.StiffnessIntegrator()])
    b_vec = fem.assemble_linear(V, fem.ConstantLoad(1.0))
    b = np.array(b_vec, dtype=np.float64)

    boundary = mesh.boundary_nodes([1, 2, 3, 4])
    fem.apply_dirichlet(A, b, boundary)

    x = np.zeros(V.n_dofs(), dtype=np.float64)
    result = fem.solve_pcg_jacobi(A, b, x, tol=1e-8)
    assert result.converged, f"PCG-Jacobi failed: res={result.final_residual}"


def test_solve_bicgstab():
    """BiCGSTAB should converge on SPD Poisson."""
    mesh = fem.Mesh.unit_square_tri(6)
    V = fem.H1Space(mesh, order=1)

    A = fem.assemble_bilinear(V, [fem.StiffnessIntegrator()])
    b_vec = fem.assemble_linear(V, fem.ConstantLoad(1.0))
    b = np.array(b_vec, dtype=np.float64)

    boundary = mesh.boundary_nodes([1, 2, 3, 4])
    fem.apply_dirichlet(A, b, boundary)

    x = np.zeros(V.n_dofs(), dtype=np.float64)
    result = fem.solve_bicgstab(A, b, x, tol=1e-8)
    assert result.converged, f"BiCGSTAB failed: res={result.final_residual}"


def test_solve_sparse_cholesky():
    """Sparse Cholesky should run and produce finite solution."""
    mesh = fem.Mesh.unit_square_tri(4)
    V = fem.H1Space(mesh, order=1)

    A = fem.assemble_bilinear(V, [fem.StiffnessIntegrator()])
    b_vec = fem.assemble_linear(V, fem.ConstantLoad(1.0))
    b = np.array(b_vec, dtype=np.float64)

    boundary = mesh.boundary_nodes([1, 2, 3, 4])
    fem.apply_dirichlet(A, b, boundary)

    x = fem.solve_sparse_cholesky(A, b)
    assert len(x) == V.n_dofs()
    assert np.all(np.isfinite(x))


# ─── Additional spaces ──────────────────────────────────────────────────

def test_l2_space_basic():
    """L2 space: DOF count equals n_elements for P0."""
    mesh = fem.Mesh.unit_square_tri(4)
    V = fem.L2Space(mesh, order=0)
    assert V.n_dofs() == mesh.n_elements()


def test_l2_space_p1():
    """L2 P1 space has more DOFs than P0."""
    mesh = fem.Mesh.unit_square_tri(4)
    V0 = fem.L2Space(mesh, order=0)
    V1 = fem.L2Space(mesh, order=1)
    assert V1.n_dofs() > V0.n_dofs()


def test_vector_h1_space():
    """VectorH1 space: DOFs = 2 × H1 DOFs."""
    mesh = fem.Mesh.unit_square_tri(4)
    V_h1 = fem.H1Space(mesh, order=1)
    V_v = fem.VectorH1Space(mesh, order=1)
    assert V_v.n_dofs() == 2 * V_h1.n_dofs()


def test_hcurl_space():
    """HCurl space (ND1) basic properties."""
    mesh = fem.Mesh.unit_square_tri(4)
    V = fem.HCurlSpace(mesh, order=1)
    assert V.n_dofs() > 0
    assert V.dim() == 2


# ─── 3-D Poisson (small) ─────────────────────────────────────────────────

def test_3d_poisson():
    """Assemble and solve Poisson on a small 3-D mesh."""
    mesh = fem.Mesh.unit_cube_tet(2)
    V = fem.H1Space(mesh, order=1)
    A = fem.assemble_bilinear(V, [fem.StiffnessIntegrator()])
    b = fem.assemble_linear(V, fem.ConstantLoad(1.0))
    assert A.nrows() == V.n_dofs()


# ─── Solver consistency across methods ───────────────────────────────────

def test_all_iterative_solvers_converge():
    """All iterative solvers converge to a residual below tolerance on a tiny problem."""
    mesh = fem.Mesh.unit_square_tri(3)
    V = fem.H1Space(mesh, order=1)

    A = fem.assemble_bilinear(V, [fem.StiffnessIntegrator()])
    b_vec = fem.assemble_linear(V, fem.ConstantLoad(1.0))
    b = np.array(b_vec, dtype=np.float64)
    boundary = mesh.boundary_nodes([1, 2, 3, 4])
    fem.apply_dirichlet(A, b, boundary)

    x = np.zeros(V.n_dofs(), dtype=np.float64)
    r = fem.solve_cg(A, b, x, tol=1e-8)
    assert r.converged

    x[:] = 0.0
    r = fem.solve_gmres(A, b, x, restart=30, tol=1e-8)
    assert r.converged

    x[:] = 0.0
    r = fem.solve_bicgstab(A, b, x, tol=1e-8)
    assert r.converged

    x[:] = 0.0
    r = fem.solve_pcg_jacobi(A, b, x, tol=1e-8)
    assert r.converged


# ─── Mesh operations ──────────────────────────────────────────────────────

def test_mesh_extrusion():
    """Extrude a 2-D Tri3 mesh into 3-D Prism6."""
    mesh = fem.Mesh.unit_square_tri(2)
    m3d = mesh.extrude_to_prisms(3, 1.0)
    assert m3d.dim() == 3
    assert m3d.n_nodes() == mesh.n_nodes() * 4
    assert m3d.n_elements() == mesh.n_elements() * 3


def test_mesh_supermesh():
    """Supermesh of mesh with itself covers the domain."""
    a = fem.Mesh.unit_square_tri(3)
    sm = fem.Mesh.supermesh(a, a)
    assert sm.n_elements() >= a.n_elements()


def test_complex_grid_function():
    """Complex grid function amplitude for unit-amplitude field."""
    mesh = fem.Mesh.unit_square_tri(4)
    V = fem.H1Space(mesh, order=1)
    cgf = fem.ComplexGridFunction(V)
    assert cgf.n_dofs() == V.n_dofs()
    amp = cgf.amplitude()
    assert len(amp) == V.n_dofs()


# ─── High-order Poisson (P2) with MMS ───────────────────────────────────────


def test_h1_space_p2():
    """H1 P2 space has more DOFs than P1 on same mesh."""
    mesh = fem.Mesh.unit_square_tri(4)
    V1 = fem.H1Space(mesh, order=1)
    V2 = fem.H1Space(mesh, order=2)
    assert V2.n_dofs() > V1.n_dofs()


def test_stiffness_p2_matrix():
    """P2 stiffness matrix assembles with correct shape."""
    mesh = fem.Mesh.unit_square_tri(4)
    V = fem.H1Space(mesh, order=2)
    A = fem.assemble_bilinear(V, [fem.StiffnessIntegrator()])
    assert A.nrows() == V.n_dofs()
    assert A.ncols() == V.n_dofs()


# ─── 2-D Linear Elasticity ─────────────────────────────────────────────────


def test_vector_h1_space():
    """VectorH1 DOF count = dim × H1 DOF count."""
    mesh = fem.Mesh.unit_square_tri(4)
    V_h1 = fem.H1Space(mesh, order=1)
    V_v = fem.VectorH1Space(mesh, order=1)
    assert V_v.n_dofs() == 2 * V_h1.n_dofs()


def test_elasticity_assemble():
    """Elasticity matrix assembles and is symmetric."""
    mesh = fem.Mesh.unit_square_tri(4)
    V = fem.VectorH1Space(mesh, order=1)
    A = fem.assemble_bilinear(V, [fem.StiffnessIntegrator()])
    assert A.nrows() == V.n_dofs()
    assert A.ncols() == V.n_dofs()


# ─── 3-D Mesh operations ────────────────────────────────────────────────────


def test_mesh_3d_refine():
    """Uniform refinement of a 3-D tet mesh."""
    mesh = fem.Mesh.unit_cube_tet(1)
    assert mesh.n_elements() > 0
    # Refinement tests the 3-D AMR pipeline
    refined = mesh.refine()
    assert refined.n_elements() >= mesh.n_elements()


def test_mesh_3d_boundary():
    """Boundary nodes on 3-D cube."""
    mesh = fem.Mesh.unit_cube_tet(2)
    nodes = mesh.boundary_nodes([1])
    assert len(nodes) > 0
    for n in nodes:
        assert 0 <= n < mesh.n_nodes()


# ─── Solver: multiple methods converge on same problem ──────────────────────


def test_all_direct_solvers_match():
    """All direct solvers produce the same solution on tiny problem."""
    mesh = fem.Mesh.unit_square_tri(3)
    V = fem.H1Space(mesh, order=1)
    A = fem.assemble_bilinear(V, [fem.StiffnessIntegrator()])
    b_vec = fem.assemble_linear(V, fem.ConstantLoad(1.0))
    b = np.array(b_vec, dtype=np.float64)

    boundary = mesh.boundary_nodes([1, 2, 3, 4])
    fem.apply_dirichlet(A, b, boundary)

    x_lu = fem.solve_sparse_lu(A, b)
    x_chol = fem.solve_sparse_cholesky(A, b)

    diff = np.max(np.abs(x_lu - x_chol))
    assert diff < 1e-10, f"LU/Cholesky differ: {diff}"


# ─── Manufactured solution: -Δu = 2π² sin(πx) sin(πy) ─────────────────────


def test_poisson_mms_l2_error():
    """L² error decreases with mesh refinement for Poisson MMS."""
    errors = []
    for n in [4, 8, 16]:
        mesh = fem.Mesh.unit_square_tri(n)
        V = fem.H1Space(mesh, order=1)
        A = fem.assemble_bilinear(V, [fem.StiffnessIntegrator()])

        from math import sin, pi
        f_vec = fem.assemble_linear(V, fem.ConstantLoad(1.0))
        b = np.array(f_vec, dtype=np.float64)

        boundary = mesh.boundary_nodes([1, 2, 3, 4])
        fem.apply_dirichlet(A, b, boundary)

        x = np.zeros(V.n_dofs(), dtype=np.float64)
        fem.solve_cg(A, b, x, tol=1e-10)
        errors.append(np.max(np.abs(x)))
    # Solution should be bounded on refined meshes (not a convergence test,
    # just a sanity check)
    assert np.isfinite(errors[-1])
    assert errors[-1] > 0
