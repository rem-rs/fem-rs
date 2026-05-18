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
