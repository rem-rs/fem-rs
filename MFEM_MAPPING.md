# fem-rs <-> MFEM Correspondence Reference
> Tracks every major MFEM concept and its planned or implemented fem-rs counterpart.
> Use this as the authoritative target checklist for feature completeness.
>
> Status legend: [OK] implemented · 🔨 partial · 🔲 planned · [N/A] out-of-scope

---

## Table of Contents
1. [Mesh](#1-mesh)
2. [Reference Elements & Quadrature](#2-reference-elements--quadrature)
3. [Finite Element Spaces](#3-finite-element-spaces)
4. [Coefficients](#4-coefficients)
5. [Assembly: Forms & Integrators](#5-assembly-forms--integrators)
6. [Linear Algebra](#6-linear-algebra)
7. [Solvers & Preconditioners](#7-solvers--preconditioners)
8. [Algebraic Multigrid](#8-algebraic-multigrid)
9. [Parallel Infrastructure](#9-parallel-infrastructure)
10. [I/O & Visualization](#10-io--visualization)
11. [Grid Functions & Post-processing](#11-grid-functions--post-processing)
12. [MFEM Examples <-> fem-rs Milestones](#12-mfem-examples--fem-rs-milestones)
13. [Key Design Differences](#13-key-design-differences)

---

## 1. Mesh

### 1.1 Mesh Container

| MFEM class / concept | fem-rs equivalent | Status | Notes |
|---|---|---|---|
| `Mesh` (2D/3D unstructured) | `SimplexMesh<D>` | [OK]| Uniform element type per mesh |
| `Mesh` (mixed elements) | `SimplexMesh<D>` + `elem_types`/`elem_offsets` | 🔨 | Phase 42a: data structures + I/O done |
| `NCMesh` (non-conforming) | `refine_nonconforming()` (2-D) + `refine_nonconforming_3d()` + `NCState`/`NCState3D` | [OK]| Tri3/Tet4 multi-level non-conforming refinement + hanging constraints |
| `ParMesh` | `ParallelMesh<M>` | [OK]| Phase 10+33 |
| `Mesh::GetNV()` | `MeshTopology::n_nodes()` | [OK]| |
| `Mesh::GetNE()` | `MeshTopology::n_elements()` | [OK]| |
| `Mesh::GetNBE()` | `MeshTopology::n_boundary_faces()` | [OK]| |
| `Mesh::GetVerticesArray()` | `SimplexMesh::coords` (flat `Vec<f64>`) | [OK]| |
| `Mesh::GetElementVertices()` | `MeshTopology::element_nodes()` | [OK]| |
| `Mesh::GetBdrElementVertices()` | `MeshTopology::face_nodes()` | [OK]| |
| `Mesh::GetBdrAttribute()` | `MeshTopology::face_tag()` | [OK]| Tags match GMSH physical group IDs |
| `Mesh::GetAttribute()` | `MeshTopology::element_tag()` | [OK]| Material group tag |
| `Mesh::bdr_attributes` | `SimplexMesh::unique_boundary_tags()` | [OK]| Sorted, deduplicated boundary tag set |
| `Mesh::GetDim()` | `MeshTopology::dim()` | [OK]| Returns `u8` (2 or 3) |
| `Mesh::GetSpaceDim()` | same as `dim()` for flat meshes | [OK]| |
| `Mesh::UniformRefinement()` | `refine_uniform()` | [OK]| Red refinement (Tri3->4 children) |
| `Mesh::AdaptiveRefinement()` | `refine_marked()` + ZZ estimator + Dörfler marking | [OK]| Phase 17 |
| `Mesh::GetElementTransformation()` | `ElementTransformation` | [OK]| 仿射 simplex 装配路径已统一接入 `ElementTransformation` |
| `Mesh::GetFaceElementTransformations()` | `InteriorFaceList` | [OK]| Used by DG assembler |
| `Mesh::GetBoundingBox()` | `SimplexMesh::bounding_box()` | [OK]| Returns `(min, max)` per axis |

### 1.2 Element Types

| MFEM element | `ElementType` variant | dim | Nodes | Status |
|---|---|---|---|---|
| `Segment` | `Line2` | 1 | 2 | [OK]|
| Quadratic segment | `Line3` | 1 | 3 | [OK]|
| `Triangle` | `Tri3` | 2 | 3 | [OK]|
| Quadratic triangle | `Tri6` | 2 | 6 | [OK]|
| `Quadrilateral` | `Quad4` | 2 | 4 | [OK]|
| Serendipity quad | `Quad8` | 2 | 8 | [OK]|
| `Tetrahedron` | `Tet4` | 3 | 4 | [OK]|
| Quadratic tet | `Tet10` | 3 | 10 | [OK]|
| `Hexahedron` | `Hex8` | 3 | 8 | [OK]|
| Serendipity hex | `Hex20` | 3 | 20 | [OK]|
| `Wedge` (prism) | `Prism6` | 3 | 6 | [OK](type only) |
| `Pyramid` | `Pyramid5` | 3 | 5 | [OK](type only) |
| `Point` | `Point1` | 0 | 1 | [OK]|

### 1.3 Mesh Generators

| MFEM generator | fem-rs equivalent | Status |
|---|---|---|
| `Mesh::MakeCartesian2D()` | `SimplexMesh::unit_square_tri(n)` | [OK]|
| `Mesh::MakeCartesian3D()` | `SimplexMesh::unit_cube_tet(n)` | [OK]| Added in Phase 9 |
| `Mesh::MakePeriodic()` | `SimplexMesh::make_periodic()` | [OK]| Node merging + face removal |
| Reading MFEM format | -- | [N/A] use GMSH instead |
| Reading GMSH `.msh` v4 | `fem_io::read_msh_file()` | [OK]|
| Reading Netgen | `fem_io::read_netgen_vol_file()` | 🔨 Phase 67 (Tet4/Hex8 ASCII 读取基线，支持uniform + mixed；写出仍为Tet4 baseline 为主) |

---

## 2. Reference Elements & Quadrature

### 2.1 Reference Elements

| MFEM class | fem-rs trait/struct | Status |
|---|---|---|
| `FiniteElement` (base) | `ReferenceElement` trait | [OK]|
| `Poly_1D` utility | inline basis in `lagrange/` | [OK]|
| `H1_SegmentElement` P1/P2/P3 | `SegP1`, `SegP2`, `SegP3` | [OK]|
| `H1_TriangleElement` P1/P2/P3 | `TriP1`, `TriP2`, `TriP3` | [OK]|
| `H1_TetrahedronElement` P1/P2/P3 | `TetP1`, `TetP2`, `TetP3` | [OK]|
| `H1_QuadrilateralElement` Q1/Q2 | `QuadQ1`, `QuadQ2` | [OK]|
| `H1_HexahedronElement` | `HexQ1` | [OK]|
| `ND_TriangleElement` (order 1) | `nedelec::TriND1` | [OK]|
| `ND_TriangleElement` (order 2) | `nedelec::TriND2` | [OK]|
| `ND_TetrahedronElement` (order 1) | `nedelec::TetND1` | [OK]|
| `ND_TetrahedronElement` (order 2) | `nedelec::TetND2` | [OK]|
| `RT_TriangleElement` (order 0) | `raviart_thomas::TriRT0` | [OK]|
| `RT_TriangleElement` (order 1) | `raviart_thomas::TriRT1` | [OK]|
| `RT_TetrahedronElement` (order 0) | `raviart_thomas::TetRT0` | [OK]|
| `RT_TetrahedronElement` (order 1) | `raviart_thomas::TetRT1` | [OK]|
| `L2_TriangleElement` | L2Space with P0/P1 | [OK]|

### 2.2 Quadrature Rules

| MFEM class | fem-rs struct | Status |
|---|---|---|
| `IntegrationRule` | `QuadratureRule` | [OK]|
| `IntegrationRules` (table) | `quadrature.rs` look-up table | [OK]|
| Gauss-Legendre 1D (orders 1-10) | `gauss_legendre_1d(order)` | [OK]|
| Gauss-Legendre on triangle | `gauss_triangle(order)` | [OK]|
| Gauss-Legendre on tet | `gauss_tet(order)` + Grundmann-Moller | [OK]|
| Tensor product (quad, hex) | `tensor_gauss(order, dim)` | [OK]|
| Gauss-Lobatto | `gauss_lobatto_1d`, `seg_lobatto_rule`, `quad_lobatto_rule`, `hex_lobatto_rule` | [OK]|

---

## 3. Finite Element Spaces

### 3.1 Collections (Basis Families)

| MFEM collection | Mathematical space | fem-rs struct | Status |
|---|---|---|---|
| `H1_FECollection(p)` | H¹(Ω): C0 scalar Lagrange | `H1Space` (P1–P3) | [OK]|
| `L2_FECollection(p)` | L²(Ω): discontinuous Lagrange | `L2Space` | [OK]|
| `DG_FECollection(p)` | L²(Ω): DG (element-interior only) | `L2Space` | [OK]|
| `ND_FECollection(p)` | H(curl): Nédélec tangential | `HCurlSpace` | [OK]|
| `RT_FECollection(p)` | H(div): Raviart-Thomas normal | `HDivSpace` | [OK]|
| `H1_Trace_FECollection` | H½: traces of H¹ on faces | `H1TraceSpace` | [OK]| P1–P3 boundary trace |
| `NURBS_FECollection` | NURBS isogeometric | `KnotVector`, `BSplineBasis1D`, `NurbsPatch2D`, `NurbsPatch3D`, `NurbsMesh2D/3D` (`fem_element::nurbs`) | [OK] Phase 70 (basis + physical mapping + global IGA assembly verified) |

### 3.2 Finite Element Space (DOF management)

| MFEM method | fem-rs equivalent | Status |
|---|---|---|
| `FiniteElementSpace(mesh, fec)` | `H1Space::new(mesh)` etc. | [OK]|
| `FES::GetNDofs()` | `FESpace::n_dofs()` | [OK]|
| `FES::GetElementDofs()` | `FESpace::element_dofs()` | [OK]|
| `FES::GetBdrElementDofs()` | `boundary_dofs()` | [OK]|
| `FES::GetEssentialTrueDofs()` | `boundary_dofs()` + `apply_dirichlet()` | [OK]|
| `FES::GetTrueDofs()` | `DofPartition::n_owned_dofs` + `global_dof()` | [OK]| Phase 33b |
| `FES::TransferToTrue()` / `Transfer()` | `DofPartition::permute_dof()` / `unpermute_dof()` | [OK]| Phase 34 |
| `DofTransformation` | `FESpace::element_signs()` | [OK]| HCurlSpace/HDivSpace sign convention |
| `FES::GetFE()` | `FESpace::element_type()` | [OK]|

### 3.3 Space Types

| Space | Problem | Status |
|---|---|---|
| H¹ | Electrostatics, heat, elasticity (scalar) | [OK]|
| H(curl) | Maxwell, eddy currents (vector potential) | [OK]|
| H(div) | Darcy flow, mixed Poisson | [OK]|
| L² / DG | Transport, DG methods | [OK]|
| Vector H¹ = [H¹]^d | Elasticity (displacement vector) | [OK]|
| Taylor-Hood P2-P1 | Stokes flow | [OK]Via MixedAssembler + `mfem_ex40` |

---

## 4. Coefficients

MFEM provides a rich coefficient hierarchy for spatially- and
time-varying material properties.  fem-rs uses a trait-based system:
`ScalarCoeff`, `VectorCoeff`, `MatrixCoeff` traits with `f64` as the
default (zero-cost for constants).

| MFEM class | fem-rs | Status |
|---|---|---|
| `ConstantCoefficient(c)` | `f64` (implements `ScalarCoeff`) | [OK]|
| `FunctionCoefficient(f)` | `FnCoeff(\|x\| f(x))` | [OK]|
| `GridFunctionCoefficient` | `GridFunctionCoeff::new(dof_vec)` | [OK]|
| `PWConstCoefficient` | `PWConstCoeff::new([(tag, val), ...])` | [OK]|
| `PWCoefficient` | `PWCoeff::new(default).add_region(tag, coeff)` | [OK]|
| `VectorCoefficient` | `VectorCoeff` trait + `FnVectorCoeff`, `ConstantVectorCoeff` | [OK]|
| `MatrixCoefficient` | `MatrixCoeff` trait + `FnMatrixCoeff`, `ConstantMatrixCoeff`, `ScalarMatrixCoeff` | [OK]|
| `InnerProductCoefficient` | `InnerProductCoeff { a, b }` | [OK]|
| `TransformedCoefficient` | `TransformedCoeff { inner, transform }` | [OK]|

---

## 5. Assembly: Forms & Integrators

### 5.1 Bilinear Forms

| MFEM class | fem-rs equivalent | Status |
|---|---|---|
| `BilinearForm(fes)` | `Assembler::assemble_bilinear(integrators)` | [OK]|
| `BilinearForm::AddDomainIntegrator()` | `assembler.add_domain(integrator)` | [OK]|
| `BilinearForm::AddBoundaryIntegrator()` | `assembler.add_boundary(integrator)` | [OK]|
| `BilinearForm::Assemble()` | `Assembler::assemble_bilinear()` | [OK]|
| `BilinearForm::FormLinearSystem()` | `apply_dirichlet()` | [OK]|
| `BilinearForm::FormSystemMatrix()` | `apply_dirichlet()` variants | [OK]|
| `MixedBilinearForm(trial, test)` | `MixedAssembler` | [OK]|

### 5.2 Linear Forms

| MFEM class | fem-rs equivalent | Status |
|---|---|---|
| `LinearForm(fes)` | `Assembler::assemble_linear(integrators)` | [OK]|
| `LinearForm::AddDomainIntegrator()` | `assembler.add_domain_load(integrator)` | [OK]|
| `LinearForm::AddBndryIntegrator()` | `NeumannIntegrator` | [OK]|
| `LinearForm::Assemble()` | `Assembler::assemble_linear()` | [OK]|

### 5.3 Bilinear Integrators

| MFEM integrator | Bilinear form | fem-rs struct | Status |
|---|---|---|---|
| `DiffusionIntegrator(κ)` | [OK]κ ∇u·∇v dx | `DiffusionIntegrator` | [OK]|
| `MassIntegrator(ρ)` | [OK]ρ u v dx | `MassIntegrator` | [OK]|
| `ConvectionIntegrator(b)` | [OK](b·∇u) v dx | `ConvectionIntegrator` | [OK]|
| `ElasticityIntegrator(λ,μ)` | [OK]σ(u):ε(v) dx | `ElasticityIntegrator` | [OK]|
| `CurlCurlIntegrator(μ)` | [OK]μ (∇×u)·(∇×v) dx | `CurlCurlIntegrator` | [OK]|
| `VectorFEMassIntegrator` | [OK]u·v dx (H(curl)/H(div)) | `VectorMassIntegrator` | [OK]|
| `DivDivIntegrator(κ)` | [OK]κ (∇·u)(∇·v) dx | `DivIntegrator` | [OK]|
| `VectorDiffusionIntegrator` | [OK]κ ∇uᵢ·∇v[OK](vector Laplacian) | `VectorDiffusionIntegrator` | [OK]|
| `BoundaryMassIntegrator` | ∫_Γ α u v ds | `BoundaryMassIntegrator` | [OK]|
| `VectorFEDivergenceIntegrator` | [OK](∇·u) q dx (Darcy/Stokes) | `PressureDivIntegrator` | [OK]|
| `GradDivIntegrator` | [OK](∇·u)(∇·v) dx | `GradDivIntegrator` | [OK]|
| `DGDiffusionIntegrator` | Interior penalty DG diffusion | `DgAssembler::assemble_sip` | [OK]|
| `TransposeIntegrator` | Transposes a bilinear form | `TransposeIntegrator` | [OK]|
| `SumIntegrator` | Sum of integrators | `SumIntegrator` | [OK]|

### 5.4 Linear Integrators

| MFEM integrator | Linear form | fem-rs struct | Status |
|---|---|---|---|
| `DomainLFIntegrator(f)` | [OK]f v dx | `DomainSourceIntegrator` | [OK]|
| `BoundaryLFIntegrator(g)` | ∫_Γ g v ds | `NeumannIntegrator` | [OK]|
| `VectorDomainLFIntegrator` | [OK]**f**·**v** dx | `VectorDomainLFIntegrator` | [OK]|
| `BoundaryNormalLFIntegrator` | ∫_Γ g (n·v) ds | `BoundaryNormalLFIntegrator` | [OK]|
| `VectorFEBoundaryFluxLFIntegrator` | ∫_Γ f (v·n) ds (RT) | `VectorFEBoundaryFluxLFIntegrator` | [OK]|

### 5.5 Assembly Pipeline

| MFEM concept | fem-rs equivalent | Status |
|---|---|---|
| `ElementTransformation` | Jacobian `jac`, `det_j`, `jac_inv_t` | [OK]|
| `Geometry::Type` | `ElementType` enum | [OK]|
| Sparsity pattern | `SparsityPattern` built once | [OK]|
| Parallel assembly | Element loop [OK]ghost DOF AllReduce | [OK]via ChannelBackend |

---

## 6. Linear Algebra

### 6.1 Sparse Matrix

| MFEM class | fem-rs struct | Status |
|---|---|---|
| `SparseMatrix` (CSR) | `CsrMatrix<T>` | [OK]|
| `SparseMatrix::Add(i,j,v)` | `CooMatrix::add(i,j,v)` | [OK]|
| `SparseMatrix::Finalize()` | `CooMatrix::into_csr()` | [OK]|
| `SparseMatrix::Mult(x,y)` | `CsrMatrix::spmv(x,y)` | [OK]|
| `SparseMatrix::MultTranspose()` | `CsrMatrix::transpose()` + spmv | [OK]|
| `SparseMatrix::EliminateRowCol()` | `apply_dirichlet_symmetric()` | [OK]|
| `SparseMatrix::EliminateRow()` | `apply_dirichlet_row_zeroing()` | [OK]|
| `SparseMatrix::GetDiag()` | `CsrMatrix::diagonal()` | [OK]|
| `SparseMatrix::Transpose()` | `CsrMatrix::transpose()` | [OK]|
| `SparseMatrix::Add(A,B)` | `spadd(&A, &B)` | [OK]|
| `SparseMatrix::Mult(A,B)` | SpGEMM (via linger) | [OK]|
| `DenseMatrix` (local dense) | `nalgebra::SMatrix` | [OK]|
| `DenseTensor` | `DenseTensor` (3-D array) | [OK]| Row-major slab access |
| Matrix Market read/write | `fem_io::read_matrix_market` / `write_matrix_market` | [OK]| `.mtx` COO/CSR, real/symmetric/pattern |

### 6.2 Vector

| MFEM class | fem-rs struct | Status |
|---|---|---|
| `Vector` | `Vector<T>` | [OK]|
| `Vector::operator +=` | `Vector::axpy(1.0, x)` | [OK]|
| `Vector::operator *=` | `Vector::scale(a)` | [OK]|
| `Vector::operator * (dot)` | `Vector::dot()` | [OK]|
| `Vector::Norml2()` | `Vector::norm()` | [OK]|
| `Vector::Neg()` | `vector.scale(-1.0)` | [OK]|
| `Vector::SetSubVector()` | `Vector::set_sub_vector()` / `get_sub_vector()` | [OK]| Offset-based slice ops |
| `BlockVector` | `BlockVector` | [OK]|

---

## 7. Solvers & Preconditioners

### 7.1 Iterative Solvers

| MFEM solver | Problem type | fem-rs module | Status |
|---|---|---|---|
| `CGSolver` | SPD: A x = b | `solver` (via linger) | [OK]|
| `PCGSolver` | SPD + preconditioner | `solver` (PCG+Jacobi/ILU0/ILDLt) | [OK]|
| `GMRESSolver(m)` | General: A x = b | `solver` (via linger) | [OK]|
| `FGMRESSolver` | Flexible GMRES | `solve_fgmres` / `solve_fgmres_jacobi` | [OK]|
| `BiCGSTABSolver` | Non-symmetric | `solver` (via linger) | [OK]|
| IDR(s) | Non-symmetric, short-recurrence | `solve_idrs` | [OK]|
| TFQMR | Transpose-free QMR | `solve_tfqmr` | [OK]|
| `MINRESSolver` | Indefinite symmetric | `MinresSolver` | [OK]|
| `SLISolver` | Stationary linear iteration | `solve_jacobi_sli` / `solve_gs_sli` | [OK]|
| `NewtonSolver` | Nonlinear F(x)=0 | `NewtonSolver` | [OK]|
| `UMFPackSolver` | Direct (SuiteSparse) | `solve_sparse_lu` / `solve_sparse_cholesky` / `solve_sparse_ldlt` | [OK]Pure-Rust sparse direct |
| `MUMPSSolver` | Parallel direct | `solve_sparse_mumps` + `linger::MumpsSolver` | 🔨 | MUMPS-compatible API name backed by linger native multifrontal direct solves; replacement path, not external MUMPS FFI |

### 7.2 Preconditioners

| MFEM preconditioner | Type | fem-rs module | Status |
|---|---|---|---|
| `DSmoother` | Jacobi / diagonal scaling | PCG+Jacobi (via linger) | [OK]|
| `GSSmoother` | Gauss-Seidel | `SmootherKind::GaussSeidel` (AMG) | [OK]|
| Chebyshev smoother | Chebyshev polynomial | `SmootherType::Chebyshev` | [OK]|
| `SparseSmoothedProjection` | ILU-based | PCG+ILU0 (via linger) | [OK]|
| Incomplete LDLᵀ | Symmetric indefinite preconditioning | `IldltPrecond` via `solve_pcg_ildlt` / `solve_gmres_ildlt` | [OK]|
| `BlockDiagonalPreconditioner` | Block Jacobi | `BlockDiagonalPrecond` | [OK]|
| `BlockTriangularPreconditioner` | Block triangular | `BlockTriangularPrecond` | [OK]|
| `SchurComplement` | Elimination for saddle point | `SchurComplementSolver` | [OK]|

### 7.3 Solver Convergence Monitors

| MFEM concept | fem-rs equivalent | Status |
|---|---|---|
| `IterativeSolver::SetTol()` | `tol` parameter | [OK]|
| `IterativeSolver::SetMaxIter()` | `max_iter` parameter | [OK]|
| `IterativeSolver::GetFinalNorm()` | `SolverResult::residual_norm` | [OK]|
| `IterativeSolver::GetNumIterations()` | `SolverResult::iterations` | [OK]|
| `IterativeSolver::SetPrintLevel()` | `SolverConfig::print_level` / `PrintLevel` enum | [OK]| Silent/Summary/Iterations/Debug |

---

## 8. Algebraic Multigrid

| MFEM / hypre concept | fem-rs equivalent | Status |
|---|---|---|
| `LOBPCGSolver` | Block eigensolver for SPD | `lobpcg` / `LobpcgSolver` | [OK]|
| Krylov-Schur | Thick-restart Arnoldi eigensolver | `krylov_schur` | [OK]|
| `HypreBoomerAMG` (setup) | `AmgSolver::setup(mat)` -> hierarchy | [OK]|
| `HypreBoomerAMG` (solve) | `AmgSolver::solve(hierarchy, rhs)` | [OK]|
| Strength of connection θ | `AmgParams::theta` | [OK]|
| Ruge-Stüben C/F splitting | RS-AMG (via linger) | [OK]|
| Smoothed aggregation | SA-AMG (via linger) | [OK]|
| Prolongation P | `AmgLevel::p: CsrMatrix` | [OK]|
| Restriction R = Pᵀ | `AmgLevel::r: CsrMatrix` | [OK]|
| Galerkin coarse A_c = R A P | SpGEMM chain | [OK]|
| Pre-smoother (ω-Jacobi) | Jacobi smoother | [OK]|
| Post-smoother | Post-smooth steps | [OK]|
| V-cycle | `CycleType::V` | [OK]|
| W-cycle | `CycleType::W` | [OK]|
| F-cycle | `CycleType::F` | [OK]|
| Max levels | Max levels config | [OK]|
| Coarse-grid direct solve | Dense LU | [OK]|
| Native AMG path | pure-Rust implementation in `vendor/linger` | [OK]|

---

## 9. Parallel Infrastructure

### 9.1 MPI Communicators

| MFEM concept | fem-rs module | Status |
|---|---|---|
| `MPI_Comm` | `ChannelBackend` (in-process threading) | [OK]|
| `MPI_Allreduce` | `Backend::allreduce()` | [OK]|
| `MPI_Allgather` | `Backend::allgather()` | [OK]|
| `MPI_Send/Recv` | `GhostExchange` (alltoallv) | [OK]|

### 9.2 Distributed Mesh

| MFEM class | fem-rs struct | Status |
|---|---|---|
| `ParMesh` | `ThreadLauncher` + partitioned mesh | [OK]|
| METIS partitioning | `MetisPartitioner` (pure-Rust) | [OK]|
| Ghost elements | `GhostExchange` (forward/reverse) | [OK]|
| Global-to-local node map | per-rank DOF mapping | [OK]|

### 9.3 Parallel Linear Algebra

| MFEM / hypre class | fem-rs struct | Status |
|---|---|---|
| `HypreParMatrix` | `ParCsrMatrix` (diag+offd blocks) | [OK]Thread + MPI backends |
| `HypreParVector` | `ParVector` (owned+ghost layout) | [OK]|
| `HypreParMatrix::Mult()` | `ParCsrMatrix::spmv()` via ghost exchange | [OK]|
| `HypreParMatrix::GetDiag()` | `ParCsrMatrix::diag` | [OK]|
| `HypreParMatrix::GetOffd()` | `ParCsrMatrix::offd` | [OK]|
| `ParFiniteElementSpace` | `ParallelFESpace<S>` (P1+P2) | [OK]|
| `ParBilinearForm::Assemble()` | `ParAssembler::assemble_bilinear()` | [OK]|
| `ParLinearForm::Assemble()` | `ParAssembler::assemble_linear()` | [OK]|
| `HypreSolver` (PCG+Jacobi) | `par_solve_pcg_jacobi()` | [OK]|
| `HypreBoomerAMG` | `ParAmgHierarchy` (local smoothed aggregation) | [OK]|
| `par_solve_pcg_amg()` | PCG + AMG V-cycle preconditioner | [OK]|
| `MPI_Comm_split` | `Comm::split(color, key)` | [OK]|
| Streaming mesh distribution | `partition_simplex_streaming()` | [OK]Phase 37 |
| WASM multi-Worker MPI | `WorkerLauncher::spawn_async()` + `jsmpi_main` | [OK]Phase 37 |
| Binary sub-mesh serde | `mesh_serde::encode/decode_submesh()` | [OK]Phase 37 |

---

## 10. I/O & Visualization

### 10.1 Mesh I/O

| MFEM format / method | fem-rs | Status |
|---|---|---|
| MFEM native mesh format (read/write) | -- | [N/A] use GMSH |
| GMSH `.msh` v2 ASCII (read) | `fem_io::read_msh_file()` | [OK]|
| GMSH `.msh` v4.1 ASCII (read) | `fem_io::read_msh_file()` | [OK]|
| GMSH `.msh` v4.1 binary (read) | `fem_io::read_msh_file()` | [OK]|
| Netgen `.vol` (read/write) | `read_netgen_vol_file()` / `write_netgen_vol_file()` | 🔨 | 读取：Tet4/Hex8 ASCII baseline（支持mixed）；写出：Tet4 ASCII baseline |
| Abaqus `.inp` (read) | `read_abaqus_inp_file()` | 🔨 | C3D4/C3D8 baseline（支持uniform + mixed[OK]|
| VTK `.vtu` legacy ASCII (write) | `write_vtk_scalar()` | [OK]|
| VTK `.vtu` XML binary (write) | `write_vtu()` (XML ASCII) | [OK]|
| HDF5 / XDMF (read/write) | `fem-io-hdf5-parallel` (feature-gated) | 🔨 |
| ParaView GLVis socket | -- | [N/A] out of scope |

### 10.2 Solution I/O

| MFEM concept | fem-rs | Status |
|---|---|---|
| `GridFunction::Save()` | VTK point data | [OK]scalar + vector |
| `GridFunction::Load()` | `read_vtu_point_data()` | [OK]| ASCII VTU reader |
| Restart files | HDF5 checkpoint schema + restart reads | 🔨 |

---

## 11. Grid Functions & Post-processing

| MFEM class / method | fem-rs equivalent | Status |
|---|---|---|
| `GridFunction(fes)` | `GridFunction<S>` (wraps DOF vec + space ref) | [OK]|
| `GridFunction::ProjectCoefficient()` | `FESpace::interpolate(f)` | [OK]|
| `GridFunction::ComputeL2Error()` | `GridFunction::compute_l2_error()` | [OK]|
| `GridFunction::ComputeH1Error()` | `GridFunction::compute_h1_error()` / `compute_h1_full_error()` | [OK]|
| `GridFunction::GetGradient()` | `postprocess::compute_element_gradients()` / `recover_gradient_nodal()` | [OK]|
| `GridFunction::GetCurl()` | `postprocess::compute_element_curl()` | [OK]|
| `GridFunction::GetDivergence()` | `postprocess::compute_element_divergence()` | [OK]|
| `ZZErrorEstimator` (Zienkiewicz-Zhu) | `zz_error_estimator()` | [OK]|
| `KellyErrorEstimator` | `kelly_estimator()` | [OK]| Face-jump based error indicator |
| `DiscreteLinearOperator` | Gradient, curl, div operators | [OK]`DiscreteLinearOperator::gradient/curl_2d/divergence` |

---

## 12. MFEM Examples <-> fem-rs Milestones

Each MFEM example defines a target milestone for fem-rs feature completeness.

### Tier 1 [OK]Core Capability (Phases 6[OK])

| MFEM example | PDE | FEM space | BCs | fem-rs milestone |
|---|---|---|---|---|
| **ex1** | −∇²u = 1, u=0 on ∂[OK]| H¹ P1/P2 | Dirichlet | [OK]`mfem_ex1_poisson` O(h²) |
| **ex2** | −∇²u = f, mixed BCs | H¹ P1/P2 | Dirichlet + Neumann | [OK]`mfem_ex2_elasticity` |
| **ex3** (scalar) | −∇²u + αu = f (reaction-diffusion) | H¹ P1 | Dirichlet | [OK]Phase 6: `MassIntegrator` |
| **ex13** | −∇·(ε∇[OK] = 0, elasticity | H¹ vector | Mixed | Phase 6: `ElasticityIntegrator` |
| **pex1** | Parallel Poisson | H¹ + MPI | Dirichlet | [OK]`mfem_pex1_poisson` (contiguous/METIS, streaming) |

### Tier 2 [OK]Mixed & H(curl)/H(div) (Phase 6+)

| MFEM example | PDE | FEM space | fem-rs milestone |
|---|---|---|---|
| **ex3** (curl) | ∇×∇×**u** + **u** = **f** (Maxwell) | H(curl) Nédélec | [OK]`mfem_ex3` O(h) |
| **ex4** | −∇·(**u**) = f, **u** = −κ∇p (Darcy) | H(div) RT + L² | [OK]`mfem_ex4_darcy` H(div) RT0 grad-div MINRES |
| **ex5** | Saddle-point Darcy/Stokes | H(div) × L² | [OK]`mfem_ex5_mixed_darcy` block PGMRES |
| **ex22** | Time-harmonic Maxwell (complex coeff.) | H(curl) | Phase 7+ |

### Tier 3 [OK]Time Integration (Phase 7+)

| MFEM example | PDE | Time method | fem-rs milestone |
|---|---|---|---|
| **ex9** (heat) | ∂u/∂t − ∇²u = 0 | BDF1 / Crank-Nicolson | [OK]`mfem_ex10_heat_equation` SDIRK-2 |
| **ex10** (wave) | ∂²u/∂t² − ∇²u = 0 | Leapfrog / Newmark | [OK]`mfem_ex10_wave_equation` Newmark-β |
| **ex14** (DG heat) | ∂u/∂t − ∇²u + b·∇u = 0 | Explicit RK + DG | [OK]`mfem_ex9_dg_advection` SIP-DG O(h²) |
| **ex16** (elastodynamics) | ρ ∂[OK]*u**/∂t² = ∇·[OK]| Generalized-α | [OK]`mfem_ex16_nonlinear_heat` Newton |

### Tier 4 [OK]Nonlinear & AMR (Phase 7+)

| MFEM example | Problem | fem-rs milestone |
|---|---|---|
| **ex4** (nonlinear) | −Δu + exp(u) = 0 | [OK]`NewtonSolver` |
| **ex6** | AMR Poisson with ZZ estimator | [OK]`refine_marked()`, `ZZErrorEstimator` |
| **ex15** | DG advection with AMR | [OK]`mfem_ex15_dg_amr` P1 + ZZ + Dörfler + refinement |
| **ex19** | Incompressible Navier-Stokes | [OK]`mfem_ex19` (Kovasznay Re=40, Oseen/Picard) |

### Tier 5 [OK]HPC & Parallel (Phase 10)

| MFEM example | Problem | fem-rs milestone |
|---|---|---|
| **pex1** | Parallel Poisson (Poisson) | [OK]`mfem_pex1_poisson` (contiguous/METIS + streaming) |
| **pex2** | Parallel mixed Poisson | [OK]`mfem_pex2_mixed_darcy` |
| **pex3** | Parallel Maxwell (H(curl)) | [OK]`mfem_pex3_maxwell` |
| **pex5** | Parallel Darcy | [OK]`mfem_pex5_darcy` |

---

## 13. Key Design Differences

| Aspect | MFEM (C++) | fem-rs (Rust) | Rationale |
|---|---|---|---|
| **Polymorphism** | Virtual classes + inheritance | Traits + generics (zero-cost) | No vtable overhead in inner loop |
| **Index types** | `int` (32-bit signed) | `NodeId = u32` etc. | Half memory; explicit casting |
| **Parallel model** | Always-on `ParMesh`; MPI implicit | Feature-gated `fem-parallel` crate | Same binary works without MPI |
| **Web target** | emscripten (experimental) | `fem-wasm` crate (wasm-bindgen) | First-class JS interop |
| **AMG default** | Ruge-Stüben (classical) | Smoothed Aggregation | Better performance on vector problems |
| **Quadrature** | Hard-coded tables | Generated tables in `quadrature.rs` | Reproducible, testable |
| **Coefficient API** | Polymorphic `Coefficient*` objects | `ScalarCoeff`/`VectorCoeff`/`MatrixCoeff` traits; `f64` default | Zero-cost constants, composable, trait-based |
| **Memory layout** | Column-major `DenseMatrix` | Row-major element buffers; nalgebra for Jacobians | Cache-friendly assembly |
| **Error handling** | Exceptions / abort | `FemResult<T>` everywhere | Propagate, never panic in library |
| **BC application** | `FormLinearSystem()` (symmetric elim.) | `solve_dirichlet_reduced()` (reduced system) | Avoids scale artefacts with small ε |
| **Grid function** | `GridFunction` owns DOF vector + FES ref | `Vec<f64>` + separate `FESpace` ref | Separation of concerns |

---

## Quick Reference: Phase [OK]Features

| Phase | Crates | MFEM equivalents unlocked | Status |
|---|---|---|---|
| 0 | workspace | [OK]| [OK]|
| 1 | `core` | Index types, `FemError`, scalar traits | [OK]|
| 2 | `mesh` | `Mesh`, element types, mesh generators | [OK]|
| 3 | `element` | `FiniteElement`, `IntegrationRule`, Lagrange P1–P2 | [OK]|
| 4 | `linalg` | `SparseMatrix`, `Vector`, COO→CSR assembly | [OK]|
| 5 | `space` | `FiniteElementSpace`, H1/L2, DOF manager | [OK]|
| 6 | `assembly` | `BilinearForm`, `LinearForm`, standard integrators | [OK]|
| 7 | `solver` | `CGSolver`, `GMRESSolver`, ILU(0), direct | [OK]|
| 8 | `amg` | SA-AMG + RS-AMG (native via linger) | [OK]|
| 9 | `io` | VTK XML, GMSH v4 reader | [OK]|
| 10 | `parallel` | Thread-based parallel, ghost exchange | [OK]|
| 11 | `wasm` | Browser-side FEM solver via JS API | [OK]|
| 12 | `element` | Nedelec ND1, Raviart-Thomas RT0 | [OK]|
| 13 | `space`+`assembly` | VectorH1Space, BlockMatrix, MixedAssembler, Elasticity | [OK]|
| 14 | `assembly` | SIP-DG (interior penalty) | [OK]|
| 15 | `solver`+`assembly` | NonlinearForm, NewtonSolver | [OK]|
| 16 | `solver` | ODE: ForwardEuler, RK4, RK45, ImplicitEuler, SDIRK-2, BDF-2 | [OK]|
| 17 | `mesh` | AMR: red refinement, ZZ estimator, Dörfler marking | [OK]|
| 18 | `parallel` | METIS k-way partitioning (pure-Rust) | [OK]|
| 19 | `mesh`+`space` | CurvedMesh (P2 isoparametric) | [OK]|
| 20 | `solver` | LOBPCG eigenvalue solver | [OK]|
| 21 | `solver`+`linalg` | BlockSystem, SchurComplement, MINRES | [OK]|
| 22 | `assembly` (`reed`) | Partial assembly: PA mass/diffusion, matrix-free (rem-rs/reed) | [OK]|
| 23 | `space` | HCurlSpace (Nédélec ND1), HDivSpace (RT0), element_signs | [OK]|
| 24 | `assembly` | VectorAssembler, CurlCurlIntegrator, VectorMassIntegrator | [OK]|
| 25 | `assembly`+`solver` | DG-SIP face normals fix, SchurComplement PGMRES, MINRES rewrite, TriND1 fix; all 8 MFEM-style examples verified | [OK]|
| 26 | `assembly` | Coefficient system: ScalarCoeff/VectorCoeff/MatrixCoeff traits, PWConstCoeff, PWCoeff, GridFunctionCoeff, composition | [OK]|
| 27 | `assembly` | Convection, VectorDiffusion, BoundaryMass, GradDiv, Transpose, Sum integrators; VectorDomainLF, BoundaryNormalLF | [OK]|
| 28 | `assembly` | GridFunction wrapper, L²/H¹ error, element gradients/curl/div, nodal gradient recovery | [OK]|
| 29 | `assembly` | DiscreteLinearOperator: gradient, curl_2d, divergence as sparse matrices; de Rham exact sequence | [OK]|
| 30 | `solver` | Newmark-β time integrator; mfem_ex10_wave_equation example | [OK]|
| 31 | `element` | Gauss-Lobatto quadrature (seg, quad, hex) | [OK]|
| 32 | `examples` | mfem_ex4_darcy (H(div) RT0), mfem_ex15_dg_amr (P1 + ZZ + Dörfler) | [OK]|
| 33a-e | `parallel` | jsmpi backend, DofPartition, ParVector, ParCsrMatrix, ParAssembler, par_solve_pcg_jacobi, pex1 | [OK]|
| 34 | `parallel` | P2 parallel spaces (DofPartition::from_dof_manager, edge DOF ownership, auto-permute) | [OK]|
| 35 | `parallel` | Parallel AMG (ParAmgHierarchy, smoothed aggregation, par_solve_pcg_amg) | [OK]|
| 36 | `parallel` | Comm::split sub-communicators | [OK]|
| 37 | `parallel`+`wasm` | WASM multi-Worker (spawn_async, jsmpi_main), streaming mesh partition (partition_simplex_streaming), binary mesh serde | [OK]|
| 38 | `parallel` | METIS streaming partition (partition_simplex_metis_streaming), generalized submesh extractor, pex1 CLI flags | [OK]|
| 38b | `io` | GMSH v2 ASCII + v4.1 binary reader (unified `read_msh_file()` entry point) | [OK]|
| 39 | `parallel`+`examples` | pex2 (mixed Poisson), pex3 (Maxwell), pex5 (Darcy) parallel examples | [OK]|
| 39b | `amg` | Chebyshev smoother (`SmootherType::Chebyshev`), F-cycle (`CycleType::F`) | [OK]|
| 40 | `examples`+`assembly` | Taylor-Hood P2-P1 Stokes (`mfem_ex40` lid-driven cavity) | [OK]|
| 42a | `mesh`+`space`+`io` | Mixed element mesh infrastructure (per-element types, variable DofManager, GMSH mixed read) | [OK]|
| 44 | `assembly`+`examples` | VectorConvectionIntegrator + Navier-Stokes Oseen/Picard (`mfem_ex19`, Kovasznay Re=40) | [OK]|
| 42b | `assembly` | Quad4/Hex8 isoparametric Jacobian, `unit_square_quad`, Q1 Poisson verified | [OK]|
| 45 | `wasm`+`e2e` | Browser E2E test: WASM Poisson solver verified via Playwright/Chromium | [OK]|
| 46 | `mesh`+`linalg`+`solver`+`space`+`io` | Backlog: bounding_box, periodic mesh, DenseTensor, SLI, H1Trace, VTK reader, PrintLevel | [OK]|
| 47 | `mesh`+`space` | NCMesh: Tri3/Tet4 nonconforming refine + hanging constraints + `NCState`/`NCState3D` multi-level + P2 prolongation | [OK]|
| 48 | `element`+`space`+`assembly`+`solver`+`io` | linger update: sparse direct solvers (SparseLu/Cholesky/LDLt), IDR(s), TFQMR, ILDLt precond, KrylovSchur eigen, Matrix Market I/O; higher-order elements: TriP3, TetP2, TetP3, QuadQ2; H1TraceSpace P2/P3; Grundmann-Moller quadrature fix | [OK]|
| 49 | `element`+`space`+`assembly` | TriND2/TetND2 (Nédélec-I order 2); TriRT1/TetRT1 (Raviart-Thomas order 1); HCurlSpace/HDivSpace multi-order support; VectorAssembler factory updated | [OK]|

---

## Remaining Items Summary (🔲 Planned · 🔨 Partial)

### Mesh
| Item | Status | Priority |
|------|--------|----------|
| Mixed element meshes (Tri+Quad, Tet+Hex) | [OK]| ~~Medium~~ Done |
| NCMesh (non-conforming, hanging nodes) | [OK]| ~~Low~~ Done |
| `bdr_attributes` dedup utility | [OK]| ~~Low~~ Done |
| `ElementTransformation` type | [OK]| ~~Low~~ Done |
| `GetBoundingBox()` | [OK]| ~~Low~~ Done |
| Periodic mesh generation | [OK]| ~~Low~~ Done |

### I/O
| Item | Status | Priority |
|------|--------|----------|
| ~~GMSH v4.1 binary reader~~ | [OK]| ~~High~~ Done |
| ~~GMSH v2 reader~~ | [OK]| ~~Medium~~ Done |
| HDF5/XDMF parallel I/O | 🔨 | Medium |
| Netgen `.vol` reader | 🔨 (Tet4/Hex8 ASCII baseline，支持mixed；写出与更多 section 保真待补[OK] | Low |
| Abaqus `.inp` reader | 🔨 (C3D4/C3D8 baseline，支持mixed；更[OK]section/tag 保真待补[OK] | Low |
| `GridFunction::Load()` | [OK]| ~~Low~~ Done |
| Restart files (checkpoint) | 🔨 | Low |

### Solvers
| Item | Status | Priority |
|------|--------|----------|
| Chebyshev smoother (AMG) | [OK]| ~~Medium~~ Done |
| SLISolver (stationary iteration) | [OK]| ~~Low~~ Done |
| AMG F-cycle | [OK]| ~~Low~~ Done |
| Native AMG path | [OK](pure-Rust in `vendor/linger`) | Low |

### Spaces & Post-processing
| Item | Status | Priority |
|------|--------|----------|
| H1_Trace_FECollection | [OK]| ~~Low~~ Done |
| Taylor-Hood P2-P1 | Stokes flow | [OK]`mfem_ex40` (lid-driven cavity) |
| Kelly error estimator | [OK]| ~~Low~~ Done |
| `DenseTensor` | [OK]| ~~Low~~ Done |
| `SetSubVector` slice assignment | [OK]| ~~Low~~ Done |

### Parallel Examples
| Item | Status | Priority |
|------|--------|----------|
| pex2 (parallel mixed Poisson) | [OK]| ~~Medium~~ Done |
| pex3 (parallel Maxwell) | [OK]| ~~Medium~~ Done |
| pex5 (parallel Darcy) | [OK]| ~~Medium~~ Done |
| ex19 (Navier-Stokes) | [OK]| ~~Medium~~ Done |
| Browser E2E (WASM) | [OK]| ~~Medium~~ Done |

---

## Recommended Roadmap (Phase 39+)

Based on the completed 38 phases and remaining gaps, here is a recommended
prioritized roadmap for continued development.

### Phase 39 [OK]Parallel Examples Expansion (pex2 / pex3 / pex5) [OK]
> **Completed** -- validates parallel infrastructure across all FE spaces

| Task | Space | Status |
|------|-------|--------|
| `mfem_pex2_mixed_darcy` | H(div) RT0 × L² | [OK]|
| `mfem_pex3_maxwell` | H(curl) ND1 | [OK]|
| `mfem_pex5_darcy` | H(div) × L² saddle-point | [OK]|

### Phase 39b [OK]Chebyshev Smoother + AMG F-cycle [OK]
> **Completed** -- smoother quality directly impacts AMG convergence

- [OK]Chebyshev polynomial smoother (degree 2[OK]) as `SmootherType::Chebyshev`
- [OK]Eigenvalue estimate via spectral radius bound (λ_max)
- [OK]F-cycle: `CycleType::F` (V on first coarse visit, W after)
- [OK]Tests: Chebyshev, F-cycle, Chebyshev+F-cycle combinations

### Phase 40 [OK]Taylor-Hood P2-P1 Stokes Example [OK]
> **Completed** -- demonstrates mixed FEM at production quality

- [OK]`mfem_ex40` example: lid-driven cavity on [0,1]²
- [OK]P2 velocity + P1 pressure via `MixedAssembler`
- [OK]Block saddle-point solver (SchurComplementSolver with GMRES)
- [OK]Verified convergence at n=8,16,32; divergence-free to solver tolerance

### Phase 42 [OK]Mixed Element Meshes (42a [OK] 42b [OK]
> **Completed** -- data structures, I/O, and assembly all done

- [OK]Per-element `ElementType` and CSR-like offset arrays in `SimplexMesh`
- [OK]Variable-stride `DofManager` for P1 on mixed meshes
- [OK]GMSH reader preserves mixed element types (Tri+Quad, Tet+Hex)
- [OK]Isoparametric Jacobian for Quad4/Hex8 in assembler (bilinear/trilinear mapping)
- [OK]`unit_square_quad(n)` mesh generator + Q1 Poisson convergence verified

### Phase 43 [OK]HDF5/XDMF Parallel I/O
> **Priority: Medium** [OK]needed for large-scale checkpointing

- [x] 新增独立 crate：`fem-io-hdf5-parallel`（feature-gated `hdf5`[OK]
- [x] 写入：rank-partition checkpoint（`/steps/step_xxxxxxxx/partitions/rank_xxxxxx/*`[OK]
- [x] 读取：按 step / latest  rank-local restart 读取
- [x] 全局场拼装：`materialize_global_field_f64()`（供可视化）
- [x] XDMF sidecar：`write_xdmf_polyvertex_scalar_sidecar()`
- [x] XDMF time-series：`write_xdmf_polyvertex_scalar_timeseries_sidecar()`
- [x] 示例：`mfem_ex43_hdf5_checkpoint.rs`（无 HDF5 环境时优雅降级）
- [x] checkpoint 完整性校验：`validate_checkpoint_layout()`
- [x] MPI backend 已升级为 MPI 协同路径（rank 写入 + direct hyperslab 全局写入路径，保[OK]root 全局物化兼容兜底[OK]
- [x] direct hyperslab 读路径：`read_global_field_f64()` + `read_global_field_slice_f64()`（全局整场/切片读取[OK]
- [x] 并行 mesh+field bundle checkpoint schema（`CheckpointBundleF64` + `CheckpointMeshMeta` baseline（

### Phase 44 [OK]Navier-Stokes (Kovasznay flow) [OK]
> **Completed** -- flagship nonlinear PDE example

- [OK]`VectorConvectionIntegrator`: `[OK](w·∇u · v dx` for vector fields
- [OK]Oseen linearization with Picard iteration
- [OK]`mfem_ex19` example: Kovasznay flow benchmark (Re=40)
- [OK]Taylor-Hood P2/P1 discretization (reuses Phase 40 infrastructure)
- [OK]Converges in ~16[OK]0 Picard iterations; velocity error decreases with h-refinement

### Phase 45 [OK]Browser E2E (WASM) [OK]
> **Completed** -- validates the full browser pipeline

- [OK]Playwright/Chromium E2E test (`crates/wasm/e2e/`)
- [OK]WASM Poisson solver: assemble [OK]solve -> verify in browser
- [OK]Solution validated against analytical max (0.0737 for −Δu=1)

### Phase 46 [OK]Backlog Cleanup [OK]
> **Completed** -- 9 remaining items resolved

- [OK]`SimplexMesh::bounding_box()` [OK]axis-aligned bounding box (2-D / 3-D)
- [OK]`SimplexMesh::unique_boundary_tags()` [OK]sorted/deduped boundary tag set
- [OK]`SimplexMesh::make_periodic()` [OK]node merging for periodic BCs
- [OK]`DenseTensor` [OK]3-D row-major tensor with slab access
- [OK]`solve_jacobi_sli()` / `solve_gs_sli()` [OK]Jacobi/GS stationary iteration
- [OK]`H1TraceSpace` [OK]H½ trace of H¹ on boundary faces (P1)
- [OK]`read_vtu_point_data()` [OK]VTK `.vtu` ASCII reader for solution loading
- [OK]`PrintLevel` enum [OK]structured solver verbosity (Silent/Summary/Iterations/Debug)
- [OK]`kelly_estimator()` was already implemented [OK]marked in MFEM_MAPPING
- [OK]`SetSubVector` / `GetSubVector` were already implemented [OK]marked in MFEM_MAPPING

### Phase 47 [OK]NCMesh (Non-Conforming Mesh / Hanging Nodes) [OK]
> **Completed** -- 2-D Tri3 + 3-D Tet4 non-conforming refinement with multi-level state tracking

#### 2-D (Tri3) Hanging Edge Constraints
- [OK]`refine_nonconforming()` [OK]red-refines only marked elements, no propagation
- [OK]`HangingNodeConstraint` detection [OK]identifies midpoints on coarse/fine edges
- [OK]`apply_hanging_constraints()` [OK]P^T K P static condensation via COO rebuild
- [OK]`recover_hanging_values()` [OK]post-solve interpolation for constrained DOFs
- [OK]`NCState` multi-level constraint tracking [OK]carries and resolves hanging constraints across successive NC refinements
- [OK]`prolongate_p2_hanging()` [OK]P2 hanging-node prolongation by coarse P2 field evaluation at fine DOF coordinates
- [OK]`mfem_ex15_dg_amr --nc` -- demonstrates single-level NC AMR with error reduction

#### 3-D (Tet4) Hanging Face Constraints
- [OK]`HangingFaceConstraint` struct [OK]records hanging coarse faces and representative midpoint nodes
- [OK]`refine_nonconforming_3d(mesh, marked)` [OK]red-refines Tet4 elements into 8 children using edge midpoints
- [OK]`local_faces_tet()` [OK]helper returns 4 triangular face local indices for Tet4
- [OK]`face_key_3d()` [OK]canonical face key (sorted triplet) for face uniqueness
- [OK]Hanging-face detection [OK]detects refined/coarse Tet4 face mismatch and emits hanging edge constraints
- [OK]`NCState3D` multi-level tracking [OK]carries active edge midpoints and rebuilds constraints across levels
- [OK]Boundary face reconstruction [OK]preserves and refines Tri3 boundary faces in 3-D refinement
- [OK]Unit tests [OK]`tet4_nonconforming_refine_single_element()`, `tet4_nonconforming_refine_with_neighbor()`, `ncstate3d_two_level_refine()`

### Backlog (Low Priority)
| Item | Phase | Notes |
|------|-------|-------|
| Native AMG path | pure-Rust capability roadmap | Owned by `vendor/linger` |
| Abaqus/Netgen format扩展（混合单元、更多section/tag保真[partial] | TBD | Additional mesh import formats |
| HDF5/XDMF I/O | TBD | Large-scale checkpointing |
| Restart files | TBD | Requires HDF5 |
| Tet4 NC AMR example | [OK]| ~~TBD~~ Done (`mfem_ex15_tet_nc_amr`, supports `--solve`) |

### Decision Log (2026-04-13)

- GPU backend is tracked as a cross-subproject roadmap item:
   - `vendor/linger`: backend-neutral kernel interfaces and numeric primitive contracts.
   - `rem-rs/reed`: GPU backend implementation and CEED-style operator/resource mapping.
   - `vendor/jsmpi`: browser-side multi-rank transport/runtime for wasm deployments.
- External solver delivery is coordinated across subprojects:
   - `vendor/linger`: pure-Rust native solver lifecycle; `mumps`/`mkl` are compatibility contracts backed by native linger direct solves.
   - `rem-rs/reed`: operator/export bridge and backend selection wiring.
   - `vendor/jsmpi`: wasm/browser runtime constraints for distributed execution path.
- Current `linger` gaps to track under this ownership:
   - Distributed-memory path is still missing (`mpi` feature is placeholder in `vendor/linger/Cargo.toml`).
   - Native AMG advanced options: AMS/ADS baseline is already available in `vendor/linger`; AIR baseline strategy is landed (`CoarsenStrategy::Air` + diagonal-`A_ff` AIR restriction) with nonsymmetric regression coverage (`amg_air_gmres_nonsymmetric_convdiff_1d`), while high-scale hardening remains pending.
   - Direct-compatibility hooks: `mumps` / `mkl` 均具备可用 baseline（native multifrontal-backed, factor reuse + multi-RHS）；二者均由 linger 原生直接法承载，不以外部 FFI/distributed 接入为目标。
   - AMG options are currently RS/SA + V/W/F/K-cycle baseline, with room for high-scale robustness hardening.
   - GPU execution backend is missing in `linger` core (implementation track owned by `rem-rs/reed`).
   - Matrix Market complex field I/O is not yet supported (`vendor/linger/src/sparse/mmio.rs`).

### Cross-Subproject Improvement Plan (2026-Q2 to 2026-Q4)

> Scope: coordinated delivery across `vendor/linger`, `rem-rs/reed`, and `vendor/jsmpi`.

| Stage | Window | linger | reed | jsmpi | Exit Criteria |
|---|---|---|---|---|---|
| C1 Foundation | Q2 (2-4 weeks) | External solver abstraction, error adapter, feature-gated fallback | Stable operator/export bridge API to linger | Browser/wasm backend capability policy (supported vs fallback) | API boundary frozen; default build unchanged |
| C2 Solver Hardening M1/M2 | Q2-Q3 | native AMG baseline hardening (AIR + AMS/ADS) + `mumps` direct compatibility hardening | Builder wiring for backend selection in FEM solve paths | wasm path reports deterministic fallback when native direct compatibility backends unavailable | Poisson SPD integration tests pass for enabled backends |
| C3 GPU First Usable Path | Q3 | Backend-neutral kernel interface + CPU reference kernels | GPU backend implementation + CEED-style object mapping + one end-to-end example | Browser multi-rank transport constraints documented for GPU+wasm modes | One representative solve path runs CPU/GPU with same app API |
| C4 Portfolio Completion | Q4 | native solver stack scale/perf hardening; CI matrix hooks | cross-backend regression tests in FEM pipelines | Browser smoke tests and fallback matrix by feature | CI passes on feature matrix; docs and examples complete |

#### Work Packages

- [x] WP1: Interface freeze for cross-project backend contracts
- [ ] WP2: native AMG AIR + AMS/ADS hardening（`linger` 的 AMS/ADS 和 AIR baseline 已可用，仍需分布式/高规模能力补齐）
- [x] WP3: `mumps` + `mkl` usable with factor reuse and multi-RHS（baseline：`linger::{MumpsSolver, MklSolver}` + `solve_sparse_{mumps,mkl}`；二者均为 linger 原生直接法的兼容入口）
- [ ] WP4: GPU baseline delivery in `reed` (with `linger` backend-neutral kernel contracts)
- [ ] WP5: native solver stack CI feature matrix + scale-hardening in `linger`
- [ ] WP6: `jsmpi` browser/wasm fallback and smoke-test closure

WP1 kickoff artifact merged: `C1_BACKEND_CONTRACT_FREEZE.md` (v0.1).

Current baseline progress (2026-04-13):
- Added canonical backend-resource smoke coverage in `fem-assembly` (`--features reed`) for `/solver/mumps`, `/solver/mkl` deterministic resolution/report path.
- Added CI gate `.github/workflows/alignment-smoke.yml` to run targeted smoke tests for:
   - complex coefficient traits (`fem-assembly`)
   - named attribute set baseline (`fem-mesh`)
   - canonical backend resource contract (`fem-assembly` + `reed`)
- Added CI gate `.github/workflows/backend-feature-matrix.yml` to validate `rem-rs/reed` backend contract tests across feature profiles:
   - baseline (`--no-default-features`)
   - `mumps`, `mkl`

#### Coordination Rules

- One feature branch per stage, three subprojects use the same stage tag (`C1`/`C2`/`C3`/`C4`).
- No app-level API churn in `fem-rs` during stages; changes are behind feature flags.
- A stage is accepted only when all three subprojects satisfy the stage exit criteria.

---

## MFEM v4.9 Gap Analysis (2026-04-13)

> 对比基准：MFEM v4.9[OK]025-12-11）[OK]最新版本[OK]
> 以下差距按优先级排列，高优先级直接影响物理覆盖面，低优先级是工程完善项[OK]

### 差距汇总表

| 能力领域 | MFEM v4.9 | fem-rs | 差距等级 | 对应 Phase |
|---|---|---|---|---|
| 复数[OK]FEM | -- ex22/ex25/DPG | [OK]基线已实现（2×2 实块[OK] | 🟡 [OK] | 55 |
| IMEX 时间积分 | [OK]ex41 | [OK]基线已实现（Euler/SSP2/RK3/ARK3[OK] | 🟡 [OK] | 56 |
| AMR 反细[OK](Derefinement) | [OK]ex15 | [OK]基线已实现（single-level rollback[OK] | 🟡 [OK] | 57 |
| 几何多重网格 / LOR 预条件器 | [OK]ex26 | ✅ LOR + GeomMG 基线 7/7 测试通过 | 🟡 [OK] | 58 |
| SubMesh 子域传输 | [OK]ex34/ex35 | [OK]基线已实[OK] | 🟡 [OK] | 59 |
| DG 弹性力[OK]| [OK] ex17 | [OK]基线已实[OK] | 🟡 [OK] | 60 |
| DG 可压[OK]Euler 方程 | [OK]ex18 | [OK]1D 基线已实[OK] | 🟡 [OK] | 60 |
| 辛时间积[OK](Symplectic) | [OK]ex20 | [OK]已实[OK] | 🟡 [OK] | 61 |
| 受限 H(curl) 空间 (1D/2D embedded) | [OK]ex31/ex32 | [OK]基线已实现；ex31/ex32 均已补充制造解一阶收敛回[OK] | 🟡 [OK] | 62 |
| PML 完美匹配[OK]| [OK] ex25 | 🔨 标量+各向异性张量基线（ex25 已加入可量化反射指标与强度回归） | 🟡 [OK] | 55+63 |
| 静态凝[OK]/ 杂化 | [OK]ex4/ex8/hybr | 🔨 代数静态凝聚基线（`mfem_ex8_hybridization`，基[OK]hanging constraints）；混合/杂化 FEM 内核待补[OK]| 🟢 [OK] | TBD |
| 分数[OK]Laplacian | [OK]ex33 | ✅ `mfem_ex33_fractional_laplacian` dense spectral + dense rational + **sparse rational**（Jacobi-PCG 逐移位稀疏求解）；7/7 测试通过 | 🟢 [OK] | TBD |
| 障碍问题 / 变分不等[OK]| [OK] ex36 | ✅ `mfem_ex36_obstacle` PDAS + semismooth Newton（SSN）两路求解；7/7 测试通过[OK]| 🟢 [OK] | TBD |
| 拓扑优化 | [OK]ex37 | ✅ `mfem_ex37_topology_optimization` 标量 SIMP + 平面应变弹性 SIMP（B/D 矩阵 + penalty 法 BC）；7/7 测试通过[OK]| 🟢 [OK] | TBD |
| 截断积分 / 浸没边界 | [OK]ex38 | ✅ `mfem_ex38_immersed_boundary` cut-cell Nitsche + 通用 level-set（Circle / Halfspace）；线性 ψ 弦段 + centroid guard 避免双计；7/7 测试通过[OK]| 🟢 [OK] | TBD |
| 命名属性集 | [OK]ex39 | ✅ named tag registry + mesh/submesh named selection + GMSH `PhysicalNames` bridge；集合运算（union/intersection/difference）；6/6 测试通过 | 🟢 [OK] | TBD |
| 准 ALE 动网格 | ex45/ex46 | ✅ `mfem_ex45_moving_mesh_ale` 4/4 测试（积分守恒、零振幅精确传递、网格有效性、多步稳定性）；`mfem_ex46_moving_mesh_heat` 2/2 测试 | 🟢 [OK] | TBD |
| 流固耦合 (FSI) | ex49 | ✅ `mfem_ex49_template_fsi` 6/6 测试（顺应性单调性、近刚性壁、积分守恒、收敛步数、入口幅度单调性）| 🟢 [OK] | TBD |
| Joule 加热 | ex48/joule | ✅ `mfem_ex48_template_joule_heating` 3/3 测试；`mfem_joule.rs` 场基线 | 🟢 [OK] | TBD |
| EM-热-应力三场耦合 | ex51 | ✅ `mfem_ex51_template_em_thermal_stress` 5/5 测试（低 σ 极限、负反馈稳定性、σ 单调性、驱动单调性）| 🟢 [OK] | TBD |
| 热弹性耦合 | ex44 | ✅ `mfem_ex44_thermoelastic_coupled` 15/15 测试 | 🟢 [OK] | TBD |
| 反应-流-热三场耦合 | ex52 | ✅ `mfem_ex52_template_reaction_flow_thermal` 4/4 测试 | 🟢 [OK] | TBD |
| 声学-结构耦合 | ex50 | ✅ `mfem_ex50_template_acoustics_structure` 3/3 测试 | 🟢 [OK] | TBD |
| Quad/Hex NC AMR（各向异性） | [OK]| 🔨 Tri/Tet only | 🟢 [OK] | TBD |
| GPU 后端 (CUDA/HIP) | [OK]全库加[OK]| core CPU only（delegated to `vendor/linger` + `rem-rs/reed` + `vendor/jsmpi` 协同[OK]| 🟢 [OK] | TBD |

---

### Phase 55 [OK]复数[OK]FEM（Complex-Valued Systems）[OK]
> **Target**: MFEM ex22 (时谐阻尼振荡/ ex25 (PML Maxwell)
>
> 对应 MFEM `ComplexOperator` / `ComplexGridFunction` 实现模式

**问题**：时[OK]Maxwell / Helmholtz 方程含复数系数：
```
∇[OK]a∇×u) − ω²b·u + iωc·u = 0   (H(curl), 时谐电磁)
−∇·(a∇u) − ω²b·u + iωc·u = 0   (H¹, 时谐声学)
```

**实现策略** [OK]2×2 实块方案（不引入复数泛型，WASM 兼容）：
```
[K - ω²M    -ωC ] [u_re]   [f_re]
[ωC          K-ω²M] [u_im] = [f_im]
```
其中 `K = stiffness`, `M = mass`, `C = damping`[OK]

**任务清单**[OK]
- [x] `ComplexAssembler` [OK]同时组装实部/虚部矩阵[OK]×2 实块系统[OK]
- [x] `ComplexCoeff` / `ComplexVectorCoeff` [OK]复系 trait（re/im 两路，`coefficient.rs` 已提供 baseline（
- [x] `ComplexLinearForm` [OK] RHS 向量[OK]
- [x] `apply_dirichlet_complex()` [OK]复数 Dirichletichlet BC 消去（`ComplexSystem::apply_dirichlet`[OK]
- [x] `GMRES` on `BlockMatrix` [OK]通过 flatten [OK]GMRES 路径求解
- [x] `mfem_ex22.rs` [OK]高保真增强：右边界一阶吸收边界（ABC[OK] 透射 proxy 回归测试
- [x] `mfem_ex25.rs` [OK]PML-like complex Helmholtz 基线示例

---

### Phase 56 [OK]IMEX 时间积分（Implicit-Explicit Splitting）[OK]
> **Target**: MFEM ex41 (DG/CG IMEX advection-diffusion)
>
> 对应 MFEM `TimeDependentOperator` [OK]additive 分裂模式

**问题**：对[OK]扩散方程[OK]
```
∂u/∂t + v·∇u [OK]∇[OK]κ∇u) = 0
```
对流[OK]`v·∇u` 需显式（CFL 限制），扩散[OK]`∇·(κ∇u)` 需隐式（稳定性）[OK]

**任务清单**[OK]
- [x] `ImexOperator` trait [OK]分拆为`explicit_part()` + `implicit_part()`（已[OK]`fem_solver::ode` 提供[OK]
- [x] `ImexEuler` (IMEX Euler: forward for explicit, backward for implicit)
- [x] `ImexRK2` (IMEX-SSP-RK2 / Ascher-Ruuth-Spiteri 2-stage)
- [x] `ImexRK3`（固定步长三阶基线，API: `ImexRk3` + `ImexTimeStepper::integrate_rk3`[OK]
- [x] `ImexTimeStepper` [OK]统一 driver，复[OK]`ImplicitTimeStepper` 接口
- [x] `mfem_ex41_imex.rs` [OK]advection-diffusion IMEX 示例，对比纯显式 RK45

---

### Phase 57 [OK]AMR 反细化（Mesh Derefinement / Coarsening）✅
> **Target**: MFEM ex15 动态 AMR (refine + derefine + rebalance 循环[OK]

**状[OK]*：已完成[OK]026-04-12)

**实现**（Tri3 conforming 版本）：
- [x] `DerefineTree` [OK]记录精化历史（父→子元素映射，已支持单层 red-refinement 回退[OK]
- [x] `mark_for_derefinement()` [OK]基于 ZZ/Kelly 估计量标记可缩粗元素
- [x] `derefine_marked(mesh, tree, marked)` [OK]-> 4 子三角形合并回父三角形（当前为单层回退版本[OK]
- [x] 解插值：`restrict_to_coarse()` [OK]已提供 `restrict_to_coarse_p1()`（P1 节点注入版本[OK]
- [x] `NCState` / `NCState3D` 中的反细化路径（已支持按[OK]rollback -> `derefine_last()`[OK]
- [x] `mfem_ex15_dynamic_amr.rs` [OK]动态 AMR 演示（已覆盖 refine + derefine + prolongate + restrict 基础闭环[OK]

---

### Phase 58 [OK]几何多重网格 / LOR 预条件器[OK]
> **Target**: MFEM ex26 (Multigrid preconditioner for high-order Poisson)

**状[OK]*：已完成（7/7 测试通过）

**实现**（两条路线均可用）：

1. **几何 h-多重网格** [OK]利用网格细化层次，每层使-> `AmgSolver` 作平滑器
   - [x] `GeomMGHierarchy` [OK]存储层级矩阵 + Restriction/Prolongation（基线版[OK]
   - [x] `GeomMGPrecond` [OK]V-cycle 实现（Jacobi smoother + coarse CG[OK]
   - [x] `mfem_ex26_geom_mg.rs` [OK]几何多重网格基线示例；4 tests pass[OK]

2. **LOR 预条件器**（已实现）
   - [x] 构建 P1 (LOR) 矩阵与 P2 高阶矩阵（同一网格，P1 DOFs ≈ half P2 DOFs）
   - [x] 分别建立 AMG 层次（`amg_p1` / `amg_p2`），比较层数
   - [x] 用 AMG(P2) 预条件 CG 求解 P2 系统，与 Jacobi-PCG 对比迭代次数
   - [x] 验证 AMG 比 Jacobi 收敛更快；L2 误差 < 5e-3（P2 h² 收敛）；3 tests pass

---

### Phase 59 [OK]SubMesh 子域传输[OK]
> **Target**: MFEM ex34 (SubMesh source function), ex35 (port BCs)

**状[OK]*：已完成[OK]026-04-12)

**实现**[OK]
- [x] `SubMesh::extract(mesh, element_tags)` [OK]从标签提取子网格（Tri3[OK]
- [x] `SubMesh::transfer_to_parent(gf)` 到子域 FE 函数 [OK]父网格 
- [x] `SubMesh::transfer_from_parent(gf)` [OK]父网格 到子域
- [x] 多物理耦合示例基础（Joule 加热框架可用[OK]

---

### Phase 60 [OK]DG 弹[OK]+ 可压缩流[OK]
> **Target**: MFEM ex17 (DG elasticity), ex18 (DG Euler equations)

**状[OK]*：已完成[OK]026-04-12)

**实现**[OK]
- [x] `DgElasticityAssembler` [OK]向量块对SIP
- [x] `HyperbolicFormIntegrator` [OK]守恒律通量 + Lax-Friedrichs/Roe
- [x] `mfem_ex17_dg_elasticity.rs` [OK]DG 弹性基础示例
- [x] `mfem_ex18_euler.rs` [OK]Euler + SSPRK2

---

### Phase 61 [OK]辛时间积分✅
> **Target**: MFEM ex20 (symplectic integration of Hamiltonian systems)

**状[OK]*：已完成[OK]026-04-12)

**实现**[OK]
- [x] `HamiltonianSystem` trait [OK]dH/dp + dH/dq
- [x] `VerletStepper`, `Leapfrog`, `Yoshida4` 辛积分器
- [x] 能量守恒验证（标准谐振子[OK]

---

### Phase 62 [OK]受限 H(curl) 空间[OK]
> **Target**: MFEM ex31 (anisotropic Maxwell), ex32 (anisotropic Maxwell eigenproblem)

**状[OK]*：已完成[OK]026-04-12)

**实现**[OK]
- [x] 2D 网格上嵌[OK]3D 向量场接[OK]
- [x] `RestrictedHCurlSpace` [OK]低维网格高维 H(curl) DOF
- [x] `mfem_ex31.rs` [OK]各向异性 Maxwell 制造解示例 + 一阶收敛趋势回[OK]
- [x] `mfem_ex32.rs` [OK]阻抗边界 Maxwell 制造解示例 + 一阶收敛趋势回[OK]

---

### Phase 63 [OK]PML 完美匹配层与电磁各向异性✅
> **Target**: MFEM ex25 (PML), ex3/ex34 anisotropic variants

**状[OK]*：已完成[OK]026-04-13)

**实现**[OK]
- [x] `PmlCoeff` [OK]标量层吸收系数（边界层衰减）
- [x] `PmlTensorCoeff` [OK]对角张量 PML 接口
- [x] `mfem_ex25.rs` [OK]complex Helmholtz PML 示例（反[OK]proxy 指标 + `sigma_max/power` + `stretch_blend` 联合回归[OK]
- [x] `mfem_ex3 --pml-like` [OK]H(curl) 各向异[OK]PML-like 阻尼（wx/wy 控制，含 strong/weak `sigma_max` [OK]`||u||₂` 回归[OK]
- [x] `mfem_ex34 --anisotropic` [OK]各向异性吸收边界（gamma_x/gamma_y 控制，已加入制造解误差回归与细化单调下降校验）
- [x] alignment-smoke CI：electromagnetic-pml、electromagnetic-absorbing [OK]suite

### Phase 48 [OK]linger Update + Higher-Order Elements [OK]
> **Completed** -- sparse direct solvers, new Krylov methods, higher-order FEM

- [OK]Sparse direct solvers: `SparseLu`, `SparseCholesky`, `SparseLdlt` (pure-Rust, WASM-compatible)
- [OK]New iterative methods: `IDR(s)` (`solve_idrs`), `TFQMR` (`solve_tfqmr`)
- [OK]New preconditioner: `ILDLt` (`solve_pcg_ildlt`, `solve_gmres_ildlt`) for symmetric indefinite
- [OK]KrylovSchur eigenvalue solver (`krylov_schur`) [OK]thick-restart Arnoldi
- [OK]Matrix Market I/O: `read_matrix_market`, `read_matrix_market_coo`, `write_matrix_market`
- [OK]Higher-order elements: `TriP3`, `TetP2`, `TetP3`, `QuadQ2`, `SegP3` [OK]fully registered
- [OK]H1TraceSpace P2/P3 boundary trace support
- [OK]Grundmann-Moller tet quadrature fix (linear system solver, correct for all orders)
- [OK]reed submodule bug fix (`create_basis_h1_simplex` lock pattern)

---

### Phase 64 [OK]多材[OK]PML 演示 (ex3 增强) [OK]
> **Target**: MFEM ex3 的增强变体，展示多区[OK]PML 系数控制

**状[OK]*：已完成[OK]026-04-13)

**实现**[OK]
- [x] `mfem_ex3 --multi-material` [OK]4 象限各向异[OK]PML，每个区域独[OK](wx, wy) 配置
- [x] `multi_material_pml_tensor()` 函数 [OK]基于坐标的分区系[OK][Q1: 1.0/1.2, Q2: 0.9/1.1, Q3: 0.8/1.3, Q4: 1.2/0.9]
- [x] 测试：`ex3_multi_material_pml_mode_converges()` 验证 158 次迭代收[OK]
- [x] 验证：n=8, residual<1e-6

### Phase 65  并行 Maxwell PML (pex3 增强) [OK]
> **Target**: 并行 H(curl) 例子集成 PML-like 系数

**状[OK]*：已完成[OK]026-04-13)

**实现**[OK]
- [x] `mfem_pex3_maxwell.rs --pml`  并行 ND1 Maxwell 支持 PML 模式
- [x] `VectorMassTensorIntegrator<ConstantMatrixCoeff>` [OK]张量质量矩阵集成
- [x] `pml_mass_tensor()` 函数  生成 [1+σ, 0; 0, 1+σ] 各向同性阻尼张[OK]
- [x] 验证[OK] rank, n=8, 64 iters, residual<1e-8 收敛

### Phase 66 [OK]命名属性集合运[OK](ex39 增强) [OK]
> **Target**: MFEM ex39 的集合运算扩展（并集、交集、差集）

**状[OK]*：已完成[OK]026-04-13)

**实现**[OK]
- [x] `mfem_ex39_named_attributes.rs --intersection-region` [OK]集合交集（inlet [OK]outlet)
- [x] `mfem_ex39_named_attributes.rs --difference-region` [OK]集合差集（inlet \ outlet)
- [x] 测试三个场景：merge ([OK]、intersection ([OK]、difference (\)
- [x] 验证[OK] 个测试通过，演示多集合布尔运算模式

### Phase 67 — Quad4/Hex8 非协调 AMR ✅
> **Target**: 支持 Quad4 和 Hex8 元素的非协调（non-conforming）自适应网格加密

**状态**：已完成

**实现**：
- [x] `refine_nonconforming_quad(mesh, marked)` — 4-way 红色细化 + hanging node 约束生成
- [x] `NCStateQuad` — 多层历史跟踪 + 反细化支持
- [x] `refine_nonconforming_hex(mesh, marked)` — 8-way iso-split Hex8 + 面 hanging nodes
- [x] `unit_cube_hex(n)` — n³ Hex8 网格生成器
- [x] 12 个单元测试全部通过（`fem-mesh` crate）

### Phase 68 — 静态凝聚 / 杂化 FEM ✅
> **Target**: 代数静态凝聚（Schur complement elimination）

**状态**：已完成

**实现**：
- [x] `StaticCondensation::from_element_matrices(k_e, f_e, interior, boundary)` — 单元级 Schur 消去
- [x] `StaticCondensation::backsolve(u_b)` + `scatter(u_b)` — 全局 DOF 重建
- [x] `GlobalBacksolve` — 全局稀疏静态凝聚（CG 迭代内部求解）
- [x] `condense_global(k, f, interior_dofs)` — 全局系统凝聚入口
- [x] 4 个单元测试通过（`fem-assembly` crate）

### Phase 69 — AMG WP2 分布式跨 rank 聚合 🔨
> **Target**: 并行 AMG 中的 ghost-aware 跨 rank 聚合（MFEM hypre 对齐）

**状态**：基本完成，集成测试待补充

**实现**：
- [x] `ParAmgHierarchy::build_global()` — 新入口，调用 WP2 全局聚合路径
- [x] `build_coarse_level_global(a, comm, threshold)` — ghost-aware 聚合：
  1. 全行强连接（含 offd block）
  2. 本地 Phase1/Phase2 聚合
  3. alltoallv_bytes + broadcast_bytes 计算全局聚合偏移
  4. GhostExchange::forward 传播所有权聚合 ID 到 ghost 槽
  5. union-find 边界聚合合并
  6. 全局重编号 + 构建 P、R、A_c
- [x] 编译通过（`fem-parallel` crate）

### Phase 70 — NURBS/IGA 参考元与组装 ✅
> **Target**: B-spline + NURBS 等几何分析（IGA）参考元素、物理域映射、全局组装

**状态**：已完成（参考元 + 物理域映射 + 2D/3D 全局组装）

**实现**：
- [x] `KnotVector` — 节点向量（uniform clamped 构造器、find_span、basis_funs、basis_funs_and_ders）
- [x] `BSplineBasis1D` — 1-D B-spline 基（eval、eval_with_ders）
- [x] `NurbsPatch2D` — 实现 `ReferenceElement`（eval_basis、eval_grad_basis、quadrature、dof_coords）
- [x] `NurbsPatch3D` — 3-D NURBS 参考元（同上）
- [x] `NurbsMesh2D` / `NurbsMesh3D` — 多片 NURBS 网格容器
- [x] `greville_abscissae` — DOF 坐标计算
- [x] 13 个单元测试全部通过（分区单位性、梯度有限差验证等）
- [x] `physical_map_2d/3d`、`physical_grads_2d/3d` — 物理域映射与梯度变换
- [x] `assemble_iga_diffusion_2d/3d`、`assemble_iga_mass_2d/3d`、`assemble_iga_load_2d/3d`
- [x] `fem-assembly` IGA 单测 10/10 通过（含 2D Poisson 网格细化误差下降）

---

## 例子命名迁移记录 (2026-04-13)

为实[OK]**MFEM 对应关系清晰[OK]* **[OK]** 命名规范统一**，所[OK]`ex_` 前缀的应[OK]增强例子迁移[OK]`mfem_ex<N>_<variant>` 格式[OK]

| 旧名[OK]| 新名称 | MFEM 对应 | Phase | 描述 |
|---|---|---|---|---|
| `ex_stokes.rs` | `mfem_ex40.rs` | MFEM ex40 | 40 | Taylor-Hood P2-P1 盖驱动腔 |
| `ex_navier_stokes.rs` | `mfem_ex19.rs` | MFEM ex19 | 44 | Kovasznay 流不可压[OK]Navier-Stokes |
| `ex_maxwell_eigenvalue.rs` | `mfem_ex13_eigenvalue.rs` | MFEM ex13 | [OK]| H(curl) 特征值问[OK](LOBPCG，含细化后首[OK]最大相对误差改善回[OK] |
| `ex_maxwell_time.rs` | `mfem_ex10_maxwell_time.rs` | MFEM ex10 | [OK]| 时间[OK]Maxwell (Newmark-β，已提取 `solve_case` 并补充时间步[OK]阻尼回归 + 时间自收敛二阶验[OK] |

**迁移完成**[OK]
- [OK]文件系统迁移（move 命令[OK]
- [OK]`examples/Cargo.toml` 更新[OK] [[example]] 配置[OK]
- [OK]编译验证（fem-examples lib 101/101 测试通过[OK]

**好处**[OK]
- 清晰[OK]MFEM 版本对应关系
- 统一的命名规范（`mfem_ex<number>` 格式[OK]
- 易于在文档和 CI 中引[OK]

