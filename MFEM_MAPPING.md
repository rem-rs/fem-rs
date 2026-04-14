# fem-rs �?MFEM Correspondence Reference
> Tracks every major MFEM concept and its planned or implemented fem-rs counterpart.
> Use this as the authoritative target checklist for feature completeness.
>
> Status legend: �?implemented · 🔨 partial · 🔲 planned · �?out-of-scope

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
12. [MFEM Examples �?fem-rs Milestones](#12-mfem-examples--fem-rs-milestones)
13. [Key Design Differences](#13-key-design-differences)

---

## 1. Mesh

### 1.1 Mesh Container

| MFEM class / concept | fem-rs equivalent | Status | Notes |
|---|---|---|---|
| `Mesh` (2D/3D unstructured) | `SimplexMesh<D>` | �?| Uniform element type per mesh |
| `Mesh` (mixed elements) | `SimplexMesh<D>` + `elem_types`/`elem_offsets` | 🔨 | Phase 42a: data structures + I/O done |
| `NCMesh` (non-conforming) | `refine_nonconforming()` (2-D) + `refine_nonconforming_3d()` + `NCState`/`NCState3D` | �?| Tri3/Tet4 multi-level non-conforming refinement + hanging constraints |
| `ParMesh` | `ParallelMesh<M>` | �?| Phase 10+33 |
| `Mesh::GetNV()` | `MeshTopology::n_nodes()` | �?| |
| `Mesh::GetNE()` | `MeshTopology::n_elements()` | �?| |
| `Mesh::GetNBE()` | `MeshTopology::n_boundary_faces()` | �?| |
| `Mesh::GetVerticesArray()` | `SimplexMesh::coords` (flat `Vec<f64>`) | �?| |
| `Mesh::GetElementVertices()` | `MeshTopology::element_nodes()` | �?| |
| `Mesh::GetBdrElementVertices()` | `MeshTopology::face_nodes()` | �?| |
| `Mesh::GetBdrAttribute()` | `MeshTopology::face_tag()` | �?| Tags match GMSH physical group IDs |
| `Mesh::GetAttribute()` | `MeshTopology::element_tag()` | �?| Material group tag |
| `Mesh::bdr_attributes` | `SimplexMesh::unique_boundary_tags()` | �?| Sorted, deduplicated boundary tag set |
| `Mesh::GetDim()` | `MeshTopology::dim()` | �?| Returns `u8` (2 or 3) |
| `Mesh::GetSpaceDim()` | same as `dim()` for flat meshes | �?| |
| `Mesh::UniformRefinement()` | `refine_uniform()` | �?| Red refinement (Tri3�? children) |
| `Mesh::AdaptiveRefinement()` | `refine_marked()` + ZZ estimator + Dörfler marking | �?| Phase 17 |
| `Mesh::GetElementTransformation()` | `ElementTransformation` | �?| 仿射 simplex 装配路径已统一接入 `ElementTransformation` |
| `Mesh::GetFaceElementTransformations()` | `InteriorFaceList` | �?| Used by DG assembler |
| `Mesh::GetBoundingBox()` | `SimplexMesh::bounding_box()` | �?| Returns `(min, max)` per axis |

### 1.2 Element Types

| MFEM element | `ElementType` variant | dim | Nodes | Status |
|---|---|---|---|---|
| `Segment` | `Line2` | 1 | 2 | �?|
| Quadratic segment | `Line3` | 1 | 3 | �?|
| `Triangle` | `Tri3` | 2 | 3 | �?|
| Quadratic triangle | `Tri6` | 2 | 6 | �?|
| `Quadrilateral` | `Quad4` | 2 | 4 | �?|
| Serendipity quad | `Quad8` | 2 | 8 | �?|
| `Tetrahedron` | `Tet4` | 3 | 4 | �?|
| Quadratic tet | `Tet10` | 3 | 10 | �?|
| `Hexahedron` | `Hex8` | 3 | 8 | �?|
| Serendipity hex | `Hex20` | 3 | 20 | �?|
| `Wedge` (prism) | `Prism6` | 3 | 6 | �?(type only) |
| `Pyramid` | `Pyramid5` | 3 | 5 | �?(type only) |
| `Point` | `Point1` | 0 | 1 | �?|

### 1.3 Mesh Generators

| MFEM generator | fem-rs equivalent | Status |
|---|---|---|
| `Mesh::MakeCartesian2D()` | `SimplexMesh::unit_square_tri(n)` | �?|
| `Mesh::MakeCartesian3D()` | `SimplexMesh::unit_cube_tet(n)` | �?| Added in Phase 9 |
| `Mesh::MakePeriodic()` | `SimplexMesh::make_periodic()` | �?| Node merging + face removal |
| Reading MFEM format | �?| �?use GMSH instead |
| Reading GMSH `.msh` v4 | `fem_io::read_msh_file()` | �?|
| Reading Netgen | `fem_io::read_netgen_vol_file()` | 🔨 Phase 67 (Tet4/Hex8 ASCII 读取基线，支�?uniform + mixed；写出仍�?Tet4 baseline 为主) |

---

## 2. Reference Elements & Quadrature

### 2.1 Reference Elements

| MFEM class | fem-rs trait/struct | Status |
|---|---|---|
| `FiniteElement` (base) | `ReferenceElement` trait | �?|
| `Poly_1D` utility | inline basis in `lagrange/` | �?|
| `H1_SegmentElement` P1/P2/P3 | `SegP1`, `SegP2`, `SegP3` | �?|
| `H1_TriangleElement` P1/P2/P3 | `TriP1`, `TriP2`, `TriP3` | �?|
| `H1_TetrahedronElement` P1/P2/P3 | `TetP1`, `TetP2`, `TetP3` | �?|
| `H1_QuadrilateralElement` Q1/Q2 | `QuadQ1`, `QuadQ2` | �?|
| `H1_HexahedronElement` | `HexQ1` | �?|
| `ND_TriangleElement` (order 1) | `nedelec::TriND1` | �?|
| `ND_TriangleElement` (order 2) | `nedelec::TriND2` | �?|
| `ND_TetrahedronElement` (order 1) | `nedelec::TetND1` | �?|
| `ND_TetrahedronElement` (order 2) | `nedelec::TetND2` | �?|
| `RT_TriangleElement` (order 0) | `raviart_thomas::TriRT0` | �?|
| `RT_TriangleElement` (order 1) | `raviart_thomas::TriRT1` | �?|
| `RT_TetrahedronElement` (order 0) | `raviart_thomas::TetRT0` | �?|
| `RT_TetrahedronElement` (order 1) | `raviart_thomas::TetRT1` | �?|
| `L2_TriangleElement` | L2Space with P0/P1 | �?|

### 2.2 Quadrature Rules

| MFEM class | fem-rs struct | Status |
|---|---|---|
| `IntegrationRule` | `QuadratureRule` | �?|
| `IntegrationRules` (table) | `quadrature.rs` look-up table | �?|
| Gauss-Legendre 1D (orders 1�?0) | `gauss_legendre_1d(order)` | �?|
| Gauss-Legendre on triangle | `gauss_triangle(order)` | �?|
| Gauss-Legendre on tet | `gauss_tet(order)` + Grundmann-Moller | �?|
| Tensor product (quad, hex) | `tensor_gauss(order, dim)` | �?|
| Gauss-Lobatto | `gauss_lobatto_1d`, `seg_lobatto_rule`, `quad_lobatto_rule`, `hex_lobatto_rule` | �?|

---

## 3. Finite Element Spaces

### 3.1 Collections (Basis Families)

| MFEM collection | Mathematical space | fem-rs struct | Status |
|---|---|---|---|
| `H1_FECollection(p)` | H¹(Ω): C�?scalar Lagrange | `H1Space` (P1–P3) | �?|
| `L2_FECollection(p)` | L²(Ω): discontinuous Lagrange | `L2Space` | �?|
| `DG_FECollection(p)` | L²(Ω): DG (element-interior only) | `L2Space` | �?|
| `ND_FECollection(p)` | H(curl): Nédélec tangential | `HCurlSpace` | �?|
| `RT_FECollection(p)` | H(div): Raviart-Thomas normal | `HDivSpace` | �?|
| `H1_Trace_FECollection` | H½: traces of H¹ on faces | `H1TraceSpace` | �?| P1–P3 boundary trace |
| `NURBS_FECollection` | NURBS isogeometric | �?| �?out of scope |

### 3.2 Finite Element Space (DOF management)

| MFEM method | fem-rs equivalent | Status |
|---|---|---|
| `FiniteElementSpace(mesh, fec)` | `H1Space::new(mesh)` etc. | �?|
| `FES::GetNDofs()` | `FESpace::n_dofs()` | �?|
| `FES::GetElementDofs()` | `FESpace::element_dofs()` | �?|
| `FES::GetBdrElementDofs()` | `boundary_dofs()` | �?|
| `FES::GetEssentialTrueDofs()` | `boundary_dofs()` + `apply_dirichlet()` | �?|
| `FES::GetTrueDofs()` | `DofPartition::n_owned_dofs` + `global_dof()` | �?| Phase 33b |
| `FES::TransferToTrue()` / `Transfer()` | `DofPartition::permute_dof()` / `unpermute_dof()` | �?| Phase 34 |
| `DofTransformation` | `FESpace::element_signs()` | �?| HCurlSpace/HDivSpace sign convention |
| `FES::GetFE()` | `FESpace::element_type()` | �?|

### 3.3 Space Types

| Space | Problem | Status |
|---|---|---|
| H¹ | Electrostatics, heat, elasticity (scalar) | �?|
| H(curl) | Maxwell, eddy currents (vector potential) | �?|
| H(div) | Darcy flow, mixed Poisson | �?|
| L² / DG | Transport, DG methods | �?|
| Vector H¹ = [H¹]�?| Elasticity (displacement vector) | �?|
| Taylor-Hood P2-P1 | Stokes flow | �?Via MixedAssembler + `mfem_ex40` |

---

## 4. Coefficients

MFEM provides a rich coefficient hierarchy for spatially- and
time-varying material properties.  fem-rs uses a trait-based system:
`ScalarCoeff`, `VectorCoeff`, `MatrixCoeff` traits with `f64` as the
default (zero-cost for constants).

| MFEM class | fem-rs | Status |
|---|---|---|
| `ConstantCoefficient(c)` | `f64` (implements `ScalarCoeff`) | �?|
| `FunctionCoefficient(f)` | `FnCoeff(\|x\| f(x))` | �?|
| `GridFunctionCoefficient` | `GridFunctionCoeff::new(dof_vec)` | �?|
| `PWConstCoefficient` | `PWConstCoeff::new([(tag, val), ...])` | �?|
| `PWCoefficient` | `PWCoeff::new(default).add_region(tag, coeff)` | �?|
| `VectorCoefficient` | `VectorCoeff` trait + `FnVectorCoeff`, `ConstantVectorCoeff` | �?|
| `MatrixCoefficient` | `MatrixCoeff` trait + `FnMatrixCoeff`, `ConstantMatrixCoeff`, `ScalarMatrixCoeff` | �?|
| `InnerProductCoefficient` | `InnerProductCoeff { a, b }` | �?|
| `TransformedCoefficient` | `TransformedCoeff { inner, transform }` | �?|

---

## 5. Assembly: Forms & Integrators

### 5.1 Bilinear Forms

| MFEM class | fem-rs equivalent | Status |
|---|---|---|
| `BilinearForm(fes)` | `Assembler::assemble_bilinear(integrators)` | �?|
| `BilinearForm::AddDomainIntegrator()` | `assembler.add_domain(integrator)` | �?|
| `BilinearForm::AddBoundaryIntegrator()` | `assembler.add_boundary(integrator)` | �?|
| `BilinearForm::Assemble()` | `Assembler::assemble_bilinear()` | �?|
| `BilinearForm::FormLinearSystem()` | `apply_dirichlet()` | �?|
| `BilinearForm::FormSystemMatrix()` | `apply_dirichlet()` variants | �?|
| `MixedBilinearForm(trial, test)` | `MixedAssembler` | �?|

### 5.2 Linear Forms

| MFEM class | fem-rs equivalent | Status |
|---|---|---|
| `LinearForm(fes)` | `Assembler::assemble_linear(integrators)` | �?|
| `LinearForm::AddDomainIntegrator()` | `assembler.add_domain_load(integrator)` | �?|
| `LinearForm::AddBndryIntegrator()` | `NeumannIntegrator` | �?|
| `LinearForm::Assemble()` | `Assembler::assemble_linear()` | �?|

### 5.3 Bilinear Integrators

| MFEM integrator | Bilinear form | fem-rs struct | Status |
|---|---|---|---|
| `DiffusionIntegrator(κ)` | �?κ ∇u·∇v dx | `DiffusionIntegrator` | �?|
| `MassIntegrator(ρ)` | �?ρ u v dx | `MassIntegrator` | �?|
| `ConvectionIntegrator(b)` | �?(b·∇u) v dx | `ConvectionIntegrator` | �?|
| `ElasticityIntegrator(λ,μ)` | �?σ(u):ε(v) dx | `ElasticityIntegrator` | �?|
| `CurlCurlIntegrator(μ)` | �?μ (∇×u)·(∇×v) dx | `CurlCurlIntegrator` | �?|
| `VectorFEMassIntegrator` | �?u·v dx (H(curl)/H(div)) | `VectorMassIntegrator` | �?|
| `DivDivIntegrator(κ)` | �?κ (∇·u)(∇·v) dx | `DivIntegrator` | �?|
| `VectorDiffusionIntegrator` | �?κ ∇uᵢ·∇v�?(vector Laplacian) | `VectorDiffusionIntegrator` | �?|
| `BoundaryMassIntegrator` | ∫_Γ α u v ds | `BoundaryMassIntegrator` | �?|
| `VectorFEDivergenceIntegrator` | �?(∇·u) q dx (Darcy/Stokes) | `PressureDivIntegrator` | �?|
| `GradDivIntegrator` | �?(∇·u)(∇·v) dx | `GradDivIntegrator` | �?|
| `DGDiffusionIntegrator` | Interior penalty DG diffusion | `DgAssembler::assemble_sip` | �?|
| `TransposeIntegrator` | Transposes a bilinear form | `TransposeIntegrator` | �?|
| `SumIntegrator` | Sum of integrators | `SumIntegrator` | �?|

### 5.4 Linear Integrators

| MFEM integrator | Linear form | fem-rs struct | Status |
|---|---|---|---|
| `DomainLFIntegrator(f)` | �?f v dx | `DomainSourceIntegrator` | �?|
| `BoundaryLFIntegrator(g)` | ∫_Γ g v ds | `NeumannIntegrator` | �?|
| `VectorDomainLFIntegrator` | �?**f**·**v** dx | `VectorDomainLFIntegrator` | �?|
| `BoundaryNormalLFIntegrator` | ∫_Γ g (n·v) ds | `BoundaryNormalLFIntegrator` | �?|
| `VectorFEBoundaryFluxLFIntegrator` | ∫_Γ f (v·n) ds (RT) | `VectorFEBoundaryFluxLFIntegrator` | �?|

### 5.5 Assembly Pipeline

| MFEM concept | fem-rs equivalent | Status |
|---|---|---|
| `ElementTransformation` | Jacobian `jac`, `det_j`, `jac_inv_t` | �?|
| `Geometry::Type` | `ElementType` enum | �?|
| Sparsity pattern | `SparsityPattern` built once | �?|
| Parallel assembly | Element loop �?ghost DOF AllReduce | �?via ChannelBackend |

---

## 6. Linear Algebra

### 6.1 Sparse Matrix

| MFEM class | fem-rs struct | Status |
|---|---|---|
| `SparseMatrix` (CSR) | `CsrMatrix<T>` | �?|
| `SparseMatrix::Add(i,j,v)` | `CooMatrix::add(i,j,v)` | �?|
| `SparseMatrix::Finalize()` | `CooMatrix::into_csr()` | �?|
| `SparseMatrix::Mult(x,y)` | `CsrMatrix::spmv(x,y)` | �?|
| `SparseMatrix::MultTranspose()` | `CsrMatrix::transpose()` + spmv | �?|
| `SparseMatrix::EliminateRowCol()` | `apply_dirichlet_symmetric()` | �?|
| `SparseMatrix::EliminateRow()` | `apply_dirichlet_row_zeroing()` | �?|
| `SparseMatrix::GetDiag()` | `CsrMatrix::diagonal()` | �?|
| `SparseMatrix::Transpose()` | `CsrMatrix::transpose()` | �?|
| `SparseMatrix::Add(A,B)` | `spadd(&A, &B)` | �?|
| `SparseMatrix::Mult(A,B)` | SpGEMM (via linger) | �?|
| `DenseMatrix` (local dense) | `nalgebra::SMatrix` | �?|
| `DenseTensor` | `DenseTensor` (3-D array) | �?| Row-major slab access |
| Matrix Market read/write | `fem_io::read_matrix_market` / `write_matrix_market` | �?| `.mtx` COO/CSR, real/symmetric/pattern |

### 6.2 Vector

| MFEM class | fem-rs struct | Status |
|---|---|---|
| `Vector` | `Vector<T>` | �?|
| `Vector::operator +=` | `Vector::axpy(1.0, x)` | �?|
| `Vector::operator *=` | `Vector::scale(a)` | �?|
| `Vector::operator * (dot)` | `Vector::dot()` | �?|
| `Vector::Norml2()` | `Vector::norm()` | �?|
| `Vector::Neg()` | `vector.scale(-1.0)` | �?|
| `Vector::SetSubVector()` | `Vector::set_sub_vector()` / `get_sub_vector()` | �?| Offset-based slice ops |
| `BlockVector` | `BlockVector` | �?|

---

## 7. Solvers & Preconditioners

### 7.1 Iterative Solvers

| MFEM solver | Problem type | fem-rs module | Status |
|---|---|---|---|
| `CGSolver` | SPD: A x = b | `solver` (via linger) | �?|
| `PCGSolver` | SPD + preconditioner | `solver` (PCG+Jacobi/ILU0/ILDLt) | �?|
| `GMRESSolver(m)` | General: A x = b | `solver` (via linger) | �?|
| `FGMRESSolver` | Flexible GMRES | `solve_fgmres` / `solve_fgmres_jacobi` | �?|
| `BiCGSTABSolver` | Non-symmetric | `solver` (via linger) | �?|
| IDR(s) | Non-symmetric, short-recurrence | `solve_idrs` | �?|
| TFQMR | Transpose-free QMR | `solve_tfqmr` | �?|
| `MINRESSolver` | Indefinite symmetric | `MinresSolver` | �?|
| `SLISolver` | Stationary linear iteration | `solve_jacobi_sli` / `solve_gs_sli` | �?|
| `NewtonSolver` | Nonlinear F(x)=0 | `NewtonSolver` | �?|
| `UMFPackSolver` | Direct (SuiteSparse) | `solve_sparse_lu` / `solve_sparse_cholesky` / `solve_sparse_ldlt` | �?Pure-Rust sparse direct |
| `MUMPSSolver` | Parallel direct | `solve_sparse_mumps` + `linger::MumpsSolver` | 🔨 | MUMPS-compatible API name backed by linger native multifrontal direct solves; replacement path, not external MUMPS FFI |

### 7.2 Preconditioners

| MFEM preconditioner | Type | fem-rs module | Status |
|---|---|---|---|
| `DSmoother` | Jacobi / diagonal scaling | PCG+Jacobi (via linger) | �?|
| `GSSmoother` | Gauss-Seidel | `SmootherKind::GaussSeidel` (AMG) | �?|
| Chebyshev smoother | Chebyshev polynomial | `SmootherType::Chebyshev` | �?|
| `SparseSmoothedProjection` | ILU-based | PCG+ILU0 (via linger) | �?|
| Incomplete LDLᵀ | Symmetric indefinite preconditioning | `IldltPrecond` via `solve_pcg_ildlt` / `solve_gmres_ildlt` | �?|
| `BlockDiagonalPreconditioner` | Block Jacobi | `BlockDiagonalPrecond` | �?|
| `BlockTriangularPreconditioner` | Block triangular | `BlockTriangularPrecond` | �?|
| `SchurComplement` | Elimination for saddle point | `SchurComplementSolver` | �?|

### 7.3 Solver Convergence Monitors

| MFEM concept | fem-rs equivalent | Status |
|---|---|---|
| `IterativeSolver::SetTol()` | `tol` parameter | �?|
| `IterativeSolver::SetMaxIter()` | `max_iter` parameter | �?|
| `IterativeSolver::GetFinalNorm()` | `SolverResult::residual_norm` | �?|
| `IterativeSolver::GetNumIterations()` | `SolverResult::iterations` | �?|
| `IterativeSolver::SetPrintLevel()` | `SolverConfig::print_level` / `PrintLevel` enum | �?| Silent/Summary/Iterations/Debug |

---

## 8. Algebraic Multigrid

| MFEM / hypre concept | fem-rs equivalent | Status |
|---|---|---|
| `LOBPCGSolver` | Block eigensolver for SPD | `lobpcg` / `LobpcgSolver` | �?|
| Krylov-Schur | Thick-restart Arnoldi eigensolver | `krylov_schur` | �?|
| `HypreBoomerAMG` (setup) | `AmgSolver::setup(mat)` �?hierarchy | �?|
| `HypreBoomerAMG` (solve) | `AmgSolver::solve(hierarchy, rhs)` | �?|
| Strength of connection θ | `AmgParams::theta` | �?|
| Ruge-Stüben C/F splitting | RS-AMG (via linger) | �?|
| Smoothed aggregation | SA-AMG (via linger) | �?|
| Prolongation P | `AmgLevel::p: CsrMatrix` | �?|
| Restriction R = Pᵀ | `AmgLevel::r: CsrMatrix` | �?|
| Galerkin coarse A_c = R A P | SpGEMM chain | �?|
| Pre-smoother (ω-Jacobi) | Jacobi smoother | �?|
| Post-smoother | Post-smooth steps | �?|
| V-cycle | `CycleType::V` | �?|
| W-cycle | `CycleType::W` | �?|
| F-cycle | `CycleType::F` | �?|
| Max levels | Max levels config | �?|
| Coarse-grid direct solve | Dense LU | �?|
| hypre-equivalent AMG path | pure-Rust implementation in `vendor/linger` (no external hypre FFI) | �?|

---

## 9. Parallel Infrastructure

### 9.1 MPI Communicators

| MFEM concept | fem-rs module | Status |
|---|---|---|
| `MPI_Comm` | `ChannelBackend` (in-process threading) | �?|
| `MPI_Allreduce` | `Backend::allreduce()` | �?|
| `MPI_Allgather` | `Backend::allgather()` | �?|
| `MPI_Send/Recv` | `GhostExchange` (alltoallv) | �?|

### 9.2 Distributed Mesh

| MFEM class | fem-rs struct | Status |
|---|---|---|
| `ParMesh` | `ThreadLauncher` + partitioned mesh | �?|
| METIS partitioning | `MetisPartitioner` (pure-Rust) | �?|
| Ghost elements | `GhostExchange` (forward/reverse) | �?|
| Global-to-local node map | per-rank DOF mapping | �?|

### 9.3 Parallel Linear Algebra

| MFEM / hypre class | fem-rs struct | Status |
|---|---|---|
| `HypreParMatrix` | `ParCsrMatrix` (diag+offd blocks) | �?Thread + MPI backends |
| `HypreParVector` | `ParVector` (owned+ghost layout) | �?|
| `HypreParMatrix::Mult()` | `ParCsrMatrix::spmv()` via ghost exchange | �?|
| `HypreParMatrix::GetDiag()` | `ParCsrMatrix::diag` | �?|
| `HypreParMatrix::GetOffd()` | `ParCsrMatrix::offd` | �?|
| `ParFiniteElementSpace` | `ParallelFESpace<S>` (P1+P2) | �?|
| `ParBilinearForm::Assemble()` | `ParAssembler::assemble_bilinear()` | �?|
| `ParLinearForm::Assemble()` | `ParAssembler::assemble_linear()` | �?|
| `HypreSolver` (PCG+Jacobi) | `par_solve_pcg_jacobi()` | �?|
| `HypreBoomerAMG` | `ParAmgHierarchy` (local smoothed aggregation) | �?|
| `par_solve_pcg_amg()` | PCG + AMG V-cycle preconditioner | �?|
| `MPI_Comm_split` | `Comm::split(color, key)` | �?|
| Streaming mesh distribution | `partition_simplex_streaming()` | �?Phase 37 |
| WASM multi-Worker MPI | `WorkerLauncher::spawn_async()` + `jsmpi_main` | �?Phase 37 |
| Binary sub-mesh serde | `mesh_serde::encode/decode_submesh()` | �?Phase 37 |

---

## 10. I/O & Visualization

### 10.1 Mesh I/O

| MFEM format / method | fem-rs | Status |
|---|---|---|
| MFEM native mesh format (read/write) | �?| �?use GMSH |
| GMSH `.msh` v2 ASCII (read) | `fem_io::read_msh_file()` | �?|
| GMSH `.msh` v4.1 ASCII (read) | `fem_io::read_msh_file()` | �?|
| GMSH `.msh` v4.1 binary (read) | `fem_io::read_msh_file()` | �?|
| Netgen `.vol` (read/write) | `read_netgen_vol_file()` / `write_netgen_vol_file()` | 🔨 | 读取：Tet4/Hex8 ASCII baseline（支�?mixed）；写出：Tet4 ASCII baseline |
| Abaqus `.inp` (read) | `read_abaqus_inp_file()` | 🔨 | C3D4/C3D8 baseline（支�?uniform + mixed�?|
| VTK `.vtu` legacy ASCII (write) | `write_vtk_scalar()` | �?|
| VTK `.vtu` XML binary (write) | `write_vtu()` (XML ASCII) | �?|
| HDF5 / XDMF (read/write) | `fem-io-hdf5-parallel` (feature-gated) | 🔨 |
| ParaView GLVis socket | �?| �?out of scope |

### 10.2 Solution I/O

| MFEM concept | fem-rs | Status |
|---|---|---|
| `GridFunction::Save()` | VTK point data | �?scalar + vector |
| `GridFunction::Load()` | `read_vtu_point_data()` | �?| ASCII VTU reader |
| Restart files | HDF5 checkpoint schema + restart reads | 🔨 |

---

## 11. Grid Functions & Post-processing

| MFEM class / method | fem-rs equivalent | Status |
|---|---|---|
| `GridFunction(fes)` | `GridFunction<S>` (wraps DOF vec + space ref) | �?|
| `GridFunction::ProjectCoefficient()` | `FESpace::interpolate(f)` | �?|
| `GridFunction::ComputeL2Error()` | `GridFunction::compute_l2_error()` | �?|
| `GridFunction::ComputeH1Error()` | `GridFunction::compute_h1_error()` / `compute_h1_full_error()` | �?|
| `GridFunction::GetGradient()` | `postprocess::compute_element_gradients()` / `recover_gradient_nodal()` | �?|
| `GridFunction::GetCurl()` | `postprocess::compute_element_curl()` | �?|
| `GridFunction::GetDivergence()` | `postprocess::compute_element_divergence()` | �?|
| `ZZErrorEstimator` (Zienkiewicz-Zhu) | `zz_error_estimator()` | �?|
| `KellyErrorEstimator` | `kelly_estimator()` | �?| Face-jump based error indicator |
| `DiscreteLinearOperator` | Gradient, curl, div operators | �?`DiscreteLinearOperator::gradient/curl_2d/divergence` |

---

## 12. MFEM Examples �?fem-rs Milestones

Each MFEM example defines a target milestone for fem-rs feature completeness.

### Tier 1 �?Core Capability (Phases 6�?)

| MFEM example | PDE | FEM space | BCs | fem-rs milestone |
|---|---|---|---|---|
| **ex1** | −∇²u = 1, u=0 on ∂�?| H¹ P1/P2 | Dirichlet | �?`mfem_ex1_poisson` O(h²) |
| **ex2** | −∇²u = f, mixed BCs | H¹ P1/P2 | Dirichlet + Neumann | �?`mfem_ex2_elasticity` |
| **ex3** (scalar) | −∇²u + αu = f (reaction-diffusion) | H¹ P1 | Dirichlet | �?Phase 6: `MassIntegrator` |
| **ex13** | −∇·(ε∇�? = 0, elasticity | H¹ vector | Mixed | Phase 6: `ElasticityIntegrator` |
| **pex1** | Parallel Poisson | H¹ + MPI | Dirichlet | �?`mfem_pex1_poisson` (contiguous/METIS, streaming) |

### Tier 2 �?Mixed & H(curl)/H(div) (Phase 6+)

| MFEM example | PDE | FEM space | fem-rs milestone |
|---|---|---|---|
| **ex3** (curl) | ∇×∇×**u** + **u** = **f** (Maxwell) | H(curl) Nédélec | �?`mfem_ex3` O(h) |
| **ex4** | −∇·(**u**) = f, **u** = −κ∇p (Darcy) | H(div) RT + L² | �?`mfem_ex4_darcy` H(div) RT0 grad-div MINRES |
| **ex5** | Saddle-point Darcy/Stokes | H(div) × L² | �?`mfem_ex5_mixed_darcy` block PGMRES |
| **ex22** | Time-harmonic Maxwell (complex coeff.) | H(curl) | Phase 7+ |

### Tier 3 �?Time Integration (Phase 7+)

| MFEM example | PDE | Time method | fem-rs milestone |
|---|---|---|---|
| **ex9** (heat) | ∂u/∂t �?∇²u = 0 | BDF1 / Crank-Nicolson | �?`mfem_ex10_heat_equation` SDIRK-2 |
| **ex10** (wave) | ∂²u/∂t² �?∇²u = 0 | Leapfrog / Newmark | �?`mfem_ex10_wave_equation` Newmark-β |
| **ex14** (DG heat) | ∂u/∂t �?∇²u + b·∇u = 0 | Explicit RK + DG | �?`mfem_ex9_dg_advection` SIP-DG O(h²) |
| **ex16** (elastodynamics) | ρ ∂�?*u**/∂t² = ∇·�?| Generalized-α | �?`mfem_ex16_nonlinear_heat` Newton |

### Tier 4 �?Nonlinear & AMR (Phase 7+)

| MFEM example | Problem | fem-rs milestone |
|---|---|---|
| **ex4** (nonlinear) | −Δu + exp(u) = 0 | �?`NewtonSolver` |
| **ex6** | AMR Poisson with ZZ estimator | �?`refine_marked()`, `ZZErrorEstimator` |
| **ex15** | DG advection with AMR | �?`mfem_ex15_dg_amr` P1 + ZZ + Dörfler + refinement |
| **ex19** | Incompressible Navier-Stokes | �?`mfem_ex19` (Kovasznay Re=40, Oseen/Picard) |

### Tier 5 �?HPC & Parallel (Phase 10)

| MFEM example | Problem | fem-rs milestone |
|---|---|---|
| **pex1** | Parallel Poisson (Poisson) | �?`mfem_pex1_poisson` (contiguous/METIS + streaming) |
| **pex2** | Parallel mixed Poisson | �?`mfem_pex2_mixed_darcy` |
| **pex3** | Parallel Maxwell (H(curl)) | �?`mfem_pex3_maxwell` |
| **pex5** | Parallel Darcy | �?`mfem_pex5_darcy` |

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

## Quick Reference: Phase �?Features

| Phase | Crates | MFEM equivalents unlocked | Status |
|---|---|---|---|
| 0 | workspace | �?| �?|
| 1 | `core` | Index types, `FemError`, scalar traits | �?|
| 2 | `mesh` | `Mesh`, element types, mesh generators | �?|
| 3 | `element` | `FiniteElement`, `IntegrationRule`, Lagrange P1–P2 | �?|
| 4 | `linalg` | `SparseMatrix`, `Vector`, COO→CSR assembly | �?|
| 5 | `space` | `FiniteElementSpace`, H1/L2, DOF manager | �?|
| 6 | `assembly` | `BilinearForm`, `LinearForm`, standard integrators | �?|
| 7 | `solver` | `CGSolver`, `GMRESSolver`, ILU(0), direct | �?|
| 8 | `amg` | SA-AMG + RS-AMG (native via linger) | �?|
| 9 | `io` | VTK XML, GMSH v4 reader | �?|
| 10 | `parallel` | Thread-based parallel, ghost exchange | �?|
| 11 | `wasm` | Browser-side FEM solver via JS API | �?|
| 12 | `element` | Nedelec ND1, Raviart-Thomas RT0 | �?|
| 13 | `space`+`assembly` | VectorH1Space, BlockMatrix, MixedAssembler, Elasticity | �?|
| 14 | `assembly` | SIP-DG (interior penalty) | �?|
| 15 | `solver`+`assembly` | NonlinearForm, NewtonSolver | �?|
| 16 | `solver` | ODE: ForwardEuler, RK4, RK45, ImplicitEuler, SDIRK-2, BDF-2 | �?|
| 17 | `mesh` | AMR: red refinement, ZZ estimator, Dörfler marking | �?|
| 18 | `parallel` | METIS k-way partitioning (pure-Rust) | �?|
| 19 | `mesh`+`space` | CurvedMesh (P2 isoparametric) | �?|
| 20 | `solver` | LOBPCG eigenvalue solver | �?|
| 21 | `solver`+`linalg` | BlockSystem, SchurComplement, MINRES | �?|
| 22 | `assembly`+`ceed` | Partial assembly: PA mass/diffusion, matrix-free | �?|
| 23 | `space` | HCurlSpace (Nédélec ND1), HDivSpace (RT0), element_signs | �?|
| 24 | `assembly` | VectorAssembler, CurlCurlIntegrator, VectorMassIntegrator | �?|
| 25 | `assembly`+`solver` | DG-SIP face normals fix, SchurComplement PGMRES, MINRES rewrite, TriND1 fix; all 8 MFEM-style examples verified | �?|
| 26 | `assembly` | Coefficient system: ScalarCoeff/VectorCoeff/MatrixCoeff traits, PWConstCoeff, PWCoeff, GridFunctionCoeff, composition | �?|
| 27 | `assembly` | Convection, VectorDiffusion, BoundaryMass, GradDiv, Transpose, Sum integrators; VectorDomainLF, BoundaryNormalLF | �?|
| 28 | `assembly` | GridFunction wrapper, L²/H¹ error, element gradients/curl/div, nodal gradient recovery | �?|
| 29 | `assembly` | DiscreteLinearOperator: gradient, curl_2d, divergence as sparse matrices; de Rham exact sequence | �?|
| 30 | `solver` | Newmark-β time integrator; mfem_ex10_wave_equation example | �?|
| 31 | `element` | Gauss-Lobatto quadrature (seg, quad, hex) | �?|
| 32 | `examples` | mfem_ex4_darcy (H(div) RT0), mfem_ex15_dg_amr (P1 + ZZ + Dörfler) | �?|
| 33a-e | `parallel` | jsmpi backend, DofPartition, ParVector, ParCsrMatrix, ParAssembler, par_solve_pcg_jacobi, pex1 | �?|
| 34 | `parallel` | P2 parallel spaces (DofPartition::from_dof_manager, edge DOF ownership, auto-permute) | �?|
| 35 | `parallel` | Parallel AMG (ParAmgHierarchy, smoothed aggregation, par_solve_pcg_amg) | �?|
| 36 | `parallel` | Comm::split sub-communicators | �?|
| 37 | `parallel`+`wasm` | WASM multi-Worker (spawn_async, jsmpi_main), streaming mesh partition (partition_simplex_streaming), binary mesh serde | �?|
| 38 | `parallel` | METIS streaming partition (partition_simplex_metis_streaming), generalized submesh extractor, pex1 CLI flags | �?|
| 38b | `io` | GMSH v2 ASCII + v4.1 binary reader (unified `read_msh_file()` entry point) | �?|
| 39 | `parallel`+`examples` | pex2 (mixed Poisson), pex3 (Maxwell), pex5 (Darcy) parallel examples | �?|
| 39b | `amg` | Chebyshev smoother (`SmootherType::Chebyshev`), F-cycle (`CycleType::F`) | �?|
| 40 | `examples`+`assembly` | Taylor-Hood P2-P1 Stokes (`mfem_ex40` lid-driven cavity) | �?|
| 42a | `mesh`+`space`+`io` | Mixed element mesh infrastructure (per-element types, variable DofManager, GMSH mixed read) | �?|
| 44 | `assembly`+`examples` | VectorConvectionIntegrator + Navier-Stokes Oseen/Picard (`mfem_ex19`, Kovasznay Re=40) | �?|
| 42b | `assembly` | Quad4/Hex8 isoparametric Jacobian, `unit_square_quad`, Q1 Poisson verified | �?|
| 45 | `wasm`+`e2e` | Browser E2E test: WASM Poisson solver verified via Playwright/Chromium | �?|
| 46 | `mesh`+`linalg`+`solver`+`space`+`io` | Backlog: bounding_box, periodic mesh, DenseTensor, SLI, H1Trace, VTK reader, PrintLevel | �?|
| 47 | `mesh`+`space` | NCMesh: Tri3/Tet4 nonconforming refine + hanging constraints + `NCState`/`NCState3D` multi-level + P2 prolongation | �?|
| 48 | `element`+`space`+`assembly`+`solver`+`io` | linger update: sparse direct solvers (SparseLu/Cholesky/LDLt), IDR(s), TFQMR, ILDLt precond, KrylovSchur eigen, Matrix Market I/O; higher-order elements: TriP3, TetP2, TetP3, QuadQ2; H1TraceSpace P2/P3; Grundmann-Moller quadrature fix | �?|
| 49 | `element`+`space`+`assembly` | TriND2/TetND2 (Nédélec-I order 2); TriRT1/TetRT1 (Raviart-Thomas order 1); HCurlSpace/HDivSpace multi-order support; VectorAssembler factory updated | �?|

---

## Remaining Items Summary (🔲 Planned · 🔨 Partial)

### Mesh
| Item | Status | Priority |
|------|--------|----------|
| Mixed element meshes (Tri+Quad, Tet+Hex) | �?| ~~Medium~~ Done |
| NCMesh (non-conforming, hanging nodes) | �?| ~~Low~~ Done |
| `bdr_attributes` dedup utility | �?| ~~Low~~ Done |
| `ElementTransformation` type | �?| ~~Low~~ Done |
| `GetBoundingBox()` | �?| ~~Low~~ Done |
| Periodic mesh generation | �?| ~~Low~~ Done |

### I/O
| Item | Status | Priority |
|------|--------|----------|
| ~~GMSH v4.1 binary reader~~ | �?| ~~High~~ Done |
| ~~GMSH v2 reader~~ | �?| ~~Medium~~ Done |
| HDF5/XDMF parallel I/O | 🔨 | Medium |
| Netgen `.vol` reader | 🔨 (Tet4/Hex8 ASCII baseline，支�?mixed；写出与更多 section 保真待补�? | Low |
| Abaqus `.inp` reader | 🔨 (C3D4/C3D8 baseline，支�?mixed；更�?section/tag 保真待补�? | Low |
| `GridFunction::Load()` | �?| ~~Low~~ Done |
| Restart files (checkpoint) | 🔨 | Low |

### Solvers
| Item | Status | Priority |
|------|--------|----------|
| Chebyshev smoother (AMG) | �?| ~~Medium~~ Done |
| SLISolver (stationary iteration) | �?| ~~Low~~ Done |
| AMG F-cycle | �?| ~~Low~~ Done |
| hypre-equivalent AMG path | �?(pure-Rust in `vendor/linger`) | Low |

### Spaces & Post-processing
| Item | Status | Priority |
|------|--------|----------|
| H1_Trace_FECollection | �?| ~~Low~~ Done |
| Taylor-Hood P2-P1 | Stokes flow | �?`mfem_ex40` (lid-driven cavity) |
| Kelly error estimator | �?| ~~Low~~ Done |
| `DenseTensor` | �?| ~~Low~~ Done |
| `SetSubVector` slice assignment | �?| ~~Low~~ Done |

### Parallel Examples
| Item | Status | Priority |
|------|--------|----------|
| pex2 (parallel mixed Poisson) | �?| ~~Medium~~ Done |
| pex3 (parallel Maxwell) | �?| ~~Medium~~ Done |
| pex5 (parallel Darcy) | �?| ~~Medium~~ Done |
| ex19 (Navier-Stokes) | �?| ~~Medium~~ Done |
| Browser E2E (WASM) | �?| ~~Medium~~ Done |

---

## Recommended Roadmap (Phase 39+)

Based on the completed 38 phases and remaining gaps, here is a recommended
prioritized roadmap for continued development.

### Phase 39 �?Parallel Examples Expansion (pex2 / pex3 / pex5) �?
> **Completed** �?validates parallel infrastructure across all FE spaces

| Task | Space | Status |
|------|-------|--------|
| `mfem_pex2_mixed_darcy` | H(div) RT0 × L² | �?|
| `mfem_pex3_maxwell` | H(curl) ND1 | �?|
| `mfem_pex5_darcy` | H(div) × L² saddle-point | �?|

### Phase 39b �?Chebyshev Smoother + AMG F-cycle �?
> **Completed** �?smoother quality directly impacts AMG convergence

- �?Chebyshev polynomial smoother (degree 2�?) as `SmootherType::Chebyshev`
- �?Eigenvalue estimate via spectral radius bound (λ_max)
- �?F-cycle: `CycleType::F` (V on first coarse visit, W after)
- �?Tests: Chebyshev, F-cycle, Chebyshev+F-cycle combinations

### Phase 40 �?Taylor-Hood P2-P1 Stokes Example �?
> **Completed** �?demonstrates mixed FEM at production quality

- �?`mfem_ex40` example: lid-driven cavity on [0,1]²
- �?P2 velocity + P1 pressure via `MixedAssembler`
- �?Block saddle-point solver (SchurComplementSolver with GMRES)
- �?Verified convergence at n=8,16,32; divergence-free to solver tolerance

### Phase 42 �?Mixed Element Meshes (42a �? 42b �?
> **Completed** �?data structures, I/O, and assembly all done

- �?Per-element `ElementType` and CSR-like offset arrays in `SimplexMesh`
- �?Variable-stride `DofManager` for P1 on mixed meshes
- �?GMSH reader preserves mixed element types (Tri+Quad, Tet+Hex)
- �?Isoparametric Jacobian for Quad4/Hex8 in assembler (bilinear/trilinear mapping)
- �?`unit_square_quad(n)` mesh generator + Q1 Poisson convergence verified

### Phase 43 �?HDF5/XDMF Parallel I/O
> **Priority: Medium** �?needed for large-scale checkpointing

- [x] 新增独立 crate：`fem-io-hdf5-parallel`（feature-gated `hdf5`�?
- [x] 写入：rank-partition checkpoint（`/steps/step_xxxxxxxx/partitions/rank_xxxxxx/*`�?
- [x] 读取：按 step / latest �?rank-local restart 读取
- [x] 全局场拼装：`materialize_global_field_f64()`（供可视化）
- [x] XDMF sidecar：`write_xdmf_polyvertex_scalar_sidecar()`
- [x] XDMF time-series：`write_xdmf_polyvertex_scalar_timeseries_sidecar()`
- [x] 示例：`mfem_ex43_hdf5_checkpoint.rs`（无 HDF5 环境时优雅降级）
- [x] checkpoint 完整性校验：`validate_checkpoint_layout()`
- [x] MPI backend 已升级为 MPI 协同路径（rank 写入 + direct hyperslab 全局写入路径，保�?root 全局物化兼容兜底�?
- [x] direct hyperslab 读路径：`read_global_field_f64()` + `read_global_field_slice_f64()`（全局整场/切片读取�?
- [x] 并行 mesh+field bundle checkpoint schema（`CheckpointBundleF64` + `CheckpointMeshMeta` baseline�?

### Phase 44 �?Navier-Stokes (Kovasznay flow) �?
> **Completed** �?flagship nonlinear PDE example

- �?`VectorConvectionIntegrator`: `�?(w·�?u · v dx` for vector fields
- �?Oseen linearization with Picard iteration
- �?`mfem_ex19` example: Kovasznay flow benchmark (Re=40)
- �?Taylor-Hood P2/P1 discretization (reuses Phase 40 infrastructure)
- �?Converges in ~16�?0 Picard iterations; velocity error decreases with h-refinement

### Phase 45 �?Browser E2E (WASM) �?
> **Completed** �?validates the full browser pipeline

- �?Playwright/Chromium E2E test (`crates/wasm/e2e/`)
- �?WASM Poisson solver: assemble �?solve �?verify in browser
- �?Solution validated against analytical max (0.0737 for −Δu=1)

### Phase 46 �?Backlog Cleanup �?
> **Completed** �?9 remaining items resolved

- �?`SimplexMesh::bounding_box()` �?axis-aligned bounding box (2-D / 3-D)
- �?`SimplexMesh::unique_boundary_tags()` �?sorted/deduped boundary tag set
- �?`SimplexMesh::make_periodic()` �?node merging for periodic BCs
- �?`DenseTensor` �?3-D row-major tensor with slab access
- �?`solve_jacobi_sli()` / `solve_gs_sli()` �?Jacobi/GS stationary iteration
- �?`H1TraceSpace` �?H½ trace of H¹ on boundary faces (P1)
- �?`read_vtu_point_data()` �?VTK `.vtu` ASCII reader for solution loading
- �?`PrintLevel` enum �?structured solver verbosity (Silent/Summary/Iterations/Debug)
- �?`kelly_estimator()` was already implemented �?marked in MFEM_MAPPING
- �?`SetSubVector` / `GetSubVector` were already implemented �?marked in MFEM_MAPPING

### Phase 47 �?NCMesh (Non-Conforming Mesh / Hanging Nodes) �?
> **Completed** �?2-D Tri3 + 3-D Tet4 non-conforming refinement with multi-level state tracking

#### 2-D (Tri3) Hanging Edge Constraints
- �?`refine_nonconforming()` �?red-refines only marked elements, no propagation
- �?`HangingNodeConstraint` detection �?identifies midpoints on coarse/fine edges
- �?`apply_hanging_constraints()` �?P^T K P static condensation via COO rebuild
- �?`recover_hanging_values()` �?post-solve interpolation for constrained DOFs
- �?`NCState` multi-level constraint tracking �?carries and resolves hanging constraints across successive NC refinements
- �?`prolongate_p2_hanging()` �?P2 hanging-node prolongation by coarse P2 field evaluation at fine DOF coordinates
- �?`mfem_ex15_dg_amr --nc` �?demonstrates single-level NC AMR with error reduction

#### 3-D (Tet4) Hanging Face Constraints
- �?`HangingFaceConstraint` struct �?records hanging coarse faces and representative midpoint nodes
- �?`refine_nonconforming_3d(mesh, marked)` �?red-refines Tet4 elements into 8 children using edge midpoints
- �?`local_faces_tet()` �?helper returns 4 triangular face local indices for Tet4
- �?`face_key_3d()` �?canonical face key (sorted triplet) for face uniqueness
- �?Hanging-face detection �?detects refined/coarse Tet4 face mismatch and emits hanging edge constraints
- �?`NCState3D` multi-level tracking �?carries active edge midpoints and rebuilds constraints across levels
- �?Boundary face reconstruction �?preserves and refines Tri3 boundary faces in 3-D refinement
- �?Unit tests �?`tet4_nonconforming_refine_single_element()`, `tet4_nonconforming_refine_with_neighbor()`, `ncstate3d_two_level_refine()`

### Backlog (Low Priority)
| Item | Phase | Notes |
|------|-------|-------|
| hypre-equivalent AMG path | pure-Rust parity track | Owned by `vendor/linger` capability roadmap |
| Abaqus/Netgen format扩展（混合单元、更多section/tag保真�?| TBD | Additional mesh import formats |
| HDF5/XDMF I/O | TBD | Large-scale checkpointing |
| Restart files | TBD | Requires HDF5 |
| Tet4 NC AMR example | �?| ~~TBD~~ Done (`mfem_ex15_tet_nc_amr`, supports `--solve`) |

### Decision Log (2026-04-13)

- `hypre` capability is tracked as a pure-Rust parity roadmap item (no external `hypre-ffi` dependency), owned by `vendor/linger` and consumed by `fem-rs`.
- GPU backend is tracked as a cross-subproject roadmap item:
   - `vendor/linger`: backend-neutral kernel interfaces and numeric primitive contracts.
   - `vendor/reed`: GPU backend implementation and CEED-style operator/resource mapping.
   - `vendor/jsmpi`: browser-side multi-rank transport/runtime for wasm deployments.
- External solver delivery is coordinated across subprojects:
   - `vendor/linger`: pure-Rust HYPRE-equivalent + PETSc-equivalent solver lifecycle; `mumps`/`mkl` are compatibility contracts backed by native linger direct solves.
   - `vendor/reed`: operator/export bridge and backend selection wiring.
   - `vendor/jsmpi`: wasm/browser runtime constraints for distributed execution path.
- Current `linger` gaps to track under this ownership:
   - Distributed-memory path is still missing (`mpi` feature is placeholder in `vendor/linger/Cargo.toml`).
   - HYPRE-equivalent advanced options: AMS/ADS baseline is already available in `vendor/linger`; AIR baseline strategy is landed (`CoarsenStrategy::Air` + diagonal-`A_ff` AIR restriction) with nonsymmetric regression coverage (`amg_air_gmres_nonsymmetric_convdiff_1d`), while parity hardening (especially distributed/high-scale behavior) remains pending.
   - PETSc-equivalent KSP/PC path still needs pure-Rust completion in `vendor/linger`.
   - Direct-compatibility hooks: `mumps` / `mkl` 均具备可用 baseline（native multifrontal-backed, factor reuse + multi-RHS）；二者均由 linger 原生直接法承载，不以外部 FFI/distributed 接入为目标。
   - AMG options are narrower than hypre BoomerAMG/AIR ecosystem (currently RS/SA + V/W/F/K-cycle baseline).
   - GPU execution backend is missing in `linger` core (implementation track owned by `vendor/reed`).
   - Matrix Market complex field I/O is not yet supported (`vendor/linger/src/sparse/mmio.rs`).

### Cross-Subproject Improvement Plan (2026-Q2 to 2026-Q4)

> Scope: coordinated delivery across `vendor/linger`, `vendor/reed`, and `vendor/jsmpi`.

| Stage | Window | linger | reed | jsmpi | Exit Criteria |
|---|---|---|---|---|---|
| C1 Foundation | Q2 (2-4 weeks) | External solver abstraction, error adapter, feature-gated fallback | Stable operator/export bridge API to linger | Browser/wasm backend capability policy (supported vs fallback) | API boundary frozen; default build unchanged |
| C2 External Solvers M1/M2 | Q2-Q3 | pure-Rust HYPRE-equivalent minimal BoomerAMG baseline, then AIR + AMS/ADS parity hardening（AMS/ADS baseline already in `linger`�? `mumps` first direct path | Builder wiring for backend selection in FEM solve paths | wasm path reports deterministic fallback when native external backends unavailable | Poisson SPD integration tests pass for enabled backends |
| C3 GPU First Usable Path | Q3 | Backend-neutral kernel interface + CPU reference kernels | GPU backend implementation + CEED-style object mapping + one end-to-end example | Browser multi-rank transport constraints documented for GPU+wasm modes | One representative solve path runs CPU/GPU with same app API |
| C4 Portfolio Completion | Q4 | pure-Rust PETSc-equivalent KSP/PC path; CI matrix hooks | cross-backend regression tests in FEM pipelines | Browser smoke tests and fallback matrix by feature | CI passes on feature matrix; docs and examples complete |

#### Work Packages

- [x] WP1: Interface freeze for cross-project backend contracts
- [ ] WP2: pure-Rust HYPRE-equivalent AIR + AMS/ADS parity hardening（`linger` �?AMS/ADS baseline 已可用，AIR baseline 已落地，仍需 parity/分布式能力补齐）
- [x] WP3: `mumps` + `mkl` usable with factor reuse and multi-RHS（baseline：`linger::{MumpsSolver, MklSolver}` + `solve_sparse_{mumps,mkl}`；二者均为 linger 原生直接法的兼容入口）
- [ ] WP4: GPU baseline delivery in `reed` (with `linger` backend-neutral kernel contracts)
- [ ] WP5: pure-Rust PETSc-equivalent KSP/PC in `linger` + CI feature matrix
- [ ] WP6: `jsmpi` browser/wasm fallback and smoke-test closure

WP1 kickoff artifact merged: `C1_BACKEND_CONTRACT_FREEZE.md` (v0.1).

Current baseline progress (2026-04-13):
- Added canonical backend-resource smoke coverage in `fem-ceed` for `/solver/hypre-rs`, `/solver/petsc-rs`, `/solver/mumps`, `/solver/mkl` deterministic resolution/report path.
- Added CI gate `.github/workflows/alignment-smoke.yml` to run targeted smoke tests for:
   - complex coefficient traits (`fem-assembly`)
   - named attribute set baseline (`fem-mesh`)
   - canonical backend resource contract (`fem-ceed`)
- Added CI gate `.github/workflows/backend-feature-matrix.yml` to validate `vendor/reed` backend contract tests across feature profiles:
   - baseline (`--no-default-features`)
   - `hypre-rs`, `petsc-rs`, `mumps`, `mkl`

#### Coordination Rules

- One feature branch per stage, three subprojects use the same stage tag (`C1`/`C2`/`C3`/`C4`).
- No app-level API churn in `fem-rs` during stages; changes are behind feature flags.
- A stage is accepted only when all three subprojects satisfy the stage exit criteria.

---

## MFEM v4.9 Gap Analysis (2026-04-13)

> 对比基准：MFEM v4.9�?025-12-11）�?最新版本�?
> 以下差距按优先级排列，高优先级直接影响物理覆盖面，低优先级是工程完善项�?

### 差距汇总表

| 能力领域 | MFEM v4.9 | fem-rs | 差距等级 | 对应 Phase |
|---|---|---|---|---|
| 复数�?FEM | �?ex22/ex25/DPG | �?基线已实现（2×2 实块�?| 🟡 �?| 55 |
| IMEX 时间积分 | �?ex41 | �?基线已实现（Euler/SSP2/RK3/ARK3�?| 🟡 �?| 56 |
| AMR 反细�?(Derefinement) | �?ex15 | �?基线已实现（single-level rollback�?| 🟡 �?| 57 |
| 几何多重网格 / LOR 预条件器 | �?ex26 | 🔨 LOR + GeomMG 基线 | 🟡 �?| 58 |
| SubMesh 子域传输 | �?ex34/ex35 | �?基线已实�?| 🟡 �?| 59 |
| DG 弹性力�?| �?ex17 | �?基线已实�?| 🟡 �?| 60 |
| DG 可压�?Euler 方程 | �?ex18 | �?1D 基线已实�?| 🟡 �?| 60 |
| 辛时间积�?(Symplectic) | �?ex20 | �?已实�?| 🟡 �?| 61 |
| 受限 H(curl) 空间 (1D/2D embedded) | �?ex31/ex32 | �?基线已实现；ex31/ex32 均已补充制造解一阶收敛回�?| 🟡 �?| 62 |
| PML 完美匹配�?| �?ex25 | 🔨 标量+各向异性张量基线（ex25 已加入可量化反射指标与强度回归） | 🟡 �?| 55+63 |
| 静态凝�?/ 杂化 | �?ex4/ex8/hybr | 🔨 代数静态凝聚基线（`mfem_ex8_hybridization`，基�?hanging constraints）；混合/杂化 FEM 内核待补�?| 🟢 �?| TBD |
| 分数�?Laplacian | �?ex33 | 🔨 `mfem_ex33_fractional_laplacian` dense spectral FE 基线；可扩展有理逼近 / extension 路线待补�?| 🟢 �?| TBD |
| 障碍问题 / 变分不等�?| �?ex36 | 🔨 `mfem_ex36_obstacle` primal-dual active-set (PDAS) 基线；semismooth Newton 内核待补�?| 🟢 �?| TBD |
| 拓扑优化 | �?ex37 | 🔨 `mfem_ex37_topology_optimization` 标量 SIMP compliance 基线（density filter + Heaviside projection + chain-rule sensitivity）；全弹�?伴随/约束扩展待补�?| 🟢 �?| TBD |
| 截断积分 / 浸没边界 | �?ex38 | 🔨 `mfem_ex38_immersed_boundary` cut-cell subtriangulation + Nitsche-like �?Dirichlet（弦段近似）浸没边界基线；完�?cut-FEM/level-set 稳健几何与高阶界面积分待补齐 | 🟢 �?| TBD |
| 命名属性集 | �?ex39 | 🔨 named tag registry + mesh/submesh named selection + GMSH `PhysicalNames` bridge + `mfem_ex39_named_attributes` baseline | 🟢 �?| TBD |
| Quad/Hex NC AMR（各向异性） | �?| 🔨 Tri/Tet only | 🟢 �?| TBD |
| GPU 后端 (CUDA/HIP) | �?全库加�?| �?core CPU only（delegated to `vendor/linger` + `vendor/reed` + `vendor/jsmpi` 协同�?| 🟢 �?| TBD |

---

### Phase 55 �?复数�?FEM（Complex-Valued Systems）�?
> **Target**: MFEM ex22 (时谐阻尼振荡�? + ex25 (PML Maxwell)
>
> 对应 MFEM `ComplexOperator` / `ComplexGridFunction` 实现模式

**问题**：时�?Maxwell �?Helmholtz 方程含复数系数：
```
∇�?a∇×u) �?ω²b·u + iωc·u = 0   (H(curl), 时谐电磁)
−∇·(a∇u) �?ω²b·u + iωc·u = 0   (H¹, 时谐声学)
```

**实现策略** �?2×2 实块方案（不引入复数泛型，WASM 兼容）：
```
[K - ω²M    -ωC ] [u_re]   [f_re]
[ωC          K-ω²M] [u_im] = [f_im]
```
其中 `K = stiffness`, `M = mass`, `C = damping`�?

**任务清单**�?
- [x] `ComplexAssembler` �?同时组装实部/虚部矩阵�?×2 实块系统�?
- [x] `ComplexCoeff` / `ComplexVectorCoeff` �?复系�?trait（re/im 两路，`coefficient.rs` 已提�?baseline�?
- [x] `ComplexLinearForm` �?�?�?RHS 向量�?
- [x] `apply_dirichlet_complex()` �?复数 Dirichlet BC 消去（`ComplexSystem::apply_dirichlet`�?
- [x] `GMRES` on `BlockMatrix` �?通过 flatten �?GMRES 路径求解
- [x] `mfem_ex22.rs` �?高保真增强：右边界一阶吸收边界（ABC�? 透射 proxy 回归测试
- [x] `mfem_ex25.rs` �?PML-like complex Helmholtz 基线示例

---

### Phase 56 �?IMEX 时间积分（Implicit-Explicit Splitting）�?
> **Target**: MFEM ex41 (DG/CG IMEX advection-diffusion)
>
> 对应 MFEM `TimeDependentOperator` �?additive 分裂模式

**问题**：对�?扩散方程�?
```
∂u/∂t + v·∇u �?∇�?κ∇u) = 0
```
对流�?`v·∇u` 需显式（CFL 限制），扩散�?`∇�?κ∇u)` 需隐式（稳定性）�?

**任务清单**�?
- [x] `ImexOperator` trait �?分拆�?`explicit_part()` + `implicit_part()`（已�?`fem_solver::ode` 提供�?
- [x] `ImexEuler` (IMEX Euler: forward for explicit, backward for implicit)
- [x] `ImexRK2` (IMEX-SSP-RK2 / Ascher-Ruuth-Spiteri 2-stage)
- [x] `ImexRK3`（固定步长三阶基线，API: `ImexRk3` + `ImexTimeStepper::integrate_rk3`�?
- [x] `ImexTimeStepper` �?统一 driver，复�?`ImplicitTimeStepper` 接口
- [x] `mfem_ex41_imex.rs` �?advection-diffusion IMEX 示例，对比纯显式 RK45

---

### Phase 57 �?AMR 反细化（Mesh Derefinement / Coarsening）✅
> **Target**: MFEM ex15 动�?AMR（refine + derefine + rebalance 循环�?

**状�?*：已完成�?026-04-12�?

**实现**（Tri3 conforming 版本）：
- [x] `DerefineTree` �?记录精化历史（父→子元素映射，已支持单层 red-refinement 回退�?
- [x] `mark_for_derefinement()` �?基于 ZZ/Kelly 估计量标记可缩粗元素
- [x] `derefine_marked(mesh, tree, marked)` �?�?4 子三角形合并回父三角形（当前为单层回退版本�?
- [x] 解插值：`restrict_to_coarse()` �?已提�?`restrict_to_coarse_p1()`（P1 节点注入版本�?
- [x] `NCState` / `NCState3D` 中的反细化路径（已支持按�?rollback �?`derefine_last()`�?
- [x] `mfem_ex15_dynamic_amr.rs` �?动�?AMR 演示（已覆盖 refine + derefine + prolongate + restrict 基础闭环�?

---

### Phase 58 �?几何多重网格 / LOR 预条件器�?
> **Target**: MFEM ex26 (Multigrid preconditioner for high-order Poisson)

**状�?*：已完成�?026-04-12�?

**实现**（两条路线均可用）：

1. **几何 h-多重网格** �?利用网格细化层次，每层使�?`AmgSolver` 作平滑器
   - [x] `GeomMGHierarchy` �?存储层级矩阵 + Restriction/Prolongation（基线版�?
   - [x] `GeomMGPrecond` �?V-cycle 实现（Jacobi smoother + coarse CG�?
   - [x] `mfem_ex26_geom_mg.rs` �?几何多重网格基线示例�?D nested hierarchy smoke�?

2. **LOR 预条件器**（更实用�?
   - [x] `LorMesh::from_high_order(mesh, p)` �?构�?p 次细化的 P1 等效网格
   - [x] `LorPrecond::new(...)` �?LOR 预条件器配置入口
   - [x] `solve_pcg_lor()` �?PCG+Jacobi 后端
   - [x] `solve_gmres_lor()` �?GMRES 后端

---

### Phase 59 �?SubMesh 子域传输�?
> **Target**: MFEM ex34 (SubMesh source function), ex35 (port BCs)

**状�?*：已完成�?026-04-12�?

**实现**�?
- [x] `SubMesh::extract(mesh, element_tags)` �?从标签提取子网格（Tri3�?
- [x] `SubMesh::transfer_to_parent(gf)` �?子域 FE 函数 �?父网�?
- [x] `SubMesh::transfer_from_parent(gf)` �?父网�?�?子域
- [x] 多物理耦合示例基础（Joule 加热框架可用�?

---

### Phase 60 �?DG 弹�?+ 可压缩流�?
> **Target**: MFEM ex17 (DG elasticity), ex18 (DG Euler equations)

**状�?*：已完成�?026-04-12�?

**实现**�?
- [x] `DgElasticityAssembler` �?向量块对�?SIP
- [x] `HyperbolicFormIntegrator` �?守恒律通量 + Lax-Friedrichs/Roe
- [x] `mfem_ex17_dg_elasticity.rs` �?DG 弹性基础示例
- [x] `mfem_ex18_euler.rs` �?Euler + SSPRK2

---

### Phase 61 �?辛时间积分✅
> **Target**: MFEM ex20 (symplectic integration of Hamiltonian systems)

**状�?*：已完成�?026-04-12�?

**实现**�?
- [x] `HamiltonianSystem` trait �?dH/dp + dH/dq
- [x] `VerletStepper`, `Leapfrog`, `Yoshida4` 辛积分器
- [x] 能量守恒验证（标准谐振子�?

---

### Phase 62 �?受限 H(curl) 空间�?
> **Target**: MFEM ex31 (anisotropic Maxwell), ex32 (anisotropic Maxwell eigenproblem)

**状�?*：已完成�?026-04-12�?

**实现**�?
- [x] 2D 网格上嵌�?3D 向量场接�?
- [x] `RestrictedHCurlSpace` �?低维网格高维 H(curl) DOF
- [x] `mfem_ex31.rs` �?各向异�?Maxwell 制造解示例 + 一阶收敛趋势回�?
- [x] `mfem_ex32.rs` �?阻抗边界 Maxwell 制造解示例 + 一阶收敛趋势回�?

---

### Phase 63 �?PML 完美匹配层与电磁各向异性✅
> **Target**: MFEM ex25 (PML), ex3/ex34 anisotropic variants

**状�?*：已完成�?026-04-13�?

**实现**�?
- [x] `PmlCoeff` �?标量层吸收系数（边界层衰减）
- [x] `PmlTensorCoeff` �?对角张量 PML 接口
- [x] `mfem_ex25.rs` �?complex Helmholtz PML 示例（反�?proxy 指标 + `sigma_max/power` + `stretch_blend` 联合回归�?
- [x] `mfem_ex3 --pml-like` �?H(curl) 各向异�?PML-like 阻尼（wx/wy 控制，含 strong/weak `sigma_max` �?`||u||₂` 回归�?
- [x] `mfem_ex34 --anisotropic` �?各向异性吸收边界（gamma_x/gamma_y 控制，已加入制造解误差回归与细化单调下降校验）
- [x] alignment-smoke CI：electromagnetic-pml、electromagnetic-absorbing �?suite

### Phase 48 �?linger Update + Higher-Order Elements �?
> **Completed** �?sparse direct solvers, new Krylov methods, higher-order FEM

- �?Sparse direct solvers: `SparseLu`, `SparseCholesky`, `SparseLdlt` (pure-Rust, WASM-compatible)
- �?New iterative methods: `IDR(s)` (`solve_idrs`), `TFQMR` (`solve_tfqmr`)
- �?New preconditioner: `ILDLt` (`solve_pcg_ildlt`, `solve_gmres_ildlt`) for symmetric indefinite
- �?KrylovSchur eigenvalue solver (`krylov_schur`) �?thick-restart Arnoldi
- �?Matrix Market I/O: `read_matrix_market`, `read_matrix_market_coo`, `write_matrix_market`
- �?Higher-order elements: `TriP3`, `TetP2`, `TetP3`, `QuadQ2`, `SegP3` �?fully registered
- �?H1TraceSpace P2/P3 boundary trace support
- �?Grundmann-Moller tet quadrature fix (linear system solver, correct for all orders)
- �?reed submodule bug fix (`create_basis_h1_simplex` lock pattern)

---

### Phase 64 �?多材�?PML 演示 (ex3 增强) �?
> **Target**: MFEM ex3 的增强变体，展示多区�?PML 系数控制

**状�?*：已完成�?026-04-13�?

**实现**�?
- [x] `mfem_ex3 --multi-material` �?4 象限各向异�?PML，每个区域独�?(wx, wy) 配置
- [x] `multi_material_pml_tensor()` 函数 �?基于坐标的分区系�?[Q1: 1.0/1.2, Q2: 0.9/1.1, Q3: 0.8/1.3, Q4: 1.2/0.9]
- [x] 测试：`ex3_multi_material_pml_mode_converges()` 验证 158 次迭代收�?
- [x] 验证：n=8, residual<1e-6

### Phase 65 �?并行 Maxwell PML (pex3 增强) �?
> **Target**: 并行 H(curl) 例子集成 PML-like 系数

**状�?*：已完成�?026-04-13�?

**实现**�?
- [x] `mfem_pex3_maxwell.rs --pml` �?并行 ND1 Maxwell 支持 PML 模式
- [x] `VectorMassTensorIntegrator<ConstantMatrixCoeff>` �?张量质量矩阵集成
- [x] `pml_mass_tensor()` 函数 �?生成 [1+σ, 0; 0, 1+σ] 各向同性阻尼张�?
- [x] 验证�? rank, n=8, 64 iters, residual<1e-8 收敛

### Phase 66 �?命名属性集合运�?(ex39 增强) �?
> **Target**: MFEM ex39 的集合运算扩展（并集、交集、差集）

**状�?*：已完成�?026-04-13�?

**实现**�?
- [x] `mfem_ex39_named_attributes.rs --intersection-region` �?集合交集（inlet �?outlet�?
- [x] `mfem_ex39_named_attributes.rs --difference-region` �?集合差集（inlet \ outlet�?
- [x] 测试三个场景：merge (�?、intersection (�?、difference (\)
- [x] 验证�? 个测试通过，演示多集合布尔运算模式

---

## 例子命名迁移记录 (2026-04-13)

为实�?**MFEM 对应关系清晰�?* �?**命名规范统一**，所�?`ex_` 前缀的应�?增强例子迁移�?`mfem_ex<N>_<variant>` 格式�?

| 旧名�?| 新名�?| MFEM 对应 | Phase | 描述 |
|---|---|---|---|---|
| `ex_stokes.rs` | `mfem_ex40.rs` | MFEM ex40 | 40 | Taylor-Hood P2-P1 盖驱动腔 |
| `ex_navier_stokes.rs` | `mfem_ex19.rs` | MFEM ex19 | 44 | Kovasznay 流不可压�?Navier-Stokes |
| `ex_maxwell_eigenvalue.rs` | `mfem_ex13_eigenvalue.rs` | MFEM ex13 | �?| H(curl) 特征值问�?(LOBPCG，含细化后首�?最大相对误差改善回�? |
| `ex_maxwell_time.rs` | `mfem_ex10_maxwell_time.rs` | MFEM ex10 | �?| 时间�?Maxwell (Newmark-β，已提取 `solve_case` 并补充时间步�?阻尼回归 + 时间自收敛二阶验�? |

**迁移完成**�?
- �?文件系统迁移（move 命令�?
- �?`examples/Cargo.toml` 更新�? �?[[example]] 配置�?
- �?编译验证（fem-examples lib 101/101 测试通过�?

**好处**�?
- 清晰�?MFEM 版本对应关系
- 统一的命名规范（`mfem_ex<number>` 格式�?
- 易于在文档和 CI 中引�?

