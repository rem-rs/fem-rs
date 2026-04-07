# fem-rs ↔ MFEM Correspondence Reference
> Tracks every major MFEM concept and its planned or implemented fem-rs counterpart.
> Use this as the authoritative target checklist for feature completeness.
>
> Status legend: ✅ implemented · 🔨 partial · 🔲 planned · ❌ out-of-scope

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
12. [MFEM Examples → fem-rs Milestones](#12-mfem-examples--fem-rs-milestones)
13. [Key Design Differences](#13-key-design-differences)

---

## 1. Mesh

### 1.1 Mesh Container

| MFEM class / concept | fem-rs equivalent | Status | Notes |
|---|---|---|---|
| `Mesh` (2D/3D unstructured) | `SimplexMesh<D>` | ✅ | Uniform element type per mesh |
| `Mesh` (mixed elements) | `SimplexMesh<D>` + `elem_types`/`elem_offsets` | 🔨 | Phase 42a: data structures + I/O done |
| `NCMesh` (non-conforming) | `refine_nonconforming()` (2-D) + `refine_nonconforming_3d()` + `NCState`/`NCState3D` | ✅ | Tri3/Tet4 multi-level non-conforming refinement + hanging constraints |
| `ParMesh` | `ParallelMesh<M>` | ✅ | Phase 10+33 |
| `Mesh::GetNV()` | `MeshTopology::n_nodes()` | ✅ | |
| `Mesh::GetNE()` | `MeshTopology::n_elements()` | ✅ | |
| `Mesh::GetNBE()` | `MeshTopology::n_boundary_faces()` | ✅ | |
| `Mesh::GetVerticesArray()` | `SimplexMesh::coords` (flat `Vec<f64>`) | ✅ | |
| `Mesh::GetElementVertices()` | `MeshTopology::element_nodes()` | ✅ | |
| `Mesh::GetBdrElementVertices()` | `MeshTopology::face_nodes()` | ✅ | |
| `Mesh::GetBdrAttribute()` | `MeshTopology::face_tag()` | ✅ | Tags match GMSH physical group IDs |
| `Mesh::GetAttribute()` | `MeshTopology::element_tag()` | ✅ | Material group tag |
| `Mesh::bdr_attributes` | `SimplexMesh::unique_boundary_tags()` | ✅ | Sorted, deduplicated boundary tag set |
| `Mesh::GetDim()` | `MeshTopology::dim()` | ✅ | Returns `u8` (2 or 3) |
| `Mesh::GetSpaceDim()` | same as `dim()` for flat meshes | ✅ | |
| `Mesh::UniformRefinement()` | `refine_uniform()` | ✅ | Red refinement (Tri3→4 children) |
| `Mesh::AdaptiveRefinement()` | `refine_marked()` + ZZ estimator + Dörfler marking | ✅ | Phase 17 |
| `Mesh::GetElementTransformation()` | `ElementTransformation` + inline Jacobian paths | 🔨 | Wrapper available for simplex; full assembler migration TBD |
| `Mesh::GetFaceElementTransformations()` | `InteriorFaceList` | ✅ | Used by DG assembler |
| `Mesh::GetBoundingBox()` | `SimplexMesh::bounding_box()` | ✅ | Returns `(min, max)` per axis |

### 1.2 Element Types

| MFEM element | `ElementType` variant | dim | Nodes | Status |
|---|---|---|---|---|
| `Segment` | `Line2` | 1 | 2 | ✅ |
| Quadratic segment | `Line3` | 1 | 3 | ✅ |
| `Triangle` | `Tri3` | 2 | 3 | ✅ |
| Quadratic triangle | `Tri6` | 2 | 6 | ✅ |
| `Quadrilateral` | `Quad4` | 2 | 4 | ✅ |
| Serendipity quad | `Quad8` | 2 | 8 | ✅ |
| `Tetrahedron` | `Tet4` | 3 | 4 | ✅ |
| Quadratic tet | `Tet10` | 3 | 10 | ✅ |
| `Hexahedron` | `Hex8` | 3 | 8 | ✅ |
| Serendipity hex | `Hex20` | 3 | 20 | ✅ |
| `Wedge` (prism) | `Prism6` | 3 | 6 | ✅ (type only) |
| `Pyramid` | `Pyramid5` | 3 | 5 | ✅ (type only) |
| `Point` | `Point1` | 0 | 1 | ✅ |

### 1.3 Mesh Generators

| MFEM generator | fem-rs equivalent | Status |
|---|---|---|
| `Mesh::MakeCartesian2D()` | `SimplexMesh::unit_square_tri(n)` | ✅ |
| `Mesh::MakeCartesian3D()` | `SimplexMesh::unit_cube_tet(n)` | ✅ | Added in Phase 9 |
| `Mesh::MakePeriodic()` | `SimplexMesh::make_periodic()` | ✅ | Node merging + face removal |
| Reading MFEM format | — | ❌ use GMSH instead |
| Reading GMSH `.msh` v4 | `fem_io::read_msh_file()` | ✅ |
| Reading Netgen | — | 🔲 Phase 9 |

---

## 2. Reference Elements & Quadrature

### 2.1 Reference Elements

| MFEM class | fem-rs trait/struct | Status |
|---|---|---|
| `FiniteElement` (base) | `ReferenceElement` trait | ✅ |
| `Poly_1D` utility | inline basis in `lagrange/` | ✅ |
| `H1_SegmentElement` | `lagrange::seg::P1Seg`, `P2Seg` | ✅ |
| `H1_TriangleElement` | `lagrange::tri::P1Tri`, `P2Tri` | ✅ |
| `H1_TetrahedronElement` | `lagrange::tet::P1Tet`, `P2Tet` | ✅ |
| `H1_QuadrilateralElement` | `lagrange::quad::Q1Quad` | ✅ |
| `H1_HexahedronElement` | `lagrange::hex::Q1Hex` | ✅ |
| `ND_TriangleElement` | `nedelec::TriND1` | ✅ |
| `ND_TetrahedronElement` | `nedelec::TetND1` | ✅ |
| `RT_TriangleElement` | `raviart_thomas::TriRT0` | ✅ |
| `RT_TetrahedronElement` | `raviart_thomas::TetRT0` | ✅ |
| `L2_TriangleElement` | L2Space with P0/P1 | ✅ |

### 2.2 Quadrature Rules

| MFEM class | fem-rs struct | Status |
|---|---|---|
| `IntegrationRule` | `QuadratureRule` | ✅ |
| `IntegrationRules` (table) | `quadrature.rs` look-up table | ✅ |
| Gauss-Legendre 1D (orders 1–10) | `gauss_legendre_1d(order)` | ✅ |
| Gauss-Legendre on triangle | `gauss_triangle(order)` | ✅ |
| Gauss-Legendre on tet | `gauss_tet(order)` | ✅ |
| Tensor product (quad, hex) | `tensor_gauss(order, dim)` | ✅ |
| Gauss-Lobatto | `gauss_lobatto_1d`, `seg_lobatto_rule`, `quad_lobatto_rule`, `hex_lobatto_rule` | ✅ |

---

## 3. Finite Element Spaces

### 3.1 Collections (Basis Families)

| MFEM collection | Mathematical space | fem-rs struct | Status |
|---|---|---|---|
| `H1_FECollection(p)` | H¹(Ω): C⁰ scalar Lagrange | `H1Space` | ✅ |
| `L2_FECollection(p)` | L²(Ω): discontinuous Lagrange | `L2Space` | ✅ |
| `DG_FECollection(p)` | L²(Ω): DG (element-interior only) | `L2Space` | ✅ |
| `ND_FECollection(p)` | H(curl): Nédélec tangential | `HCurlSpace` | ✅ |
| `RT_FECollection(p)` | H(div): Raviart-Thomas normal | `HDivSpace` | ✅ |
| `H1_Trace_FECollection` | H½: traces of H¹ on faces | `H1TraceSpace` | ✅ | P1 boundary trace |
| `NURBS_FECollection` | NURBS isogeometric | — | ❌ out of scope |

### 3.2 Finite Element Space (DOF management)

| MFEM method | fem-rs equivalent | Status |
|---|---|---|
| `FiniteElementSpace(mesh, fec)` | `H1Space::new(mesh)` etc. | ✅ |
| `FES::GetNDofs()` | `FESpace::n_dofs()` | ✅ |
| `FES::GetElementDofs()` | `FESpace::element_dofs()` | ✅ |
| `FES::GetBdrElementDofs()` | `boundary_dofs()` | ✅ |
| `FES::GetEssentialTrueDofs()` | `boundary_dofs()` + `apply_dirichlet()` | ✅ |
| `FES::GetTrueDofs()` | `DofPartition::n_owned_dofs` + `global_dof()` | ✅ | Phase 33b |
| `FES::TransferToTrue()` / `Transfer()` | `DofPartition::permute_dof()` / `unpermute_dof()` | ✅ | Phase 34 |
| `DofTransformation` | `FESpace::element_signs()` | ✅ | HCurlSpace/HDivSpace sign convention |
| `FES::GetFE()` | `FESpace::element_type()` | ✅ |

### 3.3 Space Types

| Space | Problem | Status |
|---|---|---|
| H¹ | Electrostatics, heat, elasticity (scalar) | ✅ |
| H(curl) | Maxwell, eddy currents (vector potential) | ✅ |
| H(div) | Darcy flow, mixed Poisson | ✅ |
| L² / DG | Transport, DG methods | ✅ |
| Vector H¹ = [H¹]ᵈ | Elasticity (displacement vector) | ✅ |
| Taylor-Hood P2-P1 | Stokes flow | ✅ Via MixedAssembler + `ex_stokes` |

---

## 4. Coefficients

MFEM provides a rich coefficient hierarchy for spatially- and
time-varying material properties.  fem-rs uses a trait-based system:
`ScalarCoeff`, `VectorCoeff`, `MatrixCoeff` traits with `f64` as the
default (zero-cost for constants).

| MFEM class | fem-rs | Status |
|---|---|---|
| `ConstantCoefficient(c)` | `f64` (implements `ScalarCoeff`) | ✅ |
| `FunctionCoefficient(f)` | `FnCoeff(\|x\| f(x))` | ✅ |
| `GridFunctionCoefficient` | `GridFunctionCoeff::new(dof_vec)` | ✅ |
| `PWConstCoefficient` | `PWConstCoeff::new([(tag, val), ...])` | ✅ |
| `PWCoefficient` | `PWCoeff::new(default).add_region(tag, coeff)` | ✅ |
| `VectorCoefficient` | `VectorCoeff` trait + `FnVectorCoeff`, `ConstantVectorCoeff` | ✅ |
| `MatrixCoefficient` | `MatrixCoeff` trait + `FnMatrixCoeff`, `ConstantMatrixCoeff`, `ScalarMatrixCoeff` | ✅ |
| `InnerProductCoefficient` | `InnerProductCoeff { a, b }` | ✅ |
| `TransformedCoefficient` | `TransformedCoeff { inner, transform }` | ✅ |

---

## 5. Assembly: Forms & Integrators

### 5.1 Bilinear Forms

| MFEM class | fem-rs equivalent | Status |
|---|---|---|
| `BilinearForm(fes)` | `Assembler::assemble_bilinear(integrators)` | ✅ |
| `BilinearForm::AddDomainIntegrator()` | `assembler.add_domain(integrator)` | ✅ |
| `BilinearForm::AddBoundaryIntegrator()` | `assembler.add_boundary(integrator)` | ✅ |
| `BilinearForm::Assemble()` | `Assembler::assemble_bilinear()` | ✅ |
| `BilinearForm::FormLinearSystem()` | `apply_dirichlet()` | ✅ |
| `BilinearForm::FormSystemMatrix()` | `apply_dirichlet()` variants | ✅ |
| `MixedBilinearForm(trial, test)` | `MixedAssembler` | ✅ |

### 5.2 Linear Forms

| MFEM class | fem-rs equivalent | Status |
|---|---|---|
| `LinearForm(fes)` | `Assembler::assemble_linear(integrators)` | ✅ |
| `LinearForm::AddDomainIntegrator()` | `assembler.add_domain_load(integrator)` | ✅ |
| `LinearForm::AddBndryIntegrator()` | `NeumannIntegrator` | ✅ |
| `LinearForm::Assemble()` | `Assembler::assemble_linear()` | ✅ |

### 5.3 Bilinear Integrators

| MFEM integrator | Bilinear form | fem-rs struct | Status |
|---|---|---|---|
| `DiffusionIntegrator(κ)` | ∫ κ ∇u·∇v dx | `DiffusionIntegrator` | ✅ |
| `MassIntegrator(ρ)` | ∫ ρ u v dx | `MassIntegrator` | ✅ |
| `ConvectionIntegrator(b)` | ∫ (b·∇u) v dx | `ConvectionIntegrator` | ✅ |
| `ElasticityIntegrator(λ,μ)` | ∫ σ(u):ε(v) dx | `ElasticityIntegrator` | ✅ |
| `CurlCurlIntegrator(μ)` | ∫ μ (∇×u)·(∇×v) dx | `CurlCurlIntegrator` | ✅ |
| `VectorFEMassIntegrator` | ∫ u·v dx (H(curl)/H(div)) | `VectorMassIntegrator` | ✅ |
| `DivDivIntegrator(κ)` | ∫ κ (∇·u)(∇·v) dx | `DivIntegrator` | ✅ |
| `VectorDiffusionIntegrator` | ∫ κ ∇uᵢ·∇vᵢ (vector Laplacian) | `VectorDiffusionIntegrator` | ✅ |
| `BoundaryMassIntegrator` | ∫_Γ α u v ds | `BoundaryMassIntegrator` | ✅ |
| `VectorFEDivergenceIntegrator` | ∫ (∇·u) q dx (Darcy/Stokes) | `PressureDivIntegrator` | ✅ |
| `GradDivIntegrator` | ∫ (∇·u)(∇·v) dx | `GradDivIntegrator` | ✅ |
| `DGDiffusionIntegrator` | Interior penalty DG diffusion | `DgAssembler::assemble_sip` | ✅ |
| `TransposeIntegrator` | Transposes a bilinear form | `TransposeIntegrator` | ✅ |
| `SumIntegrator` | Sum of integrators | `SumIntegrator` | ✅ |

### 5.4 Linear Integrators

| MFEM integrator | Linear form | fem-rs struct | Status |
|---|---|---|---|
| `DomainLFIntegrator(f)` | ∫ f v dx | `DomainSourceIntegrator` | ✅ |
| `BoundaryLFIntegrator(g)` | ∫_Γ g v ds | `NeumannIntegrator` | ✅ |
| `VectorDomainLFIntegrator` | ∫ **f**·**v** dx | `VectorDomainLFIntegrator` | ✅ |
| `BoundaryNormalLFIntegrator` | ∫_Γ g (n·v) ds | `BoundaryNormalLFIntegrator` | ✅ |
| `VectorFEBoundaryFluxLFIntegrator` | ∫_Γ f (v·n) ds (RT) | `VectorFEBoundaryFluxLFIntegrator` | ✅ |

### 5.5 Assembly Pipeline

| MFEM concept | fem-rs equivalent | Status |
|---|---|---|
| `ElementTransformation` | Jacobian `jac`, `det_j`, `jac_inv_t` | 🔨 inline in assembly |
| `Geometry::Type` | `ElementType` enum | ✅ |
| Sparsity pattern | `SparsityPattern` built once | ✅ |
| Parallel assembly | Element loop → ghost DOF AllReduce | ✅ via ChannelBackend |

---

## 6. Linear Algebra

### 6.1 Sparse Matrix

| MFEM class | fem-rs struct | Status |
|---|---|---|
| `SparseMatrix` (CSR) | `CsrMatrix<T>` | ✅ |
| `SparseMatrix::Add(i,j,v)` | `CooMatrix::add(i,j,v)` | ✅ |
| `SparseMatrix::Finalize()` | `CooMatrix::into_csr()` | ✅ |
| `SparseMatrix::Mult(x,y)` | `CsrMatrix::spmv(x,y)` | ✅ |
| `SparseMatrix::MultTranspose()` | `CsrMatrix::transpose()` + spmv | ✅ |
| `SparseMatrix::EliminateRowCol()` | `apply_dirichlet_symmetric()` | ✅ |
| `SparseMatrix::EliminateRow()` | `apply_dirichlet_row_zeroing()` | ✅ |
| `SparseMatrix::GetDiag()` | `CsrMatrix::diagonal()` | ✅ |
| `SparseMatrix::Transpose()` | `CsrMatrix::transpose()` | ✅ |
| `SparseMatrix::Add(A,B)` | `spadd(&A, &B)` | ✅ |
| `SparseMatrix::Mult(A,B)` | SpGEMM (via linger) | ✅ |
| `DenseMatrix` (local dense) | `nalgebra::SMatrix` | ✅ |
| `DenseTensor` | `DenseTensor` (3-D array) | ✅ | Row-major slab access |

### 6.2 Vector

| MFEM class | fem-rs struct | Status |
|---|---|---|
| `Vector` | `Vector<T>` | ✅ |
| `Vector::operator +=` | `Vector::axpy(1.0, x)` | ✅ |
| `Vector::operator *=` | `Vector::scale(a)` | ✅ |
| `Vector::operator * (dot)` | `Vector::dot()` | ✅ |
| `Vector::Norml2()` | `Vector::norm()` | ✅ |
| `Vector::Neg()` | `vector.scale(-1.0)` | ✅ |
| `Vector::SetSubVector()` | `Vector::set_sub_vector()` / `get_sub_vector()` | ✅ | Offset-based slice ops |
| `BlockVector` | `BlockVector` | ✅ |

---

## 7. Solvers & Preconditioners

### 7.1 Iterative Solvers

| MFEM solver | Problem type | fem-rs module | Status |
|---|---|---|---|
| `CGSolver` | SPD: A x = b | `solver` (via linger) | ✅ |
| `PCGSolver` | SPD + preconditioner | `solver` (PCG+Jacobi/ILU0) | ✅ |
| `GMRESSolver(m)` | General: A x = b | `solver` (via linger) | ✅ |
| `FGMRESSolver` | Flexible GMRES | `solve_fgmres` / `solve_fgmres_jacobi` | ✅ |
| `BiCGSTABSolver` | Non-symmetric | `solver` (via linger) | ✅ |
| `MINRESSolver` | Indefinite symmetric | `MinresSolver` | ✅ |
| `SLISolver` | Stationary linear iteration | `solve_jacobi_sli` / `solve_gs_sli` | ✅ |
| `NewtonSolver` | Nonlinear F(x)=0 | `NewtonSolver` | ✅ |
| `UMFPackSolver` | Direct (SuiteSparse) | `dense::lu_solve` (small systems) | ✅ |
| `MUMPSSolver` | Parallel direct | — | ❌ |

### 7.2 Preconditioners

| MFEM preconditioner | Type | fem-rs module | Status |
|---|---|---|---|
| `DSmoother` | Jacobi / diagonal scaling | PCG+Jacobi (via linger) | ✅ |
| `GSSmoother` | Gauss-Seidel | `SmootherKind::GaussSeidel` (AMG) | ✅ |
| Chebyshev smoother | Chebyshev polynomial | `SmootherType::Chebyshev` | ✅ |
| `SparseSmoothedProjection` | ILU-based | PCG+ILU0 (via linger) | ✅ |
| `BlockDiagonalPreconditioner` | Block Jacobi | `BlockDiagonalPrecond` | ✅ |
| `BlockTriangularPreconditioner` | Block triangular | `BlockTriangularPrecond` | ✅ |
| `SchurComplement` | Elimination for saddle point | `SchurComplementSolver` | ✅ |

### 7.3 Solver Convergence Monitors

| MFEM concept | fem-rs equivalent | Status |
|---|---|---|
| `IterativeSolver::SetTol()` | `tol` parameter | ✅ |
| `IterativeSolver::SetMaxIter()` | `max_iter` parameter | ✅ |
| `IterativeSolver::GetFinalNorm()` | `SolverResult::residual_norm` | ✅ |
| `IterativeSolver::GetNumIterations()` | `SolverResult::iterations` | ✅ |
| `IterativeSolver::SetPrintLevel()` | `SolverConfig::print_level` / `PrintLevel` enum | ✅ | Silent/Summary/Iterations/Debug |

---

## 8. Algebraic Multigrid

| MFEM / hypre concept | fem-rs equivalent | Status |
|---|---|---|
| `HypreBoomerAMG` (setup) | `AmgSolver::setup(mat)` → hierarchy | ✅ |
| `HypreBoomerAMG` (solve) | `AmgSolver::solve(hierarchy, rhs)` | ✅ |
| Strength of connection θ | `AmgParams::theta` | ✅ |
| Ruge-Stüben C/F splitting | RS-AMG (via linger) | ✅ |
| Smoothed aggregation | SA-AMG (via linger) | ✅ |
| Prolongation P | `AmgLevel::p: CsrMatrix` | ✅ |
| Restriction R = Pᵀ | `AmgLevel::r: CsrMatrix` | ✅ |
| Galerkin coarse A_c = R A P | SpGEMM chain | ✅ |
| Pre-smoother (ω-Jacobi) | Jacobi smoother | ✅ |
| Post-smoother | Post-smooth steps | ✅ |
| V-cycle | `CycleType::V` | ✅ |
| W-cycle | `CycleType::W` | ✅ |
| F-cycle | `CycleType::F` | ✅ |
| Max levels | Max levels config | ✅ |
| Coarse-grid direct solve | Dense LU | ✅ |
| hypre binding | feature `amg/hypre` | 🔲 |

---

## 9. Parallel Infrastructure

### 9.1 MPI Communicators

| MFEM concept | fem-rs module | Status |
|---|---|---|
| `MPI_Comm` | `ChannelBackend` (in-process threading) | ✅ |
| `MPI_Allreduce` | `Backend::allreduce()` | ✅ |
| `MPI_Allgather` | `Backend::allgather()` | ✅ |
| `MPI_Send/Recv` | `GhostExchange` (alltoallv) | ✅ |

### 9.2 Distributed Mesh

| MFEM class | fem-rs struct | Status |
|---|---|---|
| `ParMesh` | `ThreadLauncher` + partitioned mesh | ✅ |
| METIS partitioning | `MetisPartitioner` (pure-Rust) | ✅ |
| Ghost elements | `GhostExchange` (forward/reverse) | ✅ |
| Global-to-local node map | per-rank DOF mapping | ✅ |

### 9.3 Parallel Linear Algebra

| MFEM / hypre class | fem-rs struct | Status |
|---|---|---|
| `HypreParMatrix` | `ParCsrMatrix` (diag+offd blocks) | ✅ Thread + MPI backends |
| `HypreParVector` | `ParVector` (owned+ghost layout) | ✅ |
| `HypreParMatrix::Mult()` | `ParCsrMatrix::spmv()` via ghost exchange | ✅ |
| `HypreParMatrix::GetDiag()` | `ParCsrMatrix::diag` | ✅ |
| `HypreParMatrix::GetOffd()` | `ParCsrMatrix::offd` | ✅ |
| `ParFiniteElementSpace` | `ParallelFESpace<S>` (P1+P2) | ✅ |
| `ParBilinearForm::Assemble()` | `ParAssembler::assemble_bilinear()` | ✅ |
| `ParLinearForm::Assemble()` | `ParAssembler::assemble_linear()` | ✅ |
| `HypreSolver` (PCG+Jacobi) | `par_solve_pcg_jacobi()` | ✅ |
| `HypreBoomerAMG` | `ParAmgHierarchy` (local smoothed aggregation) | ✅ |
| `par_solve_pcg_amg()` | PCG + AMG V-cycle preconditioner | ✅ |
| `MPI_Comm_split` | `Comm::split(color, key)` | ✅ |
| Streaming mesh distribution | `partition_simplex_streaming()` | ✅ Phase 37 |
| WASM multi-Worker MPI | `WorkerLauncher::spawn_async()` + `jsmpi_main` | ✅ Phase 37 |
| Binary sub-mesh serde | `mesh_serde::encode/decode_submesh()` | ✅ Phase 37 |

---

## 10. I/O & Visualization

### 10.1 Mesh I/O

| MFEM format / method | fem-rs | Status |
|---|---|---|
| MFEM native mesh format (read/write) | — | ❌ use GMSH |
| GMSH `.msh` v2 ASCII (read) | `fem_io::read_msh_file()` | ✅ |
| GMSH `.msh` v4.1 ASCII (read) | `fem_io::read_msh_file()` | ✅ |
| GMSH `.msh` v4.1 binary (read) | `fem_io::read_msh_file()` | ✅ |
| Netgen `.vol` (read) | — | 🔲 Phase 9+ |
| Abaqus `.inp` (read) | — | 🔲 Phase 9+ |
| VTK `.vtu` legacy ASCII (write) | `write_vtk_scalar()` | ✅ |
| VTK `.vtu` XML binary (write) | `write_vtu()` (XML ASCII) | ✅ |
| HDF5 / XDMF (read/write) | feature `io/hdf5` | 🔲 |
| ParaView GLVis socket | — | ❌ out of scope |

### 10.2 Solution I/O

| MFEM concept | fem-rs | Status |
|---|---|---|
| `GridFunction::Save()` | VTK point data | ✅ scalar + vector |
| `GridFunction::Load()` | `read_vtu_point_data()` | ✅ | ASCII VTU reader |
| Restart files | HDF5 mesh + solution | 🔲 |

---

## 11. Grid Functions & Post-processing

| MFEM class / method | fem-rs equivalent | Status |
|---|---|---|
| `GridFunction(fes)` | `GridFunction<S>` (wraps DOF vec + space ref) | ✅ |
| `GridFunction::ProjectCoefficient()` | `FESpace::interpolate(f)` | ✅ |
| `GridFunction::ComputeL2Error()` | `GridFunction::compute_l2_error()` | ✅ |
| `GridFunction::ComputeH1Error()` | `GridFunction::compute_h1_error()` / `compute_h1_full_error()` | ✅ |
| `GridFunction::GetGradient()` | `postprocess::compute_element_gradients()` / `recover_gradient_nodal()` | ✅ |
| `GridFunction::GetCurl()` | `postprocess::compute_element_curl()` | ✅ |
| `GridFunction::GetDivergence()` | `postprocess::compute_element_divergence()` | ✅ |
| `ZZErrorEstimator` (Zienkiewicz-Zhu) | `zz_error_estimator()` | ✅ |
| `KellyErrorEstimator` | `kelly_estimator()` | ✅ | Face-jump based error indicator |
| `DiscreteLinearOperator` | Gradient, curl, div operators | ✅ `DiscreteLinearOperator::gradient/curl_2d/divergence` |

---

## 12. MFEM Examples → fem-rs Milestones

Each MFEM example defines a target milestone for fem-rs feature completeness.

### Tier 1 — Core Capability (Phases 6–7)

| MFEM example | PDE | FEM space | BCs | fem-rs milestone |
|---|---|---|---|---|
| **ex1** | −∇²u = 1, u=0 on ∂Ω | H¹ P1/P2 | Dirichlet | ✅ `ex1_poisson` O(h²) |
| **ex2** | −∇²u = f, mixed BCs | H¹ P1/P2 | Dirichlet + Neumann | ✅ `ex2_elasticity` |
| **ex3** (scalar) | −∇²u + αu = f (reaction-diffusion) | H¹ P1 | Dirichlet | ✅ Phase 6: `MassIntegrator` |
| **ex13** | −∇·(ε∇φ) = 0, elasticity | H¹ vector | Mixed | Phase 6: `ElasticityIntegrator` |
| **pex1** | Parallel Poisson | H¹ + MPI | Dirichlet | ✅ `pex1_poisson` (contiguous/METIS, streaming) |

### Tier 2 — Mixed & H(curl)/H(div) (Phase 6+)

| MFEM example | PDE | FEM space | fem-rs milestone |
|---|---|---|---|
| **ex3** (curl) | ∇×∇×**u** + **u** = **f** (Maxwell) | H(curl) Nédélec | ✅ `ex3_maxwell` O(h) |
| **ex4** | −∇·(**u**) = f, **u** = −κ∇p (Darcy) | H(div) RT + L² | ✅ `ex4_darcy` H(div) RT0 grad-div MINRES |
| **ex5** | Saddle-point Darcy/Stokes | H(div) × L² | ✅ `ex5_mixed_darcy` block PGMRES |
| **ex22** | Time-harmonic Maxwell (complex coeff.) | H(curl) | Phase 7+ |
| **em_magnetostatics_2d** (this project) | −∇·(ν∇Az) = Jz | H¹ P1 (2D A_z) | ✅ |

### Tier 3 — Time Integration (Phase 7+)

| MFEM example | PDE | Time method | fem-rs milestone |
|---|---|---|---|
| **ex9** (heat) | ∂u/∂t − ∇²u = 0 | BDF1 / Crank-Nicolson | ✅ `ex10_heat_equation` SDIRK-2 |
| **ex10** (wave) | ∂²u/∂t² − ∇²u = 0 | Leapfrog / Newmark | ✅ `ex10_wave_equation` Newmark-β |
| **ex14** (DG heat) | ∂u/∂t − ∇²u + b·∇u = 0 | Explicit RK + DG | ✅ `ex9_dg_advection` SIP-DG O(h²) |
| **ex16** (elastodynamics) | ρ ∂²**u**/∂t² = ∇·σ | Generalized-α | ✅ `ex16_nonlinear_heat` Newton |

### Tier 4 — Nonlinear & AMR (Phase 7+)

| MFEM example | Problem | fem-rs milestone |
|---|---|---|
| **ex4** (nonlinear) | −Δu + exp(u) = 0 | ✅ `NewtonSolver` |
| **ex6** | AMR Poisson with ZZ estimator | ✅ `refine_marked()`, `ZZErrorEstimator` |
| **ex15** | DG advection with AMR | ✅ `ex15_dg_amr` P1 + ZZ + Dörfler + refinement |
| **ex19** | Incompressible Navier-Stokes | ✅ `ex_navier_stokes` (Kovasznay Re=40, Oseen/Picard) |

### Tier 5 — HPC & Parallel (Phase 10)

| MFEM example | Problem | fem-rs milestone |
|---|---|---|
| **pex1** | Parallel Poisson (Poisson) | ✅ `pex1_poisson` (contiguous/METIS + streaming) |
| **pex2** | Parallel mixed Poisson | ✅ `pex2_mixed_darcy` |
| **pex3** | Parallel Maxwell (H(curl)) | ✅ `pex3_maxwell` |
| **pex5** | Parallel Darcy | ✅ `pex5_darcy` |

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

## Quick Reference: Phase → Features

| Phase | Crates | MFEM equivalents unlocked | Status |
|---|---|---|---|
| 0 | workspace | — | ✅ |
| 1 | `core` | Index types, `FemError`, scalar traits | ✅ |
| 2 | `mesh` | `Mesh`, element types, mesh generators | ✅ |
| 3 | `element` | `FiniteElement`, `IntegrationRule`, Lagrange P1–P2 | ✅ |
| 4 | `linalg` | `SparseMatrix`, `Vector`, COO→CSR assembly | ✅ |
| 5 | `space` | `FiniteElementSpace`, H1/L2, DOF manager | ✅ |
| 6 | `assembly` | `BilinearForm`, `LinearForm`, standard integrators | ✅ |
| 7 | `solver` | `CGSolver`, `GMRESSolver`, ILU(0), direct | ✅ |
| 8 | `amg` | SA-AMG + RS-AMG (native via linger) | ✅ |
| 9 | `io` | VTK XML, GMSH v4 reader | ✅ |
| 10 | `parallel` | Thread-based parallel, ghost exchange | ✅ |
| 11 | `wasm` | Browser-side FEM solver via JS API | ✅ |
| 12 | `element` | Nedelec ND1, Raviart-Thomas RT0 | ✅ |
| 13 | `space`+`assembly` | VectorH1Space, BlockMatrix, MixedAssembler, Elasticity | ✅ |
| 14 | `assembly` | SIP-DG (interior penalty) | ✅ |
| 15 | `solver`+`assembly` | NonlinearForm, NewtonSolver | ✅ |
| 16 | `solver` | ODE: ForwardEuler, RK4, RK45, ImplicitEuler, SDIRK-2, BDF-2 | ✅ |
| 17 | `mesh` | AMR: red refinement, ZZ estimator, Dörfler marking | ✅ |
| 18 | `parallel` | METIS k-way partitioning (pure-Rust) | ✅ |
| 19 | `mesh`+`space` | CurvedMesh (P2 isoparametric) | ✅ |
| 20 | `solver` | LOBPCG eigenvalue solver | ✅ |
| 21 | `solver`+`linalg` | BlockSystem, SchurComplement, MINRES | ✅ |
| 22 | `assembly`+`ceed` | Partial assembly: PA mass/diffusion, matrix-free | ✅ |
| 23 | `space` | HCurlSpace (Nédélec ND1), HDivSpace (RT0), element_signs | ✅ |
| 24 | `assembly` | VectorAssembler, CurlCurlIntegrator, VectorMassIntegrator | ✅ |
| 25 | `assembly`+`solver` | DG-SIP face normals fix, SchurComplement PGMRES, MINRES rewrite, TriND1 fix; all 8 MFEM-style examples verified | ✅ |
| 26 | `assembly` | Coefficient system: ScalarCoeff/VectorCoeff/MatrixCoeff traits, PWConstCoeff, PWCoeff, GridFunctionCoeff, composition | ✅ |
| 27 | `assembly` | Convection, VectorDiffusion, BoundaryMass, GradDiv, Transpose, Sum integrators; VectorDomainLF, BoundaryNormalLF | ✅ |
| 28 | `assembly` | GridFunction wrapper, L²/H¹ error, element gradients/curl/div, nodal gradient recovery | ✅ |
| 29 | `assembly` | DiscreteLinearOperator: gradient, curl_2d, divergence as sparse matrices; de Rham exact sequence | ✅ |
| 30 | `solver` | Newmark-β time integrator; ex10_wave_equation example | ✅ |
| 31 | `element` | Gauss-Lobatto quadrature (seg, quad, hex) | ✅ |
| 32 | `examples` | ex4_darcy (H(div) RT0), ex15_dg_amr (P1 + ZZ + Dörfler) | ✅ |
| 33a-e | `parallel` | jsmpi backend, DofPartition, ParVector, ParCsrMatrix, ParAssembler, par_solve_pcg_jacobi, pex1 | ✅ |
| 34 | `parallel` | P2 parallel spaces (DofPartition::from_dof_manager, edge DOF ownership, auto-permute) | ✅ |
| 35 | `parallel` | Parallel AMG (ParAmgHierarchy, smoothed aggregation, par_solve_pcg_amg) | ✅ |
| 36 | `parallel` | Comm::split sub-communicators | ✅ |
| 37 | `parallel`+`wasm` | WASM multi-Worker (spawn_async, jsmpi_main), streaming mesh partition (partition_simplex_streaming), binary mesh serde | ✅ |
| 38 | `parallel` | METIS streaming partition (partition_simplex_metis_streaming), generalized submesh extractor, pex1 CLI flags | ✅ |
| 38b | `io` | GMSH v2 ASCII + v4.1 binary reader (unified `read_msh_file()` entry point) | ✅ |
| 39 | `parallel`+`examples` | pex2 (mixed Poisson), pex3 (Maxwell), pex5 (Darcy) parallel examples | ✅ |
| 39b | `amg` | Chebyshev smoother (`SmootherType::Chebyshev`), F-cycle (`CycleType::F`) | ✅ |
| 40 | `examples`+`assembly` | Taylor-Hood P2-P1 Stokes (`ex_stokes` lid-driven cavity) | ✅ |
| 42a | `mesh`+`space`+`io` | Mixed element mesh infrastructure (per-element types, variable DofManager, GMSH mixed read) | ✅ |
| 44 | `assembly`+`examples` | VectorConvectionIntegrator + Navier-Stokes Oseen/Picard (`ex_navier_stokes`, Kovasznay Re=40) | ✅ |
| 42b | `assembly` | Quad4/Hex8 isoparametric Jacobian, `unit_square_quad`, Q1 Poisson verified | ✅ |
| 45 | `wasm`+`e2e` | Browser E2E test: WASM Poisson solver verified via Playwright/Chromium | ✅ |
| 46 | `mesh`+`linalg`+`solver`+`space`+`io` | Backlog: bounding_box, periodic mesh, DenseTensor, SLI, H1Trace, VTK reader, PrintLevel | ✅ |
| 47 | `mesh`+`space` | NCMesh: Tri3/Tet4 nonconforming refine + hanging constraints + `NCState`/`NCState3D` multi-level + P2 prolongation | ✅ |

---

## Remaining Items Summary (🔲 Planned · 🔨 Partial)

### Mesh
| Item | Status | Priority |
|------|--------|----------|
| Mixed element meshes (Tri+Quad, Tet+Hex) | ✅ | ~~Medium~~ Done |
| NCMesh (non-conforming, hanging nodes) | ✅ | ~~Low~~ Done |
| `bdr_attributes` dedup utility | ✅ | ~~Low~~ Done |
| `ElementTransformation` type | 🔨 | Low (works inline) |
| `GetBoundingBox()` | ✅ | ~~Low~~ Done |
| Periodic mesh generation | ✅ | ~~Low~~ Done |

### I/O
| Item | Status | Priority |
|------|--------|----------|
| ~~GMSH v4.1 binary reader~~ | ✅ | ~~High~~ Done |
| ~~GMSH v2 reader~~ | ✅ | ~~Medium~~ Done |
| HDF5/XDMF parallel I/O | 🔲 | Medium |
| Netgen `.vol` reader | 🔲 | Low |
| Abaqus `.inp` reader | 🔲 | Low |
| `GridFunction::Load()` | ✅ | ~~Low~~ Done |
| Restart files (checkpoint) | 🔲 | Low |

### Solvers
| Item | Status | Priority |
|------|--------|----------|
| Chebyshev smoother (AMG) | ✅ | ~~Medium~~ Done |
| SLISolver (stationary iteration) | ✅ | ~~Low~~ Done |
| AMG F-cycle | ✅ | ~~Low~~ Done |
| hypre binding | 🔲 | Low |

### Spaces & Post-processing
| Item | Status | Priority |
|------|--------|----------|
| H1_Trace_FECollection | ✅ | ~~Low~~ Done |
| Taylor-Hood P2-P1 | Stokes flow | ✅ `ex_stokes` (lid-driven cavity) |
| Kelly error estimator | ✅ | ~~Low~~ Done |
| `DenseTensor` | ✅ | ~~Low~~ Done |
| `SetSubVector` slice assignment | ✅ | ~~Low~~ Done |

### Parallel Examples
| Item | Status | Priority |
|------|--------|----------|
| pex2 (parallel mixed Poisson) | ✅ | ~~Medium~~ Done |
| pex3 (parallel Maxwell) | ✅ | ~~Medium~~ Done |
| pex5 (parallel Darcy) | ✅ | ~~Medium~~ Done |
| ex19 (Navier-Stokes) | ✅ | ~~Medium~~ Done |
| Browser E2E (WASM) | ✅ | ~~Medium~~ Done |

---

## Recommended Roadmap (Phase 39+)

Based on the completed 38 phases and remaining gaps, here is a recommended
prioritized roadmap for continued development.

### Phase 39 — Parallel Examples Expansion (pex2 / pex3 / pex5) ✅
> **Completed** — validates parallel infrastructure across all FE spaces

| Task | Space | Status |
|------|-------|--------|
| `pex2_mixed_darcy` | H(div) RT0 × L² | ✅ |
| `pex3_maxwell` | H(curl) ND1 | ✅ |
| `pex5_darcy` | H(div) × L² saddle-point | ✅ |

### Phase 39b — Chebyshev Smoother + AMG F-cycle ✅
> **Completed** — smoother quality directly impacts AMG convergence

- ✅ Chebyshev polynomial smoother (degree 2–4) as `SmootherType::Chebyshev`
- ✅ Eigenvalue estimate via spectral radius bound (λ_max)
- ✅ F-cycle: `CycleType::F` (V on first coarse visit, W after)
- ✅ Tests: Chebyshev, F-cycle, Chebyshev+F-cycle combinations

### Phase 40 — Taylor-Hood P2-P1 Stokes Example ✅
> **Completed** — demonstrates mixed FEM at production quality

- ✅ `ex_stokes` example: lid-driven cavity on [0,1]²
- ✅ P2 velocity + P1 pressure via `MixedAssembler`
- ✅ Block saddle-point solver (SchurComplementSolver with GMRES)
- ✅ Verified convergence at n=8,16,32; divergence-free to solver tolerance

### Phase 42 — Mixed Element Meshes (42a ✅, 42b ✅)
> **Completed** — data structures, I/O, and assembly all done

- ✅ Per-element `ElementType` and CSR-like offset arrays in `SimplexMesh`
- ✅ Variable-stride `DofManager` for P1 on mixed meshes
- ✅ GMSH reader preserves mixed element types (Tri+Quad, Tet+Hex)
- ✅ Isoparametric Jacobian for Quad4/Hex8 in assembler (bilinear/trilinear mapping)
- ✅ `unit_square_quad(n)` mesh generator + Q1 Poisson convergence verified

### Phase 43 — HDF5/XDMF Parallel I/O
> **Priority: Medium** — needed for large-scale checkpointing

- Feature-gated `io/hdf5` with `hdf5-rs` crate
- Write: parallel mesh + solution to XDMF + HDF5
- Read: parallel restart from checkpoint
- Time-series output for transient problems

### Phase 44 — Navier-Stokes (Kovasznay flow) ✅
> **Completed** — flagship nonlinear PDE example

- ✅ `VectorConvectionIntegrator`: `∫ (w·∇)u · v dx` for vector fields
- ✅ Oseen linearization with Picard iteration
- ✅ `ex_navier_stokes` example: Kovasznay flow benchmark (Re=40)
- ✅ Taylor-Hood P2/P1 discretization (reuses Phase 40 infrastructure)
- ✅ Converges in ~16–20 Picard iterations; velocity error decreases with h-refinement

### Phase 45 — Browser E2E (WASM) ✅
> **Completed** — validates the full browser pipeline

- ✅ Playwright/Chromium E2E test (`crates/wasm/e2e/`)
- ✅ WASM Poisson solver: assemble → solve → verify in browser
- ✅ Solution validated against analytical max (0.0737 for −Δu=1)

### Phase 46 — Backlog Cleanup ✅
> **Completed** — 9 remaining items resolved

- ✅ `SimplexMesh::bounding_box()` — axis-aligned bounding box (2-D / 3-D)
- ✅ `SimplexMesh::unique_boundary_tags()` — sorted/deduped boundary tag set
- ✅ `SimplexMesh::make_periodic()` — node merging for periodic BCs
- ✅ `DenseTensor` — 3-D row-major tensor with slab access
- ✅ `solve_jacobi_sli()` / `solve_gs_sli()` — Jacobi/GS stationary iteration
- ✅ `H1TraceSpace` — H½ trace of H¹ on boundary faces (P1)
- ✅ `read_vtu_point_data()` — VTK `.vtu` ASCII reader for solution loading
- ✅ `PrintLevel` enum — structured solver verbosity (Silent/Summary/Iterations/Debug)
- ✅ `kelly_estimator()` was already implemented — marked in MFEM_MAPPING
- ✅ `SetSubVector` / `GetSubVector` were already implemented — marked in MFEM_MAPPING

### Phase 47 — NCMesh (Non-Conforming Mesh / Hanging Nodes) ✅
> **Completed** — 2-D Tri3 + 3-D Tet4 non-conforming refinement with multi-level state tracking

#### 2-D (Tri3) Hanging Edge Constraints
- ✅ `refine_nonconforming()` — red-refines only marked elements, no propagation
- ✅ `HangingNodeConstraint` detection — identifies midpoints on coarse/fine edges
- ✅ `apply_hanging_constraints()` — P^T K P static condensation via COO rebuild
- ✅ `recover_hanging_values()` — post-solve interpolation for constrained DOFs
- ✅ `NCState` multi-level constraint tracking — carries and resolves hanging constraints across successive NC refinements
- ✅ `prolongate_p2_hanging()` — P2 hanging-node prolongation by coarse P2 field evaluation at fine DOF coordinates
- ✅ `ex15_dg_amr --nc` — demonstrates single-level NC AMR with error reduction

#### 3-D (Tet4) Hanging Face Constraints
- ✅ `HangingFaceConstraint` struct — records hanging coarse faces and representative midpoint nodes
- ✅ `refine_nonconforming_3d(mesh, marked)` — red-refines Tet4 elements into 8 children using edge midpoints
- ✅ `local_faces_tet()` — helper returns 4 triangular face local indices for Tet4
- ✅ `face_key_3d()` — canonical face key (sorted triplet) for face uniqueness
- ✅ Hanging-face detection — detects refined/coarse Tet4 face mismatch and emits hanging edge constraints
- ✅ `NCState3D` multi-level tracking — carries active edge midpoints and rebuilds constraints across levels
- ✅ Boundary face reconstruction — preserves and refines Tri3 boundary faces in 3-D refinement
- ✅ Unit tests — `tet4_nonconforming_refine_single_element()`, `tet4_nonconforming_refine_with_neighbor()`, `ncstate3d_two_level_refine()`

### Backlog (Low Priority)
| Item | Phase | Notes |
|------|-------|-------|
| hypre binding | TBD | Optional FFI for production AMG |
| Netgen / Abaqus readers | TBD | Additional mesh import formats |
| HDF5/XDMF I/O | TBD | Large-scale checkpointing |
| Restart files | TBD | Requires HDF5 |
| Tet4 NC AMR example | ✅ | ~~TBD~~ Done (`ex15_tet_nc_amr`, supports `--solve`) |
