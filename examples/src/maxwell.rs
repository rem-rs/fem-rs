use std::collections::HashSet;

use fem_amg::{AmgConfig, AmgSolver};
use fem_assembly::{
    DiscreteLinearOperator,
    TangentialMassIntegrator,
    VectorAssembler,
    VectorBoundaryAssembler,
    coefficient::{ConstantMatrixCoeff, FnMatrixCoeff},
    standard::{CurlCurlIntegrator, TangentialTraceLFIntegrator, VectorMassIntegrator, VectorMassTensorIntegrator},
    vector_integrator::{VectorLinearIntegrator, VectorQpData},
};
use fem_element::nedelec::TriND1;
use fem_element::reference::VectorReferenceElement;
use fem_linalg::CsrMatrix;
use fem_mesh::{SimplexMesh, topology::MeshTopology};
use fem_solver::{EigenResult, LobpcgConfig, SolveResult, SolverConfig, lobpcg_constrained_preconditioned, solve_cg_operator, solve_pcg_jacobi};
use fem_space::{H1Space, HCurlSpace, HDivSpace, L2Space, constraints::{apply_dirichlet, boundary_dofs, boundary_dofs_hcurl}, fe_space::FESpace};
use nalgebra::DMatrix;

type TangentialBoundaryFn = dyn Fn(&[f64], &[f64]) -> f64 + Send + Sync;
type VectorSourceFn = dyn Fn(&[f64]) -> [f64; 2] + Send + Sync;
type Matrix2Fn = dyn Fn(&[f64]) -> [f64; 4] + Send + Sync;

enum StaticMaxwellVolumeModel {
    Isotropic { mu: f64, alpha: f64 },
    AnisotropicDiag { mu: f64, sigma_x: f64, sigma_y: f64 },
    AnisotropicMatrixFn { mu: f64, sigma: Box<Matrix2Fn> },
}

pub enum HcurlBoundaryCondition {
    PecZero {
        tags: Vec<i32>,
    },
    TangentialRobin {
        tags: Vec<i32>,
        gamma: f64,
        data: Box<TangentialBoundaryFn>,
    },
}

pub enum BoundarySelection<'a> {
    Tags(&'a [i32]),
    Marker {
        boundary_attributes: &'a [i32],
        marker: &'a [i32],
    },
}

#[derive(Default)]
pub struct HcurlBoundaryConfig {
    conditions: Vec<HcurlBoundaryCondition>,
}

pub struct BoundaryApplyReport {
    pub essential_dofs: usize,
}

pub struct HcurlConstraintSubspace {
    pub hcurl_free_dofs: Vec<usize>,
    pub h1_free_dofs: Vec<usize>,
    pub gradient_constraints: DMatrix<f64>,
}

pub struct HcurlEigenSystem {
    pub stiffness_free: CsrMatrix<f64>,
    pub mass_free: CsrMatrix<f64>,
    pub constraints: DMatrix<f64>,
    pub hcurl_free_dofs: Vec<usize>,
    pub h1_free_dofs: Vec<usize>,
}

/// Matrix-free-ish H(curl) operator for 2-D ND1 static Maxwell problems:
///
/// `A x = (1/mu) C^T M_b^{-1} C x + alpha * M_e x`,
///
/// where `C` is the discrete curl and `M_b` is the diagonal P0 mass.
/// The combined global `A` matrix is never assembled.
pub struct HcurlMatrixFreeOperator2D {
    mass_hcurl: CsrMatrix<f64>,
    curl_c: CsrMatrix<f64>,
    curl_ct: CsrMatrix<f64>,
    mb_diag: Vec<f64>,
    pec_dofs: Vec<usize>,
    n_dofs: usize,
    mu_factor: f64,
}

impl HcurlMatrixFreeOperator2D {
    pub fn new(
        hcurl: &HCurlSpace<SimplexMesh<2>>,
        mu: f64,
        alpha: f64,
        quad_order: u8,
        pec_tags: &[i32],
    ) -> Self {
        assert!(mu > 0.0, "mu must be positive");

        let mass_hcurl = VectorAssembler::assemble_bilinear(
            hcurl,
            &[&VectorMassIntegrator { alpha }],
            quad_order,
        );

        let l2 = L2Space::new(hcurl.mesh().clone(), 0);
        let curl_c = DiscreteLinearOperator::curl_2d(hcurl, &l2)
            .expect("curl_2d assembly failed in HcurlMatrixFreeOperator2D");
        let curl_ct = curl_c.transpose();

        let mut mb_diag = vec![0.0_f64; l2.n_dofs()];
        let mesh = l2.mesh();
        for e in mesh.elem_iter() {
            let nodes = mesh.element_nodes(e);
            let dof = l2.element_dofs(e)[0] as usize;
            let a = mesh.node_coords(nodes[0]);
            let b = mesh.node_coords(nodes[1]);
            let c = mesh.node_coords(nodes[2]);
            let area = 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])).abs();
            mb_diag[dof] = area;
        }

        let pec_dofs: Vec<usize> = boundary_dofs_hcurl(hcurl.mesh(), hcurl, pec_tags)
            .into_iter()
            .map(|d| d as usize)
            .collect();

        Self {
            mass_hcurl,
            curl_c,
            curl_ct,
            mb_diag,
            pec_dofs,
            n_dofs: hcurl.n_dofs(),
            mu_factor: mu,
        }
    }

    pub fn n_dofs(&self) -> usize {
        self.n_dofs
    }

    pub fn apply(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.n_dofs, "x length mismatch");
        assert_eq!(y.len(), self.n_dofs, "y length mismatch");

        y.fill(0.0);

        // Eliminate essential boundary columns by zeroing constrained entries in x.
        let mut x_work = x.to_vec();
        for &d in &self.pec_dofs {
            x_work[d] = 0.0;
        }

        // alpha * M_e x
        self.mass_hcurl.spmv(&x_work, y);

        // mu * C^T M_b^{-1} C x  with diagonal M_b.
        let mut ce = vec![0.0_f64; self.mb_diag.len()];
        self.curl_c.spmv(&x_work, &mut ce);
        for (i, v) in ce.iter_mut().enumerate() {
            let d = self.mb_diag[i];
            *v = if d.abs() > 1e-14 { self.mu_factor * (*v) / d } else { 0.0 };
        }
        self.curl_ct.spmv_add(1.0, &ce, 1.0, y);

        // Essential rows are identity rows.
        for &d in &self.pec_dofs {
            y[d] = x[d];
        }
    }
}

pub fn solve_hcurl_matrix_free(
    op: &HcurlMatrixFreeOperator2D,
    rhs: &[f64],
    cfg: &SolverConfig,
) -> Result<(Vec<f64>, SolveResult), String> {
    if rhs.len() != op.n_dofs() {
        return Err(format!(
            "rhs length mismatch: got {}, expected {}",
            rhs.len(),
            op.n_dofs()
        ));
    }

    let mut rhs_work = rhs.to_vec();
    for &d in &op.pec_dofs {
        rhs_work[d] = 0.0;
    }

    let mut x = vec![0.0_f64; op.n_dofs()];
    let res = solve_cg_operator(op.n_dofs(), op.n_dofs(), |xin, yout| op.apply(xin, yout), &rhs_work, &mut x, cfg)
        .map_err(|e| format!("matrix-free CG failed: {e}"))?;
    for &d in &op.pec_dofs {
        x[d] = 0.0;
    }
    Ok((x, res))
}

/// Solve reduced Maxwell generalized eigenproblem with constrained LOBPCG and
/// AMG-preconditioned residual blocks.
pub fn solve_hcurl_eigen_preconditioned_amg(
    eig_system: &HcurlEigenSystem,
    k: usize,
    eig_cfg: &LobpcgConfig,
    amg_cfg: AmgConfig,
    inner_cfg: &SolverConfig,
) -> Result<EigenResult, String> {
    let stiffness = &eig_system.stiffness_free;
    let amg = AmgSolver::setup(stiffness, amg_cfg);

    let precond = |r: &DMatrix<f64>| {
        let mut z = DMatrix::<f64>::zeros(r.nrows(), r.ncols());
        for j in 0..r.ncols() {
            let rhs_col: Vec<f64> = r.column(j).iter().copied().collect();
            let mut x_col = vec![0.0_f64; rhs_col.len()];
            if amg.solve(stiffness, &rhs_col, &mut x_col, inner_cfg).is_err() {
                x_col.copy_from_slice(&rhs_col);
            }
            for i in 0..x_col.len() {
                z[(i, j)] = x_col[i];
            }
        }
        z
    };

    lobpcg_constrained_preconditioned(
        &eig_system.stiffness_free,
        Some(&eig_system.mass_free),
        k,
        &eig_system.constraints,
        precond,
        eig_cfg,
    )
}

fn boundary_admittance(epsilon: f64, mu: f64) -> f64 {
    assert!(epsilon > 0.0, "epsilon must be positive");
    assert!(mu > 0.0, "mu must be positive");
    (epsilon / mu).sqrt()
}

pub struct StaticMaxwellProblem {
    space: HCurlSpace<SimplexMesh<2>>,
    mat: CsrMatrix<f64>,
    rhs: Vec<f64>,
    boundary: HcurlBoundaryConfig,
    quad_order: u8,
}

pub struct StaticMaxwellSolveOutput {
    pub space: HCurlSpace<SimplexMesh<2>>,
    pub solution: Vec<f64>,
    pub solve_result: SolveResult,
    pub boundary_report: BoundaryApplyReport,
}

pub struct StaticMaxwellBuilder {
    space: HCurlSpace<SimplexMesh<2>>,
    quad_order: u8,
    volume_model: StaticMaxwellVolumeModel,
    source: Option<Box<VectorSourceFn>>,
    boundary: HcurlBoundaryConfig,
}

struct FnVectorSourceIntegrator<'a> {
    f: &'a VectorSourceFn,
}

impl<'a> VectorLinearIntegrator for FnVectorSourceIntegrator<'a> {
    fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f_elem: &mut [f64]) {
        let src = (self.f)(qp.x_phys);
        for i in 0..qp.n_dofs {
            let dot = qp.phi_vec[i * 2] * src[0] + qp.phi_vec[i * 2 + 1] * src[1];
            f_elem[i] += qp.weight * dot;
        }
    }
}

impl HcurlBoundaryConfig {
    pub fn new() -> Self {
        Self::default()
    }

    fn selection_to_tags(selection: BoundarySelection<'_>) -> Vec<i32> {
        match selection {
            BoundarySelection::Tags(tags) => tags.to_vec(),
            BoundarySelection::Marker {
                boundary_attributes,
                marker,
            } => marker_to_tags(boundary_attributes, marker),
        }
    }

    pub fn add_pec_zero_on(&mut self, selection: BoundarySelection<'_>) -> &mut Self {
        let tags = Self::selection_to_tags(selection);
        self.conditions.push(HcurlBoundaryCondition::PecZero {
            tags,
        });
        self
    }

    pub fn add_pec_zero(&mut self, tags: &[i32]) -> &mut Self {
        self.add_pec_zero_on(BoundarySelection::Tags(tags))
    }

    pub fn add_tangential_robin_on<F>(
        &mut self,
        selection: BoundarySelection<'_>,
        gamma: f64,
        data: F,
    ) -> &mut Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        let tags = Self::selection_to_tags(selection);
        self.conditions
            .push(HcurlBoundaryCondition::TangentialRobin {
                tags,
                gamma,
                data: Box::new(data),
            });
        self
    }

    pub fn add_tangential_robin<F>(&mut self, tags: &[i32], gamma: f64, data: F) -> &mut Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        self.add_tangential_robin_on(BoundarySelection::Tags(tags), gamma, data)
    }

    pub fn add_impedance<F>(&mut self, tags: &[i32], gamma: f64, data: F) -> &mut Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        self.add_tangential_robin(tags, gamma, data)
    }

    pub fn add_impedance_on<F>(
        &mut self,
        selection: BoundarySelection<'_>,
        gamma: f64,
        data: F,
    ) -> &mut Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        self.add_tangential_robin_on(selection, gamma, data)
    }

    pub fn add_impedance_physical_on<F>(
        &mut self,
        selection: BoundarySelection<'_>,
        epsilon: f64,
        mu: f64,
        data: F,
    ) -> &mut Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        let gamma = boundary_admittance(epsilon, mu);
        self.add_impedance_on(selection, gamma, data)
    }

    pub fn add_impedance_physical<F>(
        &mut self,
        tags: &[i32],
        epsilon: f64,
        mu: f64,
        data: F,
    ) -> &mut Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        self.add_impedance_physical_on(BoundarySelection::Tags(tags), epsilon, mu, data)
    }

    pub fn add_impedance_physical_from_marker<F>(
        &mut self,
        boundary_attributes: &[i32],
        marker: &[i32],
        epsilon: f64,
        mu: f64,
        data: F,
    ) -> &mut Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        self.add_impedance_physical_on(
            BoundarySelection::Marker {
                boundary_attributes,
                marker,
            },
            epsilon,
            mu,
            data,
        )
    }

    pub fn add_absorbing<F>(&mut self, tags: &[i32], gamma_abs: f64, data: F) -> &mut Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        self.add_tangential_robin(tags, gamma_abs, data)
    }

    pub fn add_absorbing_on<F>(
        &mut self,
        selection: BoundarySelection<'_>,
        gamma_abs: f64,
        data: F,
    ) -> &mut Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        self.add_tangential_robin_on(selection, gamma_abs, data)
    }

    pub fn add_absorbing_physical_on<F>(
        &mut self,
        selection: BoundarySelection<'_>,
        epsilon: f64,
        mu: f64,
        data: F,
    ) -> &mut Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        let gamma_abs = boundary_admittance(epsilon, mu);
        self.add_absorbing_on(selection, gamma_abs, data)
    }

    pub fn add_absorbing_physical<F>(
        &mut self,
        tags: &[i32],
        epsilon: f64,
        mu: f64,
        data: F,
    ) -> &mut Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        self.add_absorbing_physical_on(BoundarySelection::Tags(tags), epsilon, mu, data)
    }

    pub fn add_absorbing_physical_from_marker<F>(
        &mut self,
        boundary_attributes: &[i32],
        marker: &[i32],
        epsilon: f64,
        mu: f64,
        data: F,
    ) -> &mut Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        self.add_absorbing_physical_on(
            BoundarySelection::Marker {
                boundary_attributes,
                marker,
            },
            epsilon,
            mu,
            data,
        )
    }

    pub fn add_tangential_drive<F>(&mut self, tags: &[i32], gamma: f64, data: F) -> &mut Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        self.add_tangential_robin(tags, gamma, data)
    }

    pub fn add_tangential_drive_on<F>(
        &mut self,
        selection: BoundarySelection<'_>,
        gamma: f64,
        data: F,
    ) -> &mut Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        self.add_tangential_robin_on(selection, gamma, data)
    }

    pub fn add_pec_zero_from_marker(
        &mut self,
        boundary_attributes: &[i32],
        marker: &[i32],
    ) -> &mut Self {
        self.add_pec_zero_on(BoundarySelection::Marker {
            boundary_attributes,
            marker,
        })
    }

    pub fn add_tangential_robin_from_marker<F>(
        &mut self,
        boundary_attributes: &[i32],
        marker: &[i32],
        gamma: f64,
        data: F,
    ) -> &mut Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        self.add_tangential_robin_on(
            BoundarySelection::Marker {
                boundary_attributes,
                marker,
            },
            gamma,
            data,
        )
    }

    pub fn add_impedance_from_marker<F>(
        &mut self,
        boundary_attributes: &[i32],
        marker: &[i32],
        gamma: f64,
        data: F,
    ) -> &mut Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        self.add_impedance_on(
            BoundarySelection::Marker {
                boundary_attributes,
                marker,
            },
            gamma,
            data,
        )
    }

    pub fn add_absorbing_from_marker<F>(
        &mut self,
        boundary_attributes: &[i32],
        marker: &[i32],
        gamma_abs: f64,
        data: F,
    ) -> &mut Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        self.add_absorbing_on(
            BoundarySelection::Marker {
                boundary_attributes,
                marker,
            },
            gamma_abs,
            data,
        )
    }

    pub fn add_tangential_drive_from_marker<F>(
        &mut self,
        boundary_attributes: &[i32],
        marker: &[i32],
        gamma: f64,
        data: F,
    ) -> &mut Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        self.add_tangential_drive_on(
            BoundarySelection::Marker {
                boundary_attributes,
                marker,
            },
            gamma,
            data,
        )
    }

    pub fn apply(
        &self,
        space: &HCurlSpace<SimplexMesh<2>>,
        mat: &mut CsrMatrix<f64>,
        rhs: &mut Vec<f64>,
        quad_order: u8,
    ) -> BoundaryApplyReport {
        let mut essential_dofs = 0;

        for condition in &self.conditions {
            if let HcurlBoundaryCondition::TangentialRobin { tags, gamma, data } = condition {
                add_tangential_robin_boundary(
                    space,
                    mat,
                    rhs,
                    tags,
                    *gamma,
                    quad_order,
                    |x, normal| data(x, normal),
                );
            }
        }

        for condition in &self.conditions {
            if let HcurlBoundaryCondition::PecZero { tags } = condition {
                essential_dofs += apply_pec_zero(space, mat, rhs, tags);
            }
        }

        BoundaryApplyReport { essential_dofs }
    }
}

/// Convert MFEM-style boundary marker array (0/1) into boundary attribute tags.
///
/// `boundary_attributes` and `marker` must have the same length.
pub fn marker_to_tags(boundary_attributes: &[i32], marker: &[i32]) -> Vec<i32> {
    assert_eq!(
        boundary_attributes.len(),
        marker.len(),
        "marker size mismatch: attributes={}, marker={}",
        boundary_attributes.len(),
        marker.len()
    );

    boundary_attributes
        .iter()
        .zip(marker.iter())
        .filter_map(|(&attr, &m)| if m != 0 { Some(attr) } else { None })
        .collect()
}

/// Return free HCurl DOFs by excluding boundary DOFs selected by marker.
pub fn free_hcurl_dofs_from_marker(
    space: &HCurlSpace<SimplexMesh<2>>,
    boundary_attributes: &[i32],
    marker: &[i32],
) -> Vec<usize> {
    let tags = marker_to_tags(boundary_attributes, marker);
    let bnd: HashSet<u32> = boundary_dofs_hcurl(space.mesh(), space, &tags)
        .into_iter()
        .collect();

    (0..space.n_dofs() as u32)
        .filter(|d| !bnd.contains(d))
        .map(|d| d as usize)
        .collect()
}

/// Return free H1 DOFs by excluding boundary DOFs selected by marker.
pub fn free_h1_dofs_from_marker(
    space: &H1Space<SimplexMesh<2>>,
    boundary_attributes: &[i32],
    marker: &[i32],
) -> Vec<usize> {
    let tags = marker_to_tags(boundary_attributes, marker);
    let bnd: HashSet<u32> = boundary_dofs(space.mesh(), space.dof_manager(), &tags)
        .into_iter()
        .collect();

    (0..space.n_dofs() as u32)
        .filter(|d| !bnd.contains(d))
        .map(|d| d as usize)
        .collect()
}

/// Extract a square submatrix by selecting rows/cols listed in `keep`.
pub fn extract_square_submatrix(mat: &CsrMatrix<f64>, keep: &[usize]) -> CsrMatrix<f64> {
    use fem_linalg::CooMatrix;

    let n = keep.len();
    let mut inv = vec![usize::MAX; mat.nrows];
    for (i, &g) in keep.iter().enumerate() {
        inv[g] = i;
    }

    let mut coo = CooMatrix::<f64>::new(n, n);
    for (ri, &gi) in keep.iter().enumerate() {
        for idx in mat.row_ptr[gi]..mat.row_ptr[gi + 1] {
            let gj = mat.col_idx[idx] as usize;
            let cj = inv[gj];
            if cj != usize::MAX {
                coo.add(ri, cj, mat.values[idx]);
            }
        }
    }
    coo.into_csr()
}

/// Extract a rectangular submatrix by selected `rows` and `cols`.
pub fn extract_rect_submatrix(
    mat: &CsrMatrix<f64>,
    rows: &[usize],
    cols: &[usize],
) -> CsrMatrix<f64> {
    use fem_linalg::CooMatrix;

    let mut col_map = vec![usize::MAX; mat.ncols];
    for (cj, &gj) in cols.iter().enumerate() {
        col_map[gj] = cj;
    }

    let mut coo = CooMatrix::<f64>::new(rows.len(), cols.len());
    for (ri, &gi) in rows.iter().enumerate() {
        for idx in mat.row_ptr[gi]..mat.row_ptr[gi + 1] {
            let gj = mat.col_idx[idx] as usize;
            let cj = col_map[gj];
            if cj != usize::MAX {
                coo.add(ri, cj, mat.values[idx]);
            }
        }
    }
    coo.into_csr()
}

/// Return `a + alpha * b` for CSR matrices with identical shape.
pub fn csr_add_scaled(a: &CsrMatrix<f64>, b: &CsrMatrix<f64>, alpha: f64) -> CsrMatrix<f64> {
    assert_eq!(a.nrows, b.nrows, "csr_add_scaled: row mismatch");
    assert_eq!(a.ncols, b.ncols, "csr_add_scaled: col mismatch");

    let mut coo = fem_linalg::CooMatrix::<f64>::new(a.nrows, a.ncols);
    for i in 0..a.nrows {
        for idx in a.row_ptr[i]..a.row_ptr[i + 1] {
            coo.add(i, a.col_idx[idx] as usize, a.values[idx]);
        }
    }
    for i in 0..b.nrows {
        for idx in b.row_ptr[i]..b.row_ptr[i + 1] {
            coo.add(i, b.col_idx[idx] as usize, alpha * b.values[idx]);
        }
    }
    coo.into_csr()
}

/// Convert CSR matrix to dense nalgebra matrix.
pub fn csr_to_dense_matrix(mat: &CsrMatrix<f64>) -> DMatrix<f64> {
    let mut dense = DMatrix::<f64>::zeros(mat.nrows, mat.ncols);
    for i in 0..mat.nrows {
        for idx in mat.row_ptr[i]..mat.row_ptr[i + 1] {
            let j = mat.col_idx[idx] as usize;
            dense[(i, j)] = mat.values[idx];
        }
    }
    dense
}

/// Build the HCurl constrained subspace data used in Maxwell eigenproblems.
///
/// Returns free DOF lists and the dense constraint matrix obtained from the
/// discrete gradient restricted to free HCurl/H1 DOFs.
pub fn build_hcurl_constraint_subspace_from_marker(
    h1: &H1Space<SimplexMesh<2>>,
    hcurl: &HCurlSpace<SimplexMesh<2>>,
    boundary_attributes: &[i32],
    marker: &[i32],
) -> HcurlConstraintSubspace {
    let hcurl_free_dofs = free_hcurl_dofs_from_marker(hcurl, boundary_attributes, marker);
    let h1_free_dofs = free_h1_dofs_from_marker(h1, boundary_attributes, marker);

    let grad = DiscreteLinearOperator::gradient(h1, hcurl)
        .expect("failed to build discrete gradient operator");
    let grad_free = extract_rect_submatrix(&grad, &hcurl_free_dofs, &h1_free_dofs);
    let gradient_constraints = csr_to_dense_matrix(&grad_free);

    HcurlConstraintSubspace {
        hcurl_free_dofs,
        h1_free_dofs,
        gradient_constraints,
    }
}

/// Assemble reduced Maxwell generalized eigen-system in the free HCurl subspace.
///
/// Builds full curl-curl and mass matrices, applies MFEM-style marker semantics
/// for essential boundaries, and returns reduced matrices plus gradient constraints.
pub fn assemble_hcurl_eigen_system_from_marker(
    h1: &H1Space<SimplexMesh<2>>,
    hcurl: &HCurlSpace<SimplexMesh<2>>,
    boundary_attributes: &[i32],
    marker: &[i32],
    mu: f64,
    epsilon: f64,
    quad_order: u8,
) -> HcurlEigenSystem {
    let k_full = VectorAssembler::assemble_bilinear(hcurl, &[&CurlCurlIntegrator { mu }], quad_order);
    let m_full =
        VectorAssembler::assemble_bilinear(hcurl, &[&VectorMassIntegrator { alpha: epsilon }], quad_order);

    let subspace = build_hcurl_constraint_subspace_from_marker(h1, hcurl, boundary_attributes, marker);
    let stiffness_free = extract_square_submatrix(&k_full, &subspace.hcurl_free_dofs);
    let mass_free = extract_square_submatrix(&m_full, &subspace.hcurl_free_dofs);

    HcurlEigenSystem {
        stiffness_free,
        mass_free,
        constraints: subspace.gradient_constraints,
        hcurl_free_dofs: subspace.hcurl_free_dofs,
        h1_free_dofs: subspace.h1_free_dofs,
    }
}

impl StaticMaxwellProblem {
    pub fn new(
        space: HCurlSpace<SimplexMesh<2>>,
        mat: CsrMatrix<f64>,
        rhs: Vec<f64>,
        quad_order: u8,
    ) -> Self {
        Self {
            space,
            mat,
            rhs,
            boundary: HcurlBoundaryConfig::new(),
            quad_order,
        }
    }

    pub fn with_boundary(mut self, boundary: HcurlBoundaryConfig) -> Self {
        self.boundary = boundary;
        self
    }

    pub fn boundary_mut(&mut self) -> &mut HcurlBoundaryConfig {
        &mut self.boundary
    }

    pub fn n_dofs(&self) -> usize {
        self.space.n_dofs()
    }

    pub fn solve(mut self) -> StaticMaxwellSolveOutput {
        let boundary_report = self.boundary.apply(
            &self.space,
            &mut self.mat,
            &mut self.rhs,
            self.quad_order,
        );
        let (solution, solve_result) = solve_hcurl_jacobi(&self.mat, &self.rhs);

        StaticMaxwellSolveOutput {
            space: self.space,
            solution,
            solve_result,
            boundary_report,
        }
    }
}

impl StaticMaxwellBuilder {
    pub fn new(space: HCurlSpace<SimplexMesh<2>>) -> Self {
        Self {
            space,
            quad_order: 4,
            volume_model: StaticMaxwellVolumeModel::Isotropic { mu: 1.0, alpha: 1.0 },
            source: None,
            boundary: HcurlBoundaryConfig::new(),
        }
    }

    pub fn with_quad_order(mut self, quad_order: u8) -> Self {
        self.quad_order = quad_order;
        self
    }

    pub fn with_isotropic_coeffs(mut self, mu: f64, alpha: f64) -> Self {
        self.volume_model = StaticMaxwellVolumeModel::Isotropic { mu, alpha };
        self
    }

    pub fn with_frequency_isotropic(mut self, mu: f64, epsilon: f64, omega: f64) -> Self {
        let alpha = epsilon * omega * omega;
        self.volume_model = StaticMaxwellVolumeModel::Isotropic { mu, alpha };
        self
    }

    pub fn with_anisotropic_diag(mut self, mu: f64, sigma_x: f64, sigma_y: f64) -> Self {
        self.volume_model = StaticMaxwellVolumeModel::AnisotropicDiag {
            mu,
            sigma_x,
            sigma_y,
        };
        self
    }

    pub fn with_source_fn<F>(mut self, source: F) -> Self
    where
        F: Fn(&[f64]) -> [f64; 2] + Send + Sync + 'static,
    {
        self.source = Some(Box::new(source));
        self
    }

    pub fn with_anisotropic_matrix_fn<F>(mut self, mu: f64, sigma: F) -> Self
    where
        F: Fn(&[f64]) -> [f64; 4] + Send + Sync + 'static,
    {
        self.volume_model = StaticMaxwellVolumeModel::AnisotropicMatrixFn {
            mu,
            sigma: Box::new(sigma),
        };
        self
    }

    pub fn with_boundary(mut self, boundary: HcurlBoundaryConfig) -> Self {
        self.boundary = boundary;
        self
    }

    pub fn add_pec_zero(self, tags: &[i32]) -> Self {
        let mut this = self;
        this.boundary.add_pec_zero_on(BoundarySelection::Tags(tags));
        this
    }

    pub fn add_pec_zero_on(self, selection: BoundarySelection<'_>) -> Self {
        let mut this = self;
        this.boundary.add_pec_zero_on(selection);
        this
    }

    pub fn add_pec_zero_from_marker(self, boundary_attributes: &[i32], marker: &[i32]) -> Self {
        self.add_pec_zero_on(BoundarySelection::Marker {
            boundary_attributes,
            marker,
        })
    }

    pub fn add_tangential_robin<F>(self, tags: &[i32], gamma: f64, data: F) -> Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        self.add_tangential_robin_on(BoundarySelection::Tags(tags), gamma, data)
    }

    pub fn add_tangential_robin_on<F>(
        self,
        selection: BoundarySelection<'_>,
        gamma: f64,
        data: F,
    ) -> Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        let mut this = self;
        this.boundary.add_tangential_robin_on(selection, gamma, data);
        this
    }

    pub fn add_impedance<F>(self, tags: &[i32], gamma: f64, data: F) -> Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        self.add_impedance_on(BoundarySelection::Tags(tags), gamma, data)
    }

    pub fn add_impedance_on<F>(self, selection: BoundarySelection<'_>, gamma: f64, data: F) -> Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        let mut this = self;
        this.boundary.add_impedance_on(selection, gamma, data);
        this
    }

    pub fn add_impedance_physical_on<F>(
        self,
        selection: BoundarySelection<'_>,
        epsilon: f64,
        mu: f64,
        data: F,
    ) -> Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        let mut this = self;
        this.boundary
            .add_impedance_physical_on(selection, epsilon, mu, data);
        this
    }

    pub fn add_impedance_physical<F>(
        self,
        tags: &[i32],
        epsilon: f64,
        mu: f64,
        data: F,
    ) -> Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        self.add_impedance_physical_on(BoundarySelection::Tags(tags), epsilon, mu, data)
    }

    pub fn add_absorbing<F>(self, tags: &[i32], gamma_abs: f64, data: F) -> Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        self.add_absorbing_on(BoundarySelection::Tags(tags), gamma_abs, data)
    }

    pub fn add_absorbing_on<F>(
        self,
        selection: BoundarySelection<'_>,
        gamma_abs: f64,
        data: F,
    ) -> Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        let mut this = self;
        this.boundary.add_absorbing_on(selection, gamma_abs, data);
        this
    }

    pub fn add_absorbing_physical_on<F>(
        self,
        selection: BoundarySelection<'_>,
        epsilon: f64,
        mu: f64,
        data: F,
    ) -> Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        let mut this = self;
        this.boundary
            .add_absorbing_physical_on(selection, epsilon, mu, data);
        this
    }

    pub fn add_absorbing_physical<F>(
        self,
        tags: &[i32],
        epsilon: f64,
        mu: f64,
        data: F,
    ) -> Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        self.add_absorbing_physical_on(BoundarySelection::Tags(tags), epsilon, mu, data)
    }

    pub fn add_tangential_drive<F>(self, tags: &[i32], gamma: f64, data: F) -> Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        self.add_tangential_drive_on(BoundarySelection::Tags(tags), gamma, data)
    }

    pub fn add_tangential_drive_on<F>(
        self,
        selection: BoundarySelection<'_>,
        gamma: f64,
        data: F,
    ) -> Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        let mut this = self;
        this.boundary.add_tangential_drive_on(selection, gamma, data);
        this
    }

    pub fn add_tangential_robin_from_marker<F>(
        self,
        boundary_attributes: &[i32],
        marker: &[i32],
        gamma: f64,
        data: F,
    ) -> Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        self.add_tangential_robin_on(
            BoundarySelection::Marker {
                boundary_attributes,
                marker,
            },
            gamma,
            data,
        )
    }

    pub fn add_impedance_from_marker<F>(
        self,
        boundary_attributes: &[i32],
        marker: &[i32],
        gamma: f64,
        data: F,
    ) -> Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        self.add_impedance_on(
            BoundarySelection::Marker {
                boundary_attributes,
                marker,
            },
            gamma,
            data,
        )
    }

    pub fn add_impedance_physical_from_marker<F>(
        self,
        boundary_attributes: &[i32],
        marker: &[i32],
        epsilon: f64,
        mu: f64,
        data: F,
    ) -> Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        self.add_impedance_physical_on(
            BoundarySelection::Marker {
                boundary_attributes,
                marker,
            },
            epsilon,
            mu,
            data,
        )
    }

    pub fn add_absorbing_from_marker<F>(
        self,
        boundary_attributes: &[i32],
        marker: &[i32],
        gamma_abs: f64,
        data: F,
    ) -> Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        self.add_absorbing_on(
            BoundarySelection::Marker {
                boundary_attributes,
                marker,
            },
            gamma_abs,
            data,
        )
    }

    pub fn add_absorbing_physical_from_marker<F>(
        self,
        boundary_attributes: &[i32],
        marker: &[i32],
        epsilon: f64,
        mu: f64,
        data: F,
    ) -> Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        self.add_absorbing_physical_on(
            BoundarySelection::Marker {
                boundary_attributes,
                marker,
            },
            epsilon,
            mu,
            data,
        )
    }

    pub fn add_tangential_drive_from_marker<F>(
        self,
        boundary_attributes: &[i32],
        marker: &[i32],
        gamma: f64,
        data: F,
    ) -> Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        self.add_tangential_drive_on(
            BoundarySelection::Marker {
                boundary_attributes,
                marker,
            },
            gamma,
            data,
        )
    }

    pub fn boundary_mut(&mut self) -> &mut HcurlBoundaryConfig {
        &mut self.boundary
    }

    pub fn build(self) -> StaticMaxwellProblem {
        let mat = match self.volume_model {
            StaticMaxwellVolumeModel::Isotropic { mu, alpha } => VectorAssembler::assemble_bilinear(
                &self.space,
                &[&CurlCurlIntegrator { mu }, &VectorMassIntegrator { alpha }],
                self.quad_order,
            ),
            StaticMaxwellVolumeModel::AnisotropicDiag {
                mu,
                sigma_x,
                sigma_y,
            } => {
                let sigma = ConstantMatrixCoeff(vec![sigma_x, 0.0, 0.0, sigma_y]);
                VectorAssembler::assemble_bilinear(
                    &self.space,
                    &[&CurlCurlIntegrator { mu }, &VectorMassTensorIntegrator { alpha: sigma }],
                    self.quad_order,
                )
            }
            StaticMaxwellVolumeModel::AnisotropicMatrixFn { mu, sigma } => {
                let sigma = FnMatrixCoeff(move |x: &[f64], out: &mut [f64]| {
                    let s = sigma(x);
                    out[0] = s[0];
                    out[1] = s[1];
                    out[2] = s[2];
                    out[3] = s[3];
                });
                VectorAssembler::assemble_bilinear(
                    &self.space,
                    &[&CurlCurlIntegrator { mu }, &VectorMassTensorIntegrator { alpha: sigma }],
                    self.quad_order,
                )
            }
        };

        let rhs = if let Some(source) = self.source {
            let source_integrator = FnVectorSourceIntegrator { f: source.as_ref() };
            VectorAssembler::assemble_linear(&self.space, &[&source_integrator], self.quad_order)
        } else {
            vec![0.0_f64; self.space.n_dofs()]
        };

        StaticMaxwellProblem::new(self.space, mat, rhs, self.quad_order).with_boundary(self.boundary)
    }
}

pub fn assemble_isotropic_hcurl_volume(
    space: &HCurlSpace<SimplexMesh<2>>,
    quad_order: u8,
) -> CsrMatrix<f64> {
    assemble_isotropic_hcurl_system(space, None, &[], quad_order)
}

pub fn assemble_isotropic_hcurl_system(
    space: &HCurlSpace<SimplexMesh<2>>,
    gamma: Option<f64>,
    boundary_tags: &[i32],
    quad_order: u8,
) -> CsrMatrix<f64> {
    let volume = VectorAssembler::assemble_bilinear(
        space,
        &[&CurlCurlIntegrator { mu: 1.0 }, &VectorMassIntegrator { alpha: 1.0 }],
        quad_order,
    );

    match gamma {
        Some(value) if value != 0.0 => {
            let boundary = VectorBoundaryAssembler::assemble_boundary_bilinear(
                space,
                &[&TangentialMassIntegrator { gamma: value }],
                boundary_tags,
                quad_order,
            );
            volume.add(&boundary)
        }
        _ => volume,
    }
}

pub fn assemble_tangential_boundary_rhs<F>(
    space: &HCurlSpace<SimplexMesh<2>>,
    boundary_tags: &[i32],
    quad_order: u8,
    g: F,
) -> Vec<f64>
where
    F: Fn(&[f64], &[f64]) -> f64 + Send + Sync,
{
    VectorBoundaryAssembler::assemble_boundary_linear(
        space,
        &[&TangentialTraceLFIntegrator::new(|ctx, normal, out| {
            out[0] = g(ctx.x, normal);
        })],
        boundary_tags,
        quad_order,
    )
}

pub fn add_assign(dst: &mut [f64], src: &[f64]) {
    assert_eq!(dst.len(), src.len(), "vector length mismatch");
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += *s;
    }
}

pub fn apply_pec_zero(
    space: &HCurlSpace<SimplexMesh<2>>,
    mat: &mut CsrMatrix<f64>,
    rhs: &mut Vec<f64>,
    boundary_tags: &[i32],
) -> usize {
    let bnd = boundary_dofs_hcurl(space.mesh(), space, boundary_tags);
    let vals = vec![0.0_f64; bnd.len()];
    apply_dirichlet(mat, rhs, &bnd, &vals);
    bnd.len()
}

pub fn add_tangential_robin_boundary<F>(
    space: &HCurlSpace<SimplexMesh<2>>,
    mat: &mut CsrMatrix<f64>,
    rhs: &mut [f64],
    boundary_tags: &[i32],
    gamma: f64,
    quad_order: u8,
    g: F,
)
where
    F: Fn(&[f64], &[f64]) -> f64 + Send + Sync,
{
    let boundary = VectorBoundaryAssembler::assemble_boundary_bilinear(
        space,
        &[&TangentialMassIntegrator { gamma }],
        boundary_tags,
        quad_order,
    );
    *mat = mat.add(&boundary);

    let boundary_rhs = assemble_tangential_boundary_rhs(space, boundary_tags, quad_order, g);
    add_assign(rhs, &boundary_rhs);
}

pub fn solve_hcurl_jacobi(mat: &CsrMatrix<f64>, rhs: &[f64]) -> (Vec<f64>, SolveResult) {
    let mut u = vec![0.0_f64; rhs.len()];
    let cfg = SolverConfig {
        rtol: 1e-10,
        atol: 0.0,
        max_iter: 10_000,
        verbose: false,
        ..SolverConfig::default()
    };
    let res = solve_pcg_jacobi(mat, rhs, &mut u, &cfg).expect("solver failed");
    (u, res)
}

pub fn l2_error_hcurl_exact<F>(
    space: &HCurlSpace<SimplexMesh<2>>,
    uh: &[f64],
    exact: F,
) -> f64
where
    F: Fn(&[f64]) -> [f64; 2],
{
    let mesh = space.mesh();
    let ref_elem = TriND1;
    let quad = ref_elem.quadrature(6);
    let n_ldofs = ref_elem.n_dofs();

    let mut err2 = 0.0_f64;
    let mut ref_phi = vec![0.0; n_ldofs * 2];

    for e in mesh.elem_iter() {
        let nodes = mesh.element_nodes(e);
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let signs = space.element_signs(e);

        let x0 = mesh.node_coords(nodes[0]);
        let x1 = mesh.node_coords(nodes[1]);
        let x2 = mesh.node_coords(nodes[2]);
        let j00 = x1[0] - x0[0];
        let j01 = x2[0] - x0[0];
        let j10 = x1[1] - x0[1];
        let j11 = x2[1] - x0[1];
        let det_j = (j00 * j11 - j01 * j10).abs();

        let inv_det = 1.0 / (j00 * j11 - j01 * j10);
        let jit00 =  j11 * inv_det;
        let jit01 = -j10 * inv_det;
        let jit10 = -j01 * inv_det;
        let jit11 =  j00 * inv_det;

        for (qi, xi) in quad.points.iter().enumerate() {
            let w = quad.weights[qi] * det_j;
            let xp = [
                x0[0] + j00 * xi[0] + j01 * xi[1],
                x0[1] + j10 * xi[0] + j11 * xi[1],
            ];

            ref_elem.eval_basis_vec(xi, &mut ref_phi);

            let mut eh = [0.0_f64; 2];
            for i in 0..n_ldofs {
                let s = signs[i];
                let phi_x = jit00 * ref_phi[i * 2] + jit01 * ref_phi[i * 2 + 1];
                let phi_y = jit10 * ref_phi[i * 2] + jit11 * ref_phi[i * 2 + 1];
                eh[0] += s * uh[dofs[i]] * phi_x;
                eh[1] += s * uh[dofs[i]] * phi_y;
            }

            let ex = exact(&xp);
            let dx = eh[0] - ex[0];
            let dy = eh[1] - ex[1];
            err2 += w * (dx * dx + dy * dy);
        }
    }

    err2.sqrt()
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::*;
    use fem_core::{ElemId, FaceId, NodeId};
    use fem_assembly::vector_integrator::{VectorLinearIntegrator, VectorQpData};

    struct ManufacturedSource;

    impl VectorLinearIntegrator for ManufacturedSource {
        fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f_elem: &mut [f64]) {
            let x = qp.x_phys;
            let fx = -PI * PI * (PI * x[0]).sin() * (PI * x[1]).cos();
            let fy = (PI * PI + 1.0) * (PI * x[0]).cos() * (PI * x[1]).sin();
            for i in 0..qp.n_dofs {
                let dot = qp.phi_vec[i * 2] * fx + qp.phi_vec[i * 2 + 1] * fy;
                f_elem[i] += qp.weight * dot;
            }
        }
    }

    #[derive(Clone)]
    struct OneHexMesh {
        nodes: Vec<[f64; 3]>,
        elem: [NodeId; 8],
        bfaces: Vec<[NodeId; 4]>,
        btags: Vec<i32>,
    }

    impl OneHexMesh {
        fn unit() -> Self {
            Self {
                nodes: vec![
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                ],
                elem: [0, 1, 2, 3, 4, 5, 6, 7],
                bfaces: vec![
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [0, 1, 5, 4],
                    [1, 2, 6, 5],
                    [2, 3, 7, 6],
                    [3, 0, 4, 7],
                ],
                btags: vec![1, 2, 3, 4, 5, 6],
            }
        }
    }

    impl MeshTopology for OneHexMesh {
        fn dim(&self) -> u8 { 3 }
        fn n_nodes(&self) -> usize { self.nodes.len() }
        fn n_elements(&self) -> usize { 1 }
        fn n_boundary_faces(&self) -> usize { self.bfaces.len() }
        fn element_nodes(&self, _elem: ElemId) -> &[NodeId] { &self.elem }
        fn element_type(&self, _elem: ElemId) -> fem_mesh::ElementType { fem_mesh::ElementType::Hex8 }
        fn element_tag(&self, _elem: ElemId) -> i32 { 1 }
        fn node_coords(&self, node: NodeId) -> &[f64] { &self.nodes[node as usize] }
        fn face_nodes(&self, face: FaceId) -> &[NodeId] { &self.bfaces[face as usize] }
        fn face_tag(&self, face: FaceId) -> i32 { self.btags[face as usize] }
        fn face_elements(&self, _face: FaceId) -> (ElemId, Option<ElemId>) { (0, None) }
    }

    #[test]
    fn ex3_hex8_zero_source_full_pec_smoke() {
        let mesh = OneHexMesh::unit();
        let space = HCurlSpace::new(mesh, 1);

        let mut mat = VectorAssembler::assemble_bilinear(
            &space,
            &[&CurlCurlIntegrator { mu: 1.0 }, &VectorMassIntegrator { alpha: 1.0 }],
            4,
        );
        let mut rhs = vec![0.0_f64; space.n_dofs()];

        let bnd = boundary_dofs_hcurl(space.mesh(), &space, &[1, 2, 3, 4, 5, 6]);
        let vals = vec![0.0_f64; bnd.len()];
        apply_dirichlet(&mut mat, &mut rhs, &bnd, &vals);

        let mut u = vec![0.0_f64; space.n_dofs()];
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 4000,
            verbose: false,
            ..SolverConfig::default()
        };
        let res = solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg).expect("hex8 ex3 smoke solve failed");
        let norm_u = u.iter().map(|v| v * v).sum::<f64>().sqrt();

        assert!(res.converged, "hex8 ex3 smoke did not converge");
        assert!(norm_u < 1e-12, "hex8 ex3 smoke nonzero solution norm = {norm_u}");
    }

    #[test]
    fn builder_zero_source_with_pec_has_zero_solution() {
        let mesh = SimplexMesh::<2>::unit_square_tri(6);
        let space = HCurlSpace::new(mesh, 1);

        let mut boundary = HcurlBoundaryConfig::new();
        boundary.add_pec_zero(&[1, 2, 3, 4]);

        let problem = StaticMaxwellBuilder::new(space)
            .with_isotropic_coeffs(1.0, 1.0)
            .with_boundary(boundary)
            .build();

        let solved = problem.solve();
        let norm_u = solved.solution.iter().map(|v| v * v).sum::<f64>().sqrt();

        assert!(solved.solve_result.converged);
        assert!(norm_u < 1e-12, "solution norm = {norm_u}");
        assert!(solved.boundary_report.essential_dofs > 0);
    }

    #[test]
    fn builder_zero_source_with_pec_has_zero_solution_on_quad_mesh() {
        let mesh = SimplexMesh::<2>::unit_square_quad(6);
        let space = HCurlSpace::new(mesh, 1);

        let mut boundary = HcurlBoundaryConfig::new();
        boundary.add_pec_zero(&[1, 2, 3, 4]);

        let problem = StaticMaxwellBuilder::new(space)
            .with_isotropic_coeffs(1.0, 1.0)
            .with_boundary(boundary)
            .build();

        let solved = problem.solve();
        let norm_u = solved.solution.iter().map(|v| v * v).sum::<f64>().sqrt();

        assert!(solved.solve_result.converged);
        assert!(norm_u < 1e-12, "quad mesh solution norm = {norm_u}");
        assert!(solved.boundary_report.essential_dofs > 0);
    }

    #[test]
    fn builder_matches_legacy_assembly_path() {
        const GAMMA: f64 = 2.0;

        let mesh1 = SimplexMesh::<2>::unit_square_tri(6);
        let space1 = HCurlSpace::new(mesh1, 1);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(6);
        let space2 = HCurlSpace::new(mesh2, 1);

        let data = move |x: &[f64], n: &[f64]| {
            let e = [0.0, (PI * x[0]).cos() * (PI * x[1]).sin()];
            GAMMA * (e[0] * n[1] - e[1] * n[0])
        };

        let mut builder_bc = HcurlBoundaryConfig::new();
        builder_bc.add_tangential_robin(&[1, 2, 3, 4], GAMMA, data);

        let builder_problem = StaticMaxwellBuilder::new(space1)
            .with_quad_order(4)
            .with_isotropic_coeffs(1.0, 1.0)
            .with_source_fn(|x| {
                [
                    -PI * PI * (PI * x[0]).sin() * (PI * x[1]).cos(),
                    (PI * PI + 1.0) * (PI * x[0]).cos() * (PI * x[1]).sin(),
                ]
            })
            .with_boundary(builder_bc)
            .build();
        let builder_solved = builder_problem.solve();

        let mut legacy_mat = assemble_isotropic_hcurl_volume(&space2, 4);
        let mut legacy_rhs = VectorAssembler::assemble_linear(&space2, &[&ManufacturedSource], 4);
        let mut legacy_bc = HcurlBoundaryConfig::new();
        legacy_bc.add_tangential_robin(&[1, 2, 3, 4], GAMMA, data);
        legacy_bc.apply(&space2, &mut legacy_mat, &mut legacy_rhs, 4);
        let (legacy_u, legacy_res) = solve_hcurl_jacobi(&legacy_mat, &legacy_rhs);

        assert!(builder_solved.solve_result.converged);
        assert!(legacy_res.converged);
        assert_eq!(builder_solved.solution.len(), legacy_u.len());

        let diff = builder_solved
            .solution
            .iter()
            .zip(legacy_u.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f64>()
            .sqrt();

        assert!(diff < 1e-11, "builder/legacy solution mismatch = {diff}");
    }

    #[test]
    fn builder_matrix_fn_identity_matches_isotropic() {
        let mesh1 = SimplexMesh::<2>::unit_square_tri(6);
        let space1 = HCurlSpace::new(mesh1, 1);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(6);
        let space2 = HCurlSpace::new(mesh2, 1);

        let src = |x: &[f64]| {
            [
                -PI * PI * (PI * x[0]).sin() * (PI * x[1]).cos(),
                (PI * PI + 1.0) * (PI * x[0]).cos() * (PI * x[1]).sin(),
            ]
        };

        let solved_iso = StaticMaxwellBuilder::new(space1)
            .with_quad_order(4)
            .with_isotropic_coeffs(1.0, 1.0)
            .with_source_fn(src)
            .build()
            .solve();

        let solved_mat_fn = StaticMaxwellBuilder::new(space2)
            .with_quad_order(4)
            .with_anisotropic_matrix_fn(1.0, |_x| [1.0, 0.0, 0.0, 1.0])
            .with_source_fn(src)
            .build()
            .solve();

        assert!(solved_iso.solve_result.converged);
        assert!(solved_mat_fn.solve_result.converged);
        assert_eq!(solved_iso.solution.len(), solved_mat_fn.solution.len());

        let diff = solved_iso
            .solution
            .iter()
            .zip(solved_mat_fn.solution.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f64>()
            .sqrt();

        assert!(diff < 1e-11, "matrix-fn/isotropic solution mismatch = {diff}");
    }

    struct MixedSource;

    impl VectorLinearIntegrator for MixedSource {
        fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f_elem: &mut [f64]) {
            let x = qp.x_phys;
            let src = [
                (PI * x[1]).sin() + 0.2 * (PI * x[0]).cos(),
                (PI * x[0]).sin() - 0.3 * (PI * x[1]).cos(),
            ];
            for i in 0..qp.n_dofs {
                let dot = qp.phi_vec[i * 2] * src[0] + qp.phi_vec[i * 2 + 1] * src[1];
                f_elem[i] += qp.weight * dot;
            }
        }
    }

    fn mixed_source_value(x: &[f64]) -> [f64; 2] {
        [
            (PI * x[1]).sin() + 0.2 * (PI * x[0]).cos(),
            (PI * x[0]).sin() - 0.3 * (PI * x[1]).cos(),
        ]
    }

    fn mixed_robin_data(x: &[f64], n: &[f64]) -> f64 {
        let e = [
            0.4 * (PI * x[1]).sin(),
            -0.25 * (PI * x[0]).sin(),
        ];
        let trace = e[0] * n[1] - e[1] * n[0];
        1.5 * trace + 0.1 * (x[0] + x[1])
    }

    #[test]
    fn builder_mixed_boundary_matches_low_level_pipeline() {
        let mesh1 = SimplexMesh::<2>::unit_square_tri(6);
        let space1 = HCurlSpace::new(mesh1, 1);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(6);
        let space2 = HCurlSpace::new(mesh2, 1);

        let mut cfg = HcurlBoundaryConfig::new();
        cfg.add_tangential_robin(&[2, 4], 1.5, mixed_robin_data)
            .add_pec_zero(&[1, 3]);

        let builder_solved = StaticMaxwellBuilder::new(space1)
            .with_quad_order(4)
            .with_isotropic_coeffs(1.0, 1.0)
            .with_source_fn(mixed_source_value)
            .with_boundary(cfg)
            .build()
            .solve();

        let mut legacy_mat = assemble_isotropic_hcurl_volume(&space2, 4);
        let mut legacy_rhs = VectorAssembler::assemble_linear(&space2, &[&MixedSource], 4);
        add_tangential_robin_boundary(
            &space2,
            &mut legacy_mat,
            &mut legacy_rhs,
            &[2, 4],
            1.5,
            4,
            mixed_robin_data,
        );
        let n_ess = apply_pec_zero(&space2, &mut legacy_mat, &mut legacy_rhs, &[1, 3]);
        let (legacy_u, legacy_res) = solve_hcurl_jacobi(&legacy_mat, &legacy_rhs);

        assert!(builder_solved.solve_result.converged);
        assert!(legacy_res.converged);
        assert_eq!(builder_solved.boundary_report.essential_dofs, n_ess);
        assert_eq!(builder_solved.solution.len(), legacy_u.len());

        let diff = builder_solved
            .solution
            .iter()
            .zip(legacy_u.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f64>()
            .sqrt();

        assert!(diff < 1e-11, "mixed boundary builder/legacy mismatch = {diff}");
    }

    #[test]
    fn boundary_config_condition_order_is_result_invariant() {
        let mesh_a = SimplexMesh::<2>::unit_square_tri(6);
        let space_a = HCurlSpace::new(mesh_a, 1);
        let mesh_b = SimplexMesh::<2>::unit_square_tri(6);
        let space_b = HCurlSpace::new(mesh_b, 1);

        let mut cfg_a = HcurlBoundaryConfig::new();
        cfg_a
            .add_pec_zero(&[1, 3])
            .add_tangential_robin(&[2, 4], 1.5, mixed_robin_data);

        let mut cfg_b = HcurlBoundaryConfig::new();
        cfg_b
            .add_tangential_robin(&[2, 4], 1.5, mixed_robin_data)
            .add_pec_zero(&[1, 3]);

        let solved_a = StaticMaxwellBuilder::new(space_a)
            .with_quad_order(4)
            .with_isotropic_coeffs(1.0, 1.0)
            .with_source_fn(mixed_source_value)
            .with_boundary(cfg_a)
            .build()
            .solve();

        let solved_b = StaticMaxwellBuilder::new(space_b)
            .with_quad_order(4)
            .with_isotropic_coeffs(1.0, 1.0)
            .with_source_fn(mixed_source_value)
            .with_boundary(cfg_b)
            .build()
            .solve();

        assert!(solved_a.solve_result.converged);
        assert!(solved_b.solve_result.converged);
        assert_eq!(
            solved_a.boundary_report.essential_dofs,
            solved_b.boundary_report.essential_dofs
        );
        assert_eq!(solved_a.solution.len(), solved_b.solution.len());

        let diff = solved_a
            .solution
            .iter()
            .zip(solved_b.solution.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f64>()
            .sqrt();

        assert!(diff < 1e-11, "boundary config order changed result, diff = {diff}");
    }

    #[test]
    fn builder_gamma_sweep_matches_low_level_pipeline() {
        let gammas = [0.0_f64, 0.25, 1.0, 2.0, 5.0];

        for gamma in gammas {
            let mesh1 = SimplexMesh::<2>::unit_square_tri(6);
            let space1 = HCurlSpace::new(mesh1, 1);
            let mesh2 = SimplexMesh::<2>::unit_square_tri(6);
            let space2 = HCurlSpace::new(mesh2, 1);

            let robin_data = move |x: &[f64], n: &[f64]| {
                let e = [0.0, (PI * x[0]).cos() * (PI * x[1]).sin()];
                gamma * (e[0] * n[1] - e[1] * n[0])
            };

            let mut cfg = HcurlBoundaryConfig::new();
            cfg.add_tangential_robin(&[1, 2, 3, 4], gamma, robin_data);

            let solved_builder = StaticMaxwellBuilder::new(space1)
                .with_quad_order(4)
                .with_isotropic_coeffs(1.0, 1.0)
                .with_source_fn(|x| {
                    [
                        -PI * PI * (PI * x[0]).sin() * (PI * x[1]).cos(),
                        (PI * PI + 1.0) * (PI * x[0]).cos() * (PI * x[1]).sin(),
                    ]
                })
                .with_boundary(cfg)
                .build()
                .solve();

            let mut legacy_mat = assemble_isotropic_hcurl_volume(&space2, 4);
            let mut legacy_rhs = VectorAssembler::assemble_linear(&space2, &[&ManufacturedSource], 4);
            add_tangential_robin_boundary(
                &space2,
                &mut legacy_mat,
                &mut legacy_rhs,
                &[1, 2, 3, 4],
                gamma,
                4,
                move |x, n| {
                    let e = [0.0, (PI * x[0]).cos() * (PI * x[1]).sin()];
                    gamma * (e[0] * n[1] - e[1] * n[0])
                },
            );
            let (legacy_u, legacy_res) = solve_hcurl_jacobi(&legacy_mat, &legacy_rhs);

            assert!(solved_builder.solve_result.converged, "builder not converged for gamma={gamma}");
            assert!(legacy_res.converged, "legacy not converged for gamma={gamma}");
            assert_eq!(solved_builder.solution.len(), legacy_u.len());

            let diff = solved_builder
                .solution
                .iter()
                .zip(legacy_u.iter())
                .map(|(a, b)| {
                    let d = a - b;
                    d * d
                })
                .sum::<f64>()
                .sqrt();

            assert!(diff < 1e-11, "gamma={gamma} builder/legacy mismatch = {diff}");
        }
    }

    #[test]
    fn builder_frequency_isotropic_matches_direct_alpha() {
        let omega = 2.75;
        let epsilon = 1.2;
        let alpha = epsilon * omega * omega;

        let mesh_a = SimplexMesh::<2>::unit_square_tri(6);
        let space_a = HCurlSpace::new(mesh_a, 1);
        let mesh_b = SimplexMesh::<2>::unit_square_tri(6);
        let space_b = HCurlSpace::new(mesh_b, 1);

        let solved_freq = StaticMaxwellBuilder::new(space_a)
            .with_quad_order(4)
            .with_frequency_isotropic(1.0, epsilon, omega)
            .with_source_fn(move |x| {
                [
                    -PI * PI * (PI * x[0]).sin() * (PI * x[1]).cos(),
                    (PI * PI + alpha) * (PI * x[0]).cos() * (PI * x[1]).sin(),
                ]
            })
            .build()
            .solve();

        let solved_alpha = StaticMaxwellBuilder::new(space_b)
            .with_quad_order(4)
            .with_isotropic_coeffs(1.0, alpha)
            .with_source_fn(move |x| {
                [
                    -PI * PI * (PI * x[0]).sin() * (PI * x[1]).cos(),
                    (PI * PI + alpha) * (PI * x[0]).cos() * (PI * x[1]).sin(),
                ]
            })
            .build()
            .solve();

        assert!(solved_freq.solve_result.converged);
        assert!(solved_alpha.solve_result.converged);
        assert_eq!(solved_freq.solution.len(), solved_alpha.solution.len());

        let diff = solved_freq
            .solution
            .iter()
            .zip(solved_alpha.solution.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f64>()
            .sqrt();

        assert!(diff < 1e-11, "frequency/direct-alpha mismatch = {diff}");
    }

    #[test]
    fn marker_api_matches_explicit_tag_api() {
        let attrs = [1, 2, 3, 4];
        let pec_marker = [1, 0, 1, 0];
        let robin_marker = [0, 1, 0, 1];

        let mesh_a = SimplexMesh::<2>::unit_square_tri(6);
        let space_a = HCurlSpace::new(mesh_a, 1);
        let mesh_b = SimplexMesh::<2>::unit_square_tri(6);
        let space_b = HCurlSpace::new(mesh_b, 1);

        let src = |x: &[f64]| {
            [
                (PI * x[1]).sin() + 0.2 * (PI * x[0]).cos(),
                (PI * x[0]).sin() - 0.3 * (PI * x[1]).cos(),
            ]
        };

        let robin_data = |x: &[f64], n: &[f64]| {
            let e = [0.4 * (PI * x[1]).sin(), -0.25 * (PI * x[0]).sin()];
            1.5 * (e[0] * n[1] - e[1] * n[0]) + 0.1 * (x[0] + x[1])
        };

        let mut cfg_marker = HcurlBoundaryConfig::new();
        cfg_marker
            .add_pec_zero_from_marker(&attrs, &pec_marker)
            .add_tangential_robin_from_marker(&attrs, &robin_marker, 1.5, robin_data);

        let solved_marker = StaticMaxwellBuilder::new(space_a)
            .with_quad_order(4)
            .with_isotropic_coeffs(1.0, 1.0)
            .with_source_fn(src)
            .with_boundary(cfg_marker)
            .build()
            .solve();

        let mut cfg_tags = HcurlBoundaryConfig::new();
        cfg_tags
            .add_pec_zero(&[1, 3])
            .add_tangential_robin(&[2, 4], 1.5, robin_data);

        let solved_tags = StaticMaxwellBuilder::new(space_b)
            .with_quad_order(4)
            .with_isotropic_coeffs(1.0, 1.0)
            .with_source_fn(src)
            .with_boundary(cfg_tags)
            .build()
            .solve();

        assert!(solved_marker.solve_result.converged);
        assert!(solved_tags.solve_result.converged);
        assert_eq!(
            solved_marker.boundary_report.essential_dofs,
            solved_tags.boundary_report.essential_dofs
        );

        let diff = solved_marker
            .solution
            .iter()
            .zip(solved_tags.solution.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f64>()
            .sqrt();

        assert!(diff < 1e-11, "marker/tag API mismatch = {diff}");
    }

    #[test]
    fn impedance_marker_api_matches_tangential_robin_marker_api() {
        let attrs = [1, 2, 3, 4];
        let marker = [1, 1, 1, 1];
        let gamma = 1.7;

        let mesh_a = SimplexMesh::<2>::unit_square_tri(6);
        let space_a = HCurlSpace::new(mesh_a, 1);
        let mesh_b = SimplexMesh::<2>::unit_square_tri(6);
        let space_b = HCurlSpace::new(mesh_b, 1);

        let source = |x: &[f64]| {
            [
                (PI * x[1]).sin() + 0.2 * (PI * x[0]).cos(),
                (PI * x[0]).sin() - 0.3 * (PI * x[1]).cos(),
            ]
        };

        let data = move |x: &[f64], n: &[f64]| {
            let e = [0.4 * (PI * x[1]).sin(), -0.25 * (PI * x[0]).sin()];
            gamma * (e[0] * n[1] - e[1] * n[0]) + 0.1 * (x[0] + x[1])
        };

        let solved_impedance = StaticMaxwellBuilder::new(space_a)
            .with_quad_order(4)
            .with_isotropic_coeffs(1.0, 1.0)
            .with_source_fn(source)
            .add_impedance_from_marker(&attrs, &marker, gamma, data)
            .build()
            .solve();

        let solved_robin = StaticMaxwellBuilder::new(space_b)
            .with_quad_order(4)
            .with_isotropic_coeffs(1.0, 1.0)
            .with_source_fn(source)
            .add_tangential_robin_from_marker(&attrs, &marker, gamma, data)
            .build()
            .solve();

        assert!(solved_impedance.solve_result.converged);
        assert!(solved_robin.solve_result.converged);

        let diff = solved_impedance
            .solution
            .iter()
            .zip(solved_robin.solution.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f64>()
            .sqrt();

        assert!(diff < 1e-11, "impedance/robin marker API mismatch = {diff}");
    }

    #[test]
    fn absorbing_marker_api_matches_tangential_robin_marker_api() {
        let attrs = [1, 2, 3, 4];
        let marker = [1, 1, 1, 1];
        let gamma_abs = 0.9;

        let mesh_a = SimplexMesh::<2>::unit_square_tri(6);
        let space_a = HCurlSpace::new(mesh_a, 1);
        let mesh_b = SimplexMesh::<2>::unit_square_tri(6);
        let space_b = HCurlSpace::new(mesh_b, 1);

        let source = |x: &[f64]| {
            [
                (PI * x[1]).sin() + 0.2 * (PI * x[0]).cos(),
                (PI * x[0]).sin() - 0.3 * (PI * x[1]).cos(),
            ]
        };

        let data = move |x: &[f64], n: &[f64]| {
            let e = [0.4 * (PI * x[1]).sin(), -0.25 * (PI * x[0]).sin()];
            -0.5 * PI * ((PI * x[0]).cos() - (PI * x[1]).cos())
                + gamma_abs * (e[0] * n[1] - e[1] * n[0])
        };

        let solved_absorbing = StaticMaxwellBuilder::new(space_a)
            .with_quad_order(4)
            .with_isotropic_coeffs(1.0, 1.0)
            .with_source_fn(source)
            .add_absorbing_from_marker(&attrs, &marker, gamma_abs, data)
            .build()
            .solve();

        let solved_robin = StaticMaxwellBuilder::new(space_b)
            .with_quad_order(4)
            .with_isotropic_coeffs(1.0, 1.0)
            .with_source_fn(source)
            .add_tangential_robin_from_marker(&attrs, &marker, gamma_abs, data)
            .build()
            .solve();

        assert!(solved_absorbing.solve_result.converged);
        assert!(solved_robin.solve_result.converged);

        let diff = solved_absorbing
            .solution
            .iter()
            .zip(solved_robin.solution.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f64>()
            .sqrt();

        assert!(diff < 1e-11, "absorbing/robin marker API mismatch = {diff}");
    }

    #[test]
    fn tangential_drive_marker_api_matches_tangential_robin_marker_api() {
        let attrs = [1, 2, 3, 4];
        let marker = [1, 1, 1, 1];
        let gamma = 1.25;

        let mesh_a = SimplexMesh::<2>::unit_square_tri(6);
        let space_a = HCurlSpace::new(mesh_a, 1);
        let mesh_b = SimplexMesh::<2>::unit_square_tri(6);
        let space_b = HCurlSpace::new(mesh_b, 1);

        let source = |x: &[f64]| {
            [
                (PI * x[1]).sin() + 0.2 * (PI * x[0]).cos(),
                (PI * x[0]).sin() - 0.3 * (PI * x[1]).cos(),
            ]
        };

        let data = move |x: &[f64], n: &[f64]| {
            let e = [0.4 * (PI * x[1]).sin(), -0.25 * (PI * x[0]).sin()];
            -0.5 * PI * ((PI * x[0]).cos() - (PI * x[1]).cos()) + gamma * (e[0] * n[1] - e[1] * n[0])
        };

        let solved_drive = StaticMaxwellBuilder::new(space_a)
            .with_quad_order(4)
            .with_isotropic_coeffs(1.0, 1.0)
            .with_source_fn(source)
            .add_tangential_drive_from_marker(&attrs, &marker, gamma, data)
            .build()
            .solve();

        let solved_robin = StaticMaxwellBuilder::new(space_b)
            .with_quad_order(4)
            .with_isotropic_coeffs(1.0, 1.0)
            .with_source_fn(source)
            .add_tangential_robin_from_marker(&attrs, &marker, gamma, data)
            .build()
            .solve();

        assert!(solved_drive.solve_result.converged);
        assert!(solved_robin.solve_result.converged);

        let diff = solved_drive
            .solution
            .iter()
            .zip(solved_robin.solution.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f64>()
            .sqrt();

        assert!(diff < 1e-11, "tangential-drive/robin marker API mismatch = {diff}");
    }

    #[test]
    fn boundary_selector_api_matches_split_tag_and_marker_paths() {
        let attrs = [1, 2, 3, 4];
        let pec_marker = [1, 0, 1, 0];
        let robin_marker = [0, 1, 0, 1];

        let mesh_a = SimplexMesh::<2>::unit_square_tri(6);
        let space_a = HCurlSpace::new(mesh_a, 1);
        let mesh_b = SimplexMesh::<2>::unit_square_tri(6);
        let space_b = HCurlSpace::new(mesh_b, 1);

        let src = |x: &[f64]| {
            [
                (PI * x[1]).sin() + 0.2 * (PI * x[0]).cos(),
                (PI * x[0]).sin() - 0.3 * (PI * x[1]).cos(),
            ]
        };

        let robin_data = |x: &[f64], n: &[f64]| {
            let e = [0.4 * (PI * x[1]).sin(), -0.25 * (PI * x[0]).sin()];
            1.5 * (e[0] * n[1] - e[1] * n[0]) + 0.1 * (x[0] + x[1])
        };

        let solved_selector = StaticMaxwellBuilder::new(space_a)
            .with_quad_order(4)
            .with_isotropic_coeffs(1.0, 1.0)
            .with_source_fn(src)
            .add_pec_zero_on(BoundarySelection::Marker {
                boundary_attributes: &attrs,
                marker: &pec_marker,
            })
            .add_tangential_robin_on(
                BoundarySelection::Marker {
                    boundary_attributes: &attrs,
                    marker: &robin_marker,
                },
                1.5,
                robin_data,
            )
            .build()
            .solve();

        let solved_split = StaticMaxwellBuilder::new(space_b)
            .with_quad_order(4)
            .with_isotropic_coeffs(1.0, 1.0)
            .with_source_fn(src)
            .add_pec_zero(&[1, 3])
            .add_tangential_robin(&[2, 4], 1.5, robin_data)
            .build()
            .solve();

        assert!(solved_selector.solve_result.converged);
        assert!(solved_split.solve_result.converged);
        assert_eq!(
            solved_selector.boundary_report.essential_dofs,
            solved_split.boundary_report.essential_dofs
        );

        let diff = solved_selector
            .solution
            .iter()
            .zip(solved_split.solution.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f64>()
            .sqrt();

        assert!(diff < 1e-11, "selector/split API mismatch = {diff}");
    }

    #[test]
    fn impedance_physical_api_matches_explicit_gamma() {
        let attrs = [1, 2, 3, 4];
        let marker = [1, 1, 1, 1];
        let epsilon = 2.25_f64;
        let mu = 0.64_f64;
        let gamma = (epsilon / mu).sqrt();

        let mesh_a = SimplexMesh::<2>::unit_square_tri(6);
        let space_a = HCurlSpace::new(mesh_a, 1);
        let mesh_b = SimplexMesh::<2>::unit_square_tri(6);
        let space_b = HCurlSpace::new(mesh_b, 1);

        let source = |x: &[f64]| {
            [
                (PI * x[1]).sin() + 0.2 * (PI * x[0]).cos(),
                (PI * x[0]).sin() - 0.3 * (PI * x[1]).cos(),
            ]
        };
        let data = |x: &[f64], n: &[f64]| {
            let e = [0.4 * (PI * x[1]).sin(), -0.25 * (PI * x[0]).sin()];
            (e[0] * n[1] - e[1] * n[0]) + 0.05 * (x[0] + x[1])
        };

        let solved_physical = StaticMaxwellBuilder::new(space_a)
            .with_quad_order(4)
            .with_isotropic_coeffs(1.0, 1.0)
            .with_source_fn(source)
            .add_impedance_physical_from_marker(&attrs, &marker, epsilon, mu, data)
            .build()
            .solve();

        let solved_explicit = StaticMaxwellBuilder::new(space_b)
            .with_quad_order(4)
            .with_isotropic_coeffs(1.0, 1.0)
            .with_source_fn(source)
            .add_impedance_from_marker(&attrs, &marker, gamma, data)
            .build()
            .solve();

        assert!(solved_physical.solve_result.converged);
        assert!(solved_explicit.solve_result.converged);

        let diff = solved_physical
            .solution
            .iter()
            .zip(solved_explicit.solution.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f64>()
            .sqrt();

        assert!(diff < 1e-11, "impedance physical/explicit mismatch = {diff}");
    }

    #[test]
    fn absorbing_physical_api_matches_explicit_gamma() {
        let attrs = [1, 2, 3, 4];
        let marker = [1, 1, 1, 1];
        let epsilon = 1.44_f64;
        let mu = 0.81_f64;
        let gamma_abs = (epsilon / mu).sqrt();

        let mesh_a = SimplexMesh::<2>::unit_square_tri(6);
        let space_a = HCurlSpace::new(mesh_a, 1);
        let mesh_b = SimplexMesh::<2>::unit_square_tri(6);
        let space_b = HCurlSpace::new(mesh_b, 1);

        let source = |x: &[f64]| {
            [
                (PI * x[1]).sin() + 0.1 * (PI * x[0]).cos(),
                (PI * x[0]).sin() - 0.2 * (PI * x[1]).cos(),
            ]
        };
        let data = |x: &[f64], n: &[f64]| {
            let e = [0.3 * (PI * x[1]).sin(), -0.2 * (PI * x[0]).sin()];
            (e[0] * n[1] - e[1] * n[0]) - 0.03 * (x[0] - x[1])
        };

        let solved_physical = StaticMaxwellBuilder::new(space_a)
            .with_quad_order(4)
            .with_isotropic_coeffs(1.0, 1.0)
            .with_source_fn(source)
            .add_absorbing_physical_from_marker(&attrs, &marker, epsilon, mu, data)
            .build()
            .solve();

        let solved_explicit = StaticMaxwellBuilder::new(space_b)
            .with_quad_order(4)
            .with_isotropic_coeffs(1.0, 1.0)
            .with_source_fn(source)
            .add_absorbing_from_marker(&attrs, &marker, gamma_abs, data)
            .build()
            .solve();

        assert!(solved_physical.solve_result.converged);
        assert!(solved_explicit.solve_result.converged);

        let diff = solved_physical
            .solution
            .iter()
            .zip(solved_explicit.solution.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f64>()
            .sqrt();

        assert!(diff < 1e-11, "absorbing physical/explicit mismatch = {diff}");
    }

    #[test]
    fn boundary_material_frequency_matrix_regression_smoke() {
        let attrs = [1, 2, 3, 4];

        // Case A: isotropic + full PEC
        let solved_a = StaticMaxwellBuilder::new(HCurlSpace::new(SimplexMesh::<2>::unit_square_tri(6), 1))
            .with_quad_order(4)
            .with_isotropic_coeffs(1.0, 1.0)
            .with_source_fn(|x| [(PI * x[1]).sin(), (PI * x[0]).sin()])
            .add_pec_zero_from_marker(&attrs, &[1, 1, 1, 1])
            .build()
            .solve();
        assert!(solved_a.solve_result.converged);

        // Case B: frequency isotropic + physical impedance on subset marker
        let solved_b = StaticMaxwellBuilder::new(HCurlSpace::new(SimplexMesh::<2>::unit_square_tri(6), 1))
            .with_quad_order(4)
            .with_frequency_isotropic(1.0, 2.0, 1.5)
            .with_source_fn(|x| [
                -PI * PI * (PI * x[0]).sin() * (PI * x[1]).cos(),
                (PI * PI + 2.0 * 1.5 * 1.5) * (PI * x[0]).cos() * (PI * x[1]).sin(),
            ])
            .add_impedance_physical_from_marker(&attrs, &[0, 1, 0, 1], 2.0, 1.0, |x, n| {
                let e = [0.0, (PI * x[0]).cos() * (PI * x[1]).sin()];
                e[0] * n[1] - e[1] * n[0]
            })
            .add_pec_zero_from_marker(&attrs, &[1, 0, 1, 0])
            .build()
            .solve();
        assert!(solved_b.solve_result.converged);

        // Case C: anisotropic + physical absorbing + partial PEC
        let solved_c = StaticMaxwellBuilder::new(HCurlSpace::new(SimplexMesh::<2>::unit_square_tri(6), 1))
            .with_quad_order(4)
            .with_anisotropic_diag(1.0, 1.7, 2.3)
            .with_source_fn(|x| [
                (PI * PI + 1.7) * (PI * x[1]).sin(),
                (PI * PI + 2.3) * (PI * x[0]).sin(),
            ])
            .add_absorbing_physical_from_marker(&attrs, &[0, 1, 1, 0], 1.7, 1.0, |x, n| {
                let e = [(PI * x[1]).sin(), (PI * x[0]).sin()];
                e[0] * n[1] - e[1] * n[0]
            })
            .add_pec_zero_from_marker(&attrs, &[1, 0, 0, 1])
            .build()
            .solve();
        assert!(solved_c.solve_result.converged);

        let norm_b = solved_b.solution.iter().map(|v| v * v).sum::<f64>().sqrt();
        let norm_c = solved_c.solution.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm_b.is_finite());
        assert!(norm_c.is_finite());
    }

    #[test]
    fn free_hcurl_dofs_from_marker_matches_manual_filter() {
        let mesh = SimplexMesh::<2>::unit_square_tri(6);
        let space = HCurlSpace::new(mesh, 1);

        let attrs = [1, 2, 3, 4];
        let marker = [1, 0, 1, 0];

        let free_helper = free_hcurl_dofs_from_marker(&space, &attrs, &marker);

        let tags = marker_to_tags(&attrs, &marker);
        let bnd: HashSet<u32> = boundary_dofs_hcurl(space.mesh(), &space, &tags)
            .into_iter()
            .collect();
        let free_manual: Vec<usize> = (0..space.n_dofs() as u32)
            .filter(|d| !bnd.contains(d))
            .map(|d| d as usize)
            .collect();

        assert_eq!(free_helper, free_manual);
    }

    #[test]
    fn free_h1_dofs_from_marker_matches_manual_filter() {
        let mesh = SimplexMesh::<2>::unit_square_tri(6);
        let h1 = H1Space::new(mesh, 1);

        let attrs = [1, 2, 3, 4];
        let marker = [0, 1, 0, 1];

        let free_helper = free_h1_dofs_from_marker(&h1, &attrs, &marker);

        let tags = marker_to_tags(&attrs, &marker);
        let bnd: HashSet<u32> = boundary_dofs(h1.mesh(), h1.dof_manager(), &tags)
            .into_iter()
            .collect();
        let free_manual: Vec<usize> = (0..h1.n_dofs() as u32)
            .filter(|d| !bnd.contains(d))
            .map(|d| d as usize)
            .collect();

        assert_eq!(free_helper, free_manual);
    }

    #[test]
    #[should_panic(expected = "marker size mismatch")]
    fn marker_to_tags_rejects_length_mismatch() {
        let attrs = [1, 2, 3, 4];
        let marker = [1, 0, 1];
        let _ = marker_to_tags(&attrs, &marker);
    }

    #[test]
    fn assemble_hcurl_eigen_system_from_marker_matches_manual_pipeline() {
        let n = 6;
        let hcurl_mesh = SimplexMesh::<2>::unit_square_tri(n);
        let hcurl = HCurlSpace::new(hcurl_mesh, 1);
        let h1_mesh = SimplexMesh::<2>::unit_square_tri(n);
        let h1 = H1Space::new(h1_mesh, 1);

        let attrs = [1, 2, 3, 4];
        let ess_bdr = [1, 1, 1, 1];

        let helper = assemble_hcurl_eigen_system_from_marker(
            &h1, &hcurl, &attrs, &ess_bdr, 1.0, 1.0, 4,
        );

        let k_full = VectorAssembler::assemble_bilinear(&hcurl, &[&CurlCurlIntegrator { mu: 1.0 }], 4);
        let m_full =
            VectorAssembler::assemble_bilinear(&hcurl, &[&VectorMassIntegrator { alpha: 1.0 }], 4);
        let subspace = build_hcurl_constraint_subspace_from_marker(&h1, &hcurl, &attrs, &ess_bdr);
        let k_manual = extract_square_submatrix(&k_full, &subspace.hcurl_free_dofs);
        let m_manual = extract_square_submatrix(&m_full, &subspace.hcurl_free_dofs);

        assert_eq!(helper.hcurl_free_dofs, subspace.hcurl_free_dofs);
        assert_eq!(helper.h1_free_dofs, subspace.h1_free_dofs);
        assert_eq!(helper.constraints.shape(), subspace.gradient_constraints.shape());

        let c_diff = (&helper.constraints - &subspace.gradient_constraints).norm();
        assert!(c_diff < 1e-13, "constraint matrix mismatch = {c_diff}");

        assert_eq!(helper.stiffness_free.nrows, k_manual.nrows);
        assert_eq!(helper.stiffness_free.ncols, k_manual.ncols);
        assert_eq!(helper.stiffness_free.row_ptr, k_manual.row_ptr);
        assert_eq!(helper.stiffness_free.col_idx, k_manual.col_idx);
        let k_diff = helper
            .stiffness_free
            .values
            .iter()
            .zip(k_manual.values.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(k_diff < 1e-13, "stiffness mismatch = {k_diff}");

        assert_eq!(helper.mass_free.nrows, m_manual.nrows);
        assert_eq!(helper.mass_free.ncols, m_manual.ncols);
        assert_eq!(helper.mass_free.row_ptr, m_manual.row_ptr);
        assert_eq!(helper.mass_free.col_idx, m_manual.col_idx);
        let m_diff = helper
            .mass_free
            .values
            .iter()
            .zip(m_manual.values.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(m_diff < 1e-13, "mass mismatch = {m_diff}");
    }

    #[test]
    fn hcurl_matrix_free_apply_enforces_pec_and_linearity() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let hcurl = HCurlSpace::new(mesh, 1);

        let mu = 1.0;
        let alpha = 0.9;
        let quad = 4;
        let pec_tags = [1, 2, 3, 4];

        let op = HcurlMatrixFreeOperator2D::new(&hcurl, mu, alpha, quad, &pec_tags);
        let pec_u32 = boundary_dofs_hcurl(hcurl.mesh(), &hcurl, &pec_tags);

        let mut x1 = vec![0.0_f64; hcurl.n_dofs()];
        let mut x2 = vec![0.0_f64; hcurl.n_dofs()];
        for (i, xi) in x1.iter_mut().enumerate() {
            *xi = ((i as f64) * 0.173).sin();
        }
        for (i, xi) in x2.iter_mut().enumerate() {
            *xi = ((i as f64) * 0.311).cos();
        }

        let mut y1 = vec![0.0_f64; x1.len()];
        op.apply(&x1, &mut y1);

        for &d in &pec_u32 {
            let di = d as usize;
            assert!((y1[di] - x1[di]).abs() < 1e-12, "PEC row not identity at dof {di}");
        }

        let mut x12 = vec![0.0_f64; x1.len()];
        for i in 0..x12.len() {
            x12[i] = x1[i] + x2[i];
        }

        let mut y2 = vec![0.0_f64; x2.len()];
        let mut y12 = vec![0.0_f64; x12.len()];
        op.apply(&x2, &mut y2);
        op.apply(&x12, &mut y12);

        let mut y_sum = vec![0.0_f64; y1.len()];
        for i in 0..y_sum.len() {
            y_sum[i] = y1[i] + y2[i];
        }

        let diff = y12
            .iter()
            .zip(y_sum.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f64>()
            .sqrt();
        assert!(diff < 1e-9, "matrix-free linearity mismatch: {diff}");
    }

    #[test]
    fn hcurl_matrix_free_solve_matches_assembled_solve() {
        let mesh = SimplexMesh::<2>::unit_square_tri(10);
        let hcurl = HCurlSpace::new(mesh, 1);

        let mu = 1.0;
        let alpha = 1.0;
        let quad = 4;
        let pec_tags = [1, 2, 3, 4];

        let source = FnVectorSourceIntegrator {
            f: &|x: &[f64]| [(PI * x[1]).sin(), (PI * x[0]).sin()],
        };
        let rhs = VectorAssembler::assemble_linear(&hcurl, &[&source], quad);

        let mut assembled = VectorAssembler::assemble_bilinear(
            &hcurl,
            &[
                &CurlCurlIntegrator { mu },
                &VectorMassIntegrator { alpha },
            ],
            quad,
        );
        let pec_u32 = boundary_dofs_hcurl(hcurl.mesh(), &hcurl, &pec_tags);
        let zero_vals = vec![0.0_f64; pec_u32.len()];
        let mut rhs_asm = rhs.clone();
        apply_dirichlet(&mut assembled, &mut rhs_asm, &pec_u32, &zero_vals);

        let (u_asm, res_asm) = solve_hcurl_jacobi(&assembled, &rhs_asm);
        assert!(res_asm.converged, "assembled solve did not converge");

        let op = HcurlMatrixFreeOperator2D::new(&hcurl, mu, alpha, quad, &pec_tags);
        let cfg = SolverConfig {
            rtol: 1e-8,
            atol: 1e-12,
            max_iter: 20_000,
            verbose: false,
            ..SolverConfig::default()
        };
        let (u_mf, res_mf) = solve_hcurl_matrix_free(&op, &rhs, &cfg)
            .expect("matrix-free solve failed");
        assert!(res_mf.converged, "matrix-free solve did not converge");

        let diff = u_asm
            .iter()
            .zip(u_mf.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f64>()
            .sqrt();
        assert!(diff < 2e-7, "matrix-free solve mismatch: {diff}");
    }

    #[test]
    fn hcurl_matrix_free_large_dof_apply_smoke() {
        // Large-DOF smoke for matrix-free apply path used by scalable Krylov solves.
        let n = 40;
        let mesh = SimplexMesh::<2>::unit_square_tri(n);
        let hcurl = HCurlSpace::new(mesh, 1);
        let op = HcurlMatrixFreeOperator2D::new(&hcurl, 1.0, 1.0, 4, &[1, 2, 3, 4]);

        assert!(op.n_dofs() > 3000, "expected large DOF matrix-free problem, got {}", op.n_dofs());

        let mut x = vec![0.0_f64; op.n_dofs()];
        for (i, xi) in x.iter_mut().enumerate() {
            *xi = ((i as f64) * 0.013).sin();
        }
        let mut y = vec![0.0_f64; op.n_dofs()];
        op.apply(&x, &mut y);

        assert!(y.iter().all(|v| v.is_finite()), "matrix-free large-DOF apply produced non-finite values");
    }

    #[test]
    fn hcurl_eigen_amg_preconditioned_lobpcg_smoke() {
        let n = 10;
        let hcurl_mesh = SimplexMesh::<2>::unit_square_tri(n);
        let hcurl = HCurlSpace::new(hcurl_mesh, 1);
        let h1_mesh = SimplexMesh::<2>::unit_square_tri(n);
        let h1 = H1Space::new(h1_mesh, 1);

        let attrs = [1, 2, 3, 4];
        let marker = [1, 1, 1, 1];
        let eig_system = assemble_hcurl_eigen_system_from_marker(
            &h1,
            &hcurl,
            &attrs,
            &marker,
            1.0,
            1.0,
            4,
        );

        let eig_cfg = LobpcgConfig {
            max_iter: 800,
            tol: 1e-7,
            verbose: false,
        };
        let inner_cfg = SolverConfig {
            rtol: 1e-2,
            atol: 1e-12,
            max_iter: 20,
            verbose: false,
            ..SolverConfig::default()
        };

        let res = solve_hcurl_eigen_preconditioned_amg(
            &eig_system,
            3,
            &eig_cfg,
            AmgConfig::default(),
            &inner_cfg,
        ).expect("amg-preconditioned lobpcg failed");

        assert!(res.converged, "preconditioned LOBPCG did not converge");
        assert_eq!(res.eigenvalues.len(), 3);
        assert!(res.eigenvalues[0] > 9.0 && res.eigenvalues[0] < 10.5);
        assert!(res.eigenvalues[1] > 9.0 && res.eigenvalues[1] < 10.5);
        assert!(res.eigenvalues[2] > 19.0 && res.eigenvalues[2] < 20.5);

        // P3 large-scale smoke target: ensure we are solving at non-trivial reduced size.
        assert!(eig_system.hcurl_free_dofs.len() > 200, "reduced problem too small for scaling smoke");
    }

    #[test]
    fn builder_anisotropic_pec_zero_source_has_zero_solution() {
        // Anisotropic diagonal material + full PEC BC + zero source must yield u=0.
        let mesh = SimplexMesh::<2>::unit_square_tri(6);
        let space = HCurlSpace::new(mesh, 1);

        let solved = StaticMaxwellBuilder::new(space)
            .with_quad_order(4)
            .with_anisotropic_diag(1.0, 2.0, 3.0)
            .add_pec_zero(&[1, 2, 3, 4])
            .build()
            .solve();

        assert!(solved.solve_result.converged);
        assert!(solved.boundary_report.essential_dofs > 0);
        let norm_u = solved.solution.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm_u < 1e-12, "anisotropic PEC solution norm = {norm_u}");
    }

    #[test]
    fn builder_anisotropic_diag_matches_anisotropic_matrix_fn_diagonal() {
        // with_anisotropic_diag(mu, sx, sy) must equal
        // with_anisotropic_matrix_fn(mu, |_| [sx, 0, 0, sy]) for any source + BC.
        let sx = 1.5_f64;
        let sy = 2.5_f64;

        let source = |x: &[f64]| {
            [
                (PI * x[1]).sin() + 0.2 * (PI * x[0]).cos(),
                (PI * x[0]).sin() - 0.3 * (PI * x[1]).cos(),
            ]
        };
        let robin_data = |x: &[f64], n: &[f64]| {
            let e = [0.4 * (PI * x[1]).sin(), -0.25 * (PI * x[0]).sin()];
            1.5 * (e[0] * n[1] - e[1] * n[0]) + 0.1 * (x[0] + x[1])
        };

        let mesh_a = SimplexMesh::<2>::unit_square_tri(6);
        let space_a = HCurlSpace::new(mesh_a, 1);
        let mesh_b = SimplexMesh::<2>::unit_square_tri(6);
        let space_b = HCurlSpace::new(mesh_b, 1);

        let solved_diag = StaticMaxwellBuilder::new(space_a)
            .with_quad_order(4)
            .with_anisotropic_diag(1.0, sx, sy)
            .with_source_fn(source)
            .add_tangential_robin(&[2, 4], 1.5, robin_data)
            .add_pec_zero(&[1, 3])
            .build()
            .solve();

        let solved_fn = StaticMaxwellBuilder::new(space_b)
            .with_quad_order(4)
            .with_anisotropic_matrix_fn(1.0, move |_x| [sx, 0.0, 0.0, sy])
            .with_source_fn(source)
            .add_tangential_robin(&[2, 4], 1.5, robin_data)
            .add_pec_zero(&[1, 3])
            .build()
            .solve();

        assert!(solved_diag.solve_result.converged);
        assert!(solved_fn.solve_result.converged);
        assert_eq!(solved_diag.solution.len(), solved_fn.solution.len());

        let diff = solved_diag
            .solution
            .iter()
            .zip(solved_fn.solution.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f64>()
            .sqrt();
        assert!(diff < 1e-11, "anisotropic diag/matrix-fn mismatch = {diff}");
    }

    #[test]
    fn builder_frequency_isotropic_with_impedance_is_consistent() {
        // Frequency-domain material (omega, epsilon) combined with impedance BC.
        // Must converge and give the same result as directly setting alpha = epsilon * omega^2.
        let omega = 1.8_f64;
        let epsilon = 0.9_f64;
        let alpha = epsilon * omega * omega;

        let source = move |x: &[f64]| {
            [
                (PI * PI + alpha) * (PI * x[1]).sin(),
                (PI * PI + alpha) * (PI * x[0]).sin(),
            ]
        };
        let impedance = move |x: &[f64], n: &[f64]| {
            let e = [(PI * x[1]).sin(), (PI * x[0]).sin()];
            2.0 * (e[0] * n[1] - e[1] * n[0])
        };

        let mesh_a = SimplexMesh::<2>::unit_square_tri(6);
        let space_a = HCurlSpace::new(mesh_a, 1);
        let mesh_b = SimplexMesh::<2>::unit_square_tri(6);
        let space_b = HCurlSpace::new(mesh_b, 1);

        let solved_freq = StaticMaxwellBuilder::new(space_a)
            .with_quad_order(4)
            .with_frequency_isotropic(1.0, epsilon, omega)
            .with_source_fn(source)
            .add_impedance_from_marker(&[1, 2, 3, 4], &[1, 1, 1, 1], 2.0, impedance)
            .build()
            .solve();

        let solved_alpha = StaticMaxwellBuilder::new(space_b)
            .with_quad_order(4)
            .with_isotropic_coeffs(1.0, alpha)
            .with_source_fn(source)
            .add_impedance_from_marker(&[1, 2, 3, 4], &[1, 1, 1, 1], 2.0, impedance)
            .build()
            .solve();

        assert!(solved_freq.solve_result.converged, "frequency+impedance not converged");
        assert!(solved_alpha.solve_result.converged, "direct-alpha+impedance not converged");
        assert_eq!(solved_freq.solution.len(), solved_alpha.solution.len());

        let diff = solved_freq
            .solution
            .iter()
            .zip(solved_alpha.solution.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f64>()
            .sqrt();
        assert!(diff < 1e-11, "frequency+impedance / direct-alpha+impedance mismatch = {diff}");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // h-refinement convergence tests
    // ──────────────────────────────────────────────────────────────────────────

    /// Manufactured solution for anisotropic convergence test:
    ///   E(x,y) = (sin(πy), sin(πx))
    /// Material: σ = diag(2, 3), μ = 1.
    /// Source = curl curl E + σ E, computed analytically.
    fn aniso_exact(x: &[f64]) -> [f64; 2] {
        [(PI * x[1]).sin(), (PI * x[0]).sin()]
    }

    fn aniso_source(x: &[f64]) -> [f64; 2] {
        // curl curl E = (PI² sin(πy), PI² sin(πx))
        // σ E = (2 sin(πy), 3 sin(πx))
        [
            (PI * PI + 2.0) * (PI * x[1]).sin(),
            (PI * PI + 3.0) * (PI * x[0]).sin(),
        ]
    }

    fn solve_aniso_pec(n: usize) -> f64 {
        let mesh = SimplexMesh::<2>::unit_square_tri(n);
        let space = HCurlSpace::new(mesh, 1);
        let solved = StaticMaxwellBuilder::new(space)
            .with_quad_order(5)
            .with_anisotropic_diag(1.0, 2.0, 3.0)
            .with_source_fn(aniso_source)
            .add_pec_zero(&[1, 2, 3, 4])
            .build()
            .solve();
        assert!(solved.solve_result.converged, "not converged n={n}");
        l2_error_hcurl_exact(&solved.space, &solved.solution, aniso_exact)
    }

    #[test]
    fn anisotropic_pec_h_refinement_converges_order1() {
        // ND1 on triangles should give O(h) in L2(H(curl)) norm.
        // We check that halving h reduces error by at least factor 1.5
        // (theoretical rate ~2.0, this is a conservative lower bound).
        let err_coarse = solve_aniso_pec(8);
        let err_fine   = solve_aniso_pec(16);
        let rate = (err_coarse / err_fine).ln() / 2.0_f64.ln();
        assert!(
            rate > 0.8,
            "anisotropic PEC convergence rate = {rate:.3} (expected > 0.8), \
             err_coarse={err_coarse:.3e}, err_fine={err_fine:.3e}"
        );
    }

    /// Manufactured solution for anisotropic + Robin convergence test:
    ///   E(x,y) = (cos(πx) sin(πy), 0)
    /// Material: σ = diag(1.5, 2.5), μ = 1.
    /// Full Robin (γ=1) on all boundaries.
    fn aniso_robin_exact(x: &[f64]) -> [f64; 2] {
        [(PI * x[0]).cos() * (PI * x[1]).sin(), 0.0]
    }

    fn aniso_robin_source(x: &[f64]) -> [f64; 2] {
        // curl E = ∂Ey/∂x - ∂Ex/∂y = 0 - π cos(πx)cos(πy) = -π cos(πx)cos(πy)
        // curl curl E = ∇⊥(curl E) = (∂(curl E)/∂y, -∂(curl E)/∂x)
        //   = (π² cos(πx)sin(πy), -π² sin(πx)cos(πy))
        // σ E = (1.5 cos(πx)sin(πy), 0)
        [
            (PI * PI + 1.5) * (PI * x[0]).cos() * (PI * x[1]).sin(),
            -(PI * PI) * (PI * x[0]).sin() * (PI * x[1]).cos(),
        ]
    }

    fn aniso_robin_bc(x: &[f64], normal: &[f64]) -> f64 {
        // g = -curl_E + γ * (Ex*ny - Ey*nx)
        // curl_E = ∂Ey/∂x - ∂Ex/∂y = 0 - π cos(πx)cos(πy) = -π cos(πx)cos(πy)
        // -curl_E = +π cos(πx)cos(πy)
        let neg_curl_e = PI * (PI * x[0]).cos() * (PI * x[1]).cos();
        let e = aniso_robin_exact(x);
        let ntrace = e[0] * normal[1] - e[1] * normal[0];
        neg_curl_e + 1.0 * ntrace
    }

    fn solve_aniso_robin(n: usize) -> f64 {
        let mesh = SimplexMesh::<2>::unit_square_tri(n);
        let space = HCurlSpace::new(mesh, 1);
        let solved = StaticMaxwellBuilder::new(space)
            .with_quad_order(5)
            .with_anisotropic_diag(1.0, 1.5, 2.5)
            .with_source_fn(aniso_robin_source)
            .add_tangential_robin(&[1, 2, 3, 4], 1.0, aniso_robin_bc)
            .build()
            .solve();
        assert!(solved.solve_result.converged, "not converged n={n}");
        l2_error_hcurl_exact(&solved.space, &solved.solution, aniso_robin_exact)
    }

    #[test]
    fn anisotropic_robin_h_refinement_converges_order1() {
        let err_coarse = solve_aniso_robin(8);
        let err_fine   = solve_aniso_robin(16);
        let rate = (err_coarse / err_fine).ln() / 2.0_f64.ln();
        assert!(
            rate > 0.8,
            "anisotropic Robin convergence rate = {rate:.3} (expected > 0.8), \
             err_coarse={err_coarse:.3e}, err_fine={err_fine:.3e}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// First-order E-B Maxwell time-domain operator
// ─────────────────────────────────────────────────────────────────────────────

/// First-order full-wave Maxwell system in H(curl) × L2 (2-D TM mode).
///
/// The coupled system is:
/// ```text
///   ε M_E ∂E/∂t = (1/μ) Cᵀ B − σ M_E E + f
///       M_B ∂B/∂t = −C E
/// ```
/// where E ∈ H(curl) (ND1 edges, 2-D) and B ∈ L2 (P0 scalars, 2-D).
///
/// * `m_e`    = ε M_E — H(curl) mass matrix scaled by ε, with PEC BC applied.
/// * `m_b`    = M_B  — L2/P0 scalar mass matrix.
/// * `curl_c` = C    : H(curl) → L2 (rows = triangles, cols = edges).
/// * `curl_ct`= Cᵀ  : L2 → H(curl).
///
/// **Leapfrog** (staggered Euler, symplectic):
/// ```text
///   B^{n+½} = B^{n-½} − dt · M_B⁻¹ · C · E^n
///   E^{n+1} = E^n     + dt · M_E⁻¹ · [(1/μ) Cᵀ B^{n+½} − (σ/ε) m_e E^n + f]
/// ```
/// Total energy (ε/2)||E||²_{M_E} + (1/2μ)||B||²_{M_B} is conserved for σ=0, f=0.
pub struct FirstOrderMaxwellOp {
    /// ε M_E — H(curl) mass matrix scaled by ε (PEC rows/cols zeroed, diagonal=1).
    pub m_e: CsrMatrix<f64>,
    /// Diagonal of M_B: mb_diag[i] = area of triangle i (L2/P0 is exactly diagonal).
    pub mb_diag: Vec<f64>,
    /// Discrete curl C : H(curl)→L2 (rows=triangles, cols=edges).
    pub curl_c:  CsrMatrix<f64>,
    /// Transpose Cᵀ : L2→H(curl).
    pub curl_ct: CsrMatrix<f64>,
    /// PEC (tangential) boundary DOF indices in H(curl).
    pub pec_dofs: Vec<usize>,
    pub eps:   f64,
    pub mu:    f64,
    pub sigma: f64,
    /// Number of H(curl) DOFs.
    pub n_e: usize,
    /// Number of L2 DOFs.
    pub n_b: usize,
}

impl FirstOrderMaxwellOp {
    /// Assemble operators on the unit square [0,1]² triangulated with an
    /// `n×n` uniform triangular mesh. Boundary tags 1‥4 impose PEC (n×E=0).
    pub fn new_unit_square(n: usize, eps: f64, mu: f64, sigma: f64) -> Self {
        let mesh  = SimplexMesh::<2>::unit_square_tri(n);
        let hcurl = HCurlSpace::new(mesh.clone(), 1);
        let l2    = L2Space::new(mesh, 0);

        // ε M_E
        let mut m_e = VectorAssembler::assemble_bilinear(
            &hcurl, &[&VectorMassIntegrator { alpha: eps }], 4,
        );
        // Discrete curl C : H(curl)→L2
        let curl_c  = DiscreteLinearOperator::curl_2d(&hcurl, &l2)
            .expect("curl_2d assembly failed");
        let curl_ct = curl_c.transpose();

        // PEC BC: identify all tangential boundary DOFs
        let pec_u32: Vec<u32> = boundary_dofs_hcurl(hcurl.mesh(), &hcurl, &[1, 2, 3, 4]);
        let pec_dofs: Vec<usize> = pec_u32.iter().map(|&d| d as usize).collect();

        // Apply PEC to m_e (zero boundary rows/cols, diagonal=1)
        let zero_vals: Vec<f64> = vec![0.0; pec_u32.len()];
        let mut dummy_rhs = vec![0.0_f64; hcurl.n_dofs()];
        apply_dirichlet(&mut m_e, &mut dummy_rhs, &pec_u32, &zero_vals);

        // L2/P0 mass is exactly diagonal: mb_diag[j] = area(T_j).
        let mb_diag: Vec<f64> = {
            let mesh_ref = l2.mesh();
            let mut areas = vec![0.0_f64; l2.n_dofs()];
            for e in mesh_ref.elem_iter() {
                let nodes = mesh_ref.element_nodes(e);
                let dofs  = l2.element_dofs(e);
                let a = mesh_ref.node_coords(nodes[0]);
                let b = mesh_ref.node_coords(nodes[1]);
                let c = mesh_ref.node_coords(nodes[2]);
                let area = 0.5
                    * ((b[0] - a[0]) * (c[1] - a[1])
                      - (c[0] - a[0]) * (b[1] - a[1]))
                    .abs();
                areas[dofs[0] as usize] = area;
            }
            areas
        };

        let n_e = hcurl.n_dofs();
        let n_b = l2.n_dofs();
        Self { m_e, mb_diag, curl_c, curl_ct, pec_dofs, eps, mu, sigma, n_e, n_b }
    }

    /// Half-step the B field:
    /// `B^{n+½} = B^{n-½} − dt · M_B⁻¹ · C · E^n`
    ///
    /// Since M_B is diagonal (P0 mass = area per element), this is a simple
    /// element-wise scaling — no linear solve required.
    pub fn b_half_step(
        &self,
        dt: f64,
        e:  &[f64],
        b:  &mut Vec<f64>,
    ) {
        let mut ce = vec![0.0_f64; self.n_b];
        self.curl_c.spmv(e, &mut ce);               // ce = C e
        // b^{n+½} = b^{n-½} − dt · diag(area)⁻¹ · C · e
        for i in 0..self.n_b {
            b[i] -= dt * ce[i] / self.mb_diag[i];
        }
    }

    /// Full-step the E field using B^{n+½}:
    /// `E^{n+1} = E^n + dt · M_E⁻¹ · [(1/μ) Cᵀ B^{n+½} − (σ/ε) m_e E^n + force]`
    ///
    /// `force` is the raw HCurl load vector (e.g. `∫ J·φ`); pass zeros for no source.
    /// PEC BC is enforced on `E^{n+1}` automatically.
    pub fn e_full_step(
        &self,
        dt:     f64,
        e_in:   &[f64],
        b_half: &[f64],
        force:  &[f64],
        cfg:    &SolverConfig,
    ) -> Vec<f64> {
        // rhs = (1/μ) Cᵀ b^{n+½}
        let mut rhs = vec![0.0_f64; self.n_e];
        self.curl_ct.spmv_add(1.0 / self.mu, b_half, 0.0, &mut rhs);

        // rhs -= (σ/ε) · m_e · e  =  σ M_E e
        if self.sigma != 0.0 {
            let mut me_e = vec![0.0_f64; self.n_e];
            self.m_e.spmv(e_in, &mut me_e);
            let coeff = self.sigma / self.eps;
            for i in 0..self.n_e { rhs[i] -= coeff * me_e[i]; }
        }

        // rhs += force
        for i in 0..self.n_e { rhs[i] += force[i]; }

        // Scale by dt: solve m_e δe = dt · rhs
        for v in &mut rhs { *v *= dt; }
        for &d in &self.pec_dofs { rhs[d] = 0.0; }   // enforce PEC on RHS

        let mut delta_e = vec![0.0_f64; self.n_e];
        solve_pcg_jacobi(&self.m_e, &rhs, &mut delta_e, cfg)
            .expect("M_E solve failed in e_full_step");

        // e_out = e_in + δe  (with PEC BC enforced)
        let mut e_out = e_in.to_vec();
        for i in 0..self.n_e { e_out[i] += delta_e[i]; }
        for &d in &self.pec_dofs { e_out[d] = 0.0; }
        e_out
    }

    /// Total electromagnetic energy:
    /// `(ε/2) ||E||²_{M_E}  +  (1/2μ) ||B||²_{M_B}`
    ///
    /// Discrete form: `(1/2) eᵀ m_e e  +  (1/2μ) Σ_j area_j b_j²`
    /// where `m_e = ε M_E`.
    pub fn compute_energy(&self, e: &[f64], b: &[f64]) -> f64 {
        let mut me_e = vec![0.0_f64; self.n_e];
        self.m_e.spmv(e, &mut me_e);
        let e_energy: f64 = 0.5 * e.iter().zip(&me_e).map(|(ei, mi)| ei * mi).sum::<f64>();

        // b^T M_B b = Σ_j area_j · b_j²  (diagonal M_B = diag(areas))
        let b_energy: f64 = 0.5 / self.mu
            * b.iter().enumerate().map(|(i, &bi)| self.mb_diag[i] * bi * bi).sum::<f64>();

        e_energy + b_energy
    }
}

/// Minimal 3-D first-order Maxwell assembly skeleton.
///
/// This struct provides ND1↔RT0 discrete curl plus basic 3-D first-order
/// E/B single-step updates with mass-matrix solves.
pub struct FirstOrderMaxwell3DSkeleton {
    /// ε M_E in H(curl).
    pub m_e: CsrMatrix<f64>,
    /// M_B in H(div).
    pub m_b: CsrMatrix<f64>,
    /// Discrete curl C : H(curl, ND1) -> H(div, RT0).
    pub curl_c: CsrMatrix<f64>,
    /// Transpose C^T : H(div, RT0) -> H(curl, ND1).
    pub curl_ct: CsrMatrix<f64>,
    pub eps: f64,
    pub mu: f64,
    pub sigma: f64,
    /// Effective PEC boundary tags after normalization (valid + unique + sorted).
    pub pec_tags: Vec<i32>,
    /// Effective absorbing-boundary tags after normalization (valid + unique + sorted).
    pub abc_tags: Vec<i32>,
    /// Absorbing boundary damping coefficient used by tangential mass term.
    pub abc_gamma: f64,
    /// Optional tangential boundary damping matrix assembled on `abc_tags`.
    pub m_e_abc: Option<CsrMatrix<f64>>,
    /// Effective impedance-boundary tags after normalization (valid + unique + sorted).
    pub impedance_tags: Vec<i32>,
    /// Impedance boundary damping coefficient used by tangential mass term.
    pub impedance_gamma: f64,
    /// Optional tangential impedance matrix assembled on `impedance_tags`.
    pub m_e_impedance: Option<CsrMatrix<f64>>,
    /// Total number of distinct boundary tags on the mesh.
    pub n_boundary_tags: usize,
    /// PEC-constrained H(curl) boundary edge DOFs.
    pub pec_dofs: Vec<usize>,
    pub n_e: usize,
    pub n_b: usize,
}

/// High-level mixed-operator view for H(curl)-H(div) couplings in 3-D.
pub struct HCurlHDivMixedOperators3D {
    pub curl: CsrMatrix<f64>,
    pub curl_t: CsrMatrix<f64>,
    pub n_hcurl: usize,
    pub n_hdiv: usize,
}

impl HCurlHDivMixedOperators3D {
    pub fn apply_hcurl_to_hdiv(&self, e: &[f64], y: &mut [f64]) {
        assert_eq!(e.len(), self.n_hcurl, "apply_hcurl_to_hdiv: input size mismatch");
        assert_eq!(y.len(), self.n_hdiv, "apply_hcurl_to_hdiv: output size mismatch");
        self.curl.spmv(e, y);
    }

    pub fn apply_hdiv_to_hcurl(&self, b: &[f64], x: &mut [f64]) {
        assert_eq!(b.len(), self.n_hdiv, "apply_hdiv_to_hcurl: input size mismatch");
        assert_eq!(x.len(), self.n_hcurl, "apply_hdiv_to_hcurl: output size mismatch");
        self.curl_t.spmv(b, x);
    }
}

/// Time stepping strategy for 3-D first-order Maxwell skeleton.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FirstOrderTimeStepper3D {
    /// Explicit `B` half-step + explicit `E` full-step.
    Explicit,
    /// Explicit `B` half-step + Crank-Nicolson `E` full-step.
    CrankNicolson,
}

/// Per-step configuration for 3-D first-order Maxwell time stepping.
#[derive(Clone, Copy, Debug)]
pub struct FirstOrderStepConfig3D {
    pub dt: f64,
    pub strategy: FirstOrderTimeStepper3D,
}

impl FirstOrderStepConfig3D {
    pub fn explicit(dt: f64) -> Self {
        Self {
            dt,
            strategy: FirstOrderTimeStepper3D::Explicit,
        }
    }

    pub fn crank_nicolson(dt: f64) -> Self {
        Self {
            dt,
            strategy: FirstOrderTimeStepper3D::CrankNicolson,
        }
    }
}

pub type FirstOrderForceFn3D = dyn Fn(f64, &mut [f64]) + Send + Sync;

pub enum FirstOrderForceModel3D {
    Static(Vec<f64>),
    TimeDependent(Box<FirstOrderForceFn3D>),
}

/// Stateful 3-D first-order Maxwell solver wrapper.
///
/// This object owns `(E, B)` state and advances it with either explicit or
/// Crank-Nicolson strategy using an underlying `FirstOrderMaxwell3DSkeleton`.
pub struct FirstOrderMaxwellSolver3D {
    pub op: FirstOrderMaxwell3DSkeleton,
    pub e: Vec<f64>,
    pub b: Vec<f64>,
    pub force: Vec<f64>,
    pub force_model: FirstOrderForceModel3D,
    pub cfg: SolverConfig,
    pub step_cfg: FirstOrderStepConfig3D,
    pub time: f64,
}

impl FirstOrderMaxwellSolver3D {
    pub fn new(
        op: FirstOrderMaxwell3DSkeleton,
        cfg: SolverConfig,
        step_cfg: FirstOrderStepConfig3D,
    ) -> Self {
        let e = vec![0.0_f64; op.n_e];
        let b = vec![0.0_f64; op.n_b];
        let force = vec![0.0_f64; op.n_e];
        let force_model = FirstOrderForceModel3D::Static(force.clone());
        Self {
            op,
            e,
            b,
            force,
            force_model,
            cfg,
            step_cfg,
            time: 0.0,
        }
    }

    pub fn with_state(mut self, e: &[f64], b: &[f64]) -> Self {
        assert_eq!(e.len(), self.op.n_e, "with_state: E length mismatch");
        assert_eq!(b.len(), self.op.n_b, "with_state: B length mismatch");
        self.e.copy_from_slice(e);
        self.b.copy_from_slice(b);
        for &d in &self.op.pec_dofs {
            self.e[d] = 0.0;
        }
        self
    }

    pub fn set_force(&mut self, force: &[f64]) {
        assert_eq!(force.len(), self.op.n_e, "set_force: force length mismatch");
        self.force.copy_from_slice(force);
        self.force_model = FirstOrderForceModel3D::Static(self.force.clone());
    }

    pub fn set_time_dependent_force<F>(&mut self, force_fn: F)
    where
        F: Fn(f64, &mut [f64]) + Send + Sync + 'static,
    {
        self.force_model = FirstOrderForceModel3D::TimeDependent(Box::new(force_fn));
    }

    pub fn clear_force(&mut self) {
        self.force.fill(0.0);
        self.force_model = FirstOrderForceModel3D::Static(self.force.clone());
    }

    fn refresh_force_at_time(&mut self, t: f64) {
        match &self.force_model {
            FirstOrderForceModel3D::Static(f) => {
                self.force.copy_from_slice(f);
            }
            FirstOrderForceModel3D::TimeDependent(force_fn) => {
                self.force.fill(0.0);
                force_fn(t, &mut self.force);
            }
        }
    }

    pub fn set_step_config(&mut self, step_cfg: FirstOrderStepConfig3D) {
        self.step_cfg = step_cfg;
    }

    pub fn advance_one(&mut self) {
        self.refresh_force_at_time(self.time);
        self.op.step_with_config(
            self.step_cfg,
            &mut self.e,
            &mut self.b,
            &self.force,
            &self.cfg,
        );
        self.time += self.step_cfg.dt;
    }

    pub fn advance_with_config(&mut self, step_cfg: FirstOrderStepConfig3D) {
        self.refresh_force_at_time(self.time);
        self.op.step_with_config(step_cfg, &mut self.e, &mut self.b, &self.force, &self.cfg);
        self.time += step_cfg.dt;
    }

    pub fn advance_n(&mut self, n_steps: usize) {
        for _ in 0..n_steps {
            self.advance_one();
        }
    }

    pub fn energy(&self) -> f64 {
        self.op.compute_energy(&self.e, &self.b)
    }
}

impl FirstOrderMaxwell3DSkeleton {
    fn assemble_with_mesh_and_pec_tags(
        mesh3: SimplexMesh<3>,
        eps: f64,
        mu: f64,
        sigma: f64,
        pec_tags: &[i32],
        abc_tags: &[i32],
        abc_gamma: f64,
        impedance_tags: &[i32],
        impedance_gamma: f64,
    ) -> Self {
        let hcurl = HCurlSpace::new(mesh3.clone(), 1);
        let hdiv  = HDivSpace::new(mesh3.clone(), 0);

        let mut m_e = VectorAssembler::assemble_bilinear(
            &hcurl,
            &[&VectorMassIntegrator { alpha: eps }],
            4,
        );
        let m_b = VectorAssembler::assemble_bilinear(
            &hdiv,
            &[&VectorMassIntegrator { alpha: 1.0 }],
            4,
        );

        let curl_c = DiscreteLinearOperator::curl_3d(&hcurl, &hdiv)
            .expect("curl_3d ND1->RT0 assembly failed");
        let curl_ct = curl_c.transpose();

        // Normalize incoming PEC tags to keep behavior deterministic and robust:
        // only real boundary tags are kept, duplicates are removed, and order is sorted.
        let mut boundary_tags_set: HashSet<i32> = HashSet::new();
        for f in 0..mesh3.n_boundary_faces() as u32 {
            boundary_tags_set.insert(mesh3.face_tag(f));
        }
        let n_boundary_tags = boundary_tags_set.len();
        let mut normalized_tags: Vec<i32> = pec_tags
            .iter()
            .copied()
            .filter(|t| boundary_tags_set.contains(t))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        normalized_tags.sort_unstable();

        let mut normalized_abc_tags: Vec<i32> = abc_tags
            .iter()
            .copied()
            .filter(|t| boundary_tags_set.contains(t))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        normalized_abc_tags.sort_unstable();

        let m_e_abc = if abc_gamma != 0.0 && !normalized_abc_tags.is_empty() {
            Some(VectorBoundaryAssembler::assemble_boundary_bilinear(
                &hcurl,
                &[&TangentialMassIntegrator { gamma: abc_gamma }],
                &normalized_abc_tags,
                4,
            ))
        } else {
            None
        };

        let mut normalized_impedance_tags: Vec<i32> = impedance_tags
            .iter()
            .copied()
            .filter(|t| boundary_tags_set.contains(t))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        normalized_impedance_tags.sort_unstable();

        let m_e_impedance = if impedance_gamma != 0.0 && !normalized_impedance_tags.is_empty() {
            Some(VectorBoundaryAssembler::assemble_boundary_bilinear(
                &hcurl,
                &[&TangentialMassIntegrator {
                    gamma: impedance_gamma,
                }],
                &normalized_impedance_tags,
                4,
            ))
        } else {
            None
        };

        let mut pec_u32: Vec<u32> = boundary_dofs_hcurl(hcurl.mesh(), &hcurl, &normalized_tags);
        pec_u32.sort_unstable();
        pec_u32.dedup();
        let pec_dofs: Vec<usize> = pec_u32.iter().map(|&d| d as usize).collect();

        let zero_vals = vec![0.0_f64; pec_u32.len()];
        let mut dummy_rhs = vec![0.0_f64; hcurl.n_dofs()];
        apply_dirichlet(&mut m_e, &mut dummy_rhs, &pec_u32, &zero_vals);

        Self {
            m_e,
            m_b,
            eps,
            mu,
            sigma,
            pec_tags: normalized_tags,
            abc_tags: normalized_abc_tags,
            abc_gamma,
            m_e_abc,
            impedance_tags: normalized_impedance_tags,
            impedance_gamma,
            m_e_impedance,
            n_boundary_tags,
            pec_dofs,
            n_e: hcurl.n_dofs(),
            n_b: hdiv.n_dofs(),
            curl_c,
            curl_ct,
        }
    }

    /// Assemble ND1/RT0 spaces with default parameters `eps=1, mu=1, sigma=0`.
    pub fn new_unit_cube(n: usize) -> Self {
        Self::new_unit_cube_with_params(n, 1.0, 1.0, 0.0)
    }

    /// Assemble ND1/RT0 spaces with no PEC constraints.
    pub fn new_unit_cube_without_pec(n: usize, eps: f64, mu: f64, sigma: f64) -> Self {
        Self::new_unit_cube_with_pec_abc_and_impedance_tags(n, eps, mu, sigma, &[], &[], 0.0, &[], 0.0)
    }

    /// Assemble ND1/RT0 spaces with a single PEC boundary tag.
    pub fn new_unit_cube_with_pec_tag(n: usize, eps: f64, mu: f64, sigma: f64, pec_tag: i32) -> Self {
        Self::new_unit_cube_with_pec_abc_and_impedance_tags(
            n,
            eps,
            mu,
            sigma,
            &[pec_tag],
            &[],
            0.0,
            &[],
            0.0,
        )
    }

    /// Assemble ND1/RT0 spaces, mass matrices, and curl operators on unit cube tet mesh.
    pub fn new_unit_cube_with_params(n: usize, eps: f64, mu: f64, sigma: f64) -> Self {
        let mesh3 = SimplexMesh::<3>::unit_cube_tet(n);

        // Use all boundary tags as PEC by default for the 3-D skeleton.
        let mut tags_set: HashSet<i32> = HashSet::new();
        for f in 0..mesh3.n_boundary_faces() as u32 {
            tags_set.insert(mesh3.face_tag(f));
        }
        let mut tags: Vec<i32> = tags_set.into_iter().collect();
        tags.sort_unstable();

        Self::assemble_with_mesh_and_pec_tags(mesh3, eps, mu, sigma, &tags, &[], 0.0, &[], 0.0)
    }

    /// Assemble ND1/RT0 spaces with configurable PEC boundary tags.
    pub fn new_unit_cube_with_pec_tags(
        n: usize,
        eps: f64,
        mu: f64,
        sigma: f64,
        pec_tags: &[i32],
    ) -> Self {
        Self::new_unit_cube_with_pec_abc_and_impedance_tags(n, eps, mu, sigma, pec_tags, &[], 0.0, &[], 0.0)
    }

    /// Assemble ND1/RT0 spaces with configurable PEC and absorbing boundary tags.
    pub fn new_unit_cube_with_pec_and_abc_tags(
        n: usize,
        eps: f64,
        mu: f64,
        sigma: f64,
        pec_tags: &[i32],
        abc_tags: &[i32],
        abc_gamma: f64,
    ) -> Self {
        Self::new_unit_cube_with_pec_abc_and_impedance_tags(
            n,
            eps,
            mu,
            sigma,
            pec_tags,
            abc_tags,
            abc_gamma,
            &[],
            0.0,
        )
    }

    /// Assemble ND1/RT0 spaces with configurable PEC, ABC and impedance tags.
    pub fn new_unit_cube_with_pec_abc_and_impedance_tags(
        n: usize,
        eps: f64,
        mu: f64,
        sigma: f64,
        pec_tags: &[i32],
        abc_tags: &[i32],
        abc_gamma: f64,
        impedance_tags: &[i32],
        impedance_gamma: f64,
    ) -> Self {
        let mesh3 = SimplexMesh::<3>::unit_cube_tet(n);
        Self::assemble_with_mesh_and_pec_tags(
            mesh3,
            eps,
            mu,
            sigma,
            pec_tags,
            abc_tags,
            abc_gamma,
            impedance_tags,
            impedance_gamma,
        )
    }

    /// Assemble with physical `(epsilon, mu)` based ABC/impedance damping.
    pub fn new_unit_cube_with_physical_abc_and_impedance_tags(
        n: usize,
        eps: f64,
        mu: f64,
        sigma: f64,
        pec_tags: &[i32],
        abc_tags: &[i32],
        abc_scale: f64,
        impedance_tags: &[i32],
        impedance_scale: f64,
    ) -> Self {
        let gamma = boundary_admittance(eps, mu);
        Self::new_unit_cube_with_pec_abc_and_impedance_tags(
            n,
            eps,
            mu,
            sigma,
            pec_tags,
            abc_tags,
            abc_scale * gamma,
            impedance_tags,
            impedance_scale * gamma,
        )
    }

    /// Export high-level H(curl)-H(div) mixed operators for external workflows.
    pub fn mixed_operators(&self) -> HCurlHDivMixedOperators3D {
        HCurlHDivMixedOperators3D {
            curl: self.curl_c.clone(),
            curl_t: self.curl_ct.clone(),
            n_hcurl: self.n_e,
            n_hdiv: self.n_b,
        }
    }

    /// Apply `y = C * e` where `C: H(curl)->H(div)`.
    pub fn apply_curl(&self, e: &[f64], y: &mut [f64]) {
        assert_eq!(e.len(), self.n_e, "apply_curl: input size mismatch");
        assert_eq!(y.len(), self.n_b, "apply_curl: output size mismatch");
        self.curl_c.spmv(e, y);
    }

    /// Apply `x = C^T * b` where `C^T: H(div)->H(curl)`.
    pub fn apply_curl_t(&self, b: &[f64], x: &mut [f64]) {
        assert_eq!(b.len(), self.n_b, "apply_curl_t: input size mismatch");
        assert_eq!(x.len(), self.n_e, "apply_curl_t: output size mismatch");
        self.curl_ct.spmv(b, x);
    }

    /// Whether PEC constraints are active on this operator.
    pub fn has_pec_constraints(&self) -> bool {
        !self.pec_dofs.is_empty()
    }

    /// Number of constrained H(curl) DOFs from PEC boundary conditions.
    pub fn n_pec_dofs(&self) -> usize {
        self.pec_dofs.len()
    }

    /// Whether any PEC boundary tags are active on this operator.
    pub fn has_pec_tags(&self) -> bool {
        !self.pec_tags.is_empty()
    }

    /// Number of effective PEC boundary tags after normalization.
    pub fn n_pec_tags(&self) -> usize {
        self.pec_tags.len()
    }

    /// Returns true if the given boundary tag is active in the normalized PEC tag set.
    pub fn is_pec_tag(&self, tag: i32) -> bool {
        self.pec_tags.binary_search(&tag).is_ok()
    }

    /// Returns true if the given boundary tag is not active in the normalized PEC tag set.
    pub fn is_free_tag(&self, tag: i32) -> bool {
        if tag < 0 {
            return false;
        }
        !self.is_pec_tag(tag)
    }

    /// Returns true if the given H(curl) dof is constrained by PEC.
    pub fn is_pec_dof(&self, dof: usize) -> bool {
        if dof >= self.n_e {
            return false;
        }
        self.pec_dofs.binary_search(&dof).is_ok()
    }

    /// Returns true if the given H(curl) dof is free (not constrained by PEC).
    pub fn is_free_dof(&self, dof: usize) -> bool {
        if dof >= self.n_e {
            return false;
        }
        !self.is_pec_dof(dof)
    }

    /// Fraction of H(curl) dofs constrained by PEC, in [0, 1].
    pub fn pec_dof_fraction(&self) -> f64 {
        if self.n_e == 0 {
            return 0.0;
        }
        self.n_pec_dofs() as f64 / self.n_e as f64
    }

    /// Number of unconstrained (free) H(curl) dofs.
    pub fn n_free_dofs(&self) -> usize {
        self.n_e.saturating_sub(self.n_pec_dofs())
    }

    /// Fraction of unconstrained (free) H(curl) dofs, in [0, 1].
    pub fn free_dof_fraction(&self) -> f64 {
        if self.n_e == 0 {
            return 0.0;
        }
        self.n_free_dofs() as f64 / self.n_e as f64
    }

    /// Fraction of boundary tags activated as PEC, in [0, 1].
    pub fn pec_tag_fraction(&self) -> f64 {
        if self.n_boundary_tags == 0 {
            return 0.0;
        }
        self.n_pec_tags() as f64 / self.n_boundary_tags as f64
    }

    /// Number of boundary tags that are not activated as PEC.
    pub fn n_free_tags(&self) -> usize {
        self.n_boundary_tags.saturating_sub(self.n_pec_tags())
    }

    /// Fraction of boundary tags that are not activated as PEC, in [0, 1].
    pub fn free_tag_fraction(&self) -> f64 {
        if self.n_boundary_tags == 0 {
            return 0.0;
        }
        self.n_free_tags() as f64 / self.n_boundary_tags as f64
    }

    /// 3-D B half step:
    /// `M_B (B^{n+1/2} - B^{n-1/2})/dt = - C E^n`.
    pub fn b_half_step(
        &self,
        dt: f64,
        e: &[f64],
        b: &mut Vec<f64>,
        cfg: &SolverConfig,
    ) {
        assert_eq!(e.len(), self.n_e);
        assert_eq!(b.len(), self.n_b);

        let mut rhs = vec![0.0_f64; self.n_b];
        self.curl_c.spmv(e, &mut rhs);
        for v in &mut rhs { *v = -*v; }

        let mut delta_b = vec![0.0_f64; self.n_b];
        solve_pcg_jacobi(&self.m_b, &rhs, &mut delta_b, cfg)
            .expect("3D b_half_step: M_B solve failed");
        for i in 0..self.n_b { b[i] += dt * delta_b[i]; }
    }

    /// 3-D E full step:
    /// `ε M_E (E^{n+1}-E^n)/dt = (1/μ) C^T B^{n+1/2} - σ M_E E^n + f`.
    pub fn e_full_step(
        &self,
        dt: f64,
        e_in: &[f64],
        b_half: &[f64],
        force: &[f64],
        cfg: &SolverConfig,
    ) -> Vec<f64> {
        assert_eq!(e_in.len(), self.n_e);
        assert_eq!(b_half.len(), self.n_b);
        assert_eq!(force.len(), self.n_e);

        let mut rhs = vec![0.0_f64; self.n_e];
        self.curl_ct.spmv_add(1.0 / self.mu, b_half, 0.0, &mut rhs);

        if self.sigma != 0.0 {
            let mut me_e = vec![0.0_f64; self.n_e];
            self.m_e.spmv(e_in, &mut me_e);
            let coeff = self.sigma / self.eps;
            for i in 0..self.n_e { rhs[i] -= coeff * me_e[i]; }
        }

        // Optional first-order absorbing boundary contribution:
        //   rhs -= M_abc * E^n, where M_abc = γ * ∫_Γ (n×u)·(n×v) dS.
        if let Some(m_abc) = &self.m_e_abc {
            let mut mabc_e = vec![0.0_f64; self.n_e];
            m_abc.spmv(e_in, &mut mabc_e);
            for i in 0..self.n_e { rhs[i] -= mabc_e[i]; }
        }

        if let Some(m_imp) = &self.m_e_impedance {
            let mut mimp_e = vec![0.0_f64; self.n_e];
            m_imp.spmv(e_in, &mut mimp_e);
            for i in 0..self.n_e { rhs[i] -= mimp_e[i]; }
        }

        for i in 0..self.n_e { rhs[i] += force[i]; }
        for v in &mut rhs { *v *= dt; }
        for &d in &self.pec_dofs { rhs[d] = 0.0; }

        let mut delta_e = vec![0.0_f64; self.n_e];
        solve_pcg_jacobi(&self.m_e, &rhs, &mut delta_e, cfg)
            .expect("3D e_full_step: M_E solve failed");

        let mut e_out = e_in.to_vec();
        for i in 0..self.n_e { e_out[i] += delta_e[i]; }
        for &d in &self.pec_dofs { e_out[d] = 0.0; }
        e_out
    }

    /// 3-D E full step with Crank-Nicolson treatment of damping terms.
    ///
    /// Uses
    /// `eps M_E (E^{n+1}-E^n)/dt = (1/mu) C^T B^{n+1/2} + f - D * (E^{n+1}+E^n)/2`,
    /// where `D = (sigma/eps) * (eps M_E) + M_abc + M_impedance`.
    pub fn e_full_step_crank_nicolson(
        &self,
        dt: f64,
        e_in: &[f64],
        b_half: &[f64],
        force: &[f64],
        cfg: &SolverConfig,
    ) -> Vec<f64> {
        assert_eq!(e_in.len(), self.n_e);
        assert_eq!(b_half.len(), self.n_b);
        assert_eq!(force.len(), self.n_e);

        let mut rhs = vec![0.0_f64; self.n_e];
        self.curl_ct.spmv_add(1.0 / self.mu, b_half, 0.0, &mut rhs);
        for i in 0..self.n_e {
            rhs[i] += force[i];
        }

        // Build D * e_in and LHS = m_e + 0.5*dt*D.
        let mut d_e = vec![0.0_f64; self.n_e];
        let mut lhs = self.m_e.clone();

        if self.sigma != 0.0 {
            let coeff = self.sigma / self.eps;
            let mut me_e = vec![0.0_f64; self.n_e];
            self.m_e.spmv(e_in, &mut me_e);
            for i in 0..self.n_e {
                d_e[i] += coeff * me_e[i];
            }
            lhs = csr_add_scaled(&lhs, &self.m_e, 0.5 * dt * coeff);
        }
        if let Some(m_abc) = &self.m_e_abc {
            let mut tmp = vec![0.0_f64; self.n_e];
            m_abc.spmv(e_in, &mut tmp);
            for i in 0..self.n_e {
                d_e[i] += tmp[i];
            }
            lhs = csr_add_scaled(&lhs, m_abc, 0.5 * dt);
        }
        if let Some(m_imp) = &self.m_e_impedance {
            let mut tmp = vec![0.0_f64; self.n_e];
            m_imp.spmv(e_in, &mut tmp);
            for i in 0..self.n_e {
                d_e[i] += tmp[i];
            }
            lhs = csr_add_scaled(&lhs, m_imp, 0.5 * dt);
        }

        let mut me_e_in = vec![0.0_f64; self.n_e];
        self.m_e.spmv(e_in, &mut me_e_in);
        for i in 0..self.n_e {
            rhs[i] = me_e_in[i] + dt * rhs[i] - 0.5 * dt * d_e[i];
        }
        for &d in &self.pec_dofs {
            rhs[d] = 0.0;
        }

        let mut e_out = vec![0.0_f64; self.n_e];
        solve_pcg_jacobi(&lhs, &rhs, &mut e_out, cfg)
            .expect("3D e_full_step_crank_nicolson: linear solve failed");
        for &d in &self.pec_dofs {
            e_out[d] = 0.0;
        }
        e_out
    }

    /// Advance one 3-D first-order step using explicit `B` half-step and
    /// Crank-Nicolson `E` full-step.
    pub fn step_crank_nicolson(
        &self,
        dt: f64,
        e: &mut Vec<f64>,
        b: &mut Vec<f64>,
        force: &[f64],
        cfg: &SolverConfig,
    ) {
        self.b_half_step(dt, e, b, cfg);
        *e = self.e_full_step_crank_nicolson(dt, e, b, force, cfg);
    }

    /// Advance one 3-D first-order step using explicit `B` half-step and
    /// explicit `E` full-step.
    pub fn step_explicit(
        &self,
        dt: f64,
        e: &mut Vec<f64>,
        b: &mut Vec<f64>,
        force: &[f64],
        cfg: &SolverConfig,
    ) {
        self.b_half_step(dt, e, b, cfg);
        *e = self.e_full_step(dt, e, b, force, cfg);
    }

    /// Advance one 3-D first-order step with a selectable time stepping strategy.
    pub fn step(
        &self,
        dt: f64,
        e: &mut Vec<f64>,
        b: &mut Vec<f64>,
        force: &[f64],
        cfg: &SolverConfig,
        strategy: FirstOrderTimeStepper3D,
    ) {
        match strategy {
            FirstOrderTimeStepper3D::Explicit => {
                self.step_explicit(dt, e, b, force, cfg);
            }
            FirstOrderTimeStepper3D::CrankNicolson => {
                self.step_crank_nicolson(dt, e, b, force, cfg);
            }
        }
    }

    /// Advance one step using packed step configuration.
    pub fn step_with_config(
        &self,
        cfg_step: FirstOrderStepConfig3D,
        e: &mut Vec<f64>,
        b: &mut Vec<f64>,
        force: &[f64],
        cfg: &SolverConfig,
    ) {
        self.step(cfg_step.dt, e, b, force, cfg, cfg_step.strategy);
    }

    /// Discrete 3-D electromagnetic energy:
    /// `(1/2) e^T (eps M_E) e + (1/2mu) b^T M_B b`.
    pub fn compute_energy(&self, e: &[f64], b: &[f64]) -> f64 {
        assert_eq!(e.len(), self.n_e);
        assert_eq!(b.len(), self.n_b);

        let mut me_e = vec![0.0_f64; self.n_e];
        self.m_e.spmv(e, &mut me_e);
        let e_energy = 0.5 * e.iter().zip(&me_e).map(|(ei, mi)| ei * mi).sum::<f64>();

        let mut mb_b = vec![0.0_f64; self.n_b];
        self.m_b.spmv(b, &mut mb_b);
        let b_energy = 0.5 / self.mu * b.iter().zip(&mb_b).map(|(bi, mi)| bi * mi).sum::<f64>();

        e_energy + b_energy
    }
}

#[cfg(test)]
mod first_order_tests {
    use super::*;

    /// Verify that the staggered leapfrog conserves electromagnetic energy
    /// (σ=0, J=0, PEC BC on all walls) to within 2 % over 100 steps.
    ///
    /// Initial condition: E=0 on interior DOFs, B=0 initially; then the
    /// first b_half_step triggers energy exchange through the curl coupling.
    /// We seed E with nonzero interior DOFs so there is something to evolve.
    #[test]
    fn first_order_maxwell_energy_conserved() {
        let op  = FirstOrderMaxwellOp::new_unit_square(8, 1.0, 1.0, 0.0);
        let cfg = SolverConfig {
            rtol: 1e-10, atol: 0.0, max_iter: 600, verbose: false,
            ..SolverConfig::default()
        };

        // Seed E with 1.0 on all interior (non-PEC) DOFs.
        let mut e = vec![1.0_f64; op.n_e];
        for &d in &op.pec_dofs { e[d] = 0.0; }
        // B starts at zero; after the first half-step B will get energy from curl E.
        let mut b     = vec![0.0_f64; op.n_b];
        let force     = vec![0.0_f64; op.n_e];
        let dt        = 0.01_f64;   // safe below CFL ( h≈0.09 for 8×8 mesh )

        // Compute reference energy (E=seeded, B=0 → purely electric energy)
        let e_ref = op.compute_energy(&e, &b);
        assert!(e_ref > 0.0, "initial energy must be positive");

        // 100 leapfrog steps  (t = 0..1.0)
        for _ in 0..100 {
            op.b_half_step(dt, &e, &mut b);
            e = op.e_full_step(dt, &e, &b, &force, &cfg);
        }

        let e_final = op.compute_energy(&e, &b);
        let rel_err = (e_final - e_ref).abs() / e_ref;
        assert!(
            rel_err < 0.02,
            "Energy conservation violated: E_ref={e_ref:.4e}, E_final={e_final:.4e}, \
             rel_err={rel_err:.4} (expected < 0.02)"
        );
    }

    #[test]
    fn first_order_maxwell_with_sigma_dissipates_energy() {
        let op  = FirstOrderMaxwellOp::new_unit_square(8, 1.0, 1.0, 0.2);
        let cfg = SolverConfig {
            rtol: 1e-10, atol: 0.0, max_iter: 600, verbose: false,
            ..SolverConfig::default()
        };

        let mut e = vec![1.0_f64; op.n_e];
        for &d in &op.pec_dofs { e[d] = 0.0; }
        let mut b = vec![0.0_f64; op.n_b];
        let force = vec![0.0_f64; op.n_e];
        let dt = 0.01_f64;

        let e_init = op.compute_energy(&e, &b);
        assert!(e_init > 0.0, "initial energy must be positive");

        for _ in 0..200 {
            op.b_half_step(dt, &e, &mut b);
            e = op.e_full_step(dt, &e, &b, &force, &cfg);
        }

        let e_final = op.compute_energy(&e, &b);
        assert!(
            e_final < 0.9 * e_init,
            "Damped energy should decay: E_init={e_init:.4e}, E_final={e_final:.4e}"
        );
    }

    #[test]
    fn first_order_maxwell_mode_period_recovery() {
        let n = 8;
        let op  = FirstOrderMaxwellOp::new_unit_square(n, 1.0, 1.0, 0.0);
        let cfg = SolverConfig {
            rtol: 1e-10, atol: 0.0, max_iter: 600, verbose: false,
            ..SolverConfig::default()
        };

        // Manufactured cavity mode used by ex_maxwell_firstorder:
        // E(x,t) = sin(pi t) * (sin(pi y), sin(pi x))
        // B(x,t) = cos(pi t) * (cos(pi x) - cos(pi y))
        // So at t=0: E=0, B=B0; at t=1: E=0, B=-B0.
        let mut e = vec![0.0_f64; op.n_e];
        let mesh = SimplexMesh::<2>::unit_square_tri(n);
        let l2 = L2Space::new(mesh.clone(), 0);
        let mut b0 = vec![0.0_f64; op.n_b];
        for elem in l2.mesh().elem_iter() {
            let nodes = l2.mesh().element_nodes(elem);
            let dof = l2.element_dofs(elem)[0] as usize;
            let a = l2.mesh().node_coords(nodes[0]);
            let b = l2.mesh().node_coords(nodes[1]);
            let c = l2.mesh().node_coords(nodes[2]);
            let cx = (a[0] + b[0] + c[0]) / 3.0;
            let cy = (a[1] + b[1] + c[1]) / 3.0;
            b0[dof] = (std::f64::consts::PI * cx).cos() - (std::f64::consts::PI * cy).cos();
        }

        let mut b = b0.clone();
        let force = vec![0.0_f64; op.n_e];
        let dt = 0.01;
        let n_steps = 100; // t_end = 1.0

        for _ in 0..n_steps {
            op.b_half_step(dt, &e, &mut b);
            e = op.e_full_step(dt, &e, &b, &force, &cfg);
        }

        let mut e_me = vec![0.0_f64; op.n_e];
        op.m_e.spmv(&e, &mut e_me);
        let e_l2 = e.iter().zip(&e_me).map(|(ei, mei)| ei * mei).sum::<f64>().abs().sqrt();

        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for i in 0..op.n_b {
            let bi_exact = -b0[i];
            let diff = b[i] - bi_exact;
            num += op.mb_diag[i] * diff * diff;
            den += op.mb_diag[i] * bi_exact * bi_exact;
        }
        let rel_b = num.sqrt() / den.sqrt().max(1e-30);

        assert!(
            e_l2 < 6e-2,
            "E should return close to zero at one period: ||E||_M={e_l2:.3e}"
        );
        assert!(
            rel_b < 2e-2,
            "B one-period recovery too large: rel_B={rel_b:.3e}"
        );
    }

    #[test]
    fn first_order_3d_curl_nd1_rt0_operator_is_well_formed() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube(1);

        assert_eq!(skel.curl_c.nrows, skel.n_b);
        assert_eq!(skel.curl_c.ncols, skel.n_e);
        assert!(skel.curl_c.nnz() > 0, "curl_3d matrix should not be empty");

        let x = vec![1.0_f64; skel.n_e];
        let mut y = vec![0.0_f64; skel.n_b];
        skel.curl_c.spmv(&x, &mut y);
        let y_norm = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(y_norm > 0.0, "curl_3d action should be non-trivial on test vector");
    }

    #[test]
    fn first_order_3d_curl_transpose_shapes_match() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube(1);
        assert_eq!(skel.curl_ct.nrows, skel.n_e);
        assert_eq!(skel.curl_ct.ncols, skel.n_b);
        assert_eq!(skel.curl_ct.nnz(), skel.curl_c.nnz());
    }

    #[test]
    fn first_order_3d_curl_adjoint_identity_holds() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube(1);

        // Deterministic nontrivial vectors.
        let e: Vec<f64> = (0..skel.n_e).map(|i| (i as f64 + 1.0) / skel.n_e as f64).collect();
        let b: Vec<f64> = (0..skel.n_b).map(|i| (2.0 * i as f64 + 1.0) / skel.n_b as f64).collect();

        let mut ce = vec![0.0_f64; skel.n_b];
        skel.apply_curl(&e, &mut ce);
        let lhs = b.iter().zip(&ce).map(|(bi, cei)| bi * cei).sum::<f64>();

        let mut ctb = vec![0.0_f64; skel.n_e];
        skel.apply_curl_t(&b, &mut ctb);
        let rhs = e.iter().zip(&ctb).map(|(ei, ctbi)| ei * ctbi).sum::<f64>();

        let denom = lhs.abs().max(rhs.abs()).max(1.0);
        let rel = (lhs - rhs).abs() / denom;
        assert!(
            rel < 1e-12,
            "Adjoint identity mismatch: lhs={lhs:.6e}, rhs={rhs:.6e}, rel={rel:.3e}"
        );
    }

    #[test]
    fn first_order_3d_single_step_smoke() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_params(1, 1.0, 1.0, 0.05);
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 600,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e = vec![0.0_f64; skel.n_e];
        for (i, v) in e.iter_mut().enumerate() {
            *v = (i as f64 + 1.0) / skel.n_e as f64;
        }
        let mut b = vec![0.0_f64; skel.n_b];
        let force = vec![0.0_f64; skel.n_e];

        skel.b_half_step(0.01, &e, &mut b, &cfg);
        let e_new = skel.e_full_step(0.01, &e, &b, &force, &cfg);

        assert_eq!(e_new.len(), skel.n_e);
        assert_eq!(b.len(), skel.n_b);
        let e_norm = e_new.iter().map(|v| v * v).sum::<f64>().sqrt();
        let b_norm = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(e_norm.is_finite() && b_norm.is_finite(), "3D step produced non-finite values");
        assert!(e_norm > 0.0 || b_norm > 0.0, "3D step should produce non-trivial state");
    }

    #[test]
    fn first_order_3d_large_dof_single_step_smoke() {
        // Large-DOF smoke for first-order transient Maxwell scalability path.
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_params(3, 1.0, 1.0, 0.02);
        let cfg = SolverConfig {
            rtol: 1e-9,
            atol: 0.0,
            max_iter: 1200,
            verbose: false,
            ..SolverConfig::default()
        };

        assert!(skel.n_e > 250, "expected large H(curl) DOF count, got {}", skel.n_e);
        assert!(skel.n_b > 120, "expected large H(div) DOF count, got {}", skel.n_b);

        let mut e = vec![0.0_f64; skel.n_e];
        for (i, v) in e.iter_mut().enumerate() {
            *v = (i as f64 + 1.0) / skel.n_e as f64;
        }
        let mut b = vec![0.0_f64; skel.n_b];
        let force = vec![0.0_f64; skel.n_e];

        skel.b_half_step(0.005, &e, &mut b, &cfg);
        let e_new = skel.e_full_step(0.005, &e, &b, &force, &cfg);

        assert!(e_new.iter().all(|v| v.is_finite()), "large-DOF E update produced non-finite values");
        assert!(b.iter().all(|v| v.is_finite()), "large-DOF B update produced non-finite values");
    }

    #[test]
    fn first_order_3d_sigma_dissipates_energy() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_params(1, 1.0, 1.0, 0.1);
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e = vec![0.0_f64; skel.n_e];
        for (i, v) in e.iter_mut().enumerate() {
            *v = (i as f64 + 1.0) / skel.n_e as f64;
        }
        let mut b = vec![0.0_f64; skel.n_b];
        let force = vec![0.0_f64; skel.n_e];
        let dt = 0.01;

        let e0 = skel.compute_energy(&e, &b);
        assert!(e0 > 0.0, "initial 3D energy must be positive");

        for _ in 0..120 {
            skel.b_half_step(dt, &e, &mut b, &cfg);
            e = skel.e_full_step(dt, &e, &b, &force, &cfg);
        }

        let e1 = skel.compute_energy(&e, &b);
        assert!(
            e1 < 0.95 * e0,
            "3D damped energy should decay: E0={e0:.6e}, E1={e1:.6e}"
        );
    }

    #[test]
    fn first_order_3d_pec_dofs_are_zero_after_step() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_params(1, 1.0, 1.0, 0.0);
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        assert!(!skel.pec_dofs.is_empty(), "expected non-empty 3D PEC boundary dofs");

        let mut e = vec![1.0_f64; skel.n_e];
        let mut b = vec![0.0_f64; skel.n_b];
        let force = vec![0.0_f64; skel.n_e];

        skel.b_half_step(0.01, &e, &mut b, &cfg);
        e = skel.e_full_step(0.01, &e, &b, &force, &cfg);

        for &d in &skel.pec_dofs {
            assert_eq!(e[d], 0.0, "PEC DOF {d} should be clamped to zero");
        }
    }

    #[test]
    fn first_order_3d_energy_conserved_sigma0() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_params(1, 1.0, 1.0, 0.0);
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e = vec![0.0_f64; skel.n_e];
        for (i, v) in e.iter_mut().enumerate() {
            *v = (i as f64 + 1.0) / skel.n_e as f64;
        }
        for &d in &skel.pec_dofs {
            e[d] = 0.0;
        }

        let mut b = vec![0.0_f64; skel.n_b];
        let force = vec![0.0_f64; skel.n_e];
        let dt = 0.01;

        let e0 = skel.compute_energy(&e, &b);
        assert!(e0 > 0.0, "initial 3D energy must be positive");

        for _ in 0..100 {
            skel.b_half_step(dt, &e, &mut b, &cfg);
            e = skel.e_full_step(dt, &e, &b, &force, &cfg);
        }

        let e1 = skel.compute_energy(&e, &b);
        let rel = (e1 - e0).abs() / e0;
        assert!(
            rel < 0.05,
            "3D sigma=0 energy drift too large: E0={e0:.6e}, E1={e1:.6e}, rel={rel:.3e}"
        );
    }

    #[test]
    fn first_order_3d_empty_pec_tags_do_not_clamp_all_boundary_dofs() {
        let skel_all = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_params(1, 1.0, 1.0, 0.0);
        let skel_free =
            FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_tags(1, 1.0, 1.0, 0.0, &[]);
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        assert!(
            !skel_all.pec_dofs.is_empty(),
            "default 3D constructor should produce non-empty PEC dofs"
        );
        assert!(
            skel_free.pec_dofs.is_empty(),
            "empty PEC tag list should produce zero constrained dofs"
        );

        let mut e_free = vec![1.0_f64; skel_free.n_e];
        let mut b_free = vec![0.0_f64; skel_free.n_b];
        let force_free = vec![0.0_f64; skel_free.n_e];

        skel_free.b_half_step(0.01, &e_free, &mut b_free, &cfg);
        e_free = skel_free.e_full_step(0.01, &e_free, &b_free, &force_free, &cfg);

        let boundary_kept_nonzero = skel_all
            .pec_dofs
            .iter()
            .any(|&d| e_free[d].abs() > 1e-12);
        assert!(
            boundary_kept_nonzero,
            "with empty PEC tags, boundary dofs should not be forcibly clamped"
        );
    }

    #[test]
    fn first_order_3d_subset_pec_tags_clamp_only_selected_boundary_dofs() {
        let mesh3 = SimplexMesh::<3>::unit_cube_tet(1);
        let hcurl = HCurlSpace::new(mesh3.clone(), 1);

        let mut tags_set: HashSet<i32> = HashSet::new();
        for f in 0..mesh3.n_boundary_faces() as u32 {
            tags_set.insert(mesh3.face_tag(f));
        }
        let mut tags: Vec<i32> = tags_set.into_iter().collect();
        tags.sort_unstable();
        assert!(tags.len() >= 2, "unit cube should have at least two boundary tags");

        let selected = vec![tags[0]];
        let mut expected_sel: Vec<usize> = boundary_dofs_hcurl(hcurl.mesh(), &hcurl, &selected)
            .into_iter()
            .map(|d| d as usize)
            .collect();
        expected_sel.sort_unstable();
        expected_sel.dedup();

        let mut rest_tags = tags.clone();
        rest_tags.remove(0);
        let mut expected_rest: Vec<usize> =
            boundary_dofs_hcurl(hcurl.mesh(), &hcurl, &rest_tags)
                .into_iter()
                .map(|d| d as usize)
                .collect();
        expected_rest.sort_unstable();
        expected_rest.dedup();

        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_tags(
            1,
            1.0,
            1.0,
            0.0,
            &selected,
        );
        let mut got = skel.pec_dofs.clone();
        got.sort_unstable();
        got.dedup();
        assert_eq!(got, expected_sel, "selected PEC tags should map to expected dof set");

        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e = vec![1.0_f64; skel.n_e];
        let mut b = vec![0.0_f64; skel.n_b];
        let force = vec![0.0_f64; skel.n_e];
        skel.b_half_step(0.01, &e, &mut b, &cfg);
        e = skel.e_full_step(0.01, &e, &b, &force, &cfg);

        for &d in &expected_sel {
            assert_eq!(e[d], 0.0, "selected-tag PEC DOF {d} should be clamped");
        }
        let non_selected_kept = expected_rest.iter().any(|&d| e[d].abs() > 1e-12);
        assert!(
            non_selected_kept,
            "non-selected boundary dofs should not all be clamped"
        );
    }

    #[test]
    fn first_order_3d_default_constructor_matches_explicit_all_tags() {
        let mesh3 = SimplexMesh::<3>::unit_cube_tet(1);
        let mut tags_set: HashSet<i32> = HashSet::new();
        for f in 0..mesh3.n_boundary_faces() as u32 {
            tags_set.insert(mesh3.face_tag(f));
        }
        let mut tags: Vec<i32> = tags_set.into_iter().collect();
        tags.sort_unstable();

        let skel_default = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_params(1, 1.0, 1.0, 0.0);
        let skel_explicit =
            FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_tags(1, 1.0, 1.0, 0.0, &tags);

        let mut d0 = skel_default.pec_dofs.clone();
        let mut d1 = skel_explicit.pec_dofs.clone();
        d0.sort_unstable();
        d1.sort_unstable();
        d0.dedup();
        d1.dedup();
        assert_eq!(d0, d1, "default constructor should match explicit all-tags PEC dofs");

        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e0 = vec![1.0_f64; skel_default.n_e];
        let mut b0 = vec![0.0_f64; skel_default.n_b];
        let force0 = vec![0.0_f64; skel_default.n_e];

        let mut e1 = vec![1.0_f64; skel_explicit.n_e];
        let mut b1 = vec![0.0_f64; skel_explicit.n_b];
        let force1 = vec![0.0_f64; skel_explicit.n_e];

        skel_default.b_half_step(0.01, &e0, &mut b0, &cfg);
        e0 = skel_default.e_full_step(0.01, &e0, &b0, &force0, &cfg);

        skel_explicit.b_half_step(0.01, &e1, &mut b1, &cfg);
        e1 = skel_explicit.e_full_step(0.01, &e1, &b1, &force1, &cfg);

        let diff_e = e0
            .iter()
            .zip(&e1)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let diff_b = b0
            .iter()
            .zip(&b1)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(diff_e < 1e-12, "E mismatch between default/all-tags constructors: {diff_e:.3e}");
        assert!(diff_b < 1e-12, "B mismatch between default/all-tags constructors: {diff_b:.3e}");
    }

    #[test]
    fn first_order_3d_new_unit_cube_matches_default_params_constructor() {
        let skel_a = FirstOrderMaxwell3DSkeleton::new_unit_cube(1);
        let skel_b = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_params(1, 1.0, 1.0, 0.0);

        let mut da = skel_a.pec_dofs.clone();
        let mut db = skel_b.pec_dofs.clone();
        da.sort_unstable();
        db.sort_unstable();
        da.dedup();
        db.dedup();
        assert_eq!(da, db, "new_unit_cube should match default-params constructor PEC dofs");

        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e0 = vec![1.0_f64; skel_a.n_e];
        let mut b0 = vec![0.0_f64; skel_a.n_b];
        let force0 = vec![0.0_f64; skel_a.n_e];

        let mut e1 = vec![1.0_f64; skel_b.n_e];
        let mut b1 = vec![0.0_f64; skel_b.n_b];
        let force1 = vec![0.0_f64; skel_b.n_e];

        skel_a.b_half_step(0.01, &e0, &mut b0, &cfg);
        e0 = skel_a.e_full_step(0.01, &e0, &b0, &force0, &cfg);

        skel_b.b_half_step(0.01, &e1, &mut b1, &cfg);
        e1 = skel_b.e_full_step(0.01, &e1, &b1, &force1, &cfg);

        let diff_e = e0
            .iter()
            .zip(&e1)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let diff_b = b0
            .iter()
            .zip(&b1)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(diff_e < 1e-12, "E mismatch for new_unit_cube/default params: {diff_e:.3e}");
        assert!(diff_b < 1e-12, "B mismatch for new_unit_cube/default params: {diff_b:.3e}");
    }

    #[test]
    fn first_order_3d_without_pec_constructor_matches_empty_tag_constructor() {
        let skel_a = FirstOrderMaxwell3DSkeleton::new_unit_cube_without_pec(1, 1.0, 1.0, 0.0);
        let skel_b = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_tags(1, 1.0, 1.0, 0.0, &[]);

        assert!(skel_a.pec_dofs.is_empty(), "without_pec constructor should have no constrained dofs");
        assert!(skel_b.pec_dofs.is_empty(), "empty-tag constructor should have no constrained dofs");

        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e0 = vec![1.0_f64; skel_a.n_e];
        let mut b0 = vec![0.0_f64; skel_a.n_b];
        let force0 = vec![0.0_f64; skel_a.n_e];

        let mut e1 = vec![1.0_f64; skel_b.n_e];
        let mut b1 = vec![0.0_f64; skel_b.n_b];
        let force1 = vec![0.0_f64; skel_b.n_e];

        skel_a.b_half_step(0.01, &e0, &mut b0, &cfg);
        e0 = skel_a.e_full_step(0.01, &e0, &b0, &force0, &cfg);

        skel_b.b_half_step(0.01, &e1, &mut b1, &cfg);
        e1 = skel_b.e_full_step(0.01, &e1, &b1, &force1, &cfg);

        let diff_e = e0
            .iter()
            .zip(&e1)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let diff_b = b0
            .iter()
            .zip(&b1)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(diff_e < 1e-12, "E mismatch for without_pec/empty-tag constructors: {diff_e:.3e}");
        assert!(diff_b < 1e-12, "B mismatch for without_pec/empty-tag constructors: {diff_b:.3e}");
    }

    #[test]
    fn first_order_3d_single_tag_constructor_matches_singleton_tag_list() {
        let mesh3 = SimplexMesh::<3>::unit_cube_tet(1);
        let mut tags_set: HashSet<i32> = HashSet::new();
        for f in 0..mesh3.n_boundary_faces() as u32 {
            tags_set.insert(mesh3.face_tag(f));
        }
        let mut tags: Vec<i32> = tags_set.into_iter().collect();
        tags.sort_unstable();
        let tag = tags[0];

        let skel_a = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_tag(1, 1.0, 1.0, 0.0, tag);
        let skel_b = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_tags(1, 1.0, 1.0, 0.0, &[tag]);

        let mut da = skel_a.pec_dofs.clone();
        let mut db = skel_b.pec_dofs.clone();
        da.sort_unstable();
        db.sort_unstable();
        da.dedup();
        db.dedup();
        assert_eq!(da, db, "single-tag constructor should match singleton tag-list constructor");

        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e0 = vec![1.0_f64; skel_a.n_e];
        let mut b0 = vec![0.0_f64; skel_a.n_b];
        let force0 = vec![0.0_f64; skel_a.n_e];

        let mut e1 = vec![1.0_f64; skel_b.n_e];
        let mut b1 = vec![0.0_f64; skel_b.n_b];
        let force1 = vec![0.0_f64; skel_b.n_e];

        skel_a.b_half_step(0.01, &e0, &mut b0, &cfg);
        e0 = skel_a.e_full_step(0.01, &e0, &b0, &force0, &cfg);

        skel_b.b_half_step(0.01, &e1, &mut b1, &cfg);
        e1 = skel_b.e_full_step(0.01, &e1, &b1, &force1, &cfg);

        let diff_e = e0
            .iter()
            .zip(&e1)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let diff_b = b0
            .iter()
            .zip(&b1)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(diff_e < 1e-12, "E mismatch for single-tag constructor equivalence: {diff_e:.3e}");
        assert!(diff_b < 1e-12, "B mismatch for single-tag constructor equivalence: {diff_b:.3e}");
    }

    #[test]
    fn first_order_3d_pec_tags_order_and_duplicates_do_not_change_constraints() {
        let mesh3 = SimplexMesh::<3>::unit_cube_tet(1);
        let mut tags_set: HashSet<i32> = HashSet::new();
        for f in 0..mesh3.n_boundary_faces() as u32 {
            tags_set.insert(mesh3.face_tag(f));
        }
        let mut tags: Vec<i32> = tags_set.into_iter().collect();
        tags.sort_unstable();
        assert!(tags.len() >= 2, "unit cube should have at least two boundary tags");

        let canonical = vec![tags[0], tags[1]];
        let permuted_with_dups = vec![tags[1], tags[0], tags[1], tags[0]];

        let skel_a =
            FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_tags(1, 1.0, 1.0, 0.0, &canonical);
        let skel_b = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_tags(
            1,
            1.0,
            1.0,
            0.0,
            &permuted_with_dups,
        );

        let mut da = skel_a.pec_dofs.clone();
        let mut db = skel_b.pec_dofs.clone();
        da.sort_unstable();
        db.sort_unstable();
        da.dedup();
        db.dedup();
        assert_eq!(da, db, "PEC dof set should be invariant to tag order/duplicates");

        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e0 = vec![1.0_f64; skel_a.n_e];
        let mut b0 = vec![0.0_f64; skel_a.n_b];
        let force0 = vec![0.0_f64; skel_a.n_e];

        let mut e1 = vec![1.0_f64; skel_b.n_e];
        let mut b1 = vec![0.0_f64; skel_b.n_b];
        let force1 = vec![0.0_f64; skel_b.n_e];

        skel_a.b_half_step(0.01, &e0, &mut b0, &cfg);
        e0 = skel_a.e_full_step(0.01, &e0, &b0, &force0, &cfg);

        skel_b.b_half_step(0.01, &e1, &mut b1, &cfg);
        e1 = skel_b.e_full_step(0.01, &e1, &b1, &force1, &cfg);

        let diff_e = e0
            .iter()
            .zip(&e1)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let diff_b = b0
            .iter()
            .zip(&b1)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(diff_e < 1e-12, "E mismatch for order/duplicate-equivalent tags: {diff_e:.3e}");
        assert!(diff_b < 1e-12, "B mismatch for order/duplicate-equivalent tags: {diff_b:.3e}");
    }

    #[test]
    fn first_order_3d_invalid_pec_tags_match_empty_tag_behavior() {
        let skel_empty =
            FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_tags(1, 1.0, 1.0, 0.0, &[]);
        let skel_invalid =
            FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_tags(1, 1.0, 1.0, 0.0, &[-999]);

        assert!(
            skel_empty.pec_dofs.is_empty(),
            "empty tag list should have no constrained dofs"
        );
        assert!(
            skel_invalid.pec_dofs.is_empty(),
            "invalid tags should not constrain any dofs"
        );

        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e0 = vec![1.0_f64; skel_empty.n_e];
        let mut b0 = vec![0.0_f64; skel_empty.n_b];
        let force0 = vec![0.0_f64; skel_empty.n_e];

        let mut e1 = vec![1.0_f64; skel_invalid.n_e];
        let mut b1 = vec![0.0_f64; skel_invalid.n_b];
        let force1 = vec![0.0_f64; skel_invalid.n_e];

        skel_empty.b_half_step(0.01, &e0, &mut b0, &cfg);
        e0 = skel_empty.e_full_step(0.01, &e0, &b0, &force0, &cfg);

        skel_invalid.b_half_step(0.01, &e1, &mut b1, &cfg);
        e1 = skel_invalid.e_full_step(0.01, &e1, &b1, &force1, &cfg);

        let diff_e = e0
            .iter()
            .zip(&e1)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let diff_b = b0
            .iter()
            .zip(&b1)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(diff_e < 1e-12, "E mismatch for empty vs invalid tags: {diff_e:.3e}");
        assert!(diff_b < 1e-12, "B mismatch for empty vs invalid tags: {diff_b:.3e}");
    }

    #[test]
    fn first_order_3d_mixed_valid_invalid_duplicate_tags_match_valid_unique_subset() {
        let mesh3 = SimplexMesh::<3>::unit_cube_tet(1);
        let mut tags_set: HashSet<i32> = HashSet::new();
        for f in 0..mesh3.n_boundary_faces() as u32 {
            tags_set.insert(mesh3.face_tag(f));
        }
        let mut tags: Vec<i32> = tags_set.into_iter().collect();
        tags.sort_unstable();
        assert!(tags.len() >= 2, "unit cube should have at least two boundary tags");

        let valid_unique = vec![tags[0], tags[1]];
        let mixed = vec![tags[1], -999, tags[0], tags[1], 123456, tags[0]];

        let skel_a = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_tags(
            1,
            1.0,
            1.0,
            0.0,
            &valid_unique,
        );
        let skel_b =
            FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_tags(1, 1.0, 1.0, 0.0, &mixed);

        let mut da = skel_a.pec_dofs.clone();
        let mut db = skel_b.pec_dofs.clone();
        da.sort_unstable();
        db.sort_unstable();
        da.dedup();
        db.dedup();
        assert_eq!(da, db, "mixed tags should reduce to valid unique subset");
    }

    #[test]
    fn first_order_3d_effective_pec_tags_are_sorted_unique_and_valid() {
        let mesh3 = SimplexMesh::<3>::unit_cube_tet(1);
        let mut tags: Vec<i32> = (0..mesh3.n_boundary_faces() as u32)
            .map(|f| mesh3.face_tag(f))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        tags.sort_unstable();
        assert!(tags.len() >= 2, "unit cube should have at least two boundary tags");

        let noisy = vec![tags[1], -999, tags[0], tags[1], tags[0], 123456];
        let skel =
            FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_tags(1, 1.0, 1.0, 0.0, &noisy);

        assert_eq!(
            skel.pec_tags,
            vec![tags[0], tags[1]],
            "effective PEC tags should be valid, unique, and sorted"
        );
    }

    #[test]
    fn first_order_3d_pec_state_helpers_reflect_constructor_choice() {
        let with_pec = FirstOrderMaxwell3DSkeleton::new_unit_cube(1);
        assert!(with_pec.has_pec_constraints(), "default constructor should enable PEC constraints");
        assert!(with_pec.n_pec_dofs() > 0, "default constructor should constrain at least one dof");

        let without_pec = FirstOrderMaxwell3DSkeleton::new_unit_cube_without_pec(1, 1.0, 1.0, 0.0);
        assert!(!without_pec.has_pec_constraints(), "without_pec constructor should disable PEC constraints");
        assert_eq!(without_pec.n_pec_dofs(), 0, "without_pec constructor should have zero constrained dofs");
    }

    #[test]
    fn first_order_3d_is_pec_dof_matches_normalized_constraint_set() {
        let with_pec = FirstOrderMaxwell3DSkeleton::new_unit_cube(1);
        assert!(with_pec.n_pec_dofs() > 0, "expected non-empty PEC set for default constructor");

        for dof in 0..with_pec.n_e {
            let listed = with_pec.pec_dofs.binary_search(&dof).is_ok();
            assert_eq!(
                with_pec.is_pec_dof(dof),
                listed,
                "is_pec_dof mismatch at dof {dof}"
            );
        }
        assert!(
            !with_pec.is_pec_dof(with_pec.n_e),
            "out-of-range dof should not be reported as PEC"
        );

        let without_pec = FirstOrderMaxwell3DSkeleton::new_unit_cube_without_pec(1, 1.0, 1.0, 0.0);
        for dof in 0..without_pec.n_e {
            assert!(!without_pec.is_pec_dof(dof), "without_pec constructor should report no constrained dofs");
        }
    }

    #[test]
    fn first_order_3d_pec_tag_helpers_reflect_constructor_choice() {
        let default_pec = FirstOrderMaxwell3DSkeleton::new_unit_cube(1);
        assert!(default_pec.has_pec_tags(), "default constructor should enable PEC tags");
        assert!(default_pec.n_pec_tags() > 0, "default constructor should have non-empty PEC tag set");

        let no_pec = FirstOrderMaxwell3DSkeleton::new_unit_cube_without_pec(1, 1.0, 1.0, 0.0);
        assert!(!no_pec.has_pec_tags(), "without_pec constructor should disable PEC tags");
        assert_eq!(no_pec.n_pec_tags(), 0, "without_pec constructor should have zero PEC tags");

        let mesh3 = SimplexMesh::<3>::unit_cube_tet(1);
        let mut tags: Vec<i32> = (0..mesh3.n_boundary_faces() as u32)
            .map(|f| mesh3.face_tag(f))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        tags.sort_unstable();
        let single = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_tag(1, 1.0, 1.0, 0.0, tags[0]);
        assert!(single.has_pec_tags(), "single-tag constructor should have PEC tags");
        assert_eq!(single.n_pec_tags(), 1, "single-tag constructor should report exactly one PEC tag");
    }

    #[test]
    fn first_order_3d_is_pec_tag_matches_effective_tag_set() {
        let mesh3 = SimplexMesh::<3>::unit_cube_tet(1);
        let mut all_tags: Vec<i32> = (0..mesh3.n_boundary_faces() as u32)
            .map(|f| mesh3.face_tag(f))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        all_tags.sort_unstable();

        let selected = vec![all_tags[0], all_tags[1]];
        let skel =
            FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_tags(1, 1.0, 1.0, 0.0, &selected);

        for &t in &all_tags {
            let expected = selected.contains(&t);
            assert_eq!(
                skel.is_pec_tag(t),
                expected,
                "is_pec_tag mismatch for tag {t}"
            );
        }

        assert!(!skel.is_pec_tag(-12345), "invalid tag should not be active PEC tag");

        let no_pec = FirstOrderMaxwell3DSkeleton::new_unit_cube_without_pec(1, 1.0, 1.0, 0.0);
        for &t in &all_tags {
            assert!(!no_pec.is_pec_tag(t), "without_pec constructor should not activate any tag");
        }
    }

    #[test]
    fn first_order_3d_pec_dof_fraction_reflects_constraint_density() {
        let with_pec = FirstOrderMaxwell3DSkeleton::new_unit_cube(1);
        let frac_with = with_pec.pec_dof_fraction();
        assert!(frac_with > 0.0, "default constructor should have positive PEC dof fraction");
        assert!(frac_with <= 1.0, "PEC dof fraction must be <= 1");

        let without_pec = FirstOrderMaxwell3DSkeleton::new_unit_cube_without_pec(1, 1.0, 1.0, 0.0);
        let frac_without = without_pec.pec_dof_fraction();
        assert_eq!(frac_without, 0.0, "without_pec constructor should have zero PEC dof fraction");
    }

    #[test]
    fn first_order_3d_pec_tag_fraction_reflects_constraint_density() {
        let with_pec = FirstOrderMaxwell3DSkeleton::new_unit_cube(1);
        let frac_with = with_pec.pec_tag_fraction();
        assert_eq!(frac_with, 1.0, "default constructor should activate all boundary tags");

        let mesh3 = SimplexMesh::<3>::unit_cube_tet(1);
        let mut tags: Vec<i32> = (0..mesh3.n_boundary_faces() as u32)
            .map(|f| mesh3.face_tag(f))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        tags.sort_unstable();
        let single = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_tag(1, 1.0, 1.0, 0.0, tags[0]);
        let frac_single = single.pec_tag_fraction();
        assert!(frac_single > 0.0 && frac_single < 1.0, "single-tag constructor should have fractional tag coverage");

        let without_pec = FirstOrderMaxwell3DSkeleton::new_unit_cube_without_pec(1, 1.0, 1.0, 0.0);
        let frac_without = without_pec.pec_tag_fraction();
        assert_eq!(frac_without, 0.0, "without_pec constructor should have zero PEC tag fraction");
    }

    #[test]
    fn first_order_3d_free_dof_helpers_are_consistent_with_pec_helpers() {
        let with_pec = FirstOrderMaxwell3DSkeleton::new_unit_cube(1);
        assert_eq!(
            with_pec.n_pec_dofs() + with_pec.n_free_dofs(),
            with_pec.n_e,
            "PEC/free dof counts should partition total dofs"
        );
        let frac_sum_with = with_pec.pec_dof_fraction() + with_pec.free_dof_fraction();
        assert!(
            (frac_sum_with - 1.0).abs() < 1e-15,
            "PEC/free fractions should sum to one"
        );

        let without_pec = FirstOrderMaxwell3DSkeleton::new_unit_cube_without_pec(1, 1.0, 1.0, 0.0);
        assert_eq!(without_pec.n_free_dofs(), without_pec.n_e, "without_pec should leave all dofs free");
        assert_eq!(without_pec.free_dof_fraction(), 1.0, "without_pec should have unit free-dof fraction");
        assert_eq!(without_pec.pec_dof_fraction(), 0.0, "without_pec should have zero PEC-dof fraction");
    }

    #[test]
    fn first_order_3d_free_tag_helpers_are_consistent_with_pec_tag_helpers() {
        let with_pec = FirstOrderMaxwell3DSkeleton::new_unit_cube(1);
        assert_eq!(
            with_pec.n_pec_tags() + with_pec.n_free_tags(),
            with_pec.n_boundary_tags,
            "PEC/free tag counts should partition total boundary tags"
        );
        let frac_sum_with = with_pec.pec_tag_fraction() + with_pec.free_tag_fraction();
        assert!(
            (frac_sum_with - 1.0).abs() < 1e-15,
            "PEC/free tag fractions should sum to one"
        );

        let without_pec = FirstOrderMaxwell3DSkeleton::new_unit_cube_without_pec(1, 1.0, 1.0, 0.0);
        assert_eq!(
            without_pec.n_free_tags(),
            without_pec.n_boundary_tags,
            "without_pec should leave all boundary tags free"
        );
        assert_eq!(without_pec.free_tag_fraction(), 1.0, "without_pec should have unit free-tag fraction");
        assert_eq!(without_pec.pec_tag_fraction(), 0.0, "without_pec should have zero PEC-tag fraction");
    }

    #[test]
    fn first_order_3d_is_free_dof_complements_is_pec_dof() {
        let with_pec = FirstOrderMaxwell3DSkeleton::new_unit_cube(1);
        for dof in 0..with_pec.n_e {
            assert_eq!(
                with_pec.is_free_dof(dof),
                !with_pec.is_pec_dof(dof),
                "free/pec complement mismatch at dof {dof}"
            );
        }
        assert!(!with_pec.is_free_dof(with_pec.n_e), "out-of-range dof should not be free");

        let without_pec = FirstOrderMaxwell3DSkeleton::new_unit_cube_without_pec(1, 1.0, 1.0, 0.0);
        for dof in 0..without_pec.n_e {
            assert!(without_pec.is_free_dof(dof), "without_pec constructor should mark all dofs free");
        }
    }

    #[test]
    fn first_order_3d_is_free_tag_complements_is_pec_tag() {
        let mesh3 = SimplexMesh::<3>::unit_cube_tet(1);
        let mut all_tags: Vec<i32> = (0..mesh3.n_boundary_faces() as u32)
            .map(|f| mesh3.face_tag(f))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        all_tags.sort_unstable();

        let selected = vec![all_tags[0], all_tags[1]];
        let with_pec = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_tags(1, 1.0, 1.0, 0.0, &selected);
        for &tag in &all_tags {
            assert_eq!(
                with_pec.is_free_tag(tag),
                !with_pec.is_pec_tag(tag),
                "free/pec tag complement mismatch for tag {tag}"
            );
        }

        assert!(!with_pec.is_free_tag(-12345), "invalid negative tag should not be reported as free tag");

        let without_pec = FirstOrderMaxwell3DSkeleton::new_unit_cube_without_pec(1, 1.0, 1.0, 0.0);
        for &tag in &all_tags {
            assert!(without_pec.is_free_tag(tag), "without_pec constructor should mark all tags free");
        }
    }

    #[test]
    fn first_order_3d_pec_dofs_are_sorted_unique_after_normalization() {
        let mesh3 = SimplexMesh::<3>::unit_cube_tet(1);
        let mut tags_set: HashSet<i32> = HashSet::new();
        for f in 0..mesh3.n_boundary_faces() as u32 {
            tags_set.insert(mesh3.face_tag(f));
        }
        let mut tags: Vec<i32> = tags_set.into_iter().collect();
        tags.sort_unstable();
        assert!(tags.len() >= 2, "unit cube should have at least two boundary tags");

        let noisy_tags = vec![tags[1], tags[0], tags[1], tags[0], -1, 12345];
        let skel =
            FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_tags(1, 1.0, 1.0, 0.0, &noisy_tags);

        assert!(
            skel.pec_dofs.windows(2).all(|w| w[0] < w[1]),
            "PEC dofs should be strictly increasing after normalization"
        );
    }

    #[test]
    fn first_order_3d_absorbing_boundary_term_dissipates_energy_without_sigma() {
        let mesh3 = SimplexMesh::<3>::unit_cube_tet(1);
        let mut tags: Vec<i32> = (0..mesh3.n_boundary_faces() as u32)
            .map(|f| mesh3.face_tag(f))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        tags.sort_unstable();
        assert!(!tags.is_empty(), "unit cube should have boundary tags");

        let skel_no_abc = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_and_abc_tags(
            1,
            1.0,
            1.0,
            0.0,
            &[],
            &[],
            0.0,
        );
        let skel_abc = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_and_abc_tags(
            1,
            1.0,
            1.0,
            0.0,
            &[],
            &[tags[0]],
            0.2,
        );

        assert_eq!(skel_no_abc.abc_tags.len(), 0, "no-abc constructor should have empty abc tag set");
        assert_eq!(skel_abc.abc_tags, vec![tags[0]], "abc tags should be normalized and preserved");
        assert!(skel_abc.m_e_abc.is_some(), "abc matrix should be assembled when gamma and tags are provided");

        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e_no_abc = vec![0.0_f64; skel_no_abc.n_e];
        let mut e_abc = vec![0.0_f64; skel_abc.n_e];
        for i in 0..skel_no_abc.n_e {
            let val = (i as f64 + 1.0) / skel_no_abc.n_e as f64;
            e_no_abc[i] = val;
            e_abc[i] = val;
        }
        let mut b_no_abc = vec![0.0_f64; skel_no_abc.n_b];
        let mut b_abc = vec![0.0_f64; skel_abc.n_b];
        let force_no_abc = vec![0.0_f64; skel_no_abc.n_e];
        let force_abc = vec![0.0_f64; skel_abc.n_e];

        let e0_no_abc = skel_no_abc.compute_energy(&e_no_abc, &b_no_abc);
        let e0_abc = skel_abc.compute_energy(&e_abc, &b_abc);
        assert!(e0_no_abc > 0.0 && e0_abc > 0.0, "initial energies must be positive");

        let dt = 0.01;
        for _ in 0..120 {
            skel_no_abc.b_half_step(dt, &e_no_abc, &mut b_no_abc, &cfg);
            e_no_abc = skel_no_abc.e_full_step(dt, &e_no_abc, &b_no_abc, &force_no_abc, &cfg);

            skel_abc.b_half_step(dt, &e_abc, &mut b_abc, &cfg);
            e_abc = skel_abc.e_full_step(dt, &e_abc, &b_abc, &force_abc, &cfg);
        }

        let e1_no_abc = skel_no_abc.compute_energy(&e_no_abc, &b_no_abc);
        let e1_abc = skel_abc.compute_energy(&e_abc, &b_abc);

        assert!(
            e1_abc < e1_no_abc,
            "ABC damping should lower final energy versus no-ABC case: no_abc={e1_no_abc:.6e}, abc={e1_abc:.6e}"
        );
        assert!(
            e1_abc < 0.95 * e0_abc,
            "ABC damping should dissipate energy with sigma=0: E0={e0_abc:.6e}, E1={e1_abc:.6e}"
        );
    }

    #[test]
    fn first_order_3d_impedance_boundary_term_dissipates_energy_without_sigma() {
        let mesh3 = SimplexMesh::<3>::unit_cube_tet(1);
        let mut tags: Vec<i32> = (0..mesh3.n_boundary_faces() as u32)
            .map(|f| mesh3.face_tag(f))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        tags.sort_unstable();
        assert!(!tags.is_empty(), "unit cube should have boundary tags");

        let skel_no_imp = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.0,
            &[],
            &[],
            0.0,
            &[],
            0.0,
        );
        let skel_imp = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.0,
            &[],
            &[],
            0.0,
            &[tags[0]],
            0.2,
        );

        assert!(
            skel_no_imp.impedance_tags.is_empty(),
            "no-impedance constructor should have empty impedance tags"
        );
        assert_eq!(
            skel_imp.impedance_tags,
            vec![tags[0]],
            "impedance tags should be normalized and preserved"
        );
        assert!(
            skel_imp.m_e_impedance.is_some(),
            "impedance matrix should be assembled when gamma and tags are provided"
        );

        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e_no_imp = vec![0.0_f64; skel_no_imp.n_e];
        let mut e_imp = vec![0.0_f64; skel_imp.n_e];
        for i in 0..skel_no_imp.n_e {
            let val = (i as f64 + 1.0) / skel_no_imp.n_e as f64;
            e_no_imp[i] = val;
            e_imp[i] = val;
        }
        let mut b_no_imp = vec![0.0_f64; skel_no_imp.n_b];
        let mut b_imp = vec![0.0_f64; skel_imp.n_b];
        let force_no_imp = vec![0.0_f64; skel_no_imp.n_e];
        let force_imp = vec![0.0_f64; skel_imp.n_e];

        let e0_no_imp = skel_no_imp.compute_energy(&e_no_imp, &b_no_imp);
        let e0_imp = skel_imp.compute_energy(&e_imp, &b_imp);
        assert!(e0_no_imp > 0.0 && e0_imp > 0.0, "initial energies must be positive");

        let dt = 0.01;
        for _ in 0..120 {
            skel_no_imp.b_half_step(dt, &e_no_imp, &mut b_no_imp, &cfg);
            e_no_imp = skel_no_imp.e_full_step(dt, &e_no_imp, &b_no_imp, &force_no_imp, &cfg);

            skel_imp.b_half_step(dt, &e_imp, &mut b_imp, &cfg);
            e_imp = skel_imp.e_full_step(dt, &e_imp, &b_imp, &force_imp, &cfg);
        }

        let e1_no_imp = skel_no_imp.compute_energy(&e_no_imp, &b_no_imp);
        let e1_imp = skel_imp.compute_energy(&e_imp, &b_imp);

        assert!(
            e1_imp < e1_no_imp,
            "impedance damping should lower final energy versus no-impedance case: no_imp={e1_no_imp:.6e}, imp={e1_imp:.6e}"
        );
        assert!(
            e1_imp < 0.95 * e0_imp,
            "impedance damping should dissipate energy with sigma=0: E0={e0_imp:.6e}, E1={e1_imp:.6e}"
        );
    }

    #[test]
    fn first_order_3d_crank_nicolson_matches_explicit_without_damping() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.0,
            &[],
            &[],
            0.0,
            &[],
            0.0,
        );
        let cfg = SolverConfig {
            rtol: 1e-12,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e = vec![0.0_f64; skel.n_e];
        for (i, v) in e.iter_mut().enumerate() {
            *v = (i as f64 + 1.0) / skel.n_e as f64;
        }
        let mut b = vec![0.0_f64; skel.n_b];
        let force = vec![0.0_f64; skel.n_e];
        let dt = 0.01;

        skel.b_half_step(dt, &e, &mut b, &cfg);
        let e_explicit = skel.e_full_step(dt, &e, &b, &force, &cfg);
        let e_cn = skel.e_full_step_crank_nicolson(dt, &e, &b, &force, &cfg);

        let max_diff = e_explicit
            .iter()
            .zip(&e_cn)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 1e-10,
            "CN and explicit should match without damping terms: max_diff={max_diff:.3e}"
        );
    }

    #[test]
    fn first_order_3d_crank_nicolson_sigma_dissipates_energy() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.2,
            &[],
            &[],
            0.0,
            &[],
            0.0,
        );
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e = vec![0.0_f64; skel.n_e];
        for (i, v) in e.iter_mut().enumerate() {
            *v = (i as f64 + 1.0) / skel.n_e as f64;
        }
        let mut b = vec![0.0_f64; skel.n_b];
        let force = vec![0.0_f64; skel.n_e];
        let dt = 0.03;

        let e0 = skel.compute_energy(&e, &b);
        assert!(e0 > 0.0, "initial energy must be positive");

        for _ in 0..80 {
            skel.b_half_step(dt, &e, &mut b, &cfg);
            e = skel.e_full_step_crank_nicolson(dt, &e, &b, &force, &cfg);
        }

        let e1 = skel.compute_energy(&e, &b);
        assert!(
            e1 < 0.9 * e0,
            "CN with sigma should dissipate energy: E0={e0:.6e}, E1={e1:.6e}"
        );
    }

    #[test]
    fn first_order_3d_step_crank_nicolson_matches_manual_split_update() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.2,
            &[],
            &[],
            0.0,
            &[],
            0.0,
        );
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e_manual = vec![0.0_f64; skel.n_e];
        let mut e_api = vec![0.0_f64; skel.n_e];
        for i in 0..skel.n_e {
            let val = (i as f64 + 1.0) / skel.n_e as f64;
            e_manual[i] = val;
            e_api[i] = val;
        }
        let mut b_manual = vec![0.0_f64; skel.n_b];
        let mut b_api = vec![0.0_f64; skel.n_b];
        let force = vec![0.0_f64; skel.n_e];
        let dt = 0.02;

        skel.b_half_step(dt, &e_manual, &mut b_manual, &cfg);
        e_manual = skel.e_full_step_crank_nicolson(dt, &e_manual, &b_manual, &force, &cfg);

        skel.step_crank_nicolson(dt, &mut e_api, &mut b_api, &force, &cfg);

        let e_diff = e_manual
            .iter()
            .zip(&e_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let b_diff = b_manual
            .iter()
            .zip(&b_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(e_diff < 1e-12, "CN step wrapper E mismatch: {e_diff:.3e}");
        assert!(b_diff < 1e-12, "CN step wrapper B mismatch: {b_diff:.3e}");
    }

    #[test]
    fn first_order_3d_step_strategy_api_matches_existing_steppers() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.2,
            &[],
            &[1],
            0.1,
            &[2],
            0.1,
        );
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e_ref_exp = vec![0.0_f64; skel.n_e];
        let mut e_api_exp = vec![0.0_f64; skel.n_e];
        let mut e_ref_cn = vec![0.0_f64; skel.n_e];
        let mut e_api_cn = vec![0.0_f64; skel.n_e];
        for i in 0..skel.n_e {
            let val = (i as f64 + 1.0) / skel.n_e as f64;
            e_ref_exp[i] = val;
            e_api_exp[i] = val;
            e_ref_cn[i] = val;
            e_api_cn[i] = val;
        }

        let mut b_ref_exp = vec![0.0_f64; skel.n_b];
        let mut b_api_exp = vec![0.0_f64; skel.n_b];
        let mut b_ref_cn = vec![0.0_f64; skel.n_b];
        let mut b_api_cn = vec![0.0_f64; skel.n_b];
        let force = vec![0.0_f64; skel.n_e];
        let dt = 0.02;

        skel.b_half_step(dt, &e_ref_exp, &mut b_ref_exp, &cfg);
        e_ref_exp = skel.e_full_step(dt, &e_ref_exp, &b_ref_exp, &force, &cfg);
        skel.step(
            dt,
            &mut e_api_exp,
            &mut b_api_exp,
            &force,
            &cfg,
            FirstOrderTimeStepper3D::Explicit,
        );

        skel.step_crank_nicolson(dt, &mut e_ref_cn, &mut b_ref_cn, &force, &cfg);
        skel.step(
            dt,
            &mut e_api_cn,
            &mut b_api_cn,
            &force,
            &cfg,
            FirstOrderTimeStepper3D::CrankNicolson,
        );

        let e_diff_exp = e_ref_exp
            .iter()
            .zip(&e_api_exp)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let b_diff_exp = b_ref_exp
            .iter()
            .zip(&b_api_exp)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let e_diff_cn = e_ref_cn
            .iter()
            .zip(&e_api_cn)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let b_diff_cn = b_ref_cn
            .iter()
            .zip(&b_api_cn)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        assert!(e_diff_exp < 1e-12, "strategy Explicit E mismatch: {e_diff_exp:.3e}");
        assert!(b_diff_exp < 1e-12, "strategy Explicit B mismatch: {b_diff_exp:.3e}");
        assert!(e_diff_cn < 1e-12, "strategy CrankNicolson E mismatch: {e_diff_cn:.3e}");
        assert!(b_diff_cn < 1e-12, "strategy CrankNicolson B mismatch: {b_diff_cn:.3e}");
    }

    #[test]
    fn first_order_3d_step_explicit_strategy_matches_manual_with_nonzero_force() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.0,
            &[],
            &[],
            0.0,
            &[],
            0.0,
        );
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e_manual = vec![0.0_f64; skel.n_e];
        let mut e_api = vec![0.0_f64; skel.n_e];
        for i in 0..skel.n_e {
            let val = (i as f64 + 1.0) / skel.n_e as f64;
            e_manual[i] = val;
            e_api[i] = val;
        }
        let mut b_manual = vec![0.0_f64; skel.n_b];
        let mut b_api = vec![0.0_f64; skel.n_b];
        let force: Vec<f64> = (0..skel.n_e)
            .map(|i| 0.1 * (i as f64 + 1.0) / skel.n_e as f64)
            .collect();
        let dt = 0.02;

        skel.b_half_step(dt, &e_manual, &mut b_manual, &cfg);
        e_manual = skel.e_full_step(dt, &e_manual, &b_manual, &force, &cfg);

        skel.step(
            dt,
            &mut e_api,
            &mut b_api,
            &force,
            &cfg,
            FirstOrderTimeStepper3D::Explicit,
        );

        let e_diff = e_manual
            .iter()
            .zip(&e_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let b_diff = b_manual
            .iter()
            .zip(&b_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(e_diff < 1e-12, "strategy Explicit with force E mismatch: {e_diff:.3e}");
        assert!(b_diff < 1e-12, "strategy Explicit with force B mismatch: {b_diff:.3e}");
    }

    #[test]
    fn first_order_3d_step_crank_nicolson_strategy_matches_existing_stepper_with_nonzero_force() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.2,
            &[],
            &[1],
            0.1,
            &[2],
            0.1,
        );
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e_ref = vec![0.0_f64; skel.n_e];
        let mut e_api = vec![0.0_f64; skel.n_e];
        for i in 0..skel.n_e {
            let val = (i as f64 + 1.0) / skel.n_e as f64;
            e_ref[i] = val;
            e_api[i] = val;
        }
        let mut b_ref = vec![0.0_f64; skel.n_b];
        let mut b_api = vec![0.0_f64; skel.n_b];
        let force: Vec<f64> = (0..skel.n_e)
            .map(|i| 0.05 * (i as f64 + 1.0) / skel.n_e as f64)
            .collect();
        let dt = 0.02;

        skel.step_crank_nicolson(dt, &mut e_ref, &mut b_ref, &force, &cfg);

        skel.step(
            dt,
            &mut e_api,
            &mut b_api,
            &force,
            &cfg,
            FirstOrderTimeStepper3D::CrankNicolson,
        );

        let e_diff = e_ref
            .iter()
            .zip(&e_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let b_diff = b_ref
            .iter()
            .zip(&b_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(e_diff < 1e-12, "strategy CrankNicolson with force E mismatch: {e_diff:.3e}");
        assert!(b_diff < 1e-12, "strategy CrankNicolson with force B mismatch: {b_diff:.3e}");
    }

    #[test]
    fn first_order_3d_step_strategy_multistep_matches_manual_with_nonzero_force() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.2,
            &[],
            &[1],
            0.1,
            &[2],
            0.1,
        );
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let init_e: Vec<f64> = (0..skel.n_e)
            .map(|i| (i as f64 + 1.0) / skel.n_e as f64)
            .collect();
        let force: Vec<f64> = (0..skel.n_e)
            .map(|i| 0.03 * (i as f64 + 1.0) / skel.n_e as f64)
            .collect();
        let dt = 0.02;
        let n_steps = 8;

        let mut e_exp_manual = init_e.clone();
        let mut b_exp_manual = vec![0.0_f64; skel.n_b];
        let mut e_exp_api = init_e.clone();
        let mut b_exp_api = vec![0.0_f64; skel.n_b];

        for _ in 0..n_steps {
            skel.b_half_step(dt, &e_exp_manual, &mut b_exp_manual, &cfg);
            e_exp_manual = skel.e_full_step(dt, &e_exp_manual, &b_exp_manual, &force, &cfg);

            skel.step(
                dt,
                &mut e_exp_api,
                &mut b_exp_api,
                &force,
                &cfg,
                FirstOrderTimeStepper3D::Explicit,
            );
        }

        let e_exp_diff = e_exp_manual
            .iter()
            .zip(&e_exp_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let b_exp_diff = b_exp_manual
            .iter()
            .zip(&b_exp_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            e_exp_diff < 1e-12,
            "multistep explicit strategy E mismatch: {e_exp_diff:.3e}"
        );
        assert!(
            b_exp_diff < 1e-12,
            "multistep explicit strategy B mismatch: {b_exp_diff:.3e}"
        );

        let mut e_cn_manual = init_e;
        let mut b_cn_manual = vec![0.0_f64; skel.n_b];
        let mut e_cn_api = e_cn_manual.clone();
        let mut b_cn_api = vec![0.0_f64; skel.n_b];

        for _ in 0..n_steps {
            skel.step_crank_nicolson(dt, &mut e_cn_manual, &mut b_cn_manual, &force, &cfg);

            skel.step(
                dt,
                &mut e_cn_api,
                &mut b_cn_api,
                &force,
                &cfg,
                FirstOrderTimeStepper3D::CrankNicolson,
            );
        }

        let e_cn_diff = e_cn_manual
            .iter()
            .zip(&e_cn_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let b_cn_diff = b_cn_manual
            .iter()
            .zip(&b_cn_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            e_cn_diff < 1e-12,
            "multistep CN strategy E mismatch: {e_cn_diff:.3e}"
        );
        assert!(
            b_cn_diff < 1e-12,
            "multistep CN strategy B mismatch: {b_cn_diff:.3e}"
        );
    }

    #[test]
    fn first_order_3d_step_strategy_energy_trajectory_matches_manual() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.15,
            &[],
            &[1],
            0.08,
            &[2],
            0.08,
        );
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let e0: Vec<f64> = (0..skel.n_e)
            .map(|i| (i as f64 + 1.0) / skel.n_e as f64)
            .collect();
        let force: Vec<f64> = (0..skel.n_e)
            .map(|i| 0.02 * (i as f64 + 1.0) / skel.n_e as f64)
            .collect();
        let dt = 0.015;
        let n_steps = 10;

        let mut e_manual = e0.clone();
        let mut b_manual = vec![0.0_f64; skel.n_b];
        let mut e_api = e0;
        let mut b_api = vec![0.0_f64; skel.n_b];

        let mut energy_manual = Vec::with_capacity(n_steps + 1);
        let mut energy_api = Vec::with_capacity(n_steps + 1);
        energy_manual.push(skel.compute_energy(&e_manual, &b_manual));
        energy_api.push(skel.compute_energy(&e_api, &b_api));

        for _ in 0..n_steps {
            skel.step_crank_nicolson(dt, &mut e_manual, &mut b_manual, &force, &cfg);
            skel.step(
                dt,
                &mut e_api,
                &mut b_api,
                &force,
                &cfg,
                FirstOrderTimeStepper3D::CrankNicolson,
            );
            energy_manual.push(skel.compute_energy(&e_manual, &b_manual));
            energy_api.push(skel.compute_energy(&e_api, &b_api));
        }

        let max_state_e = e_manual
            .iter()
            .zip(&e_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let max_state_b = b_manual
            .iter()
            .zip(&b_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_state_e < 1e-12, "energy-trajectory test E mismatch: {max_state_e:.3e}");
        assert!(max_state_b < 1e-12, "energy-trajectory test B mismatch: {max_state_b:.3e}");

        let max_energy_diff = energy_manual
            .iter()
            .zip(&energy_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_energy_diff < 1e-12,
            "energy trajectory mismatch between manual and strategy API: {max_energy_diff:.3e}"
        );
    }

    #[test]
    fn first_order_3d_step_explicit_strategy_energy_trajectory_matches_manual() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.1,
            &[],
            &[1],
            0.05,
            &[2],
            0.05,
        );
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let e0: Vec<f64> = (0..skel.n_e)
            .map(|i| (i as f64 + 1.0) / skel.n_e as f64)
            .collect();
        let force: Vec<f64> = (0..skel.n_e)
            .map(|i| 0.02 * (i as f64 + 1.0) / skel.n_e as f64)
            .collect();
        let dt = 0.015;
        let n_steps = 10;

        let mut e_manual = e0.clone();
        let mut b_manual = vec![0.0_f64; skel.n_b];
        let mut e_api = e0;
        let mut b_api = vec![0.0_f64; skel.n_b];

        let mut energy_manual = Vec::with_capacity(n_steps + 1);
        let mut energy_api = Vec::with_capacity(n_steps + 1);
        energy_manual.push(skel.compute_energy(&e_manual, &b_manual));
        energy_api.push(skel.compute_energy(&e_api, &b_api));

        for _ in 0..n_steps {
            skel.b_half_step(dt, &e_manual, &mut b_manual, &cfg);
            e_manual = skel.e_full_step(dt, &e_manual, &b_manual, &force, &cfg);

            skel.step(
                dt,
                &mut e_api,
                &mut b_api,
                &force,
                &cfg,
                FirstOrderTimeStepper3D::Explicit,
            );
            energy_manual.push(skel.compute_energy(&e_manual, &b_manual));
            energy_api.push(skel.compute_energy(&e_api, &b_api));
        }

        let max_state_e = e_manual
            .iter()
            .zip(&e_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let max_state_b = b_manual
            .iter()
            .zip(&b_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_state_e < 1e-12, "explicit energy-trajectory test E mismatch: {max_state_e:.3e}");
        assert!(max_state_b < 1e-12, "explicit energy-trajectory test B mismatch: {max_state_b:.3e}");

        let max_energy_diff = energy_manual
            .iter()
            .zip(&energy_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_energy_diff < 1e-12,
            "explicit energy trajectory mismatch between manual and strategy API: {max_energy_diff:.3e}"
        );
    }

    #[test]
    fn first_order_3d_step_explicit_wrapper_matches_strategy_api() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.1,
            &[],
            &[1],
            0.05,
            &[2],
            0.05,
        );
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e_wrap: Vec<f64> = (0..skel.n_e)
            .map(|i| (i as f64 + 1.0) / skel.n_e as f64)
            .collect();
        let mut e_api = e_wrap.clone();
        let mut b_wrap = vec![0.0_f64; skel.n_b];
        let mut b_api = vec![0.0_f64; skel.n_b];
        let force: Vec<f64> = (0..skel.n_e)
            .map(|i| 0.02 * (i as f64 + 1.0) / skel.n_e as f64)
            .collect();

        let dt = 0.02;
        let n_steps = 6;
        for _ in 0..n_steps {
            skel.step_explicit(dt, &mut e_wrap, &mut b_wrap, &force, &cfg);
            skel.step(
                dt,
                &mut e_api,
                &mut b_api,
                &force,
                &cfg,
                FirstOrderTimeStepper3D::Explicit,
            );
        }

        let e_diff = e_wrap
            .iter()
            .zip(&e_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let b_diff = b_wrap
            .iter()
            .zip(&b_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(e_diff < 1e-12, "step_explicit wrapper E mismatch: {e_diff:.3e}");
        assert!(b_diff < 1e-12, "step_explicit wrapper B mismatch: {b_diff:.3e}");
    }

    #[test]
    fn first_order_3d_step_with_config_matches_step_api() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.1,
            &[],
            &[1],
            0.05,
            &[2],
            0.05,
        );
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e_cfg: Vec<f64> = (0..skel.n_e)
            .map(|i| (i as f64 + 1.0) / skel.n_e as f64)
            .collect();
        let mut e_api = e_cfg.clone();
        let mut b_cfg = vec![0.0_f64; skel.n_b];
        let mut b_api = vec![0.0_f64; skel.n_b];
        let force: Vec<f64> = (0..skel.n_e)
            .map(|i| 0.02 * (i as f64 + 1.0) / skel.n_e as f64)
            .collect();

        let cfg_step = FirstOrderStepConfig3D::crank_nicolson(0.02);
        let n_steps = 5;
        for _ in 0..n_steps {
            skel.step_with_config(cfg_step, &mut e_cfg, &mut b_cfg, &force, &cfg);
            skel.step(
                cfg_step.dt,
                &mut e_api,
                &mut b_api,
                &force,
                &cfg,
                cfg_step.strategy,
            );
        }

        let e_diff = e_cfg
            .iter()
            .zip(&e_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let b_diff = b_cfg
            .iter()
            .zip(&b_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(e_diff < 1e-12, "step_with_config E mismatch: {e_diff:.3e}");
        assert!(b_diff < 1e-12, "step_with_config B mismatch: {b_diff:.3e}");
    }

    #[test]
    fn first_order_3d_step_with_config_explicit_matches_step_explicit() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.1,
            &[],
            &[1],
            0.05,
            &[2],
            0.05,
        );
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e_cfg: Vec<f64> = (0..skel.n_e)
            .map(|i| (i as f64 + 1.0) / skel.n_e as f64)
            .collect();
        let mut e_wrap = e_cfg.clone();
        let mut b_cfg = vec![0.0_f64; skel.n_b];
        let mut b_wrap = vec![0.0_f64; skel.n_b];
        let force: Vec<f64> = (0..skel.n_e)
            .map(|i| 0.02 * (i as f64 + 1.0) / skel.n_e as f64)
            .collect();

        let cfg_step = FirstOrderStepConfig3D::explicit(0.02);
        let n_steps = 5;
        for _ in 0..n_steps {
            skel.step_with_config(cfg_step, &mut e_cfg, &mut b_cfg, &force, &cfg);
            skel.step_explicit(cfg_step.dt, &mut e_wrap, &mut b_wrap, &force, &cfg);
        }

        let e_diff = e_cfg
            .iter()
            .zip(&e_wrap)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let b_diff = b_cfg
            .iter()
            .zip(&b_wrap)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(e_diff < 1e-12, "step_with_config explicit E mismatch: {e_diff:.3e}");
        assert!(b_diff < 1e-12, "step_with_config explicit B mismatch: {b_diff:.3e}");
    }

    #[test]
    fn first_order_3d_step_with_config_crank_nicolson_matches_step_crank_nicolson() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.1,
            &[],
            &[1],
            0.05,
            &[2],
            0.05,
        );
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e_cfg: Vec<f64> = (0..skel.n_e)
            .map(|i| (i as f64 + 1.0) / skel.n_e as f64)
            .collect();
        let mut e_wrap = e_cfg.clone();
        let mut b_cfg = vec![0.0_f64; skel.n_b];
        let mut b_wrap = vec![0.0_f64; skel.n_b];
        let force: Vec<f64> = (0..skel.n_e)
            .map(|i| 0.02 * (i as f64 + 1.0) / skel.n_e as f64)
            .collect();

        let cfg_step = FirstOrderStepConfig3D::crank_nicolson(0.02);
        let n_steps = 5;
        for _ in 0..n_steps {
            skel.step_with_config(cfg_step, &mut e_cfg, &mut b_cfg, &force, &cfg);
            skel.step_crank_nicolson(cfg_step.dt, &mut e_wrap, &mut b_wrap, &force, &cfg);
        }

        let e_diff = e_cfg
            .iter()
            .zip(&e_wrap)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let b_diff = b_cfg
            .iter()
            .zip(&b_wrap)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            e_diff < 1e-12,
            "step_with_config crank-nicolson E mismatch: {e_diff:.3e}"
        );
        assert!(
            b_diff < 1e-12,
            "step_with_config crank-nicolson B mismatch: {b_diff:.3e}"
        );
    }

    #[test]
    fn first_order_3d_step_with_config_crank_nicolson_energy_trajectory_matches_step_api() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.08,
            &[],
            &[1],
            0.04,
            &[2],
            0.04,
        );
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e_cfg: Vec<f64> = (0..skel.n_e)
            .map(|i| (i as f64 + 1.0) / skel.n_e as f64)
            .collect();
        let mut e_api = e_cfg.clone();
        let mut b_cfg = vec![0.0_f64; skel.n_b];
        let mut b_api = vec![0.0_f64; skel.n_b];
        let force: Vec<f64> = (0..skel.n_e)
            .map(|i| 0.015 * (i as f64 + 1.0) / skel.n_e as f64)
            .collect();

        let cfg_step = FirstOrderStepConfig3D::crank_nicolson(0.02);
        let n_steps = 8;
        let mut energy_cfg = vec![skel.compute_energy(&e_cfg, &b_cfg)];
        let mut energy_api = vec![skel.compute_energy(&e_api, &b_api)];
        for _ in 0..n_steps {
            skel.step_with_config(cfg_step, &mut e_cfg, &mut b_cfg, &force, &cfg);
            skel.step(
                cfg_step.dt,
                &mut e_api,
                &mut b_api,
                &force,
                &cfg,
                cfg_step.strategy,
            );
            energy_cfg.push(skel.compute_energy(&e_cfg, &b_cfg));
            energy_api.push(skel.compute_energy(&e_api, &b_api));
        }

        let max_state_e = e_cfg
            .iter()
            .zip(&e_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let max_state_b = b_cfg
            .iter()
            .zip(&b_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_state_e < 1e-12,
            "config CN energy-trajectory test E mismatch: {max_state_e:.3e}"
        );
        assert!(
            max_state_b < 1e-12,
            "config CN energy-trajectory test B mismatch: {max_state_b:.3e}"
        );

        let max_energy_diff = energy_cfg
            .iter()
            .zip(&energy_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_energy_diff < 1e-12,
            "config CN energy trajectory mismatch between step_with_config and step API: {max_energy_diff:.3e}"
        );
    }

    #[test]
    fn first_order_3d_step_with_config_explicit_energy_trajectory_matches_step_api() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.08,
            &[],
            &[1],
            0.04,
            &[2],
            0.04,
        );
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e_cfg: Vec<f64> = (0..skel.n_e)
            .map(|i| (i as f64 + 1.0) / skel.n_e as f64)
            .collect();
        let mut e_api = e_cfg.clone();
        let mut b_cfg = vec![0.0_f64; skel.n_b];
        let mut b_api = vec![0.0_f64; skel.n_b];
        let force: Vec<f64> = (0..skel.n_e)
            .map(|i| 0.015 * (i as f64 + 1.0) / skel.n_e as f64)
            .collect();

        let cfg_step = FirstOrderStepConfig3D::explicit(0.02);
        let n_steps = 8;
        let mut energy_cfg = vec![skel.compute_energy(&e_cfg, &b_cfg)];
        let mut energy_api = vec![skel.compute_energy(&e_api, &b_api)];
        for _ in 0..n_steps {
            skel.step_with_config(cfg_step, &mut e_cfg, &mut b_cfg, &force, &cfg);
            skel.step(
                cfg_step.dt,
                &mut e_api,
                &mut b_api,
                &force,
                &cfg,
                cfg_step.strategy,
            );
            energy_cfg.push(skel.compute_energy(&e_cfg, &b_cfg));
            energy_api.push(skel.compute_energy(&e_api, &b_api));
        }

        let max_state_e = e_cfg
            .iter()
            .zip(&e_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let max_state_b = b_cfg
            .iter()
            .zip(&b_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_state_e < 1e-12,
            "config explicit energy-trajectory test E mismatch: {max_state_e:.3e}"
        );
        assert!(
            max_state_b < 1e-12,
            "config explicit energy-trajectory test B mismatch: {max_state_b:.3e}"
        );

        let max_energy_diff = energy_cfg
            .iter()
            .zip(&energy_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_energy_diff < 1e-12,
            "config explicit energy trajectory mismatch between step_with_config and step API: {max_energy_diff:.3e}"
        );
    }

    #[test]
    fn first_order_3d_step_with_config_explicit_energy_trajectory_matches_step_explicit() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.08,
            &[],
            &[1],
            0.04,
            &[2],
            0.04,
        );
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e_cfg: Vec<f64> = (0..skel.n_e)
            .map(|i| (i as f64 + 1.0) / skel.n_e as f64)
            .collect();
        let mut e_wrap = e_cfg.clone();
        let mut b_cfg = vec![0.0_f64; skel.n_b];
        let mut b_wrap = vec![0.0_f64; skel.n_b];
        let force: Vec<f64> = (0..skel.n_e)
            .map(|i| 0.015 * (i as f64 + 1.0) / skel.n_e as f64)
            .collect();

        let cfg_step = FirstOrderStepConfig3D::explicit(0.02);
        let n_steps = 8;
        let mut energy_cfg = vec![skel.compute_energy(&e_cfg, &b_cfg)];
        let mut energy_wrap = vec![skel.compute_energy(&e_wrap, &b_wrap)];
        for _ in 0..n_steps {
            skel.step_with_config(cfg_step, &mut e_cfg, &mut b_cfg, &force, &cfg);
            skel.step_explicit(cfg_step.dt, &mut e_wrap, &mut b_wrap, &force, &cfg);
            energy_cfg.push(skel.compute_energy(&e_cfg, &b_cfg));
            energy_wrap.push(skel.compute_energy(&e_wrap, &b_wrap));
        }

        let max_state_e = e_cfg
            .iter()
            .zip(&e_wrap)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let max_state_b = b_cfg
            .iter()
            .zip(&b_wrap)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_state_e < 1e-12,
            "config explicit wrapper energy-trajectory test E mismatch: {max_state_e:.3e}"
        );
        assert!(
            max_state_b < 1e-12,
            "config explicit wrapper energy-trajectory test B mismatch: {max_state_b:.3e}"
        );

        let max_energy_diff = energy_cfg
            .iter()
            .zip(&energy_wrap)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_energy_diff < 1e-12,
            "config explicit energy trajectory mismatch between step_with_config and step_explicit: {max_energy_diff:.3e}"
        );
    }

    #[test]
    fn first_order_3d_step_with_config_crank_nicolson_energy_trajectory_matches_step_crank_nicolson() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.08,
            &[],
            &[1],
            0.04,
            &[2],
            0.04,
        );
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e_cfg: Vec<f64> = (0..skel.n_e)
            .map(|i| (i as f64 + 1.0) / skel.n_e as f64)
            .collect();
        let mut e_wrap = e_cfg.clone();
        let mut b_cfg = vec![0.0_f64; skel.n_b];
        let mut b_wrap = vec![0.0_f64; skel.n_b];
        let force: Vec<f64> = (0..skel.n_e)
            .map(|i| 0.015 * (i as f64 + 1.0) / skel.n_e as f64)
            .collect();

        let cfg_step = FirstOrderStepConfig3D::crank_nicolson(0.02);
        let n_steps = 8;
        let mut energy_cfg = vec![skel.compute_energy(&e_cfg, &b_cfg)];
        let mut energy_wrap = vec![skel.compute_energy(&e_wrap, &b_wrap)];
        for _ in 0..n_steps {
            skel.step_with_config(cfg_step, &mut e_cfg, &mut b_cfg, &force, &cfg);
            skel.step_crank_nicolson(cfg_step.dt, &mut e_wrap, &mut b_wrap, &force, &cfg);
            energy_cfg.push(skel.compute_energy(&e_cfg, &b_cfg));
            energy_wrap.push(skel.compute_energy(&e_wrap, &b_wrap));
        }

        let max_state_e = e_cfg
            .iter()
            .zip(&e_wrap)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let max_state_b = b_cfg
            .iter()
            .zip(&b_wrap)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_state_e < 1e-12,
            "config CN wrapper energy-trajectory test E mismatch: {max_state_e:.3e}"
        );
        assert!(
            max_state_b < 1e-12,
            "config CN wrapper energy-trajectory test B mismatch: {max_state_b:.3e}"
        );

        let max_energy_diff = energy_cfg
            .iter()
            .zip(&energy_wrap)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_energy_diff < 1e-12,
            "config CN energy trajectory mismatch between step_with_config and step_crank_nicolson: {max_energy_diff:.3e}"
        );
    }

    #[test]
    fn first_order_3d_step_with_config_mixed_strategy_sequence_matches_step_api() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.08,
            &[],
            &[1],
            0.04,
            &[2],
            0.04,
        );
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e_cfg: Vec<f64> = (0..skel.n_e)
            .map(|i| (i as f64 + 1.0) / skel.n_e as f64)
            .collect();
        let mut e_api = e_cfg.clone();
        let mut b_cfg = vec![0.0_f64; skel.n_b];
        let mut b_api = vec![0.0_f64; skel.n_b];
        let force: Vec<f64> = (0..skel.n_e)
            .map(|i| 0.015 * (i as f64 + 1.0) / skel.n_e as f64)
            .collect();

        let seq = [
            FirstOrderStepConfig3D::explicit(0.02),
            FirstOrderStepConfig3D::crank_nicolson(0.02),
            FirstOrderStepConfig3D::explicit(0.02),
            FirstOrderStepConfig3D::crank_nicolson(0.02),
            FirstOrderStepConfig3D::crank_nicolson(0.02),
            FirstOrderStepConfig3D::explicit(0.02),
        ];
        let mut energy_cfg = vec![skel.compute_energy(&e_cfg, &b_cfg)];
        let mut energy_api = vec![skel.compute_energy(&e_api, &b_api)];
        for cfg_step in seq {
            skel.step_with_config(cfg_step, &mut e_cfg, &mut b_cfg, &force, &cfg);
            skel.step(
                cfg_step.dt,
                &mut e_api,
                &mut b_api,
                &force,
                &cfg,
                cfg_step.strategy,
            );
            energy_cfg.push(skel.compute_energy(&e_cfg, &b_cfg));
            energy_api.push(skel.compute_energy(&e_api, &b_api));
        }

        let max_state_e = e_cfg
            .iter()
            .zip(&e_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let max_state_b = b_cfg
            .iter()
            .zip(&b_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_state_e < 1e-12,
            "config mixed-sequence test E mismatch: {max_state_e:.3e}"
        );
        assert!(
            max_state_b < 1e-12,
            "config mixed-sequence test B mismatch: {max_state_b:.3e}"
        );

        let max_energy_diff = energy_cfg
            .iter()
            .zip(&energy_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_energy_diff < 1e-12,
            "config mixed-sequence energy trajectory mismatch between step_with_config and step API: {max_energy_diff:.3e}"
        );
    }

    #[test]
    fn first_order_3d_step_with_config_variable_dt_sequence_matches_step_api() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.08,
            &[],
            &[1],
            0.04,
            &[2],
            0.04,
        );
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e_cfg: Vec<f64> = (0..skel.n_e)
            .map(|i| (i as f64 + 1.0) / skel.n_e as f64)
            .collect();
        let mut e_api = e_cfg.clone();
        let mut b_cfg = vec![0.0_f64; skel.n_b];
        let mut b_api = vec![0.0_f64; skel.n_b];
        let force: Vec<f64> = (0..skel.n_e)
            .map(|i| 0.015 * (i as f64 + 1.0) / skel.n_e as f64)
            .collect();

        let seq = [
            FirstOrderStepConfig3D::explicit(0.015),
            FirstOrderStepConfig3D::crank_nicolson(0.010),
            FirstOrderStepConfig3D::explicit(0.020),
            FirstOrderStepConfig3D::crank_nicolson(0.012),
            FirstOrderStepConfig3D::explicit(0.018),
            FirstOrderStepConfig3D::crank_nicolson(0.016),
        ];
        let mut energy_cfg = vec![skel.compute_energy(&e_cfg, &b_cfg)];
        let mut energy_api = vec![skel.compute_energy(&e_api, &b_api)];
        for cfg_step in seq {
            skel.step_with_config(cfg_step, &mut e_cfg, &mut b_cfg, &force, &cfg);
            skel.step(
                cfg_step.dt,
                &mut e_api,
                &mut b_api,
                &force,
                &cfg,
                cfg_step.strategy,
            );
            energy_cfg.push(skel.compute_energy(&e_cfg, &b_cfg));
            energy_api.push(skel.compute_energy(&e_api, &b_api));
        }

        let max_state_e = e_cfg
            .iter()
            .zip(&e_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let max_state_b = b_cfg
            .iter()
            .zip(&b_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_state_e < 1e-12,
            "config variable-dt test E mismatch: {max_state_e:.3e}"
        );
        assert!(
            max_state_b < 1e-12,
            "config variable-dt test B mismatch: {max_state_b:.3e}"
        );

        let max_energy_diff = energy_cfg
            .iter()
            .zip(&energy_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_energy_diff < 1e-12,
            "config variable-dt energy trajectory mismatch between step_with_config and step API: {max_energy_diff:.3e}"
        );
    }

    #[test]
    fn first_order_3d_step_with_config_crank_nicolson_variable_dt_sequence_matches_wrapper() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.08,
            &[],
            &[1],
            0.04,
            &[2],
            0.04,
        );
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e_cfg: Vec<f64> = (0..skel.n_e)
            .map(|i| (i as f64 + 1.0) / skel.n_e as f64)
            .collect();
        let mut e_wrap = e_cfg.clone();
        let mut b_cfg = vec![0.0_f64; skel.n_b];
        let mut b_wrap = vec![0.0_f64; skel.n_b];
        let force: Vec<f64> = (0..skel.n_e)
            .map(|i| 0.015 * (i as f64 + 1.0) / skel.n_e as f64)
            .collect();

        let dt_seq = [0.015_f64, 0.010, 0.020, 0.012, 0.018, 0.016];
        let mut energy_cfg = vec![skel.compute_energy(&e_cfg, &b_cfg)];
        let mut energy_wrap = vec![skel.compute_energy(&e_wrap, &b_wrap)];
        for dt in dt_seq {
            let cfg_step = FirstOrderStepConfig3D::crank_nicolson(dt);
            skel.step_with_config(cfg_step, &mut e_cfg, &mut b_cfg, &force, &cfg);
            skel.step_crank_nicolson(dt, &mut e_wrap, &mut b_wrap, &force, &cfg);
            energy_cfg.push(skel.compute_energy(&e_cfg, &b_cfg));
            energy_wrap.push(skel.compute_energy(&e_wrap, &b_wrap));
        }

        let max_state_e = e_cfg
            .iter()
            .zip(&e_wrap)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let max_state_b = b_cfg
            .iter()
            .zip(&b_wrap)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_state_e < 1e-12,
            "config CN variable-dt test E mismatch: {max_state_e:.3e}"
        );
        assert!(
            max_state_b < 1e-12,
            "config CN variable-dt test B mismatch: {max_state_b:.3e}"
        );

        let max_energy_diff = energy_cfg
            .iter()
            .zip(&energy_wrap)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_energy_diff < 1e-12,
            "config CN variable-dt energy trajectory mismatch between step_with_config and step_crank_nicolson: {max_energy_diff:.3e}"
        );
    }

    #[test]
    fn first_order_3d_step_with_config_explicit_variable_dt_sequence_matches_wrapper() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.08,
            &[],
            &[1],
            0.04,
            &[2],
            0.04,
        );
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e_cfg: Vec<f64> = (0..skel.n_e)
            .map(|i| (i as f64 + 1.0) / skel.n_e as f64)
            .collect();
        let mut e_wrap = e_cfg.clone();
        let mut b_cfg = vec![0.0_f64; skel.n_b];
        let mut b_wrap = vec![0.0_f64; skel.n_b];
        let force: Vec<f64> = (0..skel.n_e)
            .map(|i| 0.015 * (i as f64 + 1.0) / skel.n_e as f64)
            .collect();

        let dt_seq = [0.015_f64, 0.010, 0.020, 0.012, 0.018, 0.016];
        let mut energy_cfg = vec![skel.compute_energy(&e_cfg, &b_cfg)];
        let mut energy_wrap = vec![skel.compute_energy(&e_wrap, &b_wrap)];
        for dt in dt_seq {
            let cfg_step = FirstOrderStepConfig3D::explicit(dt);
            skel.step_with_config(cfg_step, &mut e_cfg, &mut b_cfg, &force, &cfg);
            skel.step_explicit(dt, &mut e_wrap, &mut b_wrap, &force, &cfg);
            energy_cfg.push(skel.compute_energy(&e_cfg, &b_cfg));
            energy_wrap.push(skel.compute_energy(&e_wrap, &b_wrap));
        }

        let max_state_e = e_cfg
            .iter()
            .zip(&e_wrap)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let max_state_b = b_cfg
            .iter()
            .zip(&b_wrap)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_state_e < 1e-12,
            "config explicit variable-dt test E mismatch: {max_state_e:.3e}"
        );
        assert!(
            max_state_b < 1e-12,
            "config explicit variable-dt test B mismatch: {max_state_b:.3e}"
        );

        let max_energy_diff = energy_cfg
            .iter()
            .zip(&energy_wrap)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_energy_diff < 1e-12,
            "config explicit variable-dt energy trajectory mismatch between step_with_config and step_explicit: {max_energy_diff:.3e}"
        );
    }

    #[test]
    fn first_order_3d_step_with_config_variable_dt_mixed_sequence_matches_manual_wrappers() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.08,
            &[],
            &[1],
            0.04,
            &[2],
            0.04,
        );
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e_cfg: Vec<f64> = (0..skel.n_e)
            .map(|i| (i as f64 + 1.0) / skel.n_e as f64)
            .collect();
        let mut e_manual = e_cfg.clone();
        let mut b_cfg = vec![0.0_f64; skel.n_b];
        let mut b_manual = vec![0.0_f64; skel.n_b];
        let force: Vec<f64> = (0..skel.n_e)
            .map(|i| 0.015 * (i as f64 + 1.0) / skel.n_e as f64)
            .collect();

        let seq = [
            FirstOrderStepConfig3D::explicit(0.015),
            FirstOrderStepConfig3D::crank_nicolson(0.010),
            FirstOrderStepConfig3D::explicit(0.020),
            FirstOrderStepConfig3D::crank_nicolson(0.012),
            FirstOrderStepConfig3D::crank_nicolson(0.018),
            FirstOrderStepConfig3D::explicit(0.016),
        ];

        let mut energy_cfg = vec![skel.compute_energy(&e_cfg, &b_cfg)];
        let mut energy_manual = vec![skel.compute_energy(&e_manual, &b_manual)];
        for cfg_step in seq {
            skel.step_with_config(cfg_step, &mut e_cfg, &mut b_cfg, &force, &cfg);
            match cfg_step.strategy {
                FirstOrderTimeStepper3D::Explicit => {
                    skel.step_explicit(cfg_step.dt, &mut e_manual, &mut b_manual, &force, &cfg);
                }
                FirstOrderTimeStepper3D::CrankNicolson => {
                    skel.step_crank_nicolson(
                        cfg_step.dt,
                        &mut e_manual,
                        &mut b_manual,
                        &force,
                        &cfg,
                    );
                }
            }
            energy_cfg.push(skel.compute_energy(&e_cfg, &b_cfg));
            energy_manual.push(skel.compute_energy(&e_manual, &b_manual));
        }

        let max_state_e = e_cfg
            .iter()
            .zip(&e_manual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let max_state_b = b_cfg
            .iter()
            .zip(&b_manual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_state_e < 1e-12,
            "config mixed variable-dt wrapper test E mismatch: {max_state_e:.3e}"
        );
        assert!(
            max_state_b < 1e-12,
            "config mixed variable-dt wrapper test B mismatch: {max_state_b:.3e}"
        );

        let max_energy_diff = energy_cfg
            .iter()
            .zip(&energy_manual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_energy_diff < 1e-12,
            "config mixed variable-dt energy trajectory mismatch against manual wrappers: {max_energy_diff:.3e}"
        );
    }

    #[test]
    fn first_order_3d_step_with_config_variable_dt_mixed_sequence_nonzero_b_matches_manual_wrappers() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.08,
            &[],
            &[1],
            0.04,
            &[2],
            0.04,
        );
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e_cfg: Vec<f64> = (0..skel.n_e)
            .map(|i| (i as f64 + 1.0) / skel.n_e as f64)
            .collect();
        let mut e_manual = e_cfg.clone();
        let mut b_cfg: Vec<f64> = (0..skel.n_b)
            .map(|i| 0.01 * (i as f64 + 1.0) / skel.n_b as f64)
            .collect();
        let mut b_manual = b_cfg.clone();
        let force: Vec<f64> = (0..skel.n_e)
            .map(|i| 0.015 * (i as f64 + 1.0) / skel.n_e as f64)
            .collect();

        let seq = [
            FirstOrderStepConfig3D::explicit(0.015),
            FirstOrderStepConfig3D::crank_nicolson(0.010),
            FirstOrderStepConfig3D::explicit(0.020),
            FirstOrderStepConfig3D::crank_nicolson(0.012),
            FirstOrderStepConfig3D::crank_nicolson(0.018),
            FirstOrderStepConfig3D::explicit(0.016),
        ];

        let mut energy_cfg = vec![skel.compute_energy(&e_cfg, &b_cfg)];
        let mut energy_manual = vec![skel.compute_energy(&e_manual, &b_manual)];
        for cfg_step in seq {
            skel.step_with_config(cfg_step, &mut e_cfg, &mut b_cfg, &force, &cfg);
            match cfg_step.strategy {
                FirstOrderTimeStepper3D::Explicit => {
                    skel.step_explicit(cfg_step.dt, &mut e_manual, &mut b_manual, &force, &cfg);
                }
                FirstOrderTimeStepper3D::CrankNicolson => {
                    skel.step_crank_nicolson(
                        cfg_step.dt,
                        &mut e_manual,
                        &mut b_manual,
                        &force,
                        &cfg,
                    );
                }
            }
            energy_cfg.push(skel.compute_energy(&e_cfg, &b_cfg));
            energy_manual.push(skel.compute_energy(&e_manual, &b_manual));
        }

        let max_state_e = e_cfg
            .iter()
            .zip(&e_manual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let max_state_b = b_cfg
            .iter()
            .zip(&b_manual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_state_e < 1e-12,
            "config mixed variable-dt nonzero-B test E mismatch: {max_state_e:.3e}"
        );
        assert!(
            max_state_b < 1e-12,
            "config mixed variable-dt nonzero-B test B mismatch: {max_state_b:.3e}"
        );

        let max_energy_diff = energy_cfg
            .iter()
            .zip(&energy_manual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_energy_diff < 1e-12,
            "config mixed variable-dt nonzero-B energy trajectory mismatch against manual wrappers: {max_energy_diff:.3e}"
        );
    }

    #[test]
    fn first_order_3d_step_with_config_variable_dt_mixed_sequence_nonzero_b_matches_step_api() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.08,
            &[],
            &[1],
            0.04,
            &[2],
            0.04,
        );
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e_cfg: Vec<f64> = (0..skel.n_e)
            .map(|i| (i as f64 + 1.0) / skel.n_e as f64)
            .collect();
        let mut e_api = e_cfg.clone();
        let mut b_cfg: Vec<f64> = (0..skel.n_b)
            .map(|i| 0.01 * (i as f64 + 1.0) / skel.n_b as f64)
            .collect();
        let mut b_api = b_cfg.clone();
        let force: Vec<f64> = (0..skel.n_e)
            .map(|i| 0.015 * (i as f64 + 1.0) / skel.n_e as f64)
            .collect();

        let seq = [
            FirstOrderStepConfig3D::explicit(0.015),
            FirstOrderStepConfig3D::crank_nicolson(0.010),
            FirstOrderStepConfig3D::explicit(0.020),
            FirstOrderStepConfig3D::crank_nicolson(0.012),
            FirstOrderStepConfig3D::crank_nicolson(0.018),
            FirstOrderStepConfig3D::explicit(0.016),
        ];

        let mut energy_cfg = vec![skel.compute_energy(&e_cfg, &b_cfg)];
        let mut energy_api = vec![skel.compute_energy(&e_api, &b_api)];
        for cfg_step in seq {
            skel.step_with_config(cfg_step, &mut e_cfg, &mut b_cfg, &force, &cfg);
            skel.step(
                cfg_step.dt,
                &mut e_api,
                &mut b_api,
                &force,
                &cfg,
                cfg_step.strategy,
            );
            energy_cfg.push(skel.compute_energy(&e_cfg, &b_cfg));
            energy_api.push(skel.compute_energy(&e_api, &b_api));
        }

        let max_state_e = e_cfg
            .iter()
            .zip(&e_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let max_state_b = b_cfg
            .iter()
            .zip(&b_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_state_e < 1e-12,
            "config mixed variable-dt nonzero-B vs step API E mismatch: {max_state_e:.3e}"
        );
        assert!(
            max_state_b < 1e-12,
            "config mixed variable-dt nonzero-B vs step API B mismatch: {max_state_b:.3e}"
        );

        let max_energy_diff = energy_cfg
            .iter()
            .zip(&energy_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_energy_diff < 1e-12,
            "config mixed variable-dt nonzero-B energy trajectory mismatch between step_with_config and step API: {max_energy_diff:.3e}"
        );
    }

    #[test]
    fn first_order_3d_step_with_config_explicit_variable_dt_sequence_nonzero_b_matches_step_api() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.08,
            &[],
            &[1],
            0.04,
            &[2],
            0.04,
        );
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e_cfg: Vec<f64> = (0..skel.n_e)
            .map(|i| (i as f64 + 1.0) / skel.n_e as f64)
            .collect();
        let mut e_api = e_cfg.clone();
        let mut b_cfg: Vec<f64> = (0..skel.n_b)
            .map(|i| 0.01 * (i as f64 + 1.0) / skel.n_b as f64)
            .collect();
        let mut b_api = b_cfg.clone();
        let force: Vec<f64> = (0..skel.n_e)
            .map(|i| 0.015 * (i as f64 + 1.0) / skel.n_e as f64)
            .collect();

        let dt_seq = [0.015_f64, 0.010, 0.020, 0.012, 0.018, 0.016];
        let mut energy_cfg = vec![skel.compute_energy(&e_cfg, &b_cfg)];
        let mut energy_api = vec![skel.compute_energy(&e_api, &b_api)];
        for dt in dt_seq {
            let cfg_step = FirstOrderStepConfig3D::explicit(dt);
            skel.step_with_config(cfg_step, &mut e_cfg, &mut b_cfg, &force, &cfg);
            skel.step(dt, &mut e_api, &mut b_api, &force, &cfg, FirstOrderTimeStepper3D::Explicit);
            energy_cfg.push(skel.compute_energy(&e_cfg, &b_cfg));
            energy_api.push(skel.compute_energy(&e_api, &b_api));
        }

        let max_state_e = e_cfg
            .iter()
            .zip(&e_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let max_state_b = b_cfg
            .iter()
            .zip(&b_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_state_e < 1e-12,
            "config explicit variable-dt nonzero-B test E mismatch: {max_state_e:.3e}"
        );
        assert!(
            max_state_b < 1e-12,
            "config explicit variable-dt nonzero-B test B mismatch: {max_state_b:.3e}"
        );

        let max_energy_diff = energy_cfg
            .iter()
            .zip(&energy_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_energy_diff < 1e-12,
            "config explicit variable-dt nonzero-B energy trajectory mismatch between step_with_config and step API: {max_energy_diff:.3e}"
        );
    }

    #[test]
    fn first_order_3d_step_with_config_zero_dt_is_noop_for_both_strategies() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.1,
            &[],
            &[1],
            0.05,
            &[2],
            0.05,
        );
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let e0: Vec<f64> = (0..skel.n_e)
            .map(|i| (i as f64 + 1.0) / skel.n_e as f64)
            .collect();
        let b0: Vec<f64> = (0..skel.n_b)
            .map(|i| 0.01 * (i as f64 + 1.0) / skel.n_b as f64)
            .collect();
        let force: Vec<f64> = (0..skel.n_e)
            .map(|i| 0.02 * (i as f64 + 1.0) / skel.n_e as f64)
            .collect();

        for strategy in [
            FirstOrderTimeStepper3D::Explicit,
            FirstOrderTimeStepper3D::CrankNicolson,
        ] {
            let mut e = e0.clone();
            let mut b = b0.clone();
            let cfg_step = FirstOrderStepConfig3D { dt: 0.0, strategy };
            skel.step_with_config(cfg_step, &mut e, &mut b, &force, &cfg);

            let e_diff = e
                .iter()
                .zip(&e0)
                .map(|(a, b0)| (a - b0).abs())
                .fold(0.0_f64, f64::max);
            let b_diff = b
                .iter()
                .zip(&b0)
                .map(|(a, b0)| (a - b0).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                e_diff < 1e-14,
                "step_with_config dt=0 should not change E for {:?}, got diff {e_diff:.3e}",
                strategy
            );
            assert!(
                b_diff < 1e-14,
                "step_with_config dt=0 should not change B for {:?}, got diff {b_diff:.3e}",
                strategy
            );
        }
    }

    #[test]
    fn first_order_3d_step_with_config_negative_dt_matches_step_api_for_both_strategies() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.1,
            &[],
            &[1],
            0.05,
            &[2],
            0.05,
        );
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let e0: Vec<f64> = (0..skel.n_e)
            .map(|i| (i as f64 + 1.0) / skel.n_e as f64)
            .collect();
        let b0: Vec<f64> = (0..skel.n_b)
            .map(|i| 0.01 * (i as f64 + 1.0) / skel.n_b as f64)
            .collect();
        let force: Vec<f64> = (0..skel.n_e)
            .map(|i| 0.02 * (i as f64 + 1.0) / skel.n_e as f64)
            .collect();

        for strategy in [
            FirstOrderTimeStepper3D::Explicit,
            FirstOrderTimeStepper3D::CrankNicolson,
        ] {
            let mut e_cfg = e0.clone();
            let mut b_cfg = b0.clone();
            let mut e_api = e0.clone();
            let mut b_api = b0.clone();
            let cfg_step = FirstOrderStepConfig3D { dt: -0.01, strategy };

            skel.step_with_config(cfg_step, &mut e_cfg, &mut b_cfg, &force, &cfg);
            skel.step(
                cfg_step.dt,
                &mut e_api,
                &mut b_api,
                &force,
                &cfg,
                cfg_step.strategy,
            );

            let e_diff = e_cfg
                .iter()
                .zip(&e_api)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            let b_diff = b_cfg
                .iter()
                .zip(&b_api)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                e_diff < 1e-12,
                "step_with_config negative dt E mismatch for {:?}: {e_diff:.3e}",
                strategy
            );
            assert!(
                b_diff < 1e-12,
                "step_with_config negative dt B mismatch for {:?}: {b_diff:.3e}",
                strategy
            );
        }
    }

    #[test]
    fn first_order_3d_step_with_config_negative_dt_mixed_sequence_matches_manual_dispatch() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.1,
            &[],
            &[1],
            0.05,
            &[2],
            0.05,
        );
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e_cfg: Vec<f64> = (0..skel.n_e)
            .map(|i| (i as f64 + 1.0) / skel.n_e as f64)
            .collect();
        let mut b_cfg: Vec<f64> = (0..skel.n_b)
            .map(|i| 0.01 * (i as f64 + 1.0) / skel.n_b as f64)
            .collect();
        let mut e_manual = e_cfg.clone();
        let mut b_manual = b_cfg.clone();

        let dts = [-0.01, -0.0075, -0.005, -0.0025];
        let strategies = [
            FirstOrderTimeStepper3D::Explicit,
            FirstOrderTimeStepper3D::CrankNicolson,
            FirstOrderTimeStepper3D::Explicit,
            FirstOrderTimeStepper3D::CrankNicolson,
        ];

        let mut energy_cfg = vec![skel.compute_energy(&e_cfg, &b_cfg)];
        let mut energy_manual = vec![skel.compute_energy(&e_manual, &b_manual)];

        for (step, (&dt, &strategy)) in dts.iter().zip(strategies.iter()).enumerate() {
            let force: Vec<f64> = (0..skel.n_e)
                .map(|i| 0.01 * (step as f64 + 1.0) * (i as f64 + 1.0) / skel.n_e as f64)
                .collect();

            let cfg_step = FirstOrderStepConfig3D { dt, strategy };
            skel.step_with_config(cfg_step, &mut e_cfg, &mut b_cfg, &force, &cfg);

            match strategy {
                FirstOrderTimeStepper3D::Explicit => {
                    skel.step_explicit(dt, &mut e_manual, &mut b_manual, &force, &cfg)
                }
                FirstOrderTimeStepper3D::CrankNicolson => {
                    skel.step_crank_nicolson(dt, &mut e_manual, &mut b_manual, &force, &cfg)
                }
            }

            energy_cfg.push(skel.compute_energy(&e_cfg, &b_cfg));
            energy_manual.push(skel.compute_energy(&e_manual, &b_manual));
        }

        let max_state_e = e_cfg
            .iter()
            .zip(&e_manual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let max_state_b = b_cfg
            .iter()
            .zip(&b_manual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_state_e < 1e-12,
            "config negative-dt mixed sequence E mismatch: {max_state_e:.3e}"
        );
        assert!(
            max_state_b < 1e-12,
            "config negative-dt mixed sequence B mismatch: {max_state_b:.3e}"
        );

        let max_energy_diff = energy_cfg
            .iter()
            .zip(&energy_manual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_energy_diff < 1e-12,
            "config negative-dt mixed sequence energy trajectory mismatch: {max_energy_diff:.3e}"
        );
    }

    #[test]
    fn first_order_3d_step_with_config_negative_dt_mixed_sequence_matches_step_api() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.1,
            &[],
            &[1],
            0.05,
            &[2],
            0.05,
        );
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e_cfg: Vec<f64> = (0..skel.n_e)
            .map(|i| (i as f64 + 1.0) / skel.n_e as f64)
            .collect();
        let mut b_cfg: Vec<f64> = (0..skel.n_b)
            .map(|i| 0.01 * (i as f64 + 1.0) / skel.n_b as f64)
            .collect();
        let mut e_api = e_cfg.clone();
        let mut b_api = b_cfg.clone();

        let dts = [-0.01, -0.0075, -0.005, -0.0025];
        let strategies = [
            FirstOrderTimeStepper3D::Explicit,
            FirstOrderTimeStepper3D::CrankNicolson,
            FirstOrderTimeStepper3D::Explicit,
            FirstOrderTimeStepper3D::CrankNicolson,
        ];

        let mut energy_cfg = vec![skel.compute_energy(&e_cfg, &b_cfg)];
        let mut energy_api = vec![skel.compute_energy(&e_api, &b_api)];

        for (step, (&dt, &strategy)) in dts.iter().zip(strategies.iter()).enumerate() {
            let force: Vec<f64> = (0..skel.n_e)
                .map(|i| 0.01 * (step as f64 + 1.0) * (i as f64 + 1.0) / skel.n_e as f64)
                .collect();

            let cfg_step = FirstOrderStepConfig3D { dt, strategy };
            skel.step_with_config(cfg_step, &mut e_cfg, &mut b_cfg, &force, &cfg);
            skel.step(dt, &mut e_api, &mut b_api, &force, &cfg, strategy);

            energy_cfg.push(skel.compute_energy(&e_cfg, &b_cfg));
            energy_api.push(skel.compute_energy(&e_api, &b_api));
        }

        let max_state_e = e_cfg
            .iter()
            .zip(&e_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let max_state_b = b_cfg
            .iter()
            .zip(&b_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_state_e < 1e-12,
            "config negative-dt mixed sequence vs step API E mismatch: {max_state_e:.3e}"
        );
        assert!(
            max_state_b < 1e-12,
            "config negative-dt mixed sequence vs step API B mismatch: {max_state_b:.3e}"
        );

        let max_energy_diff = energy_cfg
            .iter()
            .zip(&energy_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_energy_diff < 1e-12,
            "config negative-dt mixed sequence vs step API energy mismatch: {max_energy_diff:.3e}"
        );
    }

    #[test]
    fn first_order_3d_step_with_config_negative_dt_mixed_sequence_nonzero_b_matches_step_api() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.1,
            &[],
            &[1],
            0.05,
            &[2],
            0.05,
        );
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e_cfg: Vec<f64> = (0..skel.n_e)
            .map(|i| (i as f64 + 1.0) / skel.n_e as f64)
            .collect();
        let mut b_cfg: Vec<f64> = (0..skel.n_b)
            .map(|i| 0.03 * (i as f64 + 1.0) / skel.n_b as f64)
            .collect();
        let mut e_api = e_cfg.clone();
        let mut b_api = b_cfg.clone();

        let dts = [-0.012, -0.009, -0.006, -0.003];
        let strategies = [
            FirstOrderTimeStepper3D::Explicit,
            FirstOrderTimeStepper3D::CrankNicolson,
            FirstOrderTimeStepper3D::Explicit,
            FirstOrderTimeStepper3D::CrankNicolson,
        ];

        let mut energy_cfg = vec![skel.compute_energy(&e_cfg, &b_cfg)];
        let mut energy_api = vec![skel.compute_energy(&e_api, &b_api)];

        for (step, (&dt, &strategy)) in dts.iter().zip(strategies.iter()).enumerate() {
            let force: Vec<f64> = (0..skel.n_e)
                .map(|i| 0.015 * (step as f64 + 1.0) * (i as f64 + 1.0) / skel.n_e as f64)
                .collect();

            let cfg_step = FirstOrderStepConfig3D { dt, strategy };
            skel.step_with_config(cfg_step, &mut e_cfg, &mut b_cfg, &force, &cfg);
            skel.step(dt, &mut e_api, &mut b_api, &force, &cfg, strategy);

            energy_cfg.push(skel.compute_energy(&e_cfg, &b_cfg));
            energy_api.push(skel.compute_energy(&e_api, &b_api));
        }

        let max_state_e = e_cfg
            .iter()
            .zip(&e_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let max_state_b = b_cfg
            .iter()
            .zip(&b_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_state_e < 1e-12,
            "config negative-dt mixed sequence nonzero-B vs step API E mismatch: {max_state_e:.3e}"
        );
        assert!(
            max_state_b < 1e-12,
            "config negative-dt mixed sequence nonzero-B vs step API B mismatch: {max_state_b:.3e}"
        );

        let max_energy_diff = energy_cfg
            .iter()
            .zip(&energy_api)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_energy_diff < 1e-12,
            "config negative-dt mixed sequence nonzero-B vs step API energy mismatch: {max_energy_diff:.3e}"
        );
    }

    #[test]
    fn first_order_3d_step_with_config_negative_dt_mixed_sequence_nonzero_b_matches_manual_dispatch(
    ) {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.1,
            &[],
            &[1],
            0.05,
            &[2],
            0.05,
        );
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut e_cfg: Vec<f64> = (0..skel.n_e)
            .map(|i| (i as f64 + 1.0) / skel.n_e as f64)
            .collect();
        let mut b_cfg: Vec<f64> = (0..skel.n_b)
            .map(|i| 0.03 * (i as f64 + 1.0) / skel.n_b as f64)
            .collect();
        let mut e_manual = e_cfg.clone();
        let mut b_manual = b_cfg.clone();

        let dts = [-0.012, -0.009, -0.006, -0.003];
        let strategies = [
            FirstOrderTimeStepper3D::Explicit,
            FirstOrderTimeStepper3D::CrankNicolson,
            FirstOrderTimeStepper3D::Explicit,
            FirstOrderTimeStepper3D::CrankNicolson,
        ];

        let mut energy_cfg = vec![skel.compute_energy(&e_cfg, &b_cfg)];
        let mut energy_manual = vec![skel.compute_energy(&e_manual, &b_manual)];

        for (step, (&dt, &strategy)) in dts.iter().zip(strategies.iter()).enumerate() {
            let force: Vec<f64> = (0..skel.n_e)
                .map(|i| 0.015 * (step as f64 + 1.0) * (i as f64 + 1.0) / skel.n_e as f64)
                .collect();

            let cfg_step = FirstOrderStepConfig3D { dt, strategy };
            skel.step_with_config(cfg_step, &mut e_cfg, &mut b_cfg, &force, &cfg);

            match strategy {
                FirstOrderTimeStepper3D::Explicit => {
                    skel.step_explicit(dt, &mut e_manual, &mut b_manual, &force, &cfg)
                }
                FirstOrderTimeStepper3D::CrankNicolson => {
                    skel.step_crank_nicolson(dt, &mut e_manual, &mut b_manual, &force, &cfg)
                }
            }

            energy_cfg.push(skel.compute_energy(&e_cfg, &b_cfg));
            energy_manual.push(skel.compute_energy(&e_manual, &b_manual));
        }

        let max_state_e = e_cfg
            .iter()
            .zip(&e_manual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let max_state_b = b_cfg
            .iter()
            .zip(&b_manual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_state_e < 1e-12,
            "config negative-dt mixed sequence nonzero-B vs manual E mismatch: {max_state_e:.3e}"
        );
        assert!(
            max_state_b < 1e-12,
            "config negative-dt mixed sequence nonzero-B vs manual B mismatch: {max_state_b:.3e}"
        );

        let max_energy_diff = energy_cfg
            .iter()
            .zip(&energy_manual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_energy_diff < 1e-12,
            "config negative-dt mixed sequence nonzero-B vs manual energy mismatch: {max_energy_diff:.3e}"
        );
    }

    #[test]
    fn first_order_3d_step_config_constructors_set_expected_fields() {
        let dt_explicit = 0.0125;
        let cfg_explicit = FirstOrderStepConfig3D::explicit(dt_explicit);
        assert_eq!(cfg_explicit.strategy, FirstOrderTimeStepper3D::Explicit);
        assert!((cfg_explicit.dt - dt_explicit).abs() < 1e-15);

        let dt_cn = 0.02;
        let cfg_cn = FirstOrderStepConfig3D::crank_nicolson(dt_cn);
        assert_eq!(cfg_cn.strategy, FirstOrderTimeStepper3D::CrankNicolson);
        assert!((cfg_cn.dt - dt_cn).abs() < 1e-15);
    }

    #[test]
    fn first_order_3d_solver_wrapper_fixed_step_matches_manual_step_with_config() {
        let op_solver = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.1,
            &[],
            &[1],
            0.05,
            &[2],
            0.05,
        );
        let op_manual = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.1,
            &[],
            &[1],
            0.05,
            &[2],
            0.05,
        );
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let e0: Vec<f64> = (0..op_solver.n_e)
            .map(|i| (i as f64 + 1.0) / op_solver.n_e as f64)
            .collect();
        let b0: Vec<f64> = (0..op_solver.n_b)
            .map(|i| 0.02 * (i as f64 + 1.0) / op_solver.n_b as f64)
            .collect();
        let force: Vec<f64> = (0..op_solver.n_e)
            .map(|i| 0.01 * (i as f64 + 1.0) / op_solver.n_e as f64)
            .collect();

        let step_cfg = FirstOrderStepConfig3D::explicit(0.01);
        let mut solver = FirstOrderMaxwellSolver3D::new(
            op_solver,
            SolverConfig {
                rtol: cfg.rtol,
                atol: cfg.atol,
                max_iter: cfg.max_iter,
                verbose: cfg.verbose,
                ..SolverConfig::default()
            },
            step_cfg,
        )
        .with_state(&e0, &b0);
        solver.set_force(&force);

        let mut e_manual = e0.clone();
        let mut b_manual = b0.clone();
        for _ in 0..8 {
            solver.advance_one();
            op_manual.step_with_config(step_cfg, &mut e_manual, &mut b_manual, &force, &cfg);
        }

        let e_diff = solver
            .e
            .iter()
            .zip(&e_manual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let b_diff = solver
            .b
            .iter()
            .zip(&b_manual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(e_diff < 1e-12, "solver wrapper fixed-step E mismatch: {e_diff:.3e}");
        assert!(b_diff < 1e-12, "solver wrapper fixed-step B mismatch: {b_diff:.3e}");
        assert!((solver.time - 0.08).abs() < 1e-15, "solver wrapper time mismatch");
    }

    #[test]
    fn first_order_3d_solver_wrapper_mixed_configs_match_manual_step_with_config() {
        let op_solver = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.1,
            &[],
            &[1],
            0.05,
            &[2],
            0.05,
        );
        let op_manual = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.1,
            &[],
            &[1],
            0.05,
            &[2],
            0.05,
        );
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };

        let e0: Vec<f64> = (0..op_solver.n_e)
            .map(|i| (i as f64 + 1.0) / op_solver.n_e as f64)
            .collect();
        let b0: Vec<f64> = (0..op_solver.n_b)
            .map(|i| 0.025 * (i as f64 + 1.0) / op_solver.n_b as f64)
            .collect();

        let mut solver = FirstOrderMaxwellSolver3D::new(
            op_solver,
            SolverConfig {
                rtol: cfg.rtol,
                atol: cfg.atol,
                max_iter: cfg.max_iter,
                verbose: cfg.verbose,
                ..SolverConfig::default()
            },
            FirstOrderStepConfig3D::explicit(0.01),
        )
        .with_state(&e0, &b0);

        let mut e_manual = e0.clone();
        let mut b_manual = b0.clone();
        let mut energy_solver = vec![solver.energy()];
        let mut energy_manual = vec![op_manual.compute_energy(&e_manual, &b_manual)];

        let step_cfgs = [
            FirstOrderStepConfig3D {
                dt: 0.01,
                strategy: FirstOrderTimeStepper3D::Explicit,
            },
            FirstOrderStepConfig3D {
                dt: 0.0075,
                strategy: FirstOrderTimeStepper3D::CrankNicolson,
            },
            FirstOrderStepConfig3D {
                dt: -0.005,
                strategy: FirstOrderTimeStepper3D::Explicit,
            },
            FirstOrderStepConfig3D {
                dt: -0.0025,
                strategy: FirstOrderTimeStepper3D::CrankNicolson,
            },
        ];

        for (k, step_cfg) in step_cfgs.iter().copied().enumerate() {
            let force: Vec<f64> = (0..solver.op.n_e)
                .map(|i| 0.012 * (k as f64 + 1.0) * (i as f64 + 1.0) / solver.op.n_e as f64)
                .collect();
            solver.set_force(&force);
            solver.advance_with_config(step_cfg);

            op_manual.step_with_config(step_cfg, &mut e_manual, &mut b_manual, &force, &cfg);

            energy_solver.push(solver.energy());
            energy_manual.push(op_manual.compute_energy(&e_manual, &b_manual));
        }

        let e_diff = solver
            .e
            .iter()
            .zip(&e_manual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let b_diff = solver
            .b
            .iter()
            .zip(&b_manual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let en_diff = energy_solver
            .iter()
            .zip(&energy_manual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        assert!(e_diff < 1e-12, "solver wrapper mixed-config E mismatch: {e_diff:.3e}");
        assert!(b_diff < 1e-12, "solver wrapper mixed-config B mismatch: {b_diff:.3e}");
        assert!(en_diff < 1e-12, "solver wrapper mixed-config energy mismatch: {en_diff:.3e}");

        let expected_t: f64 = step_cfgs.iter().map(|c| c.dt).sum();
        assert!((solver.time - expected_t).abs() < 1e-15, "solver wrapper time mismatch");
    }

    #[test]
    fn first_order_3d_solver_wrapper_time_dependent_force_matches_static_when_constant() {
        let op_static = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.05,
            &[],
            &[1],
            0.02,
            &[2],
            0.03,
        );
        let op_td = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            1.0,
            1.0,
            0.05,
            &[],
            &[1],
            0.02,
            &[2],
            0.03,
        );
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 800,
            verbose: false,
            ..SolverConfig::default()
        };
        let step_cfg = FirstOrderStepConfig3D::explicit(0.01);

        let e0: Vec<f64> = (0..op_static.n_e)
            .map(|i| (i as f64 + 1.0) / op_static.n_e as f64)
            .collect();
        let b0: Vec<f64> = (0..op_static.n_b)
            .map(|i| 0.015 * (i as f64 + 1.0) / op_static.n_b as f64)
            .collect();
        let force: Vec<f64> = (0..op_static.n_e)
            .map(|i| 0.01 * (i as f64 + 1.0) / op_static.n_e as f64)
            .collect();

        let mut solver_static = FirstOrderMaxwellSolver3D::new(op_static, cfg.clone(), step_cfg)
            .with_state(&e0, &b0);
        solver_static.set_force(&force);

        let force_const = force.clone();
        let mut solver_td = FirstOrderMaxwellSolver3D::new(op_td, cfg, step_cfg).with_state(&e0, &b0);
        solver_td.set_time_dependent_force(move |_t, out| out.copy_from_slice(&force_const));

        solver_static.advance_n(8);
        solver_td.advance_n(8);

        let e_diff = solver_static
            .e
            .iter()
            .zip(&solver_td.e)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let b_diff = solver_static
            .b
            .iter()
            .zip(&solver_td.b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            e_diff < 1e-12,
            "time-dependent constant force should match static force for E, diff={e_diff:.3e}"
        );
        assert!(
            b_diff < 1e-12,
            "time-dependent constant force should match static force for B, diff={b_diff:.3e}"
        );
        assert!((solver_static.time - solver_td.time).abs() < 1e-15);
    }

    #[test]
    fn first_order_3d_physical_abc_impedance_constructor_matches_explicit_gamma() {
        let eps = 2.5;
        let mu = 1.25;
        let sigma = 0.08;
        let abc_scale = 0.7;
        let imp_scale = 1.1;
        let gamma = boundary_admittance(eps, mu);

        let skel_physical =
            FirstOrderMaxwell3DSkeleton::new_unit_cube_with_physical_abc_and_impedance_tags(
                1,
                eps,
                mu,
                sigma,
                &[],
                &[1],
                abc_scale,
                &[2],
                imp_scale,
            );
        let skel_explicit = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_pec_abc_and_impedance_tags(
            1,
            eps,
            mu,
            sigma,
            &[],
            &[1],
            abc_scale * gamma,
            &[2],
            imp_scale * gamma,
        );

        assert!((skel_physical.abc_gamma - skel_explicit.abc_gamma).abs() < 1e-15);
        assert!((skel_physical.impedance_gamma - skel_explicit.impedance_gamma).abs() < 1e-15);
        assert!(skel_physical.m_e_abc.is_some());
        assert!(skel_physical.m_e_impedance.is_some());

        let cfg = SolverConfig::default();
        let step_cfg = FirstOrderStepConfig3D::explicit(0.01);
        let mut e_a: Vec<f64> = (0..skel_physical.n_e)
            .map(|i| (i as f64 + 1.0) / skel_physical.n_e as f64)
            .collect();
        let mut b_a: Vec<f64> = (0..skel_physical.n_b)
            .map(|i| 0.01 * (i as f64 + 1.0) / skel_physical.n_b as f64)
            .collect();
        let mut e_b = e_a.clone();
        let mut b_b = b_a.clone();
        let force = vec![0.0_f64; skel_physical.n_e];

        skel_physical.step_with_config(step_cfg, &mut e_a, &mut b_a, &force, &cfg);
        skel_explicit.step_with_config(step_cfg, &mut e_b, &mut b_b, &force, &cfg);

        let e_diff = e_a
            .iter()
            .zip(&e_b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let b_diff = b_a
            .iter()
            .zip(&b_b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(e_diff < 1e-12, "physical/explicit gamma constructor E mismatch: {e_diff:.3e}");
        assert!(b_diff < 1e-12, "physical/explicit gamma constructor B mismatch: {b_diff:.3e}");
    }

    #[test]
    fn first_order_3d_mixed_operator_view_matches_direct_curl_paths() {
        let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube(1);
        let mixed = skel.mixed_operators();
        assert_eq!(mixed.n_hcurl, skel.n_e);
        assert_eq!(mixed.n_hdiv, skel.n_b);

        let e: Vec<f64> = (0..skel.n_e)
            .map(|i| (i as f64 + 1.0) / skel.n_e as f64)
            .collect();
        let b: Vec<f64> = (0..skel.n_b)
            .map(|i| 0.02 * (i as f64 + 1.0) / skel.n_b as f64)
            .collect();

        let mut y_direct = vec![0.0_f64; skel.n_b];
        let mut y_mixed = vec![0.0_f64; skel.n_b];
        skel.apply_curl(&e, &mut y_direct);
        mixed.apply_hcurl_to_hdiv(&e, &mut y_mixed);

        let mut x_direct = vec![0.0_f64; skel.n_e];
        let mut x_mixed = vec![0.0_f64; skel.n_e];
        skel.apply_curl_t(&b, &mut x_direct);
        mixed.apply_hdiv_to_hcurl(&b, &mut x_mixed);

        let y_diff = y_direct
            .iter()
            .zip(&y_mixed)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let x_diff = x_direct
            .iter()
            .zip(&x_mixed)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        assert!(y_diff < 1e-14, "mixed hcurl->hdiv mismatch: {y_diff:.3e}");
        assert!(x_diff < 1e-14, "mixed hdiv->hcurl mismatch: {x_diff:.3e}");
    }
}