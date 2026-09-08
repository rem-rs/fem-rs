//! PDE-based extrapolation across level-set interfaces — 1:1 serial port of
//! MFEM `miniapps/shifted/extrapolator.{hpp,cpp}`:
//!
//! - [`Extrapolator`]: DG-advection extrapolation of a field from the region
//!   where a level set is positive into the rest of the domain
//!   (Aslam, JCP 193(1), 2004; Bochkov & Gibou, SISC 42(4), 2020).
//! - [`AdvectionOper`]: the time-dependent advection operator with the HO
//!   (element-block mass solve) and LO (discrete upwind diffusion) modes.
//! - [`DiscreteUpwindLOSolver`]: low-order upwind operator `D` built from the
//!   high-order advection matrix `K`.
//!
//! The face normal `n = -∇ls/|∇ls|` is projected onto a continuous H1 vector
//! field (MFEM `LevelSetNormalGradCoeff` + `ProjectDiscCoefficient`), and the
//! transport equation `∂u/∂t - (n·∇)u = 0` — discretized with the volume
//! `ConvectionIntegrator(n, -1)` plus the non-conservative DG trace term of
//! MFEM's `NonconservativeDGTraceIntegrator` — is advanced with MFEM's
//! `RK2Solver(1.0)` up to `time_period`.

use nalgebra::DMatrix;

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, L2Space};

use crate::assembler::ref_elem_vol_for_space;
use crate::assembler::Assembler;
use crate::interior_faces::InteriorFaceList;
use crate::postproc::grid_function::GridFunction;
use crate::standard::MassIntegrator;

use super::distance::{assemble_lf_scalar};
use super::{
    elem_geometry, project_disc_average, project_disc_average_vec, transform_grads, DistScalarCoeff,
    DistVectorCoeff, LevelSetFn, Rk2Solver,
};

// ─── Coefficients (MFEM extrapolator.hpp) ────────────────────────────────────

/// MFEM `LevelSetNormalGradCoeff`: `n = -∇ls / |∇ls|` — points out of the known
/// (level set > 0) region.
pub struct LevelSetNormalGradCoeff<'c, 'x, M: MeshTopology> {
    ls_gf: &'c GridFunction<'x, H1Space<M>>,
}

impl<'c, 'x, M: MeshTopology> LevelSetNormalGradCoeff<'c, 'x, M> {
    pub fn new(ls_gf: &'c GridFunction<'x, H1Space<M>>) -> Self {
        LevelSetNormalGradCoeff { ls_gf }
    }
}

impl<M: MeshTopology> DistVectorCoeff for LevelSetNormalGradCoeff<'_, '_, M> {
    fn eval(&self, elem: u32, xi: &[f64], _x: &[f64], out: &mut [f64]) {
        let grad_ls = self.ls_gf.evaluate_gradient_at_element(elem, xi);
        let norm_grad: f64 = grad_ls.iter().map(|g| g * g).sum::<f64>().sqrt();
        for (o, g) in out.iter_mut().zip(grad_ls.iter()) {
            // Transport into the opposite direction of the gradient.
            *o = if norm_grad > 0.0 { -g / norm_grad } else { -g };
        }
    }
}

/// Component of the gradient of a grid function (MFEM `GradComponentCoeff`).
pub struct GradComponentCoeff<'c, 'x, S: FESpace> {
    u_gf: &'c GridFunction<'x, S>,
    comp: usize,
}

impl<'c, 'x, S: FESpace> GradComponentCoeff<'c, 'x, S> {
    pub fn new(u_gf: &'c GridFunction<'x, S>, comp: usize) -> Self {
        GradComponentCoeff { u_gf, comp }
    }
}

impl<S: FESpace> DistScalarCoeff for GradComponentCoeff<'_, '_, S> {
    fn eval(&self, elem: u32, xi: &[f64], _x: &[f64]) -> f64 {
        self.u_gf.evaluate_gradient_at_element(elem, xi)[self.comp]
    }
}

/// Projected (continuous) normal field `n(x)` given by component-major DOFs on
/// an H1 space — the stand-in for MFEM's
/// `VectorGridFunctionCoefficient(lsn_gf)`.
pub(crate) struct NormalGfCoeff<'a, M: MeshTopology + 'static> {
    space: &'a H1Space<M>,
    dofs: Vec<f64>, // component-major: dofs[c * n + dof]
}

impl<M: MeshTopology + 'static> DistVectorCoeff for NormalGfCoeff<'_, M> {
    fn eval(&self, elem: u32, xi: &[f64], _x: &[f64], out: &mut [f64]) {
        let re = ref_elem_vol_for_space(
            self.space,
            self.space.mesh().element_type(elem),
            self.space.element_order(elem),
        );
        let nd = re.n_dofs();
        let mut phi = vec![0.0_f64; nd];
        re.eval_basis(xi, &mut phi);
        let n = self.space.n_dofs();
        let dim = out.len();
        let dofs = self.space.element_dofs(elem);
        for c in 0..dim {
            let comp = &self.dofs[c * n..(c + 1) * n];
            out[c] = (0..nd).map(|i| comp[dofs[i] as usize] * phi[i]).sum();
        }
    }
}

/// `n · ∇u` with a projected normal field (MFEM `NormalGradCoeff` with
/// `VectorGridFunctionCoefficient`).
pub struct NormalGradCoeff<'c, 'x, 'p, 'd, M: MeshTopology + 'static> {
    u_gf: &'c GridFunction<'x, L2Space<M>>,
    n_coeff: &'p NormalGfCoeff<'d, M>,
}

impl<M: MeshTopology + 'static> DistScalarCoeff for NormalGradCoeff<'_, '_, '_, '_, M> {
    fn eval(&self, elem: u32, xi: &[f64], x: &[f64]) -> f64 {
        let dim = self.n_coeff.space.mesh().topological_dim() as usize;
        let mut n = vec![0.0_f64; dim];
        self.n_coeff.eval(elem, xi, x, &mut n);
        let grad_u = self.u_gf.evaluate_gradient_at_element(elem, xi);
        (0..dim).map(|d| n[d] * grad_u[d]).sum()
    }
}

/// `n · (du/dx, du/dy)` with two scalar L2 grid functions
/// (MFEM `NormalGradComponentCoeff`, Bochkov variant).
pub struct NormalGradComponentCoeff<'c, 'x, 'p, 'd, M: MeshTopology + 'static> {
    du_dx: &'c GridFunction<'x, L2Space<M>>,
    du_dy: &'c GridFunction<'x, L2Space<M>>,
    n_coeff: &'p NormalGfCoeff<'d, M>,
}

impl<M: MeshTopology + 'static> DistScalarCoeff for NormalGradComponentCoeff<'_, '_, '_, '_, M> {
    fn eval(&self, elem: u32, xi: &[f64], x: &[f64]) -> f64 {
        let dim = self.n_coeff.space.mesh().topological_dim() as usize;
        let mut n = vec![0.0_f64; dim];
        self.n_coeff.eval(elem, xi, x, &mut n);
        let gx = self.du_dx.evaluate_at_element(elem, xi);
        let gy = self.du_dy.evaluate_at_element(elem, xi);
        let grad_u = [gx, gy, 0.0];
        (0..dim).map(|d| n[d] * grad_u[d]).sum()
    }
}

// ─── DiscreteUpwindLOSolver ──────────────────────────────────────────────────

/// Low-order discrete upwind solver (MFEM `DiscreteUpwindLOSolver`).
///
/// Builds the upwind-biased matrix `D` from the (structurally symmetric)
/// high-order advection matrix `K`:
/// `d_ij = k_ij + max(0, -k_ij, -k_ji)`,
/// `d_ii = k_ii - Σ_{j≠i} max(0, -k_ij, -k_ji)`.
pub struct DiscreteUpwindLOSolver<'a> {
    d: CsrMatrix<f64>,
    m_lumped: Vec<f64>,
    _k: std::marker::PhantomData<&'a CsrMatrix<f64>>,
}

impl<'a> DiscreteUpwindLOSolver<'a> {
    /// `k`: the finalized advection matrix; `m_lumped`: the lumped mass vector
    /// (MFEM `M_lumped`).
    pub fn new(k: &'a CsrMatrix<f64>, m_lumped: Vec<f64>) -> Self {
        assert_eq!(k.nrows, k.ncols);
        let mut d = k.clone();
        for i in 0..k.nrows {
            let mut rowsum = 0.0_f64;
            for idx in k.row_ptr[i]..k.row_ptr[i + 1] {
                let j = k.col_idx[idx] as usize;
                let kij = k.values[idx];
                let sym = k
                    .find_entry(j, i)
                    .unwrap_or_else(|| panic!("DiscreteUpwindLOSolver: missing symmetric entry for ({i},{j})"));
                let kji = k.values[sym];
                let dij = 0.0_f64.max(-kij).max(-kji);
                d.values[idx] = kij + dij;
                d.values[sym] = kji + dij;
                if i != j {
                    rowsum += dij;
                }
            }
            // D(i, i) = K(i, i) - rowsum.
            let diag = k.find_entry(i, i).expect("missing diagonal entry");
            d.values[diag] = k.values[diag] - rowsum;
        }
        DiscreteUpwindLOSolver {
            d,
            m_lumped,
            _k: std::marker::PhantomData,
        }
    }

    /// `du = (D·u + rhs) / M_lumped` (MFEM `CalcLOSolution`; the serial port
    /// has no face-neighbour data, so all entries of `u` are local).
    pub fn calc_lo_solution(&self, u: &[f64], rhs: &[f64], du: &mut [f64]) {
        self.d.spmv(u, du);
        for i in 0..du.len() {
            du[i] = (du[i] + rhs[i]) / self.m_lumped[i];
        }
    }

    /// The upwind matrix (MFEM `D`).
    pub fn matrix(&self) -> &CsrMatrix<f64> {
        &self.d
    }
}

// ─── AdvectionOper ───────────────────────────────────────────────────────────

/// Advection mode (MFEM `AdvectionOper::AdvectionMode`): `HO` is the standard
/// FE advection solve (element-block mass solve), `LO` is upwind diffusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdvectionMode {
    Ho,
    Lo,
}

/// Time-dependent advection operator (MFEM `AdvectionOper`).
///
/// `dx = M⁻¹ (K x + b)` per active element block (HO) or the discrete upwind
/// low-order update (LO), zeroed outside the active zones.
pub struct AdvectionOper<'a> {
    active_zones: &'a [bool],
    /// Element-local LU factorizations of the mass matrix blocks (MFEM factors
    /// `M_loc` inside `Mult`; the blocks are constant, so the factorizations
    /// are hoisted here — identical results).
    m_lu: Vec<nalgebra::LU<f64, nalgebra::Dyn, nalgebra::Dyn>>,
    dof_blocks: Vec<Vec<usize>>,
    k: &'a CsrMatrix<f64>,
    /// RHS of the transport equation (MFEM `b`; set via [`Self::set_rhs`]).
    rhs: Vec<f64>,
    lo_solver: DiscreteUpwindLOSolver<'a>,
    pub adv_mode: AdvectionMode,
}

impl<'a> AdvectionOper<'a> {
    /// `mass`: block-diagonal mass matrix; `k`: the advection matrix
    /// (volume convection + non-conservative DG trace).
    pub fn new(
        active_zones: &'a [bool],
        mass: &CsrMatrix<f64>,
        k: &'a CsrMatrix<f64>,
        m_lumped: Vec<f64>,
        dof_blocks: Vec<Vec<usize>>,
    ) -> Self {
        let mut m_lu = Vec::with_capacity(dof_blocks.len());
        for dofs in &dof_blocks {
            let nd = dofs.len();
            let mut m_loc = DMatrix::<f64>::zeros(nd, nd);
            for (r, &dr) in dofs.iter().enumerate() {
                for (c, &dc) in dofs.iter().enumerate() {
                    m_loc[(r, c)] = mass.get(dr, dc);
                }
            }
            m_lu.push(nalgebra::LU::new(m_loc));
        }
        let n = k.nrows;
        AdvectionOper {
            active_zones,
            m_lu,
            dof_blocks,
            k,
            rhs: vec![0.0; n],
            lo_solver: DiscreteUpwindLOSolver::new(k, m_lumped),
            adv_mode: AdvectionMode::Ho,
        }
    }

    /// Update the transport RHS (MFEM fills `b` before the time loops).
    pub fn set_rhs(&mut self, rhs: Vec<f64>) {
        assert_eq!(rhs.len(), self.k.nrows);
        self.rhs = rhs;
    }

    /// `dx = d/dt x` at state `x` (MFEM `AdvectionOper::Mult`).
    pub fn mult(&self, x: &[f64], dx: &mut [f64]) {
        if self.adv_mode == AdvectionMode::Lo {
            self.lo_solver.calc_lo_solution(x, &self.rhs, dx);
            for (k, dofs) in self.dof_blocks.iter().enumerate() {
                if !self.active_zones[k] {
                    for &d in dofs {
                        dx[d] = 0.0;
                    }
                }
            }
            return;
        }
        // HO: rhs = K·x + b, then per-element block mass solve.
        let n = self.k.nrows;
        let mut r = vec![0.0_f64; n];
        self.k.spmv(x, &mut r);
        for i in 0..n {
            r[i] += self.rhs[i];
        }
        for (k, dofs) in self.dof_blocks.iter().enumerate() {
            if !self.active_zones[k] {
                for &d in dofs {
                    dx[d] = 0.0;
                }
                continue;
            }
            let nd = dofs.len();
            let mut rhs_loc = nalgebra::DVector::<f64>::zeros(nd);
            for (i, &d) in dofs.iter().enumerate() {
                rhs_loc[i] = r[d];
            }
            let dx_loc = self.m_lu[k].solve(&rhs_loc).expect("mass block singular");
            for (i, &d) in dofs.iter().enumerate() {
                dx[d] = dx_loc[i];
            }
        }
    }
}

// ─── Advection matrix assembly ───────────────────────────────────────────────

/// Volume convection `K_ij = −∫ φ_i (n·∇φ_j) dx` — MFEM's
/// `ConvectionIntegrator(n, α = −1)`, whose element matrix is
/// `elmat(i,j) = α·∫ φ_i (Q·∇φ_j)` (verified against the MFEM 4.9 source;
/// the sign follows `alpha`, unlike the DGAdvectionIntegrator convention).
fn assemble_convection_volume<M: MeshTopology + 'static>(
    mesh: &M,
    space: &L2Space<M>,
    n_coeff: &NormalGfCoeff<'_, M>,
    quad_order: u8,
    coo: &mut CooMatrix<f64>,
) {
    let dim = mesh.topological_dim() as usize;
    let mut n = vec![0.0_f64; dim];
    for e in mesh.elem_iter() {
        let order = space.element_order(e);
        let re = ref_elem_vol_for_space(space, mesh.element_type(e), order);
        let nd = re.n_dofs();
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let rule = re.quadrature(quad_order);
        let mut phi = vec![0.0_f64; nd];
        let mut grad_ref = vec![0.0_f64; nd * dim];
        let mut grad_phys = vec![0.0_f64; nd * dim];
        let mut ke = vec![0.0_f64; nd * nd];
        for (q, xi) in rule.points.iter().enumerate() {
            let (jit, det, xp) = elem_geometry(mesh, e, xi, dim);
            let w = rule.weights[q] * det.abs();
            re.eval_basis(xi, &mut phi);
            re.eval_grad_basis(xi, &mut grad_ref);
            transform_grads(&jit, &grad_ref, &mut grad_phys, nd, dim);
            n_coeff.eval(e, xi, &xp, &mut n);
            for i in 0..nd {
                for j in 0..nd {
                    let mut ndotg = 0.0_f64;
                    for d in 0..dim {
                        ndotg += n[d] * grad_phys[j * dim + d];
                    }
                    ke[i * nd + j] -= w * phi[i] * ndotg;
                }
            }
        }
        for (i, &gi) in dofs.iter().enumerate() {
            for (j, &gj) in dofs.iter().enumerate() {
                coo.add(gi, gj, ke[i * nd + j]);
            }
        }
    }
}

/// Interior-face contribution of MFEM's `NonconservativeDGTraceIntegrator(n, α)`
/// with `α = −1`: `TransposeIntegrator(DGTraceIntegrator(n, 1, −1/2))`, i.e. the
/// numerical transpose of the standard DGTrace face matrix
/// (`TransposeIntegrator` transposes *without* swapping the element roles).
///
/// Final blocks (`W` = integration weight × face measure, `vn` = n·unit
/// normal, normal pointing left→right):
/// - `vn < 0`: `K_ll += W·vn·φ⁻φ⁻`, `K_lr += −W·vn·φ⁻φ⁺`
/// - `vn > 0`: `K_rr += −W·vn·φ⁺φ⁺`, `K_rl += +W·vn·φ⁺φ⁻`
#[allow(clippy::too_many_arguments)]
fn assemble_nonconservative_faces<M: MeshTopology + 'static>(
    mesh: &M,
    space: &L2Space<M>,
    n_coeff: &NormalGfCoeff<'_, M>,
    ifl: &InteriorFaceList,
    quad_order: u8,
    coo: &mut CooMatrix<f64>,
) {
    let dim = mesh.topological_dim() as usize;
    let mut n = vec![0.0_f64; dim];
    for face in &ifl.faces {
        let el = face.elem_left;
        let er = face.elem_right;
        let face_nodes = &face.face_nodes;

        // Face geometry: unit normal (left→right) and measure.
        let (normal_l, h_f): (Vec<f64>, f64) = if dim == 2 {
            let x0 = mesh.node_coords(face_nodes[0]);
            let x1 = mesh.node_coords(face_nodes[1]);
            let dx = x1[0] - x0[0];
            let dy = x1[1] - x0[1];
            let h = (dx * dx + dy * dy).sqrt();
            (vec![dy / h, -dx / h], h)
        } else if face_nodes.len() == 3 {
            let x0 = mesh.node_coords(face_nodes[0]);
            let x1 = mesh.node_coords(face_nodes[1]);
            let x2 = mesh.node_coords(face_nodes[2]);
            let v1 = [x1[0] - x0[0], x1[1] - x0[1], x1[2] - x0[2]];
            let v2 = [x2[0] - x0[0], x2[1] - x0[1], x2[2] - x0[2]];
            let nx = v1[1] * v2[2] - v1[2] * v2[1];
            let ny = v1[2] * v2[0] - v1[0] * v2[2];
            let nz = v1[0] * v2[1] - v1[1] * v2[0];
            let h = (nx * nx + ny * ny + nz * nz).sqrt();
            (vec![nx / h, ny / h, nz / h], h)
        } else {
            // Quadrilateral face: diagonals cross product.
            let x0 = mesh.node_coords(face_nodes[0]);
            let x1 = mesh.node_coords(face_nodes[1]);
            let x2 = mesh.node_coords(face_nodes[2]);
            let x3 = mesh.node_coords(face_nodes[3]);
            let d1 = [x2[0] - x0[0], x2[1] - x0[1], x2[2] - x0[2]];
            let d2 = [x3[0] - x1[0], x3[1] - x1[1], x3[2] - x1[2]];
            let nx = d1[1] * d2[2] - d1[2] * d2[1];
            let ny = d1[2] * d2[0] - d1[0] * d2[2];
            let nz = d1[0] * d2[1] - d1[1] * d2[0];
            let h = (nx * nx + ny * ny + nz * nz).sqrt();
            (vec![nx / h, ny / h, nz / h], h)
        };
        // Orient the normal outward from the left element (centroid test).
        let mut normal_l = normal_l;
        {
            let nl = mesh.element_nodes(el);
            let mut cen = [0.0_f64; 3];
            for &nd_ in nl {
                let xc = mesh.node_coords(nd_);
                for i in 0..dim {
                    cen[i] += xc[i];
                }
            }
            for v in cen.iter_mut() {
                *v /= nl.len() as f64;
            }
            let mut fc = [0.0_f64; 3];
            for &nd_ in face_nodes {
                let xc = mesh.node_coords(nd_);
                for i in 0..dim {
                    fc[i] += xc[i];
                }
            }
            for v in fc.iter_mut() {
                *v /= face_nodes.len() as f64;
            }
            let dot: f64 = (0..dim).map(|i| normal_l[i] * (fc[i] - cen[i])).sum();
            if dot < 0.0 {
                for v in normal_l.iter_mut() {
                    *v = -*v;
                }
            }
        }

        // Face reference rule + quadrature.
        let (ref_face, face_xp): (Box<dyn fem_element::ReferenceElement>, _) = if dim == 2 {
            (ElementType::Line2.ref_elem(1), 0)
        } else if face_nodes.len() == 3 {
            (ElementType::Tri3.ref_elem(1), 0)
        } else {
            (ElementType::Quad4.ref_elem(1), 0)
        };
        let _ = face_xp;
        let q_face = ref_face.quadrature(quad_order);

        // Element data.
        let dofs_l: Vec<usize> = space.element_dofs(el).iter().map(|&d| d as usize).collect();
        let dofs_r: Vec<usize> = space.element_dofs(er).iter().map(|&d| d as usize).collect();
        let n_l = dofs_l.len();
        let n_r = dofs_r.len();
        let re_l = ref_elem_vol_for_space(space, mesh.element_type(el), space.element_order(el));
        let re_r = ref_elem_vol_for_space(space, mesh.element_type(er), space.element_order(er));

        // Affine Jacobians for the physical→reference maps.
        let (jac_l, x0_l) = affine_jac(mesh, mesh.element_nodes(el), dim);
        let (jac_r, x0_r) = affine_jac(mesh, mesh.element_nodes(er), dim);

        let mut k_ll = vec![0.0_f64; n_l * n_l];
        let mut k_lr = vec![0.0_f64; n_l * n_r];
        let mut k_rl = vec![0.0_f64; n_r * n_l];
        let mut k_rr = vec![0.0_f64; n_r * n_r];
        let mut phi_l = vec![0.0_f64; n_l];
        let mut phi_r = vec![0.0_f64; n_r];

        for (qi, xi_f) in q_face.points.iter().enumerate() {
            let w_f = q_face.weights[qi] * h_f;
            // Physical quadrature point on the face.
            let xp: Vec<f64> = if dim == 2 {
                let c0 = mesh.node_coords(face_nodes[0]);
                let c1 = mesh.node_coords(face_nodes[1]);
                let t = xi_f[0];
                (0..dim).map(|i| (1.0 - t) * c0[i] + t * c1[i]).collect()
            } else if face_nodes.len() == 3 {
                let c0 = mesh.node_coords(face_nodes[0]);
                let c1 = mesh.node_coords(face_nodes[1]);
                let c2 = mesh.node_coords(face_nodes[2]);
                let (u, v) = (xi_f[0], xi_f[1]);
                (0..dim)
                    .map(|i| (1.0 - u - v) * c0[i] + u * c1[i] + v * c2[i])
                    .collect()
            } else {
                bilinear_face_point(mesh, face_nodes, xi_f, dim)
            };

            // Map to the reference coordinates of each adjacent element.
            let xi_l = phys_to_ref(&jac_l, &x0_l, &xp, dim);
            let xi_r = phys_to_ref(&jac_r, &x0_r, &xp, dim);
            re_l.eval_basis(&xi_l, &mut phi_l);
            re_r.eval_basis(&xi_r, &mut phi_r);

            // Velocity: MFEM evaluates u at Elem1's integration point.
            n_coeff.eval(el, &xi_l, &xp, &mut n);
            let vn: f64 = (0..dim).map(|i| n[i] * normal_l[i]).sum();
            let w = w_f;
            if vn < 0.0 {
                for i in 0..n_l {
                    for j in 0..n_l {
                        k_ll[i * n_l + j] += w * vn * phi_l[i] * phi_l[j];
                    }
                    for j in 0..n_r {
                        k_lr[i * n_r + j] += -w * vn * phi_l[i] * phi_r[j];
                    }
                }
            } else if vn > 0.0 {
                for i in 0..n_r {
                    for j in 0..n_r {
                        k_rr[i * n_r + j] += -w * vn * phi_r[i] * phi_r[j];
                    }
                    for j in 0..n_l {
                        k_rl[i * n_l + j] += w * vn * phi_r[i] * phi_l[j];
                    }
                }
            }
        }

        for (i, &gi) in dofs_l.iter().enumerate() {
            for (j, &gj) in dofs_l.iter().enumerate() {
                coo.add(gi, gj, k_ll[i * n_l + j]);
            }
            for (j, &gj) in dofs_r.iter().enumerate() {
                coo.add(gi, gj, k_lr[i * n_r + j]);
            }
        }
        for (i, &gi) in dofs_r.iter().enumerate() {
            for (j, &gj) in dofs_l.iter().enumerate() {
                coo.add(gi, gj, k_rl[i * n_l + j]);
            }
            for (j, &gj) in dofs_r.iter().enumerate() {
                coo.add(gi, gj, k_rr[i * n_r + j]);
            }
        }
    }
}

/// Affine Jacobian + origin node coordinates of an element.
fn affine_jac<M: MeshTopology>(mesh: &M, nodes: &[u32], dim: usize) -> (DMatrix<f64>, Vec<f64>) {
    let x0 = mesh.node_coords(nodes[0]).to_vec();
    let mut j = DMatrix::<f64>::zeros(dim, dim);
    let axis_nodes: &[usize] = match (dim, nodes.len()) {
        (2, 4) => &[1, 3],
        (3, 8) => &[1, 3, 4],
        _ => &[1, 2, 3],
    };
    for (col, &an) in axis_nodes.iter().enumerate().take(dim) {
        let xc = mesh.node_coords(nodes[an.max(0)]);
        for row in 0..dim {
            j[(row, col)] = xc[row] - x0[row];
        }
    }
    (j, x0)
}

/// Map a physical point to reference coordinates through the affine map
/// `x ↦ x0 + J ξ` (inverse: `ξ = J⁻¹ (x − x0)`).
fn phys_to_ref(jac: &DMatrix<f64>, x0: &[f64], xp: &[f64], dim: usize) -> Vec<f64> {
    let ji = jac
        .clone()
        .try_inverse()
        .expect("affine_jac: singular element Jacobian");
    let mut xi = vec![0.0_f64; dim];
    for r in 0..dim {
        let mut s = 0.0;
        for c in 0..dim {
            s += ji[(r, c)] * (xp[c] - x0[c]);
        }
        xi[r] = s;
    }
    xi
}

/// Bilinear interpolation of a quadrilateral face point (`xi` on `[0,1]²`).
fn bilinear_face_point<M: MeshTopology>(
    mesh: &M,
    face_nodes: &[u32],
    xi: &[f64],
    dim: usize,
) -> Vec<f64> {
    let c = [
        mesh.node_coords(face_nodes[0]),
        mesh.node_coords(face_nodes[1]),
        mesh.node_coords(face_nodes[3]),
        mesh.node_coords(face_nodes[2]),
    ];
    let (u, v) = (xi[0], xi[1]);
    let w = [
        (1.0 - u) * (1.0 - v),
        u * (1.0 - v),
        (1.0 - u) * v,
        u * v,
    ];
    (0..dim)
        .map(|i| w[0] * c[0][i] + w[1] * c[1][i] + w[2] * c[2][i] + w[3] * c[3][i])
        .collect()
}

// ─── Extrapolator ────────────────────────────────────────────────────────────

/// Extrapolation strategy (MFEM `Extrapolator::XtrapType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XtrapType {
    Aslam,
    Bochkov,
}

/// PDE-based extrapolator (MFEM `Extrapolator`).
pub struct Extrapolator {
    pub xtrap_type: XtrapType,
    pub advection_mode: AdvectionMode,
    /// Extrapolation degree: 0 (constant), 1 (linear), 2 (quadratic, Aslam only).
    pub xtrap_degree: i32,
    /// Visualization flags kept for API parity (no-op in this serial port).
    pub visualization: bool,
    pub vis_steps: u32,
}

impl Default for Extrapolator {
    fn default() -> Self {
        Extrapolator {
            xtrap_type: XtrapType::Aslam,
            advection_mode: AdvectionMode::Ho,
            xtrap_degree: 1,
            visualization: false,
            vis_steps: 5,
        }
    }
}

/// Advance `sltn` from `t = 0` to `t_final` with MFEM `RK2Solver(1.0)`.
fn time_loop(oper: &AdvectionOper<'_>, sltn: &mut [f64], t_final: f64, dt: f64) {
    let ode = Rk2Solver;
    let mut t = 0.0_f64;
    loop {
        let dt_real = dt.min(t_final - t);
        ode.step(&|x: &[f64], dx: &mut [f64]| oper.mult(x, dx), dt_real, sltn);
        t += dt_real;
        if t >= t_final - 1e-8 * dt {
            break;
        }
    }
}

impl Extrapolator {
    /// Extrapolate `input` from the elements where `level_set > 0` into all
    /// other elements over the time horizon `time_period`
    /// (MFEM `Extrapolator::Extrapolate`).
    ///
    /// `input` and `xtrap` are DOF vectors on `input_space`; the internal
    /// transport runs on the matching L2 space (MFEM: `pfes_L2`).
    pub fn extrapolate<M, S>(
        &self,
        level_set: &LevelSetFn,
        input: &GridFunction<'_, S>,
        time_period: f64,
        xtrap: &mut [f64],
    ) where
        M: MeshTopology + Clone + 'static,
        S: FESpace<Mesh = M>,
    {
        let mesh = input.space().mesh();
        let order = input.space().order();
        let dim = mesh.dim() as usize;
        let ne = mesh.n_elements();
        let p = order as usize;

        // H1 space + level set grid function.
        let pfes_h1 = H1Space::new(mesh.clone(), order);
        let ls_dofs: Vec<f64> = pfes_h1.interpolate(level_set).as_slice().to_vec();
        let ls_gf = GridFunction::new(&pfes_h1, ls_dofs);

        // Mark elements (the Extrapolator uses `include_cut_cell = false`).
        let mut marker = super::marking::ShiftedFaceMarker::new(mesh, &pfes_h1, false);
        let mut elem_marker: Vec<i32> = Vec::new();
        marker.mark_elements(&ls_gf, &mut elem_marker);

        // The active zones are where we extrapolate (where the PDE is solved):
        // zones that are CUT or OUTSIDE.
        let active_zones: Vec<bool> = (0..ne)
            .map(|k| elem_marker[k] != super::SBElementType::Inside as i32)
            .collect();

        // Continuous normal field n = -∇ls/|∇ls|, projected (ARITHMETIC) onto
        // the H1 DOF layout.
        let ls_n = LevelSetNormalGradCoeff::new(&ls_gf);
        let lsn_dofs =
            project_disc_average_vec(&pfes_h1, dim, |e, xi, x, out| ls_n.eval(e, &xi, &x, out));
        let ls_n_coeff = NormalGfCoeff {
            space: &pfes_h1,
            dofs: lsn_dofs,
        };

        // L2 space for the transport; initial solution trimmed to the known
        // region (only elements inside the level set keep their values).
        let pfes_l2 = L2Space::new(mesh.clone(), order);
        let mut u = project_disc_average(&pfes_l2, |e, xi, _x| input.evaluate_at_element(e, &xi));
        for k in 0..ne {
            if elem_marker[k] != super::SBElementType::Inside as i32 {
                for &d in pfes_l2.element_dofs(k as u32) {
                    u[d as usize] = 0.0;
                }
            }
        }

        // Normal derivative fields (Aslam chains).
        let u_gf0 = GridFunction::new(&pfes_l2, u.clone());
        let mut n_grad_u = project_disc_average(&pfes_l2, |e, xi, x| {
            let c = NormalGradCoeff { u_gf: &u_gf0, n_coeff: &ls_n_coeff };
            c.eval(e, &xi, &x)
        });
        let n_grad_u_gf = GridFunction::new(&pfes_l2, n_grad_u.clone());
        let mut n_grad_n_grad_u = project_disc_average(&pfes_l2, |e, xi, x| {
            let c = NormalGradCoeff { u_gf: &n_grad_u_gf, n_coeff: &ls_n_coeff };
            c.eval(e, &xi, &x)
        });

        // lhs: mass matrix; rhs operator: volume convection + interior-face
        // non-conservative trace.
        let mass = Assembler::assemble_bilinear(
            &pfes_l2,
            &[&MassIntegrator { rho: 1.0 }],
            (2 * p) as u8,
        );
        let mut coo = CooMatrix::<f64>::new(pfes_l2.n_dofs(), pfes_l2.n_dofs());
        assemble_convection_volume(mesh, &pfes_l2, &ls_n_coeff, (2 * p - 1) as u8, &mut coo);
        let ifl = InteriorFaceList::build(mesh);
        assemble_nonconservative_faces(mesh, &pfes_l2, &ls_n_coeff, &ifl, (2 * p) as u8, &mut coo);
        let k = coo.into_csr();

        // Lumped mass (MFEM: LumpedIntegrator(MassIntegrator) diagonal —
        // row sums of the block-diagonal L2 mass equal ∫φ_i).
        let lumped: Vec<f64> = (0..pfes_l2.n_dofs())
            .map(|i| {
                (mass.row_ptr[i]..mass.row_ptr[i + 1])
                    .map(|idx| mass.values[idx])
                    .sum()
            })
            .collect();

        // Element DOF blocks (L2: one block per element).
        let dof_blocks: Vec<Vec<usize>> = (0..ne)
            .map(|e| {
                pfes_l2
                    .element_dofs(e as u32)
                    .iter()
                    .map(|&d| d as usize)
                    .collect()
            })
            .collect();

        // CFL time step; the propagation speed is 1 (MFEM: dt = 0.25 h_min/order;
        // h from the element volume: h = |K|^(1/dim)).
        let hd = mesh.topological_dim() as usize;
        let h_min = (0..ne)
            .map(|e| super::element_volume(mesh, e as u32).powf(1.0 / hd as f64))
            .fold(f64::INFINITY, f64::min);
        let dt = 0.25 * h_min / order as f64;
        let half_dt = 0.5 * dt;

        let mut oper = AdvectionOper::new(&active_zones, &mass, &k, lumped, dof_blocks);
        let mode = self.advection_mode;

        let n_dofs_l2 = pfes_l2.n_dofs();

        if self.xtrap_degree == 0 {
            // Constant extrapolation of u (always LO).
            oper.set_rhs(vec![0.0; n_dofs_l2]);
            oper.adv_mode = AdvectionMode::Lo;
            time_loop(&oper, &mut u, time_period, half_dt);
            xtrap.copy_from_slice(&u);
            return;
        }

        assert!(
            self.xtrap_degree == 1 || self.xtrap_degree == 2,
            "Wrong order input."
        );

        match self.xtrap_type {
            XtrapType::Aslam => {
                if self.xtrap_degree == 1 {
                    // Constant extrapolation of [n·∇u] (always LO).
                    oper.set_rhs(vec![0.0; n_dofs_l2]);
                    oper.adv_mode = AdvectionMode::Lo;
                    time_loop(&oper, &mut n_grad_u, time_period, half_dt);

                    // Linear extrapolation of u.
                    oper.adv_mode = mode;
                    let mut rhs = vec![0.0_f64; n_dofs_l2];
                    mass.spmv(&n_grad_u, &mut rhs);
                    oper.set_rhs(rhs);
                    time_loop(&oper, &mut u, time_period, dt);
                }

                if self.xtrap_degree == 2 {
                    // Constant extrapolation of [n·∇(n·∇u)] (always LO).
                    oper.set_rhs(vec![0.0; n_dofs_l2]);
                    oper.adv_mode = AdvectionMode::Lo;
                    time_loop(&oper, &mut n_grad_n_grad_u, time_period, half_dt);

                    // Linear extrapolation of [n·∇u].
                    oper.adv_mode = mode;
                    let mut rhs = vec![0.0_f64; n_dofs_l2];
                    mass.spmv(&n_grad_n_grad_u, &mut rhs);
                    oper.set_rhs(rhs);
                    time_loop(&oper, &mut n_grad_u, time_period, dt);

                    // Quadratic extrapolation of u.
                    let mut rhs = vec![0.0_f64; n_dofs_l2];
                    mass.spmv(&n_grad_u, &mut rhs);
                    oper.set_rhs(rhs);
                    time_loop(&oper, &mut u, time_period, dt);
                }
            }
            XtrapType::Bochkov => {
                if self.xtrap_degree == 2 {
                    panic!("Quadratic Bochkov method is not implemented.");
                }
                if self.xtrap_degree == 1 {
                    // Constant extrapolation of all ∇u components (always LO).
                    let u_gf = GridFunction::new(&pfes_l2, u.clone());
                    let mut grad_u_0 = project_disc_average(&pfes_l2, |e, xi, _x| {
                        GradComponentCoeff::new(&u_gf, 0).eval(e, &xi, &[])
                    });
                    let mut grad_u_1 = project_disc_average(&pfes_l2, |e, xi, _x| {
                        GradComponentCoeff::new(&u_gf, 1).eval(e, &xi, &[])
                    });
                    oper.set_rhs(vec![0.0; n_dofs_l2]);
                    oper.adv_mode = AdvectionMode::Lo;
                    time_loop(&oper, &mut grad_u_0, time_period, half_dt);
                    time_loop(&oper, &mut grad_u_1, time_period, half_dt);

                    // Linear extrapolation of u.
                    oper.adv_mode = mode;
                    let grad_u_0_gf = GridFunction::new(&pfes_l2, grad_u_0.clone());
                    let grad_u_1_gf = GridFunction::new(&pfes_l2, grad_u_1.clone());
                    let coeff = NormalGradComponentCoeff {
                        du_dx: &grad_u_0_gf,
                        du_dy: &grad_u_1_gf,
                        n_coeff: &ls_n_coeff,
                    };
                    let rhs = assemble_lf_scalar(&pfes_l2, &|e, xi, x| coeff.eval(e, xi, x), order);
                    oper.set_rhs(rhs);
                    time_loop(&oper, &mut u, time_period, dt);
                }
            }
        }

        // Copy the transported field into `xtrap` (MFEM:
        // `xtrap.ProjectGridFunction(u)` — element-wise nodal evaluation of
        // the L2 solution at the target space's DOF nodes; when both spaces
        // have the same layout this is a plain copy).
        if xtrap.len() == u.len() {
            xtrap.copy_from_slice(&u);
        } else {
            let u_gf = GridFunction::new(&pfes_l2, u);
            let transferred = project_disc_average(input.space(), |e, xi, _x| {
                u_gf.evaluate_at_element(e, &xi)
            });
            xtrap.copy_from_slice(&transferred);
        }
    }

    /// Errors in the cut elements, given an exact solution
    /// (MFEM `Extrapolator::ComputeLocalErrors`).  Returns
    /// `(err_L1, err_L2, err_LI)` normalized by the cut volume (L1/L2).
    pub fn compute_local_errors<M, S>(
        &self,
        level_set: &LevelSetFn,
        exact: &GridFunction<'_, S>,
        xtrap: &GridFunction<'_, S>,
    ) -> (f64, f64, f64)
    where
        M: MeshTopology + Clone + 'static,
        S: FESpace<Mesh = M>,
    {
        let mesh = exact.space().mesh();
        let order = exact.space().order();
        let ne = mesh.n_elements();

        // Mark elements (include_cut_cell = false).
        let pfes_h1 = H1Space::new(mesh.clone(), order);
        let ls_dofs: Vec<f64> = pfes_h1.interpolate(level_set).as_slice().to_vec();
        let ls_gf = GridFunction::new(&pfes_h1, ls_dofs);
        let mut marker = super::marking::ShiftedFaceMarker::new(mesh, &pfes_h1, false);
        let mut elem_marker: Vec<i32> = Vec::new();
        marker.mark_elements(&ls_gf, &mut elem_marker);

        // Per-element L1/L2/L∞ errors on a quadrature rule of order 2p+1.
        let dim = mesh.topological_dim() as usize;
        let mut err_l1 = 0.0_f64;
        let mut err_l2 = 0.0_f64;
        let mut err_li = 0.0_f64;
        let mut cut_volume = 0.0_f64;
        for e in mesh.elem_iter() {
            if elem_marker[e as usize] != super::SBElementType::Cut as i32 {
                continue;
            }
            let space = exact.space();
            let re = ref_elem_vol_for_space(space, mesh.element_type(e), exact.space().element_order(e));
            let nd = re.n_dofs();
            let rule = re.quadrature((2 * order as usize + 1) as u8);
            let mut phi = vec![0.0_f64; nd];
            let dofs = space.element_dofs(e);
            let mut e1 = 0.0_f64;
            let mut e2 = 0.0_f64;
            let mut eli = 0.0_f64;
            let mut vol = 0.0_f64;
            for (q, xi) in rule.points.iter().enumerate() {
                let (_jit, det, _xp) = elem_geometry(mesh, e, xi, dim);
                let w = rule.weights[q] * det.abs();
                re.eval_basis(xi, &mut phi);
                let u_h: f64 = (0..nd)
                    .map(|i| xtrap.dofs()[dofs[i] as usize] * phi[i])
                    .sum();
                let diff = u_h - exact.evaluate_at_element(e, xi);
                e1 += w * diff.abs();
                e2 += w * diff * diff;
                eli = eli.max(diff.abs());
                vol += w;
            }
            err_l1 += e1;
            err_l2 += e2.sqrt();
            err_li = err_li.max(eli);
            cut_volume += vol;
        }
        let _ = ne;
        (err_l1 / cut_volume, err_l2 / cut_volume, err_li)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postproc::grid_function::GridFunction;
    use fem_mesh::Mesh;
    use fem_space::H1Space;

    /// Planar-interface transport: the assembled operator must not blow up and
    /// must transport the known field stably for a few steps.
    #[test]
    fn planar_transport_stability() {
        let n = 16_usize;
        let mesh = Mesh::<2>::unit_square_tri(n);
        let order = 1_u8;
        let p = order as usize;

        let pfes_h1 = H1Space::new(mesh.clone(), order);
        let ls = |x: &[f64]| x[0] - 0.5;
        let ls_dofs: Vec<f64> = pfes_h1.interpolate(&ls).as_slice().to_vec();
        let ls_gf = GridFunction::new(&pfes_h1, ls_dofs);

        let mut marker = crate::dist_solver::ShiftedFaceMarker::new(&mesh, &pfes_h1, false);
        let mut elem_marker: Vec<i32> = Vec::new();
        marker.mark_elements(&ls_gf, &mut elem_marker);
        let active_zones: Vec<bool> = (0..mesh.n_elements())
            .map(|k| elem_marker[k] != crate::dist_solver::SBElementType::Inside as i32)
            .collect();
        println!("active zones: {}", active_zones.iter().filter(|&&a| a).count());

        let ls_n = LevelSetNormalGradCoeff::new(&ls_gf);
        let lsn_dofs = project_disc_average_vec(&pfes_h1, 2, |e, xi, x, out| {
            ls_n.eval(e, &xi, &x, out)
        });
        let ls_n_coeff = NormalGfCoeff { space: &pfes_h1, dofs: lsn_dofs.clone() };
        println!("lsn[0..4] = {:?}", &lsn_dofs[..4.min(lsn_dofs.len())]);

        let pfes_l2 = L2Space::new(mesh.clone(), order);
        let mut u = project_disc_average(&pfes_l2, |e, xi, _x| ls_gf.evaluate_at_element(e, &xi));
        let rms = |v: &[f64]| (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt();
        println!("initial u rms: {}", rms(&u));

        let mass = Assembler::assemble_bilinear(&pfes_l2, &[&MassIntegrator { rho: 1.0 }], (2 * p) as u8);
        let mut coo = CooMatrix::<f64>::new(pfes_l2.n_dofs(), pfes_l2.n_dofs());
        assemble_convection_volume(&mesh, &pfes_l2, &ls_n_coeff, (2 * p - 1) as u8, &mut coo);
        let ifl = InteriorFaceList::build(&mesh);
        println!("interior faces: {}", ifl.len());
        assemble_nonconservative_faces(&mesh, &pfes_l2, &ls_n_coeff, &ifl, (2 * p) as u8, &mut coo);
        let k = coo.into_csr();
        let mut k1 = vec![0.0_f64; pfes_l2.n_dofs()];
        k.spmv(&vec![1.0_f64; pfes_l2.n_dofs()], &mut k1);
        println!("‖K·1‖_inf = {}", k1.iter().fold(0.0_f64, |m, &v| m.max(v.abs())));

        let lumped: Vec<f64> = (0..pfes_l2.n_dofs())
            .map(|i| (mass.row_ptr[i]..mass.row_ptr[i + 1]).map(|idx| mass.values[idx]).sum())
            .collect();
        let dof_blocks: Vec<Vec<usize>> = (0..mesh.n_elements())
            .map(|e| pfes_l2.element_dofs(e as u32).iter().map(|&d| d as usize).collect())
            .collect();
        let h_min = (0..mesh.n_elements())
            .map(|e| super::super::element_volume(&mesh, e as u32).powf(0.5))
            .fold(f64::INFINITY, f64::min);
        let dt = 0.25 * h_min / order as f64;
        let mut oper = AdvectionOper::new(&active_zones, &mass, &k, lumped, dof_blocks);

        // LO phase, one step.
        oper.adv_mode = AdvectionMode::Lo;
        let ode = Rk2Solver;
        ode.step(&|x: &[f64], dx: &mut [f64]| oper.mult(x, dx), dt, &mut u);
        println!("after 1 LO step: rms={}", rms(&u));
        for _ in 0..8 {
            ode.step(&|x: &[f64], dx: &mut [f64]| oper.mult(x, dx), dt, &mut u);
        }
        println!("after 9 LO steps: rms={}", rms(&u));
        assert!(rms(&u) < 1e3, "LO transport unstable: rms={}", rms(&u));

        // HO phase with rhs = M·n_grad_u (the Aslam linear extrapolation);
        // n·∇u for u = x along n = (−1, 0) is −1 inside the known region.
        let n_grad_u: Vec<f64> = (0..mesh.n_elements())
            .flat_map(|e| {
                let v = if elem_marker[e] == crate::dist_solver::SBElementType::Inside as i32 {
                    -1.0
                } else {
                    0.0
                };
                std::iter::repeat(v).take(pfes_l2.element_dofs(0).len())
            })
            .collect();
        let mut rhs = vec![0.0_f64; pfes_l2.n_dofs()];
        mass.spmv(&n_grad_u, &mut rhs);
        oper.set_rhs(rhs);
        oper.adv_mode = AdvectionMode::Ho;
        let mut u2 = u.clone();
        for step in 0..16 {
            ode.step(&|x: &[f64], dx: &mut [f64]| oper.mult(x, dx), dt, &mut u2);
            println!("HO step {}: rms={}", step + 1, rms(&u2));
            assert!(
                rms(&u2) < 1e3,
                "HO transport unstable at step {}: rms={}",
                step + 1,
                rms(&u2)
            );
        }
    }

    /// The exact linear extension must be a steady state of the transport:
    /// u = x, n = (−1,0) ⇒ n·∇u = −1 and ‖M⁻¹(Ku + M·n·∇u)‖ ≈ 0.
    #[test]
    fn linear_extension_is_steady_state() {
        let n = 8_usize;
        let mesh = Mesh::<2>::unit_square_tri(n);
        let order = 1_u8;
        let p = order as usize;

        let pfes_h1 = H1Space::new(mesh.clone(), order);
        let ls = |x: &[f64]| x[0] - 0.5;
        let ls_dofs: Vec<f64> = pfes_h1.interpolate(&ls).as_slice().to_vec();
        let _ = ls_dofs;
        // Constant projected normal n = (−1, 0).
        let n_dofs = pfes_h1.n_dofs();
        let ls_n_coeff = NormalGfCoeff {
            space: &pfes_h1,
            dofs: (0..2 * n_dofs).map(|i| if i < n_dofs { -1.0 } else { 0.0 }).collect(),
        };

        let pfes_l2 = L2Space::new(mesh.clone(), order);
        // u = x (the exact extension).
        let u = project_disc_average(&pfes_l2, |_e, _xi, x| x[0]);

        let mass = Assembler::assemble_bilinear(&pfes_l2, &[&MassIntegrator { rho: 1.0 }], (2 * p) as u8);
        let mut coo = CooMatrix::<f64>::new(pfes_l2.n_dofs(), pfes_l2.n_dofs());
        assemble_convection_volume(&mesh, &pfes_l2, &ls_n_coeff, (2 * p - 1) as u8, &mut coo);
        let ifl = InteriorFaceList::build(&mesh);
        assemble_nonconservative_faces(&mesh, &pfes_l2, &ls_n_coeff, &ifl, (2 * p) as u8, &mut coo);
        let k = coo.into_csr();

        let n_grad_u: Vec<f64> = (0..mesh.n_elements())
            .flat_map(|_| std::iter::repeat(-1.0).take(pfes_l2.element_dofs(0).len()))
            .collect();
        let mut ku = vec![0.0_f64; pfes_l2.n_dofs()];
        k.spmv(&u, &mut ku);
        let mut mngu = vec![0.0_f64; pfes_l2.n_dofs()];
        mass.spmv(&n_grad_u, &mut mngu);
        // dx = M⁻¹(Ku + M·(n·∇u)) — solve with the block-diagonal mass.
        let mut res = vec![0.0_f64; pfes_l2.n_dofs()];
        for i in 0..pfes_l2.n_dofs() {
            res[i] = ku[i] + mngu[i];
        }
        // Block-diagonal solve: mass is block diagonal for L2; use row-scaling
        // (Jacobi) as a proxy — enough to reveal order-of-magnitude mismatch.
        let diag = mass.diagonal();
        let dx_rms = (res.iter().zip(diag.iter()).map(|(r, &d)| (r / d) * (r / d)).sum::<f64>()
            / res.len() as f64)
            .sqrt();
        // Both contributions must vanish identically for the exact linear
        // field: the operator is a strongly consistent discretization of
        // −(n·∇), so the linear extension is an exact steady state.
        assert!(dx_rms < 1e-10, "linear extension is NOT a steady state: {dx_rms}");

        // Single-element structural check: on tri1 = (0,0),(1,0),(1,1) with
        // n = (−1,0), K_ij = ∫ φ_i ∂λ_j/∂x gives the exact entries
        // (−|T|/3, +|T|/3, 0) per row.
        {
            let mesh1 = Mesh::<2>::unit_square_tri(1);
            let h1 = H1Space::new(mesh1.clone(), 1);
            let n1 = h1.n_dofs();
            let l1 = NormalGfCoeff {
                space: &h1,
                dofs: (0..2 * n1).map(|i| if i < n1 { -1.0 } else { 0.0 }).collect(),
            };
            let l2s = L2Space::new(mesh1.clone(), 1);
            let mut coo1 = CooMatrix::<f64>::new(l2s.n_dofs(), l2s.n_dofs());
            assemble_convection_volume(&mesh1, &l2s, &l1, 2, &mut coo1);
            let kv1 = coo1.into_csr();
            assert_eq!(kv1.nnz(), 18, "expected 6x6 dense-block structure");
            // Triangle 1 (dofs 0..3, vertices (0,0),(1,0),(1,1)) has |T| = 0.5
            // and vertex x-gradients (−1, +1, 0), so its 3x3 block is exactly
            // (|T|/3)·[(−1, +1, 0) per row].
            let expected: [[f64; 3]; 2] = [
                [-1.0 / 6.0, 1.0 / 6.0, 0.0],
                [0.0, 1.0 / 6.0, -1.0 / 6.0],
            ];
            for (e, row_base) in [(0_usize, 0_usize), (1_usize, 3_usize)] {
                for (j, col) in (row_base..row_base + 3).enumerate() {
                    let got = kv1.get(row_base, col);
                    assert!(
                        (got - expected[e][j]).abs() < 1e-14,
                        "K_vol[{row_base},{col}] = {got}, expected {}",
                        expected[e][j]
                    );
                }
            }
            // Partition of unity: every row sum must vanish (n·∇1 = 0).
            for row in 0..kv1.nrows {
                let rs: f64 = (kv1.row_ptr[row]..kv1.row_ptr[row + 1])
                    .map(|idx| kv1.values[idx])
                    .sum();
                assert!(rs.abs() < 1e-14, "row sum {rs} for row {row}");
            }
        }
    }
}

#[cfg(test)]
mod dof_points_probe {
    /// Probe elem_dof_points on a single triangle.
    #[test]
    fn probe_dof_points() {
        use crate::dist_solver::elem_dof_points;
        use fem_mesh::Mesh;
        use fem_space::L2Space;

        let mesh = Mesh::<2>::unit_square_tri(1);
        let l2 = L2Space::new(mesh.clone(), 1);
        let pts = elem_dof_points(&l2, 0);
        println!("elem0 dof points: {:?}", pts);
        let h1 = fem_space::H1Space::new(mesh, 1);
        let pts1 = elem_dof_points(&h1, 0);
        println!("h1 elem0 dof points: {:?}", pts1);
    }
}
