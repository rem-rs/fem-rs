//! TMOP mesh-optimization nonlinear form and Newton solver.
//!
//! 1:1 port of the MFEM serial machinery used by `miniapps/meshing/mesh-optimizer`:
//! - `fem/tmop.cpp`: `TMOP_Integrator` (LEGACY full assembly: `GetElementEnergy`,
//!   `AssembleElementVectorExact`, `AssembleElementGradExact`) and
//!   `TargetConstructor::ComputeElementTargets` (ideal-shape target types).
//! - `fem/tmop_tools.cpp`: `TMOPNewtonSolver` (Newton iteration, energy/min-det
//!   line search via `ComputeScalingFactor`, shared `min_det` for untangling).
//!
//! Scope (v1): serial, full (matrix) assembly, quad/hex meshes with Qk nodal
//! geometry, metric ids {1, 2, 7, 9, 14, 22, 50, 55, 56, 58, 77} in 2D and
//! {301, 302, 303, 304, 315, 316, 318, 321, 323, 360} in 3D, target types
//! {IDEAL_SHAPE_UNIT_SIZE, IDEAL_SHAPE_EQUAL_SIZE, IDEAL_SHAPE_GIVEN_SIZE}.
//! Limiting / adaptive limiting / normalization / discrete-adaptivity targets
//! are NOT implemented and rejected by the driver.

use crate::assembler::ref_elem_vol_h1;
use fem_element::quadrature::{gauss_lobatto_01_arbitrary, gauss_lobatto_arbitrary};
use fem_element::reference::ReferenceElement;
use fem_linalg::{CooMatrix, CsrMatrix, SolveResult, SolverConfig};
use fem_mesh::element_type::ElementType;
use fem_mesh::tmop::metrics::{
    TmopMetric001, TmopMetric002, TmopMetric007, TmopMetric009, TmopMetric014, TmopMetric022,
    TmopMetric050, TmopMetric055, TmopMetric056, TmopMetric058, TmopMetric077, TmopMetric301,
    TmopMetric302, TmopMetric303, TmopMetric304, TmopMetric315, TmopMetric316, TmopMetric318,
    TmopMetric321, TmopMetric323, TmopMetric360, TmopQualityMetric, TmopQualityMetric3D,
};
use fem_mesh::topology::MeshTopology;
use fem_solver::{solve_cg, solve_minres, solve_minres_jacobi, solve_minres_precond};
use fem_space::dof_manager::DofManager;
use std::cell::Cell;
use std::rc::Rc;

/// Target-matrix construction type (MFEM `TargetConstructor::TargetType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TmopTargetType {
    /// IDEAL_SHAPE_UNIT_SIZE (target_id 1).
    IdealShapeUnitSize,
    /// IDEAL_SHAPE_EQUAL_SIZE (target_id 2).
    IdealShapeEqualSize,
    /// IDEAL_SHAPE_GIVEN_SIZE (target_id 3): target size from the initial mesh.
    IdealShapeGivenSize,
}

/// Target constructor state for one integrator (MFEM `TargetConstructor`).
#[derive(Clone)]
pub struct TmopTarget {
    pub target_type: TmopTargetType,
    /// MFEM `SetVolumeScale` (IDEAL_SHAPE_EQUAL_SIZE).
    pub volume_scale: f64,
    /// MFEM `ComputeAvgVolume`: total physical volume / NE (from `x0`).
    pub avg_volume: f64,
}

impl TmopTarget {
    pub fn new(target_type: TmopTargetType) -> Self {
        Self {
            target_type,
            volume_scale: 1.0,
            avg_volume: 0.0,
        }
    }
}

/// Quality metric + constant coefficient, dimension-tagged (MFEM
/// `TMOP_QualityMetric` held by a `TMOP_Integrator`).
pub enum TmopMetric {
    D2(Box<dyn TmopQualityMetric>),
    D3(Box<dyn TmopQualityMetric3D>),
}

/// Shared `min_detJ` cell for untangling metrics (MFEM passes `real_t&` into
/// `TMOP_Metric_022`; Rust gets an `Rc<Cell<f64>>`).
#[derive(Clone)]
pub struct SharedMinDet(Rc<Cell<f64>>);

impl SharedMinDet {
    pub fn new(value: f64) -> Self {
        Self(Rc::new(Cell::new(value)))
    }
    pub fn get(&self) -> f64 {
        self.0.get()
    }
    pub fn set(&self, value: f64) {
        self.0.set(value);
    }
}

/// `TMOP_Metric_022` wrapper reading tau0 from the shared min-det cell, so the
/// solver's `SetMinDetPtr` updates are visible to the metric exactly like the
/// C++ reference binding `real_t &min_detT`.
#[derive(Clone)]
pub struct Metric22Shared {
    tau0: SharedMinDet,
}

impl TmopQualityMetric for Metric22Shared {
    fn eval_w(&self, jpt: &[[f64; 2]; 2]) -> f64 {
        let t = |c: usize, r: usize| jpt[r][c];
        let i1 = t(0, 0) * t(0, 0) + t(1, 0) * t(1, 0) + t(0, 1) * t(0, 1) + t(1, 1) * t(1, 1);
        let i2b = t(0, 0) * t(1, 1) - t(1, 0) * t(0, 1);
        let min_det_t = self.tau0.get();
        let mut d = i2b - min_det_t;
        if d < 0.0 && min_det_t == 0.0 {
            d = -i2b * 0.1;
        }
        (0.5 * i1 - i2b) / d
    }

    fn eval_p(&self, jpt: &[[f64; 2]; 2], p: &mut [[f64; 2]; 2]) {
        let m = TmopMetric022 {
            min_det_t: self.tau0.get(),
        };
        m.eval_p(jpt, p);
    }

    fn assemble_h(&self, jpt: &[[f64; 2]; 2], ds: &[[f64; 2]], weight: f64, a: &mut [f64]) {
        let m = TmopMetric022 {
            min_det_t: self.tau0.get(),
        };
        m.assemble_h(jpt, ds, weight, a);
    }

    fn id(&self) -> i32 {
        22
    }
}

/// Build a metric from a mesh-optimizer `-mid` id. Returns `None` for ids the
/// metric zoo does not cover.
pub fn metric_from_id_2d(id: i32, min_det: &SharedMinDet) -> Option<TmopMetric> {
    let m = match id {
        1 => TmopMetric::D2(Box::new(TmopMetric001)),
        2 => TmopMetric::D2(Box::new(TmopMetric002)),
        7 => TmopMetric::D2(Box::new(TmopMetric007)),
        9 => TmopMetric::D2(Box::new(TmopMetric009)),
        14 => TmopMetric::D2(Box::new(TmopMetric014)),
        22 => TmopMetric::D2(Box::new(Metric22Shared { tau0: min_det.clone() })),
        50 => TmopMetric::D2(Box::new(TmopMetric050)),
        55 => TmopMetric::D2(Box::new(TmopMetric055)),
        56 => TmopMetric::D2(Box::new(TmopMetric056)),
        58 => TmopMetric::D2(Box::new(TmopMetric058)),
        77 => TmopMetric::D2(Box::new(TmopMetric077)),
        _ => return None,
    };
    Some(m)
}

pub fn metric_from_id_3d(id: i32, _min_det: &SharedMinDet) -> Option<TmopMetric> {
    let m = match id {
        301 => TmopMetric::D3(Box::new(TmopMetric301)),
        302 => TmopMetric::D3(Box::new(TmopMetric302)),
        303 => TmopMetric::D3(Box::new(TmopMetric303)),
        304 => TmopMetric::D3(Box::new(TmopMetric304)),
        315 => TmopMetric::D3(Box::new(TmopMetric315)),
        316 => TmopMetric::D3(Box::new(TmopMetric316)),
        318 => TmopMetric::D3(Box::new(TmopMetric318)),
        321 => TmopMetric::D3(Box::new(TmopMetric321)),
        323 => TmopMetric::D3(Box::new(TmopMetric323)),
        360 => TmopMetric::D3(Box::new(TmopMetric360)),
        _ => return None,
    };
    Some(m)
}

/// Surface fitting to prescribed node positions (MFEM
/// `TMOP_Integrator::EnableSurfaceFitting(pos, smarker, coeff)`, the third
/// overload in fem/tmop.hpp). It adds the term
/// `sum_{i in S} c * 1/2 * (x_i - x_{t,i})^2` over the marked dofs S, with the
/// per-dof weight divided by the number of elements sharing the dof (MFEM
/// `surf_fit_dof_count` from `GridFunction::CountElementsPerVDof`). The
/// gradient/Hessian of the term are the exact derivatives of the quadratic
/// (`TMOP_QuadraticLimiter` with dist = 1).
#[derive(Clone)]
pub struct SurfFitPos {
    /// Target positions `x_t`, layout `[c*n_scalar + s]` (MFEM `surf_fit_pos`,
    /// Ordering::byNODES).
    pub pos: Rc<Vec<f64>>,
    /// Marked scalar dofs (MFEM `surf_fit_marker`).
    pub marker: Rc<Vec<bool>>,
    /// Element-sharing count per scalar dof (MFEM `surf_fit_dof_count`, the
    /// count is identical for every component of a dof).
    pub dof_count: Rc<Vec<f64>>,
    /// Fitting weight `c` (MFEM `surf_fit_coeff`, a ConstantCoefficient mutated
    /// by the adaptive surface fitting of `TMOPNewtonSolver`).
    pub coeff: Rc<Cell<f64>>,
}

impl SurfFitPos {
    pub fn new(
        pos: Vec<f64>,
        marker: Vec<bool>,
        dof_count: Vec<f64>,
        coeff: f64,
    ) -> Self {
        Self {
            pos: Rc::new(pos),
            marker: Rc::new(marker),
            dof_count: Rc::new(dof_count),
            coeff: Rc::new(Cell::new(coeff)),
        }
    }
}

/// MFEM `GridFunction::CountElementsPerVDof`: for every scalar dof, the number
/// of elements whose dof tables reference it.
pub fn count_elements_per_dof(topo: &dyn MeshTopology, dm: &DofManager) -> Vec<f64> {
    let mut count = vec![0.0_f64; dm.n_dofs];
    for e in 0..topo.n_elements() {
        for &dof in dm.element_dofs(e as u32) {
            count[dof as usize] += 1.0;
        }
    }
    count
}

/// One `TMOP_Integrator`: metric + target (+ constant metric coefficient).
pub struct TmopIntegrator {
    pub metric: TmopMetric,
    pub target: TmopTarget,
    /// MFEM `SetCoefficient` (ConstantCoefficient); 1.0 when unset.
    pub coeff: f64,
    /// MFEM `surf_fit_pos`/`surf_fit_marker`/`surf_fit_coeff` when surface
    /// fitting to prescribed positions is enabled (None otherwise).
    pub surf_fit: Option<SurfFitPos>,
}

/// Per-element cached data: scalar dofs, reference element, quadrature rule.
struct TmopElemData {
    edofs: Vec<usize>,
    re: Box<dyn ReferenceElement>,
    /// (point[p][d], weight[p]).
    quad_points: Vec<Vec<f64>>,
    quad_weights: Vec<f64>,
}

/// Quadrature family (mesh-optimizer `-qt` / MFEM `IntegrationRules`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TmopQuadType {
    /// GaussLobatto (`IntRulesLo`) — tensor-product, exact on quad/hex.
    GaussLobatto,
    /// GaussLegendre (`IntRules`) — from the reference element.
    GaussLegendre,
}

/// The TMOP nonlinear form: E(dx) = Σ_integ ∫ μ(Jpt) over the mesh, with
/// element positions x = x0 + dx. Global layout: byNODES (`c*n_scalar + s`),
/// matching MFEM `PMatI(i, d) = elfun[i + d*dof]` per element.
pub struct TmopForm<'a> {
    topo: &'a dyn MeshTopology,
    dm: &'a DofManager,
    n_scalar: usize,
    dim: usize,
    order: u8,
    elems: Vec<TmopElemData>,
    integrators: Vec<TmopIntegrator>,
    /// Initial positions, len = dim * n_scalar, layout [c*n_scalar + s].
    x0: Vec<f64>,
    /// Global ess vdofs (gradient zeroed, Hessian eliminated here).
    pub ess_vdofs: Vec<usize>,
}

impl<'a> TmopForm<'a> {
    /// Build element data for a quad(2D)/hex(3D) mesh with a Qk nodal space of
    /// the given order.
    pub fn new(
        topo: &'a dyn MeshTopology,
        dm: &'a DofManager,
        order: u8,
        quad_type: TmopQuadType,
        quad_order: u8,
    ) -> Self {
        let dim = topo.dim() as usize;
        let n_scalar = dm.n_dofs;
        let mut elems = Vec::with_capacity(topo.n_elements() as usize);
        for e in 0..topo.n_elements() {
            let e = e as u32;
            let et = topo.element_type(e);
            let raw_re = ref_elem_vol_h1(et, order);
            // Normalize the reference domain to MFEM's [0,1]^d convention.
            let re: Box<dyn ReferenceElement> = if el_domain_is_unit(raw_re.as_ref()) {
                raw_re
            } else {
                Box::new(UnitDomainElem {
                    inner: raw_re,
                    dim,
                })
            };
            let (points, weights) = match (quad_type, dim) {
                (TmopQuadType::GaussLegendre, _) => {
                    let rule = re.quadrature(quad_order);
                    (rule.points, rule.weights)
                }
                (TmopQuadType::GaussLobatto, 2) | (TmopQuadType::GaussLobatto, 3) => {
                    // MFEM Quadrature1D::GaussLobatto: n = Order/2 + 2 points
                    // per dimension (exact for degree 2n-3), tensor product on
                    // the reference element. The rule domain must match the
                    // reference element's domain ([0,1] for QuadQk, [-1,1]
                    // for HexQk).
                    let n = (quad_order as usize) / 2 + 2;
                    let domain_unit = el_domain_is_unit(re.as_ref());
                    let (xs, ws) = if domain_unit {
                        gauss_lobatto_01_arbitrary(n)
                    } else {
                        gauss_lobatto_arbitrary(n)
                    };
                    let mut points = Vec::with_capacity(xs.len().pow(dim as u32));
                    let mut weights = Vec::with_capacity(xs.len().pow(dim as u32));
                    match dim {
                        2 => {
                            for (xi, wi) in xs.iter().zip(ws.iter()) {
                                for (xj, wj) in xs.iter().zip(ws.iter()) {
                                    points.push(vec![*xi, *xj]);
                                    weights.push(wi * wj);
                                }
                            }
                        }
                        3 => {
                            for (xi, wi) in xs.iter().zip(ws.iter()) {
                                for (xj, wj) in xs.iter().zip(ws.iter()) {
                                    for (xk, wk) in xs.iter().zip(ws.iter()) {
                                        points.push(vec![*xi, *xj, *xk]);
                                        weights.push(wi * wj * wk);
                                    }
                                }
                            }
                        }
                        _ => unreachable!(),
                    }
                    (points, weights)
                }
                _ => unreachable!("dim checked above"),
            };
            let edofs = dm
                .element_dofs(e)
                .iter()
                .map(|&d| d as usize)
                .collect::<Vec<_>>();
            elems.push(TmopElemData {
                edofs,
                re,
                quad_points: points,
                quad_weights: weights,
            });
        }
        Self {
            topo,
            dm,
            n_scalar,
            dim,
            order,
            elems,
            integrators: Vec::new(),
            x0: vec![0.0; dim * n_scalar],
            ess_vdofs: Vec::new(),
        }
    }

    pub fn order(&self) -> u8 {
        self.order
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn n_dofs(&self) -> usize {
        self.dim * self.n_scalar
    }

    /// Set the initial positions `x0` (layout `[c*n_scalar + s]`).
    pub fn set_x0(&mut self, x0: Vec<f64>) {
        assert_eq!(x0.len(), self.dim * self.n_scalar);
        self.x0 = x0;
    }

    pub fn x0(&self) -> &[f64] {
        &self.x0
    }

    pub fn push_integrator(&mut self, integ: TmopIntegrator) {
        self.integrators.push(integ);
    }

    /// MFEM `TargetConstructor::ComputeAvgVolume` for every integrator whose
    /// target needs it: total physical volume of the `x0` mesh / NE, using a
    /// Gauss-Legendre rule of order `2*order` (exact for the polynomial det).
    pub fn finalize_targets(&mut self) {
        let ne = self.elems.len();
        let gl_order = 2 * self.order;
        for integ in &mut self.integrators {
            if integ.target.avg_volume == 0.0
                && matches!(
                    integ.target.target_type,
                    TmopTargetType::IdealShapeEqualSize | TmopTargetType::IdealShapeGivenSize
                )
            {
                let mut volume = 0.0;
                for el in &self.elems {
                    let nd = el.edofs.len();
                    let mut pos = vec![0.0; nd * self.dim];
                    for (k, &dof) in el.edofs.iter().enumerate() {
                        for c in 0..self.dim {
                            pos[k + c * nd] = self.x0[c * self.n_scalar + dof];
                        }
                    }
                    // Gauss-Legendre rule on the reference element.
                    let rule = el.re.quadrature(gl_order);
                    let mut dsh = vec![0.0; nd * self.dim];
                    for (q, xi) in rule.points.iter().enumerate() {
                        el.re.eval_grad_basis(xi, &mut dsh);
                        let mut jpr = [[0.0f64; 3]; 3];
                        for a in 0..self.dim {
                            for b in 0..self.dim {
                                let mut s = 0.0;
                                for i in 0..nd {
                                    s += pos[i + a * nd] * dsh[i * self.dim + b];
                                }
                                jpr[a][b] = s;
                            }
                        }
                        volume += rule.weights[q] * det_small(&jpr, self.dim);
                    }
                }
                integ.target.avg_volume = volume / ne as f64;
            }
        }
    }

    /// Per-element target Jacobians Jtr (column-major) at every quadrature
    /// point. `Wideal` = identity for quad/hex (MFEM
    /// `Geometries.GetGeomToPerfGeomJac`); tri/tet meshes are rejected in
    /// `TmopForm::new` via `ref_elem_for`.
    fn compute_element_targets(
        &self,
        integ: &TmopIntegrator,
        el: &TmopElemData,
        jtr_out: &mut Vec<[f64; 4]>,
        jtr_out_3d: &mut Vec<[f64; 9]>,
        dsh_cache: &mut Vec<f64>,
    ) {
        let dim = self.dim;
        let nd = el.edofs.len();
        jtr_out.clear();
        jtr_out_3d.clear();
        match integ.target.target_type {
            TmopTargetType::IdealShapeUnitSize => {
                // Jtr = I.
                for _ in 0..el.quad_points.len() {
                    if dim == 2 {
                        jtr_out.push([1.0, 0.0, 0.0, 1.0]);
                    } else {
                        jtr_out_3d.push([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
                    }
                }
            }
            TmopTargetType::IdealShapeEqualSize => {
                // W = c * I, c = (volume_scale * el_volume / det(W))^(1/dim),
                // el_volume = avg_volume (serial, no NC).
                assert!(integ.target.avg_volume > 0.0, "call finalize_targets");
                let c = (integ.target.volume_scale * integ.target.avg_volume).powf(1.0 / dim as f64);
                let nqp = el.quad_points.len();
                for _ in 0..nqp {
                    if dim == 2 {
                        jtr_out.push([c, 0.0, 0.0, c]);
                    } else {
                        jtr_out_3d.push([c, 0.0, 0.0, 0.0, c, 0.0, 0.0, 0.0, c]);
                    }
                }
            }
            TmopTargetType::IdealShapeGivenSize => {
                // Jtr(q) = (det(Jpr_x0(q)) / det(W))^(1/dim) * W, W = I. The
                // initial-size field comes from the x0 positions (MFEM uses
                // the `nodes` grid function set via SetNodes(x0)).
                let nqp = el.quad_points.len();
                dsh_cache.resize(nd * dim, 0.0);
                let mut pos = vec![0.0; nd * dim];
                for (k, &dof) in el.edofs.iter().enumerate() {
                    for c in 0..dim {
                        pos[k + c * nd] = self.x0[c * self.n_scalar + dof];
                    }
                }
                for q in 0..nqp {
                    el.re.eval_grad_basis(&el.quad_points[q], dsh_cache);
                    let mut jpr = [[0.0f64; 3]; 3];
                    for a in 0..dim {
                        for b in 0..dim {
                            let mut s = 0.0;
                            for i in 0..nd {
                                s += pos[i + a * nd] * dsh_cache[i * dim + b];
                            }
                            jpr[a][b] = s;
                        }
                    }
                    let det = det_small(&jpr, dim);
                    assert!(det > 0.0, "The given mesh is inverted!");
                    let c = det.powf(1.0 / dim as f64);
                    if dim == 2 {
                        jtr_out.push([c, 0.0, 0.0, c]);
                    } else {
                        jtr_out_3d.push([c, 0.0, 0.0, 0.0, c, 0.0, 0.0, 0.0, c]);
                    }
                }
            }
        }
    }

    /// Gather element positions for displacement `dx`: `pos[k + c*nd] =
    /// x0[c*n_scalar + dof] + dx[c*n_scalar + dof]`.
    fn element_positions(&self, el: &TmopElemData, dx: &[f64], pos: &mut Vec<f64>) {
        let nd = el.edofs.len();
        pos.clear();
        pos.resize(nd * self.dim, 0.0);
        for (k, &dof) in el.edofs.iter().enumerate() {
            for c in 0..self.dim {
                pos[k + c * nd] = self.x0[c * self.n_scalar + dof] + dx[c * self.n_scalar + dof];
            }
        }
    }

    /// MFEM `TMOP_Integrator::GetElementEnergy` (LEGACY, integ_over_target).
    /// Mirrors `NonlinearForm::GetGridFunctionEnergy`: one accumulator per
    /// element, added to the total element-by-element (same summation tree as
    /// the C++, so the line search sees bit-identical energies).
    pub fn energy(&self, dx: &[f64]) -> f64 {
        let dim = self.dim;
        let mut total = 0.0;
        let mut pos = Vec::new();
        let mut dsh = Vec::new();
        let mut jtr2: Vec<[f64; 4]> = Vec::new();
        let mut jtr3: Vec<[f64; 9]> = Vec::new();
        for el in &self.elems {
            let nd = el.edofs.len();
            let nqp = el.quad_points.len();
            self.element_positions(el, dx, &mut pos);
            let mut ds = vec![0.0; nd * dim];
            dsh.resize(nd * dim, 0.0);
            let mut energy = 0.0;
            for integ in &self.integrators {
                self.compute_element_targets(integ, el, &mut jtr2, &mut jtr3, &mut dsh);
                for q in 0..nqp {
                    // weight = ip.weight * det(Jtr) (integ_over_target).
                    let weight = el.quad_weights[q]
                        * if dim == 2 { det_jtr2(&jtr2[q]) } else { det_jtr3(&jtr3[q]) };
                    el.re.eval_grad_basis(&el.quad_points[q], &mut dsh);
                    // Jrt = Jtr^{-1}, Jpt = Jpr * Jrt.
                    let mut jpt_cm = [0.0f64; 9];
                    match (&integ.metric, dim) {
                        (TmopMetric::D2(m), 2) => {
                            let jrt = invert_2x2(&jtr2[q]);
                            // DS = DSh * Jrt (dof x dim, column-major).
                            for i in 0..nd {
                                for d in 0..2 {
                                    let mut s = 0.0;
                                    for j in 0..2 {
                                        s += dsh[i * 2 + j] * jrt[j + d * 2];
                                    }
                                    ds[i + d * nd] = s;
                                }
                            }
                            // Jpt = PMatI^T * DS.
                            for a in 0..2 {
                                for b in 0..2 {
                                    let mut s = 0.0;
                                    for i in 0..nd {
                                        s += pos[i + a * nd] * ds[i + b * nd];
                                    }
                                    jpt_cm[a + b * 2] = s;
                                }
                            }
                            let jpt = [[jpt_cm[0], jpt_cm[2]], [jpt_cm[1], jpt_cm[3]]];
                            energy += weight * integ.coeff * m.eval_w(&jpt);
                        }
                        (TmopMetric::D3(m), 3) => {
                            let jrt = invert_3x3(&jtr3[q]);
                            for i in 0..nd {
                                for d in 0..3 {
                                    let mut s = 0.0;
                                    for j in 0..3 {
                                        s += dsh[i * 3 + j] * jrt[j + d * 3];
                                    }
                                    ds[i + d * nd] = s;
                                }
                            }
                            for a in 0..3 {
                                for b in 0..3 {
                                    let mut s = 0.0;
                                    for i in 0..nd {
                                        s += pos[i + a * nd] * ds[i + b * nd];
                                    }
                                    jpt_cm[a + b * 3] = s;
                                }
                            }
                            let jpt = [
                                [jpt_cm[0], jpt_cm[3], jpt_cm[6]],
                                [jpt_cm[1], jpt_cm[4], jpt_cm[7]],
                                [jpt_cm[2], jpt_cm[5], jpt_cm[8]],
                            ];
                            energy += weight * integ.coeff * m.eval_w(&jpt);
                        }
                        _ => panic!("metric dimension mismatch"),
                    }
                }
                // Contribution from the surface fitting term (MFEM
                // GetElementEnergy, the `if (surface_fit)` block): evaluated at
                // the FE nodal points, one term per marked dof.
                if let Some(sf) = &integ.surf_fit {
                    for (k, &dof) in el.edofs.iter().enumerate() {
                        if !sf.marker[dof] {
                            continue;
                        }
                        let w = sf.coeff.get() / sf.dof_count[dof];
                        let mut d2 = 0.0;
                        for c in 0..dim {
                            let d = pos[k + c * nd] - sf.pos[c * self.n_scalar + dof];
                            d2 += d * d;
                        }
                        // TMOP_QuadraticLimiter::Eval = 0.5*d2/dist^2, dist = 1.
                        energy += w * (0.5 * d2);
                    }
                }
            }
            total += energy;
        }
        total
    }

    /// MFEM `TMOP_Integrator::AssembleElementVectorExact`:
    /// r = dE/d(dx); entries at essential vdofs are zeroed.
    pub fn gradient(&self, dx: &[f64], r: &mut [f64]) {
        let dim = self.dim;
        for v in r.iter_mut() {
            *v = 0.0;
        }
        let mut pos = Vec::new();
        let mut dsh = Vec::new();
        let mut jtr2: Vec<[f64; 4]> = Vec::new();
        let mut jtr3: Vec<[f64; 9]> = Vec::new();
        let mut elvec = Vec::new();
        for el in &self.elems {
            let nd = el.edofs.len();
            let nqp = el.quad_points.len();
            self.element_positions(el, dx, &mut pos);
            let mut ds = vec![0.0; nd * dim];
            dsh.resize(nd * dim, 0.0);
            elvec.resize(nd * dim, 0.0);
            // elvec is a reused scratch buffer: `resize` only fills new slots,
            // so zero it explicitly or each element would accumulate the
            // previous elements' contributions.
            for v in elvec.iter_mut() {
                *v = 0.0;
            }
            for integ in &self.integrators {
                self.compute_element_targets(integ, el, &mut jtr2, &mut jtr3, &mut dsh);
                for q in 0..nqp {
                    let weight_m = el.quad_weights[q]
                        * if dim == 2 { det_jtr2(&jtr2[q]) } else { det_jtr3(&jtr3[q]) }
                        * integ.coeff;
                    el.re.eval_grad_basis(&el.quad_points[q], &mut dsh);
                    match (&integ.metric, dim) {
                        (TmopMetric::D2(m), 2) => {
                            let jrt = invert_2x2(&jtr2[q]);
                            // DS = DSh * Jrt.
                            for i in 0..nd {
                                for d in 0..2 {
                                    let mut s = 0.0;
                                    for j in 0..2 {
                                        s += dsh[i * 2 + j] * jrt[j + d * 2];
                                    }
                                    ds[i + d * nd] = s;
                                }
                            }
                            let mut jpt_cm = [0.0f64; 4];
                            for a in 0..2 {
                                for b in 0..2 {
                                    let mut s = 0.0;
                                    for i in 0..nd {
                                        s += pos[i + a * nd] * ds[i + b * nd];
                                    }
                                    jpt_cm[a + b * 2] = s;
                                }
                            }
                            let jpt = [[jpt_cm[0], jpt_cm[2]], [jpt_cm[1], jpt_cm[3]]];
                            let mut p = [[0.0f64; 2]; 2];
                            m.eval_p(&jpt, &mut p);
                            // MFEM: `P *= weight_m;` then `AddMultABt(DS, P,
                            // PMatO)` — the weight is folded into P before the
                            // contraction, which runs k-outer / j / i-inner,
                            // accumulating each product directly into elvect.
                            for row in p.iter_mut() {
                                for v in row.iter_mut() {
                                    *v *= weight_m;
                                }
                            }
                            for k in 0..2 {
                                for j in 0..2 {
                                    let bjk = p[j][k];
                                    for i in 0..nd {
                                        elvec[i + j * nd] += ds[i + k * nd] * bjk;
                                    }
                                }
                            }
                        }
                        (TmopMetric::D3(m), 3) => {
                            let jrt = invert_3x3(&jtr3[q]);
                            for i in 0..nd {
                                for d in 0..3 {
                                    let mut s = 0.0;
                                    for j in 0..3 {
                                        s += dsh[i * 3 + j] * jrt[j + d * 3];
                                    }
                                    ds[i + d * nd] = s;
                                }
                            }
                            let mut jpt_cm = [0.0f64; 9];
                            for a in 0..3 {
                                for b in 0..3 {
                                    let mut s = 0.0;
                                    for i in 0..nd {
                                        s += pos[i + a * nd] * ds[i + b * nd];
                                    }
                                    jpt_cm[a + b * 3] = s;
                                }
                            }
                            let jpt = [
                                [jpt_cm[0], jpt_cm[3], jpt_cm[6]],
                                [jpt_cm[1], jpt_cm[4], jpt_cm[7]],
                                [jpt_cm[2], jpt_cm[5], jpt_cm[8]],
                            ];
                            let mut p = [[0.0f64; 3]; 3];
                            m.eval_p(&jpt, &mut p);
                            // MFEM: `P *= weight_m;` then AddMultABt (see 2D).
                            for row in p.iter_mut() {
                                for v in row.iter_mut() {
                                    *v *= weight_m;
                                }
                            }
                            for k in 0..3 {
                                for j in 0..3 {
                                    let bjk = p[j][k];
                                    for i in 0..nd {
                                        elvec[i + j * nd] += ds[i + k * nd] * bjk;
                                    }
                                }
                            }
                        }
                        _ => panic!("metric dimension mismatch"),
                    }
                }
                // MFEM `AssembleElemVecSurfFit`: elvect(s, d) += w * (x_s -
                // x_{t,s})_d at the nodal point of every marked dof s
                // (TMOP_QuadraticLimiter::Eval_d1, dist = 1).
                if let Some(sf) = &integ.surf_fit {
                    for (k, &dof) in el.edofs.iter().enumerate() {
                        if !sf.marker[dof] {
                            continue;
                        }
                        let w = sf.coeff.get() / sf.dof_count[dof];
                        for c in 0..dim {
                            elvec[k + c * nd] += w * (pos[k + c * nd] - sf.pos[c * self.n_scalar + dof]);
                        }
                    }
                }
            }
            // Scatter into the global gradient (layout [c*n_scalar + dof]).
            for (k, &dof) in el.edofs.iter().enumerate() {
                for c in 0..dim {
                    r[c * self.n_scalar + dof] += elvec[k + c * nd];
                }
            }
        }
        for &dof in &self.ess_vdofs {
            r[dof] = 0.0;
        }
    }

    /// MFEM `TMOP_Integrator::AssembleElementGradExact`: Hessian with essential
    /// rows/columns eliminated (diagonal = 1), i.e. `NonlinearForm::GetGradient`.
    pub fn hessian(&self, dx: &[f64]) -> CsrMatrix<f64> {
        let dim = self.dim;
        let n = self.dim * self.n_scalar;
        let mut coo = CooMatrix::<f64>::new(n, n);
        let mut pos = Vec::new();
        let mut dsh = Vec::new();
        let mut jtr2: Vec<[f64; 4]> = Vec::new();
        let mut jtr3: Vec<[f64; 9]> = Vec::new();
        for el in &self.elems {
            let nd = el.edofs.len();
            let ah = nd * dim;
            let nqp = el.quad_points.len();
            self.element_positions(el, dx, &mut pos);
            let mut ds = vec![0.0; nd * dim];
            dsh.resize(nd * dim, 0.0);
            // MFEM keeps one dense element matrix per element, accumulated over
            // all quadrature points (and the surface fitting term), and adds it
            // into the sparse matrix once; mirror that exactly so the
            // floating-point sums (and hence the Newton path) match the C++.
            let mut elmat = vec![0.0f64; ah * ah];
            for integ in &self.integrators {
                self.compute_element_targets(integ, el, &mut jtr2, &mut jtr3, &mut dsh);
                for q in 0..nqp {
                    let weight_m = el.quad_weights[q]
                        * if dim == 2 { det_jtr2(&jtr2[q]) } else { det_jtr3(&jtr3[q]) }
                        * integ.coeff;
                    el.re.eval_grad_basis(&el.quad_points[q], &mut dsh);
                    match (&integ.metric, dim) {
                        (TmopMetric::D2(m), 2) => {
                            let jrt = invert_2x2(&jtr2[q]);
                            for i in 0..nd {
                                for d in 0..2 {
                                    let mut s = 0.0;
                                    for j in 0..2 {
                                        s += dsh[i * 2 + j] * jrt[j + d * 2];
                                    }
                                    ds[i + d * nd] = s;
                                }
                            }
                            let mut jpt_cm = [0.0f64; 4];
                            for a in 0..2 {
                                for b in 0..2 {
                                    let mut s = 0.0;
                                    for i in 0..nd {
                                        s += pos[i + a * nd] * ds[i + b * nd];
                                    }
                                    jpt_cm[a + b * 2] = s;
                                }
                            }
                            let jpt = [[jpt_cm[0], jpt_cm[2]], [jpt_cm[1], jpt_cm[3]]];
                            // assemble_h expects DS rows as per-dof pairs.
                            let ds_rows: Vec<[f64; 2]> =
                                (0..nd).map(|i| [ds[i], ds[i + nd]]).collect();
                            m.assemble_h(&jpt, &ds_rows, weight_m, &mut elmat);
                        }
                        (TmopMetric::D3(m), 3) => {
                            let jrt = invert_3x3(&jtr3[q]);
                            for i in 0..nd {
                                for d in 0..3 {
                                    let mut s = 0.0;
                                    for j in 0..3 {
                                        s += dsh[i * 3 + j] * jrt[j + d * 3];
                                    }
                                    ds[i + d * nd] = s;
                                }
                            }
                            let mut jpt_cm = [0.0f64; 9];
                            for a in 0..3 {
                                for b in 0..3 {
                                    let mut s = 0.0;
                                    for i in 0..nd {
                                        s += pos[i + a * nd] * ds[i + b * nd];
                                    }
                                    jpt_cm[a + b * 3] = s;
                                }
                            }
                            let jpt = [
                                [jpt_cm[0], jpt_cm[3], jpt_cm[6]],
                                [jpt_cm[1], jpt_cm[4], jpt_cm[7]],
                                [jpt_cm[2], jpt_cm[5], jpt_cm[8]],
                            ];
                            let ds_rows: Vec<[f64; 3]> = (0..nd)
                                .map(|i| [ds[i], ds[i + nd], ds[i + 2 * nd]])
                                .collect();
                            m.assemble_h(&jpt, &ds_rows, weight_m, &mut elmat);
                        }
                        _ => panic!("metric dimension mismatch"),
                    }
                }
                // MFEM `AssembleElemGradSurfFit`: for every marked dof s,
                // mat(s+c1*nd, s+c1*nd) += w with w = coeff/count (the quadratic
                // limiter's Eval_d2 = I with dist = 1). NOTE: like MFEM, the
                // outer product of the fitting gradient is deliberately omitted
                // here (its `surf_fit_pos` branch zeroes surf_fit_grad_e and
                // keeps only the limiter Hessian), so the assembled Hessian of
                // the fitting term is the constant w*I.
                if let Some(sf) = &integ.surf_fit {
                    for (k, &dof) in el.edofs.iter().enumerate() {
                        if !sf.marker[dof] {
                            continue;
                        }
                        for c1 in 0..dim {
                            let idx = c1 * nd + k;
                            // w = surf_fit_normal * coeff, then entry *= 1/count
                            // (the exact C++ operation order).
                            let mut entry = sf.coeff.get();
                            entry *= 1.0 / sf.dof_count[dof];
                            elmat[idx + ah * idx] += entry;
                        }
                    }
                }
            }
            // Single scatter of the element matrix (row = c_r*nd + i ↔ global
            // c_r*n_scalar + dof), mirroring MFEM's `grad->AddMatrix(elmat,
            // vdofs)`.
            for k in 0..ah {
                let kr = k % nd;
                let cr = k / nd;
                let grow = cr * self.n_scalar + el.edofs[kr];
                for l in 0..ah {
                    let lc = l % nd;
                    let cc = l / nd;
                    let gcol = cc * self.n_scalar + el.edofs[lc];
                    let v = elmat[k + ah * l];
                    if v != 0.0 {
                        coo.add(grow, gcol, v);
                    }
                }
            }
        }
        let mut h = coo.into_csr();
        for &dof in &self.ess_vdofs {
            h.eliminate_essential_bc_diag_symmetric(dof, 1.0);
        }
        h
    }

    /// MFEM `TMOP_Integrator::GetSurfaceFittingErrors` (serial path):
    /// average and maximum fitting error |x0 + dx - x_t| over the marked dofs,
    /// maximized over all fitting integrators.
    pub fn surf_fit_errors(&self, dx: &[f64]) -> (f64, f64) {
        let dim = self.dim;
        let mut err_avg = 0.0_f64;
        let mut err_max = 0.0_f64;
        for integ in &self.integrators {
            let Some(sf) = &integ.surf_fit else {
                continue;
            };
            let mut err_sum = 0.0_f64;
            let mut max_loc = 0.0_f64;
            let mut dof_cnt = 0_usize;
            for dof in 0..self.n_scalar {
                if !sf.marker[dof] {
                    continue;
                }
                dof_cnt += 1;
                let mut d2 = 0.0_f64;
                for c in 0..dim {
                    let p = self.x0[c * self.n_scalar + dof] + dx[c * self.n_scalar + dof];
                    let d = p - sf.pos[c * self.n_scalar + dof];
                    d2 += d * d;
                }
                let sigma = d2.sqrt(); // Vector::DistanceTo
                max_loc = max_loc.max(sigma);
                err_sum += sigma;
            }
            let avg_loc = if dof_cnt > 0 { err_sum / dof_cnt as f64 } else { 0.0 };
            err_avg = err_avg.max(avg_loc);
            err_max = err_max.max(max_loc);
        }
        (err_avg, err_max)
    }

    /// MFEM `TMOPNewtonSolver::GetSurfaceFittingWeight`: the current fitting
    /// weight of every fitting integrator.
    pub fn surf_fit_weights(&self) -> Vec<f64> {
        self.integrators
            .iter()
            .filter_map(|integ| integ.surf_fit.as_ref().map(|sf| sf.coeff.get()))
            .collect()
    }

    /// MFEM `TMOP_Integrator::UpdateSurfaceFittingWeight`: coeff *= factor.
    pub fn update_surf_fit_weight(&self, factor: f64) {
        for integ in &self.integrators {
            if let Some(sf) = &integ.surf_fit {
                sf.coeff.set(sf.coeff.get() * factor);
            }
        }
    }

    /// `TMOPNewtonSolver::ComputeMinDet`: minimum det(Jpr)/det(Wideal) over
    /// all quadrature points of the trial mesh x0 + dx.
    pub fn min_det_j(&self, dx: &[f64]) -> f64 {
        let dim = self.dim;
        let mut min_det = f64::INFINITY;
        let mut pos = Vec::new();
        let mut dsh = Vec::new();
        for el in &self.elems {
            let nd = el.edofs.len();
            self.element_positions(el, dx, &mut pos);
            dsh.resize(nd * dim, 0.0);
            for q in 0..el.quad_points.len() {
                el.re.eval_grad_basis(&el.quad_points[q], &mut dsh);
                let mut jpr = [[0.0f64; 3]; 3];
                for a in 0..dim {
                    for b in 0..dim {
                        let mut s = 0.0;
                        for i in 0..nd {
                            s += pos[i + a * nd] * dsh[i * dim + b];
                        }
                        jpr[a][b] = s;
                    }
                }
                // Wideal = I for quad/hex => det(Jpt) = det(Jpr).
                min_det = min_det.min(det_small(&jpr, dim));
            }
        }
        min_det
    }

    /// Per-element MFEM `Mesh::GetElementSize(i)` (type 0):
    /// `pow(|det J(center)|, 1/dim)` — the Jacobian of the Qp nodal geometry
    /// (`x0`) at the reference-element center, in the unit reference domain.
    /// Used by the driver for the per-dof mesh size `h0` (MFEM
    /// `h0(dofs[j]) = min(h0(dofs[j]), hi)`).
    pub fn element_sizes_center(&self) -> Vec<f64> {
        let dim = self.dim;
        let center: Vec<f64> = vec![0.5; dim];
        let zeros = vec![0.0f64; self.n_dofs()];
        let mut out = Vec::with_capacity(self.elems.len());
        let mut pos = Vec::new();
        let mut dsh = Vec::new();
        for el in &self.elems {
            let nd = el.edofs.len();
            self.element_positions(el, &zeros, &mut pos);
            dsh.resize(nd * dim, 0.0);
            el.re.eval_grad_basis(&center, &mut dsh);
            let mut jpr = [[0.0f64; 3]; 3];
            for a in 0..dim {
                for b in 0..dim {
                    let mut s = 0.0;
                    for i in 0..nd {
                        s += pos[i + a * nd] * dsh[i * dim + b];
                    }
                    jpr[a][b] = s;
                }
            }
            out.push(det_small(&jpr, dim).abs().powf(1.0 / dim as f64));
        }
        out
    }
}

/// MFEM `Geometries`/collection factory: H1 Gauss-Lobatto reference element per
/// `ElementType` (quad/hex only in the v1 scope).
fn ref_elem_for(et: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match et {
        ElementType::Quad4 | ElementType::Hex8 => ref_elem_vol_h1(et, order),
        other => panic!(
            "TmopForm: unsupported element type {:?} (v1 scope: Quad4/Hex8 meshes)",
            other
        ),
    }
}

#[inline]
fn invert_2x2(m: &[f64; 4]) -> [f64; 4] {
    let det = m[0] * m[3] - m[1] * m[2];
    let inv = 1.0 / det;
    [m[3] * inv, -m[1] * inv, -m[2] * inv, m[0] * inv]
}

#[inline]
fn invert_3x3(m: &[f64; 9]) -> [f64; 9] {
    let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    let inv = 1.0 / det;
    let mut r = [0.0; 9];
    r[0] = (m[4] * m[8] - m[5] * m[7]) * inv;
    r[1] = (m[2] * m[7] - m[1] * m[8]) * inv;
    r[2] = (m[1] * m[5] - m[2] * m[4]) * inv;
    r[3] = (m[5] * m[6] - m[3] * m[8]) * inv;
    r[4] = (m[0] * m[8] - m[2] * m[6]) * inv;
    r[5] = (m[2] * m[3] - m[0] * m[5]) * inv;
    r[6] = (m[3] * m[7] - m[4] * m[6]) * inv;
    r[7] = (m[1] * m[6] - m[0] * m[7]) * inv;
    r[8] = (m[0] * m[4] - m[1] * m[3]) * inv;
    r
}

/// Affine re-domaining wrapper: presents an arbitrary-tensor-domain reference
/// element (e.g. `HexQk` on `[-1,1]^3`) as MFEM's `[0,1]^d` convention.
/// TMOP math (det(Jpr), target scaling, min-det thresholds) assumes the MFEM
/// unit reference element.
struct UnitDomainElem {
    inner: Box<dyn ReferenceElement>,
    dim: usize,
}

impl ReferenceElement for UnitDomainElem {
    fn dim(&self) -> u8 {
        self.inner.dim()
    }
    fn order(&self) -> u8 {
        self.inner.order()
    }
    fn n_dofs(&self) -> usize {
        self.inner.n_dofs()
    }
    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        self.inner.eval_basis(xi, values);
    }
    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        self.inner.eval_grad_basis(xi, grads);
        // dφ/dξ_unit = 2 · dφ/dη for ξ = (η+1)/2.
        for g in grads.iter_mut() {
            *g *= 2.0;
        }
    }
    fn eval_hessian(&self, xi: &[f64], hess: &mut [f64]) {
        self.inner.eval_hessian(xi, hess);
        for h in hess.iter_mut() {
            *h *= 4.0;
        }
    }
    fn quadrature(&self, order: u8) -> fem_element::reference::QuadratureRule {
        let mut rule = self.inner.quadrature(order);
        for p in rule.points.iter_mut() {
            for c in p.iter_mut() {
                *c = 0.5 * (*c + 1.0);
            }
        }
        let scale = 0.5_f64.powi(self.dim as i32);
        for w in rule.weights.iter_mut() {
            *w *= scale;
        }
        rule
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        self.inner
            .dof_coords()
            .into_iter()
            .map(|p| p.into_iter().map(|c| 0.5 * (c + 1.0)).collect())
            .collect()
    }
}

/// Whether the reference element's tensor-product domain is `[0,1]^d` (true)
/// or `[-1,1]^d` (false), probed from the first dof coordinate.
fn el_domain_is_unit(re: &dyn ReferenceElement) -> bool {
    re.dof_coords()[0].first().map(|x| *x > -0.25).unwrap_or(true)
}

#[inline]
fn det_jtr2(j: &[f64; 4]) -> f64 {
    j[0] * j[3] - j[1] * j[2]
}

#[inline]
fn det_jtr3(j: &[f64; 9]) -> f64 {
    j[0] * (j[4] * j[8] - j[5] * j[7]) - j[1] * (j[3] * j[8] - j[5] * j[6])
        + j[2] * (j[3] * j[7] - j[4] * j[6])
}

#[inline]
fn det_small(j: &[[f64; 3]; 3], dim: usize) -> f64 {
    match dim {
        2 => j[0][0] * j[1][1] - j[1][0] * j[0][1],
        3 => {
            j[0][0] * (j[1][1] * j[2][2] - j[2][1] * j[1][2])
                - j[0][1] * (j[1][0] * j[2][2] - j[2][0] * j[1][2])
                + j[0][2] * (j[1][0] * j[2][1] - j[2][0] * j[1][1])
        }
        _ => unreachable!(),
    }
}

/// Inner linear solver choices (mesh-optimizer `-ls`).
pub enum TmopLinSolver {
    /// 0: l1-Jacobi stationary iteration, `max_lin_iter` sweeps (DSmoother(1,1,it)).
    L1Jacobi(usize),
    /// 1: CG, rtol 1e-12.
    Cg(usize),
    /// 2: MINRES, rtol 1e-12.
    Minres(usize),
    /// 3: MINRES + Jacobi (positive diagonal).
    MinresJacobi(usize),
    /// 4: MINRES + l1-Jacobi.
    MinresL1Jacobi(usize),
}

/// Newton result (MFEM `IterativeSolver` state).
pub struct TmopNewtonResult {
    pub converged: bool,
    pub iterations: usize,
    pub final_norm: f64,
    pub initial_norm: f64,
}

/// MFEM `TMOPNewtonSolver` (Newton path): solve for the displacement dx with
/// the exact energy/min-det/residual-norm line search of `ComputeScalingFactor`.
///
/// `min_det` is the shared cell bound to untangling metrics; the driver
/// initializes it to the (adjusted) min det of the starting mesh.
pub fn tmop_newton_solve(
    form: &TmopForm,
    lin: &TmopLinSolver,
    rtol: f64,
    max_iter: usize,
    verbosity: u8,
    min_det: &SharedMinDet,
) -> (Vec<f64>, TmopNewtonResult) {
    let n = form.n_dofs();
    let mut x = vec![0.0f64; n]; // iterative_mode = false: dx starts at 0.
    let mut r = vec![0.0f64; n];
    form.gradient(&x, &mut r);
    let norm0 = l2_norm(&r);
    let norm_goal = rtol * norm0; // abs_tol = 0
    let mut norm = norm0;
    let mut converged = false;
    let mut c = vec![0.0f64; n];
    let mut final_iter = 0usize;

    for it in 0..=max_iter {
        if verbosity > 0 {
            println!("Newton iteration {:2} : ||r|| = {:.6e}", it, norm);
            if it > 0 {
                println!("   ||r||/||r_0|| = {:.6e}", norm / norm0);
            }
        }
        if norm <= norm_goal {
            converged = true;
            break;
        }
        if it >= max_iter {
            break;
        }
        final_iter = it + 1;
        let hess = form.hessian(&x);
        solve_linear(&hess, &r, &mut c, lin);
        for &dof in &form.ess_vdofs {
            c[dof] = 0.0;
        }
        let scale = compute_scaling_factor(form, &x, &r, &c, verbosity, min_det);
        if scale == 0.0 {
            converged = false;
            break;
        }
        for i in 0..n {
            x[i] -= scale * c[i];
        }
        form.gradient(&x, &mut r);
        norm = l2_norm(&r);
    }

    (
        x,
        TmopNewtonResult {
            converged,
            iterations: final_iter,
            final_norm: norm,
            initial_norm: norm0,
        },
    )
}

/// MFEM `TMOPNewtonSolver::ComputeScalingFactor` (serial, no surface fitting).
fn compute_scaling_factor(
    form: &TmopForm,
    d_in: &[f64],
    r: &[f64],
    c: &[f64],
    verbosity: u8,
    min_det: &SharedMinDet,
) -> f64 {
    let energy_in = form.energy(d_in);
    let min_det_t_in = form.min_det_j(d_in);
    let untangling = min_det_t_in <= 0.0;
    let untangle_factor = 1.5;
    let min_detj_limit = 0.0;
    if untangling {
        min_det.set(untangle_factor * min_det_t_in);
    }

    let norm_in = l2_norm(r);
    let mut scale = 1.0;
    let mut x_out_ok = false;
    let mut energy_out = f64::NAN;
    let mut min_det_t_out = min_det_t_in;
    let detj_factor = 0.5;
    let n = d_in.len();
    let mut d_out = vec![0.0f64; n];
    let mut r_out = vec![0.0f64; n];

    for _i in 0..12 {
        for j in 0..n {
            d_out[j] = d_in[j] - scale * c[j];
        }
        min_det_t_out = form.min_det_j(&d_out);
        if !untangling && min_det_t_out <= min_detj_limit {
            if verbosity > 0 {
                println!("Scale = {} Neg det(J) found.", scale);
            }
            scale *= detj_factor;
            continue;
        }
        if untangling && min_det_t_out < min_det.get() {
            if verbosity > 0 {
                println!("Scale = {} Neg det(J) decreased.", scale);
            }
            scale *= detj_factor;
            continue;
        }
        if untangling {
            x_out_ok = true;
            break;
        }
        energy_out = form.energy(&d_out);
        if energy_out > energy_in + 0.2 * energy_in.abs() || energy_out.is_nan() {
            if verbosity > 0 {
                println!(
                    "Scale = {} Increasing energy: {} --> {}",
                    scale, energy_in, energy_out
                );
            }
            scale *= 0.5;
            continue;
        }
        form.gradient(&d_out, &mut r_out);
        let norm_out = l2_norm(&r_out);
        if norm_out > 1.2 * norm_in {
            if verbosity > 0 {
                println!(
                    "Scale = {} Norm increased: {} --> {}",
                    scale, norm_in, norm_out
                );
            }
            scale *= 0.5;
            continue;
        } else {
            x_out_ok = true;
            break;
        }
    }

    if untangling {
        if min_det_t_out > 0.0 {
            min_det.set(0.0);
            if verbosity > 0 {
                println!("The mesh has been untangled at the used points!");
            }
        } else {
            min_det.set(untangle_factor * min_det_t_out);
        }
    } else if verbosity > 0 {
        println!(
            "Energy decrease: {} --> {} or {}% with {} scaling.",
            energy_in,
            energy_out,
            (energy_in - energy_out) / energy_in * 100.0,
            scale
        );
    }
    if !x_out_ok {
        scale = 0.0;
    }
    scale
}

// ─── TMOPNewtonSolver with surface fitting (fit-node-position) ───────────────

/// MFEM `IterativeSolver::PrintLevel` flags derived from the legacy
/// `SetPrintLevel(int)` levels (linalg/solvers.cpp, `FromLegacyPrintLevel`).
#[derive(Debug, Clone, Copy)]
pub struct TmopPrintLevel {
    pub none: bool,
    pub errors: bool,
    pub warnings: bool,
    pub iterations: bool,
    pub summary: bool,
    pub first_and_last: bool,
}

impl TmopPrintLevel {
    pub fn from_legacy(level: i32) -> Self {
        let base = Self {
            none: false,
            errors: true,
            warnings: true,
            iterations: false,
            summary: false,
            first_and_last: false,
        };
        match level {
            -1 => Self {
                none: true,
                errors: false,
                warnings: false,
                iterations: false,
                summary: false,
                first_and_last: false,
            },
            0 => base,
            1 => Self {
                iterations: true,
                ..base
            },
            2 => Self {
                summary: true,
                ..base
            },
            3 => Self {
                first_and_last: true,
                ..base
            },
            _ => base, // MFEM warns and defaults to level 0
        }
    }
}

/// Adaptive surface fitting parameters of MFEM `TMOPNewtonSolver` (defaults
/// from fem/tmop_tools.hpp).
#[derive(Debug, Clone, Copy)]
pub struct SurfFitNewtonParams {
    /// `SetAdaptiveSurfaceFittingScalingFactor`: the fitting weight is
    /// multiplied by at most this factor when the average fitting error does
    /// not decrease sufficiently. 0 disables the adaptive weight updates.
    pub scale_factor: f64,
    /// `SetTerminationWithMaxSurfaceFittingError`: terminate the line search
    /// (with scale 0) once the maximum fitting error drops below this
    /// threshold. Negative disables it and makes the residual norm the
    /// convergence criterion (MFEM `surf_fit_converge_error = false`).
    pub max_err_limit: f64,
    /// MFEM `surf_fit_err_rel_change_limit` (default 1e-3): increase the
    /// fitting weight when the relative decrease of the average fitting error
    /// per iteration is below this value.
    pub err_rel_change_limit: f64,
    /// MFEM `surf_fit_weight_limit` (default 1e10).
    pub weight_limit: f64,
    /// MFEM `surf_fit_adapt_count_limit` (default 10): terminate after this
    /// many weight increases.
    pub adapt_count_limit: usize,
}

impl SurfFitNewtonParams {
    /// MFEM member defaults.
    pub fn new() -> Self {
        Self {
            scale_factor: 0.0,
            max_err_limit: -1.0,
            err_rel_change_limit: 0.001,
            weight_limit: 1e10,
            adapt_count_limit: 10,
        }
    }

    /// MFEM `surf_fit_converge_error` (set together with `max_err_limit`).
    pub fn converge_error(&self) -> bool {
        self.max_err_limit >= 0.0
    }
}

impl Default for SurfFitNewtonParams {
    fn default() -> Self {
        Self::new()
    }
}

/// C `std::cout`/`%g` formatting with `sig` significant digits (MFEM prints
/// doubles through `mfem::out` with the default precision of 6).
fn fmt_gp(x: f64, sig: usize) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    let exp = x.abs().log10().floor() as i32;
    if !(-4..(sig as i32)).contains(&exp) {
        fix_exp(format!("{:.*e}", sig - 1, x))
    } else {
        let decimals = (sig as i32 - 1 - exp).max(0) as usize;
        let mut s = format!("{:.*}", decimals, x);
        if s.contains('.') {
            while s.ends_with('0') {
                s.pop();
            }
            if s.ends_with('.') {
                s.pop();
            }
        }
        s
    }
}

/// C `%g` with 6 significant digits.
fn fmt_g6(x: f64) -> String {
    fmt_gp(x, 6)
}

/// glibc `hypot` (sysdeps/ieee754/dbl-64/e_hypot.c, Borges 2019 correction),
/// 1:1 port of the x86-64 baseline build (no FMA branch, which is what the
/// reference libmfem links against on glibc systems). MFEM's MINRES calls
/// `std::hypot`; a truncated (max_iter-limited) MINRES amplifies last-ulp
/// differences of the platform libm into the Newton direction, so the Rust
/// port must reproduce glibc's exact result rather than the UCRT one.
fn hypot_glibc(x: f64, y: f64) -> f64 {
    // #define SCALE 0x1p-600, LARGE_VAL 0x1p+511, TINY_VAL 0x1p-459, EPS 0x1p-54
    const SCALE: f64 = f64::from_bits((((-600i64) + 1023) as u64) << 52);
    const LARGE_VAL: f64 = f64::from_bits(((511u64 + 1023) << 52) as u64);
    const TINY_VAL: f64 = f64::from_bits((((-459i64) + 1023) as u64) << 52);
    const EPS: f64 = f64::from_bits((((-54i64) + 1023) as u64) << 52);

    if !x.is_finite() || !y.is_finite() {
        if x.is_infinite() || y.is_infinite() {
            return f64::INFINITY;
        }
        return x + y;
    }
    let x = x.abs();
    let y = y.abs();
    let (ax, ay) = if x < y { (y, x) } else { (x, y) };

    if ax > LARGE_VAL {
        if ay <= ax * EPS {
            return ax + ay;
        }
        return hypot_kernel(ax * SCALE, ay * SCALE) / SCALE;
    }
    if ay < TINY_VAL {
        if ax >= ay / EPS {
            return ax + ay;
        }
        return hypot_kernel(ax / SCALE, ay / SCALE) * SCALE;
    }
    if ay <= ax * EPS {
        return ax + ay;
    }
    hypot_kernel(ax, ay)
}

/// Kernel of the glibc `hypot` port above: the no-FMA (`!__FP_FAST_FMA`)
/// branch of glibc's e_hypot.c, which is what the x86-64 baseline reference
/// libm uses. The FMA variant is intentionally not ported (dead code).
fn hypot_kernel(ax: f64, ay: f64) -> f64 {
    // Borges 2019 correction, no-FMA branch.
    let mut h = (ax * ax + ay * ay).sqrt();
    if h <= 2.0 * ay {
        let delta = h - ay;
        let t1 = ax * (2.0 * delta - ax);
        let t2 = (delta - 2.0 * (ax - ay)) * delta;
        h -= (t1 + t2) / (2.0 * h);
    } else {
        let delta = h - ax;
        let t1 = 2.0 * delta * (ax - 2.0 * ay);
        let t2 = (4.0 * delta - ay) * ay + delta * delta;
        h -= (t1 + t2) / (2.0 * h);
    }
    h
}

/// Normalize Rust's `1.8633e-1` to C's `1.8633e-01`.
fn fix_exp(s: String) -> String {
    match s.find('e') {
        Some(pos) => {
            let (m, e) = s.split_at(pos);
            let exp: i32 = e[1..].parse().unwrap();
            format!("{}e{}{:02}", m, if exp < 0 { "-" } else { "+" }, exp.abs())
        }
        None => s,
    }
}

#[inline]
fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

/// MFEM `MINRESSolver::Mult` (linalg/solvers.cpp, van der Vorst Fig. 6.9
/// formulation) without a preconditioner, solving `H c = r` from `c = 0`.
///
/// This is a bit-level port of MFEM's recurrence; it intentionally does not
/// reuse `fem_solver::solve_minres` (a classical Lanczos + back-substitution
/// formulation) because the Newton iteration path of the 1:1 TMOP port must
/// reproduce MFEM's floating-point rounding, or the outer line search may
/// take different branches near its thresholds.
fn minres_mfem(
    h: &CsrMatrix<f64>,
    b: &[f64],
    x: &mut [f64],
    rtol: f64,
    atol: f64,
    max_iter: usize,
) {
    let n = b.len();
    for v in x.iter_mut() {
        *v = 0.0;
    }
    let mut v0 = vec![0.0_f64; n];
    let mut v1 = b.to_vec(); // iterative_mode = false: v1 = b
    let mut q = vec![0.0_f64; n];
    let mut w0 = vec![0.0_f64; n];
    let mut w1 = vec![0.0_f64; n];

    let mut eta = dot(&v1, &v1).sqrt(); // z = v1 (no preconditioner)
    let mut beta = eta;
    let norm_goal = (rtol * eta).max(atol);
    if eta <= norm_goal {
        return;
    }
    let (mut gamma0, mut gamma1) = (1.0_f64, 1.0_f64);
    let (mut sigma0, mut sigma1) = (0.0_f64, 0.0_f64);

    let mut it = 1_usize;
    while it <= max_iter {
        // v1 /= beta — MFEM's Vector::operator/= multiplies by the reciprocal.
        let beta_inv = 1.0 / beta;
        for v in v1.iter_mut() {
            *v *= beta_inv;
        }
        // MFEM SparseMatrix::Mult: y = 0 then per-row sequential accumulation
        // over the row's entries in storage order (NOT the 8-way blocked
        // grouping of fem_linalg's spmv, which rounds differently and would
        // derail the truncated MINRES iteration relative to the reference).
        for row in 0..n {
            let (s, e) = (h.row_ptr[row], h.row_ptr[row + 1]);
            let mut sum = 0.0_f64;
            for k in s..e {
                sum += h.values[k] * v1[h.col_idx[k] as usize];
            }
            q[row] = sum;
        }
        let alpha = dot(&v1, &q);
        if it > 1 {
            // q.Add(-beta, v0) with v0 the PREVIOUS search direction
            for i in 0..n {
                q[i] -= beta * v0[i];
            }
        }
        // add(q, -alpha, v1, v0): v0 = q - alpha*v1
        for i in 0..n {
            v0[i] = q[i] - alpha * v1[i];
        }

        let delta = gamma1 * alpha - gamma0 * sigma1 * beta;
        let rho3 = sigma0 * beta;
        let rho2 = sigma1 * alpha + gamma0 * gamma1 * beta;
        beta = dot(&v0, &v0).sqrt(); // Norm(v0)
        let rho1 = hypot_glibc(delta, beta);
        if it == 1 {
            // w0.Set(1./rho1, *z) with z == v1
            let s = 1.0 / rho1;
            for i in 0..n {
                w0[i] = s * v1[i];
            }
        } else if it == 2 {
            // add(1./rho1, *z, -rho2/rho1, w1, w0)
            let s = 1.0 / rho1;
            let t = -rho2 / rho1;
            for i in 0..n {
                w0[i] = s * v1[i] + t * w1[i];
            }
        } else {
            // add(-rho3/rho1, w0, -rho2/rho1, w1, w0); w0.Add(1./rho1, *z)
            let a = -rho3 / rho1;
            let b2 = -rho2 / rho1;
            for i in 0..n {
                w0[i] = a * w0[i] + b2 * w1[i];
            }
            let s = 1.0 / rho1;
            for i in 0..n {
                w0[i] += s * v1[i];
            }
        }

        gamma0 = gamma1;
        gamma1 = delta / rho1;
        // x.Add(gamma1*eta, w0)
        let t = gamma1 * eta;
        for i in 0..n {
            x[i] += t * w0[i];
        }
        sigma0 = sigma1;
        sigma1 = beta / rho1;
        eta = -sigma1 * eta;
        if eta.abs() <= norm_goal {
            return; // converged = true
        }
        std::mem::swap(&mut v0, &mut v1);
        std::mem::swap(&mut w0, &mut w1);
        it += 1;
    }
    // Max-iter exhaustion: converged = false. The Newton solver ignores the
    // MINRES convergence flag and uses the iterate as-is (MFEM semantics).
}

/// MFEM `TMOPNewtonSolver` for problems with (adaptive) surface fitting:
/// the `NewtonSolver::Mult` loop with the TMOP `ProcessNewState` (fitting
/// weight update) and `ComputeScalingFactor` (fitting-aware line search and
/// fit-error termination) hooks. The essential dofs of the form see zero
/// residual rows and an identity diagonal of the Hessian, exactly like the
/// serial `NonlinearForm` with `SetEssentialVDofs`.
#[allow(clippy::too_many_arguments)]
pub fn tmop_newton_solve_surf_fit(
    form: &TmopForm,
    params: SurfFitNewtonParams,
    newton_max_iter: usize,
    newton_rtol: f64,
    newton_abs_tol: f64,
    lin_max_iter: usize,
    lin_rtol: f64,
    print_level: i32,
    min_det: &SharedMinDet,
) -> (Vec<f64>, TmopNewtonResult) {
    let pl = TmopPrintLevel::from_legacy(print_level);
    let n = form.n_dofs();
    let mut x = vec![0.0_f64; n]; // displacement, iterative_mode = false
    let mut r = vec![0.0_f64; n];
    let mut c = vec![0.0_f64; n];

    // MFEM mutable solver state.
    let mut avg_err_prvs = 10000.0_f64; // surf_fit_avg_err_prvs
    let mut adapt_count: usize = 0; // surf_fit_adapt_count
    let mut coeff_update = false; // surf_fit_coeff_update (set by ComputeScalingFactor)
    // The initial ProcessNewState(x = 0) is a no-op while coeff_update is
    // false (the flag is only set at the end of ComputeScalingFactor).

    form.gradient(&x, &mut r); // oper->Mult(x, r); b is empty: no subtraction
    let norm0 = l2_norm(&r);
    let mut norm = norm0;
    let norm_goal = (newton_rtol * norm0).max(newton_abs_tol);
    let converged;
    let mut it = 0_usize;
    let final_iter;

    loop {
        if pl.iterations {
            print!("Newton iteration {:2} : ||r|| = {}", it, fmt_g6(norm));
            if it > 0 {
                print!(", ||r||/||r_0|| = {}", fmt_g6(norm / norm0));
            }
            println!();
        }
        if norm <= norm_goal {
            converged = true;
            final_iter = it;
            break;
        }
        if it >= newton_max_iter {
            converged = false;
            final_iter = it;
            break;
        }

        // prec->SetOperator(gradient); prec->Mult(r, c): MINRES from c = 0.
        let hess = form.hessian(&x);
        minres_mfem(&hess, &r, &mut c, lin_rtol, 0.0, lin_max_iter);
        // The Hessian is eliminated at the essential dofs (identity diagonal,
        // zero rows/columns) and the residual is zero there, so the Krylov
        // solution vanishes at those dofs without explicit elimination of c
        // (MFEM does not touch c either).

        let scale = compute_scaling_factor_surf_fit(
            form,
            &params,
            &x,
            &r,
            &c,
            &pl,
            &mut coeff_update,
            &mut adapt_count,
            min_det,
        );
        if scale == 0.0 {
            converged = false;
            final_iter = it;
            break;
        }
        for i in 0..n {
            x[i] -= scale * c[i];
        }

        // ProcessNewState(x): updates the adaptive fitting weight. This is the
        // only place where `coeff_update` is consumed: inside the line search
        // the flag is always false (it is set at the end of every
        // ComputeScalingFactor and consumed right here).
        process_new_state_surf_fit(
            form,
            &params,
            &pl,
            &x,
            &mut coeff_update,
            &mut adapt_count,
            &mut avg_err_prvs,
        );

        form.gradient(&x, &mut r);
        norm = l2_norm(&r);
        it += 1;
    }

    // Summary printing (NewtonSolver::Mult tail).
    if pl.summary || (!converged && pl.warnings) || pl.first_and_last {
        println!("Newton: Number of iterations: {}", final_iter);
        println!(
            "   ||r|| = {},  ||r||/||r_0|| = {}",
            fmt_g6(norm),
            fmt_g6(norm / norm0)
        );
    }
    if !converged && (pl.summary || pl.warnings) {
        println!("Newton: No convergence!");
    }

    (
        x,
        TmopNewtonResult {
            converged,
            iterations: final_iter,
            final_norm: norm,
            initial_norm: norm0,
        },
    )
}

/// MFEM `TMOPNewtonSolver::ProcessNewState` (surface fitting part only): on
/// the first call after every line search, adapt the fitting weight if the
/// average fitting error did not decrease sufficiently.
fn process_new_state_surf_fit(
    form: &TmopForm,
    params: &SurfFitNewtonParams,
    pl: &TmopPrintLevel,
    dx: &[f64],
    coeff_update: &mut bool,
    adapt_count: &mut usize,
    avg_err_prvs: &mut f64,
) {
    if !*coeff_update {
        return;
    }
    // Get surface fitting errors.
    let (avg_err, max_err) = form.surf_fit_errors(dx);
    // Get array with surface fitting weights.
    let weights = form.surf_fit_weights();
    let wmax = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let wmin = weights.iter().cloned().fold(f64::INFINITY, f64::min);
    if pl.iterations {
        println!(
            "Avg/Max surface fitting error: {} {}",
            fmt_g6(avg_err),
            fmt_g6(max_err)
        );
        println!(
            "Min/Max surface fitting weight: {} {}",
            fmt_g6(wmin),
            fmt_g6(wmax)
        );
    }
    let change = *avg_err_prvs - avg_err;
    let rel_change = change / *avg_err_prvs;
    if rel_change < params.err_rel_change_limit
        && (params.converge_error()
            || (wmax < params.weight_limit && max_err > params.max_err_limit))
    {
        let factor = params.scale_factor.min(params.weight_limit / wmax);
        form.update_surf_fit_weight(factor);
        *adapt_count += 1;
    } else {
        *adapt_count = 0;
    }
    *avg_err_prvs = avg_err;
    *coeff_update = false;
}

/// MFEM `TMOPNewtonSolver::ComputeScalingFactor` with surface fitting
/// (serial, full assembly, no adaptive limiting).
#[allow(clippy::too_many_arguments)]
fn compute_scaling_factor_surf_fit(
    form: &TmopForm,
    params: &SurfFitNewtonParams,
    d_in: &[f64],
    r: &[f64],
    c: &[f64],
    pl: &TmopPrintLevel,
    coeff_update: &mut bool,
    adapt_count: &mut usize,
    min_det: &SharedMinDet,
) -> f64 {
    let energy_in = form.energy(d_in);

    let fitting = form.surf_fit_weights().len() > 0; // IsSurfaceFittingEnabled
    let init_fit_max_err;
    if fitting && params.converge_error() {
        let (_avg, max) = form.surf_fit_errors(d_in);
        init_fit_max_err = max;
        if max < params.max_err_limit {
            if pl.iterations || pl.warnings {
                println!("TMOPNewtonSolver converged based on the surface fitting error.");
            }
            return 0.0;
        }
    } else {
        init_fit_max_err = 0.0;
    }

    if *adapt_count >= params.adapt_count_limit {
        if pl.iterations {
            println!(
                "TMOPNewtonSolver terminated based on max number of times \
surface fitting weight canbe increased. "
            );
        }
        return 0.0;
    }

    let min_det_t_in = form.min_det_j(d_in);
    let untangling = min_det_t_in <= 0.0;
    let untangle_factor = 1.5;
    let min_detj_limit = 0.0;
    if untangling {
        min_det.set(untangle_factor * min_det_t_in);
    }

    let norm_in = l2_norm(r);
    let mut scale = 1.0_f64;
    let mut x_out_ok = false;
    let mut energy_out = 0.0_f64;
    let mut min_det_t_out = min_det_t_in;
    let detj_factor = 0.5;
    let n = d_in.len();
    let mut d_out = vec![0.0_f64; n];

    for _i in 0..12 {
        for j in 0..n {
            d_out[j] = d_in[j] - scale * c[j];
        }
        min_det_t_out = form.min_det_j(&d_out);
        if !untangling && min_det_t_out <= min_detj_limit {
            if pl.iterations {
                println!("Scale = {} Neg det(J) found.", fmt_g6(scale));
            }
            scale *= detj_factor;
            continue;
        }
        if untangling && min_det_t_out < min_det.get() {
            if pl.iterations {
                println!("Scale = {} Neg det(J) decreased.", fmt_g6(scale));
            }
            scale *= detj_factor;
            continue;
        }
        if untangling {
            x_out_ok = true;
            break;
        }

        // ProcessNewState(d_out) is a no-op here: the fitting weight update
        // flag is always false inside the line search (see
        // process_new_state_surf_fit in the Newton loop).

        // Ensure sufficient decrease in fitting error when converging based
        // on the error.
        if fitting && params.converge_error() {
            let (_avg, max_fit_err) = form.surf_fit_errors(&d_out);
            if max_fit_err >= 1.2 * init_fit_max_err {
                if pl.iterations {
                    println!("Scale = {} Surf fit err increased.", fmt_g6(scale));
                }
                scale *= 0.5;
                continue;
            }
        }

        energy_out = form.energy(&d_out);
        if energy_out > energy_in + 0.2 * energy_in.abs() || energy_out.is_nan() {
            if pl.iterations {
                println!(
                    "Scale = {} Increasing energy: {} --> {}",
                    fmt_g6(scale),
                    fmt_g6(energy_in),
                    fmt_g6(energy_out)
                );
            }
            scale *= 0.5;
            continue;
        }

        let mut r_out = vec![0.0_f64; n];
        form.gradient(&d_out, &mut r_out);
        let norm_out = l2_norm(&r_out);
        if norm_out > 1.2 * norm_in {
            if pl.iterations {
                println!(
                    "Scale = {} Norm increased: {} --> {}",
                    fmt_g6(scale),
                    fmt_g6(norm_in),
                    fmt_g6(norm_out)
                );
            }
            scale *= 0.5;
            continue;
        }
        x_out_ok = true;
        break;
    }

    if untangling {
        if min_det_t_out > 0.0 {
            min_det.set(0.0);
            if pl.iterations || pl.summary || pl.first_and_last {
                println!("The mesh has been untangled at the used points!");
            }
        } else {
            min_det.set(untangle_factor * min_det_t_out);
        }
    }

    if pl.iterations || pl.summary || pl.first_and_last {
        if untangling {
            println!(
                "Min det(T) change: {} -> {} with {} scaling.",
                fmt_g6(min_det_t_in),
                fmt_g6(min_det_t_out),
                fmt_g6(scale)
            );
        } else {
            println!(
                "Energy decrease: {} --> {} or {}% with {} scaling.",
                fmt_g6(energy_in),
                fmt_g6(energy_out),
                fmt_g6((energy_in - energy_out) / energy_in * 100.0),
                fmt_g6(scale)
            );
        }
    }

    if !x_out_ok {
        scale = 0.0;
    }
    if params.scale_factor > 0.0 {
        *coeff_update = true;
    }
    scale
}

fn solve_linear(h: &CsrMatrix<f64>, r: &[f64], c: &mut [f64], lin: &TmopLinSolver) {
    let n = r.len();
    let linsol_rtol = 1e-12;
    match lin {
        TmopLinSolver::L1Jacobi(iters) => {
            // DSmoother(1, 1.0, iters), iterative_mode = false: stationary
            // iteration with the l1 diagonal.
            let d = l1_diagonal(h);
            for v in c.iter_mut() {
                *v = 0.0;
            }
            let mut tmp = vec![0.0f64; n];
            for _ in 0..*iters {
                h.spmv(c, &mut tmp);
                for i in 0..n {
                    c[i] += (r[i] - tmp[i]) / d[i];
                }
            }
        }
        TmopLinSolver::Cg(max_it) => {
            for v in c.iter_mut() {
                *v = 0.0;
            }
            let cfg = SolverConfig {
                rtol: linsol_rtol,
                atol: 0.0,
                max_iter: *max_it,
                ..SolverConfig::default()
            };
            let _ : Result<SolveResult, _> = solve_cg(h, r, c, &cfg);
        }
        TmopLinSolver::Minres(max_it) => {
            for v in c.iter_mut() {
                *v = 0.0;
            }
            let cfg = SolverConfig {
                rtol: linsol_rtol,
                atol: 0.0,
                max_iter: *max_it,
                ..SolverConfig::default()
            };
            let _: Result<SolveResult, _> = solve_minres(h, r, c, &cfg);
        }
        TmopLinSolver::MinresJacobi(max_it) => {
            for v in c.iter_mut() {
                *v = 0.0;
            }
            let cfg = SolverConfig {
                rtol: linsol_rtol,
                atol: 0.0,
                max_iter: *max_it,
                ..SolverConfig::default()
            };
            let _: Result<SolveResult, _> = solve_minres_jacobi(h, r, c, &cfg);
        }
        TmopLinSolver::MinresL1Jacobi(max_it) => {
            for v in c.iter_mut() {
                *v = 0.0;
            }
            let d = l1_diagonal(h);
            let cfg = SolverConfig {
                rtol: linsol_rtol,
                atol: 0.0,
                max_iter: *max_it,
                ..SolverConfig::default()
            };
            let prec = |rr: &[f64], z: &mut [f64]| {
                for i in 0..z.len() {
                    z[i] = rr[i] / d[i];
                }
            };
            let _: Result<SolveResult, _> = solve_minres_precond(h, &prec, r, c, &cfg);
        }
    }
}

fn l1_diagonal(h: &CsrMatrix<f64>) -> Vec<f64> {
    let n = h.nrows;
    let mut d = vec![0.0f64; n];
    for row in 0..n {
        let (s, e) = (h.row_ptr[row], h.row_ptr[row + 1]);
        let mut sum = 0.0;
        for k in s..e {
            sum += h.values[k].abs();
        }
        d[row] = if sum > 0.0 { sum } else { 1.0 };
    }
    d
}

fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Build the nodal position vector for a *linear* input mesh and a Qk nodal
/// space of the given order: interpolate the vertex coordinates with the
/// order-1 basis at every dof's reference location (MFEM initializes the nodal
/// GridFunction of `SetNodalFESpace` the same way). Layout `[c*n_scalar + s]`.
pub fn linear_mesh_positions(
    topo: &dyn MeshTopology,
    dm: &DofManager,
    order: u8,
    dim: usize,
) -> Vec<f64> {
    let n_scalar = dm.n_dofs;
    let mut x0 = vec![0.0f64; dim * n_scalar];
    for e in 0..topo.n_elements() {
        let e = e as u32;
        let et = topo.element_type(e);
        let re = ref_elem_vol_h1(et, order);
        let lin = ref_elem_vol_h1(et, 1);
        let edofs: Vec<usize> = dm.element_dofs(e).iter().map(|&d| d as usize).collect();
        let vnodes = topo.element_nodes(e).to_vec();
        for (k, &dof) in edofs.iter().enumerate() {
            let xi = re.dof_coords()[k].clone();
            let mut shape = vec![0.0f64; lin.n_dofs()];
            lin.eval_basis(&xi, &mut shape);
            let mut coord = vec![0.0f64; dim];
            for (v, &nv) in vnodes.iter().enumerate() {
                let c = topo.node_coords(nv);
                for d in 0..dim {
                    coord[d] += shape[v] * c[d];
                }
            }
            for d in 0..dim {
                x0[d * n_scalar + dof] = coord[d];
            }
        }
    }
    x0
}

/// Positions of a curved (high-order geometry) mesh on the Qp nodal space,
/// with MFEM `Mesh::SetNodalFESpace` semantics: the new nodal grid function is
/// initialized by `Mesh::GetNodes` → `GridFunction::ProjectCoefficient(xyz)`
/// with `ProjectType::Default`, which for nodal elements is
/// `NodalFiniteElement::Project` — interpolation of the mesh's existing
/// geometry map at the new space's dof reference points (NOT an L2
/// projection).
pub fn curved_mesh_positions(
    topo: &dyn MeshTopology,
    dm: &DofManager,
    order: u8,
    dim: usize,
) -> Vec<f64> {
    let n_scalar = dm.n_dofs;
    let geo_order = topo.geom_order();
    let mut x0 = vec![0.0f64; dim * n_scalar];
    let mut shape = Vec::new();
    let mut gcoords: Vec<f64> = Vec::new();
    for e in 0..topo.n_elements() {
        let e = e as u32;
        let et = topo.element_type(e);
        let re_new = ref_elem_vol_h1(et, order);
        let re_old = ref_elem_vol_h1(et, geo_order);
        // Old geometry dof coordinates for this element (geometry_nodes /
        // geom_coords_of carry the file's high-order `nodes` section).
        let gn = topo.geometry_nodes(e);
        gcoords.clear();
        gcoords.resize(gn.len() * dim, 0.0);
        for (i, &g) in gn.iter().enumerate() {
            let c = topo.geom_coords_of(g);
            gcoords[i * dim..i * dim + dim].copy_from_slice(c);
        }
        let edofs = dm.element_dofs(e);
        let dcoords = re_new.dof_coords();
        shape.clear();
        shape.resize(re_old.n_dofs(), 0.0);
        for (k, &dof) in edofs.iter().enumerate() {
            re_old.eval_basis(&dcoords[k], &mut shape);
            let mut coord = [0.0f64; 3];
            for (i, &s) in shape.iter().enumerate() {
                for d in 0..dim {
                    coord[d] += s * gcoords[i * dim + d];
                }
            }
            for d in 0..dim {
                x0[d * n_scalar + dof as usize] = coord[d];
            }
        }
    }
    x0
}

/// MFEM `Mesh::CheckElementOrientation(false)` (the serial loader check, run
/// on the file's geometry before any nodal space conversion): count elements
/// whose Jacobian determinant at the reference-element center is negative.
pub fn count_wrong_orientations(topo: &dyn MeshTopology) -> usize {
    let dim = topo.dim() as usize;
    let mut wrong = 0;
    for e in 0..topo.n_elements() {
        let e = e as u32;
        let et = topo.element_type(e);
        let raw = ref_elem_vol_h1(et, topo.geom_order());
        // Normalize the reference domain to MFEM's [0,1]^d so the center is
        // 0.5 (Geometry::GetCenter).
        let re: Box<dyn ReferenceElement> = if el_domain_is_unit(raw.as_ref()) {
            raw
        } else {
            Box::new(UnitDomainElem { inner: raw, dim })
        };
        let center: Vec<f64> = vec![0.5; dim];
        let mut dsh = vec![0.0f64; re.n_dofs() * dim];
        re.eval_grad_basis(&center, &mut dsh);
        let gn = topo.geometry_nodes(e);
        let mut jpr = [[0.0f64; 3]; 3];
        for (i, &g) in gn.iter().enumerate() {
            let c = topo.geom_coords_of(g);
            for a in 0..dim {
                for b in 0..dim {
                    jpr[a][b] += c[a] * dsh[i * dim + b];
                }
            }
        }
        if det_small(&jpr, dim) < 0.0 {
            wrong += 1;
        }
    }
    wrong
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;

    /// DofManager element DOF order must match the reference element's
    /// dof_coords order (this is what makes the TMOP position gather
    /// `pos[k + c*nd]` and the assembled geometry tables consistent).
    #[test]

    /// Straight hex mesh with P2 curvature: TMOP min det must be the exact
    /// linear value (1/8 for the unit cube split into 8 hexes).
    #[test]
    fn hex_tmop_straight_curved_min_det() {
        let mesh: Mesh<3> =
            Mesh::make_cartesian_3d(2, 2, 2, ElementType::Hex8, 1.0, 1.0, 1.0, false);
        assert_eq!(mesh.n_elements(), 8);
        let mut mesh = mesh;
        mesh.set_curvature(2);
        let order = 2u8;
        let dm = DofManager::new(&mesh, order);
        let topo: &dyn MeshTopology = &mesh;
        let mut form = TmopForm::new(topo, &dm, order, TmopQuadType::GaussLegendre, 8);
        let x0 = linear_mesh_positions(topo, &dm, order, 3);
        form.set_x0(x0);
        let zeros = vec![0.0f64; form.n_dofs()];
        let d = form.min_det_j(&zeros);
        assert!((d - 0.125).abs() < 1e-12, "min det = {}", d);
    }

    /// Multi-element gradient must match central FD differences of the energy
    /// at every free dof.  Guards against cross-element accumulation of the
    /// reused element-vector scratch buffer (2×2 quad mesh, order 1 — every
    /// dof is shared by ≥ 2 elements, so a per-element pollution shows up
    /// immediately).
    #[test]
    fn gradient_fd_multi_element() {
        let mesh: Mesh<2> = Mesh::make_cartesian_2d(2, 2, 1.0, 1.0);
        let order = 1u8;
        let dm = DofManager::new(&mesh, order);
        let topo: &dyn MeshTopology = &mesh;
        let mut form = TmopForm::new(topo, &dm, order, TmopQuadType::GaussLobatto, 8);
        let mut x0 = linear_mesh_positions(topo, &dm, order, 2);
        // Perturb x0 so the mesh is not at an energy stationary point.
        for (i, v) in x0.iter_mut().enumerate() {
            *v += 0.01 * (((i % 5) as f64) - 2.0);
        }
        form.set_x0(x0);
        form.push_integrator(TmopIntegrator {
            metric: metric_from_id_2d(1, &SharedMinDet::new(0.0)).unwrap(),
            target: TmopTarget::new(TmopTargetType::IdealShapeUnitSize),
            coeff: 1.0,
            surf_fit: None,
        });
        form.finalize_targets();

        let n = form.n_dofs();
        let dx = vec![0.0f64; n];
        let mut g = vec![0.0f64; n];
        form.gradient(&dx, &mut g);
        let h = 1e-6;
        for k in (0..n).step_by(3) {
            let mut p = dx.clone();
            p[k] += h;
            let mut m = dx.clone();
            m[k] -= h;
            let fd = (form.energy(&p) - form.energy(&m)) / (2.0 * h);
            assert!(
                (fd - g[k]).abs() < 1e-5,
                "grad mismatch at {k}: analytic {} fd {}",
                g[k],
                fd
            );
        }
    }

    /// Same check for quads.
    #[test]
    fn quad_dof_order_matches_reference() {
        let mesh: Mesh<2> = Mesh::make_cartesian_2d(1, 1, 1.0, 1.0);
        let dm = DofManager::new(&mesh, 2);
        let quad = fem_element::lagrange::factory::QuadQk::new(2);
        let coords = quad.dof_coords();
        for e in 0..mesh.n_elements() {
            let edofs = dm.element_dofs(e as u32);
            for (k, &dof) in edofs.iter().enumerate() {
                let dc = dm.dof_coord(dof);
                let hc = &coords[k];
                for d in 0..2 {
                    assert!(
                        (dc[d] - hc[d]).abs() < 1e-12,
                        "elem {} dof {} mismatch {:?} vs {:?}",
                        e,
                        k,
                        dc,
                        hc
                    );
                }
            }
        }
    }

    /// Hex reference element may live on [-1,1]^3 while DofManager dof
    /// coordinates are on [0,1]^3 — compare after normalizing the domain
    /// (detected once from the first dof).
    #[test]
    fn hex_dof_order_matches_reference_normalized() {
        let mesh: Mesh<3> =
            Mesh::make_cartesian_3d(1, 1, 1, ElementType::Hex8, 1.0, 1.0, 1.0, false);
        let dm = DofManager::new(&mesh, 2);
        let hex = fem_element::lagrange::factory::HexQk::new(2);
        let coords = hex.dof_coords();
        let minus_one_domain = coords[0][0] < -0.25;
        for e in 0..mesh.n_elements() {
            let edofs = dm.element_dofs(e as u32);
            assert_eq!(edofs.len(), coords.len());
            for (k, &dof) in edofs.iter().enumerate() {
                let dc = dm.dof_coord(dof);
                let hc = &coords[k];
                for d in 0..3 {
                    let hn = if minus_one_domain { (hc[d] + 1.0) / 2.0 } else { hc[d] };
                    assert!(
                        (dc[d] - hn).abs() < 1e-12,
                        "elem {} dof {} mismatch {:?} vs {:?}",
                        e,
                        k,
                        dc,
                        hc
                    );
                }
            }
        }
    }

    /// One linear quad mesh, metric 1 (|T|^2), unit-size target: the optimum is
    /// the identity displacement; energy at x0 equals NE * int |J|^2 dx.
    #[test]
    fn energy_one_quad() {
        let mesh: Mesh<2> = Mesh::make_cartesian_2d(1, 1, 1.0, 1.0);
        let order = 2u8;
        let dm = DofManager::new(&mesh, order);
        let topo: &dyn MeshTopology = &mesh;
        let mut form = TmopForm::new(
            topo,
            &dm,
            order,
            TmopQuadType::GaussLegendre,
            4,
        );
        // x0 = positions by interpolation of the linear geometry.
        let x0 = linear_mesh_positions(topo, &dm, order, 2);
        form.set_x0(x0.clone());
        let min_det = SharedMinDet::new(0.0);
        form.push_integrator(TmopIntegrator {
            metric: metric_from_id_2d(1, &min_det).unwrap(),
            target: TmopTarget::new(TmopTargetType::IdealShapeUnitSize),
            coeff: 1.0,
            surf_fit: None,
        });
        form.finalize_targets();
        let e = form.energy(&vec![0.0; form.n_dofs()]);
        // Unit square: Jpr = Jtr = I, mu_1 = |T|^2 = FNorm^2(I) = 2.
        assert!((e - 2.0).abs() < 1e-12, "energy = {}", e);
        // Gradient/Hessian consistency at a perturbed point via FD.
        let n = form.n_dofs();
        let mut dx = vec![0.0; n];
        for (i, v) in dx.iter_mut().enumerate() {
            *v = 0.01 * ((i % 7) as f64 - 3.0);
        }
        let mut g = vec![0.0; n];
        form.gradient(&dx, &mut g);
        let h = 1e-7;
        for i in (0..n).step_by(3) {
            let mut dp = dx.clone();
            dp[i] += h;
            let fd = (form.energy(&dp) - form.energy(&dx)) / h;
            assert!(
                (fd - g[i]).abs() < 1e-4,
                "grad mismatch at {}: analytic {} fd {}",
                i,
                g[i],
                fd
            );
        }
    }

    /// Surface fitting to prescribed positions (fit-node-position machinery):
    /// the fitting term enters energy/gradient/Hessian as the exact quadratic
    /// sum_i c/count_i * |x_i - x_{t,i}|^2 / 2, checked against finite
    /// differences on a 2x2 quad mesh with the y=0 dofs fitted.
    #[test]
    fn surf_fit_energy_gradient_hessian_fd() {
        let mesh: Mesh<2> = Mesh::make_cartesian_2d(2, 2, 1.0, 1.0);
        let order = 2u8;
        let dm = DofManager::new(&mesh, order);
        let topo: &dyn MeshTopology = &mesh;
        let mut form = TmopForm::new(topo, &dm, order, TmopQuadType::GaussLegendre, 7);
        let x0 = linear_mesh_positions(topo, &dm, order, 2);
        form.set_x0(x0);
        let n = dm.n_dofs;
        let mut marker = vec![false; n];
        let mut target = form.x0().to_vec();
        let mut n_marked = 0;
        for dof in 0..n {
            if dm.dof_coord(dof as u32)[1] == 0.0 {
                marker[dof] = true;
                target[n + dof] -= 0.1;
                n_marked += 1;
            }
        }
        assert!(n_marked > 0);
        let dof_count = count_elements_per_dof(topo, &dm);
        form.push_integrator(TmopIntegrator {
            metric: metric_from_id_2d(2, &SharedMinDet::new(0.0)).unwrap(),
            target: TmopTarget::new(TmopTargetType::IdealShapeUnitSize),
            coeff: 1.0,
            surf_fit: Some(SurfFitPos::new(
                target.clone(),
                marker.clone(),
                dof_count.clone(),
                100.0,
            )),
        });
        form.finalize_targets();

        // Fitting errors at dx = 0: every marked dof is off by exactly 0.1.
        let zeros = vec![0.0_f64; form.n_dofs()];
        let (avg, max) = form.surf_fit_errors(&zeros);
        assert!((max - 0.1).abs() < 1e-14, "max err = {max}");
        assert!((avg - 0.1).abs() < 1e-14, "avg err = {avg}");

        // Energy: metric part (mu_2 = 0 at J = I) + fit part. Total per-dof
        // fitting weight: each element contributes c/count, so a dof shared by
        // `count` elements sums up to the full weight c.
        let mut fit = 0.0;
        for dof in 0..n {
            if marker[dof] {
                fit += 100.0 * 0.5 * (0.1 * 0.1);
            }
        }
        let e0 = form.energy(&zeros);
        assert!((e0 - fit).abs() < 1e-12, "energy = {e0}, fit = {fit}");

        // Gradient and Hessian vs finite differences at a perturbed point.
        let n_tot = form.n_dofs();
        let mut dx = vec![0.0_f64; n_tot];
        for (i, v) in dx.iter_mut().enumerate() {
            *v = 0.01 * (((i * 7) % 11) as f64 - 5.0);
        }
        let mut g = vec![0.0_f64; n_tot];
        form.gradient(&dx, &mut g);
        let h = 1e-6;
        for k in (0..n_tot).step_by(5) {
            let mut p = dx.clone();
            p[k] += h;
            let mut m = dx.clone();
            m[k] -= h;
            let fd = (form.energy(&p) - form.energy(&m)) / (2.0 * h);
            assert!(
                (fd - g[k]).abs() < 1e-5,
                "grad mismatch at {k}: analytic {} fd {}",
                g[k],
                fd
            );
        }
        // Hessian: the metric Hessian is the exact second derivative (check by
        // FD of the metric-only gradient); the surf-fit Hessian is MFEM's
        // Gauss-Newton form c*I per marked dof (block-diagonal, no gradient
        // outer product) — check it analytically.
        let hess = form.hessian(&dx);
        let mut v = vec![0.0_f64; n_tot];
        for (i, vi) in v.iter_mut().enumerate() {
            *vi = (((i * 13) % 7) as f64 - 3.0) * 0.05;
        }
        {
            // Metric-only form: H v vs FD of gradient.
            let mut f2 = TmopForm::new(topo, &dm, order, TmopQuadType::GaussLegendre, 7);
            f2.set_x0(form.x0().to_vec());
            f2.push_integrator(TmopIntegrator {
                metric: metric_from_id_2d(2, &SharedMinDet::new(0.0)).unwrap(),
                target: TmopTarget::new(TmopTargetType::IdealShapeUnitSize),
                coeff: 1.0,
                surf_fit: None,
            });
            f2.finalize_targets();
            let hm = f2.hessian(&dx);
            let mut hv = vec![0.0_f64; n_tot];
            hm.spmv(&v, &mut hv);
            let eps = 1e-4;
            let mut xp = dx.clone();
            let mut xm = dx.clone();
            for i in 0..n_tot {
                xp[i] += eps * v[i];
                xm[i] -= eps * v[i];
            }
            let mut gp = vec![0.0_f64; n_tot];
            let mut gm = vec![0.0_f64; n_tot];
            f2.gradient(&xp, &mut gp);
            f2.gradient(&xm, &mut gm);
            for k in 0..n_tot {
                let fd = (gp[k] - gm[k]) / (2.0 * eps);
                assert!(
                    (fd - hv[k]).abs() < 1e-5,
                    "metric hess action mismatch at {k}: analytic {} fd {}",
                    hv[k],
                    fd
                );
            }
        }
        // Surf-fit part: (H_total - H_metric) v == c * v at the marked dofs'
        // rows (c = 100, all components), 0 elsewhere.
        let hv_metric = {
            let mut f2 = TmopForm::new(topo, &dm, order, TmopQuadType::GaussLegendre, 7);
            f2.set_x0(form.x0().to_vec());
            f2.push_integrator(TmopIntegrator {
                metric: metric_from_id_2d(2, &SharedMinDet::new(0.0)).unwrap(),
                target: TmopTarget::new(TmopTargetType::IdealShapeUnitSize),
                coeff: 1.0,
                surf_fit: None,
            });
            f2.finalize_targets();
            let hm = f2.hessian(&dx);
            let mut hv = vec![0.0_f64; n_tot];
            hm.spmv(&v, &mut hv);
            hv
        };
        let mut hv_tot = vec![0.0_f64; n_tot];
        hess.spmv(&v, &mut hv_tot);
        for k in 0..n_tot {
            let surf = hv_tot[k] - hv_metric[k];
            let expect = if marker[k % n] { 100.0 * v[k] } else { 0.0 };
            assert!(
                (surf - expect).abs() < 1e-9,
                "surf hess mismatch at {k}: {} vs {}",
                surf,
                expect
            );
        }
    }

    /// `minres_mfem` solves a small SPD system to the requested tolerance.
    #[test]
    fn minres_mfem_spd_system() {
        // H = tridiag(-1, 4, -1), n = 20; well-conditioned SPD.
        let n = 20;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 4.0);
            if i > 0 {
                coo.add(i, i - 1, -1.0);
            }
            if i + 1 < n {
                coo.add(i, i + 1, -1.0);
            }
        }
        let h = coo.into_csr();
        let b: Vec<f64> = (0..n).map(|i| ((i % 5) as f64) - 2.0).collect();
        let mut x = vec![0.0_f64; n];
        minres_mfem(&h, &b, &mut x, 1e-12, 0.0, 100);
        let mut r = vec![0.0_f64; n];
        h.spmv(&x, &mut r);
        for i in 0..n {
            assert!(
                (r[i] - b[i]).abs() < 1e-9,
                "residual at {i}: {} vs {}",
                r[i],
                b[i]
            );
        }
    }
}

