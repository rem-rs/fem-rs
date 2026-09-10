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
//!
//! Scope (v2 additions):
//! - Limiting of the mesh displacements (`TMOP_Integrator::EnableLimiting`
//!   with `TMOP_QuadraticLimiter` / `TMOP_ExponentialLimiter`), with a uniform
//!   limiting distance (the C++ driver's constant `dist` grid function) and a
//!   constant limiting weight `lim_coeff` (`ConstantCoefficient`).
//! - Normalization (`EnableNormalization` / `ComputeNormalizationEnergies`):
//!   `metric_normal = 1/E_metric(x0)`, `lim_normal = 1/E_lim(x0)`, and the
//!   resulting `surf_fit_normal`.
//! - Discrete-adaptivity target construction (`DiscreteAdaptTC` with
//!   IDEAL_SHAPE_GIVEN_SIZE / GIVEN_SHAPE_AND_SIZE and discrete size /
//!   aspect-ratio / skew / orientation coefficient fields), including the
//!   `UpdateTargetSpecification` remap of the fields onto the moving mesh
//!   through the `AdvectorCG` / `InterpolatorFP` evaluators
//!   ([`TmopRemapEvaluator`], driven by `TmopForm::process_new_state` at the
//!   exact `TMOPNewtonSolver::ProcessNewState` call points).
//! - Adaptive limiting (`TMOP_Integrator::EnableAdaptiveLimiting`): the
//!   limiting fields are remapped like the target specification and penalized
//!   quadratically (`-alc` of mesh-optimizer).

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
use std::cell::{Cell, RefCell};
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
    /// IDEAL_SHAPE_GIVEN_SIZE built by a `DiscreteAdaptTC` with a discrete
    /// size field (mesh-optimizer target_id 5).
    IdealShapeGivenSizeDiscrete,
    /// GIVEN_SHAPE_AND_SIZE built by a `DiscreteAdaptTC` with discrete
    /// size/aspect-ratio/skew/orientation fields (mesh-optimizer target ids
    /// 6/7/8).
    GivenShapeAndSizeDiscrete,
}

/// Target constructor state for one integrator (MFEM `TargetConstructor`).
#[derive(Clone)]
pub struct TmopTarget {
    pub target_type: TmopTargetType,
    /// MFEM `SetVolumeScale` (IDEAL_SHAPE_EQUAL_SIZE).
    pub volume_scale: f64,
    /// MFEM `ComputeAvgVolume`: total physical volume / NE (from `x0`).
    pub avg_volume: f64,
    /// Discrete-adaptivity specification (`DiscreteAdaptTC` fields); required
    /// by the `*Discrete` target types, `None` otherwise.
    pub discrete: Option<TmopDiscreteSpec>,
}

impl TmopTarget {
    pub fn new(target_type: TmopTargetType) -> Self {
        Self {
            target_type,
            volume_scale: 1.0,
            avg_volume: 0.0,
            discrete: None,
        }
    }

    /// MFEM `TargetConstructor::ContainsVolumeInfo`.
    pub fn contains_volume_info(&self) -> bool {
        match self.target_type {
            TmopTargetType::IdealShapeUnitSize => false,
            TmopTargetType::IdealShapeEqualSize
            | TmopTargetType::IdealShapeGivenSize
            | TmopTargetType::IdealShapeGivenSizeDiscrete
            | TmopTargetType::GivenShapeAndSizeDiscrete => true,
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
    /// MFEM `surf_fit_normal` (1.0 unless normalization is enabled, in which
    /// case it becomes the limiting normalization factor).
    pub normal: Cell<f64>,
}

impl SurfFitPos {
    pub fn new(pos: Vec<f64>, marker: Vec<bool>, dof_count: Vec<f64>, coeff: f64) -> Self {
        Self {
            pos: Rc::new(pos),
            marker: Rc::new(marker),
            dof_count: Rc::new(dof_count),
            coeff: Rc::new(Cell::new(coeff)),
            normal: Cell::new(1.0),
        }
    }
}

/// MFEM `TMOP_LimiterFunction` (fem/tmop.hpp): the scalar limiting function
/// f(x, x0, d) of the physical position x, the reference position x0 and the
/// limiting distance d.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TmopLimiterFunction {
    /// `TMOP_QuadraticLimiter` (the MFEM default limiter).
    Quadratic,
    /// `TMOP_ExponentialLimiter`.
    Exponential,
}

/// `|x - x0|^2` (MFEM `Vector::DistanceSquaredTo`: ascending component sum).
fn dist_sq(x: &[f64], x0: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..x.len() {
        let d = x[i] - x0[i];
        sum += d * d;
    }
    sum
}

impl TmopLimiterFunction {
    /// MFEM `TMOP_LimiterFunction::Eval(x, x0, dist)`.
    pub fn eval(&self, x: &[f64], x0: &[f64], dist: f64) -> f64 {
        let dist2 = dist * dist;
        match self {
            Self::Quadratic => 0.5 * dist_sq(x, x0) / dist2,
            Self::Exponential => (10.0 * (dist_sq(x, x0) / dist2 - 1.0)).exp(),
        }
    }

    /// MFEM `TMOP_LimiterFunction::Eval_d1`: gradient w.r.t. x.
    pub fn eval_d1(&self, x: &[f64], x0: &[f64], dist: f64, d1: &mut [f64]) {
        let dist2 = dist * dist;
        match self {
            Self::Quadratic => {
                // subtract(1.0/(dist*dist), x, x0, d1)
                let c = 1.0 / dist2;
                for i in 0..x.len() {
                    d1[i] = c * (x[i] - x0[i]);
                }
            }
            Self::Exponential => {
                let c = 20.0 * (10.0 * (dist_sq(x, x0) / dist2 - 1.0)).exp() / dist2;
                for i in 0..x.len() {
                    d1[i] = c * (x[i] - x0[i]);
                }
            }
        }
    }

    /// MFEM `TMOP_LimiterFunction::Eval_d2`: Hessian w.r.t. x, written
    /// row-major into `d2` (dim x dim).
    pub fn eval_d2(&self, x: &[f64], x0: &[f64], dist: f64, d2: &mut [f64]) {
        let dim = x.len();
        let dist2 = dist * dist;
        match self {
            Self::Quadratic => {
                let c = 1.0 / dist2;
                for i in 0..dim {
                    for j in 0..dim {
                        d2[i * dim + j] = if i == j { c } else { 0.0 };
                    }
                }
            }
            Self::Exponential => {
                // TMOP_ExponentialLimiter::Eval_d2.
                let f = (10.0 * (dist_sq(x, x0) / dist2 - 1.0)).exp();
                let d2s2 = dist2 * dist2;
                let tmp: Vec<f64> = (0..dim).map(|i| x[i] - x0[i]).collect();
                for i in 0..dim {
                    for j in 0..dim {
                        d2[i * dim + j] = 400.0 * tmp[i] * tmp[j] * f / d2s2;
                    }
                    d2[i * dim + i] += 20.0 * f / dist2;
                }
            }
        }
    }
}

/// The limiting term of one `TMOP_Integrator` (MFEM `EnableLimiting(n0, w0,
/// lfunc)` / `EnableLimiting(n0, dist, w0, lfunc)`): `lim_nodes0` (the
/// reference positions), `lim_coeff` (a constant `Coefficient`), the uniform
/// value of the `dist` grid function (the C++ driver keeps `dist` constant in
/// space, so the `GridFunction::GetValues` quadrature values reduce to the Qk
/// interpolation of that constant), the limiter function and the
/// normalization factor `lim_normal`.
#[derive(Clone)]
pub struct TmopLimiting {
    /// MFEM `lim_nodes0`: reference positions, layout `[c*n_scalar + s]`.
    pub nodes0: Rc<Vec<f64>>,
    /// MFEM `lim_coeff` (ConstantCoefficient).
    pub coeff: f64,
    /// The constant value of the `dist` grid function dofs (1.0 by default in
    /// the C++ driver; `small_phys_size` with normalization).
    pub dist: f64,
    /// MFEM `lim_func`.
    pub lim_func: TmopLimiterFunction,
    /// MFEM `lim_normal` (set by `TmopForm::enable_normalization`).
    pub normal: f64,
}

/// Discrete-adaptivity target specification (MFEM `DiscreteAdaptTC`): the
/// geometric parameters of the target given as FE fields. The fields are
/// stored as scalar dof values of an H1 indicator space (Ordering::byNODES,
/// component-major), evaluated at the metric quadrature points through that
/// space's shape functions.
///
/// MFEM remaps these fields onto the moved mesh at every Newton step
/// (`DiscreteAdaptTC::UpdateTargetSpecification` through `AdvectorCG` /
/// `InterpolatorFP`, see [`TmopRemapEvaluator`]). The remapped values are
/// stored in `remapped` (packed byNODES in the field order size, aspect
/// ratio, skew, orientation) and read by the target construction once the
/// remap has been set up with `TmopForm::set_discrete_remapper`; before the
/// first remap the initial fields are used.
#[derive(Clone)]
pub struct TmopDiscreteSpec {
    /// Target size eta(x) dofs (`SetSerialDiscreteTargetSize`).
    pub size: Option<Rc<Vec<f64>>>,
    /// Target aspect ratio dofs, 1 component in 2D, 3 in 3D
    /// (`SetSerialDiscreteTargetAspectRatio`).
    pub aspect_ratio: Option<Rc<Vec<f64>>>,
    /// Target skew dofs, 1 component in 2D, 3 in 3D
    /// (`SetSerialDiscreteTargetSkew`).
    pub skew: Option<Rc<Vec<f64>>>,
    /// Target orientation dofs, 1 component in 2D, 3 in 3D
    /// (`SetSerialDiscreteTargetOrientation`).
    pub orientation: Option<Rc<Vec<f64>>>,
    /// `SetMinSizeForTargets` (MFEM `lim_min_size`; negative disables it).
    pub min_size: f64,
    /// Order of the indicator FE space (C++ `ind_fec_order`).
    pub order: u8,
    /// Number of scalar indicator dofs.
    pub n_ind: usize,
    /// Indicator-space scalar dof table per mesh element.
    pub element_dofs: Rc<Vec<Vec<usize>>>,
    /// Per-element indicator shape values at the metric quadrature points,
    /// `[e][q * nd_ind + k]`, prepared by `TmopForm::finalize_targets`.
    pub qp_shapes: Option<Rc<Vec<Vec<f64>>>>,
    /// MFEM `tspec`: the packed initial field values (component order: size,
    /// aspect ratio, skew, orientation).
    pub(crate) packed0: Rc<Vec<f64>>,
    /// MFEM `tspec` after the last `UpdateTargetSpecification` remap; `None`
    /// until a remap evaluator is attached and the solver calls
    /// `TmopForm::process_new_state`.
    pub(crate) remapped: RefCell<Option<Rc<Vec<f64>>>>,
}

impl TmopDiscreteSpec {
    /// Create a spec from the indicator-space dof tables (the values are
    /// provided component-wise in `size`/`aspect_ratio`/`skew`/`orientation`,
    /// each of length `n_ind`, component-major byNODES layout).
    pub fn new(
        size: Option<Rc<Vec<f64>>>,
        aspect_ratio: Option<Rc<Vec<f64>>>,
        skew: Option<Rc<Vec<f64>>>,
        orientation: Option<Rc<Vec<f64>>>,
        min_size: f64,
        order: u8,
        n_ind: usize,
        element_dofs: Rc<Vec<Vec<usize>>>,
    ) -> Self {
        let mut packed0 = Vec::new();
        for f in [&size, &aspect_ratio, &skew, &orientation] {
            if let Some(v) = f {
                packed0.extend_from_slice(v);
            }
        }
        Self {
            size,
            aspect_ratio,
            skew,
            orientation,
            min_size,
            order,
            n_ind,
            element_dofs,
            qp_shapes: None,
            packed0: Rc::new(packed0),
            remapped: RefCell::new(None),
        }
    }

    /// Number of packed components (`DiscreteAdaptTC::ncomp`).
    pub fn ncomp(&self) -> usize {
        self.packed0.len() / self.n_ind
    }

    /// The current tspec values: the last remap result, or the initial fields.
    pub(crate) fn packed_current(&self) -> Vec<f64> {
        match self.remapped.borrow().as_ref() {
            Some(v) => v.as_ref().clone(),
            None => self.packed0.as_ref().clone(),
        }
    }

    /// The packed values the element-target construction reads (the remapped
    /// state when present, the initial fields otherwise).
    fn current(&self) -> Rc<Vec<f64>> {
        match self.remapped.borrow().as_ref() {
            Some(v) => v.clone(),
            None => self.packed0.clone(),
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

// ─── Field remap evaluators (fem/tmop_tools.cpp) ─────────────────────────────

/// The evaluator kind used to remap discrete fields onto the moving mesh
/// (mesh-optimizer `-ae`): `AdvectorCG` (0, the default) or `InterpolatorFP`
/// (1, MFEM_USE_GSLIB path).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TmopRemapKind {
    /// MFEM `AdvectorCG`: conservative remap by advecting the field with the
    /// mesh displacement velocity (RK4 in time + CG solve of the mass matrix).
    AdvectorCG,
    /// MFEM `InterpolatorFP`: pointwise interpolation of the initial field at
    /// the new mesh node positions.
    InterpolatorFP,
}

/// One scalar FE space on the mesh, owned by a remap evaluator (MFEM keeps
/// these in the copied `Mesh` / `FiniteElementSpace`).
struct RemapSpaceElem {
    dofs: Vec<usize>,
    re: Box<dyn ReferenceElement>,
}

/// MFEM `AdvectorCG` / `InterpolatorFP` (serial, H1 fields, quad/hex meshes):
/// remaps a packed byNODES multi-component field from the initial mesh
/// positions (`nodes0`) to new mesh positions.
///
/// Like the C++ evaluators, the state is *incremental*: after every
/// `compute_at_new_position` the internal `nodes0`/`field0` are updated, so
/// consecutive remaps transport the field through the increment only.
pub struct TmopRemapEvaluator {
    kind: TmopRemapKind,
    dim: usize,
    /// Nodal (mesh geometry) space.
    elems_nodal: Vec<RemapSpaceElem>,
    n_nodal: usize,
    /// Field (advection) space.
    elems_field: Vec<RemapSpaceElem>,
    n_field: usize,
    /// MFEM `AdvectorCG::dt_scale` (0.5).
    dt_scale: f64,
    /// MFEM `nodes0` / `field0`: incremental remap state (packed byNODES).
    nodes0: Vec<f64>,
    field0: Vec<f64>,
    ncomp: usize,
    /// Per-element quadrature rules of the advection operators (mass:
    /// `trial+test+OrderW`, convection: `OrderGrad+Order+el_order`, both
    /// Gauss-Legendre tensor rules as in MFEM's `IntRules`).
    quad_mass: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
    quad_conv: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
    /// InterpolatorFP: per-element bounding boxes of the initial geometry and
    /// the field-space dof coordinates.
    interp_boxes: Vec<[f64; 6]>,
    field_dof_coords: Vec<Vec<f64>>,
}

impl TmopRemapEvaluator {
    /// Build the two spaces (MFEM
    /// `AdaptivityEvaluator::SetSerialMetaInfo(m, f)` copies the mesh and
    /// creates the field space on it).
    pub fn new(
        kind: TmopRemapKind,
        topo: &dyn MeshTopology,
        dm_nodal: &DofManager,
        order_nodal: u8,
        dm_field: &DofManager,
        order_field: u8,
        dt_scale: f64,
    ) -> Self {
        let dim = topo.dim() as usize;
        let elem = |dm: &DofManager, order: u8, e: u32| -> RemapSpaceElem {
            let raw = ref_elem_for(topo.element_type(e), order);
            let re: Box<dyn ReferenceElement> = if el_domain_is_unit(raw.as_ref()) {
                raw
            } else {
                Box::new(UnitDomainElem {
                    inner: raw,
                    dim,
                })
            };
            RemapSpaceElem {
                dofs: dm
                    .element_dofs(e)
                    .iter()
                    .map(|&d| d as usize)
                    .collect(),
                re,
            }
        };
        let ne = topo.n_elements() as usize;
        let elems_nodal: Vec<RemapSpaceElem> =
            (0..ne as u32).map(|e| elem(dm_nodal, order_nodal, e)).collect();
        let elems_field: Vec<RemapSpaceElem> =
            (0..ne as u32).map(|e| elem(dm_field, order_field, e)).collect();
        // Advection quadrature rules (only needed by AdvectorCG; MFEM
        // ConvectionIntegrator: order = OrderGrad + Order + test order with
        // IsoparametricTransformation OrderGrad = k*(d-1)+(l-1), Order = k for
        // Qk spaces; MassIntegrator: order = trial + test + OrderW with
        // OrderW = k*d - 1).
        let mut quad_mass = Vec::with_capacity(ne);
        let mut quad_conv = Vec::with_capacity(ne);
        for e in 0..ne {
            let (mass, conv) = if kind == TmopRemapKind::AdvectorCG {
                let k = order_nodal as usize;
                let l = order_field as usize;
                let d = dim;
                let m_order = (l + l + (k * d - 1)) as u8;
                let c_order = (k * (d - 1) + (l - 1) + k + l) as u8;
                let rule = |re: &dyn ReferenceElement, order: u8| {
                    let r = re.quadrature(order);
                    (r.points, r.weights)
                };
                (
                    rule(elems_field[e].re.as_ref(), m_order),
                    rule(elems_field[e].re.as_ref(), c_order),
                )
            } else {
                ((Vec::new(), Vec::new()), (Vec::new(), Vec::new()))
            };
            quad_mass.push(mass);
            quad_conv.push(conv);
        }
        let field_dof_coords: Vec<Vec<f64>> = elems_field
            .first()
            .map(|el| el.re.dof_coords())
            .unwrap_or_default();
        Self {
            kind,
            dim,
            n_nodal: dm_nodal.n_dofs,
            n_field: dm_field.n_dofs,
            dt_scale,
            nodes0: Vec::new(),
            field0: Vec::new(),
            ncomp: 0,
            quad_mass,
            quad_conv,
            interp_boxes: Vec::new(),
            field_dof_coords,
            elems_nodal,
            elems_field,
        }
    }

    /// MFEM `AdaptivityEvaluator::SetInitialField`.
    pub fn set_initial_field(&mut self, init_nodes: &[f64], init_field: &[f64]) {
        assert_eq!(init_nodes.len(), self.dim * self.n_nodal);
        assert_eq!(init_field.len() % self.n_field, 0);
        self.nodes0 = init_nodes.to_vec();
        self.field0 = init_field.to_vec();
        self.ncomp = init_field.len() / self.n_field;
        if self.kind == TmopRemapKind::InterpolatorFP {
            // Element bounding boxes of the initial geometry (findpoints).
            self.interp_boxes = self
                .elems_nodal
                .iter()
                .map(|el| {
                    let mut box_ = [f64::INFINITY; 6];
                    for &dof in &el.dofs {
                        for c in 0..self.dim {
                            let v = init_nodes[c * self.n_nodal + dof];
                            box_[c] = box_[c].min(v);
                            box_[c + 3] = box_[c + 3].max(v);
                        }
                    }
                    box_
                })
                .collect();
        }
    }

    /// MFEM `AdaptivityEvaluator::ComputeAtNewPosition` (byNODES).
    pub fn compute_at_new_position(&mut self, new_mesh_nodes: &[f64], new_field: &mut Vec<f64>) {
        assert_eq!(self.nodes0.len(), new_mesh_nodes.len());
        assert_eq!(new_field.len(), self.ncomp * self.n_field);
        new_field.copy_from_slice(&self.field0);
        match self.kind {
            TmopRemapKind::AdvectorCG => {
                for c in 0..self.ncomp {
                    self.advector_scalar(new_mesh_nodes, new_field, c);
                }
            }
            TmopRemapKind::InterpolatorFP => {
                self.interpolator(new_mesh_nodes, new_field);
            }
        }
        // Without this, the next remap would start from the initial mesh, i.e.,
        // every consecutive remap would be more expensive (C++ comment).
        self.field0.copy_from_slice(new_field);
        self.nodes0.copy_from_slice(new_mesh_nodes);
    }

    /// Minimum MFEM `Mesh::GetElementSize(i)` over the mesh at `nodes`
    /// (type 0 size: `pow(|det J(center)|, 1/dim)` of the nodal geometry).
    fn h_min(&self, nodes: &[f64]) -> f64 {
        let dim = self.dim;
        let center = vec![0.5_f64; dim];
        let mut h_min = f64::INFINITY;
        for el in &self.elems_nodal {
            let nd = el.dofs.len();
            let mut dsh = vec![0.0_f64; nd * dim];
            el.re.eval_grad_basis(&center, &mut dsh);
            let mut jpr = [[0.0_f64; 3]; 3];
            for a in 0..dim {
                for b in 0..dim {
                    let mut s = 0.0;
                    for (i, &dof) in el.dofs.iter().enumerate() {
                        s += nodes[a * self.n_nodal + dof] * dsh[i * dim + b];
                    }
                    jpr[a][b] = s;
                }
            }
            let det = det_small(&jpr, dim);
            let h = det.abs().powf(1.0 / dim as f64);
            h_min = h_min.min(h);
        }
        h_min
    }

    /// MFEM `AdvectorCG::ComputeAtNewPositionScalar` for component `c`.
    fn advector_scalar(&self, new_mesh_nodes: &[f64], new_field: &mut [f64], c: usize) {
        let nf = self.n_field;
        let nn = self.n_nodal;
        let comp = &mut new_field[c * nf..(c + 1) * nf];
        let minv = comp.iter().cloned().fold(f64::INFINITY, f64::min);
        let maxv = comp.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        // Velocity of the positions: u = new_mesh_nodes - nodes0 (on the
        // nodal space).
        let u: Vec<f64> = (0..self.dim * nn)
            .map(|i| new_mesh_nodes[i] - self.nodes0[i])
            .collect();
        // Compute some time step [mesh_size / speed].
        let h_min = self.h_min(&self.nodes0);
        let mut v2_max = 0.0_f64;
        for i in 0..nn {
            let mut vel = 0.0;
            for d in 0..self.dim {
                vel += u[d * nn + i] * u[d * nn + i];
            }
            v2_max = v2_max.max(vel);
        }
        if v2_max == 0.0 {
            // No need to change the field.
            return;
        }
        let v_max = v2_max.sqrt();
        let dt = self.dt_scale * h_min / v_max;
        let mut t = 0.0_f64;
        let mut last_step = false;
        while !last_step {
            let mut step = dt;
            if t + dt >= 1.0 {
                step = 1.0 - t;
                last_step = true;
            }
            self.rk4_step(comp, &u, &mut t, step);
        }
        // Trim the overshoots and undershoots.
        for v in comp.iter_mut() {
            if *v < minv {
                *v = minv;
            }
            if *v > maxv {
                *v = maxv;
            }
        }
    }

    /// MFEM `RK4Solver::Step` applied to the scalar component with the
    /// advection operator evaluated at the moving mesh (`f->SetTime(t)`).
    fn rk4_step(&self, x: &mut [f64], u: &[f64], t: &mut f64, dt: f64) {
        let k1 = self.advector_mult(x, u, *t);
        let y: Vec<f64> = x.iter().zip(k1.iter()).map(|(a, b)| a + dt / 2.0 * b).collect();
        let mut z: Vec<f64> = x.iter().zip(k1.iter()).map(|(a, b)| a + dt / 6.0 * b).collect();

        let k2 = self.advector_mult(&y, u, *t + dt / 2.0);
        let y: Vec<f64> = x.iter().zip(k2.iter()).map(|(a, b)| a + dt / 2.0 * b).collect();
        for (zv, kv) in z.iter_mut().zip(k2.iter()) {
            *zv += dt / 3.0 * kv;
        }

        let k3 = self.advector_mult(&y, u, *t + dt / 2.0);
        let y: Vec<f64> = x.iter().zip(k3.iter()).map(|(a, b)| a + dt * b).collect();
        for (zv, kv) in z.iter_mut().zip(k3.iter()) {
            *zv += dt / 3.0 * kv;
        }

        let k4 = self.advector_mult(&y, u, *t + dt);
        for (xv, (zv, kv)) in x.iter_mut().zip(z.iter().zip(k4.iter())) {
            *xv = zv + dt / 6.0 * kv;
        }
        *t += dt;
    }

    /// MFEM `SerialAdvectorCGOper::Mult`: move the mesh to `x0 + t*u`, reassemble
    /// the convection (K) and mass (M) operators on it and solve
    /// `M di/dt = K ind` by PCG (Jacobi, rtol 1e-12, max 100 iterations).
    fn advector_mult(&self, ind: &[f64], u: &[f64], t: f64) -> Vec<f64> {
        let dim = self.dim;
        let n = self.n_nodal;
        let nf = self.n_field;
        // Current mesh positions (nodal space, byNODES).
        let xnow: Vec<f64> = (0..dim * n).map(|i| self.nodes0[i] + t * u[i]).collect();
        let ne = self.elems_field.len();
        let mut coo = CooMatrix::<f64>::new(nf, nf);
        let mut coo_m = CooMatrix::<f64>::new(nf, nf);
        let mut rhs = vec![0.0_f64; nf];
        for e in 0..ne {
            let fel = &self.elems_field[e];
            let nel = &self.elems_nodal[e];
            let ndf = fel.dofs.len();
            let (cp, cw) = &self.quad_conv[e];
            let (mp, mw) = &self.quad_mass[e];
            let mut kel = vec![0.0_f64; ndf * ndf];
            let mut mel = vec![0.0_f64; ndf * ndf];
            let mut dsh = vec![0.0_f64; ndf * dim];
            let mut sh = vec![0.0_f64; ndf];
            let mut shn = vec![0.0_f64; nel.dofs.len()];
            let mut dshn = vec![0.0_f64; nel.dofs.len() * dim];
            for q in 0..cp.len() {
                // Geometry (Jpr) of the element at the moved positions.
                nel.re.eval_basis(&cp[q], &mut shn);
                nel.re.eval_grad_basis(&cp[q], &mut dshn);
                let mut jpr = [[0.0_f64; 3]; 3];
                for a in 0..dim {
                    for b in 0..dim {
                        let mut s = 0.0;
                        for (i, &dof) in nel.dofs.iter().enumerate() {
                            s += xnow[a * n + dof] * dshn[i * dim + b];
                        }
                        jpr[a][b] = s;
                    }
                }
                // Velocity values at the quadrature point.
                let mut vec1 = [0.0_f64; 3];
                for d in 0..dim {
                    let mut s = 0.0;
                    for (i, &dof) in nel.dofs.iter().enumerate() {
                        s += u[d * n + dof] * shn[i];
                    }
                    // MFEM: vec1 = Q * (alpha * ip.weight), alpha = 1.
                    vec1[d] = s * cw[q];
                }
                // adjJ = adj(Jpr) (cofactor matrix, no det division).
                let adj = adjugate(&jpr, dim);
                let mut vec2 = [0.0_f64; 3];
                for m in 0..dim {
                    let mut s = 0.0;
                    for d in 0..dim {
                        s += adj[m][d] * vec1[d];
                    }
                    vec2[m] = s;
                }
                fel.re.eval_basis(&cp[q], &mut sh);
                fel.re.eval_grad_basis(&cp[q], &mut dsh);
                // BdFidxT(j) = dshape(j,:) . vec2 (reference gradients).
                let mut bdf = vec![0.0_f64; ndf];
                for j in 0..ndf {
                    let mut s = 0.0;
                    for m in 0..dim {
                        s += dsh[j * dim + m] * vec2[m];
                    }
                    bdf[j] = s;
                }
                // AddMultVWt(shape, BdFidxT, elmat).
                for i in 0..ndf {
                    for j in 0..ndf {
                        kel[i * ndf + j] += sh[i] * bdf[j];
                    }
                }
            }
            for q in 0..mp.len() {
                fel.re.eval_basis(&mp[q], &mut sh);
                nel.re.eval_grad_basis(&mp[q], &mut dshn);
                let mut jpr = [[0.0_f64; 3]; 3];
                for a in 0..dim {
                    for b in 0..dim {
                        let mut s = 0.0;
                        for (i, &dof) in nel.dofs.iter().enumerate() {
                            s += xnow[a * n + dof] * dshn[i * dim + b];
                        }
                        jpr[a][b] = s;
                    }
                }
                // MassIntegrator: w = ip.weight * Trans.Weight().
                let w = mw[q] * det_small(&jpr, dim);
                // AddMult_a_VVt(w, shape, elmat).
                for i in 0..ndf {
                    for j in 0..ndf {
                        mel[i * ndf + j] += w * sh[i] * sh[j];
                    }
                }
            }
            // Scatter (row = field dof).
            for (i, &gi) in fel.dofs.iter().enumerate() {
                for (j, &gj) in fel.dofs.iter().enumerate() {
                    let v = kel[i * ndf + j];
                    if v != 0.0 {
                        coo.add(gi, gj, v);
                    }
                }
                let mut acc = 0.0_f64;
                for (j, &gj) in fel.dofs.iter().enumerate() {
                    acc += kel[i * ndf + j] * ind[gj];
                }
                rhs[gi] += acc;
            }
            for (i, &gi) in fel.dofs.iter().enumerate() {
                for (j, &gj) in fel.dofs.iter().enumerate() {
                    let v = mel[i * ndf + j];
                    if v != 0.0 {
                        coo_m.add(gi, gj, v);
                    }
                }
            }
        }
        let m = coo_m.into_csr();
        // PCG (MFEM CGSolver, x0 = 0) with the Jacobi smoother as
        // preconditioner.
        pcg_jacobi(&m, &rhs, 1e-12, 100)
    }

    /// MFEM `InterpolatorFP::ComputeAtNewPosition` (same-FE-space path):
    /// interpolate the initial field at the new node positions.
    fn interpolator(&self, new_mesh_nodes: &[f64], new_field: &mut [f64]) {
        // Query positions: the field-space dof positions on the new geometry.
        let queries: Vec<Vec<f64>> = if self.n_field == self.n_nodal
            && self.field_dof_coords.len() == self.elems_nodal[0].re.dof_coords().len()
        {
            (0..self.n_field)
                .map(|i| {
                    (0..self.dim)
                        .map(|c| new_mesh_nodes[c * self.n_nodal + i])
                        .collect()
                })
                .collect()
        } else {
            // MFEM `FiniteElementSpace::GetNodePositions`: the new mesh
            // geometry evaluated at the field dof points.
            let mut mapped = vec![0.0_f64; self.dim * self.n_field];
            let ndn = self.elems_nodal[0].dofs.len();
            let mut shn = vec![0.0_f64; ndn];
            for (e, fel) in self.elems_field.iter().enumerate() {
                let nel = &self.elems_nodal[e];
                for (k, xi) in self.field_dof_coords.iter().enumerate() {
                    self.elems_nodal[e].re.eval_basis(xi, &mut shn);
                    for c in 0..self.dim {
                        let mut s = 0.0;
                        for (i, &dof) in nel.dofs.iter().enumerate() {
                            s += new_mesh_nodes[c * self.n_nodal + dof] * shn[i];
                        }
                        mapped[c * self.n_field + fel.dofs[k]] = s;
                    }
                }
                let _ = ndn;
            }
            (0..self.n_field)
                .map(|i| {
                    (0..self.dim)
                        .map(|c| mapped[c * self.n_field + i])
                        .collect()
                })
                .collect()
        };
        // FindPointsGSLIB equivalent: locate the element containing each query
        // point on the initial (nodes0) geometry by Newton inversion, then
        // interpolate the initial field there.
        for (i, p) in queries.iter().enumerate() {
            let (e, xi) = self
                .find_point(p)
                .unwrap_or_else(|| panic!("InterpolatorFP: point {:?} not found", p));
            let fel = &self.elems_field[e];
            let mut sh = vec![0.0_f64; fel.dofs.len()];
            fel.re.eval_basis(&xi, &mut sh);
            for c in 0..self.ncomp {
                let mut s = 0.0;
                for (k, &dof) in fel.dofs.iter().enumerate() {
                    s += sh[k] * self.field0[c * self.n_field + dof];
                }
                new_field[c * self.n_field + i] = s;
            }
        }
    }

    /// FindPointsGSLIB replacement: Newton inversion of the isoparametric map
    /// of every candidate element (bounding-box prefilter). Returns the
    /// element index and reference coordinates. Query points outside the
    /// initial mesh (possible when the optimized mesh bulges past its initial
    /// bounding box) are extrapolated from the best-converged candidate.
    fn find_point(&self, p: &[f64]) -> Option<(usize, Vec<f64>)> {
        let dim = self.dim;
        let tol = 1.0e-10;
        let mut best: Option<(f64, usize, Vec<f64>)> = None;
        // First pass with the bounding-box prefilter; query points outside the
        // initial mesh bounding box (optimized mesh bulging past its starting
        // shape) retry the full element list and are extrapolated.
        for use_boxes in [true, false] {
            'elements: for (e, nel) in self.elems_nodal.iter().enumerate() {
                if use_boxes {
                    let b = &self.interp_boxes[e];
                    for c in 0..dim {
                        if p[c] < b[c] - tol || p[c] > b[c + 3] + tol {
                            continue 'elements;
                        }
                    }
                } else if best.is_some() {
                    break;
                }
                let nd = nel.dofs.len();
                let mut xi: Vec<f64> = vec![0.5; dim];
                let mut sh = vec![0.0_f64; nd];
                let mut dsh = vec![0.0_f64; nd * dim];
                for _ in 0..50 {
                    nel.re.eval_basis(&xi, &mut sh);
                    let mut f = [0.0_f64; 3];
                    for c in 0..dim {
                        let mut s = 0.0;
                        for (k, &dof) in nel.dofs.iter().enumerate() {
                            s += self.nodes0[c * self.n_nodal + dof] * sh[k];
                        }
                        f[c] = s - p[c];
                    }
                    let res = f[..dim].iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
                    if res < 1.0e-12 {
                        let inside = xi.iter().enumerate().all(|(d, v)| {
                            let lo = if nel.re.dof_coords()[0][d] < -0.25 { -1.0 } else { 0.0 };
                            let hi = lo + 2.0;
                            *v >= lo - tol && *v <= hi + tol
                        });
                        if inside {
                            return Some((e, xi));
                        }
                        if best.as_ref().map_or(true, |(r, _, _)| res < *r) {
                            best = Some((res, e, xi.clone()));
                        }
                        continue 'elements;
                    }
                    if best.as_ref().map_or(true, |(r, _, _)| res < *r) {
                        best = Some((res, e, xi.clone()));
                    }
                    nel.re.eval_grad_basis(&xi, &mut dsh);
                    let mut jpr = [[0.0_f64; 3]; 3];
                    for a in 0..dim {
                        for b in 0..dim {
                            let mut s = 0.0;
                            for (i, &dof) in nel.dofs.iter().enumerate() {
                                s += self.nodes0[a * self.n_nodal + dof] * dsh[i * dim + b];
                            }
                            jpr[a][b] = s;
                        }
                    }
                    let jac = jac_inverse(&jpr, dim);
                    let mut dx = [0.0_f64; 3];
                    for m in 0..dim {
                        let mut s = 0.0;
                        for d in 0..dim {
                            s += jac[m][d] * f[d];
                        }
                        dx[m] = s;
                    }
                    for d in 0..dim {
                        xi[d] -= dx[d];
                    }
                }
            }
            if best.is_some() {
                break;
            }
        }
        best.map(|(_, e, xi)| (e, xi))
    }
}

/// MFEM `CalcAdjugate` (square): the adjugate (cofactor-transpose) matrix, no
/// determinant division.
fn adjugate(j: &[[f64; 3]; 3], dim: usize) -> [[f64; 3]; 3] {
    match dim {
        2 => [
            [j[1][1], -j[0][1], 0.0],
            [-j[1][0], j[0][0], 0.0],
            [0.0, 0.0, 0.0],
        ],
        3 => [
            [
                j[1][1] * j[2][2] - j[1][2] * j[2][1],
                j[0][2] * j[2][1] - j[0][1] * j[2][2],
                j[0][1] * j[1][2] - j[0][2] * j[1][1],
            ],
            [
                j[1][2] * j[2][0] - j[1][0] * j[2][2],
                j[0][0] * j[2][2] - j[0][2] * j[2][0],
                j[0][2] * j[1][0] - j[0][0] * j[1][2],
            ],
            [
                j[1][0] * j[2][1] - j[1][1] * j[2][0],
                j[0][1] * j[2][0] - j[0][0] * j[2][1],
                j[0][0] * j[1][1] - j[0][1] * j[1][0],
            ],
        ],
        _ => unreachable!(),
    }
}

/// Full matrix inverse (MFEM `CalcInverse`), row-major `jac[m][d]`.
fn jac_inverse(j: &[[f64; 3]; 3], dim: usize) -> [[f64; 3]; 3] {
    match dim {
        2 => {
            let flat: [f64; 4] = [j[0][0], j[0][1], j[1][0], j[1][1]];
            let inv = invert_2x2(&flat);
            [
                [inv[0], inv[1], 0.0],
                [inv[2], inv[3], 0.0],
                [0.0, 0.0, 0.0],
            ]
        }
        3 => {
            let m: [f64; 9] = [
                j[0][0], j[0][1], j[0][2], j[1][0], j[1][1], j[1][2], j[2][0], j[2][1], j[2][2],
            ];
            let i = invert_3x3(&m);
            [[i[0], i[1], i[2]], [i[3], i[4], i[5]], [i[6], i[7], i[8]]]
        }
        _ => unreachable!(),
    }
}

/// MFEM `PCG` with a Jacobi (diagonal) preconditioner: `M z = b` solved to
/// relative tolerance `rtol` (abs tol 0), at most `max_iter` iterations.
fn pcg_jacobi(m: &CsrMatrix<f64>, b: &[f64], rtol: f64, max_iter: usize) -> Vec<f64> {
    let n = b.len();
    let mut diag = vec![1.0_f64; n];
    for row in 0..n {
        let (s, e) = (m.row_ptr[row], m.row_ptr[row + 1]);
        for k in s..e {
            if m.col_idx[k] as usize == row {
                diag[row] = m.values[k];
            }
        }
        if diag[row] == 0.0 {
            diag[row] = 1.0;
        }
    }
    let mut x = vec![0.0_f64; n];
    let mut r = b.to_vec();
    let bnorm = l2_norm(b);
    if bnorm == 0.0 {
        return x;
    }
    let tol = rtol * bnorm;
    let mut z = vec![0.0_f64; n];
    let mut p = vec![0.0_f64; n];
    let mut q = vec![0.0_f64; n];
    for i in 0..n {
        z[i] = r[i] / diag[i];
    }
    p.copy_from_slice(&z);
    let mut rdotz: f64 = r.iter().zip(z.iter()).map(|(a, c)| a * c).sum();
    for _ in 0..max_iter {
        if l2_norm(&r) <= tol {
            break;
        }
        m.spmv(p.as_slice(), q.as_mut_slice());
        let pq: f64 = p.iter().zip(q.iter()).map(|(a, c)| a * c).sum();
        if pq == 0.0 {
            break;
        }
        let alpha = rdotz / pq;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }
        if l2_norm(&r) <= tol {
            break;
        }
        for i in 0..n {
            z[i] = r[i] / diag[i];
        }
        let rdotz1: f64 = r.iter().zip(z.iter()).map(|(a, c)| a * c).sum();
        let beta = rdotz1 / rdotz;
        for i in 0..n {
            p[i] = beta * p[i] + z[i];
        }
        rdotz = rdotz1;
    }
    x
}

/// Adaptive limiting term of one integrator (MFEM
/// `TMOP_Integrator::EnableAdaptiveLimiting`): the energy
/// `lim_normal * Σ_c coeff_c * ((z_c(x) - z_{0,c}(x))/delta_max_c)^2` added at
/// the metric quadrature points, where `z_c` are the limiting fields remapped
/// onto the moving mesh through a [`TmopRemapEvaluator`].
pub struct TmopAdaptiveLimiting {
    /// MFEM `adapt_lim_gf0`: initial (fixed) field dofs, packed byNODES as
    /// `[c * n_field + i]`.
    pub z0: Rc<Vec<f64>>,
    /// MFEM `adapt_lim_gf`: current remapped field dofs, updated by
    /// `TmopForm::process_new_state` through the adaptive-limiting evaluator.
    pub(crate) z_cur: RefCell<Rc<Vec<f64>>>,
    /// MFEM `adapt_lim_coeff` (one ConstantCoefficient per field). Mutated by
    /// the driver around metric-only energy evaluations and by
    /// `enable_normalization` (never).
    pub coeffs: RefCell<Vec<f64>>,
    /// MFEM `adapt_lim_delta_max` per field.
    pub delta_max: Vec<f64>,
    /// MFEM `lim_normal` (1.0 until `TmopForm::enable_normalization`).
    pub(crate) normal: Cell<f64>,
    /// Order of the limiting-field FE space (must equal the nodal space order:
    /// the C++ driver only uses `-alc` when `ind_fec_order == mesh_poly_deg`,
    /// which `AssembleElemVecAdaptLim`'s `el.ProjectGrad(el, ...)` requires).
    pub order: u8,
    /// Number of scalar limiting-field dofs.
    pub n_field: usize,
    /// Limiting-field scalar dof table per mesh element.
    pub element_dofs: Rc<Vec<Vec<usize>>>,
    /// Per-element limiting-space shape values at the metric quadrature
    /// points, `[e][q * nd + k]`.
    pub qp_shapes: Rc<Vec<Vec<f64>>>,
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
    /// MFEM `metric_normal`: 1.0 until normalization is enabled
    /// (`EnableNormalization`), then 1/E_metric(x0).
    pub metric_normal: f64,
    /// MFEM `EnableLimiting` state (None when limiting is disabled).
    pub limiting: Option<TmopLimiting>,
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
    /// Per-integrator discrete-target-spec remap evaluator
    /// (`DiscreteAdaptTC::adapt_eval`), attached with
    /// [`TmopForm::set_discrete_remapper`].
    remap_evals: Vec<Option<RefCell<TmopRemapEvaluator>>>,
    /// Per-integrator adaptive limiting term (`TMOP_Integrator`
    /// adapt_lim_* state), attached with
    /// [`TmopForm::enable_adaptive_limiting`].
    adaptive_limiting: Vec<Option<TmopAdaptiveLimiting>>,
    /// Per-integrator adaptive-limiting remap evaluator (MFEM
    /// `adapt_lim_eval`).
    al_evals: Vec<Option<RefCell<TmopRemapEvaluator>>>,
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
            remap_evals: Vec::new(),
            adaptive_limiting: Vec::new(),
            al_evals: Vec::new(),
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
        self.remap_evals.push(None);
        self.adaptive_limiting.push(None);
        self.al_evals.push(None);
    }

    pub fn integrators(&self) -> &[TmopIntegrator] {
        &self.integrators
    }

    pub fn integrators_mut(&mut self) -> &mut [TmopIntegrator] {
        &mut self.integrators
    }

    /// Attach the discrete-target-spec remap evaluator of integrator
    /// `integ_idx` (MFEM `DiscreteAdaptTC::SetAdaptivityEvaluator` +
    /// `FinalizeSerialDiscreteTargetSpec`, which initialize the evaluator with
    /// the mesh nodes and the packed initial tspec). The integrator's target
    /// must carry a `TmopDiscreteSpec`.
    pub fn set_discrete_remapper(&mut self, integ_idx: usize, evaluator: TmopRemapEvaluator) {
        let spec = self.integrators[integ_idx]
            .target
            .discrete
            .as_ref()
            .expect("set_discrete_remapper requires a discrete target spec");
        let packed0 = spec.packed0.clone();
        let mut ev = evaluator;
        ev.set_initial_field(&self.x0, &packed0);
        self.remap_evals[integ_idx] = Some(RefCell::new(ev));
    }

    /// MFEM `TMOP_Integrator::EnableAdaptiveLimiting`: enable the adaptive
    /// limiting term of integrator `integ_idx` with the initial limiting
    /// fields `z0` (one scalar dof vector per field, on the H1 space
    /// (`dm_al`, `order_al`)), constant coefficients `coeffs` and maximum
    /// field deltas `delta_max`; `evaluator` remaps the fields onto the moving
    /// mesh during the optimization.
    ///
    /// The C++ driver only enables adaptive limiting when the limiting-field
    /// space order equals the nodal space order (`ind_fec_order ==
    /// mesh_poly_deg`); the gradient/Hessian assembly mirrors
    /// `el.ProjectGrad(el, ...)` which requires it.
    #[allow(clippy::too_many_arguments)]
    pub fn enable_adaptive_limiting(
        &mut self,
        integ_idx: usize,
        dm_al: &DofManager,
        order_al: u8,
        z0: &[Vec<f64>],
        coeffs: Vec<f64>,
        delta_max: Vec<f64>,
        evaluator: TmopRemapEvaluator,
    ) {
        assert!(!z0.is_empty(), "Requires at least one field.");
        assert_eq!(z0.len(), coeffs.len(), "Requires one coefficient per field.");
        assert_eq!(z0.len(), delta_max.len(), "Requires one delta_max per field.");
        assert!(delta_max.iter().all(|d| *d > 0.0), "Requires delta_max > 0.0.");
        assert_eq!(
            order_al, self.order,
            "EnableAdaptiveLimiting requires the limiting-field space order to \
             equal the mesh nodal space order (see AssembleElemVecAdaptLim)."
        );
        let n_al = dm_al.n_dofs;
        let dim = self.dim;
        let mut evaluator = evaluator;
        // Packed initial fields (byNODES).
        let mut packed = vec![0.0_f64; z0.len() * n_al];
        for (c, zf) in z0.iter().enumerate() {
            assert_eq!(zf.len(), n_al);
            packed[c * n_al..(c + 1) * n_al].copy_from_slice(zf);
        }
        evaluator.set_initial_field(&self.x0, &packed);
        // Limiting-space shape values at the metric quadrature points (as in
        // the discrete-spec path of finalize_targets).
        let mut tables: Vec<Vec<f64>> = Vec::with_capacity(self.elems.len());
        for (e, el) in self.elems.iter().enumerate() {
            let al_re = ref_elem_for(self.topo.element_type(e as u32), order_al);
            let al_re: Box<dyn ReferenceElement> = if el_domain_is_unit(al_re.as_ref()) {
                al_re
            } else {
                Box::new(UnitDomainElem {
                    inner: al_re,
                    dim,
                })
            };
            let nd_ind = al_re.n_dofs();
            let nqp = el.quad_points.len();
            let mut tab = vec![0.0_f64; nqp * nd_ind];
            let mut sh = vec![0.0_f64; nd_ind];
            for q in 0..nqp {
                al_re.eval_basis(&el.quad_points[q], &mut sh);
                for (k, v) in sh.iter().enumerate() {
                    tab[q * nd_ind + k] = *v;
                }
            }
            tables.push(tab);
        }
        let element_dofs: Rc<Vec<Vec<usize>>> = Rc::new(
            (0..self.topo.n_elements())
                .map(|e| {
                    dm_al
                        .element_dofs(e as u32)
                        .iter()
                        .map(|&d| d as usize)
                        .collect()
                })
                .collect(),
        );
        self.adaptive_limiting[integ_idx] = Some(TmopAdaptiveLimiting {
            z0: Rc::new(packed.clone()),
            z_cur: RefCell::new(Rc::new(packed.clone())),
            coeffs: RefCell::new(coeffs),
            delta_max,
            normal: Cell::new(1.0),
            order: order_al,
            n_field: n_al,
            element_dofs,
            qp_shapes: Rc::new(tables),
        });
        self.al_evals[integ_idx] = Some(RefCell::new(evaluator));
    }

    /// Replace the adaptive limiting coefficients of integrator `integ_idx`
    /// (the driver zeroes/restores them around metric-only energy reports).
    pub fn set_adaptive_limiting_coeffs(&self, integ_idx: usize, vals: &[f64]) {
        let al = self.adaptive_limiting[integ_idx]
            .as_ref()
            .expect("no adaptive limiting on this integrator");
        *al.coeffs.borrow_mut() = vals.to_vec();
    }

    pub fn adaptive_limiting_coeffs(&self, integ_idx: usize) -> Vec<f64> {
        match self.adaptive_limiting[integ_idx].as_ref() {
            Some(al) => al.coeffs.borrow().clone(),
            None => Vec::new(),
        }
    }

    /// MFEM `TMOPNewtonSolver::ProcessNewState`: remap the discrete target
    /// specification and the adaptive limiting fields onto the mesh positions
    /// `x0 + dx` (the evaluators are called in the C++ order: tspec first,
    /// adaptive limiting second). A no-op when none are enabled.
    pub fn process_new_state(&self, dx: &[f64]) {
        let n = self.n_dofs();
        let has_remap = self.remap_evals.iter().any(Option::is_some)
            || self.al_evals.iter().any(Option::is_some);
        if !has_remap {
            return;
        }
        let mut x_loc = vec![0.0_f64; n];
        for i in 0..n {
            x_loc[i] = self.x0[i] + dx[i];
        }
        for ii in 0..self.integrators.len() {
            // DiscreteAdaptTC::UpdateTargetSpecification(x_loc, ...).
            if let Some(ev) = &self.remap_evals[ii] {
                if let Some(spec) = &self.integrators[ii].target.discrete {
                    let mut tspec = spec.packed_current();
                    ev.borrow_mut().compute_at_new_position(&x_loc, &mut tspec);
                    *spec.remapped.borrow_mut() = Some(Rc::new(tspec));
                }
            }
            // Adaptive limiting field remap (UpdateAfterMeshPositionChange).
            if let Some(al) = &self.adaptive_limiting[ii] {
                let ev = self.al_evals[ii].as_ref().unwrap();
                let mut z = al.z_cur.borrow().as_ref().clone();
                ev.borrow_mut().compute_at_new_position(&x_loc, &mut z);
                *al.z_cur.borrow_mut() = Rc::new(z);
            }
        }
    }

    /// MFEM `TargetConstructor::ComputeAvgVolume` for every integrator whose
    /// target needs it: total physical volume of the `x0` mesh / NE, using a
    /// Gauss-Legendre rule of order `2*order` (exact for the polynomial det).
    /// Also prepares the quadrature shape tables of discrete-adaptivity
    /// targets (`TmopDiscreteSpec::qp_shapes`).
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
            // Discrete adaptivity: precompute the indicator-space shape values
            // at the metric quadrature points of every element (MFEM evaluates
            // `src_fes->GetFE(e_id)->CalcShape(ip, shape)` on the fly; the
            // values are deterministic, so precomputing them is equivalent).
            if let Some(spec) = &mut integ.target.discrete {
                let mut tables: Vec<Vec<f64>> = Vec::with_capacity(ne);
                for (e, el) in self.elems.iter().enumerate() {
                    let ind_re = ref_elem_for(self.topo.element_type(e as u32), spec.order);
                    // Normalize an arbitrary reference domain to MFEM's [0,1]^d.
                    let ind_re: Box<dyn ReferenceElement> = if el_domain_is_unit(ind_re.as_ref()) {
                        ind_re
                    } else {
                        Box::new(UnitDomainElem {
                            inner: ind_re,
                            dim: self.dim,
                        })
                    };
                    let nd_ind = ind_re.n_dofs();
                    let nqp = el.quad_points.len();
                    let mut tab = vec![0.0f64; nqp * nd_ind];
                    let mut sh = vec![0.0f64; nd_ind];
                    for q in 0..nqp {
                        ind_re.eval_basis(&el.quad_points[q], &mut sh);
                        for (k, v) in sh.iter().enumerate() {
                            tab[q * nd_ind + k] = *v;
                        }
                    }
                    tables.push(tab);
                }
                spec.qp_shapes = Some(Rc::new(tables));
            }
        }
    }

    /// MFEM `TMOP_Integrator::EnableNormalization(x)` (serial LEGACY path):
    /// computes `metric_normal = 1/E_metric(x0)` and
    /// `lim_normal = 1/E_lim(x0)` with `ComputeNormalizationEnergies`, where
    /// `E_lim` is the integral of the unit weight over the targets. The
    /// surface fitting weight normalization (`surf_fit_normal`) becomes the
    /// limiting factor. Must be called after `set_x0`, the integrator setup
    /// (including limiting) and `finalize_targets`.
    pub fn enable_normalization(&mut self) {
        let dim = self.dim;
        let zeros = vec![0.0_f64; self.n_dofs()];
        // Pass 1 (immutable): ComputeNormalizationEnergies per integrator.
        let mut normals: Vec<(f64, f64)> = Vec::with_capacity(self.integrators.len());
        for integ in &self.integrators {
            let mut metric_energy = 0.0f64;
            let mut lim_energy = 0.0f64;
            let mut pos = Vec::new();
            let mut dsh = Vec::new();
            let mut jtr2: Vec<[f64; 4]> = Vec::new();
            let mut jtr3: Vec<[f64; 9]> = Vec::new();
            for (ei, el) in self.elems.iter().enumerate() {
                let nd = el.edofs.len();
                let nqp = el.quad_points.len();
                self.element_positions(el, &zeros, &mut pos);
                dsh.resize(nd * dim, 0.0);
                self.compute_element_targets(integ, ei, el, &mut jtr2, &mut jtr3, &mut dsh);
                for q in 0..nqp {
                    let weight = el.quad_weights[q]
                        * if dim == 2 { det_jtr2(&jtr2[q]) } else { det_jtr3(&jtr3[q]) };
                    el.re.eval_grad_basis(&el.quad_points[q], &mut dsh);
                    // Jpr = PMatI^T * DSh, Jpt = Jpr * Jrt (MFEM
                    // `MultAtB(PMatI, DSh, Jpr); Mult(Jpr, Jrt, Jpt)`).
                    match (&integ.metric, dim) {
                        (TmopMetric::D2(m), 2) => {
                            let mut jpr = [[0.0f64; 2]; 2];
                            for a in 0..2 {
                                for b in 0..2 {
                                    let mut s = 0.0;
                                    for i in 0..nd {
                                        s += pos[i + a * nd] * dsh[i * 2 + b];
                                    }
                                    jpr[a][b] = s;
                                }
                            }
                            let jrt = invert_2x2(&jtr2[q]);
                            let mut jpt = [[0.0f64; 2]; 2];
                            for a in 0..2 {
                                for b in 0..2 {
                                    let mut s = 0.0;
                                    for j in 0..2 {
                                        s += jpr[a][j] * jrt[j + b * 2];
                                    }
                                    jpt[a][b] = s;
                                }
                            }
                            metric_energy += weight * m.eval_w(&jpt);
                        }
                        (TmopMetric::D3(m), 3) => {
                            let mut jpr = [[0.0f64; 3]; 3];
                            for a in 0..3 {
                                for b in 0..3 {
                                    let mut s = 0.0;
                                    for i in 0..nd {
                                        s += pos[i + a * nd] * dsh[i * 3 + b];
                                    }
                                    jpr[a][b] = s;
                                }
                            }
                            let jrt = invert_3x3(&jtr3[q]);
                            let mut jpt = [[0.0f64; 3]; 3];
                            for a in 0..3 {
                                for b in 0..3 {
                                    let mut s = 0.0;
                                    for j in 0..3 {
                                        s += jpr[a][j] * jrt[j + b * 3];
                                    }
                                    jpt[a][b] = s;
                                }
                            }
                            metric_energy += weight * m.eval_w(&jpt);
                        }
                        _ => panic!("metric dimension mismatch"),
                    }
                    lim_energy += weight;
                }
            }
            normals.push((1.0 / metric_energy, 1.0 / lim_energy));
        }
        // Pass 2 (mutable): metric_normal, lim_normal, surf_fit_normal.
        for (i, (integ, (metric_normal, lim_normal))) in
            self.integrators.iter_mut().zip(normals.into_iter()).enumerate()
        {
            integ.metric_normal = metric_normal;
            if let Some(lim) = &mut integ.limiting {
                lim.normal = lim_normal;
            }
            if let Some(sf) = &integ.surf_fit {
                sf.normal.set(lim_normal);
            }
            // The adaptive limiting term is normalized with lim_normal too
            // (MFEM AssembleElemVecAdaptLim uses lim_normal).
            if let Some(al) = &self.adaptive_limiting[i] {
                al.normal.set(lim_normal);
            }
        }
    }

    /// Total physical volume of the `x0` mesh (MFEM `Mesh::GetElementVolume`
    /// summed over all elements): `Σ ∫ det(Jpr)` with a Gauss-Legendre rule of
    /// order `order` (MFEM `OrderJ()` for a Qk geometry; exact for det(Jpr)).
    pub fn mesh_volume(&self) -> f64 {
        let dim = self.dim;
        let mut volume = 0.0;
        let zeros = vec![0.0_f64; self.n_dofs()];
        let mut pos = Vec::new();
        let mut dsh = Vec::new();
        for el in &self.elems {
            let nd = el.edofs.len();
            self.element_positions(el, &zeros, &mut pos);
            dsh.resize(nd * dim, 0.0);
            let rule = el.re.quadrature(self.order);
            for (q, xi) in rule.points.iter().enumerate() {
                el.re.eval_grad_basis(xi, &mut dsh);
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
                volume += rule.weights[q] * det_small(&jpr, dim);
            }
        }
        volume
    }

    /// Per-element target Jacobians Jtr (column-major) at every quadrature
    /// point. `Wideal` = identity for quad/hex (MFEM
    /// `Geometries.GetGeomToPerfGeomJac`); tri/tet meshes are rejected in
    /// `TmopForm::new` via `ref_elem_for`.
    fn compute_element_targets(
        &self,
        integ: &TmopIntegrator,
        e: usize,
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
            TmopTargetType::IdealShapeGivenSizeDiscrete
            | TmopTargetType::GivenShapeAndSizeDiscrete => {
                let spec = integ.target.discrete.as_ref().expect("call finalize_targets");
                let tables = spec
                    .qp_shapes
                    .as_ref()
                    .expect("call finalize_targets (qp_shapes)");
                let vals = spec.current();
                Self::discrete_element_targets(
                    &vals,
                    spec.size.is_some(),
                    spec.aspect_ratio.is_some(),
                    spec.skew.is_some(),
                    spec.orientation.is_some(),
                    &spec.element_dofs[e],
                    spec.n_ind,
                    spec.min_size,
                    integ.target.target_type == TmopTargetType::GivenShapeAndSizeDiscrete,
                    dim,
                    &tables[e],
                    jtr_out,
                    jtr_out_3d,
                );
            }
        }
    }

    /// Discrete-adaptivity element targets (MFEM
    /// `DiscreteAdaptTC::ComputeElementTargets` for IDEAL_SHAPE_GIVEN_SIZE /
    /// GIVEN_SHAPE_AND_SIZE). `shapes` holds the indicator-space shape values
    /// at the quadrature points of element `e` (`spec.qp_shapes[e]`); `vals`
    /// holds the packed byNODES tspec dofs (the remapped state when a remap
    /// evaluator is attached, the initial fields otherwise) with components in
    /// the order size, aspect ratio, skew, orientation, matching the C++
    /// `SetTspecAtIndex` append order. The `has_*` flags mirror the C++
    /// `sizeidx`/`aspectratioidx`/`skewidx`/`orientationidx >= 0` checks.
    #[allow(clippy::too_many_arguments)]
    fn discrete_element_targets(
        vals: &[f64],
        has_size: bool,
        has_aspr: bool,
        has_skew: bool,
        has_ori: bool,
        dofs: &[usize],
        n_ind: usize,
        lim_min_size: f64,
        given_shape_and_size: bool,
        dim: usize,
        shapes: &[f64],
        jtr_out: &mut Vec<[f64; 4]>,
        jtr_out_3d: &mut Vec<[f64; 9]>,
    ) {
        // byNODES multi-component layout: index = c*n_ind + dof.
        let comp = |c: usize, k: usize| -> f64 { vals[c * n_ind + dofs[k]] };
        let min_of = |c: usize| -> f64 {
            let mut m = f64::INFINITY;
            for k in 0..dofs.len() {
                m = m.min(comp(c, k));
            }
            m
        };
        let nqp = shapes.len() / dofs.len();
        let mut jtr = [[0.0f64; 3]; 3];
        for q in 0..nqp {
            // Component cursor (MFEM sizeidx / aspectratioidx / skewidx /
            // orientationidx): the fields are appended in the order they are
            // set; it restarts at every quadrature point.
            let mut idx = 0_usize;
            let sh = &shapes[q * dofs.len()..(q + 1) * dofs.len()];
            // shape * par_vals (MFEM Vector dot product, ascending sum).
            let dot = |c: usize| -> f64 {
                let mut s = 0.0;
                for (k, &sv) in sh.iter().enumerate() {
                    s += sv * comp(c, k);
                }
                s
            };
            // Jtr = Wideal = I (quad/hex).
            for r in jtr.iter_mut() {
                *r = [0.0; 3];
            }
            for d in 0..dim {
                jtr[d][d] = 1.0;
            }
            // Set size.
            if has_size {
                let mut min_size = min_of(idx);
                if lim_min_size > 0.0 {
                    min_size = lim_min_size;
                }
                assert!(min_size > 0.0, "Non-positive size propagated in the target definition.");
                let size_q = dot(idx).max(min_size);
                let sc = size_q.powf(1.0 / dim as f64);
                for r in jtr.iter_mut().take(dim) {
                    for v in r.iter_mut().take(dim) {
                        *v *= sc;
                    }
                }
                idx += 1;
            }
            if given_shape_and_size {
                // aspect ratio
                if has_aspr {
                    let d_rho = if dim == 2 {
                        assert!(
                            min_of(idx) > 0.0,
                            "Non-positive aspect-ratio propagated in the target definition."
                        );
                        let aspr = dot(idx);
                        [
                            [1.0 / aspr.powf(0.5), 0.0, 0.0],
                            [0.0, aspr.powf(0.5), 0.0],
                            [0.0, 0.0, 0.0],
                        ]
                    } else {
                        [
                            [dot(idx).powf(2.0 / 3.0), 0.0, 0.0],
                            [0.0, dot(idx + 1).powf(2.0 / 3.0), 0.0],
                            [0.0, 0.0, dot(idx + 2).powf(2.0 / 3.0)],
                        ]
                    };
                    jtr = mat3_mul_dim(&d_rho, &jtr, dim);
                    idx += if dim == 2 { 1 } else { 3 };
                }
                // skew
                if has_skew {
                    let q_phi = if dim == 2 {
                        let skew = dot(idx);
                        [
                            [1.0, skew.cos(), 0.0],
                            [0.0, skew.sin(), 0.0],
                            [0.0, 0.0, 0.0],
                        ]
                    } else {
                        let phi12 = dot(idx);
                        let phi13 = dot(idx + 1);
                        let chi = dot(idx + 2);
                        [
                            [1.0, phi12.cos(), phi13.cos()],
                            [0.0, phi12.sin(), phi13.sin() * chi.cos()],
                            [0.0, 0.0, phi13.sin() * chi.sin()],
                        ]
                    };
                    jtr = mat3_mul_dim(&q_phi, &jtr, dim);
                    idx += if dim == 2 { 1 } else { 3 };
                }
                // orientation
                if has_ori {
                    let r_theta = if dim == 2 {
                        let theta = dot(idx);
                        let (ct, st) = (theta.cos(), theta.sin());
                        [[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 0.0]]
                    } else {
                        let theta = dot(idx);
                        let psi = dot(idx + 1);
                        let beta = dot(idx + 2);
                        let (ct, st) = (theta.cos(), theta.sin());
                        let (cp, sp) = (psi.cos(), psi.sin());
                        let (cb, sb) = (beta.cos(), beta.sin());
                        // Ported verbatim from MFEM (the (0,0)/(1,0)/(2,0)
                        // entries are assigned twice; the final values are the
                        // last ones).
                        [
                            [-st * sb - ct * cp * cb, -st * cb + ct * cp * sb, 0.0],
                            [ct * sb - st * cp * cb, ct * cb + st * cp * sb, 0.0],
                            [sp * cb, -sp * sb, 0.0],
                        ]
                    };
                    jtr = mat3_mul_dim(&r_theta, &jtr, dim);
                }
            }
            let _ = idx; // the cursor value is only read across branches above
            if dim == 2 {
                // Row-major [r*2 + c], the convention of `det_jtr2`/`invert_2x2`.
                jtr_out.push([jtr[0][0], jtr[0][1], jtr[1][0], jtr[1][1]]);
            } else {
                jtr_out_3d.push([
                    jtr[0][0], jtr[0][1], jtr[0][2], jtr[1][0], jtr[1][1], jtr[1][2], jtr[2][0],
                    jtr[2][1], jtr[2][2],
                ]);
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
        let mut shape: Vec<f64> = Vec::new();
        let mut pos0: Vec<f64> = Vec::new();
        let mut pt = [0.0f64; 3];
        let mut pt0 = [0.0f64; 3];
        let mut d_val = 0.0f64;
        for (ei, el) in self.elems.iter().enumerate() {
            let nd = el.edofs.len();
            let nqp = el.quad_points.len();
            self.element_positions(el, dx, &mut pos);
            let mut ds = vec![0.0; nd * dim];
            dsh.resize(nd * dim, 0.0);
            let mut energy = 0.0;
            for (ii, integ) in self.integrators.iter().enumerate() {
                // Adaptive limiting state (None when disabled).
                let al = self.adaptive_limiting[ii].as_ref();
                self.compute_element_targets(integ, ei, el, &mut jtr2, &mut jtr3, &mut dsh);
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
                            // MFEM: val = metric_normal*EvalW; val *= coeff;
                            // val += lim_normal*lim_func*lim_coeff;
                            // val += adapt_lim_coeff*lim_normal*diff^2;
                            // energy += weight*val. The v1 path (normal == 1,
                            // no limiting, no adaptive limiting) keeps its
                            // original accumulation.
                            if integ.metric_normal == 1.0
                                && integ.limiting.is_none()
                                && al.is_none()
                            {
                                energy += weight * integ.coeff * m.eval_w(&jpt);
                            } else {
                                let mut val = integ.metric_normal * m.eval_w(&jpt);
                                val *= integ.coeff;
                                if let Some(lim) = &integ.limiting {
                                    limiting_point_data(
                                        el, &pos, &lim.nodes0, self.n_scalar, dim, lim, q,
                                        &mut shape, &mut pos0, &mut pt, &mut pt0, &mut d_val,
                                    );
                                    val += lim.normal
                                        * lim.lim_func.eval(&pt[..dim], &pt0[..dim], d_val)
                                        * lim.coeff;
                                }
                                if let Some(al) = al {
                                    val += al_energy_at_point(al, ei, q);
                                }
                                energy += weight * val;
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
                            if integ.metric_normal == 1.0
                                && integ.limiting.is_none()
                                && al.is_none()
                            {
                                energy += weight * integ.coeff * m.eval_w(&jpt);
                            } else {
                                let mut val = integ.metric_normal * m.eval_w(&jpt);
                                val *= integ.coeff;
                                if let Some(lim) = &integ.limiting {
                                    limiting_point_data(
                                        el, &pos, &lim.nodes0, self.n_scalar, dim, lim, q,
                                        &mut shape, &mut pos0, &mut pt, &mut pt0, &mut d_val,
                                    );
                                    val += lim.normal
                                        * lim.lim_func.eval(&pt[..dim], &pt0[..dim], d_val)
                                        * lim.coeff;
                                }
                                if let Some(al) = al {
                                    val += al_energy_at_point(al, ei, q);
                                }
                                energy += weight * val;
                            }
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
                        // MFEM: w = surf_fit_coeff * surf_fit_normal * 1.0/count.
                        let w = (sf.coeff.get() * sf.normal.get()) / sf.dof_count[dof];
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
        let mut shape: Vec<f64> = Vec::new();
        let mut pos0: Vec<f64> = Vec::new();
        let mut pt = [0.0f64; 3];
        let mut pt0 = [0.0f64; 3];
        let mut d_val = 0.0f64;
        let mut d1v = [0.0f64; 3];
        for (ei, el) in self.elems.iter().enumerate() {
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
            for (ii, integ) in self.integrators.iter().enumerate() {
                self.compute_element_targets(integ, ei, el, &mut jtr2, &mut jtr3, &mut dsh);
                for q in 0..nqp {
                    // MFEM: weights(q) = ip.weight*det(Jtr);
                    // weight_m = weights(q)*metric_normal (*= metric_coeff).
                    let weight = el.quad_weights[q]
                        * if dim == 2 { det_jtr2(&jtr2[q]) } else { det_jtr3(&jtr3[q]) };
                    let weight_m = weight * integ.metric_normal * integ.coeff;
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
                    // MFEM `if (lim_coeff)` in AssembleElementVectorExact:
                    // lim_func->Eval_d1(p, p0, d_vals(q), grad);
                    // grad *= weights(q)*lim_normal*lim_coeff;
                    // AddMultVWt(shape, grad, PMatO).
                    if let Some(lim) = &integ.limiting {
                        limiting_point_data(
                            el, &pos, &lim.nodes0, self.n_scalar, dim, lim, q, &mut shape,
                            &mut pos0, &mut pt, &mut pt0, &mut d_val,
                        );
                        lim.lim_func.eval_d1(&pt[..dim], &pt0[..dim], d_val, &mut d1v);
                        let w_l = (weight * lim.normal) * lim.coeff;
                        for c in 0..dim {
                            d1v[c] *= w_l;
                        }
                        for (k, &sv) in shape.iter().enumerate() {
                            for c in 0..dim {
                                elvec[k + c * nd] += sv * d1v[c];
                            }
                        }
                    }
                }
                // MFEM `AssembleElemVecAdaptLim`: for every limiting field c,
                // project its gradient on the current element geometry and add
                // `shape(i) * gq(d)` with `gq = ∇z_c(q) * 2*(z-z0)/δ² *
                // weights(q)*lim_normal*coeff_c` to elvect(i, d).
                if let Some(al) = &self.adaptive_limiting[ii] {
                    let cur = al.z_cur.borrow();
                    let coeffs = al.coeffs.borrow();
                    let nf = al.n_field;
                    let mut gphys = vec![0.0_f64; nd * dim * nd];
                    let mut grad_e = vec![0.0_f64; nd * dim];
                    let mut gq = [0.0_f64; 3];
                    let mut zq_al = vec![0.0_f64; coeffs.len()];
                    let mut z0q_al = vec![0.0_f64; coeffs.len()];
                    al_project_grad_matrix(el.re.as_ref(), &el.edofs, &pos, nd, dim, &mut gphys);
                    for (c, coeff_c) in coeffs.iter().enumerate() {
                        al_grad_e_from(
                            &gphys,
                            &cur[c * nf..(c + 1) * nf],
                            &el.edofs,
                            nd,
                            dim,
                            &mut grad_e,
                        );
                        for q in 0..nqp {
                            shape.resize(nd, 0.0);
                            el.re.eval_basis(&el.quad_points[q], &mut shape);
                            for d in 0..dim {
                                let mut s = 0.0;
                                for i in 0..nd {
                                    s += shape[i] * grad_e[i + d * nd];
                                }
                                gq[d] = s;
                            }
                            al_values_at_point(al, &cur, ei, q, &mut zq_al);
                            al_values_at_point(al, &al.z0, ei, q, &mut z0q_al);
                            let delta2 = al.delta_max[c] * al.delta_max[c];
                            let weight = el.quad_weights[q]
                                * if dim == 2 {
                                    det_jtr2(&jtr2[q])
                                } else {
                                    det_jtr3(&jtr3[q])
                                };
                            for d in 0..dim {
                                gq[d] *= 2.0 * (zq_al[c] - z0q_al[c]) / delta2;
                            }
                            for d in 0..dim {
                                gq[d] *= weight * al.normal.get() * coeff_c;
                            }
                            // AddMultVWt(shape, gq, PMatO).
                            for (k, &sv) in shape.iter().enumerate() {
                                for d in 0..dim {
                                    elvec[k + d * nd] += sv * gq[d];
                                }
                            }
                        }
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
                        let w = (sf.normal.get() * sf.coeff.get()) / sf.dof_count[dof];
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
        let mut shape: Vec<f64> = Vec::new();
        let mut pos0: Vec<f64> = Vec::new();
        let mut pt = [0.0f64; 3];
        let mut pt0 = [0.0f64; 3];
        let mut d_val = 0.0f64;
        let mut hess_lm = [0.0f64; 9];
        for (ei, el) in self.elems.iter().enumerate() {
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
            for (ii, integ) in self.integrators.iter().enumerate() {
                self.compute_element_targets(integ, ei, el, &mut jtr2, &mut jtr3, &mut dsh);
                for q in 0..nqp {
                    // MFEM: weights(q) = ip.weight*det(Jtr);
                    // weight_m = weights(q)*metric_normal (*= metric_coeff).
                    let weight = el.quad_weights[q]
                        * if dim == 2 { det_jtr2(&jtr2[q]) } else { det_jtr3(&jtr3[q]) };
                    let weight_m = weight * integ.metric_normal * integ.coeff;
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
                    // MFEM `if (lim_coeff)` in AssembleElementGradExact:
                    // weight_m = weights(q)*lim_normal*lim_coeff;
                    // lim_func->Eval_d2(p, p0, d_vals(q), hess);
                    // elmat(d1*dof+i, d2*dof+j) += weight_m*shape_i*shape_j*hess(d1,d2).
                    if let Some(lim) = &integ.limiting {
                        limiting_point_data(
                            el, &pos, &lim.nodes0, self.n_scalar, dim, lim, q, &mut shape,
                            &mut pos0, &mut pt, &mut pt0, &mut d_val,
                        );
                        lim.lim_func.eval_d2(&pt[..dim], &pt0[..dim], d_val, &mut hess_lm);
                        let w_l = (weight * lim.normal) * lim.coeff;
                        for i in 0..nd {
                            let w_shape_i = w_l * shape[i];
                            for j in 0..nd {
                                let w = w_shape_i * shape[j];
                                for d1 in 0..dim {
                                    for d2 in 0..dim {
                                        let row = d1 * nd + i;
                                        let col = d2 * nd + j;
                                        elmat[row + ah * col] += w * hess_lm[d1 * dim + d2];
                                    }
                                }
                            }
                        }
                    }
                }
                // MFEM `AssembleElemGradAdaptLim`: per limiting field c,
                // entry(i,j) = factor * (grad outer product +
                // (z(q)-z0(q)) * hess_q(d1,d2) * shape outer product) with
                // factor = weights(q)*lim_normal*coeff*2/δ², added for i, j <= i
                // (symmetric), i = idof + idim*nd.
                if let Some(al) = &self.adaptive_limiting[ii] {
                    let cur = al.z_cur.borrow();
                    let coeffs = al.coeffs.borrow();
                    let nf = al.n_field;
                    let mut gphys = vec![0.0_f64; nd * dim * nd];
                    let mut grad_e = vec![0.0_f64; nd * dim];
                    let mut hess_e = vec![0.0_f64; nd * dim * dim];
                    let mut gq = [0.0_f64; 3];
                    let mut gg = [0.0_f64; 9]; // gg[d + m*dim]
                    let mut zq_al = vec![0.0_f64; coeffs.len()];
                    let mut z0q_al = vec![0.0_f64; coeffs.len()];
                    al_project_grad_matrix(el.re.as_ref(), &el.edofs, &pos, nd, dim, &mut gphys);
                    for (c, coeff_c) in coeffs.iter().enumerate() {
                        al_grad_e_from(
                            &gphys,
                            &cur[c * nf..(c + 1) * nf],
                            &el.edofs,
                            nd,
                            dim,
                            &mut grad_e,
                        );
                        // hess_e(k + d*nd + m*(nd*dim)) =
                        // Σ_j gphys(k + d*nd, j) * grad_e(j + m*nd).
                        for v in hess_e.iter_mut() {
                            *v = 0.0;
                        }
                        for k in 0..nd {
                            for d in 0..dim {
                                for m in 0..dim {
                                    let mut s = 0.0;
                                    for j in 0..nd {
                                        s += gphys[k + d * nd + j * (nd * dim)]
                                            * grad_e[j + m * nd];
                                    }
                                    hess_e[k + d * nd + m * (nd * dim)] = s;
                                }
                            }
                        }
                        let delta2 = al.delta_max[c] * al.delta_max[c];
                        for q in 0..nqp {
                            shape.resize(nd, 0.0);
                            el.re.eval_basis(&el.quad_points[q], &mut shape);
                            for d in 0..dim {
                                let mut s = 0.0;
                                for i in 0..nd {
                                    s += shape[i] * grad_e[i + d * nd];
                                }
                                gq[d] = s;
                            }
                            // gg(c = d + m*dim) = Σ_i shape(i) *
                            // hess_e(i + c*nd)  (the (dof, dim*dim) reshape).
                            for c2 in 0..dim * dim {
                                let mut s = 0.0;
                                for i in 0..nd {
                                    s += shape[i] * hess_e[i + c2 * nd];
                                }
                                gg[c2] = s;
                            }
                            al_values_at_point(al, &cur, ei, q, &mut zq_al);
                            al_values_at_point(al, &al.z0, ei, q, &mut z0q_al);
                            let weight = el.quad_weights[q]
                                * if dim == 2 {
                                    det_jtr2(&jtr2[q])
                                } else {
                                    det_jtr3(&jtr3[q])
                                };
                            let factor = weight * al.normal.get() * coeff_c * 2.0 / delta2;
                            for i in 0..nd * dim {
                                let idof = i % nd;
                                let idim = i / nd;
                                for j in 0..=i {
                                    let jdof = j % nd;
                                    let jdim = j / nd;
                                    let entry = factor
                                        * (gq[idim]
                                            * shape[idof]
                                            * gq[jdim]
                                            * shape[jdof]
                                            + (zq_al[c] - z0q_al[c])
                                                * gg[idim + jdim * dim]
                                                * shape[idof]
                                                * shape[jdof]);
                                    elmat[i + ah * j] += entry;
                                    if i != j {
                                        elmat[j + ah * i] += entry;
                                    }
                                }
                            }
                        }
                    }
                }
                // MFEM `AssembleElemGradSurfFit`: for every marked dof s,
                // mat(s+c1*nd, s+c1*nd) += w with w = normal*coeff/count (the
                // quadratic limiter's Eval_d2 = I with dist = 1). NOTE: like
                // MFEM, the outer product of the fitting gradient is
                // deliberately omitted here (its `surf_fit_pos` branch zeroes
                // surf_fit_grad_e and keeps only the limiter Hessian), so the
                // assembled Hessian of the fitting term is the constant w*I.
                if let Some(sf) = &integ.surf_fit {
                    for (k, &dof) in el.edofs.iter().enumerate() {
                        if !sf.marker[dof] {
                            continue;
                        }
                        let w = sf.normal.get() * sf.coeff.get();
                        for c1 in 0..dim {
                            let idx = c1 * nd + k;
                            // entry = w*(0 + 1*hess(c1,c1)) with hess = I, then
                            // entry *= 1/count (the exact C++ operation order).
                            let mut entry = w * 1.0;
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

/// H1 Gauss-Lobatto reference element on MFEM's unit reference domain
/// `[0,1]^dim` (the same normalization as the element data of `TmopForm`).
/// Exposed for drivers that evaluate discrete fields (nodal projections,
/// diffusion smoothing, derivatives) on the mesh outside of the form.
pub fn tmop_ref_elem(et: ElementType, order: u8, dim: usize) -> Box<dyn ReferenceElement> {
    let raw = ref_elem_for(et, order);
    if el_domain_is_unit(raw.as_ref()) {
        raw
    } else {
        Box::new(UnitDomainElem {
            inner: raw,
            dim,
        })
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

/// C = A * B on the top-left dim x dim block (MFEM `Mult(A, B, C)`).
fn mat3_mul_dim(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3], dim: usize) -> [[f64; 3]; 3] {
    let mut c = [[0.0f64; 3]; 3];
    for i in 0..dim {
        for j in 0..dim {
            let mut s = 0.0;
            for k in 0..dim {
                s += a[i][k] * b[k][j];
            }
            c[i][j] = s;
        }
    }
    c
}

/// Quadrature-point data of the limiting term (MFEM: `el.CalcShape(ip, shape)`,
/// `PMatI.MultTranspose(shape, p)`, `pos0.MultTranspose(shape, p0)` and
/// `lim_dist->GetValues(el_id, ir, d_vals)` — the driver's `dist` grid function
/// is constant in space, so the latter reduces to the Qk interpolation of that
/// constant).
#[allow(clippy::too_many_arguments)]
fn limiting_point_data(
    el: &TmopElemData,
    pos: &[f64],
    nodes0: &[f64],
    n_scalar: usize,
    dim: usize,
    lim: &TmopLimiting,
    q: usize,
    shape: &mut Vec<f64>,
    pos0: &mut Vec<f64>,
    p: &mut [f64; 3],
    p0: &mut [f64; 3],
    d_val: &mut f64,
) {
    let nd = el.edofs.len();
    shape.resize(nd, 0.0);
    el.re.eval_basis(&el.quad_points[q], shape);
    pos0.resize(nd * dim, 0.0);
    for (k, &dof) in el.edofs.iter().enumerate() {
        for c in 0..dim {
            pos0[k + c * nd] = nodes0[c * n_scalar + dof];
        }
    }
    *p = [0.0; 3];
    *p0 = [0.0; 3];
    for (k, &sv) in shape.iter().enumerate() {
        for c in 0..dim {
            p[c] += sv * pos[k + c * nd];
            p0[c] += sv * pos0[k + c * nd];
        }
    }
    *d_val = 0.0;
    for &sv in shape.iter() {
        *d_val += sv * lim.dist;
    }
}

/// Adaptive limiting energy term at quadrature point `q` of element `ei`
/// (MFEM `GetElementEnergy`): `Σ_c coeff_c * lim_normal * diff_c^2` with
/// `diff_c = (z_c(q) - z_{0,c}(q)) / delta_max_c`.
fn al_energy_at_point(al: &TmopAdaptiveLimiting, ei: usize, q: usize) -> f64 {
    let nd_al = al.element_dofs[ei].len();
    let sh = &al.qp_shapes[ei][q * nd_al..(q + 1) * nd_al];
    let dofs_al = &al.element_dofs[ei];
    let cur = al.z_cur.borrow();
    let coeffs = al.coeffs.borrow();
    let mut val = 0.0;
    for (c, coeff) in coeffs.iter().enumerate() {
        let mut zq = 0.0;
        let mut z0q = 0.0;
        for (k, &sv) in sh.iter().enumerate() {
            zq += sv * cur[c * al.n_field + dofs_al[k]];
            z0q += sv * al.z0[c * al.n_field + dofs_al[k]];
        }
        let diff = (zq - z0q) / al.delta_max[c];
        val += coeff * al.normal.get() * diff * diff;
    }
    val
}

/// MFEM `FiniteElement::ProjectGrad` (NodalFiniteElement) for the adaptive
/// limiting term: the raw projection matrix of the current element geometry
/// `pos`, `gphys[k + d*nd + j*(nd*dim)] = dφ_j/∂x_d` at the nodal point `k`.
fn al_project_grad_matrix(
    re: &dyn ReferenceElement,
    dofs: &[usize],
    pos: &[f64],
    nd: usize,
    dim: usize,
    gphys: &mut [f64],
) {
    let coords = re.dof_coords();
    let mut dsh = vec![0.0_f64; nd * dim];
    let mut jac = [[0.0_f64; 3]; 3];
    for v in gphys.iter_mut() {
        *v = 0.0;
    }
    for (k, xi) in coords.iter().enumerate().take(nd) {
        re.eval_grad_basis(xi, &mut dsh);
        // Jpr at the nodal point (current trial geometry).
        for a in 0..dim {
            for b in 0..dim {
                let mut s = 0.0;
                for (i, _) in dofs.iter().enumerate() {
                    s += pos[i + a * nd] * dsh[i * dim + b];
                }
                jac[a][b] = s;
            }
        }
        let jac_inv = jac_inverse(&jac, dim);
        for d in 0..dim {
            for j in 0..nd {
                let mut s = 0.0;
                for m in 0..dim {
                    s += dsh[j * dim + m] * jac_inv[m][d];
                }
                gphys[k + d * nd + j * (nd * dim)] = s;
            }
        }
    }
}

/// `adapt_lim_gf_grad_e = grad_phys * z_e` (projected gradient dofs of the
/// field `z`): `grad_e[k + d*nd] = ∂z/∂x_d at nodal point k`.
fn al_grad_e_from(gphys: &[f64], z: &[f64], dofs: &[usize], nd: usize, dim: usize, grad_e: &mut [f64]) {
    for v in grad_e.iter_mut() {
        *v = 0.0;
    }
    for k in 0..nd {
        for d in 0..dim {
            let mut s = 0.0;
            for (j, &dof) in dofs.iter().enumerate() {
                s += gphys[k + d * nd + j * (nd * dim)] * z[dof];
            }
            grad_e[k + d * nd] = s;
        }
    }
}

/// Interpolate the adaptive limiting field values `z` (current or initial) at
/// quadrature point `q` of element `ei`, for all components (`nal` values).
fn al_values_at_point(al: &TmopAdaptiveLimiting, z: &[f64], ei: usize, q: usize, out: &mut [f64]) {
    let nd_al = al.element_dofs[ei].len();
    let sh = &al.qp_shapes[ei][q * nd_al..(q + 1) * nd_al];
    let dofs_al = &al.element_dofs[ei];
    for c in 0..al.coeffs.borrow().len() {
        let mut s = 0.0;
        for (k, &sv) in sh.iter().enumerate() {
            s += sv * z[c * al.n_field + dofs_al[k]];
        }
        out[c] = s;
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
    // NewtonSolver::Mult starts with ProcessNewState(x = 0): for remap
    // evaluators this consumes a zero-velocity remap cycle (a value no-op).
    form.process_new_state(&x);
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
        form.process_new_state(&x);
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
        // ProcessNewState(d_out): remap the discrete target specification and
        // the adaptive limiting fields onto the trial mesh before the
        // energy/residual checks.
        form.process_new_state(&d_out);
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
    // false (the flag is only set at the end of ComputeScalingFactor); it
    // consumes the initial remap cycle of the field evaluators.

    form.process_new_state(&x);
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

        // ProcessNewState(x): updates the adaptive fitting weight and remaps
        // the discrete target spec / adaptive limiting fields.
        form.process_new_state(&x);
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

        // ProcessNewState(d_out): a no-op for the fitting weight (the update
        // flag is always false inside the line search) but it remaps the
        // discrete target spec / adaptive limiting fields when enabled.
        form.process_new_state(&d_out);

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
            metric_normal: 1.0,
            limiting: None,
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
            metric_normal: 1.0,
            limiting: None,
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
            metric_normal: 1.0,
            limiting: None,
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
                metric_normal: 1.0,
                limiting: None,
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
                metric_normal: 1.0,
                limiting: None,
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

    // ─── v2: limiting / normalization / discrete adaptivity ─────────────────

    /// A 2x2 quad mesh with a perturbed nodal configuration, metric 2 and the
    /// quadratic limiter bound to x0 (the mesh-optimizer driver setup): the
    /// energy/gradient/Hessian including the limiting term must satisfy the
    /// exact derivative relations.
    #[test]
    fn limiting_energy_gradient_hessian_fd() {
        let mesh: Mesh<2> = Mesh::make_cartesian_2d(2, 2, 1.0, 1.0);
        let order = 2u8;
        let dm = DofManager::new(&mesh, order);
        let topo: &dyn MeshTopology = &mesh;
        let mut form = TmopForm::new(topo, &dm, order, TmopQuadType::GaussLegendre, 7);
        let mut x0 = linear_mesh_positions(topo, &dm, order, 2);
        for (i, v) in x0.iter_mut().enumerate() {
            *v += 0.01 * (((i % 5) as f64) - 2.0);
        }
        form.set_x0(x0.clone());
        form.push_integrator(TmopIntegrator {
            metric: metric_from_id_2d(2, &SharedMinDet::new(0.0)).unwrap(),
            target: TmopTarget::new(TmopTargetType::IdealShapeUnitSize),
            coeff: 1.0,
            surf_fit: None,
            metric_normal: 1.0,
            limiting: Some(TmopLimiting {
                nodes0: Rc::new(x0),
                coeff: 0.4,
                dist: 1.0,
                lim_func: TmopLimiterFunction::Quadratic,
                normal: 1.0,
            }),
        });
        form.finalize_targets();

        let n = form.n_dofs();
        let mut dx = vec![0.0_f64; n];
        for (i, v) in dx.iter_mut().enumerate() {
            *v = 0.01 * (((i * 7) % 11) as f64 - 5.0);
        }
        let mut g = vec![0.0_f64; n];
        form.gradient(&dx, &mut g);
        let h = 1e-6;
        for k in (0..n).step_by(5) {
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
        // Hessian action: on the limiting-only functional (metric coeff 0) the
        // energy is an exact quadratic in dx, so its Hessian is constant and a
        // coarse central difference of the gradient is exact up to roundoff.
        // (The metric part of the Hessian is covered by
        // `surf_fit_energy_gradient_hessian_fd`.)
        let mut form_lim = TmopForm::new(topo, &dm, order, TmopQuadType::GaussLegendre, 7);
        form_lim.set_x0(form.x0().to_vec());
        form_lim.push_integrator(TmopIntegrator {
            metric: metric_from_id_2d(2, &SharedMinDet::new(0.0)).unwrap(),
            target: TmopTarget::new(TmopTargetType::IdealShapeUnitSize),
            coeff: 0.0,
            surf_fit: None,
            metric_normal: 1.0,
            limiting: Some(TmopLimiting {
                nodes0: Rc::new(form.x0().to_vec()),
                coeff: 0.4,
                dist: 1.0,
                lim_func: TmopLimiterFunction::Quadratic,
                normal: 1.0,
            }),
        });
        form_lim.finalize_targets();
        let hess = form_lim.hessian(&dx);
        let v: Vec<f64> = (0..n).map(|i| (((i * 13) % 7) as f64 - 3.0) * 0.05).collect();
        let mut hv = vec![0.0_f64; n];
        hess.spmv(&v, &mut hv);
        let eps = 1e-2;
        let mut xp = dx.clone();
        let mut xm = dx.clone();
        for i in 0..n {
            xp[i] += eps * v[i];
            xm[i] -= eps * v[i];
        }
        let mut gp = vec![0.0_f64; n];
        let mut gm = vec![0.0_f64; n];
        form_lim.gradient(&xp, &mut gp);
        form_lim.gradient(&xm, &mut gm);
        for k in 0..n {
            let fd = (gp[k] - gm[k]) / (2.0 * eps);
            assert!(
                (fd - hv[k]).abs() < 1e-9,
                "hess action mismatch at {k}: analytic {} fd {}",
                hv[k],
                fd
            );
        }
    }

    /// Same derivative checks for the exponential limiter (energy/gradient).
    #[test]
    fn limiting_exponential_limiter_fd() {
        let mesh: Mesh<2> = Mesh::make_cartesian_2d(2, 2, 1.0, 1.0);
        let order = 2u8;
        let dm = DofManager::new(&mesh, order);
        let topo: &dyn MeshTopology = &mesh;
        let mut form = TmopForm::new(topo, &dm, order, TmopQuadType::GaussLegendre, 7);
        let mut x0 = linear_mesh_positions(topo, &dm, order, 2);
        for (i, v) in x0.iter_mut().enumerate() {
            *v += 0.008 * (((i % 4) as f64) - 1.5);
        }
        form.set_x0(x0.clone());
        form.push_integrator(TmopIntegrator {
            metric: metric_from_id_2d(2, &SharedMinDet::new(0.0)).unwrap(),
            target: TmopTarget::new(TmopTargetType::IdealShapeUnitSize),
            coeff: 1.0,
            surf_fit: None,
            metric_normal: 1.0,
            limiting: Some(TmopLimiting {
                nodes0: Rc::new(x0),
                coeff: 0.3,
                dist: 0.5,
                lim_func: TmopLimiterFunction::Exponential,
                normal: 1.0,
            }),
        });
        form.finalize_targets();

        let n = form.n_dofs();
        let mut dx = vec![0.0_f64; n];
        for (i, v) in dx.iter_mut().enumerate() {
            *v = 0.005 * (((i * 7) % 11) as f64 - 5.0);
        }
        let mut g = vec![0.0_f64; n];
        form.gradient(&dx, &mut g);
        let h = 1e-7;
        for k in (0..n).step_by(5) {
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

    /// With metric coefficient 0 the functional reduces to the limiting term
    /// alone: `sum_q weight_q * 0.5*|x(q)-x0(q)|^2/dist^2 * lim_coeff`, which
    /// must be maximized at dx = 0 (limiter pulls the mesh back to x0).
    #[test]
    fn limiting_pulls_toward_reference_positions() {
        let mesh: Mesh<2> = Mesh::make_cartesian_2d(2, 2, 1.0, 1.0);
        let order = 2u8;
        let dm = DofManager::new(&mesh, order);
        let topo: &dyn MeshTopology = &mesh;
        let mut form = TmopForm::new(topo, &dm, order, TmopQuadType::GaussLegendre, 7);
        let mut x0 = linear_mesh_positions(topo, &dm, order, 2);
        for (i, v) in x0.iter_mut().enumerate() {
            *v += 0.02 * (((i % 5) as f64) - 2.0);
        }
        form.set_x0(x0.clone());
        form.push_integrator(TmopIntegrator {
            metric: metric_from_id_2d(2, &SharedMinDet::new(0.0)).unwrap(),
            target: TmopTarget::new(TmopTargetType::IdealShapeUnitSize),
            coeff: 0.0,
            surf_fit: None,
            metric_normal: 1.0,
            limiting: Some(TmopLimiting {
                nodes0: Rc::new(x0),
                coeff: 0.4,
                dist: 1.0,
                lim_func: TmopLimiterFunction::Quadratic,
                normal: 1.0,
            }),
        });
        form.finalize_targets();
        let zeros = vec![0.0_f64; form.n_dofs()];
        let e0 = form.energy(&zeros);
        // dx = 0 keeps every quadrature point at its reference position: the
        // limiting term vanishes identically.
        assert!(e0.abs() < 1e-12, "limiting energy at dx=0 = {e0}");
        // Any displacement increases the limiting energy.
        let mut dx = vec![0.0_f64; form.n_dofs()];
        for (i, v) in dx.iter_mut().enumerate() {
            *v = 0.01 * (((i * 7) % 11) as f64 - 5.0);
        }
        assert!(form.energy(&dx) > e0);
        // One Newton step strictly reduces the energy.
        let (dx1, res) = tmop_newton_solve(
            &form,
            &TmopLinSolver::Minres(100),
            1e-8,
            8,
            0,
            &SharedMinDet::new(0.0),
        );
        assert!(res.converged || res.iterations > 0);
        assert!(
            form.energy(&dx1) < form.energy(&dx),
            "Newton must reduce the limiting energy"
        );
    }

    /// `enable_normalization` reproduces `ComputeNormalizationEnergies`:
    /// for the perturbed 2x2 mesh with metric 1, metric_normal = 1/E_metric
    /// and the normalized energy satisfies E_norm(x0) = E(x0)/E_metric = 1.
    #[test]
    fn normalization_factors() {
        let mesh: Mesh<2> = Mesh::make_cartesian_2d(2, 2, 1.0, 1.0);
        let order = 2u8;
        let dm = DofManager::new(&mesh, order);
        let topo: &dyn MeshTopology = &mesh;
        // Without limiting: metric_normal = 1/E(x0), lim_energy = NE
        // (IDEAL_SHAPE_UNIT_SIZE has no volume info).
        let mut form = TmopForm::new(topo, &dm, order, TmopQuadType::GaussLegendre, 7);
        let mut x0 = linear_mesh_positions(topo, &dm, order, 2);
        for (i, v) in x0.iter_mut().enumerate() {
            *v += 0.01 * (((i % 5) as f64) - 2.0);
        }
        form.set_x0(x0);
        form.push_integrator(TmopIntegrator {
            metric: metric_from_id_2d(1, &SharedMinDet::new(0.0)).unwrap(),
            target: TmopTarget::new(TmopTargetType::IdealShapeUnitSize),
            coeff: 1.0,
            surf_fit: None,
            metric_normal: 1.0,
            limiting: None,
        });
        form.finalize_targets();
        let zeros = vec![0.0_f64; form.n_dofs()];
        let e_raw = form.energy(&zeros);
        form.enable_normalization();
        assert!((form.integrators()[0].metric_normal - 1.0 / e_raw).abs() < 1e-12);
        let e_norm = form.energy(&zeros);
        assert!((e_norm - 1.0).abs() < 1e-12, "normalized E(x0) = {e_norm}");
        // The normalized functional is the unnormalized one scaled by the
        // constant metric_normal, so the Newton path must visit the identical
        // sequence of meshes: compare the two solutions.
        let (dx_norm, res_norm) = tmop_newton_solve(
            &form,
            &TmopLinSolver::Minres(100),
            1e-10,
            30,
            0,
            &SharedMinDet::new(0.0),
        );
        let mut form_raw = TmopForm::new(topo, &dm, order, TmopQuadType::GaussLegendre, 7);
        form_raw.set_x0(form.x0().to_vec());
        form_raw.push_integrator(TmopIntegrator {
            metric: metric_from_id_2d(1, &SharedMinDet::new(0.0)).unwrap(),
            target: TmopTarget::new(TmopTargetType::IdealShapeUnitSize),
            coeff: 1.0,
            surf_fit: None,
            metric_normal: 1.0,
            limiting: None,
        });
        form_raw.finalize_targets();
        let (dx_raw, res_raw) = tmop_newton_solve(
            &form_raw,
            &TmopLinSolver::Minres(100),
            1e-10,
            30,
            0,
            &SharedMinDet::new(0.0),
        );
        assert_eq!(res_norm.converged, res_raw.converged);
        let max_diff = dx_norm
            .iter()
            .zip(dx_raw.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_diff < 1e-10, "normalized vs raw dx: {max_diff}");
    }

    /// Discrete-adaptivity target: IDEAL_SHAPE_GIVEN_SIZE with a size field
    /// only. The expected element-Jacobians were generated by the C++ probe
    /// (`DiscreteAdaptTC::ComputeElementTargets`, MFEM 4.9/4.10, 2x2 unit quad
    /// mesh, order-2 nodal space, order-1 indicator space, IntRules(SQUARE,3)).
    #[test]
    fn discrete_target_size_vs_cpp() {
        let mesh: Mesh<2> = Mesh::make_cartesian_2d(2, 2, 1.0, 1.0);
        let order = 2u8;
        let dm = DofManager::new(&mesh, order);
        let dm1 = DofManager::new(&mesh, 1);
        let topo: &dyn MeshTopology = &mesh;
        let n1 = dm1.n_dofs;
        // Indicator fields interpolated at the order-1 dofs (the same
        // functions as the C++ probe).
        let size_field: Vec<f64> = (0..n1)
            .map(|d| {
                let c = dm1.dof_coord(d as u32);
                let (x, y) = (c[0], c[1]);
                0.02 + 0.05 * x * y + 0.01 * (3.0 * x).sin()
            })
            .collect();
        let min_size = size_field.iter().cloned().fold(f64::INFINITY, f64::min);
        let element_dofs: Rc<Vec<Vec<usize>>> = Rc::new(
            (0..topo.n_elements())
                .map(|e| dm1.element_dofs(e as u32).iter().map(|&d| d as usize).collect())
                .collect(),
        );
        let mut form = TmopForm::new(topo, &dm, order, TmopQuadType::GaussLegendre, 3);
        form.set_x0(linear_mesh_positions(topo, &dm, order, 2));
        let mut target = TmopTarget::new(TmopTargetType::IdealShapeGivenSizeDiscrete);
        target.discrete = Some(TmopDiscreteSpec::new(
            Some(Rc::new(size_field)),
            None,
            None,
            None,
            min_size,
            1,
            n1,
            element_dofs,
        ));
        form.push_integrator(TmopIntegrator {
            metric: metric_from_id_2d(2, &SharedMinDet::new(0.0)).unwrap(),
            target,
            coeff: 1.0,
            surf_fit: None,
            metric_normal: 1.0,
            limiting: None,
        });
        form.finalize_targets();
        // Element 0 of the mesh: Jtr(q) = size(q)^(1/2) * I (the C++ probe
        // prints the same four values at the IntRules(SQUARE, 3) points).
        let expected: [[f64; 4]; 4] = [
            [0.15055292232997675, 0.0, 0.0, 0.15055292232997675],
            [0.17306163139618605, 0.0, 0.0, 0.17306163139618605],
            [0.15553548878374299, 0.0, 0.0, 0.15553548878374299],
            [0.18879115651236916, 0.0, 0.0, 0.18879115651236916],
        ];
        let mut jtr2: Vec<[f64; 4]> = Vec::new();
        let mut jtr3: Vec<[f64; 9]> = Vec::new();
        let mut dsh = Vec::new();
        let integ = &form.integrators()[0];
        let el = &form.elems[0];
        form.compute_element_targets(integ, 0, el, &mut jtr2, &mut jtr3, &mut dsh);
        assert_eq!(jtr2.len(), 4);
        for (q, e) in expected.iter().enumerate() {
            for c in 0..4 {
                assert!(
                    (jtr2[q][c] - e[c]).abs() < 1e-13,
                    "q{q}[{c}]: {} vs {}",
                    jtr2[q][c],
                    e[c]
                );
            }
        }
    }

    /// Discrete-adaptivity target: GIVEN_SHAPE_AND_SIZE with size + aspect
    /// ratio + skew + orientation fields, element 0 (C++ probe case 2).
    #[test]
    fn discrete_target_shape_and_size_vs_cpp() {
        let mesh: Mesh<2> = Mesh::make_cartesian_2d(2, 2, 1.0, 1.0);
        let order = 2u8;
        let dm = DofManager::new(&mesh, order);
        let dm1 = DofManager::new(&mesh, 1);
        let topo: &dyn MeshTopology = &mesh;
        let n1 = dm1.n_dofs;
        let size_field: Vec<f64> = (0..n1)
            .map(|d| {
                let c = dm1.dof_coord(d as u32);
                let (x, y) = (c[0], c[1]);
                0.02 + 0.05 * x * y + 0.01 * (3.0 * x).sin()
            })
            .collect();
        let aspr_field: Vec<f64> = (0..n1)
            .map(|d| {
                let c = dm1.dof_coord(d as u32);
                let (x, y) = (c[0], c[1]);
                1.0 + 2.0 * x + 0.5 * y * y
            })
            .collect();
        let ori_field: Vec<f64> = (0..n1)
            .map(|d| {
                let c = dm1.dof_coord(d as u32);
                let (x, y) = (c[0], c[1]);
                std::f64::consts::PI * y * (1.0 - y) * (2.0 * std::f64::consts::PI * x).cos()
            })
            .collect();
        let skew_field: Vec<f64> = (0..n1)
            .map(|d| {
                let c = dm1.dof_coord(d as u32);
                let (x, y) = (c[0], c[1]);
                0.3 * x + 0.2 * y
            })
            .collect();
        let min_size = size_field.iter().cloned().fold(f64::INFINITY, f64::min);
        let element_dofs: Rc<Vec<Vec<usize>>> = Rc::new(
            (0..topo.n_elements())
                .map(|e| dm1.element_dofs(e as u32).iter().map(|&d| d as usize).collect())
                .collect(),
        );
        let mut form = TmopForm::new(topo, &dm, order, TmopQuadType::GaussLegendre, 3);
        form.set_x0(linear_mesh_positions(topo, &dm, order, 2));
        let mut target = TmopTarget::new(TmopTargetType::GivenShapeAndSizeDiscrete);
        target.discrete = Some(TmopDiscreteSpec::new(
            Some(Rc::new(size_field)),
            Some(Rc::new(aspr_field)),
            Some(Rc::new(skew_field)),
            Some(Rc::new(ori_field)),
            min_size,
            1,
            n1,
            element_dofs,
        ));
        form.push_integrator(TmopIntegrator {
            metric: metric_from_id_2d(2, &SharedMinDet::new(0.0)).unwrap(),
            target,
            coeff: 1.0,
            surf_fit: None,
            metric_normal: 1.0,
            limiting: None,
        });
        form.finalize_targets();
        let expected: [[f64; 4]; 4] = [
            [
                0.13470303647480519,
                0.16564850745144186,
                0.01294760362425087,
                0.024807727515924173,
            ],
            [
                0.12786586309735384,
                0.23293614487330014,
                -0.012290417170862504,
                0.010164446958251919,
            ],
            [
                0.12729864224224097,
                0.15885574745613223,
                0.047570654930314774,
                0.080332085438534165,
            ],
            [
                0.12873027963043773,
                0.25602514187336189,
                -0.048105648289078547,
                -0.041436951652293399,
            ],
        ];
        let mut jtr2: Vec<[f64; 4]> = Vec::new();
        let mut jtr3: Vec<[f64; 9]> = Vec::new();
        let mut dsh = Vec::new();
        let integ = &form.integrators()[0];
        let el = &form.elems[0];
        form.compute_element_targets(integ, 0, el, &mut jtr2, &mut jtr3, &mut dsh);
        assert_eq!(jtr2.len(), 4);
        for (q, e) in expected.iter().enumerate() {
            for c in 0..4 {
                assert!(
                    (jtr2[q][c] - e[c]).abs() < 1e-13,
                    "q{q}[{c}]: {} vs {}",
                    jtr2[q][c],
                    e[c]
                );
            }
        }
    }

    /// `AdvectorCG` with a linear field and a uniform mesh translation: the
    /// spatial function is preserved (C++ probe, MFEM 4.9 serial: field0 =
    /// x, mesh shifted by +0.1 ⇒ remapped dofs = x + 0.1, clamped to the
    /// original range). A zero-displacement remap (the ProcessNewState(x=0)
    /// solver start) is a value no-op.
    #[test]
    fn advector_cg_uniform_translation() {
        let mesh: Mesh<2> = Mesh::make_cartesian_2d(2, 2, 1.0, 1.0);
        let order = 2u8;
        let dm = DofManager::new(&mesh, order);
        let dm1 = DofManager::new(&mesh, 1);
        let topo: &dyn MeshTopology = &mesh;
        let x0 = linear_mesh_positions(topo, &dm, order, 2);
        let n1 = dm1.n_dofs;
        let field: Vec<f64> = (0..n1).map(|d| dm1.dof_coord(d as u32)[0]).collect();

        let shift = 0.07_f64;
        let mut new_nodes = x0.clone();
        for i in 0..dm.n_dofs {
            new_nodes[i] += shift;
        }

        let mut ev =
            TmopRemapEvaluator::new(TmopRemapKind::AdvectorCG, topo, &dm, order, &dm1, 1, 0.5);
        ev.set_initial_field(&x0, &field);
        // Zero-displacement remap: no change.
        let mut nf = field.clone();
        ev.compute_at_new_position(&x0, &mut nf);
        for (v, w) in nf.iter().zip(field.iter()) {
            assert!((v - w).abs() < 1e-12);
        }
        // Translation: z_new(p) = z0(p), clamped to [min, max] of z0.
        ev.compute_at_new_position(&new_nodes, &mut nf);
        for d in 0..n1 {
            let want = (dm1.dof_coord(d as u32)[0] + shift).min(1.0);
            assert!(
                (nf[d] - want).abs() < 1e-9,
                "dof {d}: {} vs {want}",
                nf[d]
            );
        }
    }

    /// `InterpolatorFP` (findpoints equivalent): direct interpolation of the
    /// initial linear field at the new node positions.
    #[test]
    fn interpolator_fp_linear_field() {
        let mesh: Mesh<2> = Mesh::make_cartesian_2d(2, 2, 1.0, 1.0);
        let order = 2u8;
        let dm = DofManager::new(&mesh, order);
        let dm1 = DofManager::new(&mesh, 1);
        let topo: &dyn MeshTopology = &mesh;
        let x0 = linear_mesh_positions(topo, &dm, order, 2);
        let n1 = dm1.n_dofs;
        let field: Vec<f64> = (0..n1)
            .map(|d| {
                let c = dm1.dof_coord(d as u32);
                2.0 * c[0] - 3.0 * c[1] + 1.0
            })
            .collect();

        let mut ev = TmopRemapEvaluator::new(
            TmopRemapKind::InterpolatorFP,
            topo,
            &dm,
            order,
            &dm1,
            1,
            0.5,
        );
        ev.set_initial_field(&x0, &field);
        let mut new_nodes = x0.clone();
        for i in 0..dm.n_dofs {
            new_nodes[i] += 0.03;
            new_nodes[dm.n_dofs + i] -= 0.02;
        }
        let mut nf = field.clone();
        ev.compute_at_new_position(&new_nodes, &mut nf);
        for d in 0..n1 {
            let c = dm1.dof_coord(d as u32);
            let want = 2.0 * (c[0] + 0.03) - 3.0 * (c[1] - 0.02) + 1.0;
            assert!(
                (nf[d] - want).abs() < 1e-11,
                "dof {d}: {} vs {want}",
                nf[d]
            );
        }
    }

    /// `UpdateTargetSpecification` wiring: a discrete size target with an
    /// attached AdvectorCG remaps the spec values through
    /// `process_new_state`; with a linear size field and a uniform translation
    /// the packed dofs become x + shift.
    #[test]
    fn discrete_target_remapper_updates_spec() {
        let mesh: Mesh<2> = Mesh::make_cartesian_2d(2, 2, 1.0, 1.0);
        let order = 2u8;
        let dm = DofManager::new(&mesh, order);
        let dm1 = DofManager::new(&mesh, 1);
        let topo: &dyn MeshTopology = &mesh;
        let x0 = linear_mesh_positions(topo, &dm, order, 2);
        let n1 = dm1.n_dofs;
        let size: Vec<f64> = (0..n1).map(|d| 0.1 + dm1.dof_coord(d as u32)[0]).collect();
        let element_dofs: Rc<Vec<Vec<usize>>> = Rc::new(
            (0..topo.n_elements())
                .map(|e| dm1.element_dofs(e as u32).iter().map(|&d| d as usize).collect())
                .collect(),
        );
        let min_size = 0.1_f64;

        let mut form = TmopForm::new(topo, &dm, order, TmopQuadType::GaussLegendre, 3);
        form.set_x0(x0.clone());
        let mut target = TmopTarget::new(TmopTargetType::IdealShapeGivenSizeDiscrete);
        target.discrete = Some(TmopDiscreteSpec::new(
            Some(Rc::new(size.clone())),
            None,
            None,
            None,
            min_size,
            1,
            n1,
            element_dofs,
        ));
        form.push_integrator(TmopIntegrator {
            metric: metric_from_id_2d(2, &SharedMinDet::new(0.0)).unwrap(),
            target,
            coeff: 1.0,
            surf_fit: None,
            metric_normal: 1.0,
            limiting: None,
        });
        form.finalize_targets();
        let ev = TmopRemapEvaluator::new(TmopRemapKind::AdvectorCG, topo, &dm, order, &dm1, 1, 0.5);
        form.set_discrete_remapper(0, ev);

        let shift = 0.05_f64;
        let dx: Vec<f64> = (0..form.n_dofs())
            .map(|i| if i < dm.n_dofs { shift } else { 0.0 })
            .collect();
        form.process_new_state(&dx);
        let spec = form.integrators()[0].target.discrete.as_ref().unwrap();
        let cur = spec.remapped.borrow();
        let cur = cur.as_ref().unwrap();
        for d in 0..n1 {
            // Clamp to [min, max] of the pre-advection field (C++ Trim).
            let want = (0.1 + dm1.dof_coord(d as u32)[0] + shift).min(1.1);
            assert!((cur[d] - want).abs() < 1e-9, "dof {d}: {} vs {want}", cur[d]);
        }
    }

    /// Adaptive limiting term, C++-semantics invariants: (i) the penalty
    /// energy measures the remap drift `z − z0` and vanishes identically
    /// (energy AND gradient AND Hessian coupling) while the remapped field
    /// equals its initial state; (ii) with the dofs frozen the energy is
    /// invariant under mesh motion (its `dx`-dependence is carried by the
    /// `process_new_state` remap, exactly like the C++ `ProcessNewState` /
    /// `UpdateAfterMeshPositionChange` split).
    #[test]
    fn adaptive_limiting_semantics() {
        let mesh: Mesh<2> = Mesh::make_cartesian_2d(2, 2, 1.0, 1.0);
        let order = 1u8;
        let dm = DofManager::new(&mesh, order);
        let topo: &dyn MeshTopology = &mesh;
        let n = dm.n_dofs;
        let x0 = linear_mesh_positions(topo, &dm, order, 2);
        let mut form = TmopForm::new(topo, &dm, order, TmopQuadType::GaussLegendre, 4);
        form.set_x0(x0.clone());
        form.push_integrator(TmopIntegrator {
            metric: metric_from_id_2d(1, &SharedMinDet::new(0.0)).unwrap(),
            target: TmopTarget::new(TmopTargetType::IdealShapeUnitSize),
            coeff: 1.0,
            surf_fit: None,
            metric_normal: 1.0,
            limiting: None,
        });
        let z0: Vec<f64> = (0..n)
            .map(|d| {
                let c = dm.dof_coord(d as u32);
                2.0 * c[0] - c[1] + 0.3
            })
            .collect();
        let ev =
            TmopRemapEvaluator::new(TmopRemapKind::AdvectorCG, topo, &dm, order, &dm, order, 0.5);
        form.enable_adaptive_limiting(0, &dm, order, &[z0], vec![0.7], vec![2.0], ev);
        form.finalize_targets();

        // (i) z_cur == z0 (remap has not run): penalty energy is identically 0
        // and the limiter contributes nothing to the gradient.
        let dx: Vec<f64> = (0..form.n_dofs())
            .map(|i| 0.02 * ((i % 7) as f64 - 3.0))
            .collect();
        assert!(form.energy(&dx) > 0.0); // metric part only
        form.set_adaptive_limiting_coeffs(0, &[0.0]);
        let e_metric = form.energy(&dx);
        form.set_adaptive_limiting_coeffs(0, &[0.7]);
        assert!((form.energy(&dx) - e_metric).abs() < 1e-12);
        let mut g = vec![0.0_f64; form.n_dofs()];
        form.gradient(&dx, &mut g);
        form.set_adaptive_limiting_coeffs(0, &[0.0]);
        let mut g0 = vec![0.0_f64; form.n_dofs()];
        form.gradient(&dx, &mut g0);
        for (a, b) in g.iter().zip(g0.iter()) {
            assert!((a - b).abs() < 1e-12, "zero-delta limiter must not act");
        }
        form.set_adaptive_limiting_coeffs(0, &[0.7]);

        // (ii) frozen-dof dx-invariance of the penalty energy.
        let h = 1e-6;
        let mut p = dx.clone();
        p[0] += h;
        let mut m = dx.clone();
        m[0] -= h;
        form.set_adaptive_limiting_coeffs(0, &[0.0]);
        let fd_metric = (form.energy(&p) - form.energy(&m)) / (2.0 * h);
        form.set_adaptive_limiting_coeffs(0, &[0.7]);
        let fd_full = (form.energy(&p) - form.energy(&m)) / (2.0 * h);
        assert!(
            (fd_metric - fd_full).abs() < 1e-9,
            "frozen-dof limiter energy must be dx-invariant: {fd_metric} vs {fd_full}"
        );

        // (iii) a nontrivial remap state produces a nonzero penalty and a
        // nonzero limiter gradient.
        let z_pert: Vec<f64> = (0..n)
            .map(|d| {
                let c = dm.dof_coord(d as u32);
                2.0 * c[0] - c[1] + 0.3 + 0.25 * (1.0 - c[0]) * c[1]
            })
            .collect();
        if let Some(al) = form.adaptive_limiting[0].as_ref() {
            *al.z_cur.borrow_mut() = Rc::new(z_pert);
        }
        form.set_adaptive_limiting_coeffs(0, &[0.0]);
        let e_metric2 = form.energy(&dx);
        form.set_adaptive_limiting_coeffs(0, &[0.7]);
        assert!(
            form.energy(&dx) > e_metric2 + 1e-6,
            "penalty must add energy when z drifts from z0"
        );
        form.gradient(&dx, &mut g);
        let mut max_diff = 0.0_f64;
        for (a, b) in g.iter().zip(g0.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(max_diff > 1e-4, "limiter gradient must act: {max_diff}");
        // Hessian assembles with the limiter and stays finite/positive diag.
        let hess = form.hessian(&dx);
        let mut min_diag = f64::INFINITY;
        for i in 0..form.n_dofs() {
            min_diag = min_diag.min(hess.get(i, i));
        }
        assert!(min_diag.is_finite());
    }
}

