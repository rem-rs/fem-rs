//! MFEM patch-wise integration on NURBS meshes (`fem/intrules.cpp`
//! `IntegrationRule::ApplyToKnotIntervals` + `NURBSMeshRules`, and the
//! `DiffusionIntegrator` patch-aware assembly paths of
//! `fem/integ/bilininteg_diffusion_patch.cpp`).
//!
//! This is the D497 debt: `miniapps/nurbs/nurbs_patch_ex1.rs` needs
//!
//! * `NURBSMeshRules` — a different tensor-product integration rule per patch,
//!   built by stretching one base 1-D segment rule over each knot span
//!   (`ApplyToKnotIntervals`), plus `Finalize`'s maps from patch quadrature
//!   points to knot spans and mesh elements;
//! * the *element-wise* assembly where every element uses the restriction of
//!   its patch's rule to the element's knot span
//!   (`NURBSMeshRules::GetElementRule`, reached through
//!   `Integrator::GetIntegrationRule` when a NURBS patch rule is set) — this
//!   is `nurbs_patch_ex1`'s **default** profile and it differs from the
//!   standard quadrature even without `-patcha`;
//! * the *patch-wise* sparse matrix assembly
//!   (`DiffusionIntegrator::AssemblePatchMatrix_fullQuadrature`), i.e.
//!   `-patcha -fint`.
//!
//! The NURBS space itself comes from `fem_space::nurbs_fe_space::NurbsFESpace`
//! (read-only); nothing here modifies the space crate.

use fem_element::iga::KnotVector;
use fem_element::nurbs_fe_collection::NurbsScalar3D;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_space::nurbs_extension::{NurbsExtension, NurbsKnot};
use fem_space::nurbs_fe_space::NurbsFESpace;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// IntegrationRule::ApplyToKnotIntervals (fem/intrules.cpp)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// The base 1-D segment rule `IntRules.Get(Geometry::SEGMENT, order)`: a
/// Gauss–Legendre rule on `[0,1]` with `(order + 2) / 2` points (the same
/// point-count convention as `fem_space::nurbs_fe_space::nurbs_rule`).
pub fn segment_rule(order: u8) -> Vec<(f64, f64)> {
    let n = ((order as usize + 2) / 2).max(1);
    let (xs, ws) = fem_element::quadrature::gauss_legendre_01(n);
    xs.into_iter().zip(ws).collect()
}

/// `IntegrationRule::ApplyToKnotIntervals(KnotVector const& kv)`: stretch the
/// base rule over each knot span, i.e. point `e*np + j` integrates span `e` of
/// `kv`.  The scan of the knot sequence keeps one running index (`id`) exactly
/// as the C++ does, so spans with repeated knots are handled identically.
pub fn apply_to_knot_intervals(base: &[(f64, f64)], kv: &NurbsKnot) -> Vec<(f64, f64)> {
    let knots = kv.knot_vector().as_slice();
    let ne = kv.n_elements();
    let np = base.len();

    let mut out = Vec::with_capacity(ne * np);
    let mut x1 = knots[0];
    let mut id = 0usize;
    for e in 0..ne {
        let x0 = x1;
        if e == ne - 1 {
            x1 = knots[knots.len() - 1];
        } else {
            while id < knots.len() - 1 {
                id += 1;
                if knots[id] != x0 {
                    x1 = knots[id];
                    break;
                }
            }
        }
        let s = x1 - x0;
        for (bx, bw) in base {
            out.push((x0 + (s * bx), *bw));
        }
    }
    out
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// KnotVector::CalcShape / CalcDShape (mesh/nurbs.cpp) — the *raw* B-spline
// evaluations (no rational weights, no normalisation), used by
// DiffusionIntegrator::SetupPatchBasisData for the patch B/G matrices.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// `KnotVector::CalcShape(shape, i, xi)`: the `order + 1` non-zero B-spline
/// values at the span-local coordinate `xi` of span `i`, in DOF order
/// `N_i .. N_{i+order}` (de Boor triangular scheme, MFEM's operation order).
fn knot_calc_shape(knots: &[f64], order: usize, i: usize, xi: f64, shape: &mut [f64]) {
    let ip = i + order;
    let u = xi * knots[ip + 1] + (1.0 - xi) * knots[ip];
    let mut left = [0.0_f64; 33];
    let mut right = [0.0_f64; 33];
    shape[0] = 1.0;
    for j in 1..=order {
        left[j] = u - knots[ip + 1 - j];
        right[j] = knots[ip + j] - u;
        let mut saved = 0.0_f64;
        for r in 0..j {
            let tmp = shape[r] / (right[r + 1] + left[j - r]);
            shape[r] = saved + right[r + 1] * tmp;
            saved = left[j - r] * tmp;
        }
        shape[j] = saved;
    }
}

/// `KnotVector::CalcDShape(grad, i, xi)`: the first derivatives of the same
/// `order + 1` B-splines ("The NURBS Book" algorithm A2.3), finally scaled by
/// `order * (knots[ip+1] - knots[ip])`.
fn knot_calc_dshape(knots: &[f64], order: usize, i: usize, xi: f64, grad: &mut [f64]) {
    let p = order;
    let ip = i + p;
    let u = xi * knots[ip + 1] + (1.0 - xi) * knots[ip];
    let n_max = p + 1;
    let mut ndu = vec![vec![0.0_f64; n_max]; n_max];
    let mut left = [0.0_f64; 33];
    let mut right = [0.0_f64; 33];

    ndu[0][0] = 1.0;
    for j in 1..=p {
        left[j] = u - knots[ip - j + 1];
        right[j] = knots[ip + j] - u;
        let mut saved = 0.0_f64;
        for r in 0..j {
            ndu[j][r] = right[r + 1] + left[j - r];
            let temp = ndu[r][j - 1] / ndu[j][r];
            ndu[r][j] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }

    for r in 0..=p {
        let mut d = 0.0_f64;
        let pk = p - 1;
        if r >= 1 {
            // C++ `rk = r-1`; `ndu[rk][pk] / ndu[p][rk]`.
            d = ndu[r - 1][pk] / ndu[p][r - 1];
        }
        if r <= pk {
            d -= ndu[r][pk] / ndu[p][r];
        }
        grad[r] = d;
    }

    let scale = p as f64 * (knots[ip + 1] - knots[ip]);
    for g in grad.iter_mut().take(p + 1) {
        *g *= scale;
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// NURBSMeshRules (fem/intrules.cpp)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// One patch's 1-D tensor rules and the `Finalize` maps.  `NURBSMeshRules`
/// carries `patchRules1D` (per patch, per dimension), `patchRules1D_KnotSpan`
/// (the knot span of every 1-D point) and `pointToElem` (the mesh element that
/// contains every tensor point); all three are required by the assembly paths.
#[derive(Debug, Clone)]
pub struct NurbsPatchRules {
    dim: usize,
    /// `patch_rules_1d[patch][dim]` — the `(x, weight)` list in patch
    /// parameter space.
    patch_rules_1d: Vec<Vec<Vec<(f64, f64)>>>,
    /// `knot_span[patch][dim][point]` — `Finalize`'s knot-span index.
    knot_span: Vec<Vec<Vec<usize>>>,
    /// `point_to_elem[patch]`, flattened `i + n0*(j + n1*k)` — the element
    /// containing each tensor point.
    point_to_elem: Vec<Vec<usize>>,
    /// Per-patch point counts `(n0, n1, n2)` (unused dims are 1).
    npoints: Vec<[usize; 3]>,
}

impl NurbsPatchRules {
    /// `NURBSMeshRules(numPatches, dim)`.
    pub fn new(num_patches: usize, dim: usize) -> Self {
        Self {
            dim,
            patch_rules_1d: vec![Vec::new(); num_patches],
            knot_span: Vec::new(),
            point_to_elem: Vec::new(),
            npoints: Vec::new(),
        }
    }

    /// `SetPatchRules1D(patch, ir1D)` — the per-dimension 1-D rules of one
    /// patch (already stretched with [`apply_to_knot_intervals`]).
    pub fn set_patch_rules_1d(&mut self, patch: usize, ir1d: Vec<Vec<(f64, f64)>>) {
        assert_eq!(ir1d.len(), self.dim, "SetPatchRules1D: wrong dimension");
        self.patch_rules_1d[patch] = ir1d;
    }

    /// `GetPatchRule1D(patch, dimension)`.
    pub fn patch_rule_1d(&self, patch: usize, dim: usize) -> &[(f64, f64)] {
        &self.patch_rules_1d[patch][dim]
    }

    /// `GetPatchRule1D_KnotSpan(patch, dimension)`.
    pub fn patch_rule_1d_knot_span(&self, patch: usize, dim: usize) -> &[usize] {
        &self.knot_span[patch][dim]
    }

    /// `GetPointElement(patch, i, j, k)`.
    ///
    /// The linear index must match [`Self::finalize`]'s build order (the `i`
    /// loop outermost, `k` innermost): `i*n1*n2 + j*n2 + k`.  Decoding it as
    /// x-fastest mapped every refined patch point to the wrong mesh element
    /// (invisible at `ref_levels == 0`, where every patch has a single
    /// element, but wrong from the first refinement on — D531).
    pub fn point_element(&self, patch: usize, i: usize, j: usize, k: usize) -> usize {
        let n = &self.npoints[patch];
        self.point_to_elem[patch][i * n[1] * n[2] + j * n[2] + k]
    }

    /// `GetElementRule(elem, patch, ijk, kv)` — the restriction of the patch's
    /// tensor rule to the element's knot span.  Per dimension, keep the points
    /// `x` with `kv0 <= x < kv1` (or `x <= kv1` on the last span) and map them
    /// to the span-local coordinate `(x - kv0) / (kv1 - kv0)`; the tensor
    /// weights are the products of the 1-D weights.  Returns the
    /// (points, weights) of the element rule with `x` varying fastest.
    pub fn element_rule(
        &self,
        patch: usize,
        ijk: &[usize; 3],
        kv: &[&NurbsKnot],
    ) -> (Vec<[f64; 3]>, Vec<f64>) {
        let mut el: [Vec<[f64; 2]>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for d in 0..self.dim {
            let knots = kv[d].knot_vector().as_slice();
            let order = kv[d].order();
            let kv0 = knots[order + ijk[d]];
            let kv1 = knots[order + ijk[d] + 1];
            let right_end = order + ijk[d] + 1 == knots.len() - 1;
            for &(x, w) in &self.patch_rules_1d[patch][d] {
                if kv0 <= x && (x < kv1 || right_end) {
                    el[d].push([(x - kv0) / (kv1 - kv0), w]);
                }
            }
        }
        let npd = [el[0].len(), el[1].len(), el[2].len()];
        assert!(npd[0] > 0 && npd[1] > 0, "element rule: empty span rule");
        let k_len = if self.dim > 2 { npd[2] } else { 1 };
        let np = npd[0] * npd[1] * k_len;
        let mut points = Vec::with_capacity(np);
        let mut weights = Vec::with_capacity(np);
        // x varies fastest (`i + j*npd[0] + k*npd[0]*npd[1]`), as in the C++.
        for k in 0..k_len {
            for j in 0..npd[1] {
                for i in 0..npd[0] {
                    let z = if self.dim > 2 { el[2][k][0] } else { 0.0 };
                    points.push([el[0][i][0], el[1][j][0], z]);
                    let mut w = el[0][i][1] * el[1][j][1];
                    if self.dim > 2 {
                        w *= el[2][k][1];
                    }
                    weights.push(w);
                }
            }
        }
        (points, weights)
    }

    /// `NURBSMeshRules::Finalize(mesh)` — build the knot-span index of every
    /// 1-D point and the tensor-point → element map.
    pub fn finalize(&mut self, ext: &NurbsExtension) {
        let npatches = self.patch_rules_1d.len();
        assert!(npatches > 0 && npatches == ext.n_patches());

        // First, find all the elements in each patch (ascending element order,
        // as in the C++ loop over `mesh.GetNE()`).
        let mut patch_elements: Vec<Vec<usize>> = vec![Vec::new(); npatches];
        for e in 0..ext.n_elements() {
            patch_elements[ext.element_patch(e)].push(e);
        }

        self.knot_span = vec![Vec::new(); npatches];
        self.point_to_elem = Vec::with_capacity(npatches);
        self.npoints = Vec::with_capacity(npatches);

        for p in 0..npatches {
            let pkv = ext.patch_knot_vectors(p).expect("patch knot vectors");
            let mut maxijk = [1usize; 3];
            let mut np = [1usize; 3];
            let mut ijk2elem: Vec<Vec<Vec<i64>>> = Vec::new();
            for d in 0..self.dim {
                maxijk[d] = pkv[d].nks();
                np[d] = self.patch_rules_1d[p][d].len();
            }
            for _ in 0..maxijk[0] {
                let mut yz = Vec::with_capacity(maxijk[1]);
                for _ in 0..maxijk[1] {
                    yz.push(vec![-1_i64; maxijk[2]]);
                }
                ijk2elem.push(yz);
            }
            for &e in &patch_elements[p] {
                let ijk = ext.element_ijk(e);
                assert!(ijk2elem[ijk[0]][ijk[1]][ijk[2]] == -1);
                ijk2elem[ijk[0]][ijk[1]][ijk[2]] = e as i64;
            }

            // For each point, find its knot span.
            self.knot_span[p] = vec![Vec::new(); self.dim];
            for d in 0..self.dim {
                let knots = pkv[d].knot_vector().as_slice();
                let order = pkv[d].order();
                self.knot_span[p][d] = vec![0usize; np[d]];
                for (r, &(x, _)) in self.patch_rules_1d[p][d].iter().enumerate() {
                    let mut ijk_d = 0usize;
                    loop {
                        let kv0 = knots[order + ijk_d];
                        let kv1 = knots[order + ijk_d + 1];
                        let right_end = order + ijk_d + 1 == knots.len() - 1;
                        if kv0 <= x && (x < kv1 || right_end) {
                            break;
                        }
                        ijk_d += 1;
                    }
                    self.knot_span[p][d][r] = ijk_d;
                }
            }

            self.point_to_elem.push(Vec::with_capacity(np[0] * np[1] * np[2]));
            self.npoints.push(np);
            for i in 0..np[0] {
                for j in 0..np[1] {
                    for k in 0..np[2] {
                        let elem = ijk2elem[self.knot_span[p][0][i]]
                            [self.knot_span[p][1][j]]
                            [self.knot_span[p][2][k]];
                        assert!(elem >= 0, "Finalize: point outside the mesh");
                        self.point_to_elem[p].push(elem as usize);
                    }
                }
            }
        }
    }

    /// `SetupPatchBasisData` (bilininteg_diffusion_patch.cpp): the patch-level
    /// 1-D basis matrices `B`/`G` (values and derivatives of *all* patch DOFs
    /// at *all* patch points) plus the point↔DOF interaction ranges.
    /// `pkv` are the patch's knot vectors.
    fn setup_basis_data(&self, p: usize, pkv: &[&NurbsKnot]) -> PatchBasisData {
        let dim = self.dim;
        let mut q1d = [1usize; 3];
        let mut d1d = [1usize; 3];
        let mut b: Vec<Vec<f64>> = Vec::new();
        let mut g: Vec<Vec<f64>> = Vec::new();

        let mut min_d_v = [Vec::new(), Vec::new(), Vec::new()];
        let mut max_d_v = [Vec::new(), Vec::new(), Vec::new()];
        let mut min_q_v = [Vec::new(), Vec::new(), Vec::new()];
        let mut max_q_v = [Vec::new(), Vec::new(), Vec::new()];

        for d in 0..dim {
            let ir1d = self.patch_rules_1d[p][d].clone();
            let knot_span = &self.knot_span[p][d];
            q1d[d] = ir1d.len();
            let knots = pkv[d].knot_vector().as_slice().to_vec();
            let order = pkv[d].order();
            d1d[d] = pkv[d].ncp();

            let mut bd = vec![0.0_f64; q1d[d] * d1d[d]];
            let mut gd = vec![0.0_f64; q1d[d] * d1d[d]];
            let mut mind = vec![q1d[d]; d1d[d]];
            let mut maxd = vec![0usize; d1d[d]];
            let mut minq = vec![d1d[d]; q1d[d]];
            let mut maxq = vec![0usize; q1d[d]];

            // A raw B-spline evaluator on the patch knot vector (unit
            // weights), i.e. `pkv[d]->CalcShape/CalcDShape`: `knot_calc_*`.
            let mut shape = vec![0.0_f64; order + 1];
            let mut dshape = vec![0.0_f64; order + 1];

            for i in 0..q1d[d] {
                let ipx = ir1d[i].0;
                let ijk = knot_span[i];
                let kv0 = knots[order + ijk];
                let mut kv1 = knots[0];
                for &k in knots.iter().skip(order + ijk + 1) {
                    if k > kv0 {
                        kv1 = k;
                        break;
                    }
                }
                assert!(kv1 > kv0);
                let xi = (ipx - kv0) / (kv1 - kv0);

                knot_calc_shape(&knots, order, ijk, xi, &mut shape);
                knot_calc_dshape(&knots, order, ijk, xi, &mut dshape);

                for (j, s) in shape.iter().enumerate() {
                    bd[i * d1d[d] + ijk + j] = *s;
                    mind[ijk + j] = mind[ijk + j].min(i);
                    maxd[ijk + j] = maxd[ijk + j].max(i);
                }
                for (j, s) in dshape.iter().enumerate() {
                    gd[i * d1d[d] + ijk + j] = *s;
                }
                minq[i] = minq[i].min(ijk);
                maxq[i] = maxq[i].max(ijk + order);
            }

            b.push(bd);
            g.push(gd);
            min_d_v[d] = mind;
            max_d_v[d] = maxd;
            min_q_v[d] = minq;
            max_q_v[d] = maxq;
        }

        let mut min_dd = [Vec::new(), Vec::new(), Vec::new()];
        let mut max_dd = [Vec::new(), Vec::new(), Vec::new()];
        for d in 0..dim {
            for i in 0..d1d[d] {
                min_dd[d].push(min_q_v[d][min_d_v[d][i]]);
                max_dd[d].push(max_q_v[d][max_d_v[d][i]]);
            }
        }

        PatchBasisData {
            q1d,
            d1d,
            b,
            g,
            min_d: min_d_v,
            max_d: max_d_v,
            min_q: min_q_v,
            max_q: max_q_v,
            min_dd,
            max_dd,
        }
    }
}

/// The per-patch basis data of [`NurbsPatchRules::setup_basis_data`].
#[allow(clippy::type_complexity)]
struct PatchBasisData {
    q1d: [usize; 3],
    d1d: [usize; 3],
    /// `b[d]` is the `Q1D[d] x D1D[d]` matrix, row-major in points.
    b: Vec<Vec<f64>>,
    g: Vec<Vec<f64>>,
    min_d: [Vec<usize>; 3],
    max_d: [Vec<usize>; 3],
    min_q: [Vec<usize>; 3],
    max_q: [Vec<usize>; 3],
    min_dd: [Vec<usize>; 3],
    max_dd: [Vec<usize>; 3],
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Element-wise assembly with the patch-rule restriction
// (`BilinearForm::Assemble`'s element loop with
// `Integrator::GetIntegrationRule` returning `NURBSMeshRules::GetElementRule`)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// `BilinearForm(DiffusionIntegrator(one) with a NURBS patch rule).Assemble()`
/// for a scalar `NurbsFESpace`: every element integrates `-∇·κ∇` with the
/// restriction of its patch's rule (`kappa` multiplies the result, i.e. the
/// `ConstantCoefficient` of the miniapp).
///
/// The element arithmetic (weight/`detJ` handling, `dshape · adj(J)`,
/// accumulation order) mirrors `DiffusionIntegrator::AssembleElementMatrix`
/// exactly as `NurbsFESpace::assemble_bilinear` does; only the integration
/// rule differs.
pub fn assemble_diffusion_patch_rules(
    space: &NurbsFESpace,
    rules: &NurbsPatchRules,
    kappa: f64,
) -> CsrMatrix<f64> {
    let dim = space.dim();
    assert_eq!(dim, rules.dim, "patch rules dimension mismatch");
    let ext = space.extension();
    let n = space.n_dofs();
    let mut coo = CooMatrix::new(n, n);
    coo.reserve(n * 16);

    let mut grad = Vec::new();
    let mut dshapedxt = Vec::new();
    let mut k_elem = Vec::new();
    for e in 0..space.n_elements() {
        let nd = space.element_n_dofs(e);
        let patch = ext.element_patch(e);
        let ijk = ext.element_ijk(e);
        let pkv = ext.patch_knot_vectors(patch).expect("patch knot vectors");
        let (points, weights) = rules.element_rule(patch, &ijk, &pkv);

        k_elem.clear();
        k_elem.resize(nd * nd, 0.0);
        grad.clear();
        grad.resize(nd * dim, 0.0);
        dshapedxt.clear();
        dshapedxt.resize(nd * dim, 0.0);

        let fe = space.element_fe(e);
        for (q, xi) in points.iter().enumerate() {
            let geo = space.geometry(e, xi);
            // MFEM: w = ip.weight / Weight(); dshapedxt = dshape * adj(J);
            //       elmat += w * dshapedxt * dshapedxt^T.
            let adj = adjugate(&geo.jac, dim);
            let w = weights[q] / geo.det_j;
            fe.grad(xi, &mut grad);
            for i in 0..nd {
                for k in 0..dim {
                    let mut s = 0.0;
                    for j in 0..dim {
                        s += grad[i * dim + j] * adj[j][k];
                    }
                    dshapedxt[i * dim + k] = s;
                }
            }
            for i in 0..nd {
                for j in 0..nd {
                    let mut s = 0.0;
                    for k in 0..dim {
                        s += dshapedxt[i * dim + k] * dshapedxt[j * dim + k];
                    }
                    k_elem[i * nd + j] += w * s;
                }
            }
        }
        for v in k_elem.iter_mut() {
            *v *= kappa;
        }
        coo.add_element_matrix(ext.element_dofs(e), &k_elem);
    }
    coo.into_csr()
}

/// `CalcAdjugate` of a `dim x dim` Jacobian (same helper as in
/// `fem_space::nurbs_fe_space`).
fn adjugate(j: &[[f64; 3]; 3], dim: usize) -> [[f64; 3]; 3] {
    let mut a = [[0.0_f64; 3]; 3];
    if dim == 2 {
        a[0][0] = j[1][1];
        a[0][1] = -j[0][1];
        a[1][0] = -j[1][0];
        a[1][1] = j[0][0];
    } else {
        a[0][0] = j[1][1] * j[2][2] - j[1][2] * j[2][1];
        a[0][1] = j[0][2] * j[2][1] - j[0][1] * j[2][2];
        a[0][2] = j[0][1] * j[1][2] - j[0][2] * j[1][1];
        a[1][0] = j[1][2] * j[2][0] - j[1][0] * j[2][2];
        a[1][1] = j[0][0] * j[2][2] - j[0][2] * j[2][0];
        a[1][2] = j[0][2] * j[1][0] - j[0][0] * j[1][2];
        a[2][0] = j[1][0] * j[2][1] - j[1][1] * j[2][0];
        a[2][1] = j[0][1] * j[2][0] - j[0][0] * j[2][1];
        a[2][2] = j[0][0] * j[1][1] - j[0][1] * j[1][0];
    }
    a
}

/// `DenseMatrix::Det` (row-0 cofactor expansion, MFEM's product order).
fn det3_mfem(j: &[[f64; 3]; 3]) -> f64 {
    j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
        + j[1][0] * (j[0][2] * j[2][1] - j[0][1] * j[2][2])
        + j[2][0] * (j[0][1] * j[1][2] - j[0][2] * j[1][1])
}

/// MFEM-faithful NURBS element geometry: the control points are the mesh's
/// current control net — the file's `FiniteElementSpace` block
/// (`NurbsExtension::parse_nodes`) on an unrefined mesh, or the knot-insert
/// refined net (`NurbsMeshGeometry::from_mesh_nodes` of
/// `NurbsFESpace::mesh_nodes`) after `ref_levels > 0` uniform refinements —
/// and every quantity is evaluated with MFEM's exact operation order
/// (`IsoparametricTransformation`: `J = PointMat·dshape`, `Weight = Det J`,
/// `AdjugateJacobian = CalcAdjugate J`).
///
/// This is what makes the D497 dumps byte-comparable: the rational-geometry
/// evaluation of `NurbsFESpace::geometry` is mathematically identical but
/// rounds differently in the last ulp.
pub struct NurbsMeshGeometry {
    /// Control point coordinates in extension-DOF order.
    coords: Vec<[f64; 3]>,
    /// Per-DOF rational weights, read from the file's `weights` section the
    /// way MFEM's `weights.Load(input, GetNDof())` does: the *first* `n_dofs`
    /// values of the section (ball-nurbs.mesh stores 1040 values of which
    /// only the first `GetNDof()` = 517 are consumed; `NurbsFESpace`'s
    /// loader currently rejects the mismatching count and falls back to unit
    /// weights, which is wrong for this mesh).
    weights: Vec<f64>,
}

impl NurbsMeshGeometry {
    /// Read the control points and weights of the mesh file.  Requires
    /// `orders` of the space to equal the mesh's own orders (so that the
    /// analysis DOF numbering *is* the control-point numbering).
    pub fn from_mesh_text(text: &str, ext: &NurbsExtension) -> Result<Self, String> {
        let nodes = NurbsExtension::parse_nodes(text, ext.n_dofs())?;
        if nodes.vdim != 3 {
            return Err(format!("expected vdim 3 control points, got {}", nodes.vdim));
        }
        let mut coords = Vec::with_capacity(nodes.coords.len());
        for c in &nodes.coords {
            coords.push([c[0], c[1], c[2]]);
        }
        let mut weights = vec![1.0_f64; ext.n_dofs()];
        if let Some(w) = parse_weights_section(text) {
            if w.len() >= ext.n_dofs() {
                weights.copy_from_slice(&w[..ext.n_dofs()]);
            }
        }
        Ok(Self { coords, weights })
    }

    /// Build from the mesh's **current** control net — MFEM
    /// `mesh->GetNodes()` after `NURBSUniformRefinement` — and the matching
    /// (refined) extension: the coordinates are
    /// [`NurbsExtension::refined_control_points`]' dehomogenized A5.5 net
    /// ([`NurbsFESpace::mesh_nodes`]) and the weights the refined extension's
    /// ([`NurbsExtension::uniform_refinement`]).  This is what makes the
    /// `-patcha -fint -ref > 0` assembly bit-comparable: evaluating the
    /// *refined* control points reproduces MFEM's operation order, while
    /// re-evaluating the original net over the refined parameter intervals
    /// rounds differently in the last ulp.
    pub fn from_mesh_nodes(coords: &[Vec<f64>], ext: &NurbsExtension) -> Result<Self, String> {
        let vdim = coords
            .first()
            .map(|c| c.len())
            .ok_or_else(|| "NurbsMeshGeometry::from_mesh_nodes: no control points".to_string())?;
        if vdim != 3 {
            return Err(format!("expected vdim 3 control points, got {vdim}"));
        }
        if coords.len() != ext.n_dofs() {
            return Err(format!(
                "NurbsMeshGeometry::from_mesh_nodes: {} control points for {} DOFs",
                coords.len(),
                ext.n_dofs()
            ));
        }
        let coords = coords.iter().map(|c| [c[0], c[1], c[2]]).collect();
        Ok(Self {
            coords,
            weights: ext.weights().to_vec(),
        })
    }

    fn element_dofs(&self, space: &NurbsFESpace, e: usize) -> Vec<[f64; 3]> {
        space
            .extension()
            .element_dofs(e)
            .iter()
            .map(|&g| self.coords[g])
            .collect()
    }

    /// The per-element weights, indexed like [`NurbsExtension::element_dofs`].
    pub fn element_weights(&self, space: &NurbsFESpace, e: usize) -> Vec<f64> {
        space
            .extension()
            .element_dofs(e)
            .iter()
            .map(|&g| self.weights[g])
            .collect()
    }
}

/// The numeric payload of the mesh file's `weights` section (all tokens after
/// the keyword; `Array::Load` consumes only the first `GetNDof()` of them).
pub fn parse_weights_section(text: &str) -> Option<Vec<f64>> {
    let mut in_section = false;
    let mut out = Vec::new();
    for line in text.lines() {
        let line = match line.find('#') {
            Some(i) => &line[..i],
            None => line,
        };
        for tok in line.split_whitespace() {
            if !in_section {
                if tok == "weights" {
                    in_section = true;
                }
                continue;
            }
            match tok.parse::<f64>() {
                Ok(v) => out.push(v),
                Err(_) => return Some(out),
            }
        }
    }
    if in_section { Some(out) } else { None }
}

/// The per-element data of [`NurbsMeshGeometry`]: point matrix (rows = local
/// DOFs), weights, and the element's knot vectors with span index.
pub struct ExactElement {
    pub pm: Vec<[f64; 3]>,
    pub fe: NurbsScalar3D,
    pub ijk: [usize; 3],
}

impl ExactElement {
    fn new(space: &NurbsFESpace, geo: &NurbsMeshGeometry, e: usize) -> Self {
        let ext = space.extension();
        let dim = ext.dim();
        assert_eq!(dim, 3, "exact NURBS patch assembly supports 3D meshes");
        let patch = ext.element_patch(e);
        let ijk = ext.element_ijk(e);
        let pkv = ext.patch_knot_vectors(patch).expect("patch knot vectors");
        let kv: Vec<KnotVector> = pkv
            .iter()
            .map(|k| {
                KnotVector::new_clamped(k.knot_vector().as_slice().to_vec())
                    .expect("knot vector")
            })
            .collect();
        let mut fe = NurbsScalar3D::new(
            kv[0].clone(),
            kv[1].clone(),
            kv[2].clone(),
        )
        .expect("NurbsScalar3D");
        fe.set_ijk(ijk);
        let wts = geo.element_weights(space, e);
        fe.set_weights(wts).expect("element weights");
        let pm = geo.element_dofs(space, e);
        Self { pm, fe, ijk }
    }

    /// `IsoparametricTransformation::EvalJacobian`: `J = PointMat·dshape` with
    /// the rational derivatives of `NURBS3DFiniteElement::CalcDShape`.
    pub fn jacobian(&self, xi: &[f64; 3]) -> [[f64; 3]; 3] {
        let nd = self.pm.len();
        let mut grads = vec![0.0_f64; nd * 3];
        self.fe.calc_grad(xi, &mut grads);
        let mut jac = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0;
                for o in 0..nd {
                    s += self.pm[o][i] * grads[o * 3 + j];
                }
                jac[i][j] = s;
            }
        }
        jac
    }
}

/// `LinearForm(DomainLFIntegrator(one)).Assemble()` with MFEM's rule
/// (`IntRules.Get(CUBE, 2·order)`, i.e. `p+1` Gauss points per direction) and
/// arithmetic (`val = Weight()·Q; add(elvect, ip.weight·val, shape, elvect)`).
pub fn assemble_domain_lf_exact(
    space: &NurbsFESpace,
    geo: &NurbsMeshGeometry,
    f: &dyn Fn(&[f64]) -> f64,
) -> Vec<f64> {
    let ext = space.extension();
    let mut rhs = vec![0.0_f64; space.n_dofs()];
    // `IntRules.Get(Geometry::CUBE, 2·order)`: the tensor rule of the segment
    // rule of order `2·order` (`(2·order + 2) / 2 = order + 1` points), x
    // varying fastest, as in MFEM's three-rule tensor constructor.  The
    // element order is the maximum of the space's per-direction orders
    // (`NURBSFiniteElement::GetOrder`).
    let fe_order = space.orders().iter().copied().max().unwrap_or(1);
    let base = segment_rule(2 * fe_order as u8);
    for e in 0..space.n_elements() {
        let el = ExactElement::new(space, geo, e);
        let nd = el.pm.len();
        let mut elvec = vec![0.0_f64; nd];
        let mut shape = vec![0.0_f64; nd];
        for &(bz, bzw) in base.iter() {
            for &(by, byw) in base.iter() {
                for &(bx, bxw) in base.iter() {
                    let xi = [bx, by, bz];
                    let jac = el.jacobian(&xi);
                    // `val = Tr.Weight() * Q.Eval(Tr, ip)` with `Q = 1`.
                    let val = det3_mfem(&jac) * f(&physical_point(&el, &xi));
                    // `add(elvect, ip.weight * val, shape, elvect)`.
                    let w = bxw * byw * bzw * val;
                    el.fe.calc_shape(&xi, &mut shape);
                    for (o, ev) in elvec.iter_mut().enumerate() {
                        *ev += w * shape[o];
                    }
                }
            }
        }
        for (o, &g) in ext.element_dofs(e).iter().enumerate() {
            rhs[g] += elvec[o];
        }
    }
    rhs
}

/// `IsoparametricTransformation::Transform`: `x = Σ shape·pm`.
fn physical_point(el: &ExactElement, xi: &[f64; 3]) -> [f64; 3] {
    let nd = el.pm.len();
    let mut shape = vec![0.0_f64; nd];
    el.fe.calc_shape(xi, &mut shape);
    let mut x = [0.0_f64; 3];
    for o in 0..nd {
        for i in 0..3 {
            x[i] += shape[o] * el.pm[o][i];
        }
    }
    x
}

/// `BilinearForm(DiffusionIntegrator(one) *without* patch rules).Assemble()` —
/// `nurbs_patch_ex1`'s second, comparison solve — with MFEM's exact
/// arithmetic: the default rule
/// `IntRules.Get(CUBE, 2·order + dim − 1)` (`DiffusionIntegrator::GetRule`,
/// Qk branch → `p + 2` Gauss points per direction) and [`NurbsMeshGeometry`]'s
/// point matrices.  Requires the [`NurbsMeshGeometry`] point matrix (the file
/// control net, or the refined net via [`NurbsMeshGeometry::from_mesh_nodes`]).
pub fn assemble_diffusion_standard_exact(
    space: &NurbsFESpace,
    geo: &NurbsMeshGeometry,
    kappa: f64,
) -> CsrMatrix<f64> {
    let ext = space.extension();
    let dim = ext.dim();
    assert_eq!(dim, 3, "exact NURBS patch assembly supports 3D meshes");
    let n = space.n_dofs();
    let mut coo = CooMatrix::new(n, n);
    coo.reserve(n * 16);

    let fe_order = space.orders().iter().copied().max().unwrap_or(1);
    let rule_order = 2 * fe_order as usize + dim - 1;
    let base = segment_rule(rule_order as u8);

    for e in 0..space.n_elements() {
        let el = ExactElement::new(space, geo, e);
        let nd = el.pm.len();

        let mut dshape = vec![0.0_f64; nd * 3];
        let mut dshapedxt = vec![0.0_f64; nd * 3];
        let mut k_elem = vec![0.0_f64; nd * nd];

        for &(bz, bzw) in base.iter() {
            for &(by, byw) in base.iter() {
                for &(bx, bxw) in base.iter() {
                    let xi = [bx, by, bz];
                    let ipw = bxw * byw * bzw;

                    // `el.CalcDShape(ip, dshape)` before `SetIntPoint`.
                    el.fe.calc_grad(&xi, &mut dshape);
                    let jac = el.jacobian(&xi);
                    let adj = adjugate(&jac, 3);
                    let w = ipw / det3_mfem(&jac);
                    for i in 0..nd {
                        for c in 0..3 {
                            let mut s = 0.0;
                            for j in 0..3 {
                                s += dshape[i * 3 + j] * adj[j][c];
                            }
                            dshapedxt[i * 3 + c] = s;
                        }
                    }
                    let a = w * kappa;
                    // `AddMult_a_AAt(w, dshapedxt, elmat)`.
                    for i in 0..nd {
                        for j in 0..i {
                            let mut d = 0.0_f64;
                            for c in 0..3 {
                                d += dshapedxt[i * 3 + c] * dshapedxt[j * 3 + c];
                            }
                            let d = d * a;
                            k_elem[i * nd + j] += d;
                            k_elem[j * nd + i] += d;
                        }
                        let mut d = 0.0_f64;
                        for c in 0..3 {
                            d += dshapedxt[i * 3 + c] * dshapedxt[i * 3 + c];
                        }
                        k_elem[i * nd + i] += a * d;
                    }
                }
            }
        }
        coo.add_element_matrix(ext.element_dofs(e), &k_elem);
    }
    coo.into_csr()
}

/// Test hook: element `e`'s MFEM-faithful data (`crates/assembly/tests`).
pub fn testsupport_element(space: &NurbsFESpace, geo: &NurbsMeshGeometry, e: usize) -> ExactElement {
    ExactElement::new(space, geo, e)
}

/// Test hook: `DenseMatrix::Det` (`crates/assembly/tests`).
pub fn testsupport_det(j: &[[f64; 3]; 3]) -> f64 {
    det3_mfem(j)
}

/// Test hook: `CalcAdjugate` (`crates/assembly/tests`).
pub fn testsupport_adjugate(j: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    adjugate(j, 3)
}

/// `BilinearForm(DiffusionIntegrator(one) with a NURBS patch rule).Assemble()`
/// evaluated with MFEM's exact arithmetic
/// ([`NurbsMeshGeometry`] point matrices, `w = ip.weight/Weight()`,
/// `dshapedxt = dshape·adj(J)`, `AddMult_a_AAt(w, dshapedxt, elmat)`).
///
/// Requires the mesh to be unrefined (`ref_levels == 0`), so that the file's
/// control points are the point matrix.  `kappa` multiplies the element
/// matrix (the miniapp's `ConstantCoefficient`).
pub fn assemble_diffusion_patch_rules_exact(
    space: &NurbsFESpace,
    geo: &NurbsMeshGeometry,
    rules: &NurbsPatchRules,
    kappa: f64,
) -> CsrMatrix<f64> {
    let ext = space.extension();
    assert_eq!(ext.dim(), rules.dim, "patch rules dimension mismatch");
    let n = space.n_dofs();
    let mut coo = CooMatrix::new(n, n);
    coo.reserve(n * 16);

    for e in 0..space.n_elements() {
        let el = ExactElement::new(space, geo, e);
        let nd = el.pm.len();
        let patch = ext.element_patch(e);
        let pkv = ext.patch_knot_vectors(patch).expect("patch knot vectors");
        let (points, weights) = rules.element_rule(patch, &el.ijk, &pkv);

        let mut dshape = vec![0.0_f64; nd * 3];
        let mut dshapedxt = vec![0.0_f64; nd * 3];
        let mut k_elem = vec![0.0_f64; nd * nd];

        for (q, xi) in points.iter().enumerate() {
            // `el.CalcDShape(ip, dshape)` (before `SetIntPoint`; the NURBS
            // element does not depend on the transformation).
            el.fe.calc_grad(xi, &mut dshape);
            let jac = el.jacobian(xi);
            let adj = adjugate(&jac, 3);
            // `w = ip.weight / Trans.Weight()`; `Mult(dshape, adjJ, dshapedxt)`.
            let w = weights[q] / det3_mfem(&jac);
            for i in 0..nd {
                for k in 0..3 {
                    let mut s = 0.0;
                    for j in 0..3 {
                        s += dshape[i * 3 + j] * adj[j][k];
                    }
                    dshapedxt[i * 3 + k] = s;
                }
            }
            // `AddMult_a_AAt(w, dshapedxt, elmat)` with the constant
            // coefficient folded into `w` (MFEM multiplies `w *= Q` first).
            let a = w * kappa;
            for i in 0..nd {
                for j in 0..i {
                    let mut d = 0.0_f64;
                    for k in 0..3 {
                        d += dshapedxt[i * 3 + k] * dshapedxt[j * 3 + k];
                    }
                    let d = d * a;
                    k_elem[i * nd + j] += d;
                    k_elem[j * nd + i] += d;
                }
                let mut d = 0.0_f64;
                for k in 0..3 {
                    d += dshapedxt[i * 3 + k] * dshapedxt[i * 3 + k];
                }
                k_elem[i * nd + i] += a * d;
            }
        }
        coo.add_element_matrix(ext.element_dofs(e), &k_elem);
    }
    coo.into_csr()
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Patch-wise assembly (`-patcha -fint`): DiffusionIntegrator::
// AssemblePatchMatrix_fullQuadrature + BilinearForm::Assemble's patch loop
// (`GetPatchVDofs` + `mat->AddRow(vdofs[r], ...)`).
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// `BilinearForm(DiffusionIntegrator(one) in Mode::PATCHWISE).Assemble()` for a
/// scalar `NurbsFESpace`: one sparse matrix per patch, assembled by the
/// full-quadrature patch contraction of
/// `AssemblePatchMatrix_fullQuadrature`, then scattered to global DOFs with
/// the patch's VDOF map (`GetPatchVDofs`).
///
/// Note that this path inherits MFEM's unit-weight assumption for the patch
/// basis (`SetupPatchBasisData` uses the raw B-spline knot shapes), and its
/// quadrature-point Jacobians are evaluated through the element
/// transformation exactly as `SetupPatchPA` does (the element of
/// [`NurbsPatchRules::point_element`] is transformed at the *patch-level*
/// integration point).  Requires the [`NurbsMeshGeometry`] point matrix (the
/// file control net, or the refined net via
/// [`NurbsMeshGeometry::from_mesh_nodes`]).
pub fn assemble_diffusion_patchwise(
    space: &NurbsFESpace,
    geo: &NurbsMeshGeometry,
    rules: &NurbsPatchRules,
    kappa: f64,
) -> CsrMatrix<f64> {
    let dim = space.dim();
    assert_eq!(dim, 3, "patch-wise assembly supports 3D only (as in MFEM)");
    let ext = space.extension();
    let n = space.n_dofs();
    let mut coo = CooMatrix::new(n, n);

    for p in 0..ext.n_patches() {
        let pkv = ext.patch_knot_vectors(p).expect("patch knot vectors");
        let basis = rules.setup_basis_data(p, &pkv);

        // SetupPatchPA: quadrature-point data over the whole patch (weights
        // and Jacobians, evaluated through the element containing each point).
        let nq = basis.q1d[0] * basis.q1d[1] * basis.q1d[2];
        // `pa_data`: symmetric 3x3 diffusivity at every point (6 entries).
        let mut pa_data = vec![0.0_f64; nq * 6];
        let mut jac_cache = vec![[0.0_f64; 9]; nq];
        let mut weights = vec![0.0_f64; nq];
        for k in 0..basis.q1d[2] {
            for j in 0..basis.q1d[1] {
                for i in 0..basis.q1d[0] {
                    let q = i + basis.q1d[0] * (j + basis.q1d[1] * k);
                    let ipx = rules.patch_rules_1d[p][0][i].0;
                    let ipy = rules.patch_rules_1d[p][1][j].0;
                    let ipz = rules.patch_rules_1d[p][2][k].0;
                    let w = rules.patch_rules_1d[p][0][i].1
                        * rules.patch_rules_1d[p][1][j].1
                        * rules.patch_rules_1d[p][2][k].1;
                    weights[q] = w;
                    let e = rules.point_element(p, i, j, k);
                    // C++ `SetupPatchPA`: `tr->SetIntPoint(&ip)` with the
                    // *patch-level* point on the element's transformation —
                    // the element is evaluated at span-local coordinates
                    // numerically equal to the patch coordinates.
                    let el = ExactElement::new(space, geo, e);
                    let jac = el.jacobian(&[ipx, ipy, ipz]);
                    for r in 0..dim {
                        for c in 0..dim {
                            jac_cache[q][r * 3 + c] = jac[r][c];
                        }
                    }
                }
            }
        }
        for q in 0..nq {
            let j11 = jac_cache[q][0];
            let j21 = jac_cache[q][3];
            let j31 = jac_cache[q][6];
            let j12 = jac_cache[q][1];
            let j22 = jac_cache[q][4];
            let j32 = jac_cache[q][7];
            let j13 = jac_cache[q][2];
            let j23 = jac_cache[q][5];
            let j33 = jac_cache[q][8];
            let det_j = j11 * (j22 * j33 - j32 * j23) - j21 * (j12 * j33 - j32 * j13)
                + j31 * (j12 * j23 - j22 * j13);
            let w_det_j = weights[q] / det_j;
            let a11 = (j22 * j33) - (j23 * j32);
            let a12 = (j32 * j13) - (j12 * j33);
            let a13 = (j12 * j23) - (j22 * j13);
            let a21 = (j31 * j23) - (j21 * j33);
            let a22 = (j11 * j33) - (j13 * j31);
            let a23 = (j21 * j13) - (j11 * j23);
            let a31 = (j21 * j32) - (j31 * j22);
            let a32 = (j31 * j12) - (j11 * j32);
            let a33 = (j11 * j22) - (j12 * j21);
            // Scalar (unit) diffusivity: `w/detJ adj(J) adj(J)^T`.
            let c = kappa;
            pa_data[6 * q] = w_det_j * (c * a11 * a11 + c * a12 * a12 + c * a13 * a13);
            pa_data[6 * q + 1] = w_det_j * (c * a11 * a21 + c * a12 * a22 + c * a13 * a23);
            pa_data[6 * q + 2] = w_det_j * (c * a11 * a31 + c * a12 * a32 + c * a13 * a33);
            pa_data[6 * q + 3] = w_det_j * (c * a21 * a21 + c * a22 * a22 + c * a23 * a23);
            pa_data[6 * q + 4] = w_det_j * (c * a21 * a31 + c * a22 * a32 + c * a23 * a33);
            pa_data[6 * q + 5] = w_det_j * (c * a31 * a31 + c * a32 * a32 + c * a33 * a33);
        }

        // AssemblePatchMatrix_fullQuadrature: the patch matrix in patch-DOF
        // order with its exact banded sparsity.
        let ndof = basis.d1d[0] * basis.d1d[1] * basis.d1d[2];
        let mut smat_i = vec![0usize; ndof + 1];
        for dof_j in 0..ndof {
            let (jdx, jdy, jdz) = dof_indices(dof_j, &basis.d1d);
            let mut ndd = 1usize;
            for d in 0..3 {
                let jd = [jdx, jdy, jdz][d];
                ndd *= basis.max_dd[d][jd] - basis.min_dd[d][jd] + 1;
            }
            smat_i[dof_j + 1] = smat_i[dof_j] + ndd;
        }
        let nnz = smat_i[ndof];
        let mut smat_j = vec![-1_i64; nnz];
        let mut smat_a = vec![0.0_f64; nnz];

        let mut grad: Vec<Vec<f64>> = vec![
            vec![0.0; basis.q1d[0] * basis.q1d[1] * basis.q1d[2]],
            vec![0.0; basis.q1d[0] * basis.q1d[1] * basis.q1d[2]],
            vec![0.0; basis.q1d[0] * basis.q1d[1] * basis.q1d[2]],
        ];
        let mut grad_dxy = vec![0.0_f64; basis.d1d[0] * basis.d1d[1] * 3];
        let mut grad_dx = vec![0.0_f64; basis.d1d[0] * 3];

        for dof_j in 0..ndof {
            let (jdx, jdy, jdz) = dof_indices(dof_j, &basis.d1d);
            let nd = [
                basis.max_dd[0][jdx] - basis.min_dd[0][jdx] + 1,
                basis.max_dd[1][jdy] - basis.min_dd[1][jdy] + 1,
                basis.max_dd[2][jdz] - basis.min_dd[2][jdz] + 1,
            ];

            // cdofs(i,j,k) of the interacting DOFs.
            let cdofs = |i: usize, j: usize, k: usize| -> usize {
                basis.min_dd[0][jdx]
                    + i
                    + basis.d1d[0]
                        * (basis.min_dd[1][jdy]
                            + j
                            + basis.d1d[1] * (basis.min_dd[2][jdz] + k))
            };

            let zero_grad = |grad: &mut Vec<Vec<f64>>| {
                for qz in basis.min_d[2][jdz]..=basis.max_d[2][jdz] {
                    for qy in basis.min_d[1][jdy]..=basis.max_d[1][jdy] {
                        for qx in basis.min_d[0][jdx]..=basis.max_d[0][jdx] {
                            let q = qx + (qy + qz * basis.q1d[1]) * basis.q1d[0];
                            for gv in grad.iter_mut() {
                                gv[q] = 0.0;
                            }
                        }
                    }
                }
            };
            zero_grad(&mut grad);

            let qpoint = |qx: usize, qy: usize, qz: usize| -> usize {
                qx + (qy + qz * basis.q1d[1]) * basis.q1d[0]
            };

            for qz in basis.min_d[2][jdz]..=basis.max_d[2][jdz] {
                let wz = basis.b[2][qz * basis.d1d[2] + jdz];
                let w_dz = basis.g[2][qz * basis.d1d[2] + jdz];
                for qy in basis.min_d[1][jdy]..=basis.max_d[1][jdy] {
                    let wy = basis.b[1][qy * basis.d1d[1] + jdy];
                    let w_dy = basis.g[1][qy * basis.d1d[1] + jdy];
                    for qx in basis.min_d[0][jdx]..=basis.max_d[0][jdx] {
                        let q = qpoint(qx, qy, qz);
                        let o11 = pa_data[6 * q];
                        let o12 = pa_data[6 * q + 1];
                        let o13 = pa_data[6 * q + 2];
                        let o22 = pa_data[6 * q + 3];
                        let o23 = pa_data[6 * q + 4];
                        let o33 = pa_data[6 * q + 5];
                        let wx = basis.b[0][qx * basis.d1d[0] + jdx];
                        let w_dx = basis.g[0][qx * basis.d1d[0] + jdx];

                        let grad_x = w_dx * wy * wz;
                        let grad_y = wx * w_dy * wz;
                        let grad_z = wx * wy * w_dz;

                        grad[0][q] = (o11 * grad_x) + (o12 * grad_y) + (o13 * grad_z);
                        grad[1][q] = (o12 * grad_x) + (o22 * grad_y) + (o23 * grad_z);
                        grad[2][q] = (o13 * grad_x) + (o23 * grad_y) + (o33 * grad_z);
                    }
                }
            }

            for qz in basis.min_d[2][jdz]..=basis.max_d[2][jdz] {
                for dy in basis.min_dd[1][jdy]..=basis.max_dd[1][jdy] {
                    for dx in basis.min_dd[0][jdx]..=basis.max_dd[0][jdx] {
                        for d in 0..3 {
                            grad_dxy[(dx * basis.d1d[1] + dy) * 3 + d] = 0.0;
                        }
                    }
                }
                for qy in basis.min_d[1][jdy]..=basis.max_d[1][jdy] {
                    // C++ also declares unused `wy`/`wDy` at this level.
                    for dx in basis.min_dd[0][jdx]..=basis.max_dd[0][jdx] {
                        for d in 0..3 {
                            grad_dx[dx * 3 + d] = 0.0;
                        }
                    }
                    for qx in basis.min_d[0][jdx]..=basis.max_d[0][jdx] {
                        let q = qpoint(qx, qy, qz);
                        let gx = grad[0][q];
                        let gy = grad[1][q];
                        let gz = grad[2][q];
                        for dx in basis.min_q[0][qx]..=basis.max_q[0][qx] {
                            let wx = basis.b[0][qx * basis.d1d[0] + dx];
                            let w_dx = basis.g[0][qx * basis.d1d[0] + dx];
                            grad_dx[dx * 3] += gx * w_dx;
                            grad_dx[dx * 3 + 1] += gy * wx;
                            grad_dx[dx * 3 + 2] += gz * wx;
                        }
                    }
                    for dy2 in basis.min_q[1][qy]..=basis.max_q[1][qy] {
                        let wy2 = basis.b[1][qy * basis.d1d[1] + dy2];
                        let w_dy2 = basis.g[1][qy * basis.d1d[1] + dy2];
                        for dx in basis.min_dd[0][jdx]..=basis.max_dd[0][jdx] {
                            grad_dxy[(dx * basis.d1d[1] + dy2) * 3] += grad_dx[dx * 3] * wy2;
                            grad_dxy[(dx * basis.d1d[1] + dy2) * 3 + 1] +=
                                grad_dx[dx * 3 + 1] * w_dy2;
                            grad_dxy[(dx * basis.d1d[1] + dy2) * 3 + 2] += grad_dx[dx * 3 + 2] * wy2;
                        }
                    }
                }
                for dz in basis.min_q[2][qz]..=basis.max_q[2][qz] {
                    let wz = basis.b[2][qz * basis.d1d[2] + dz];
                    let w_dz = basis.g[2][qz * basis.d1d[2] + dz];
                    for dy in basis.min_dd[1][jdy]..=basis.max_dd[1][jdy] {
                        for dx in basis.min_dd[0][jdx]..=basis.max_dd[0][jdx] {
                            let v = (grad_dxy[(dx * basis.d1d[1] + dy) * 3] * wz)
                                + (grad_dxy[(dx * basis.d1d[1] + dy) * 3 + 1] * wz)
                                + (grad_dxy[(dx * basis.d1d[1] + dy) * 3 + 2] * w_dz);
                            let loc = dx
                                - basis.min_dd[0][jdx]
                                + nd[0]
                                    * (dy - basis.min_dd[1][jdy]
                                        + nd[1] * (dz - basis.min_dd[2][jdz]));
                            let odof = cdofs(
                                dx - basis.min_dd[0][jdx],
                                dy - basis.min_dd[1][jdy],
                                dz - basis.min_dd[2][jdz],
                            );
                            let m = smat_i[dof_j] + loc;
                            debug_assert!(
                                smat_j[m] == -1 || smat_j[m] == odof as i64,
                                "patch matrix: sparsity collision"
                            );
                            smat_j[m] = odof as i64;
                            smat_a[m] += v;
                        }
                    }
                }
            }
        }

        // BilinearForm::Assemble's patch loop: map the patch rows to global
        // DOFs (`GetPatchVDofs` order = x-fastest patch-CP order).
        let mut vdofs = Vec::with_capacity(ndof);
        for k in 0..basis.d1d[2] {
            for j in 0..basis.d1d[1] {
                for i in 0..basis.d1d[0] {
                    vdofs.push(
                        ext.patch_dof(p, &[i, j, k])
                            .expect("patch DOF in range"),
                    );
                }
            }
        }
        for r in 0..ndof {
            for m in smat_i[r]..smat_i[r + 1] {
                if smat_j[m] >= 0 {
                    coo.add(vdofs[r], vdofs[smat_j[m] as usize], smat_a[m]);
                }
            }
        }
    }
    coo.into_csr()
}

/// `(jdx, jdy, jdz)` of a patch-DOF index (x fastest, as in
/// `AssemblePatchMatrix_fullQuadrature`).
fn dof_indices(dof: usize, d1d: &[usize; 3]) -> (usize, usize, usize) {
    let jdz = dof / (d1d[0] * d1d[1]);
    let jdy = (dof - jdz * d1d[0] * d1d[1]) / d1d[0];
    let jdx = dof - jdz * d1d[0] * d1d[1] - jdy * d1d[0];
    (jdx, jdy, jdz)
}
