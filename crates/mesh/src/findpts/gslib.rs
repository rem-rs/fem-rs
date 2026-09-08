//! Serial FindPointsGSLIB-equivalent locator for general meshes.
//!
//! Semantics mirror MFEM `FindPointsGSLIB` (serial, CPU): given physical
//! points, find the containing element and reference coordinates, and report
//! for every point
//!
//! - `code`: [`CODE_INSIDE`] (0) when the point lies strictly inside the
//!   element, [`CODE_BORDER`] (1) when it is found on (or within tolerance
//!   of) the element border, [`CODE_NOT_FOUND`] (2) when no element matched;
//! - `dist2`: squared physical distance between the query point and its
//!   located position (MFEM `GetDist`);
//! - `elem` / `xi`: element id and reference coordinates in `[0, 1]`
//!   (MFEM `GetElements` / `GetReferenceCoordinates` after
//!   `MapRefPosAndElemIndices`).
//!
//! Unlike [`super::find_points::FindPoints`] (affine simplices only), this
//! module runs Newton iteration on the full isoparametric element map
//! ([`Mesh::element_jacobian`]), so it handles straight and curved quads,
//! hexes, prisms, triangles and tets — the element families MFEM's
//! FindPointsGSLIB supports.
//!
//! Reference-coordinate convention: this module always works in the MFEM
//! convention `xi ∈ [0, 1]^D` (triangles: `x, y >= 0, x + y <= 1`; prisms:
//! `xi[0]` axial in `[0, 1]`, `(xi[1], xi[2])` the triangle coordinates).
//! [`Mesh::element_jacobian`] uses the `fem_element` factory conventions per
//! family ([`HexQk`] evaluates on `[-1, 1]^3`, prisms keep the axial
//! coordinate first, …), so [`to_factory_coords`] / [`from_factory_coords`]
//! translate, and the Jacobian is rescaled accordingly.
//!
//! The 0/1 code split uses MFEM's `Geometry::CheckPoint(geom, ip, -1e-12)`
//! rule: a found point is "inside" only if it is at least [`STRICT_TOL`]
//! inside the reference domain, otherwise it is "on border".
//!
//! [`HexQk`]: fem_element::lagrange::factory::HexQk

use fem_core::ElemId;

use crate::element_type::ElementType;
use crate::Mesh;

use super::bvh::Bvh;

/// Point found strictly inside an element (MFEM code 0).
pub const CODE_INSIDE: u32 = 0;
/// Point found on an element border / within tolerance of it (MFEM code 1).
pub const CODE_BORDER: u32 = 1;
/// Point not found (MFEM code 2).
pub const CODE_NOT_FOUND: u32 = 2;

/// MFEM `FindPointsGSLIB` default Newton tolerance (`newt_tol`).
pub const DEFAULT_NEWT_TOL: f64 = 1.0e-12;
/// MFEM `FindPointsGSLIB` default border distance tolerance (`bdr_tol`);
/// compared against the *squared* distance of border-found points.
pub const DEFAULT_BDR_TOL: f64 = 1.0e-8;
/// MFEM `MapRefPosAndElemIndices` `rbtol`: a found point counts as strictly
/// inside only if it is at least this far inside the reference domain.
pub const STRICT_TOL: f64 = 1.0e-12;

/// Result of locating one physical point.
#[derive(Debug, Clone, Copy)]
pub struct GslibPoint<const D: usize> {
    /// MFEM code: 0 inside, 1 on border, 2 not found.
    pub code: u32,
    /// Containing element (0 when not found, as in MFEM).
    pub elem: ElemId,
    /// Reference coordinates in `[0, 1]` (MFEM mapped reference position).
    pub xi: [f64; D],
    /// Squared distance between the query point and the located position.
    pub dist2: f64,
}

impl<const D: usize> Default for GslibPoint<D> {
    fn default() -> Self {
        Self {
            code: CODE_NOT_FOUND,
            elem: 0,
            xi: [-1.0; D],
            dist2: f64::INFINITY,
        }
    }
}

/// Element families with simplex (barycentric) reference domains.
fn is_simplex(et: ElementType) -> bool {
    matches!(et, ElementType::Tri3 | ElementType::Tri6 | ElementType::Tet4 | ElementType::Tet10)
}

/// Serial FindPointsGSLIB-equivalent locator over a general [`Mesh`].
pub struct GslibFindPoints<'a, const D: usize> {
    mesh: &'a Mesh<D>,
    bvh: Bvh<D>,
    /// Newton convergence tolerance (physical residual).
    pub newt_tol: f64,
    /// Squared-distance tolerance for border-found points (MFEM `bdr_tol`).
    pub bdr_tol: f64,
    /// Reference-space slack for accepting a Newton result (`inside_tol`).
    pub inside_tol: f64,
    /// Maximum Newton iterations per candidate element.
    pub max_iter: usize,
    /// Maximum number of candidate elements tried (BVH order).
    pub max_candidates: usize,
}

/// Element families supported by the isoparametric Newton search.
///
/// Each entry has a matching `fem_element` Lagrange factory element whose DOF
/// count equals the element's node count at the mesh geometry order.
fn is_supported(et: ElementType) -> bool {
    is_simplex(et)
        || matches!(
            et,
            ElementType::Quad4
                | ElementType::Quad9
                | ElementType::Hex8
                | ElementType::Hex27
                | ElementType::Prism6
                | ElementType::Prism18
        )
}

/// Is `xi` (factory convention) inside the factory reference domain with a
/// slack of `t`?  `simplex`: barycentric test; prism: axial + triangle; hex /
/// quad: box test (factory coords: hex on `[-1, 1]`, quad on `[0, 1]`).
fn factory_in_range(et: ElementType, fxi: &[f64], t: f64) -> bool {
    match et {
        ElementType::Tri3 | ElementType::Tri6 => {
            let mut sum = 0.0;
            for &v in fxi {
                if v < -t {
                    return false;
                }
                sum += v;
            }
            sum <= 1.0 + t
        }
        ElementType::Tet4 | ElementType::Tet10 => {
            let mut sum = 0.0;
            for &v in fxi {
                if v < -t {
                    return false;
                }
                sum += v;
            }
            sum <= 1.0 + t
        }
        ElementType::Prism6 | ElementType::Prism18 => {
            // PrismPk: xi[0] axial, (xi[1], xi[2]) triangle coordinates.
            let (s, x, y) = (fxi[0], fxi[1], fxi[2]);
            s >= -t && x >= -t && y >= -t && x + y <= 1.0 + t
        }
        ElementType::Hex8 | ElementType::Hex27 => {
            // HexQk evaluates on [-1, 1]^3.
            fxi.iter().all(|&v| v >= -1.0 - t && v <= 1.0 + t)
        }
        _ => fxi.iter().all(|&v| v >= -t && v <= 1.0 + t),
    }
}

/// Map canonical `[0, 1]^D` coordinates to the family's factory convention,
/// returning `(factory_xi, jacobian_scale)` where `jacobian_scale` is the
/// factor `d(factory_xi) / d(canonical_xi)` per axis (diagonal map).
fn to_factory_coords<const D: usize>(et: ElementType, xi: &[f64; D]) -> (Vec<f64>, [f64; D]) {
    match et {
        ElementType::Hex8 | ElementType::Hex27 => {
            let f: Vec<f64> = xi.iter().map(|&v| 2.0 * v - 1.0).collect();
            let scale = [2.0; D];
            (f, scale)
        }
        _ => (xi.to_vec(), [1.0; D]),
    }
}

/// Inverse of [`to_factory_coords`]: factory reference coordinates →
/// canonical `[0, 1]^D`.
fn from_factory_coords<const D: usize>(et: ElementType, fxi: &[f64]) -> [f64; D] {
    match et {
        ElementType::Hex8 | ElementType::Hex27 => {
            std::array::from_fn(|d| 0.5 * (fxi[d] + 1.0))
        }
        _ => std::array::from_fn(|d| fxi[d]),
    }
}

impl<'a, const D: usize> GslibFindPoints<'a, D> {
    /// Build the locator (BVH over element vertex AABBs).
    pub fn new(mesh: &'a Mesh<D>) -> Self {
        for e in 0..mesh.n_elems() as ElemId {
            let et = mesh.element_type_at(e);
            assert!(
                is_supported(et),
                "GslibFindPoints: unsupported element type {et:?}"
            );
        }
        let bvh = Bvh::new_with_aabbs(mesh, Self::geometry_aabbs(mesh));
        Self {
            mesh,
            bvh,
            newt_tol: DEFAULT_NEWT_TOL,
            bdr_tol: DEFAULT_BDR_TOL,
            inside_tol: 1.0e-3,
            max_iter: 20,
            max_candidates: 16,
        }
    }

    /// Per-element AABBs covering the curved geometry.
    ///
    /// Straight meshes use the corner vertices; curved meshes bound **all**
    /// high-order geometry nodes of each element (GSlib bounds the curved
    /// element with rigorous Lobatto polynomial bounds; the node hull is the
    /// practical equivalent since the GLL interpolation of the geometry is
    /// exact between nodes) plus a small safety margin.
    fn geometry_aabbs(mesh: &Mesh<D>) -> Vec<super::bvh::Aabb<D>> {
        let g = mesh.geometry.as_ref();
        let npe: usize = match g {
            Some(g) => g.nodes_per_elem,
            None => 0,
        };
        (0..mesh.n_elems() as ElemId)
            .map(|e| {
                let ids: Vec<u32> = match g {
                    Some(g) => g.conn[e as usize * npe..(e as usize + 1) * npe].to_vec(),
                    None => mesh.elem_nodes(e).to_vec(),
                };
                let coord = |id: u32| -> [f64; D] {
                    match g {
                        Some(g) => std::array::from_fn(|d| g.coords[id as usize * D + d]),
                        None => mesh.coords_of(id),
                    }
                };
                let mut lo = coord(ids[0]);
                let mut hi = lo;
                for &id in ids.iter().skip(1) {
                    let c = coord(id);
                    for d in 0..D {
                        lo[d] = lo[d].min(c[d]);
                        hi[d] = hi[d].max(c[d]);
                    }
                }
                // Safety margin: 25% of the element's box diagonal per side
                // (equispaced high-order nodes can undershoot the true curved
                // extent between nodes; gslib uses rigorous Lobatto bounds).
                let mut diag2 = 0.0;
                for d in 0..D {
                    diag2 += (hi[d] - lo[d]) * (hi[d] - lo[d]);
                }
                let pad = 0.25 * diag2.sqrt() + 1e-14;
                for d in 0..D {
                    lo[d] -= pad;
                    hi[d] += pad;
                }
                super::bvh::Aabb::new(lo, hi)
            })
            .collect()
    }

    /// Locate a batch of points (MFEM `FindPoints`).
    pub fn find_points(&self, points: &[[f64; D]]) -> Vec<GslibPoint<D>> {
        points.iter().map(|p| self.find_point(p)).collect()
    }

    /// Locate one physical point (MFEM `FindPoints` for a single point).
    pub fn find_point(&self, p: &[f64; D]) -> GslibPoint<D> {
        let mut best = GslibPoint::<D>::default();

        // Candidate elements: AABBs expanded by a small physical radius
        // (mirrors MFEM's border-found behaviour: points within sqrt(bdr_tol)
        // outside the mesh are reported with code 1).
        let diag: f64 = {
            let (lo, hi) = self.mesh.bounding_box();
            let mut d2: f64 = 0.0;
            for d in 0..D {
                d2 += (hi[d] - lo[d]) * (hi[d] - lo[d]);
            }
            d2.sqrt()
        };
        let cand_tol = 1.0e-4 * diag;

        let candidates = self.bvh.locate_candidates(p, cand_tol);
        let n_cand = candidates.len().min(self.max_candidates);

        // `newton` only returns results whose reference coordinates lie
        // inside the (slightly expanded) reference domain, so every surviving
        // candidate is a genuine containment.  Prefer a candidate that
        // contains the point strictly (MFEM code 0) over one where the point
        // sits on the border (code 1); among equals keep the smallest
        // residual.
        for &e in candidates.iter().take(n_cand) {
            let Some((xi, dist2)) = self.newton(e, p) else {
                continue;
            };
            if !best.dist2.is_finite() {
                best.dist2 = dist2;
                best.elem = e;
                best.xi = xi;
                continue;
            }
            let inside_new = self.strictly_inside(self.mesh.element_type_at(e), &xi);
            let inside_best = self.strictly_inside(self.mesh.element_type_at(best.elem), &best.xi);
            let better = if inside_new != inside_best {
                inside_new
            } else {
                dist2 < best.dist2
            };
            if better {
                best.dist2 = dist2;
                best.elem = e;
                best.xi = xi;
            }
        }

        if best.dist2.is_finite() {
            // Strict-inside test: MFEM CheckPoint(geom, ip, -rbtol).
            let et = self.mesh.element_type_at(best.elem);
            best.code = if self.strictly_inside(et, &best.xi) {
                CODE_INSIDE
            } else {
                CODE_BORDER
            };
            // MFEM: border-found points farther than bdr_tol are not found.
            if best.code == CODE_BORDER && best.dist2 > self.bdr_tol {
                best = GslibPoint::<D>::default();
            }
        }
        best
    }

    /// Newton iteration on the isoparametric map of element `e`.
    ///
    /// Multi-start: the (clamped and unclamped) affine inverse guess and the
    /// element center are tried in turn; the first start whose iteration
    /// converges with reference coordinates inside the (slightly expanded)
    /// reference domain wins.  On strongly curved elements a single affine
    /// start can diverge or converge to a far-away preimage of the
    /// (polynomial) extended map, so the in-range check is essential.
    ///
    /// Returns `(xi, dist2)` with `xi` in canonical `[0, 1]^D` coordinates and
    /// `dist2 = ||x(xi) - p||^2`.
    fn newton(&self, e: ElemId, p: &[f64; D]) -> Option<([f64; D], f64)> {
        let ns = self.mesh.elem_nodes(e);
        if ns.len() < D + 1 {
            return None;
        }
        let et = self.mesh.element_type_at(e);

        // Affine inverse (canonical coords) from the corner nodes.
        let x0 = self.mesh.coords_of(ns[0]);
        let mut jac0 = nalgebra::DMatrix::<f64>::zeros(D, D);
        // Corner offsets per family (canonical): axis k direction.
        let ax: [usize; D] = if et == ElementType::Prism6 || et == ElementType::Prism18 {
            // Connectivity [b0 b1 b2 t0 t1 t2]: tri axes at nodes 1, 2,
            // axial axis at node 3.  Canonical coords [axial, tx, ty].
            let mut a = [0usize; D];
            a[0] = 3;
            a[1] = 1;
            a[2] = 2;
            a
        } else {
            let mut a = [0usize; D];
            for (k, v) in a.iter_mut().enumerate() {
                *v = k + 1;
            }
            a
        };
        for (k, &nk) in ax.iter().enumerate() {
            let xk = self.mesh.coords_of(ns[nk]);
            for i in 0..D {
                jac0[(i, k)] = xk[i] - x0[i];
            }
        }
        let affine = jac0
            .clone()
            .try_inverse()
            .map(|inv| inv * nalgebra::DVector::from_fn(D, |i, _| p[i] - x0[i]));

        let mut starts: Vec<[f64; D]> = Vec::with_capacity(4);
        // Start 0: reference coordinates of the geometrically nearest
        // high-order geometry node (Gauss-Lobatto point).  From within the
        // node's cell the Newton map is well-behaved even on curved elements.
        {
            let geo_order = self.mesh.geom_order().max(1);
            let fe = fem_element::lagrange::factory::ref_elem(et.to_elem_type(), geo_order);
            let rc = fe.dof_coords();
            let npe = fe.n_dofs();
            let g = self.mesh.geometry.as_ref();
            let conn: Vec<u32> = match g {
                Some(g) => g.conn[e as usize * npe..(e as usize + 1) * npe].to_vec(),
                None => ns.to_vec(),
            };
            let coord = |k: usize| -> [f64; D] {
                let id = conn[k] as usize;
                match g {
                    Some(g) => std::array::from_fn(|d| g.coords[id * D + d]),
                    None => self.mesh.coords_of(conn[k]),
                }
            };
            let mut best_k = 0usize;
            let mut best_d2 = f64::INFINITY;
            for k in 0..npe {
                let c = coord(k);
                let d2: f64 = (0..D).map(|d| (c[d] - p[d]) * (c[d] - p[d])).sum();
                if d2 < best_d2 {
                    best_d2 = d2;
                    best_k = k;
                }
            }
            starts.push(from_factory_coords(et, &rc[best_k]));
        }
        if let Some(v) = &affine {
            let mut a = [0.0_f64; D];
            for i in 0..D {
                // Clamp the extrapolated affine guess near the element.
                a[i] = v[i].clamp(0.0, 1.0);
            }
            starts.push(a);
        }
        starts.push([0.5; D]);
        if let Some(v) = &affine {
            let mut a = [0.0_f64; D];
            for i in 0..D {
                a[i] = v[i].clamp(-0.5, 1.5);
            }
            starts.push(a);
        }

        let mut best: Option<([f64; D], f64)> = None;
        for start in &starts {
            let mut xi = *start;
            let mut converged = false;
            for _ in 0..self.max_iter {
                let (j, _det, xmap) = self.isoparametric(e, et, &xi);
                let mut r = [0.0_f64; D];
                let mut r2 = 0.0;
                for i in 0..D {
                    r[i] = p[i] - xmap[i];
                    r2 += r[i] * r[i];
                }
                if r2 < self.newt_tol * self.newt_tol {
                    converged = true;
                    // One extra quadratic polish step: the tolerance check can
                    // trigger a step before machine precision, while the
                    // converged root is exact.  Never accept a polish step
                    // that moves a strictly-inside point across the element
                    // boundary (would flip the MFEM code 0 → 1).
                    if let Some(inv) = j.clone().try_inverse() {
                        let delta = inv * nalgebra::DVector::from_fn(D, |i, _| r[i]);
                        let mut xj = xi;
                        let mut ok = true;
                        for i in 0..D {
                            xj[i] += delta[i];
                            if !xj[i].is_finite() || xj[i].abs() > 10.0 {
                                ok = false;
                                break;
                            }
                        }
                        if ok {
                            let (_j2, _d2, xm2) = self.isoparametric(e, et, &xj);
                            let mut r2n = 0.0;
                            for i in 0..D {
                                let d = p[i] - xm2[i];
                                r2n += d * d;
                            }
                            // Accept the polished root when it lowers the
                            // residual and does not move a strictly-inside
                            // point across the element boundary (which would
                            // flip the MFEM code 0 → 1).
                            let inside_before = self.strictly_inside(et, &xi);
                            let accept =
                                r2n < r2 && (!inside_before || self.strictly_inside(et, &xj));
                            if accept {
                                xi = xj;
                            }
                        }
                    }
                    break;
                }
                let Some(inv) = j.clone().try_inverse() else {
                    break;
                };
                let delta = inv * nalgebra::DVector::from_fn(D, |i, _| r[i]);
                let mut ok = true;
                for i in 0..D {
                    xi[i] += delta[i];
                    // Runaway protection (far-outside preimages of the
                    // extended polynomial map are spurious).
                    if !xi[i].is_finite() || xi[i].abs() > 10.0 {
                        ok = false;
                        break;
                    }
                }
                if !ok {
                    break;
                }
            }
            if !converged {
                continue;
            }
            let (_j, _det, xmap) = self.isoparametric(e, et, &xi);
            let mut d2 = 0.0;
            for i in 0..D {
                let d = p[i] - xmap[i];
                d2 += d * d;
            }
            let (fxi, _scale) = to_factory_coords(et, &xi);
            if !factory_in_range(et, &fxi, self.inside_tol) {
                continue;
            }
            // Keep the smallest-residual in-range result across starts.
            match &best {
                Some((_, bd2)) if d2 >= *bd2 => {}
                _ => best = Some((xi, d2)),
            }
            if d2 < self.newt_tol * self.newt_tol {
                break; // fully converged in-range result
            }
        }
        best
    }

    /// Physical map of element `e` at canonical reference coords `xi`:
    /// returns the Jacobian `dx/dxi` (canonical), `det`, and `x(xi)`.
    fn isoparametric(
        &self,
        e: ElemId,
        et: ElementType,
        xi: &[f64; D],
    ) -> (nalgebra::DMatrix<f64>, f64, Vec<f64>) {
        let (fxi, scale) = to_factory_coords(et, xi);
        let (j, _det, x) = self.mesh.element_jacobian(e, &fxi);
        // Chain rule: dx/dxi_canonical = dx/dxi_factory * dxi_factory/dxi.
        let mut jc: nalgebra::DMatrix<f64> = j.clone();
        for k in 0..D {
            for i in 0..D {
                jc[(i, k)] *= scale[k];
            }
        }
        let det = if D == 2 {
            jc[(0, 0)] * jc[(1, 1)] - jc[(0, 1)] * jc[(1, 0)]
        } else {
            jc.determinant()
        };
        (jc, det, x)
    }

    /// MFEM `CheckPoint(geom, ip, -STRICT_TOL)`: strictly inside test in
    /// canonical `[0, 1]^D` coordinates.
    fn strictly_inside(&self, et: ElementType, xi: &[f64; D]) -> bool {
        let t = STRICT_TOL;
        if is_simplex(et) {
            let mut sum = 0.0;
            for &v in xi {
                if v < t {
                    return false;
                }
                sum += v;
            }
            sum <= 1.0 - t
        } else if et == ElementType::Prism6 || et == ElementType::Prism18 {
            let (s, x, y) = (xi[0], xi[1], xi[2]);
            s >= t && x >= t && y >= t && x + y <= 1.0 - t
        } else {
            xi.iter().all(|&v| v >= t && v <= 1.0 - t)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gslib_finds_interior_quad() {
        let m = Mesh::<2>::unit_square_quad(3);
        let fp = GslibFindPoints::new(&m);
        let r = fp.find_point(&[0.42, 0.61]);
        assert_eq!(r.code, CODE_INSIDE);
        assert!(r.dist2 < 1e-20);
        let (_j, _det, xmap) = m.element_jacobian(r.elem, &r.xi);
        assert!((xmap[0] - 0.42).abs() < 1e-10);
        assert!((xmap[1] - 0.61).abs() < 1e-10);
    }

    #[test]
    fn gslib_finds_interior_tri() {
        let m = Mesh::<2>::unit_square_tri(4);
        let fp = GslibFindPoints::new(&m);
        let r = fp.find_point(&[0.37, 0.41]);
        assert_eq!(r.code, CODE_INSIDE);
        assert!(r.dist2 < 1e-24);
    }

    #[test]
    fn gslib_border_point_code_1() {
        let m = Mesh::<2>::unit_square_quad(3);
        let fp = GslibFindPoints::new(&m);
        // Point exactly on the bottom boundary of the square.
        let r = fp.find_point(&[0.5, 0.0]);
        assert_eq!(r.code, CODE_BORDER);
        assert!(r.dist2 < 1e-20);
    }

    #[test]
    fn gslib_outside_point_not_found() {
        let m = Mesh::<2>::unit_square_quad(3);
        let fp = GslibFindPoints::new(&m);
        let r = fp.find_point(&[1.5, 0.3]);
        assert_eq!(r.code, CODE_NOT_FOUND);
        // Just outside (within bdr_tol): found on border.
        let r2 = fp.find_point(&[1.0 + 1e-5, 0.3]);
        assert_eq!(r2.code, CODE_BORDER);
    }

    #[test]
    fn gslib_hex_and_tet_interior() {
        let mh = Mesh::<3>::unit_cube_hex(2);
        let fph = GslibFindPoints::new(&mh);
        let r = fph.find_point(&[0.31, 0.62, 0.13]);
        assert_eq!(r.code, CODE_INSIDE);
        assert!(r.dist2 < 1e-24);

        let mt = Mesh::<3>::unit_cube_tet(3);
        let fpt = GslibFindPoints::new(&mt);
        let r = fpt.find_point(&[0.21, 0.41, 0.17]);
        assert_eq!(r.code, CODE_INSIDE);
        assert!(r.dist2 < 1e-20);
    }

    #[test]
    fn gslib_batch_matches_single() {
        let m = Mesh::<2>::unit_square_tri(4);
        let fp = GslibFindPoints::new(&m);
        let pts = [[0.1, 0.1], [0.7, 0.2], [0.5, 0.5], [1.5, 1.5]];
        let batch = fp.find_points(&pts);
        assert_eq!(batch.len(), 4);
        for (b, p) in batch.iter().zip(pts.iter()) {
            let single = fp.find_point(p);
            assert_eq!(b.code, single.code);
            assert_eq!(b.elem, single.elem);
        }
        assert_eq!(batch[3].code, CODE_NOT_FOUND);
    }
}
