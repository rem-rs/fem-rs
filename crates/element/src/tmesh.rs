//! 2-D T-spline data structures and blending functions (Phase 3.3).
//!
//! T-splines generalise tensor-product B-splines by permitting T-junctions
//! in the control mesh, enabling local refinement without global knot-line
//! propagation.
//!
//! # Overview
//!
//! | Type        | Description                                   |
//! |-------------|-----------------------------------------------|
//! | [`TVertex`] | T-mesh control vertex with parametric indices  |
//! | [`TCell`]   | Axis-aligned rectangular element               |
//! | [`TMesh2D`] | Full 2-D T-mesh with blending-function queries |
//!
//! # T-spline evaluation flow
//!
//! 1. For each element (cell), find active vertices whose local knot vectors
//!    overlap the cell's parametric domain.
//! 2. For each active vertex, evaluate its T-spline blending function
//!    `R_a(u,v) = N_a(u) * M_a(v)` where `N_a` and `M_a` are 1-D B-spline
//!    basis functions defined on the vertex's local knot vectors.
//! 3. Assemble element matrices directly from those evaluations.
//!
//! Each local knot vector has `p+2` values, defining a single B-spline basis
//! function of degree `p`.  Evaluation extends the knot vector to a properly
//! clamped form that [`BSplineBasis1D`] can handle.
//!
//! # References
//! * Sederberg et al., "T-splines and T-NURCCs", ACM Trans. Graph. 22(3), 2003.
//! * Bazilevs et al., "Isogeometric analysis using T-splines", CMAME 2010.

use crate::nurbs::{BSplineBasis1D, KnotVector};

/// Build a properly clamped knot vector from a `p+2`-element local knot vector
/// by repeating the first and last values `p+1` times each.
///
/// Result length: `3p+2`.  Number of basis functions: `2p+1`.
/// The vertex's basis function is `N_{p, p}` (index `p`), whose support spans
/// from `knots[p] = local[0]` to `knots[2p+1] = local[p+1]`.
fn extend_local_knots(local: &[f64], p: usize) -> Vec<f64> {
    let mut ext = Vec::with_capacity(3 * p + 2);
    for _ in 0..=p {
        ext.push(local[0]);
    }
    for i in 1..=p {
        ext.push(local[i]);
    }
    for _ in 0..=p {
        ext.push(local[p + 1]);
    }
    ext
}

// ─── Data structures ────────────────────────────────────────────────────────

/// A 2-D T-mesh control vertex.
#[derive(Debug, Clone)]
pub struct TVertex {
    /// Physical x-coordinate of the control point.
    pub x: f64,
    /// Physical y-coordinate of the control point.
    pub y: f64,
    /// NURBS weight (1.0 for a polynomial B-spline).
    pub weight: f64,
    /// Knot index in the u-direction of the T-mesh topology.
    pub iu: usize,
    /// Knot index in the v-direction of the T-mesh topology.
    pub iv: usize,
}

/// A T-mesh element (cell) — axis-aligned rectangle in the parametric domain.
#[derive(Debug, Clone)]
pub struct TCell {
    /// Local knot-span indices (min, max) in u.
    pub iu_min: usize,
    pub iu_max: usize,
    /// Local knot-span indices (min, max) in v.
    pub iv_min: usize,
    pub iv_max: usize,
}

/// 2-D T-mesh for T-spline finite element analysis.
#[derive(Debug, Clone)]
pub struct TMesh2D {
    /// Control vertices in row-major order: index = `iv * nu + iu`.
    ///
    /// T-junction positions have no entry — only actual control points
    /// are stored.  The `vertex_exists` method checks whether a vertex
    /// exists at a given index pair.
    pub vertices: Vec<TVertex>,
    /// Number of unique knot indices in the u-direction.
    pub nu: usize,
    /// Number of unique knot indices in the v-direction.
    pub nv: usize,

    /// Unique knot values in the u-direction (sorted, may have gaps
    /// at T-junction positions).
    pub unique_u: Vec<f64>,
    /// Unique knot values in the v-direction (sorted, may have gaps).
    pub unique_v: Vec<f64>,

    /// All elements (cells) in the T-mesh.
    pub cells: Vec<TCell>,

    /// Polynomial degree in the u-direction.
    pub pu: usize,
    /// Polynomial degree in the v-direction.
    pub pv: usize,
}

// ─── TMesh2D methods ─────────────────────────────────────────────────────────

impl TMesh2D {
    /// Check whether a control vertex exists at T-mesh index `(iu, iv)`.
    ///
    /// T-junction positions return `false`.
    pub fn vertex_exists(&self, iu: usize, iv: usize) -> bool {
        self.vertices
            .iter()
            .any(|v| v.iu == iu && v.iv == iv)
    }

    /// Get the global vertex index for `(iu, iv)`, or `None` if no vertex exists.
    pub fn vertex_index(&self, iu: usize, iv: usize) -> Option<usize> {
        self.vertices
            .iter()
            .position(|v| v.iu == iu && v.iv == iv)
    }

    /// Build the **local u-knot vector** for the vertex at `(iu, iv)`.
    ///
    /// Returns `pu + 2` knot values: a subsequence of `unique_u` centred on `iu`
    /// with left-boundary padding.
    pub fn local_knot_vector_u(&self, iu: usize, _iv: usize) -> Vec<f64> {
        let p = self.pu;
        let n_unique = self.unique_u.len();
        let mut knots = Vec::with_capacity(p + 2);
        let iu_s = iu as isize;
        let nu_s = n_unique as isize - 1;
        for ji in 0..=p + 1 {
            let idx = (iu_s - p as isize + ji as isize).clamp(0, nu_s) as usize;
            knots.push(self.unique_u[idx]);
        }
        knots
    }

    /// Build the **local v-knot vector** for the vertex at `(iu, iv)`.
    ///
    /// Returns `pv + 2` knot values.
    pub fn local_knot_vector_v(&self, _iu: usize, iv: usize) -> Vec<f64> {
        let p = self.pv;
        let n_unique = self.unique_v.len();
        let mut knots = Vec::with_capacity(p + 2);
        let iv_s = iv as isize;
        let nv_s = n_unique as isize - 1;
        for ji in 0..=p + 1 {
            let idx = (iv_s - p as isize + ji as isize).clamp(0, nv_s) as usize;
            knots.push(self.unique_v[idx]);
        }
        knots
    }

    /// Find all **active vertices** whose blending functions overlap `cell`.
    ///
    /// A vertex `(iu, iv)` is active on the cell if its T-spline blending
    /// function's support overlaps the cell's parametric domain.
    /// The support of a vertex is `(local_u[0], local_u[pu+1])`
    /// times `(local_v[0], local_v[pv+1])`.
    ///
    /// Returns global vertex indices (indices into `self.vertices`).
    pub fn find_active_vertices(&self, cell: &TCell) -> Vec<usize> {
        let u0 = self.unique_u[cell.iu_min];
        let u1 = self.unique_u[cell.iu_max];
        let v0 = self.unique_v[cell.iv_min];
        let v1 = self.unique_v[cell.iv_max];

        self.vertices
            .iter()
            .enumerate()
            .filter(|(_, vtx)| {
                let kv_u = self.local_knot_vector_u(vtx.iu, vtx.iv);
                let kv_v = self.local_knot_vector_v(vtx.iu, vtx.iv);

                // Support = (first_knot, last_knot) for the p+2 local knots.
                let su0 = kv_u[0];
                let su1 = kv_u[self.pu + 1];
                let sv0 = kv_v[0];
                let sv1 = kv_v[self.pv + 1];

                // Overlap test: does the support intersect the cell domain?
                su0 < u1 && su1 > u0 && sv0 < v1 && sv1 > v0
            })
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Evaluate the T-spline blending functions for a specific cell at
    /// a parametric point.
    ///
    /// `(xi, eta)` is in the reference element `[0,1]²`.
    ///
    /// # Outputs
    ///
    /// * `phi` — `n_active` values of the blending functions at `(u,v)`.
    /// * `grads_u` — `dR/du` (derivative w.r.t. parametric u).
    /// * `grads_v` — `dR/dv` (derivative w.r.t. parametric v).
    pub fn eval_cell(
        &self,
        cell: &TCell,
        xi: f64,
        eta: f64,
        phi: &mut [f64],
        grads_u: &mut [f64],
        grads_v: &mut [f64],
    ) {
        let u0 = self.unique_u[cell.iu_min];
        let u1 = self.unique_u[cell.iu_max];
        let v0 = self.unique_v[cell.iv_min];
        let v1 = self.unique_v[cell.iv_max];
        let u = u0 + xi * (u1 - u0);
        let v = v0 + eta * (v1 - v0);

        let active = self.find_active_vertices(cell);
        assert_eq!(phi.len(), active.len());
        assert_eq!(grads_u.len(), active.len());
        assert_eq!(grads_v.len(), active.len());

        // Pre-extend local knot vectors to clamped form for all active vertices.
        let ext_vectors: Vec<(Vec<f64>, Vec<f64>)> = active
            .iter()
            .map(|&vidx| {
                let vtx = &self.vertices[vidx];
                let kv_u = self.local_knot_vector_u(vtx.iu, vtx.iv);
                let kv_v = self.local_knot_vector_v(vtx.iu, vtx.iv);
                (extend_local_knots(&kv_u, self.pu), extend_local_knots(&kv_v, self.pv))
            })
            .collect();

        let local_iu = self.pu;
        let local_iv = self.pv;

        for (a, (ref ext_u, ref ext_v)) in ext_vectors.iter().enumerate() {
            let basis_u = BSplineBasis1D::new(KnotVector::new(ext_u.clone(), self.pu));
            let basis_v = BSplineBasis1D::new(KnotVector::new(ext_v.clone(), self.pv));

            let (nu_b, dnu_b) = basis_u.eval_with_ders(u);
            let (nv_b, dnv_b) = basis_v.eval_with_ders(v);

            phi[a] = nu_b[local_iu] * nv_b[local_iv];
            grads_u[a] = dnu_b[local_iu] * nv_b[local_iv];
            grads_v[a] = dnv_b[local_iv] * nu_b[local_iu];
        }
    }
}

// ─── Factory functions ──────────────────────────────────────────────────────

/// Create a uniform tensor-product T-spline mesh (equivalent to a B-spline).
///
/// `nu × nv` vertices, degrees `(pu, pv)`, on the unit square `[0,1]²`.
/// The knot values are uniformly spaced: `unique_u` has `nu` entries from
/// 0 to 1, similarly for `unique_v`.
///
/// The resulting T-mesh has no T-junctions — it is a regular tensor-product
/// grid.  T-splines on this mesh reproduce standard B-spline basis functions
/// exactly.
pub fn uniform_tspline_2d(
    nu: usize,
    nv: usize,
    pu: usize,
    pv: usize,
) -> TMesh2D {
    assert!(nu >= pu + 1, "nu must be >= pu + 1");
    assert!(nv >= pv + 1, "nv must be >= pv + 1");

    let unique_u: Vec<f64> = (0..nu)
        .map(|i| i as f64 / (nu - 1).max(1) as f64)
        .collect();
    let unique_v: Vec<f64> = (0..nv)
        .map(|j| j as f64 / (nv - 1).max(1) as f64)
        .collect();

    // Vertices in row-major order (iv outer, iu inner).
    let mut vertices = Vec::with_capacity(nu * nv);
    for iv in 0..nv {
        for iu in 0..nu {
            vertices.push(TVertex {
                x: unique_u[iu],
                y: unique_v[iv],
                weight: 1.0,
                iu,
                iv,
            });
        }
    }

    // Cells: (nu-1) × (nv-1) axis-aligned rectangles.
    let mut cells = Vec::with_capacity((nu - 1) * (nv - 1));
    for iv in 0..nv - 1 {
        for iu in 0..nu - 1 {
            cells.push(TCell {
                iu_min: iu,
                iu_max: iu + 1,
                iv_min: iv,
                iv_max: iv + 1,
            });
        }
    }

    TMesh2D {
        vertices,
        nu,
        nv,
        unique_u,
        unique_v,
        cells,
        pu,
        pv,
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_tspline_2d_creates_correct_size() {
        let nu = 4;
        let nv = 3;
        let tmesh = uniform_tspline_2d(nu, nv, 2, 1);
        assert_eq!(tmesh.vertices.len(), nu * nv);
        assert_eq!(tmesh.nu, nu);
        assert_eq!(tmesh.nv, nv);
        assert_eq!(tmesh.cells.len(), (nu - 1) * (nv - 1));
    }

    #[test]
    fn vertex_exists_on_uniform_grid() {
        let tmesh = uniform_tspline_2d(4, 4, 2, 2);
        assert!(tmesh.vertex_exists(0, 0));
        assert!(tmesh.vertex_exists(3, 3));
        assert!(tmesh.vertex_exists(1, 2));
    }

    #[test]
    fn local_knot_vector_length() {
        let tmesh = uniform_tspline_2d(5, 5, 2, 2);
        let kv = tmesh.local_knot_vector_u(2, 2);
        assert_eq!(kv.len(), 4); // pu + 2
    }

    #[test]
    fn local_knot_vector_u_for_uniform_mesh() {
        let tmesh = uniform_tspline_2d(5, 5, 2, 2);
        // Vertex iu=2: local window [i-p, i+1] = [0, 3]
        // unique_u = [0.0, 0.25, 0.5, 0.75, 1.0]
        // local = [0.0, 0.25, 0.5, 0.75]
        let kv = tmesh.local_knot_vector_u(2, 2);
        assert_eq!(kv.len(), 4);
        assert!((kv[0] - 0.0).abs() < 1e-14);
        assert!((kv[1] - 0.25).abs() < 1e-14);
        assert!((kv[2] - 0.5).abs() < 1e-14);
        assert!((kv[3] - 0.75).abs() < 1e-14);
    }

    #[test]
    fn local_knot_vector_left_boundary() {
        let tmesh = uniform_tspline_2d(4, 4, 1, 1);
        // Vertex iu=0: local window [i-1, i+1] = [-1, 1] clamped to [0, 1]
        // unique_u = [0.0, 1/3, 2/3, 1.0]
        // local = [0.0, 0.0, 1/3]
        let kv = tmesh.local_knot_vector_u(0, 0);
        assert_eq!(kv.len(), 3);
        assert!((kv[0] - 0.0).abs() < 1e-14);
        assert!((kv[1] - 0.0).abs() < 1e-14);
        assert!((kv[2] - (1.0 / 3.0)).abs() < 1e-14);
    }

    #[test]
    fn local_knot_vector_right_boundary() {
        let tmesh = uniform_tspline_2d(4, 4, 1, 1);
        // Vertex iu=3: local window [i-1, i+1] = [2, 4] clamped to [2, 3]
        // local = [2/3, 1.0, 1.0]
        let kv = tmesh.local_knot_vector_u(3, 0);
        assert_eq!(kv.len(), 3);
        assert!((kv[0] - (2.0 / 3.0)).abs() < 1e-14);
        assert!((kv[1] - 1.0).abs() < 1e-14);
        assert!((kv[2] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn find_active_vertices_on_uniform_grid() {
        let tmesh = uniform_tspline_2d(4, 4, 1, 1);
        // For degree-1, each cell has exactly 4 active vertices.
        let cell = &tmesh.cells[0]; // first cell
        let active = tmesh.find_active_vertices(cell);
        assert_eq!(active.len(), 4);
    }

    #[test]
    fn eval_cell_partition_of_unity() {
        let tmesh = uniform_tspline_2d(4, 4, 1, 1);
        let cell = &tmesh.cells[0];
        let n_active = tmesh.find_active_vertices(cell).len();
        let n_qpts = 3;
        let (qpts, _qwts) = crate::quadrature::gauss_legendre_01(n_qpts);

        for &xi in &qpts {
            for &eta in &qpts {
                let mut phi = vec![0.0; n_active];
                let mut gu = vec![0.0; n_active];
                let mut gv = vec![0.0; n_active];
                tmesh.eval_cell(cell, xi, eta, &mut phi, &mut gu, &mut gv);
                let sum: f64 = phi.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-12,
                    "partition of unity failed at (xi={xi}, eta={eta}): sum={sum}"
                );
            }
        }
    }

    #[test]
    fn cell_active_dofs_vary_by_cell() {
        let tmesh = uniform_tspline_2d(4, 4, 1, 1);
        let active_0 = tmesh.find_active_vertices(&tmesh.cells[0]);
        let active_last = tmesh.find_active_vertices(&tmesh.cells[tmesh.cells.len() - 1]);
        assert_ne!(active_0, active_last);
    }
}
