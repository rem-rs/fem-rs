//! FindPoints — spatial query API for simplex meshes.
//!
//! Combines the BVH spatial index with Newton iteration to map physical
//! points back to reference coordinates within candidate elements.
//!
//! Public API:
//! - [`FindPoints`] — query struct with `locate` / `locate_batch` methods
//! - [`LocatedPoint`] — result type (element id + reference coords + barycentric)
//!
//! Semantics follow MFEM `FindPointsGSLIB`: given a set of physical points,
//! find the containing element and reference coordinates for each point.
//! Points outside the mesh return `None` (caller may use `nearest_node`
//! fallback on the underlying locator).

use nalgebra::DMatrix;

use fem_core::ElemId;

use crate::Mesh;

use super::bvh::Bvh;
use super::newton::{self, NewtonStatus};

/// Result of locating a physical point in the mesh.
#[derive(Debug, Clone)]
pub struct LocatedPoint<const D: usize> {
    /// Element id that contains the point.
    pub elem: ElemId,
    /// Reference (barycentric) coordinates of the point within the element.
    pub xi: [f64; D],
    /// Barycentric coordinates (length `D + 1`).
    pub barycentric: Vec<f64>,
}

/// Options for FindPoints queries.
#[derive(Debug, Clone)]
pub struct FindPointsOptions {
    /// Convergence tolerance for Newton iteration (physical-space residual).
    pub tol: f64,
    /// Maximum Newton iterations per candidate element.
    pub max_iter: usize,
    /// Tolerance for the "inside simplex" check on reference coordinates.
    pub inside_tol: f64,
    /// Number of nearest candidates to try (in order of AABB proximity).
    pub max_candidates: usize,
}

impl Default for FindPointsOptions {
    fn default() -> Self {
        Self {
            tol: 1e-10,
            max_iter: 20,
            inside_tol: 1e-8,
            max_candidates: 8,
        }
    }
}

/// FindPoints query struct.
///
/// Holds a pre-built BVH for efficient spatial queries.
pub struct FindPoints<'a, const D: usize> {
    mesh: &'a Mesh<D>,
    bvh: Bvh<D>,
}

impl<'a, const D: usize> FindPoints<'a, D> {
    /// Build a new FindPoints query struct from a mesh.
    pub fn new(mesh: &'a Mesh<D>) -> Self {
        let bvh = Bvh::new(mesh);
        Self { mesh, bvh }
    }

    /// Locate a single physical point in the mesh.
    ///
    /// Returns `None` if no containing element is found.
    pub fn locate(&self, p: &[f64], opts: &FindPointsOptions) -> Option<LocatedPoint<D>> {
        assert!(p.len() >= D, "point dimension mismatch");
        let x: [f64; D] = std::array::from_fn(|i| p[i]);

        // 1. Get candidate elements from BVH.
        let candidates = self.bvh.locate_candidates(&x, opts.inside_tol);

        // 2. Try candidates in order of proximity.
        let max_cands = candidates.len().min(opts.max_candidates);
        for &e in &candidates[..max_cands] {
            if let Some(lp) = self.try_element(e, &x, opts) {
                return Some(lp);
            }
        }

        // 3. Fallback: try the nearest AABB.
        if candidates.is_empty() {
            let e = self.bvh.nearest_box(&x);
            if let Some(lp) = self.try_element(e, &x, opts) {
                return Some(lp);
            }
        }

        None
    }

    /// Locate a batch of points.
    ///
    /// Returns a vector of `Option<LocatedPoint>` aligned with the input.
    pub fn locate_batch(
        &self,
        points: &[[f64; D]],
        opts: &FindPointsOptions,
    ) -> Vec<Option<LocatedPoint<D>>> {
        points.iter().map(|p| self.locate(p, opts)).collect()
    }

    /// Try to find the reference coordinates of point `x` within element `e`.
    fn try_element(
        &self,
        e: ElemId,
        x: &[f64; D],
        opts: &FindPointsOptions,
    ) -> Option<LocatedPoint<D>> {
        let ns = self.mesh.elem_nodes(e);
        let dim = D;

        // Build the affine Jacobian: J[:,k] = node_{k+1} - node_0
        let x0 = self.mesh.coords_of(ns[0]);
        let mut jac = DMatrix::<f64>::zeros(dim, dim);
        for k in 0..dim {
            let xk = self.mesh.coords_of(ns[k + 1]);
            for i in 0..dim {
                jac[(i, k)] = xk[i] - x0[i];
            }
        }

        // Run Newton iteration.
        let x0_vec: Vec<f64> = x0.to_vec();
        let (xi_vec, result) =
            newton::newton_inverse_simplex::<D>(&x0_vec, &jac, x, opts.tol, opts.max_iter);

        if result.status == NewtonStatus::InvalidJacobian
            || result.status == NewtonStatus::Diverged
        {
            return None;
        }

        // Check if the reference coordinates are inside the simplex.
        if !newton::is_inside_simplex(&xi_vec, opts.inside_tol) {
            return None;
        }

        // Verify the residual is small (for NotConverged cases).
        if !result.is_converged() {
            // Recompute residual
            let mut x_map = vec![0.0; dim];
            for i in 0..dim {
                x_map[i] = x0[i];
                for j in 0..dim {
                    x_map[i] += jac[(i, j)] * xi_vec[j];
                }
            }
            let mut rnorm = 0.0;
            for i in 0..dim {
                let r = x_map[i] - x[i];
                rnorm += r * r;
            }
            rnorm = rnorm.sqrt();
            if rnorm > opts.tol * 100.0 {
                return None;
            }
        }

        let mut xi = [0.0_f64; D];
        for i in 0..dim {
            xi[i] = xi_vec[i];
        }
        let barycentric = newton::barycentric_from_ref(&xi_vec);

        Some(LocatedPoint {
            elem: e,
            xi,
            barycentric,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn findpoints_locates_interior_2d() {
        let m = Mesh::<2>::unit_square_tri(4);
        let fp = FindPoints::new(&m);
        let opts = FindPointsOptions::default();
        let p = [0.37, 0.41];
        let r = fp.locate(&p, &opts).expect("point should be inside mesh");
        assert!((r.elem as usize) < m.n_elems());
        // Verify barycentric sums to 1
        let sum: f64 = r.barycentric.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn findpoints_returns_none_for_outside_2d() {
        let m = Mesh::<2>::unit_square_tri(4);
        let fp = FindPoints::new(&m);
        let opts = FindPointsOptions::default();
        let p = [1.5, -0.2];
        assert!(fp.locate(&p, &opts).is_none());
    }

    #[test]
    fn findpoints_locates_interior_3d() {
        let m = Mesh::<3>::unit_cube_tet(3);
        let fp = FindPoints::new(&m);
        let opts = FindPointsOptions::default();
        let p = [0.21, 0.41, 0.37];
        let r = fp.locate(&p, &opts).expect("point should be inside mesh");
        assert!((r.elem as usize) < m.n_elems());
        let sum: f64 = r.barycentric.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn findpoints_batch_2d() {
        let m = Mesh::<2>::unit_square_tri(4);
        let fp = FindPoints::new(&m);
        let opts = FindPointsOptions::default();
        let points = [[0.1, 0.1], [0.5, 0.5], [0.9, 0.9], [1.5, 1.5]];
        let results = fp.locate_batch(&points, &opts);
        assert_eq!(results.len(), 4);
        assert!(results[0].is_some());
        assert!(results[1].is_some());
        assert!(results[2].is_some());
        assert!(results[3].is_none());
    }

    #[test]
    fn findpoints_boundary_point() {
        let m = Mesh::<2>::unit_square_tri(4);
        let fp = FindPoints::new(&m);
        let opts = FindPointsOptions::default();
        // Point on the boundary (bottom edge)
        let p = [0.5, 0.0];
        let r = fp.locate(&p, &opts).expect("boundary point should be found");
        let sum: f64 = r.barycentric.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}
