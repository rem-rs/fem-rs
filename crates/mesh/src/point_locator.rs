//! Point-location helpers for simplex meshes.
//!
//! Current scope:
//! - 2-D `Tri3` meshes
//! - barycentric point-in-element test
//! - nearest-node fallback query

use fem_core::{ElemId, NodeId};

use crate::{ElementType, SimplexMesh};

/// Result of locating a point in a triangular mesh.
#[derive(Debug, Clone)]
pub struct LocatedPoint2D {
    pub elem: ElemId,
    pub barycentric: [f64; 3],
}

/// Result of locating a point in a tetrahedral mesh.
#[derive(Debug, Clone)]
pub struct LocatedPoint3D {
    pub elem: ElemId,
    pub barycentric: [f64; 4],
}

/// Naive point locator for 2-D `Tri3` meshes.
///
/// Uses per-element axis-aligned bounding boxes to cheaply reject most
/// elements, then a barycentric inclusion test for exact containment.
pub struct TriPointLocator<'a> {
    mesh: &'a SimplexMesh<2>,
    elem_bboxes: Vec<([f64; 2], [f64; 2])>,
}

/// Naive point locator for 3-D `Tet4` meshes.
///
/// Uses per-element axis-aligned bounding boxes and barycentric inclusion.
pub struct TetPointLocator<'a> {
    mesh: &'a SimplexMesh<3>,
    elem_bboxes: Vec<([f64; 3], [f64; 3])>,
}

impl<'a> TriPointLocator<'a> {
    pub fn new(mesh: &'a SimplexMesh<2>) -> Self {
        assert!(
            mesh.elem_type == ElementType::Tri3 || mesh.is_mixed(),
            "TriPointLocator::new: only Tri3 meshes are supported"
        );

        let mut elem_bboxes = Vec::with_capacity(mesh.n_elems());
        for e in 0..mesh.n_elems() as ElemId {
            let ns = mesh.elem_nodes(e);
            assert!(ns.len() >= 3, "TriPointLocator::new: Tri3 element expected");
            let a = mesh.coords_of(ns[0]);
            let b = mesh.coords_of(ns[1]);
            let c = mesh.coords_of(ns[2]);
            let lo = [a[0].min(b[0]).min(c[0]), a[1].min(b[1]).min(c[1])];
            let hi = [a[0].max(b[0]).max(c[0]), a[1].max(b[1]).max(c[1])];
            elem_bboxes.push((lo, hi));
        }

        Self { mesh, elem_bboxes }
    }

    /// Locate a physical point in the mesh.
    ///
    /// Returns `None` if no containing triangle is found within tolerance.
    pub fn locate(&self, p: &[f64], tol: f64) -> Option<LocatedPoint2D> {
        assert!(p.len() >= 2, "TriPointLocator::locate: point must be 2D");
        let x = [p[0], p[1]];
        for e in 0..self.mesh.n_elems() as ElemId {
            let (lo, hi) = self.elem_bboxes[e as usize];
            if x[0] < lo[0] - tol || x[0] > hi[0] + tol || x[1] < lo[1] - tol || x[1] > hi[1] + tol {
                continue;
            }

            let ns = self.mesh.elem_nodes(e);
            let a = self.mesh.coords_of(ns[0]);
            let b = self.mesh.coords_of(ns[1]);
            let c = self.mesh.coords_of(ns[2]);
            if let Some(lmb) = barycentric_tri2([a[0], a[1]], [b[0], b[1]], [c[0], c[1]], x) {
                if lmb[0] >= -tol && lmb[1] >= -tol && lmb[2] >= -tol {
                    return Some(LocatedPoint2D {
                        elem: e,
                        barycentric: lmb,
                    });
                }
            }
        }
        None
    }

    /// Return the nearest source node to point `p`.
    pub fn nearest_node(&self, p: &[f64]) -> NodeId {
        assert!(p.len() >= 2, "TriPointLocator::nearest_node: point must be 2D");
        let mut best = 0_u32;
        let mut best_d2 = f64::INFINITY;
        for n in 0..self.mesh.n_nodes() as NodeId {
            let c = self.mesh.coords_of(n);
            let dx = c[0] - p[0];
            let dy = c[1] - p[1];
            let d2 = dx * dx + dy * dy;
            if d2 < best_d2 {
                best_d2 = d2;
                best = n;
            }
        }
        best
    }
}

impl<'a> TetPointLocator<'a> {
    pub fn new(mesh: &'a SimplexMesh<3>) -> Self {
        assert!(
            mesh.elem_type == ElementType::Tet4 || mesh.is_mixed(),
            "TetPointLocator::new: only Tet4 meshes are supported"
        );

        let mut elem_bboxes = Vec::with_capacity(mesh.n_elems());
        for e in 0..mesh.n_elems() as ElemId {
            let ns = mesh.elem_nodes(e);
            assert!(ns.len() >= 4, "TetPointLocator::new: Tet4 element expected");
            let a = mesh.coords_of(ns[0]);
            let b = mesh.coords_of(ns[1]);
            let c = mesh.coords_of(ns[2]);
            let d = mesh.coords_of(ns[3]);
            let lo = [
                a[0].min(b[0]).min(c[0]).min(d[0]),
                a[1].min(b[1]).min(c[1]).min(d[1]),
                a[2].min(b[2]).min(c[2]).min(d[2]),
            ];
            let hi = [
                a[0].max(b[0]).max(c[0]).max(d[0]),
                a[1].max(b[1]).max(c[1]).max(d[1]),
                a[2].max(b[2]).max(c[2]).max(d[2]),
            ];
            elem_bboxes.push((lo, hi));
        }

        Self { mesh, elem_bboxes }
    }

    /// Locate a physical point in the tetrahedral mesh.
    pub fn locate(&self, p: &[f64], tol: f64) -> Option<LocatedPoint3D> {
        assert!(p.len() >= 3, "TetPointLocator::locate: point must be 3D");
        let x = [p[0], p[1], p[2]];
        for e in 0..self.mesh.n_elems() as ElemId {
            let (lo, hi) = self.elem_bboxes[e as usize];
            if x[0] < lo[0] - tol
                || x[0] > hi[0] + tol
                || x[1] < lo[1] - tol
                || x[1] > hi[1] + tol
                || x[2] < lo[2] - tol
                || x[2] > hi[2] + tol
            {
                continue;
            }

            let ns = self.mesh.elem_nodes(e);
            let a = self.mesh.coords_of(ns[0]);
            let b = self.mesh.coords_of(ns[1]);
            let c = self.mesh.coords_of(ns[2]);
            let d = self.mesh.coords_of(ns[3]);

            if let Some(lmb) = barycentric_tet3(
                [a[0], a[1], a[2]],
                [b[0], b[1], b[2]],
                [c[0], c[1], c[2]],
                [d[0], d[1], d[2]],
                x,
            ) {
                if lmb[0] >= -tol && lmb[1] >= -tol && lmb[2] >= -tol && lmb[3] >= -tol {
                    return Some(LocatedPoint3D {
                        elem: e,
                        barycentric: lmb,
                    });
                }
            }
        }
        None
    }

    /// Return nearest node id to point `p`.
    pub fn nearest_node(&self, p: &[f64]) -> NodeId {
        assert!(p.len() >= 3, "TetPointLocator::nearest_node: point must be 3D");
        let mut best = 0_u32;
        let mut best_d2 = f64::INFINITY;
        for n in 0..self.mesh.n_nodes() as NodeId {
            let c = self.mesh.coords_of(n);
            let dx = c[0] - p[0];
            let dy = c[1] - p[1];
            let dz = c[2] - p[2];
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 < best_d2 {
                best_d2 = d2;
                best = n;
            }
        }
        best
    }
}

fn barycentric_tri2(a: [f64; 2], b: [f64; 2], c: [f64; 2], p: [f64; 2]) -> Option<[f64; 3]> {
    let v0 = [b[0] - a[0], b[1] - a[1]];
    let v1 = [c[0] - a[0], c[1] - a[1]];
    let v2 = [p[0] - a[0], p[1] - a[1]];

    let det = v0[0] * v1[1] - v0[1] * v1[0];
    if det.abs() < 1e-20 {
        return None;
    }

    let inv_det = 1.0 / det;
    let l1 = (v2[0] * v1[1] - v2[1] * v1[0]) * inv_det;
    let l2 = (v0[0] * v2[1] - v0[1] * v2[0]) * inv_det;
    let l0 = 1.0 - l1 - l2;
    Some([l0, l1, l2])
}

fn det3(m: [[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn barycentric_tet3(
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
    d: [f64; 3],
    p: [f64; 3],
) -> Option<[f64; 4]> {
    let m = [
        [b[0] - a[0], c[0] - a[0], d[0] - a[0]],
        [b[1] - a[1], c[1] - a[1], d[1] - a[1]],
        [b[2] - a[2], c[2] - a[2], d[2] - a[2]],
    ];
    let det_m = det3(m);
    if det_m.abs() < 1e-24 {
        return None;
    }

    let r = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
    let m1 = [
        [r[0], m[0][1], m[0][2]],
        [r[1], m[1][1], m[1][2]],
        [r[2], m[2][1], m[2][2]],
    ];
    let m2 = [
        [m[0][0], r[0], m[0][2]],
        [m[1][0], r[1], m[1][2]],
        [m[2][0], r[2], m[2][2]],
    ];
    let m3 = [
        [m[0][0], m[0][1], r[0]],
        [m[1][0], m[1][1], r[1]],
        [m[2][0], m[2][1], r[2]],
    ];

    let l1 = det3(m1) / det_m;
    let l2 = det3(m2) / det_m;
    let l3 = det3(m3) / det_m;
    let l0 = 1.0 - l1 - l2 - l3;
    Some([l0, l1, l2, l3])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locate_point_in_unit_square_tri_mesh() {
        let m = SimplexMesh::<2>::unit_square_tri(4);
        let loc = TriPointLocator::new(&m);
        let p = [0.37, 0.41];
        let r = loc.locate(&p, 1e-12).expect("point should be inside mesh");
        let l = r.barycentric;
        assert!((l[0] + l[1] + l[2] - 1.0).abs() < 1e-12);
        assert!(l[0] >= -1e-12 && l[1] >= -1e-12 && l[2] >= -1e-12);
    }

    #[test]
    fn locate_returns_none_for_outside_point() {
        let m = SimplexMesh::<2>::unit_square_tri(4);
        let loc = TriPointLocator::new(&m);
        let p = [1.5, -0.2];
        assert!(loc.locate(&p, 1e-12).is_none());
    }

    #[test]
    fn nearest_node_returns_valid_id() {
        let m = SimplexMesh::<2>::unit_square_tri(4);
        let loc = TriPointLocator::new(&m);
        let nid = loc.nearest_node(&[0.99, 0.99]);
        assert!((nid as usize) < m.n_nodes());
    }

    #[test]
    fn locate_point_in_unit_cube_tet_mesh() {
        let m = SimplexMesh::<3>::unit_cube_tet(3);
        let loc = TetPointLocator::new(&m);
        let p = [0.21, 0.41, 0.37];
        let r = loc.locate(&p, 1e-12).expect("point should be inside mesh");
        let l = r.barycentric;
        assert!((l[0] + l[1] + l[2] + l[3] - 1.0).abs() < 1e-12);
        assert!(l[0] >= -1e-12 && l[1] >= -1e-12 && l[2] >= -1e-12 && l[3] >= -1e-12);
    }

    #[test]
    fn locate_returns_none_for_outside_point_3d() {
        let m = SimplexMesh::<3>::unit_cube_tet(3);
        let loc = TetPointLocator::new(&m);
        assert!(loc.locate(&[-0.1, 0.2, 1.5], 1e-12).is_none());
    }
}
