//! Point-location helpers for 2-D/3-D meshes.
//!
//! Current scope:
//! - 2-D `Tri3` and `Quad4` meshes (barycentric / bilinear inclusion test)
//! - 3-D `Tet4` meshes
//! - nearest-node fallback query

use fem_core::{ElemId, NodeId};

use crate::{ElementType, Mesh};

/// Result of locating a point in a 2-D mesh element.
#[derive(Debug, Clone)]
pub struct LocatedPoint2D {
    pub elem: ElemId,
    /// Shape-function weights of the containing element's nodes, in the order
    /// of [`MeshTopology::element_nodes`](crate::topology::MeshTopology::element_nodes):
    /// 3 barycentric coordinates for a triangle, 4 bilinear weights for a
    /// quadrilateral.
    pub barycentric: Vec<f64>,
}

/// Result of locating a point in a tetrahedral mesh.
#[derive(Debug, Clone)]
pub struct LocatedPoint3D {
    pub elem: ElemId,
    pub barycentric: [f64; 4],
}

/// Naive point locator for 2-D meshes of `Tri3` and/or `Quad4` elements.
///
/// Uses per-element axis-aligned bounding boxes to cheaply reject most
/// elements, then an exact inclusion test for containment: barycentric for a
/// triangle, the inverted bilinear map for a quadrilateral.
///
/// The 3-D counterpart is [`TetPointLocator`].  The name is historical (the
/// locator was simplex-only); D64 extended it to quadrilaterals so the LOR
/// H¹ prolongation works on the default quad meshes.
pub struct TriPointLocator<'a> {
    mesh: &'a Mesh<2>,
    elem_bboxes: Vec<([f64; 2], [f64; 2])>,
}

/// Naive point locator for 3-D `Tet4` meshes.
///
/// Uses per-element axis-aligned bounding boxes and barycentric inclusion.
pub struct TetPointLocator<'a> {
    mesh: &'a Mesh<3>,
    elem_bboxes: Vec<([f64; 3], [f64; 3])>,
}

impl<'a> TriPointLocator<'a> {
    pub fn new(mesh: &'a Mesh<2>) -> Self {
        assert!(
            mesh.is_mixed()
                || matches!(mesh.elem_type, ElementType::Tri3 | ElementType::Quad4),
            "TriPointLocator::new: only Tri3/Quad4 meshes are supported (got {:?})",
            mesh.elem_type
        );

        let mut elem_bboxes = Vec::with_capacity(mesh.n_elems());
        for e in 0..mesh.n_elems() as ElemId {
            let ns = mesh.elem_nodes(e);
            assert!(
                ns.len() == 3 || ns.len() == 4,
                "TriPointLocator::new: Tri3/Quad4 element expected (element {e} has {} nodes)",
                ns.len()
            );
            let c0 = mesh.coords_of(ns[0]);
            let mut lo = [c0[0], c0[1]];
            let mut hi = [c0[0], c0[1]];
            for &n in &ns[1..] {
                let c = mesh.coords_of(n);
                lo[0] = lo[0].min(c[0]);
                lo[1] = lo[1].min(c[1]);
                hi[0] = hi[0].max(c[0]);
                hi[1] = hi[1].max(c[1]);
            }
            elem_bboxes.push((lo, hi));
        }

        Self { mesh, elem_bboxes }
    }

    /// Locate a physical point in the mesh.
    ///
    /// Returns `None` if no containing element is found within tolerance.
    /// Triangles report barycentric coordinates, quadrilaterals bilinear
    /// weights (`barycentric.len() == elem_nodes(elem).len()`).
    pub fn locate(&self, p: &[f64], tol: f64) -> Option<LocatedPoint2D> {
        assert!(p.len() >= 2, "TriPointLocator::locate: point must be 2D");
        let x = [p[0], p[1]];
        for e in 0..self.mesh.n_elems() as ElemId {
            let (lo, hi) = self.elem_bboxes[e as usize];
            if x[0] < lo[0] - tol || x[0] > hi[0] + tol || x[1] < lo[1] - tol || x[1] > hi[1] + tol {
                continue;
            }

            let ns = self.mesh.elem_nodes(e);
            let lmb = match ns.len() {
                3 => {
                    let a = self.mesh.coords_of(ns[0]);
                    let b = self.mesh.coords_of(ns[1]);
                    let c = self.mesh.coords_of(ns[2]);
                    barycentric_tri2([a[0], a[1]], [b[0], b[1]], [c[0], c[1]], x)
                        .map(|l| l.to_vec())
                }
                4 => {
                    // CCW from the reference corner (0,0): (0,0),(1,0),(1,1),(0,1).
                    let mut v = [[0.0_f64; 2]; 4];
                    for (k, &n) in ns.iter().enumerate() {
                        let c = self.mesh.coords_of(n);
                        v[k] = [c[0], c[1]];
                    }
                    quad_bilinear_coords(&v, x).map(|(xi, eta)| {
                        vec![
                            (1.0 - xi) * (1.0 - eta),
                            xi * (1.0 - eta),
                            xi * eta,
                            (1.0 - xi) * eta,
                        ]
                    })
                }
                _ => None,
            };
            if let Some(lmb) = lmb {
                if lmb.iter().all(|&l| l >= -tol) {
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
    pub fn new(mesh: &'a Mesh<3>) -> Self {
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

/// Invert the bilinear map of a quadrilateral.
///
/// The four physical vertices `v0..v3` correspond to the reference corners
/// `(0,0)`, `(1,0)`, `(1,1)`, `(0,1)` (the `Mesh` counter-clockwise order), so
/// the map is
///
/// ```text
/// x(ξ, η) = v0 + ξ·b + η·c + ξη·d
/// b = v1 − v0,  c = v3 − v0,  d = v0 − v1 + v2 − v3
/// ```
///
/// Cross-multiplying `p = x − v0` by `c + ξ·d` eliminates `η` and leaves the
/// quadratic
///
/// ```text
/// [b,d]·ξ² + ([b,c] − [p,d])·ξ − [p,c] = 0          ([·,·] = 2-D cross product)
/// ```
///
/// which degenerates to a linear equation for a parallelogram (`d = 0`, hence
/// for every axis-aligned mesh).  `η` then follows exactly from
/// `p − ξ·b = η·(c + ξ·d)`.  MFEM's
/// `Quadrilateral::SetInverseTransformation` solves the same quadratic.
///
/// Returns `None` for a degenerate (zero-area) element or when no real solution
/// exists; the caller applies the containment tolerance to the resulting
/// weights.
fn quad_bilinear_coords(v: &[[f64; 2]; 4], p: [f64; 2]) -> Option<(f64, f64)> {
    let cross = |u: [f64; 2], w: [f64; 2]| u[0] * w[1] - u[1] * w[0];
    let b = [v[1][0] - v[0][0], v[1][1] - v[0][1]];
    let c = [v[3][0] - v[0][0], v[3][1] - v[0][1]];
    let d = [
        v[0][0] - v[1][0] + v[2][0] - v[3][0],
        v[0][1] - v[1][1] + v[2][1] - v[3][1],
    ];
    let q = [p[0] - v[0][0], p[1] - v[0][1]];

    let (bc, bd, qd, qc) = (cross(b, c), cross(b, d), cross(q, d), cross(q, c));
    // `a2 ξ² + a1 ξ + a0 = 0`.
    let (a2, a1, a0) = (bd, bc - qd, -qc);

    // Scale-relative degeneracy test: `d` is exactly zero on a parallelogram.
    let scale = bc.abs() + bd.abs() + qd.abs() + qc.abs();
    let mut roots = [f64::NAN; 2];
    let n_roots;
    if a2.abs() <= 1e-12 * scale.max(f64::MIN_POSITIVE) {
        if a1 == 0.0 {
            return None;
        }
        roots[0] = -a0 / a1;
        n_roots = 1;
    } else {
        let disc = a1 * a1 - 4.0 * a2 * a0;
        if disc < 0.0 {
            return None;
        }
        let s = disc.sqrt();
        // Numerically stable form: the two roots have product a0/a2.
        let r1 = if a1 >= 0.0 { (-a1 - s) / (2.0 * a2) } else { (-a1 + s) / (2.0 * a2) };
        let r2 = if r1 == 0.0 { r1 } else { a0 / (a2 * r1) };
        roots[0] = r1;
        roots[1] = r2;
        n_roots = 2;
    }

    // Two roots are possible only for a genuinely warped quad; return the one
    // whose smallest weight is largest (the caller applies the containment
    // tolerance).
    let mut best: Option<(f64, f64)> = None;
    let mut best_violation = f64::NEG_INFINITY;
    for &xi in roots.iter().take(n_roots) {
        let e = [c[0] + xi * d[0], c[1] + xi * d[1]];
        let e2 = e[0] * e[0] + e[1] * e[1];
        if e2 <= 0.0 {
            continue;
        }
        let r = [q[0] - xi * b[0], q[1] - xi * b[1]];
        let eta = (r[0] * e[0] + r[1] * e[1]) / e2;
        let w = [
            (1.0 - xi) * (1.0 - eta),
            xi * (1.0 - eta),
            xi * eta,
            (1.0 - xi) * eta,
        ];
        let violation = w.iter().fold(f64::INFINITY, |acc, &l| acc.min(l));
        if violation > best_violation {
            best_violation = violation;
            best = Some((xi, eta));
        }
    }
    best
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
        let m = Mesh::<2>::unit_square_tri(4);
        let loc = TriPointLocator::new(&m);
        let p = [0.37, 0.41];
        let r = loc.locate(&p, 1e-12).expect("point should be inside mesh");
        let l = r.barycentric;
        assert!((l[0] + l[1] + l[2] - 1.0).abs() < 1e-12);
        assert!(l[0] >= -1e-12 && l[1] >= -1e-12 && l[2] >= -1e-12);
    }

    /// D64: the locator now also covers `Quad4` meshes with the bilinear
    /// weights, which reproduce a bilinear field exactly.
    #[test]
    fn locate_point_in_unit_square_quad_mesh() {
        let m = Mesh::<2>::make_cartesian_2d(4, 4, 1.0, 1.0);
        let loc = TriPointLocator::new(&m);
        let p = [0.37, 0.41];
        let r = loc.locate(&p, 1e-12).expect("point should be inside mesh");
        let l = r.barycentric;
        assert_eq!(l.len(), 4, "a quad element has four weights");
        assert!((l.iter().sum::<f64>() - 1.0).abs() < 1e-12, "weights {l:?}");
        assert!(l.iter().all(|&w| w >= -1e-12), "weights {l:?}");

        // The weights reproduce the point from the element's nodes ...
        let ns = m.elem_nodes(r.elem);
        let mut x = [0.0_f64; 2];
        for (k, &n) in ns.iter().enumerate() {
            let c = m.coords_of(n);
            x[0] += l[k] * c[0];
            x[1] += l[k] * c[1];
        }
        assert!((x[0] - p[0]).abs() < 1e-12 && (x[1] - p[1]).abs() < 1e-12, "{x:?}");

        // ... and reproduce a bilinear field exactly (the Q1 space contains it).
        let f = |x: f64, y: f64| 1.0 + 2.0 * x - 3.0 * y + 4.0 * x * y;
        let v: f64 = l
            .iter()
            .zip(ns.iter())
            .map(|(&w, &n)| {
                let c = m.coords_of(n);
                w * f(c[0], c[1])
            })
            .sum();
        assert!((v - f(p[0], p[1])).abs() < 1e-12, "bilinear reproduction {v}");
    }

    /// A warped quadrilateral (the general `d != 0` quadratic branch of the
    /// bilinear inverse) with an off-centre query point.
    #[test]
    fn locate_point_in_warped_quad() {
        let quad = [[0.0, 0.0], [2.0, 0.1], [1.9, 1.3], [0.2, 0.9]];
        // Reference point (ξ, η) → physical point through the bilinear map.
        let (xi, eta) = (0.31, 0.62);
        let phi = [
            (1.0 - xi) * (1.0 - eta),
            xi * (1.0 - eta),
            xi * eta,
            (1.0 - xi) * eta,
        ];
        let p: [f64; 2] = std::array::from_fn(|d| {
            (0..4).map(|k| phi[k] * quad[k][d]).sum()
        });
        let (xi_h, eta_h) = quad_bilinear_coords(&quad, p).expect("invertible quad");
        assert!((xi_h - xi).abs() < 1e-12, "ξ {xi_h} vs {xi}");
        assert!((eta_h - eta).abs() < 1e-12, "η {eta_h} vs {eta}");
    }

    /// Outside the mesh (and outside the tolerance) nothing is located.
    #[test]
    fn locate_returns_none_for_outside_point_quad() {
        let m = Mesh::<2>::make_cartesian_2d(4, 4, 1.0, 1.0);
        let loc = TriPointLocator::new(&m);
        assert!(loc.locate(&[1.5, -0.2], 1e-12).is_none());
    }

    #[test]
    fn locate_returns_none_for_outside_point() {
        let m = Mesh::<2>::unit_square_tri(4);
        let loc = TriPointLocator::new(&m);
        let p = [1.5, -0.2];
        assert!(loc.locate(&p, 1e-12).is_none());
    }

    #[test]
    fn nearest_node_returns_valid_id() {
        let m = Mesh::<2>::unit_square_tri(4);
        let loc = TriPointLocator::new(&m);
        let nid = loc.nearest_node(&[0.99, 0.99]);
        assert!((nid as usize) < m.n_nodes());
    }

    #[test]
    fn locate_point_in_unit_cube_tet_mesh() {
        let m = Mesh::<3>::unit_cube_tet(3);
        let loc = TetPointLocator::new(&m);
        let p = [0.21, 0.41, 0.37];
        let r = loc.locate(&p, 1e-12).expect("point should be inside mesh");
        let l = r.barycentric;
        assert!((l[0] + l[1] + l[2] + l[3] - 1.0).abs() < 1e-12);
        assert!(l[0] >= -1e-12 && l[1] >= -1e-12 && l[2] >= -1e-12 && l[3] >= -1e-12);
    }

    #[test]
    fn locate_returns_none_for_outside_point_3d() {
        let m = Mesh::<3>::unit_cube_tet(3);
        let loc = TetPointLocator::new(&m);
        assert!(loc.locate(&[-0.1, 0.2, 1.5], 1e-12).is_none());
    }
}
