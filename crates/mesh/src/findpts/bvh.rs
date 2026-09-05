//! Bounding Volume Hierarchy (BVH) for spatial queries on simplex meshes.
//!
//! Self-contained implementation — no external crates.
//!
//! Provides:
//! - [`Bvh`] — AABB tree over element bounding boxes
//! - [`Bvh::locate_candidates`] → candidate element ids near a point
//! - [`Bvh::nearest_box`] → nearest AABB by center distance (fallback search)

use fem_core::ElemId;
use crate::Mesh;

/// Axis-aligned bounding box in `D` dimensions.
#[derive(Debug, Clone)]
pub struct Aabb<const D: usize> {
    pub lo: [f64; D],
    pub hi: [f64; D],
}

impl<const D: usize> Aabb<D> {
    pub fn new(lo: [f64; D], hi: [f64; D]) -> Self {
        Self { lo, hi }
    }

    /// Centre point.
    pub fn center(&self) -> [f64; D] {
        let mut c = [0.0_f64; D];
        for i in 0..D {
            c[i] = 0.5 * (self.lo[i] + self.hi[i]);
        }
        c
    }

    /// Squared distance from point `p` to the nearest point on the AABB.
    pub fn dist2(&self, p: &[f64]) -> f64 {
        let mut d2 = 0.0_f64;
        for i in 0..D {
            let di = if p[i] < self.lo[i] {
                self.lo[i] - p[i]
            } else if p[i] > self.hi[i] {
                p[i] - self.hi[i]
            } else {
                0.0
            };
            d2 += di * di;
        }
        d2
    }

    /// Squared distance between the AABB center and point `p`.
    pub fn center_dist2(&self, p: &[f64]) -> f64 {
        let c = self.center();
        let mut d2 = 0.0_f64;
        for i in 0..D {
            let di = c[i] - p[i];
            d2 += di * di;
        }
        d2
    }

    /// Build the union of two AABBs.
    pub fn merge(a: &Self, b: &Self) -> Self {
        let mut lo = [0.0_f64; D];
        let mut hi = [0.0_f64; D];
        for i in 0..D {
            lo[i] = a.lo[i].min(b.lo[i]);
            hi[i] = a.hi[i].max(b.hi[i]);
        }
        Self { lo, hi }
    }
}

/// BVH node — either an inner node or a leaf.
#[derive(Debug, Clone)]
enum BvhNode<const D: usize> {
    Inner {
        bbox: Aabb<D>,
        left: Box<BvhNode<D>>,
        right: Box<BvhNode<D>>,
    },
    Leaf {
        bbox: Aabb<D>,
        elem: ElemId,
    },
}

/// Bounding Volume Hierarchy over mesh element AABBs.
///
/// Construction is O(n log n) via median-split on the longest axis.
/// Point queries return candidate element ids sorted by proximity to the
/// query point (useful as a starting set for Newton iteration).
pub struct Bvh<const D: usize> {
    root: BvhNode<D>,
}

impl<const D: usize> Bvh<D> {
    /// Build a BVH from a mesh.
    pub fn new(mesh: &Mesh<D>) -> Self {
        let mut elems: Vec<ElemId> = (0..mesh.n_elems() as ElemId).collect();
        let root = Self::build_recursive(mesh, &mut elems);
        Self { root }
    }

    /// Build the BVH recursively.
    ///
    /// Splits the element list by the median along the longest axis.
    fn build_recursive(mesh: &Mesh<D>, elems: &mut [ElemId]) -> BvhNode<D> {
        if elems.len() == 1 {
            let e = elems[0];
            return BvhNode::Leaf {
                bbox: Self::elem_bbox(mesh, e),
                elem: e,
            };
        }

        // Compute the bounding box of all elements in this node.
        let node_bbox = {
            let mut b = Self::elem_bbox(mesh, elems[0]);
            for &e in elems.iter().skip(1) {
                b = Aabb::merge(&b, &Self::elem_bbox(mesh, e));
            }
            b
        };

        // Split along the longest axis by the median coordinate.
        let axis = {
            let mut axis = 0;
            let mut max_len = node_bbox.hi[0] - node_bbox.lo[0];
            for i in 1..D {
                let len = node_bbox.hi[i] - node_bbox.lo[i];
                if len > max_len {
                    max_len = len;
                    axis = i;
                }
            }
            axis
        };

        // Sort by center coordinate on the chosen axis and split at median.
        elems.sort_by(|&a, &b| {
            let ca = Self::elem_bbox(mesh, a).center()[axis];
            let cb = Self::elem_bbox(mesh, b).center()[axis];
            ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mid = elems.len() / 2;

        let left = Box::new(Self::build_recursive(mesh, &mut elems[..mid]));
        let right = Box::new(Self::build_recursive(mesh, &mut elems[mid..]));

        BvhNode::Inner {
            bbox: node_bbox,
            left,
            right,
        }
    }

    /// Compute the AABB of element `e`.
    fn elem_bbox(mesh: &Mesh<D>, e: ElemId) -> Aabb<D> {
        let ns = mesh.elem_nodes(e);
        let mut lo = mesh.coords_of(ns[0]);
        let mut hi = mesh.coords_of(ns[0]);
        for &n in ns.iter().skip(1) {
            let c = mesh.coords_of(n);
            for i in 0..D {
                lo[i] = lo[i].min(c[i]);
                hi[i] = hi[i].max(c[i]);
            }
        }
        Aabb::new(lo, hi)
    }

    /// Return all element ids whose AABBs contain `p` (with tolerance `tol`),
    /// sorted by AABB-center distance to `p`.
    pub fn locate_candidates(&self, p: &[f64], tol: f64) -> Vec<ElemId> {
        let mut result: Vec<(ElemId, f64)> = Vec::new();
        Self::collect_containing(&self.root, p, tol, &mut result);
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        result.into_iter().map(|(e, _)| e).collect()
    }

    fn collect_containing(
        node: &BvhNode<D>,
        p: &[f64],
        tol: f64,
        out: &mut Vec<(ElemId, f64)>,
    ) {
        match node {
            BvhNode::Leaf { bbox, elem } => {
                if bbox.dist2(p) <= tol * tol {
                    out.push((*elem, bbox.center_dist2(p)));
                }
            }
            BvhNode::Inner { bbox, left, right } => {
                if bbox.dist2(p) > tol * tol {
                    return;
                }
                Self::collect_containing(left, p, tol, out);
                Self::collect_containing(right, p, tol, out);
            }
        }
    }

    /// Nearest AABB to point `p` by center distance (used as fallback when
    /// no AABB contains `p`).
    pub fn nearest_box(&self, p: &[f64]) -> ElemId {
        let mut best: Option<(f64, ElemId)> = None;
        Self::nearest_recursive(&self.root, p, &mut best);
        best.expect("BVH is empty").1
    }

    fn nearest_recursive(node: &BvhNode<D>, p: &[f64], best: &mut Option<(f64, ElemId)>) {
        match node {
            BvhNode::Leaf { bbox, elem } => {
                let d2 = bbox.center_dist2(p);
                if best.map_or(true, |(b, _)| d2 < b) {
                    *best = Some((d2, *elem));
                }
            }
            BvhNode::Inner { left, right, .. } => {
                Self::nearest_recursive(left, p, best);
                Self::nearest_recursive(right, p, best);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bvh_builds_and_locates_2d() {
        let m = Mesh::<2>::unit_square_tri(4);
        let bvh = Bvh::new(&m);
        let cands = bvh.locate_candidates(&[0.37, 0.41], 1e-9);
        assert!(!cands.is_empty(), "should find a candidate");
    }

    #[test]
    fn bvh_nearest_box_2d() {
        let m = Mesh::<2>::unit_square_tri(4);
        let bvh = Bvh::new(&m);
        let e = bvh.nearest_box(&[0.37, 0.41]);
        assert!((e as usize) < m.n_elems());
    }

    #[test]
    fn bvh_builds_3d() {
        let m = Mesh::<3>::unit_cube_tet(3);
        let bvh = Bvh::new(&m);
        let cands = bvh.locate_candidates(&[0.21, 0.41, 0.37], 1e-9);
        assert!(!cands.is_empty());
    }

    #[test]
    fn bvh_candidates_sorted_by_distance() {
        let m = Mesh::<2>::unit_square_tri(8);
        let bvh = Bvh::new(&m);
        let p = [0.5, 0.5];
        let cands = bvh.locate_candidates(&p, 1e-9);
        // Verify sorted order: each candidate's AABB center should be at least
        // as far from p as the previous candidate's AABB center.
        for w in cands.windows(2) {
            let d0 = bvh_aabb_center_dist2(&m, w[0], &p);
            let d1 = bvh_aabb_center_dist2(&m, w[1], &p);
            assert!(d0 <= d1 + 1e-15, "candidates not sorted by AABB-center distance");
        }
    }

    fn bvh_aabb_center_dist2(mesh: &Mesh<2>, e: ElemId, p: &[f64]) -> f64 {
        let ns = mesh.elem_nodes(e);
        let mut lo = mesh.coords_of(ns[0]);
        let mut hi = mesh.coords_of(ns[0]);
        for &n in ns.iter().skip(1) {
            let c = mesh.coords_of(n);
            lo[0] = lo[0].min(c[0]); lo[1] = lo[1].min(c[1]);
            hi[0] = hi[0].max(c[0]); hi[1] = hi[1].max(c[1]);
        }
        let cx = 0.5 * (lo[0] + hi[0]);
        let cy = 0.5 * (lo[1] + hi[1]);
        let dx = cx - p[0];
        let dy = cy - p[1];
        dx * dx + dy * dy
    }
}
