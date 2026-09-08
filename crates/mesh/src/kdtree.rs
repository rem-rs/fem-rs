//! KD-tree point cloud and nodal GridFunction projection — 1:1 port of MFEM
//! `general/kdtree.hpp` (`KDTree<Tindex=int, Tfloat=real_t, ndim>`, `Norm_l2`)
//! and `fem/kdtree.hpp` / `fem/kdtree.cpp` (`KDTreeNodalProjection`).
//!
//! The tree stores a flat point cloud; `sort` builds the balanced kd-tree by
//! recursive median splitting (`std::nth_element` ≙
//! `[f64]::select_nth_unstable_by`), cycling the split dimension by depth.
//! `find_closest_point` replicates MFEM's `PSearch` branch-for-branch, so the
//! nearest-neighbour result (index + distance) matches the C++ implementation.
//!
//! Layout note: `nth_element`/`select_nth_unstable` only pin the element at
//! the median *position* (the unique order statistic). When several points
//! share the same split-dimension coordinate the partition *sets* (and hence
//! the internal layout) are unspecified and may differ between libstdc++ and
//! Rust. This never changes the closest point itself — `PSearch` pruning is
//! exact — but which of two *equidistant* points wins a strict `<` race can
//! depend on the layout, as in C++.
//!
//! `KdTreeNodalProjection` mirrors `BaseKDTreeNodalProjection` /
//! `KDTreeNodalProjection<kdim>`: the closest **single** source point within
//! tolerance `lerr` is copied to the target node (no interpolation, no
//! k-nearest weighting), guarded by a bounding-box rejection.

/// Vector-DOF ordering (values match MFEM `Ordering::byNODES = 0`,
/// `byVDIM = 1`; same semantics as `fem_space::Ordering`, duplicated here
/// because `fem-space` depends on `fem-mesh`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ordering {
    /// MFEM `Ordering::byNODES` — block layout: `vdof = dof + ndofs*vd`.
    ByNodes = 0,
    /// MFEM `Ordering::byVDIM` — node-major interleaved: `vdof = vd + vdim*dof`.
    ByVdim = 1,
}

/// KD-tree over `NDIM`-dimensional points with attached indices — port of
/// MFEM `KDTree<int, real_t, NDIM, Norm_l2>` (`general/kdtree.hpp`).
#[derive(Debug, Clone)]
pub struct KdTree<const NDIM: usize> {
    /// The point cloud (sorted in place by [`KdTree::sort`]).
    data: Vec<Node<NDIM>>,
}

/// One cloud entry: MFEM `KDTree::NodeND` (`PointND pt` + `Tindex ind`).
#[derive(Debug, Clone, Copy)]
struct Node<const NDIM: usize> {
    pt: [f64; NDIM],
    ind: usize,
}

/// Search state — MFEM `KDTree::PointS` (`dist`, `pos`, `level`, `sp`; the
/// C++ `level` field is write-only bookkeeping and is dropped here).
#[derive(Clone)]
struct PointS<const NDIM: usize> {
    dist: f64,
    pos: usize,
    sp: [f64; NDIM],
}

impl<const NDIM: usize> KdTree<NDIM> {
    /// MFEM `KDTreeNorms::Norm_l2`: `sqrt(x0² + … + x_{d-1}²)`, summed in the
    /// same order as the C++ loop (bitwise-comparable distances).
    #[inline]
    fn norm_l2(xx: &[f64; NDIM]) -> f64 {
        let mut tm = xx[0] * xx[0];
        for i in 1..NDIM {
            tm = tm + xx[i] * xx[i];
        }
        tm.sqrt()
    }

    /// MFEM `KDTree::Dist`: componentwise difference followed by `Norm_l2`.
    #[inline]
    fn dist(p1: &[f64; NDIM], p2: &[f64; NDIM]) -> f64 {
        let mut tp = [0.0_f64; NDIM];
        for i in 0..NDIM {
            tp[i] = p1[i] - p2[i];
        }
        Self::norm_l2(&tp)
    }

    /// Adds a new node by coordinates and an associated index
    /// (MFEM `KDTree::AddPoint`).
    pub fn add_point(&mut self, xx: &[f64; NDIM], ind: usize) {
        self.data.push(Node { pt: *xx, ind });
    }
    /// Returns the size of the point cloud.
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Clears the point cloud.
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Builds the KD-tree (MFEM `KDTree::Sort` → `SortInPlace`).
    ///
    /// If the point cloud is modified the tree needs to be rebuilt by a new
    /// call to `sort`.
    pub fn sort(&mut self) {
        let n = self.data.len();
        Self::sort_in_place(&mut self.data, 0, n, 0);
    }

    /// MFEM `KDTree::SortInPlace`: median split at `siz/2` on coordinate
    /// `level % ndim`, recursing when more than two nodes remain.
    fn sort_in_place(data: &mut [Node<NDIM>], itb: usize, ite: usize, level: usize) {
        let cdim = level % NDIM;
        let siz = ite - itb;
        if siz > 2 {
            data[itb..ite].select_nth_unstable_by(siz / 2, |a, b| {
                a.pt[cdim].partial_cmp(&b.pt[cdim]).unwrap()
            });
            let level = level + 1;
            Self::sort_in_place(data, itb, itb + siz / 2, level);
            Self::sort_in_place(data, itb + siz / 2 + 1, ite, level);
        }
    }

    /// Nearest-neighbour search — MFEM `KDTree::PSearch`, replicated
    /// branch-for-branch (including the C++ control flow in which the median
    /// node is only tested inside the second-subtree condition of the
    /// "check all" branch).
    fn p_search(data: &[Node<NDIM>], itb: usize, ite: usize, level: usize, bc: &mut PointS<NDIM>) {
        let dim = level % NDIM;
        let siz = ite - itb;
        let mtb = itb + siz / 2;
        if siz > 2 {
            // median is at itb+siz/2
            let level = level + 1;
            if (bc.sp[dim] - bc.dist) > data[mtb].pt[dim] {
                // look on the right only
                Self::p_search(data, itb + siz / 2 + 1, ite, level, bc);
            } else if (bc.sp[dim] + bc.dist) < data[mtb].pt[dim] {
                // look on the left only
                Self::p_search(data, itb, itb + siz / 2, level, bc);
            } else {
                // check all
                if bc.sp[dim] < data[mtb].pt[dim] {
                    // start with the left portion
                    Self::p_search(data, itb, itb + siz / 2, level, bc);
                    // and continue to the right
                    if !((bc.sp[dim] + bc.dist) < data[mtb].pt[dim]) {
                        Self::p_search(data, itb + siz / 2 + 1, ite, level, bc);
                        // check central one
                        let dd = Self::dist(&data[mtb].pt, &bc.sp);
                        if dd < bc.dist {
                            bc.dist = dd;
                            bc.pos = mtb;
                        }
                    } // end central point check
                } else {
                    // start with the right portion
                    Self::p_search(data, itb + siz / 2 + 1, ite, level, bc);
                    // and continue with left
                    if !((bc.sp[dim] - bc.dist) > data[mtb].pt[dim]) {
                        Self::p_search(data, itb, itb + siz / 2, level, bc);
                        // check central one
                        let dd = Self::dist(&data[mtb].pt, &bc.sp);
                        if dd < bc.dist {
                            bc.dist = dd;
                            bc.pos = mtb;
                        }
                    } // end central point check
                }
            }
        } else {
            // check the nodes
            for it in itb..ite {
                let dd = Self::dist(&data[it].pt, &bc.sp);
                if dd < bc.dist {
                    // update bc
                    bc.pos = it;
                    bc.dist = dd;
                }
            }
        }
    }

    /// Initializes the best candidate as MFEM does (`data[0]`, distance to it)
    /// and runs `PSearch` over the whole cloud.
    fn run_p_search(&self, sp: &[f64; NDIM]) -> PointS<NDIM> {
        let mut bc = PointS::<NDIM> { dist: 0.0, pos: 0, sp: *sp };
        bc.dist = Self::dist(&self.data[0].pt, sp);
        Self::p_search(&self.data, 0, self.data.len(), 0, &mut bc);
        bc
    }

    /// Finds the nearest neighbour: returns `(index, distance)`
    /// (MFEM `FindClosestPoint(pt, ind, dist)`).
    ///
    /// # Panics
    /// Panics on an empty cloud (as the C++ code would via `data[0]`).
    pub fn find_closest_point(&self, pt: &[f64; NDIM]) -> (usize, f64) {
        let bc = self.run_p_search(pt);
        (self.data[bc.pos].ind, bc.dist)
    }

    /// Finds the nearest neighbour and returns the closest point:
    /// `(index, distance, closest point)` (MFEM
    /// `FindClosestPoint(pt, ind, dist, clp)`).
    pub fn find_closest_point_location(&self, pt: &[f64; NDIM]) -> (usize, f64, [f64; NDIM]) {
        let bc = self.run_p_search(pt);
        (self.data[bc.pos].ind, bc.dist, self.data[bc.pos].pt)
    }

    /// Brute-force nearest neighbour — MFEM `FindClosestPointSlow`
    /// (debugging reference).
    pub fn find_closest_point_slow(&self, pt: &[f64; NDIM]) -> (usize, f64) {
        let mut best = 0usize;
        let mut best_dist = Self::dist(&self.data[0].pt, pt);
        for (i, node) in self.data.iter().enumerate().skip(1) {
            let dd = Self::dist(&node.pt, pt);
            if dd < best_dist {
                best = i;
                best_dist = dd;
            }
        }
        (self.data[best].ind, best_dist)
    }

    /// Finds all points within distance `r` of `pt`: returns `(indices,
    /// distances)` (MFEM `FindNeighborPoints(pt, R, res, dist)`).
    ///
    /// The C++ overload takes `&mut` vectors and appends; this port returns
    /// fresh vectors (callers can `extend` to append).
    pub fn find_neighbor_points(&self, pt: &[f64; NDIM], r: f64) -> (Vec<usize>, Vec<f64>) {
        let mut res = Vec::new();
        let mut dist = Vec::new();
        Self::find_neighbor_points_rec(&self.data, pt, r, 0, self.data.len(), 0, &mut res, &mut dist);
        (res, dist)
    }

    /// MFEM `FindNeighborPoints(pt, R, itb, ite, level, res, dist)`.
    #[allow(clippy::too_many_arguments)]
    fn find_neighbor_points_rec(
        data: &[Node<NDIM>],
        pt: &[f64; NDIM],
        r: f64,
        itb: usize,
        ite: usize,
        level: usize,
        res: &mut Vec<usize>,
        dist: &mut Vec<f64>,
    ) {
        let dim = level % NDIM;
        let siz = ite - itb;
        let mtb = itb + siz / 2;
        if siz > 2 {
            // median is at itb+siz/2
            let level = level + 1;
            if (pt[dim] - r) > data[mtb].pt[dim] {
                // look to the right only
                Self::find_neighbor_points_rec(data, pt, r, itb + siz / 2 + 1, ite, level, res, dist);
            } else if (pt[dim] + r) < data[mtb].pt[dim] {
                // look to the left only
                Self::find_neighbor_points_rec(data, pt, r, itb, itb + siz / 2, level, res, dist);
            } else {
                // check all
                Self::find_neighbor_points_rec(data, pt, r, itb + siz / 2 + 1, ite, level, res, dist); // right
                Self::find_neighbor_points_rec(data, pt, r, itb, itb + siz / 2, level, res, dist); // left

                // check central one
                let dd = Self::dist(&data[mtb].pt, pt);
                if dd < r {
                    res.push(data[mtb].ind);
                    dist.push(dd);
                }
            }
        } else {
            for it in itb..ite {
                let dd = Self::dist(&data[it].pt, pt);
                if dd < r {
                    // update bc
                    res.push(data[it].ind);
                    dist.push(dd);
                }
            }
        }
    }

    /// Brute-force neighbor search — MFEM `FindNeighborPointsSlow`
    /// (debugging reference). Returns `(indices, distances)`.
    pub fn find_neighbor_points_slow(&self, pt: &[f64; NDIM], r: f64) -> (Vec<usize>, Vec<f64>) {
        let mut res = Vec::new();
        let mut dist = Vec::new();
        for node in &self.data {
            let dd = Self::dist(&node.pt, pt);
            if dd < r {
                res.push(node.ind);
                dist.push(dd);
            }
        }
        (res, dist)
    }
}

/// Nodal projection between point clouds — port of MFEM
/// `KDTreeNodalProjection<kdim>` (`fem/kdtree.hpp` + `fem/kdtree.cpp`).
///
/// The target ("dest") cloud is fixed at construction: the tree is built over
/// the target points (each target dof added once — the C++ constructor dedups
/// shared FE dofs via an `indt` flag, so callers must pass unique points) and
/// the target bounding box `[minbb, maxbb]` is recorded.
///
/// Each `project*` call copies values from a source cloud: a source point is
/// matched against the closest target point, and the value is copied only if
/// the distance is below the tolerance `lerr` (C++ default `1e-8`). Points
/// outside the (inflated) bounding boxes are skipped — no interpolation is
/// performed, matching the C++ semantics exactly.
#[derive(Debug, Clone)]
pub struct KdTreeNodalProjection<const D: usize> {
    /// Pointer to the KDTree (MFEM `kdt`).
    kdt: KdTree<D>,
    /// Ordering of the target dof vector (MFEM reads it from the dest
    /// `FiniteElementSpace`; `fem-rs` FE spaces use `ByNodes`).
    dest_ordering: Ordering,
    /// Number of target points (MFEM `GetVSize()/GetVDim()`).
    n_dest: usize,
    /// Upper corner of the bounding box (MFEM `maxbb`).
    maxbb: [f64; D],
    /// Lower corner of the bounding box (MFEM `minbb`).
    minbb: [f64; D],
}

impl<const D: usize> KdTreeNodalProjection<D> {
    /// Builds the projection over target points `dest_coords`
    /// (MFEM `KDTreeNodalProjection(dest_)` constructor: add every nodal dof
    /// once, track the bounding box, then `Sort`). Uses `Ordering::ByNodes`
    /// for the target vector (the `fem-rs` FE-space convention).
    pub fn new(dest_coords: &[[f64; D]]) -> Self {
        Self::with_dest_ordering(dest_coords, Ordering::ByNodes)
    }

    /// Same as [`KdTreeNodalProjection::new`] with an explicit target
    /// ordering (MFEM: `dest->FESpace()->GetOrdering()`).
    pub fn with_dest_ordering(dest_coords: &[[f64; D]], dest_ordering: Ordering) -> Self {
        let mut kdt = KdTree::new();
        let mut minbb = dest_coords
            .first()
            .map(|p| *p)
            .unwrap_or([0.0; D]);
        let mut maxbb = minbb;
        for (bind, c) in dest_coords.iter().enumerate() {
            kdt.add_point(c, bind);
            for d in 0..D {
                if minbb[d] > c[d] {
                    minbb[d] = c[d];
                }
                if maxbb[d] < c[d] {
                    maxbb[d] = c[d];
                }
            }
        }
        // build the KDTree
        kdt.sort();
        Self { kdt, dest_ordering, n_dest: dest_coords.len(), maxbb, minbb }
    }

    /// Lower corner of the target bounding box.
    pub fn minbb(&self) -> &[f64; D] {
        &self.minbb
    }

    /// Upper corner of the target bounding box.
    pub fn maxbb(&self) -> &[f64; D] {
        &self.maxbb
    }

    /// Number of target points.
    pub fn n_dest(&self) -> usize {
        self.n_dest
    }

    /// Copies source values onto the closest target points — port of MFEM
    /// `KDTreeNodalProjection::Project(coords, src, ordering, lerr)`.
    ///
    /// - `dest`: target dof vector, `vdim * n_dest` entries laid out per the
    ///   target ordering given at construction (MFEM: the dest GridFunction).
    /// - `coords`: source point coordinates, `np * D` entries, node-major
    ///   (`coords(i*D + d)`), as in the C++ `coords` vector.
    /// - `src`: source values, `np * vdim` entries laid out per `ordering`.
    /// - `ordering`: source data ordering (`Ordering::ByNodes` → `src[vd*np+i]`
    ///   is point `i` component `vd`; `Ordering::ByVdim` → `src[vd + i*vd]`).
    /// - `lerr`: match tolerance (C++ default `1e-8`); a source point is also
    ///   rejected when it lies outside the target bounding box inflated by
    ///   `lerr`.
    ///
    /// `vdim` is inferred as `dest.len() / n_dest` (MFEM
    /// `dest->VectorDim()`), and `np = src.len() / vdim` — i.e. source and
    /// target must have the same number of points per component, as in C++.
    pub fn project(&self, dest: &mut [f64], coords: &[f64], src: &[f64], ordering: Ordering, lerr: f64) {
        debug_assert_eq!(dest.len() % self.n_dest, 0, "dest size must be vdim * n_dest");
        let vd = dest.len() / self.n_dest; // dimension of the vector field
        let np = src.len() / vd; // number of points
        debug_assert_eq!(coords.len(), np * D, "coords size must be np * dim");
        for i in 0..np {
            let mut pnd = [0.0_f64; D];
            for (j, p) in pnd.iter_mut().enumerate() {
                *p = coords[i * D + j];
            }

            let mut pt_inside_bbox = true;
            for j in 0..D {
                if pnd[j] > (self.maxbb[j] + lerr) {
                    pt_inside_bbox = false;
                    break;
                }
                if pnd[j] < (self.minbb[j] - lerr) {
                    pt_inside_bbox = false;
                    break;
                }
            }

            if pt_inside_bbox {
                let (ind, dist) = self.kdt.find_closest_point(&pnd);
                if dist < lerr {
                    self.copy_value(dest, ind, src, i, np, vd, ordering);
                }
            }
        }
    }

    /// Projects a source nodal field given by its dof coordinates and values —
    /// port of MFEM
    /// `KDTreeNodalProjection::Project(const GridFunction& gf, lerr)`.
    ///
    /// Unlike [`KdTreeNodalProjection::project`], the C++ GridFunction
    /// overload does **not** test each source point against the bounding box
    /// individually; instead it computes the source bounding box once and
    /// returns early when it does not intersect the (inflated) target box.
    ///
    /// - `src_coords`: source dof coordinates, `np * D`, node-major — the
    ///   C++ overload extracts these by transforming each element's nodal
    ///   integration rule; the caller supplies the (unique) source dof
    ///   coordinates here.
    /// - `src`: source values, `np * vdim`, per `ordering`.
    pub fn project_gridfunction(
        &self,
        dest: &mut [f64],
        src_coords: &[f64],
        src: &[f64],
        ordering: Ordering,
        lerr: f64,
    ) {
        debug_assert_eq!(dest.len() % self.n_dest, 0, "dest size must be vdim * n_dest");
        let vd = dest.len() / self.n_dest; // dimension of the vector field
        let np = src_coords.len() / D; // number of source points (C++: gf VSize/VDim)
        debug_assert_eq!(src.len(), np * vd, "src size must be np * vdim");

        let mut maxbb_src = [0.0_f64; D];
        let mut minbb_src = [0.0_f64; D];

        // extract the bounding box of the source coordinates
        if np > 0 {
            maxbb_src = src_coords[0..D].try_into().unwrap();
            minbb_src = maxbb_src;
            for p in 0..np {
                for d in 0..D {
                    let c = src_coords[p * D + d];
                    if maxbb_src[d] < c {
                        maxbb_src[d] = c;
                    }
                    if minbb_src[d] > c {
                        minbb_src[d] = c;
                    }
                }
            }
        }

        for d in 0..D {
            maxbb_src[d] += lerr;
            minbb_src[d] -= lerr;
        }

        // check for intersection
        for d in 0..D {
            if minbb_src[d] > self.maxbb[d] || maxbb_src[d] < self.minbb[d] {
                return;
            }
        }

        for i in 0..np {
            let mut pnd = [0.0_f64; D];
            for (j, p) in pnd.iter_mut().enumerate() {
                *p = src_coords[i * D + j];
            }

            let (ind, dist) = self.kdt.find_closest_point(&pnd);
            if dist < lerr {
                self.copy_value(dest, ind, src, i, np, vd, ordering);
            }
        }
    }

    /// The four-branch ordering-conversion copy shared by both `Project`
    /// overloads (MFEM `fem/kdtree.cpp`).
    fn copy_value(
        &self,
        dest: &mut [f64],
        ind: usize,
        src: &[f64],
        i: usize,
        np: usize,
        vd: usize,
        ordering: Ordering,
    ) {
        if self.dest_ordering == Ordering::ByNodes {
            if ordering == Ordering::ByNodes {
                for di in 0..vd {
                    dest[di * np + ind] = src[di * np + i];
                }
            } else {
                for di in 0..vd {
                    dest[di * np + ind] = src[di + i * vd];
                }
            }
        } else if ordering == Ordering::ByNodes {
            for di in 0..vd {
                dest[di + ind * vd] = src[di * np + i];
            }
        } else {
            for di in 0..vd {
                dest[di + ind * vd] = src[di + i * vd];
            }
        }
    }
}

impl<const NDIM: usize> Default for KdTree<NDIM> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const NDIM: usize> KdTree<NDIM> {
    /// Default constructor (MFEM `KDTree() = default`).
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift64* PRNG (no external rand dependency).
    struct Rng(u64);
    impl Rng {
        fn next_f64(&mut self) -> f64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            let v = x.wrapping_mul(0x2545F4914F6CDD1D);
            (v >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn fill_tree<const D: usize>(pts: &[[f64; D]], rng: &mut Rng) -> KdTree<D> {
        // shuffle insertion order so the tree structure is exercised
        let mut order: Vec<usize> = (0..pts.len()).collect();
        for i in (1..order.len()).rev() {
            let j = (rng.next_f64() * (i + 1) as f64) as usize;
            order.swap(i, j);
        }
        let mut t = KdTree::new();
        for &i in &order {
            t.add_point(&pts[i], i);
        }
        t.sort();
        t
    }

    #[test]
    fn nn_matches_brute_force_2d() {
        let mut rng = Rng(0x12345678);
        let pts: Vec<[f64; 2]> = (0..1000).map(|_| [rng.next_f64(), rng.next_f64()]).collect();
        let t = fill_tree(&pts, &mut rng);
        for _ in 0..300 {
            let q = [rng.next_f64(), rng.next_f64()];
            let (ind, dist) = t.find_closest_point(&q);
            let (ind_ref, dist_ref) = t.find_closest_point_slow(&q);
            assert_eq!(ind, ind_ref, "nn index mismatch for q={q:?}");
            assert_eq!(dist, dist_ref, "nn distance mismatch for q={q:?}");
        }
    }

    #[test]
    fn nn_matches_brute_force_3d() {
        let mut rng = Rng(0x9e3779b9);
        let pts: Vec<[f64; 3]> =
            (0..800).map(|_| [rng.next_f64(), rng.next_f64(), rng.next_f64()]).collect();
        let t = fill_tree(&pts, &mut rng);
        for _ in 0..300 {
            let q = [rng.next_f64(), rng.next_f64(), rng.next_f64()];
            let (ind, dist) = t.find_closest_point(&q);
            let (ind_ref, dist_ref) = t.find_closest_point_slow(&q);
            assert_eq!(ind, ind_ref);
            assert_eq!(dist, dist_ref);
        }
    }

    #[test]
    fn nn_matches_brute_force_on_lattice_with_tied_coordinates() {
        // Structured lattice: many points share each split-dimension
        // coordinate, so partition sets are not pinned by the median
        // guarantee — the search must still return the true nearest point.
        let n = 12;
        let mut pts = Vec::new();
        for j in 0..n {
            for i in 0..n {
                pts.push([i as f64 / (n - 1) as f64, j as f64 / (n - 1) as f64]);
            }
        }
        let mut rng = Rng(42);
        let t = fill_tree(&pts, &mut rng);
        for _ in 0..500 {
            let q = [rng.next_f64(), rng.next_f64()];
            let (ind, dist) = t.find_closest_point(&q);
            let (ind_ref, dist_ref) = t.find_closest_point_slow(&q);
            assert_eq!(dist, dist_ref);
            assert_eq!(pts[ind], pts[ind_ref], "distinct lattice points returned");
        }
    }

    #[test]
    fn neighbor_points_matches_brute_force() {
        let mut rng = Rng(0xdeadbeef);
        let pts: Vec<[f64; 3]> =
            (0..600).map(|_| [rng.next_f64(), rng.next_f64(), rng.next_f64()]).collect();
        let t = fill_tree(&pts, &mut rng);
        for _ in 0..50 {
            let q = [rng.next_f64(), rng.next_f64(), rng.next_f64()];
            let r = 0.1 + 0.3 * rng.next_f64();
            let (mut res, mut dist) = t.find_neighbor_points(&q, r);
            let (mut res_ref, mut dist_ref) = t.find_neighbor_points_slow(&q, r);
            res.sort_unstable();
            res_ref.sort_unstable();
            dist.sort_by(|a, b| a.partial_cmp(b).unwrap());
            dist_ref.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert_eq!(res, res_ref, "neighbor indices for q={q:?} r={r}");
            assert_eq!(dist, dist_ref, "neighbor distances for q={q:?} r={r}");
        }
    }

    #[test]
    fn projection_identity_on_same_grid() {
        // projecting a grid function onto its own nodes must reproduce the
        // values exactly (matched nodes => distance 0 => copy)
        let mesh: crate::Mesh<2> = crate::simplex::Mesh::unit_square_quad(4);
        let coords: Vec<[f64; 2]> =
            (0..mesh.n_nodes()).map(|n| mesh.coords_of(n as u32)).collect();
        let vals: Vec<f64> = coords.iter().map(|p| f(p)).collect();

        let proj = KdTreeNodalProjection::new(&coords);
        let mut dest = vec![0.0_f64; coords.len()];
        let flat: Vec<f64> = coords.iter().flat_map(|p| p.iter().copied()).collect();
        proj.project_gridfunction(&mut dest, &flat, &vals, Ordering::ByNodes, 1e-8);
        assert_eq!(dest, vals, "identity projection must copy values exactly");
    }

    /// Analytic field used by the projection tests: `sin(x)*cos(y) + x`.
    fn f(p: &[f64]) -> f64 {
        p[0].sin() * p[1].cos() + p[0]
    }

    #[test]
    fn projection_between_dyadic_grids_accuracy() {
        // Nodal transfer performs no interpolation: the accuracy contract is
        // that every *matched* target node (nearest source node within lerr)
        // receives the exact source value, and unmatched nodes stay at their
        // initial value. On dyadic refinements of a structured grid the
        // coincident node coordinates are bitwise equal, so the projected
        // field must equal the analytic field exactly on the matched set,
        // and the matched set must grow to everything as the source is
        // refined to the target resolution.
        let grid = |n: usize| -> Vec<[f64; 2]> {
            let m: crate::Mesh<2> = crate::simplex::Mesh::unit_square_quad(n);
            (0..m.n_nodes()).map(|v| m.coords_of(v as u32)).collect()
        };
        let target = grid(8);

        for n_src in [4usize, 8] {
            let source = grid(n_src);
            let vals: Vec<f64> = source.iter().map(|p| f(p)).collect();
            let mut flat_s = Vec::with_capacity(source.len() * 2);
            for p in &source {
                flat_s.extend_from_slice(p);
            }

            let proj = KdTreeNodalProjection::new(&target);
            let mut dest = vec![0.0_f64; target.len()];
            proj.project_gridfunction(&mut dest, &flat_s, &vals, Ordering::ByNodes, 1e-8);

            let mut n_matched = 0usize;
            let mut max_err = 0.0_f64;
            for (i, p) in target.iter().enumerate() {
                // brute-force nearest source node
                let (_, d) = source
                    .iter()
                    .enumerate()
                    .map(|(k, s)| (k, (s[0] - p[0]).powi(2) + (s[1] - p[1]).powi(2)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap();
                if d.sqrt() < 1e-8 {
                    n_matched += 1;
                    max_err = max_err.max((dest[i] - f(p)).abs());
                } else {
                    assert_eq!(dest[i], 0.0, "unmatched node {p:?} must stay 0");
                }
            }
            assert_eq!(max_err, 0.0, "matched nodes must be bitwise exact (src n={n_src})");
            if n_src == 8 {
                assert_eq!(n_matched, target.len(), "same-resolution grids must fully match");
            } else {
                // 5x5 source lattice inside the 9x9 target lattice
                assert_eq!(n_matched, 25, "coarse nodes must all be found");
            }
        }
    }

    #[test]
    fn projection_rejects_points_outside_bbox() {
        // target: unit square; source: same square shifted far in x — the
        // bounding-box check must reject every point and dest stays zero.
        let target: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let proj = KdTreeNodalProjection::new(&target);
        let shift = 10.0;
        let src: Vec<[f64; 2]> = target.iter().map(|p| [p[0] + shift, p[1]]).collect();
        let vals: Vec<f64> = (0..4).map(|i| i as f64 + 1.0).collect();
        let mut dest = vec![0.0_f64; 4];
        let flat: Vec<f64> = src.iter().flat_map(|p| p.iter().copied()).collect();
        proj.project(&mut dest, &flat, &vals, Ordering::ByNodes, 1e-8);
        assert!(dest.iter().all(|&v| v == 0.0), "no value may cross the bbox, got {dest:?}");

        // the GridFunction overload returns early via the intersection check
        let mut dest2 = vec![0.0_f64; 4];
        proj.project_gridfunction(&mut dest2, &flat, &vals, Ordering::ByNodes, 1e-8);
        assert!(dest2.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn projection_vector_field_all_orderings() {
        // vdim=2 source field in all four dest/source ordering combinations
        // must produce the same interleaved (per-point) result.
        let target: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let proj = KdTreeNodalProjection::new(&target);
        let np = target.len();
        let vd = 2;

        // source = target points; component values f0/f1 per point
        let f0 = |p: &[f64; 2]| 3.0 * p[0] + p[1];
        let f1 = |p: &[f64; 2]| p[0] - 2.0 * p[1];

        let by_nodes = {
            let mut s = Vec::with_capacity(np * vd);
            for p in &target {
                s.push(f0(p));
            }
            for p in &target {
                s.push(f1(p));
            }
            s
        };
        let by_vdim = {
            let mut s = Vec::with_capacity(np * vd);
            for p in &target {
                s.push(f0(p));
                s.push(f1(p));
            }
            s
        };

        let flat: Vec<f64> = target.iter().flat_map(|p| p.iter().copied()).collect();
        let mut dest_nodes = vec![0.0_f64; np * vd];
        proj.project(&mut dest_nodes, &flat, &by_nodes, Ordering::ByNodes, 1e-8);
        let mut dest_mixed = vec![0.0_f64; np * vd];
        proj.project(&mut dest_mixed, &flat, &by_vdim, Ordering::ByVdim, 1e-8);

        // ByNodes dest layout: component block vd: [f0 all points | f1 all]
        for (i, p) in target.iter().enumerate() {
            assert_eq!(dest_nodes[i], f0(p), "byNodes dest <- byNodes src, x-comp {i}");
            assert_eq!(dest_nodes[np + i], f1(p), "byNodes dest <- byNodes src, y-comp {i}");
            assert_eq!(dest_mixed[i], f0(p), "byNodes dest <- byVdim src, x-comp {i}");
            assert_eq!(dest_mixed[np + i], f1(p), "byNodes dest <- byVdim src, y-comp {i}");
        }

        // dest in ByVdim layout (interleaved: point i component di at
        // index di + i*vd) via a ByVdim-ordered projection object
        let proj_v = KdTreeNodalProjection::with_dest_ordering(&target, Ordering::ByVdim);
        let mut dest_v = vec![0.0_f64; np * vd];
        proj_v.project(&mut dest_v, &flat, &by_vdim, Ordering::ByVdim, 1e-8);
        let mut dest_mixed_v = vec![0.0_f64; np * vd];
        proj_v.project(&mut dest_mixed_v, &flat, &by_nodes, Ordering::ByNodes, 1e-8);
        for (i, p) in target.iter().enumerate() {
            assert_eq!(dest_v[i * vd], f0(p), "byVdim dest <- byVdim src, x-comp {i}");
            assert_eq!(dest_v[i * vd + 1], f1(p), "byVdim dest <- byVdim src, y-comp {i}");
            assert_eq!(dest_mixed_v[i * vd], f0(p), "byVdim dest <- byNodes src, x-comp {i}");
            assert_eq!(dest_mixed_v[i * vd + 1], f1(p), "byVdim dest <- byNodes src, y-comp {i}");
        }
    }

    #[test]
    fn projection_3d_tet_grids() {
        // uniformly refined cube tet mesh <- coarse cube: the target tree is
        // built over the fine nodes; every fine node coinciding with a coarse
        // node (dyadic coordinates are bitwise exact) must receive the exact
        // coarse value; unmatched fine nodes stay at 0.
        let coarse: crate::Mesh<3> = crate::simplex::Mesh::unit_cube_tet(2);
        let fine: crate::Mesh<3> = crate::amr::refine_uniform_3d(&coarse);
        let ccoords: Vec<[f64; 3]> =
            (0..coarse.n_nodes()).map(|n| coarse.coords_of(n as u32)).collect();
        let fcoords: Vec<[f64; 3]> =
            (0..fine.n_nodes()).map(|n| fine.coords_of(n as u32)).collect();

        let g = |p: &[f64; 3]| (p[0] * p[1]).sin() + p[2];
        let vals: Vec<f64> = ccoords.iter().map(|p| g(p)).collect();

        let proj = KdTreeNodalProjection::new(&fcoords);
        let mut dest = vec![0.0_f64; fcoords.len()];
        let flat: Vec<f64> = ccoords.iter().flat_map(|p| p.iter().copied()).collect();
        proj.project_gridfunction(&mut dest, &flat, &vals, Ordering::ByNodes, 1e-8);

        let mut n_matched = 0;
        for (i, p) in fcoords.iter().enumerate() {
            let (k, d) = coarse_node_dist(&ccoords, p);
            if d < 1e-8 {
                n_matched += 1;
                assert_eq!(dest[i], g(&ccoords[k]), "matched fine node {p:?} must be exact");
            } else {
                assert_eq!(dest[i], 0.0, "unmatched fine node {p:?} must stay 0");
            }
        }

        assert_eq!(n_matched, ccoords.len(), "every coarse node must be preserved & matched");
    }

    /// Brute-force nearest coarse-node distance (test helper).
    fn coarse_node_dist(ccoords: &[[f64; 3]], p: &[f64; 3]) -> (usize, f64) {
        let mut best = (usize::MAX, f64::INFINITY);
        for (i, c) in ccoords.iter().enumerate() {
            let d = ((c[0] - p[0]).powi(2) + (c[1] - p[1]).powi(2) + (c[2] - p[2]).powi(2)).sqrt();
            if d < best.1 {
                best = (i, d);
            }
        }
        best
    }


    /// Cross-check against the C++ `KDTreeNodalProjection` harness dumps
    /// (produced by `tmp/nd/nd_harness.cpp` against serial MFEM 4.9).
    ///
    /// For every scenario (2D quad order 1/2, shifted-source bbox rejection,
    /// 3D `beam-tet` tet order 1/2, and a vdim-2 case exercising all four
    /// (dest, src) ordering combinations) the C++ side dumps source
    /// coordinates/values, target coordinates and the projected target
    /// values as raw little-endian f64 binaries into `../../tmp/nd`.
    ///
    /// The Rust projection must reproduce the C++ target values **bitwise**:
    /// both sides copy the very same `f64` source values, so any difference
    /// would mean the kd-tree picked a different source point.

    /// Runs one harness scenario: reads the C++ dumps for `case`, projects
    /// with the requested (dest, src) orderings and compares bitwise.
    fn cross_check_case<const D: usize>(case: &str, dest_ordering: Ordering, src_ordering: Ordering) {
        let base = std::path::Path::new("../../tmp/nd");
        let read_doubles = |name: &str| -> Vec<f64> {
            let bytes = std::fs::read(base.join(name))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            bytes
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        // vec dumps are named vec_<kind>_<tag>.bin, scalar dumps <case>_<kind>.bin
        let fname = |kind: &str| -> String {
            if let Some(tag) = case.strip_prefix("vec_") {
                format!("vec_{kind}_{tag}.bin")
            } else {
                format!("{case}_{kind}.bin")
            }
        };
        let src_coords = read_doubles(&fname("src_coords"));
        let src_vals = read_doubles(&fname("src_vals"));
        let dest_coords = read_doubles(&fname("dest_coords"));
        let dest_vals_ref = read_doubles(&fname("dest_vals"));

        let np = src_coords.len() / D;
        let nd = dest_coords.len() / D;
        assert_eq!(src_vals.len() % np, 0, "{case}: vdim must divide value count");
        let vd = src_vals.len() / np;
        assert_eq!(dest_vals_ref.len(), vd * nd, "{case}: dest value count");

        let dest_pts: Vec<[f64; D]> = (0..nd)
            .map(|i| {
                let mut p = [0.0_f64; D];
                for d in 0..D {
                    p[d] = dest_coords[i * D + d];
                }
                p
            })
            .collect();

        let proj = KdTreeNodalProjection::with_dest_ordering(&dest_pts, dest_ordering);
        let mut out = vec![0.0_f64; nd * vd];
        proj.project_gridfunction(&mut out, &src_coords, &src_vals, src_ordering, 1e-8);

        let n_diff = out
            .iter()
            .zip(dest_vals_ref.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let max_diff = out
            .iter()
            .zip(dest_vals_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert_eq!(n_diff, 0, "{case}: bitwise mismatch at {n_diff}/{nd} (max {max_diff:e})");
    }

    /// Cross-check against the C++ `KDTreeNodalProjection` harness dumps
    /// (produced by `tmp/nd/nd_harness.cpp` against serial MFEM 4.9).
    ///
    /// For every scenario the C++ side dumps source coordinates/values,
    /// target coordinates and the projected target values as raw
    /// little-endian f64 binaries into `../../tmp/nd`. The Rust projection
    /// must reproduce the C++ target values **bitwise**: both sides copy the
    /// very same `f64` source values, so any difference would mean the
    /// kd-tree picked a different source point.
    ///
    /// Scenarios (see nd_harness.cpp):
    /// - `s2d`        2D quad order 1, source = refined target mesh
    /// - `s2d_o2`     2D quad order 2
    /// - `s2d_shift`  source shifted +0.5 in x (bbox rejection path)
    /// - `s3d`        beam-tet.mesh order 1, source = 2x refined target
    /// - `s3d_o2`     beam-tet.mesh order 2
    /// - `vec_*`      vdim = 2, all four (dest, src) ordering combinations
    #[test]
    fn cpp_cross_check_bitwise() {
        let base = std::path::Path::new("../../tmp/nd");
        if !base.is_dir() {
            // harness dumps not generated — skip silently
            return;
        }
        // (case, dest ordering, src ordering)
        const CASES: &[(&str, Ordering, Ordering)] = &[
            ("s2d", Ordering::ByNodes, Ordering::ByNodes),
            ("s2d_o2", Ordering::ByNodes, Ordering::ByNodes),
            ("s2d_shift", Ordering::ByNodes, Ordering::ByNodes),
            ("s3d", Ordering::ByNodes, Ordering::ByNodes),
            ("s3d_o2", Ordering::ByNodes, Ordering::ByNodes),
            ("vec_bn_bn", Ordering::ByNodes, Ordering::ByNodes),
            ("vec_bn_bv", Ordering::ByNodes, Ordering::ByVdim),
            ("vec_bv_bn", Ordering::ByVdim, Ordering::ByNodes),
            ("vec_bv_bv", Ordering::ByVdim, Ordering::ByVdim),
        ];
        for &(case, dest_ord, src_ord) in CASES {
            let dim = if case.starts_with("s3d") { 3 } else { 2 };
            match dim {
                2 => cross_check_case::<2>(case, dest_ord, src_ord),
                3 => cross_check_case::<3>(case, dest_ord, src_ord),
                _ => unreachable!(),
            }
        }
    }
}
