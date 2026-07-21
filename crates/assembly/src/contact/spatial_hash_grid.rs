//! Spatial hash grid for accelerating 2D N2S contact search.
//!
//! Divides the bounding box of all master segments into a uniform grid,
//! so each slave query only checks segments in nearby cells (O(1) average).
//!
//! # Usage
//! ```rust,ignore
//! use fem_assembly::contact::spatial_hash_grid::{SpatialHashGrid2D, build_segment_grid};
//!
//! let segments = build_segment_index(&master_mesh, &[1]);
//! let grid = SpatialHashGrid2D::new(&segments, cell_size);
//! let nearest = grid.find_closest(&query_point, &segments, search_dist);
//! ```

/// A segment in 2D contact represented by its two endpoints.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct IndexedSegment {
    pub x0: [f64; 2],
    pub x1: [f64; 2],
}

/// Uniform grid spatial index for 2D segments.
pub struct SpatialHashGrid2D {
    /// Grid origin (min corner of bounding box).
    origin: [f64; 2],
    /// Cell size.
    cell_size: f64,
    /// Number of cells in x and y.
    nx: usize,
    ny: usize,
    /// For each cell, list of segment indices whose bounding box overlaps it.
    cells: Vec<Vec<usize>>,
}

impl SpatialHashGrid2D {
    /// Build a spatial hash grid from a list of segments.
    ///
    /// `cell_size` controls grid resolution; smaller = fewer segments per cell
    /// but more memory. A good default is `search_dist * 2.0`.
    pub fn new(
        seg_x0: &[[f64; 2]],
        seg_x1: &[[f64; 2]],
        cell_size: f64,
    ) -> Self {
        let n_seg = seg_x0.len();
        if n_seg == 0 {
            return Self { origin: [0.0; 2], cell_size, nx: 1, ny: 1, cells: vec![vec![]] };
        }

        // Compute bounding box
        let mut xmin = f64::MAX;
        let mut ymin = f64::MAX;
        let mut xmax = f64::NEG_INFINITY;
        let mut ymax = f64::NEG_INFINITY;
        for i in 0..n_seg {
            xmin = xmin.min(seg_x0[i][0]).min(seg_x1[i][0]);
            ymin = ymin.min(seg_x0[i][1]).min(seg_x1[i][1]);
            xmax = xmax.max(seg_x0[i][0]).max(seg_x1[i][0]);
            ymax = ymax.max(seg_x0[i][1]).max(seg_x1[i][1]);
        }

        let eps = 1e-12;
        let w = (xmax - xmin).max(eps);
        let h = (ymax - ymin).max(eps);
        let nx = (w / cell_size).ceil() as usize + 1;
        let ny = (h / cell_size).ceil() as usize + 1;

        let mut cells = vec![vec![]; nx * ny];

        for i in 0..n_seg {
            // Bounding box of this segment
            let sx0 = seg_x0[i][0].min(seg_x1[i][0]);
            let sx1 = seg_x0[i][0].max(seg_x1[i][0]);
            let sy0 = seg_x0[i][1].min(seg_x1[i][1]);
            let sy1 = seg_x0[i][1].max(seg_x1[i][1]);

            let ic_min = ((sx0 - xmin) / cell_size).floor() as isize;
            let ic_max = ((sx1 - xmin) / cell_size).ceil() as isize;
            let jc_min = ((sy0 - ymin) / cell_size).floor() as isize;
            let jc_max = ((sy1 - ymin) / cell_size).ceil() as isize;

            for jc in jc_min..=jc_max {
                for ic in ic_min..=ic_max {
                    if ic >= 0 && (ic as usize) < nx && jc >= 0 && (jc as usize) < ny {
                        let idx = jc as usize * nx + ic as usize;
                        cells[idx].push(i);
                    }
                }
            }
        }

        Self { origin: [xmin, ymin], cell_size, nx, ny, cells }
    }

    /// Find the closest segment to a query point within `search_dist`.
    ///
    /// Only checks segments in the same and adjacent grid cells.
    /// Returns `(segment_index, closest_point, gap_squared, xi_parameter)`.
    pub fn find_closest(
        &self,
        query: &[f64; 2],
        seg_x0: &[[f64; 2]],
        seg_x1: &[[f64; 2]],
        search_dist: f64,
    ) -> Option<(usize, [f64; 2], f64, f64)> {
        let ic = ((query[0] - self.origin[0]) / self.cell_size).floor() as isize;
        let jc = ((query[1] - self.origin[1]) / self.cell_size).floor() as isize;
        let search_sq = search_dist * search_dist;

        let mut best_idx = 0usize;
        let mut best_closest = [0.0; 2];
        let mut best_dist_sq = search_sq;
        let mut best_xi = 0.0;
        let mut found = false;

        // Check 3×3 cell neighborhood
        for dj in -1..=1 {
            for di in -1..=1 {
                let i = ic + di;
                let j = jc + dj;
                if i < 0 || j < 0 || i as usize >= self.nx || j as usize >= self.ny {
                    continue;
                }
                let cell_idx = j as usize * self.nx + i as usize;
                for &si in &self.cells[cell_idx] {
                    if si >= seg_x0.len() { continue; }
                    let p0 = &seg_x0[si];
                    let p1 = &seg_x1[si];
                    if let Some((closest, dist_sq, xi)) =
                        point_to_segment_dist_sq(query, p0, p1)
                    {
                        if dist_sq < best_dist_sq {
                            best_dist_sq = dist_sq;
                            best_closest = closest;
                            best_idx = si;
                            best_xi = xi;
                            found = true;
                        }
                    }
                }
            }
        }

        if found { Some((best_idx, best_closest, best_dist_sq.sqrt(), best_xi)) } else { None }
    }
}

/// Minimum distance squared from a point to a line segment in 2D.
fn point_to_segment_dist_sq(
    p: &[f64; 2],
    a: &[f64; 2],
    b: &[f64; 2],
) -> Option<([f64; 2], f64, f64)> {
    let abx = b[0] - a[0];
    let aby = b[1] - a[1];
    let apx = p[0] - a[0];
    let apy = p[1] - a[1];

    let ab_sq = abx * abx + aby * aby;
    if ab_sq < 1e-30 {
        // Segment is a point
        let dx = p[0] - a[0];
        let dy = p[1] - a[1];
        return Some(([a[0], a[1]], dx * dx + dy * dy, 0.0));
    }

    let t = (apx * abx + apy * aby) / ab_sq;
    let t_clamped = t.clamp(0.0, 1.0);
    let cx = a[0] + t_clamped * abx;
    let cy = a[1] + t_clamped * aby;
    let dx = p[0] - cx;
    let dy = p[1] - cy;

    Some(([cx, cy], dx * dx + dy * dy, t_clamped))
}

/// Build a SpatialHashGrid2D from contact segments, then find closest.
///
/// Convenience wrapper matching the `find_closest_segment` interface.
pub fn find_closest_segment_grid(
    query: &[f64; 2],
    seg_x0: &[[f64; 2]],
    seg_x1: &[[f64; 2]],
    grid: &SpatialHashGrid2D,
    search_dist: f64,
) -> Option<(usize, [f64; 2], f64, f64)> {
    grid.find_closest(query, seg_x0, seg_x1, search_dist)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_dist_sq_perpendicular() {
        let p = [0.0, 1.0];
        let a = [-1.0, 0.0];
        let b = [1.0, 0.0];
        let (closest, dist_sq, _xi) = point_to_segment_dist_sq(&p, &a, &b).unwrap();
        assert!((closest[0] - 0.0).abs() < 1e-12);
        assert!((closest[1] - 0.0).abs() < 1e-12);
        assert!((dist_sq - 1.0).abs() < 1e-12);
    }

    #[test]
    fn segment_dist_sq_beyond_endpoint() {
        let p = [2.0, 0.5];
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        let (_closest, dist_sq, xi) = point_to_segment_dist_sq(&p, &a, &b).unwrap();
        assert!((xi - 1.0).abs() < 1e-12, "should clamp to b");
        assert!((dist_sq - 1.25).abs() < 1e-12);
    }

    #[test]
    fn grid_finds_closest() {
        let seg_x0 = vec![[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]];
        let seg_x1 = vec![[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]];
        let grid = SpatialHashGrid2D::new(&seg_x0, &seg_x1, 2.0);
        let query = [0.5, 0.5];
        let result = find_closest_segment_grid(&query, &seg_x0, &seg_x1, &grid, 10.0);
        assert!(result.is_some(), "should find a segment");
        if let Some((idx, closest, gap, _xi)) = result {
            assert_eq!(idx, 0, "should find segment 0 (closest)");
            let dx = query[0] - closest[0];
            let dy = query[1] - closest[1];
            assert!((gap - (dx*dx + dy*dy).sqrt()).abs() < 1e-12);
        }
    }

    #[test]
    fn grid_no_result_outside_search() {
        let seg_x0 = vec![[0.0, 0.0]];
        let seg_x1 = vec![[1.0, 0.0]];
        let grid = SpatialHashGrid2D::new(&seg_x0, &seg_x1, 1.0);
        let query = [100.0, 100.0];
        let result = find_closest_segment_grid(&query, &seg_x0, &seg_x1, &grid, 1.0);
        assert!(result.is_none(), "should not find segment far away");
    }

    #[test]
    fn empty_grid_handled() {
        let seg_x0: Vec<[f64; 2]> = vec![];
        let seg_x1: Vec<[f64; 2]> = vec![];
        let grid = SpatialHashGrid2D::new(&seg_x0, &seg_x1, 1.0);
        let query = [0.0, 0.0];
        let result = find_closest_segment_grid(&query, &seg_x0, &seg_x1, &grid, 1.0);
        assert!(result.is_none());
    }

    #[test]
    fn grid_handles_many_segments() {
        let mut seg_x0 = Vec::new();
        let mut seg_x1 = Vec::new();
        for i in 0..100 {
            let x = i as f64 * 0.1;
            seg_x0.push([x, 0.0]);
            seg_x1.push([x + 0.05, 0.0]);
        }
        let grid = SpatialHashGrid2D::new(&seg_x0, &seg_x1, 0.2);
        // Query near the middle
        let query = [5.0, 0.1];
        let result = find_closest_segment_grid(&query, &seg_x0, &seg_x1, &grid, 1.0);
        assert!(result.is_some(), "should find segment in dense grid");
    }
}
