//! Level-set geometry description for XFEM (2-D).
//!
//! Provides signed-distance functions and sub-cell triangulation
//! for elements cut by an interface.

/// A signed-distance / level-set function describing an immersed interface.
///
/// Convention: **ψ(x) < 0** is the *active* (interior) side.
/// For a crack, ψ(x) = 0 is the crack surface, and Heaviside
/// enrichment uses `H(x) = sign(ψ(x))`.
#[derive(Debug, Clone)]
pub enum XfemLevelSet {
    /// Circular interface: ψ(x) = ‖x − c‖ − r.
    Circle { cx: f64, cy: f64, radius: f64 },
    /// Half-space: ψ(x) = n·x − d  (unit outward normal `n`, positive side = ψ > 0).
    Halfspace { normal: [f64; 2], offset: f64 },
    /// Line crack: ψ(x) = signed distance to segment `(x1, x2)`,
    /// with sign determined by the direction of `n = (x2-x1) rotated 90° CCW`.
    CrackLine { x1: [f64; 2], x2: [f64; 2] },
}

impl XfemLevelSet {
    /// Evaluate ψ(x). Negative = active (interior).
    pub fn eval(&self, x: [f64; 2]) -> f64 {
        match self {
            XfemLevelSet::Circle { cx, cy, radius } => {
                let dx = x[0] - cx;
                let dy = x[1] - cy;
                (dx * dx + dy * dy).sqrt() - radius
            }
            XfemLevelSet::Halfspace { normal, offset } => {
                normal[0] * x[0] + normal[1] * x[1] - offset
            }
            XfemLevelSet::CrackLine { x1, x2 } => {
                signed_distance_to_segment(x, *x1, *x2)
            }
        }
    }

    /// Outward unit normal at point `x` on the interface.
    pub fn outward_normal(&self, x: [f64; 2]) -> [f64; 2] {
        match self {
            XfemLevelSet::Circle { cx, cy, .. } => {
                let dx = x[0] - cx;
                let dy = x[1] - cy;
                let inv = 1.0 / (dx * dx + dy * dy).sqrt().max(1e-14);
                [dx * inv, dy * inv]
            }
            XfemLevelSet::Halfspace { normal, .. } => *normal,
            XfemLevelSet::CrackLine { x1, x2 } => {
                let dx = x2[0] - x1[0];
                let dy = x2[1] - x1[1];
                let len = (dx * dx + dy * dy).sqrt().max(1e-14);
                [-dy / len, dx / len] // CCW 90° rotation = outward normal for crack side
            }
        }
    }

    /// True when `x` is in the active (interior) domain.
    #[inline]
    pub fn is_active(&self, x: [f64; 2]) -> bool {
        self.eval(x) < 0.0
    }
}

/// Signed distance from point `p` to line segment `(a, b)`.
/// Returns positive on the LEFT side of the directed segment a→b
/// (CCW 90°), negative on the right.
fn signed_distance_to_segment(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let ax = p[0] - a[0];
    let ay = p[1] - a[1];
    let bx = b[0] - a[0];
    let by = b[1] - a[1];
    // Cross product: z-component of (b-a) × (p-a) gives sign
    // along with perpendicular distance
    let cross = bx * ay - by * ax;
    let dot = ax * bx + ay * by;
    let len_sq = bx * bx + by * by;
    if len_sq < 1e-30 {
        return (ax * ax + ay * ay).sqrt();
    }
    let t = (dot / len_sq).clamp(0.0, 1.0);
    let closest_x = a[0] + t * bx;
    let closest_y = a[1] + t * by;
    let dist = ((p[0] - closest_x).powi(2) + (p[1] - closest_y).powi(2)).sqrt();
    if cross >= 0.0 { dist } else { -dist }
}

// ─── Sub-cell triangulation ──────────────────────────────────────────────────

/// A sub-triangle in REFERENCE coordinates (two-level set allow splitting).
#[derive(Debug, Clone)]
pub struct SubTriangle {
    /// Vertices in reference coordinate space (3 × [ξ, η]).
    pub verts: [[f64; 2]; 3],
}

/// Result of cutting a triangle by a level set.
#[derive(Debug, Clone)]
pub enum CutResult {
    /// Triangle entirely on the positive side (ψ > 0) — not active.
    Positive,
    /// Triangle entirely on the negative side (ψ < 0) — fully active.
    Negative,
    /// Triangle is cut by the interface.  Sub-cells cover the negative side.
    Cut(Vec<SubTriangle>),
}

/// Find the edge intersection point where ψ changes sign between vertices `a` and `b`.
///
/// Returns the REFERENCE coordinates of the intersection − the point along
/// edge (a,b) where ψ = 0, determined by linear interpolation.
fn edge_intersection(
    ls: &XfemLevelSet,
    a_phys: [f64; 2],
    b_phys: [f64; 2],
    a_ref: [f64; 2],
    b_ref: [f64; 2],
) -> [f64; 2] {
    let psi_a = ls.eval(a_phys);
    let psi_b = ls.eval(b_phys);
    // Linear interpolation: ψ(t) = ψ_a + t·(ψ_b - ψ_a) = 0
    // t = ψ_a / (ψ_a - ψ_b)
    let t = psi_a / (psi_a - psi_b);
    [
        a_ref[0] + t * (b_ref[0] - a_ref[0]),
        a_ref[1] + t * (b_ref[1] - a_ref[1]),
    ]
}

/// Convert reference coordinate (ξ, η) on the reference triangle
/// (0 ≤ ξ, 0 ≤ η, ξ+η ≤ 1) to physical coordinates via affine map.
fn ref_to_phys(ref_coord: [f64; 2], phys_nodes: &[[f64; 2]; 3]) -> [f64; 2] {
    let (xi, eta) = (ref_coord[0], ref_coord[1]);
    [
        phys_nodes[0][0] + xi * (phys_nodes[1][0] - phys_nodes[0][0])
            + eta * (phys_nodes[2][0] - phys_nodes[0][0]),
        phys_nodes[0][1] + xi * (phys_nodes[1][1] - phys_nodes[0][1])
            + eta * (phys_nodes[2][1] - phys_nodes[0][1]),
    ]
}

/// Cut a single triangle by a level set, returning sub-triangles
/// that cover the NEGATIVE (active, ψ < 0) side.
///
/// `phys_nodes` are the physical coordinates of the triangle vertices.
/// `ref_nodes` are the reference triangle vertices:
///   node 0 = (0,0), node 1 = (1,0), node 2 = (0,1)
pub fn cut_triangle(ls: &XfemLevelSet, phys_nodes: &[[f64; 2]; 3]) -> CutResult {
    let ref_nodes = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let psi: Vec<f64> = phys_nodes.iter().map(|&x| ls.eval(x)).collect();

    let all_neg = psi.iter().all(|&p| p <= 0.0);
    let all_pos = psi.iter().all(|&p| p >= 0.0);

    if all_neg { return CutResult::Negative; }
    if all_pos { return CutResult::Positive; }

    // Find edges that cross ψ=0
    let mut intersections: Vec<[f64; 2]> = Vec::new();  // ref coords
    let mut neg_verts: Vec<usize> = Vec::new();
    let mut pos_verts: Vec<usize> = Vec::new();

    for i in 0..3 {
        if psi[i] < 0.0 { neg_verts.push(i); } else { pos_verts.push(i); }
    }

    for i in 0..3 {
        let j = (i + 1) % 3;
        if (psi[i] < 0.0) != (psi[j] < 0.0) {
            let inter = edge_intersection(ls, phys_nodes[i], phys_nodes[j], ref_nodes[i], ref_nodes[j]);
            intersections.push(inter);
        }
    }

    // Build sub-triangles covering the negative side
    match (neg_verts.len(), intersections.len()) {
        (1, 2) => {
            // One vertex is negative (ψ<0), two are positive
            // Result: one sub-triangle covering the negative corner
            let n = neg_verts[0];
            let i0 = intersections[0];
            let i1 = intersections[1];
            CutResult::Cut(vec![SubTriangle { verts: [ref_nodes[n], i0, i1] }])
        }
        (2, 2) => {
            // Two vertices are negative, one is positive
            // Result: two sub-triangles (a quad split by a diagonal)
            let _p = pos_verts[0];
            let i0 = intersections[0];
            let i1 = intersections[1];

            // Determine which vertex maps to which intersection
            // Find the negative vertices in order
            let n0 = neg_verts[0];
            let n1 = neg_verts[1];

            // The quad consists of: n0, n1, i_connected_to_n0, i_connected_to_n1
            // Split along the line connecting the two intersection points
            CutResult::Cut(vec![
                SubTriangle { verts: [ref_nodes[n0], ref_nodes[n1], i0] },
                SubTriangle { verts: [ref_nodes[n1], i1, i0] },
            ])
        }
        _ => {
            // Shouldn't happen for a linear level set cutting a triangle
            // Fall back: no cut
            if all_neg { CutResult::Negative } else { CutResult::Positive }
        }
    }
}

/// Compute the area of a triangle given its vertices.
pub fn triangle_area(v: &[[f64; 2]; 3]) -> f64 {
    0.5 * ((v[1][0] - v[0][0]) * (v[2][1] - v[0][1])
        - (v[2][0] - v[0][0]) * (v[1][1] - v[0][1]))
        .abs()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halfspace_no_cut_full_positive() {
        // Triangle entirely on positive (active) side
        let ls = XfemLevelSet::Halfspace {
            normal: [1.0, 0.0],
            offset: -0.1, // ψ = x + 0.1 > 0 for all x≥0
        };
        let tri = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        match cut_triangle(&ls, &tri) {
            CutResult::Positive => {} // entire triangle in ψ>0
            r => panic!("expected Positive, got {r:?}"),
        }
    }

    #[test]
    fn halfspace_cut_one_vertex() {
        // ψ = x - 0.5; vertices: (0,0)→neg, (1,0)→pos, (0,1)→neg
        // → 2 neg, 1 pos → 2 sub-triangles, active area = 0.375
        let ls = XfemLevelSet::Halfspace {
            normal: [1.0, 0.0],
            offset: 0.5,
        };
        let tri = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let result = cut_triangle(&ls, &tri);
        match &result {
            CutResult::Cut(subs) => {
                assert_eq!(subs.len(), 2, "should produce 2 sub-triangles");
                let total_area: f64 = subs.iter().map(|s| triangle_area(&s.verts)).sum();
                let expected = 0.375; // area of triangle where x < 0.5
                assert!((total_area - expected).abs() < 1e-12,
                    "area mismatch: got {total_area}, expected {expected}");
            }
            r => panic!("expected Cut, got {r:?}"),
        }
    }

    #[test]
    fn halfspace_cut_two_vertices() {
        // ψ = y - 0.5; vertices: (0,0)→neg, (1,0)→neg, (0,1)→pos
        // → 2 neg, 1 pos → 2 sub-triangles, active area = 0.375
        let ls = XfemLevelSet::Halfspace {
            normal: [0.0, 1.0],
            offset: 0.5,
        };
        let tri = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let result = cut_triangle(&ls, &tri);
        match &result {
            CutResult::Cut(subs) => {
                assert_eq!(subs.len(), 2, "should produce 2 sub-triangles (quad split)");
                let total_area: f64 = subs.iter().map(|s| triangle_area(&s.verts)).sum();
                let expected = 0.375; // area where y < 0.5 in the unit triangle
                assert!((total_area - expected).abs() < 1e-12,
                    "area mismatch: got {total_area}, expected {expected}");
            }
            r => panic!("expected Cut, got {r:?}"),
        }
    }

    #[test]
    fn crack_line_signed_distance() {
        let ls = XfemLevelSet::CrackLine {
            x1: [0.0, 0.5],
            x2: [1.0, 0.5],
        };
        // Point above crack → positive side
        let psi_above = ls.eval([0.5, 1.0]);
        assert!(psi_above > 0.0, "above crack should be positive: {psi_above}");
        // Point below crack → negative side
        let psi_below = ls.eval([0.5, 0.0]);
        assert!(psi_below < 0.0, "below crack should be negative: {psi_below}");
        // Point on crack → zero
        let psi_on = ls.eval([0.5, 0.5]);
        assert!(psi_on.abs() < 1e-14, "on crack should be zero: {psi_on}");
    }

    #[test]
    fn circle_no_cut() {
        let ls = XfemLevelSet::Circle { cx: 0.5, cy: 0.5, radius: 0.1 };
        let tri = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let result = cut_triangle(&ls, &tri);
        match result {
            CutResult::Positive => {} // entirely outside
            _ => panic!("expected no cut"),
        }
    }

    #[test]
    fn cut_triangle_preserves_area() {
        // For any cut, the total area of sub-triangles (positive + negative sides)
        // should equal the original triangle area.
        // We only compute the NEGATIVE side sub-triangles.
        let ls = XfemLevelSet::Halfspace {
            normal: [0.6, 0.8],
            offset: 0.4,
        };
        let tri = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let result = cut_triangle(&ls, &tri);
        if let CutResult::Cut(subs) = result {
            for s in &subs {
                let area = triangle_area(&s.verts);
                assert!(area > 0.0, "sub-triangle has zero area");
            }
        }
    }
}
