//! Acceptance tests for uniform refinement of the *curved* hex mesh
//! `data/multidomain-hex.mesh` (MFEM `nodes` H1_3D_P2 geometry).
//!
//! Reference: serial MFEM 4.9 `Mesh::UniformRefinement()` on the same file,
//! dumping every refined child hex's 8 corner coordinates at `%.17e`
//! (sorted lexicographically within each element). Regenerate with:
//!
//! ```text
//! wsl g++ -std=c++17 -O2 -I$HOME/mfem49 hexdump17.cpp $HOME/mfem49/libmfem.a \
//!     -o hexdump17 && ./hexdump17 -m ../../data/multidomain-hex.mesh
//! ```
//!
//! (harness kept in `tmp/multidomain/hexdump17.cpp`; the fixture checked in
//! under `tests/data/` is its output).
//!
//! Root cause this guards against (defect D1): MFEM's `UniformRefinement`
//! on a curved mesh ends with `UpdateNodes()` → `SetVerticesFromNodes`, so
//! every fine vertex receives the coarse Qk geometry-dof value at its
//! reference point — plain straight averaging matches only on affine
//! regions (96/480 children on this mesh).
//!
//! Tolerance: MFEM's refinement operator evaluates the coarse nodal basis
//! through dense-matrix arithmetic, so its fine vertex coordinates carry
//! ~1 ulp noise (~1e-17 here) relative to the exact dof picks. The
//! acceptance criterion (≤1e-15 per coordinate) is checked with greedy
//! nearest-corner matching per element; the rust picks themselves are
//! exact, and most coordinates match the dump bit-for-bit.

use fem_io::mfem::read_mfem_file;
use fem_mesh::{refine_hex8_uniform, refine_uniform_3d, Mesh};

const PARENT_MESH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/multidomain-hex.mesh");
const CPP_DUMP: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/multidomain_hex_refined_cpp.txt");
const TOL: f64 = 1e-15;

/// Load the curved parent mesh (60 Hex8, order-2 geometry).
fn load_parent() -> Mesh<3> {
    let f = read_mfem_file(PARENT_MESH).expect("read parent mesh");
    let mesh: Mesh<3> = f.mesh3d.expect("3D mesh");
    assert_eq!(mesh.n_elems(), 60);
    assert!(mesh.geometry.is_some(), "parent must carry curved geometry");
    mesh
}

/// Parse the MFEM reference dump: one line per child hex, 24 floats
/// (8 corners × xyz), corners sorted lexicographically within the line.
fn load_cpp_dump() -> Vec<Vec<[f64; 3]>> {
    let text = std::fs::read_to_string(CPP_DUMP).expect("read reference dump");
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let v: Vec<f64> = line.split_whitespace().map(|t| t.parse().expect("float")).collect();
            assert_eq!(v.len(), 24, "reference row must hold 8 corners");
            (0..8)
                .map(|k| [v[3 * k], v[3 * k + 1], v[3 * k + 2]])
                .collect()
        })
        .collect()
}

/// Corner triples of fine element `e`.
fn corners(mesh: &Mesh<3>, e: u32) -> Vec<[f64; 3]> {
    mesh.elem_nodes(e)
        .iter()
        .map(|&n| {
            let c = mesh.coords_of(n);
            [c[0], c[1], c[2]]
        })
        .collect()
}

/// Compare all refined children against the MFEM reference dump and return
/// (coordinate-exact-match count, max |Δ|). Corners are matched by greedy
/// nearest pairing per element (MFEM's ulp noise can flip lexicographic
/// sort ties, so positional matching of sorted lists is not robust).
fn compare_with_dump(fine: &Mesh<3>, cpp: &[Vec<[f64; 3]>]) -> (usize, f64) {
    assert_eq!(fine.n_elems() as usize, cpp.len(), "child count must match reference");
    let mut exact = 0usize;
    let mut max_diff = 0.0_f64;
    for e in 0..fine.n_elems() {
        let got = corners(fine, e as u32);
        let want = &cpp[e as usize];
        let mut used = [false; 8];
        for g in &got {
            let mut best: Option<(usize, f64)> = None;
            for (k, (&u, &w)) in used.iter().zip(want.iter()).enumerate() {
                if u { continue; }
                let d = (g[0] - w[0]).abs().max((g[1] - w[1]).abs()).max((g[2] - w[2]).abs());
                if best.map_or(true, |(_, bd)| d < bd) {
                    best = Some((k, d));
                }
            }
            let (k, d) = best.expect("more rust corners than reference corners");
            assert!(
                d <= TOL,
                "child {e}: no reference corner within {TOL:e} of {g:?} (best Δ={d:e})"
            );
            used[k] = true;
            if d == 0.0 { exact += 3; }
            max_diff = max_diff.max(d);
        }
    }
    (exact, max_diff)
}

#[test]
fn refine_hex8_uniform_matches_mfem_reference() {
    let parent = load_parent();
    let all: Vec<u32> = (0..parent.n_elems() as u32).collect();
    let (fine, _, _) = refine_hex8_uniform(&parent, &all);
    let cpp = load_cpp_dump();
    let (exact, max_diff) = compare_with_dump(&fine, &cpp);
    // 480/480 children, all 8 corners within 1e-15 of MFEM.
    assert_eq!(fine.n_elems() as usize, 480);
    assert!(max_diff <= TOL, "max coordinate deviation {max_diff:e} exceeds {TOL:e}");
    eprintln!("multidomain-hex refinement: 480/480 children match; \
               exact coordinates {exact}/{}; max |Δ| = {max_diff:e}", 480 * 24);
}

#[test]
fn refine_uniform_3d_handles_curved_hex_with_quad_boundary() {
    // Before the fix this panicked: rebuild_3d_boundary looked up the new
    // quad-face centers by exact-bit coordinate match computed with a
    // different accumulation order (and, for curved geometry, with values
    // that are not recomputable averages at all). Now the hex path builds
    // the boundary topologically from the refinement's own maps.
    let parent = load_parent();
    let fine = refine_uniform_3d(&parent);
    let cpp = load_cpp_dump();
    let (_, max_diff) = compare_with_dump(&fine, &cpp);
    assert!(max_diff <= TOL);
    // Boundary: every quad boundary face is split into 4 (MFEM order).
    assert_eq!(fine.n_faces(), 4 * parent.n_faces());
}

#[test]
fn curved_geometry_is_refined_and_survives_repeated_refinement() {
    let parent = load_parent();
    let all: Vec<u32> = (0..parent.n_elems() as u32).collect();
    let (fine, _, _) = refine_hex8_uniform(&parent, &all);

    let geo = fine.geometry.as_ref().expect("refined mesh must carry geometry");
    assert_eq!(geo.order, 2);
    assert_eq!(geo.nodes_per_elem, 27);
    assert_eq!(geo.conn.len(), fine.n_elems() as usize * 27);
    assert!(geo.coords.iter().all(|v| v.is_finite()));

    // Second refinement: the child geometry must again refine consistently.
    let all2: Vec<u32> = (0..fine.n_elems() as u32).collect();
    let (fine2, _, _) = refine_hex8_uniform(&fine, &all2);
    assert_eq!(fine2.n_elems(), 8 * fine.n_elems());
    let geo2 = fine2.geometry.as_ref().expect("level-2 geometry");
    assert_eq!(geo2.conn.len(), fine2.n_elems() as usize * 27);
    assert_eq!(geo2.n_nodes, geo2.coords.len() / 3);
}

#[test]
fn partial_refinement_keeps_straight_neighbors_consistent() {
    // Mark only element 0; its neighbors stay unrefined (identity children
    // in the geometry builder) and the mesh must remain valid with hanging
    // constraints on the split edges.
    let parent = load_parent();
    let (fine, constraints, _) = refine_hex8_uniform(&parent, &[0]);
    assert_eq!(fine.n_elems(), parent.n_elems() + 7);
    assert!(fine.geometry.is_some());
    // The 12 edges of element 0 are split → hanging midpoints.
    assert!(!constraints.is_empty());
}

// ─── D111: the refined Q2 geometry must be the parent's Q2 field ────────────
//
// Regression pins for defect D111: `refine_uniform_3d` on any order-2 hex mesh
// used to scramble the *interior* fine geometry dofs (MFEM `Geometry::
// Constants<CUBE>` vertex order was confused with the bitwise
// `(v&1, (v>>1)&1, (v>>2)&1)` encoding, both in the reference-point table of
// `q2_eval` and in the child-octant origin of `build_refined_hex_geometry`).
// The child geometry then no longer reproduced the parent's field:
// `data/cube.mesh -o 2 -rs 1` (an *affine* Q2 mesh!) had min det(J) = −1.32
// instead of +1.953125e-3, which is what `mesh-optimizer` reported as
// "The input mesh is inverted!".  The fine vertex coordinates were always
// correct (they are exact geometry-dof picks), which is why the corner-only
// checks above passed while the interior dofs were wrong.

/// MFEM `Geometry::Constants<CUBE>` vertex order, `[0,1]³`.
const MFEM_HEX_VERTS: [[f64; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
];

/// Local index of the parent's body-center vertex inside child `k` (see the
/// child table in `refine_nonconforming_hex`).
const CHILD_BC_LOCAL: [usize; 8] = [6, 7, 4, 5, 2, 3, 0, 1];

/// Sample points inside the reference hex (`[-1,1]³`): the 27 Q2 dof points
/// plus the 8 sub-octant centers.
fn ref_samples() -> Vec<[f64; 3]> {
    let mut v = Vec::new();
    for k in -1..=1 {
        for j in -1..=1 {
            for i in -1..=1 {
                v.push([i as f64, j as f64, k as f64]);
            }
        }
    }
    for k in [-0.5, 0.5] {
        for j in [-0.5, 0.5] {
            for i in [-0.5, 0.5] {
                v.push([i, j, k]);
            }
        }
    }
    v
}

/// Every refined child's Q2 geometry must reproduce the parent element's own
/// Q2 field on the child's reference domain (a degree-2 field restricted to a
/// half-cell is still degree 2, so this is exact, not an approximation).
///
/// The child→parent correspondence is recovered from the mesh alone: a child
/// has exactly one coarse-vertex corner, whose local index *is* the child
/// index, and its body-center corner is a node shared by exactly the 8
/// children of one parent whose vertex set is the children's corner set.
fn check_child_geometry_reproduces_parent(parent: &Mesh<3>, fine: &Mesh<3>) {
    let n_coarse = parent.n_nodes() as u32;
    let mut groups: std::collections::HashMap<u32, Vec<(usize, u32)>> =
        std::collections::HashMap::new();
    for fe in 0..fine.n_elems() as u32 {
        let vs = fine.elem_nodes(fe);
        let own: Vec<usize> = (0..8).filter(|&k| vs[k] < n_coarse).collect();
        assert_eq!(own.len(), 1, "child {fe} must have exactly one parent-vertex corner");
        let child = own[0];
        groups.entry(vs[CHILD_BC_LOCAL[child]]).or_default().push((child, fe));
    }
    assert_eq!(groups.len(), parent.n_elems(), "one child group per parent");

    let samples = ref_samples();
    let mut max_dev = 0.0_f64;
    for (_bc, kids) in &groups {
        assert_eq!(kids.len(), 8, "8 children per parent");
        // The children's own corners are the parent's 8 vertices.
        let mut corners: Vec<u32> = kids
            .iter()
            .map(|&(child, fe)| fine.elem_nodes(fe)[child])
            .collect();
        corners.sort_unstable();
        let pe = (0..parent.n_elems() as u32)
            .find(|&p| {
                let mut pv = parent.elem_nodes(p).to_vec();
                pv.sort_unstable();
                pv == corners
            })
            .expect("parent element with this vertex set");
        for &(child, fe) in kids {
            let c = MFEM_HEX_VERTS[child];
            for xi in &samples {
                let parent_xi = [
                    c[0] - 0.5 + 0.5 * xi[0],
                    c[1] - 0.5 + 0.5 * xi[1],
                    c[2] - 0.5 + 0.5 * xi[2],
                ];
                let (_, _, x_child) = fine.element_jacobian(fe, xi);
                let (_, _, x_parent) = parent.element_jacobian(pe, &parent_xi);
                for d in 0..3 {
                    max_dev = max_dev.max((x_child[d] - x_parent[d]).abs());
                }
            }
        }
    }
    assert!(
        max_dev <= 1e-15,
        "refined Q2 geometry deviates from the parent field by {max_dev:e}"
    );
    eprintln!("D111: child geometry reproduces the parent Q2 field, max |Δ| = {max_dev:e}");
}

#[test]
fn d111_refined_curved_hex_geometry_reproduces_parent_field() {
    let parent = load_parent();
    let all: Vec<u32> = (0..parent.n_elems() as u32).collect();
    let (fine, _, _) = refine_hex8_uniform(&parent, &all);
    check_child_geometry_reproduces_parent(&parent, &fine);
    // Second level: the fine mesh is itself a valid order-2 parent now.
    let all2: Vec<u32> = (0..fine.n_elems() as u32).collect();
    let (fine2, _, _) = refine_hex8_uniform(&fine, &all2);
    check_child_geometry_reproduces_parent(&fine, &fine2);
}

#[test]
fn d111_affine_q2_cube_refines_to_exact_trilinear_children() {
    // `data/cube.mesh` is an affine 8-hex Q2 mesh of the unit cube: every
    // refined child's geometry must be *exactly* the trilinear (here affine)
    // map of its 8 vertices, and min det(J) must be > 0 at every level.
    let cube = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/cube.mesh");
    let f = read_mfem_file(cube).expect("read cube.mesh");
    let mut m: Mesh<3> = f.mesh3d.expect("3d");
    assert_eq!(m.geom_order(), 2);
    let samples = ref_samples();
    for (level, expected_min_det) in [(1usize, 1.953125e-3_f64), (2, 2.44140625e-4)] {
        m = refine_uniform_3d(&m);
        let mut min_det = f64::INFINITY;
        let mut max_dev = 0.0_f64;
        for e in 0..m.n_elems() as u32 {
            let vs: Vec<[f64; 3]> = m.elem_nodes(e).iter().map(|&n| m.coords_of(n)).collect();
            for xi in &samples {
                let (_, det, xp) = m.element_jacobian(e, xi);
                min_det = min_det.min(det);
                // trilinear map of the 8 vertices in MFEM order on [-1,1]³
                let mut want = [0.0_f64; 3];
                for k in 0..8 {
                    let r = [
                        2.0 * MFEM_HEX_VERTS[k][0] - 1.0,
                        2.0 * MFEM_HEX_VERTS[k][1] - 1.0,
                        2.0 * MFEM_HEX_VERTS[k][2] - 1.0,
                    ];
                    let w = 0.125
                        * (1.0 + r[0] * xi[0])
                        * (1.0 + r[1] * xi[1])
                        * (1.0 + r[2] * xi[2]);
                    for d in 0..3 {
                        want[d] += w * vs[k][d];
                    }
                }
                for d in 0..3 {
                    max_dev = max_dev.max((xp[d] - want[d]).abs());
                }
            }
        }
        assert!(min_det > 0.0, "level {level}: min det(J) = {min_det:e} (mesh inverted)");
        assert!(
            (min_det - expected_min_det).abs() < 1e-15,
            "level {level}: min det(J) = {min_det:e}, expected {expected_min_det:e}"
        );
        assert!(max_dev <= 1e-15, "level {level}: geometry is not trilinear (Δ = {max_dev:e})");
    }
}
