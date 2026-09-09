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
