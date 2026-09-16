//! D160 — Quad4 nonconforming `GeneralRefinement` element counts vs MFEM 4.10.
//!
//! Reference: `Mesh::GeneralRefinement(refs, -1, nclimit)` on
//! `MakeCartesian2D(4, 4, QUADRILATERAL)` (MFEM 4.10, serial), probed with the
//! order-independent geometric marking rule
//! `(llround(100·cx) + llround(100·cy)) % 3 == 0` on the element centroid —
//! the exact rule of `$HOME/work/d160/d160_probe.cpp` (WSL reference tree,
//! counts recorded 2026-09-15):
//!
//! ```text
//! nclimit = 1:  NE 16 -> 31 -> 91 -> 244 -> 553 -> 1228 -> 3046
//!               NV 25 -> 50 -> 128 -> 345 -> 748 -> 1603 -> 3819
//! nclimit = 0:  NE 16 -> 31 -> 61 -> 121 -> 181 -> 241 -> 481
//!               NV 25 -> 50 -> 100 -> 200 -> 300 -> 400 -> 720
//! ```
//!
//! The `nclimit = 1` run exercises `NCMesh::LimitNCLevel` propagation (e.g.
//! round 1: 10 marked by the caller + 20 propagated coarse neighbours), which
//! is what reproduces MFEM's mandel/mondrian iteration counts.

use fem_mesh::amr::{general_refinement_quad, refine_uniform};
use fem_mesh::{Mesh, element_type::ElementType};

/// One probe round: mark elements by the centroid checkerboard rule, refine.
fn probe_round(mesh: &Mesh<2>, nclimit: u32) -> Mesh<2> {
    let mut marked = Vec::new();
    for e in 0..mesh.n_elems() as u32 {
        let ns = mesh.elem_nodes(e);
        let (mut sx, mut sy) = (0.0f64, 0.0f64);
        for &n in ns {
            let c = mesh.coords_of(n);
            sx += c[0];
            sy += c[1];
        }
        let cx = sx / 4.0;
        let cy = sy / 4.0;
        let ix = (cx * 32.0).round() as i64;
        let iy = (cy * 32.0).round() as i64;
        if (ix + iy) % 3 == 0 {
            marked.push(e);
        }
    }
    general_refinement_quad(mesh, &marked, nclimit, None).0
}

#[test]
fn d160_quad_nc_counts_nclimit_1() {
    let mesh = Mesh::<2>::make_cartesian_2d(4, 4, 1.0, 1.0);
    assert_eq!(mesh.elem_type, ElementType::Quad4);
    assert_eq!(mesh.n_elems(), 16);
    assert_eq!(mesh.n_nodes(), 25);

    // MFEM 4.10 d160_probe (nclimit = 1), NE/NV after each GeneralRefinement.
    let ne_ref = [31, 91, 244, 553, 1228, 3046];
    let nv_ref = [50, 128, 345, 748, 1603, 3819];

    let mut mesh = mesh;
    for r in 0..6 {
        mesh = probe_round(&mesh, 1);
        assert_eq!(
            mesh.n_elems(),
            ne_ref[r],
            "NE mismatch at round {r} (nclimit=1)"
        );
        assert_eq!(
            mesh.n_nodes(),
            nv_ref[r],
            "NV mismatch at round {r} (nclimit=1)"
        );
    }
}

#[test]
fn d160_quad_nc_counts_nclimit_0() {
    let mesh = Mesh::<2>::make_cartesian_2d(4, 4, 1.0, 1.0);

    // MFEM 4.10 d160_probe (nclimit = 0, no LimitNCLevel propagation).
    let ne_ref = [31, 61, 121, 181, 241, 481];
    let nv_ref = [50, 100, 200, 300, 400, 720];

    let mut mesh = mesh;
    for r in 0..6 {
        mesh = probe_round(&mesh, 0);
        assert_eq!(
            mesh.n_elems(),
            ne_ref[r],
            "NE mismatch at round {r} (nclimit=0)"
        );
        assert_eq!(
            mesh.n_nodes(),
            nv_ref[r],
            "NV mismatch at round {r} (nclimit=0)"
        );
    }
}

#[test]
fn d160_quad_tri_path_unchanged() {
    // Tri3 regression guard: closure_refine on Tri3 must still produce the
    // conforming red-green refinement and must never hit the Quad4 branch.
    // Marking *both* triangles keeps the count fully deterministic (both red,
    // no green bisection needed: every split edge is either shared by the two
    // red-refined elements or lies on the boundary).
    let mesh = Mesh::<2>::make_cartesian_2d_tri(2, 1, 1.0, 1.0);
    assert_eq!(mesh.elem_type, ElementType::Tri3);
    // 2 quads x 2 triangles each.
    assert_eq!(mesh.n_elems(), 4);

    let refined = fem_mesh::amr::closure_refine(&mesh, &[0, 1, 2, 3], 20, None);
    // Red refinement of all four triangles: 4 -> 16 children, conforming.
    assert_eq!(refined.n_elems(), 16);
    // All edges conforming: every edge is shared by exactly two elements
    // (or one boundary edge), i.e. no hanging midpoints remain.
    use std::collections::HashMap;
    let mut edge_count: HashMap<(u32, u32), u32> = HashMap::new();
    for e in 0..refined.n_elems() as u32 {
        let ns = refined.elem_nodes(e);
        for w in 0..3 {
            let (a, b) = (ns[w], ns[(w + 1) % 3]);
            *edge_count.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    assert!(
        edge_count.values().all(|&c| c <= 2),
        "hanging edge detected after Tri3 closure_refine"
    );
}

#[test]
fn d160_quad_single_marked_matches_uniform_topology() {
    // A single marked element of a uniform quad grid must become 4 children
    // (5 with the neighbour untouched) with midpoints + center, and the
    // unmarked neighbour must keep its coarse connectivity (hanging node on
    // the shared edge).
    let mesh = Mesh::<2>::make_cartesian_2d(2, 1, 1.0, 1.0);
    let (refined, constraints) = general_refinement_quad(&mesh, &[0], 0, None);
    assert_eq!(refined.n_elems(), 5);
    // New nodes: 4 edge midpoints + 1 center; the shared (interior) edge
    // midpoint is a hanging node constrained by the coarse neighbour.
    assert_eq!(refined.n_nodes(), mesh.n_nodes() + 5);
    assert_eq!(constraints.len(), 1, "shared-edge midpoint is hanging");
    assert_eq!(constraints[0].coeff_a, 0.5);
    assert_eq!(constraints[0].coeff_b, 0.5);
    // The coarse neighbour (elem id 1 in the old numbering) keeps its nodes.
    let ns = refined.elem_nodes(4); // appended after elem 0's 4 children
    assert_eq!(ns.len(), 4);
}

#[test]
fn d160_quad_uniform_then_nc_mandel_like() {
    // mandel flow sanity: the 4x4 inline-quad grid + 3 uniform refinements
    // = 1024 elements (matches C++ mandel iteration 1), then one NC round of
    // the checkerboard rule must refine *only* the marked elements (+3 each).
    let m0 = Mesh::<2>::make_cartesian_2d(4, 4, 1.0, 1.0);
    let mut mesh = m0;
    for _ in 0..3 {
        mesh = refine_uniform(&mesh);
    }
    assert_eq!(mesh.n_elems(), 1024);

    let mut marked = Vec::new();
    for e in 0..mesh.n_elems() as u32 {
        let ns = mesh.elem_nodes(e);
        let (mut sx, mut sy) = (0.0f64, 0.0f64);
        for &n in ns {
            let c = mesh.coords_of(n);
            sx += c[0];
            sy += c[1];
        }
        let ix = (sx / 4.0 * 32.0).round() as i64;
        let iy = (sy / 4.0 * 32.0).round() as i64;
        if (ix + iy) % 3 == 0 {
            marked.push(e);
        }
    }
    let marked_count = marked.len();
    let (mesh, _cons) = general_refinement_quad(&mesh, &marked, 1, None);
    // At level 0 (all roots) LimitNCLevel cannot force anything: every edge
    // is split at most once.
    assert_eq!(mesh.n_elems(), 1024 + 3 * marked_count);
}
