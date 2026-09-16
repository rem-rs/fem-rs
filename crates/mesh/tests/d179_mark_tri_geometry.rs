//! D179: `mark_tri_mesh_for_refinement` must permute a curved mesh's
//! **geometry-table slots together with the connectivity**.
//!
//! MFEM 4.10 loads a curved 2-D triangle mesh with refine=1 through
//! `Mesh::Finalize` (`mesh/mesh.cpp:3766`): `PrepareNodeReorder` →
//! `MarkForRefinement()` (rotate each triangle so its longest edge is local
//! edge (0,1); lengths from the vertices — for a curved mesh
//! `SetVerticesFromNodes`) → `DoNodeReorder` (`mesh/mesh.cpp:3255`), which
//! moves every `nodes` grid-function value into the rotated element-local dof
//! slot.  The fem-rs 1:1 counterpart stores the geometry dof ids in an
//! explicit per-element table ([`GeometryData::conn`]), so rotating the
//! connectivity without permuting those slots leaves the geometry field
//! expressed in the *old* local frame: every curved tri mesh that MFEM marks
//! comes out desynchronised.
//!
//! The fixtures `d179_antidiag_p{2,4}.mesh` (generator `tmp/d242/d179_gen.cpp`,
//! MFEM 4.10) are 3×2 **anti-diagonal** triangle grids promoted to `H1_2D_P`
//! geometry with `SetCurvature(p, ...)` — unlike MFEM's MakeCartesian2D
//! main-diagonal split, every triangle's longest edge is not local edge (0,1),
//! so the mark genuinely rotates all 12 elements (MFEM reports
//! `mark rotated 12`).  Order 2 exercises the vertex + edge slots, order 4
//! additionally the 3 interior slots of the H1 lattice.  The references
//! `d179_antidiag_p{2,4}_r1.mesh` are MFEM's own refine=1 load +
//! `UniformRefinement` + 17-digit `Save`.
//!
//! Acceptance: mark → refine → write must reproduce MFEM's file dof-for-dof
//! (the writer walks MFEM's `H1_2D_P` dof numbering — the D178-pinned slot
//! maps), and the straight-tri paths keep their behaviour unchanged.

use fem_io::mfem::{read_mfem, read_mfem_file, write_mfem_nodes, NodesSpace};
use fem_mesh::amr::mark_tri_mesh_for_refinement;
use fem_mesh::{refine_uniform, Mesh};
use fem_mesh::topology::MeshTopology;

/// Quantum of the straight-grid arithmetic differences between MFEM's
/// interpolation matrices and fem-rs' dof picks is ≪ 1e-12 (both sides land
/// within a few ulp); a desynced slot is wrong at O(0.1).
const TOL: f64 = 1e-12;

const PARENT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/d179_antidiag_p2.mesh");

/// Whole-file token comparison (numbers to relative `tol`).
fn compare_mesh(ours: &str, theirs: &str, tol: f64, label: &str) {
    let tok = |t: &str| -> Vec<Vec<String>> {
        t.lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|l| l.split_whitespace().map(str::to_string).collect())
            .collect()
    };
    let (a, b) = (tok(ours), tok(theirs));
    assert_eq!(a.len(), b.len(), "{label}: line count {} vs {}", a.len(), b.len());
    for (i, (ra, rb)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(ra.len(), rb.len(), "{label}: line {i} token count");
        for (ta, tb) in ra.iter().zip(rb.iter()) {
            if ta == tb {
                continue;
            }
            let (x, y) = (
                ta.parse::<f64>().unwrap_or_else(|_| panic!("{label}: line {i} '{ta}'")),
                tb.parse::<f64>().unwrap_or_else(|_| panic!("{label}: line {i} '{tb}'")),
            );
            assert!(
                (x - y).abs() <= tol * (1.0 + x.abs().max(y.abs())),
                "{label}: line {i}: {ta} vs {tb}"
            );
        }
    }
}

// ─── tests ──────────────────────────────────────────────────────────────────

/// The mark must leave the mesh's geometry table in sync with the rotated
/// connectivity: for every element the geometry table's vertex slots are the
/// rotated vertex list (MFEM `DoNodeReorder`).
#[test]
fn d179_mark_keeps_geometry_slots_in_sync() {
    let f = read_mfem_file(PARENT).expect("read fixture");
    let mut mesh: Mesh<2> = f.mesh2d.expect("2-D mesh");
    assert_eq!(mesh.n_elems(), 12);
    assert_eq!(mesh.geom_order(), 2, "fixture carries H1_2D_P2 geometry");

    // The fixture's triangles all have their longest edge off slot (0,1):
    // before the mark, geometry vertex slots match the connectivity.
    let mut misaligned = 0usize;
    for e in 0..mesh.n_elems() as u32 {
        let top = mesh.elem_nodes(e).to_vec();
        let geo = mesh.geometry_nodes(e)[..3].to_vec();
        if top != geo {
            misaligned += 1;
        }
    }
    assert_eq!(misaligned, 0, "fixture loads with consistent geometry slots");

    mark_tri_mesh_for_refinement(&mut mesh);

    let mut desynced = 0usize;
    for e in 0..mesh.n_elems() as u32 {
        let top = mesh.elem_nodes(e).to_vec();
        let geo = mesh.geometry_nodes(e)[..3].to_vec();
        if top != geo {
            desynced += 1;
        }
    }
    assert_eq!(
        desynced, 0,
        "after mark, {desynced} of 12 elements have geometry vertex slots \
         out of sync with the rotated connectivity"
    );

    // The geometry VALUES must survive the permutation: for each element the
    // vertex-slot coordinates read through the geometry table equal the
    // topology vertex coordinates.
    for e in 0..mesh.n_elems() as u32 {
        let geo = mesh.geometry_nodes(e);
        for (s, &gn) in geo[..3].iter().enumerate() {
            let gx = mesh.geom_coords_of(gn);
            let tx = mesh.node_coords(mesh.elem_nodes(e)[s]);
            assert!(
                (gx[0] - tx[0]).abs() <= TOL && (gx[1] - tx[1]).abs() <= TOL,
                "elem {e} slot {s}: geometry value moved with the wrong slot"
            );
        }
    }
}

/// End to end: mark → refine → write must reproduce MFEM's refine=1 load +
/// UniformRefinement + Save dof-for-dof, at order 2 (vertex + edge slots) and
/// order 4 (adds the 3 interior slots of the H1 lattice).
#[test]
fn d179_mark_refine_write_matches_mfem() {
    for (order, dpe) in [(2usize, 6usize), (4, 15)] {
        let parent =
            format!("{}/tests/data/d179_antidiag_p{order}.mesh", env!("CARGO_MANIFEST_DIR"));
        let refined =
            format!("{}/tests/data/d179_antidiag_p{order}_r1.mesh", env!("CARGO_MANIFEST_DIR"));
        let f = read_mfem_file(&parent).unwrap_or_else(|e| panic!("{parent}: {e}"));
        let mut mesh: Mesh<2> = f.mesh2d.expect("2-D mesh");
        assert_eq!(mesh.geom_order(), order as u8);
        mark_tri_mesh_for_refinement(&mut mesh);

        let fine = refine_uniform(&mesh);
        assert_eq!(fine.n_elems(), 48, "12 tris × 4 children");
        let g = fine.geometry.as_ref().expect("refined mesh keeps its geometry");
        assert_eq!(g.order, order as u8);
        assert_eq!(g.nodes_per_elem, dpe);

        // Whole-file write (the "写→读回" acceptance): the writer walks
        // MFEM's `H1_2D_P` dof numbering, so token equality with MFEM's own
        // 17-digit Save pins every element, boundary and `nodes` dof value.
        let mut buf: Vec<u8> = Vec::new();
        write_mfem_nodes(&mut buf, &fine, None, NodesSpace::Continuous).expect("write");
        let ours = String::from_utf8(buf).expect("utf-8");
        let theirs = std::fs::read_to_string(&refined).expect("read reference");
        compare_mesh(&ours, &theirs, TOL, &format!("d179 p{order} whole-file"));

        // 读回: our own file must round-trip through the reader with identical
        // connectivity and a geometry field that is still the (straight-sided
        // fixture's) affine map — evaluated at non-dof interior points, which
        // pins the per-element *slot placement* of the read-back dofs without
        // depending on the reader's own dof-id ordering.
        let back = read_mfem(ours.as_bytes())
            .expect("read back the written mesh")
            .mesh2d
            .expect("2-D mesh");
        assert_eq!(back.n_elems(), 48);
        assert_eq!(back.geom_order(), order as u8);
        for e in 0..48u32 {
            assert_eq!(back.elem_nodes(e), fine.elem_nodes(e), "round-trip elem {e}");
        }
        for m in [&fine, &back] {
            let mut max_dev = 0.0_f64;
            for e in 0..48u32 {
                let ns = m.elem_nodes(e);
                for &[a, b] in &[[0.3_f64, 0.25], [0.2, 0.6], [0.45, 0.1]] {
                    let (_, _, xp) = m.element_jacobian(e, &[a, b]);
                    let lam = [1.0 - a - b, a, b];
                    let mut exact = [0.0_f64; 2];
                    for (k, &l) in lam.iter().enumerate() {
                        let c = m.node_coords(ns[k]);
                        exact[0] += l * c[0];
                        exact[1] += l * c[1];
                    }
                    max_dev = max_dev.max(((xp[0] - exact[0]).abs()).max((xp[1] - exact[1]).abs()));
                }
            }
            eprintln!("d179 p{order}: geometry affine deviation {max_dev:.3e}");
            assert!(
                max_dev <= 1e-12,
                "round-trip geometry deviates from the affine map by {max_dev:.3e}"
            );
        }
    }
}

/// Straight-tri regression: refining a mesh **without** curved geometry keeps
/// the mark + refine behaviour of the linear path (ids unchanged).
#[test]
fn d179_straight_tri_mark_refine_ids_unchanged() {
    // Anti-diagonal straight grid, same split as the fixture (built by hand:
    // 3×2 cells, vertices j*(nx+1)+i).
    let (nx, ny) = (3usize, 2usize);
    let mut coords = Vec::new();
    for j in 0..=ny {
        for i in 0..=nx {
            coords.push(i as f64 / nx as f64);
            coords.push(j as f64 / ny as f64);
        }
    }
    let vid = |i: usize, j: usize| (j * (nx + 1) + i) as u32;
    let mut conn = Vec::new();
    let mut tags = Vec::new();
    for j in 0..ny {
        for i in 0..nx {
            conn.extend([vid(i, j), vid(i + 1, j), vid(i, j + 1)]);
            tags.push(1);
            conn.extend([vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)]);
            tags.push(1);
        }
    }
    let mut mesh = Mesh::uniform(
        coords, conn, tags, fem_mesh::ElementType::Tri3,
        Vec::new(), Vec::new(), fem_mesh::ElementType::Line2,
    );
    let before: Vec<Vec<u32>> = (0..mesh.n_elems() as u32)
        .map(|e| mesh.elem_nodes(e).to_vec())
        .collect();
    mark_tri_mesh_for_refinement(&mut mesh);
    let mut rotated = 0usize;
    for e in 0..mesh.n_elems() as u32 {
        if mesh.elem_nodes(e) != before[e as usize].as_slice() {
            rotated += 1;
        }
    }
    assert_eq!(rotated, 12, "all 12 anti-diagonal triangles rotate");
    let fine = refine_uniform(&mesh);
    assert_eq!(fine.n_elems(), 48);
    assert!(fine.geometry.is_none(), "straight mesh gains no geometry");
}

