//! D446 — `build_prolongation_hdiv` must walk HDiv face blocks by the face's
//! **shape**, not by a global triangular stride.
//!
//! Debt: `hdiv_face_dofs_per_face(dim, order)` sized every 3-D face block
//! `(k+1)(k+2)/2` (the triangular size) and the 3-D walk enumerated only the
//! tet's local face table.  After D377/D393 made `HDivSpace::face_dofs` size
//! blocks by shape — a quadrilateral face carries `(k+1)^2` — the
//! prolongation on any mesh with quad faces (hex/prism/pyramid/mixed)
//! mis-walked: on a pure hex hierarchy the tet-tuple `FaceKey`s degenerate to
//! one accidental bottom-face hit per element, the sub-face fallback never
//! matches, and the returned P is **empty** even though every coarse face dof
//! has fine children (RT1 quad faces: 4 dofs, not 3).
//!
//! Fix (same shape rule as `HDivSpace`): enumerate each element's faces from
//! its element type (3-entry slice = tri face, 4-entry = quad), build the
//! quad `FaceKey` as the sorted first 3 of the 4 verts (the HDivSpace
//! builders' rule), and take the per-face block size from the face's vertex
//! count.  Sub-face matching extends by the coarse face's own shape (tri:
//! edge midpoints; quad: edge midpoints + face center).
//!
//! The tests re-derive the expected structure from the meshes alone and
//! assert, for every fine face that is a coarse face or one of its
//! uniform-refinement sub-faces:
//!   1. every dof of its `face_dofs` block carries exactly one P entry, in
//!      the same block offset, with value `ratio × visits` (ratio 1.0 for an
//!      unrefined face, the 0.25 area ratio for a sub-face; `visits` = the
//!      number of fine elements sharing the face — P is piecewise identity
//!      per face block);
//!   2. every coarse face dof is claimed by at least one entry (Pᵀ covers
//!      all face blocks), while coarse interior dofs receive none (the
//!      documented higher-order approximation);
//!   3. every nonzero P row's support lies inside a single coarse element's
//!      dof set (Pᵀ single-parent partition, d386 style);
//!   4. fine faces strictly inside a coarse element stay zero (new flux
//!      information has no coarse counterpart).
//! A tet hierarchy runs the same independent audit (pinning the pre-existing
//! simplex behavior), a prism case covers mixed tri+quad faces on one element.

use fem_assembly::build_prolongation_hdiv;
use fem_mesh::topology::MeshTopology;
use fem_mesh::{ElementType, Mesh};
use fem_space::dof_manager::FaceKey;
use fem_space::HDivSpace;

/// Quantized coordinate (the generators use dyadic rationals — exact).
type Qc = i64;
fn q(x: f64) -> Qc {
    (x * 1_048_576.0).round() as Qc
}

/// Local faces of each supported 3-D element shape: a 3-entry slice is a
/// triangular face, a 4-entry slice a quadrilateral face.  Vertex sets mirror
/// the HDivSpace builders' face tables (`crates/space/src/hdiv.rs`); the
/// `FaceKey` lookup sorts internally, so only the vertex *set* matters.
fn elem_faces(et: ElementType) -> &'static [&'static [usize]] {
    static TET: [&[usize]; 4] = [&[1, 2, 3], &[0, 2, 3], &[0, 1, 3], &[0, 1, 2]];
    static HEX: [&[usize]; 6] = [
        &[0, 1, 2, 3],
        &[4, 5, 6, 7],
        &[0, 1, 5, 4],
        &[2, 3, 7, 6],
        &[0, 3, 7, 4],
        &[1, 2, 6, 5],
    ];
    static PRISM: [&[usize]; 5] = [
        &[0, 1, 2],
        &[3, 4, 5],
        &[0, 1, 4, 3],
        &[1, 2, 5, 4],
        &[0, 2, 5, 3],
    ];
    static PYRAMID: [&[usize]; 5] = [
        &[0, 1, 4],
        &[1, 2, 4],
        &[2, 3, 4],
        &[3, 0, 4],
        &[0, 1, 2, 3],
    ];
    match et {
        ElementType::Tet4 | ElementType::Tet10 => &TET,
        ElementType::Hex8 => &HEX,
        ElementType::Prism6 => &PRISM,
        ElementType::Pyramid5 => &PYRAMID,
        other => panic!("unsupported 3-D element type {other:?}"),
    }
}

/// Canonical `HDivSpace` face key of a face's global vertices (quad: sorted
/// first 3 of 4).
fn face_key(verts: &[u32]) -> FaceKey {
    if verts.len() == 3 {
        FaceKey::new(verts[0], verts[1], verts[2])
    } else {
        let mut v = [verts[0], verts[1], verts[2], verts[3]];
        v.sort_unstable();
        FaceKey::new(v[0], v[1], v[2])
    }
}

/// Unique faces of a mesh as `(sorted vertex set, canonical perimeter-order
/// verts, face_dofs block)` triples.  The canonical order (the element face
/// table's) is what makes the quad's *consecutive* pairs its perimeter edges
/// — the sorted order's pairs are diagonals.
fn mesh_faces(mesh: &Mesh<3>, space: &HDivSpace<Mesh<3>>) -> Vec<(Vec<u32>, Vec<u32>, Vec<u32>)> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for e in 0..mesh.n_elements() as u32 {
        let nodes = mesh.element_nodes(e);
        for fv in elem_faces(mesh.element_type(e)) {
            let canon: Vec<u32> = fv.iter().map(|&i| nodes[i]).collect();
            let mut verts = canon.clone();
            verts.sort_unstable();
            if seen.insert(verts.clone()) {
                let block = space
                    .face_dofs(face_key(&verts))
                    .unwrap_or_else(|| panic!("{:?} face {verts:?} missing from space", mesh.element_type(e)));
                out.push((verts, canon, block));
            }
        }
    }
    out
}

/// Independent audit of the prolongation on one hierarchy (see module doc).
fn check_hierarchy(tag: &str, coarse_mesh: Mesh<3>, fine_mesh: Mesh<3>, order: u8) {
    let coarse = HDivSpace::new(coarse_mesh.clone(), order);
    let fine = HDivSpace::new(fine_mesh.clone(), order);
    let (p, stats) = build_prolongation_hdiv(&coarse, &fine);
    assert_eq!(p.nrows, fine.n_dofs(), "{tag}: P rows");
    assert_eq!(p.ncols, coarse.n_dofs(), "{tag}: P cols");

    let qc = |m: &Mesh<3>, n: u32| -> [Qc; 3] {
        let c = m.node_coords(n);
        [q(c[0]), q(c[1]), q(c[2])]
    };
    // Midpoint node of a coarse edge, found in the fine mesh by quantized
    // coordinate (midpoints/centers of dyadic meshes are exact).
    let mut by_coord: std::collections::HashMap<[Qc; 3], u32> = std::collections::HashMap::new();
    for n in 0..fine_mesh.n_nodes() as u32 {
        by_coord.insert(qc(&fine_mesh, n), n);
    }
    let mid_of = |m: &Mesh<3>, a: u32, b: u32| -> u32 {
        let ca = m.node_coords(a);
        let cb = m.node_coords(b);
        let key = [q(0.5 * (ca[0] + cb[0])), q(0.5 * (ca[1] + cb[1])), q(0.5 * (ca[2] + cb[2]))];
        by_coord[&key]
    };

    let coarse_faces = mesh_faces(&coarse_mesh, &coarse);
    let fine_faces = mesh_faces(&fine_mesh, &fine);

    // Fine-element visit count per fine face (vertex set).
    let mut visits: std::collections::HashMap<Vec<u32>, usize> = std::collections::HashMap::new();
    for e in 0..fine_mesh.n_elements() as u32 {
        let nodes = fine_mesh.element_nodes(e);
        for fv in elem_faces(fine_mesh.element_type(e)) {
            let mut verts: Vec<u32> = fv.iter().map(|&i| nodes[i]).collect();
            verts.sort_unstable();
            *visits.entry(verts).or_insert(0) += 1;
        }
    }

    // Row sums (the P·1 column), built while auditing.
    let mut row_sum = vec![0.0_f64; fine.n_dofs()];
    for r in 0..fine.n_dofs() {
        for k in p.row_ptr[r]..p.row_ptr[r + 1] {
            row_sum[r] += p.values[k];
        }
    }

    let mut expected_entries = 0usize;
    let mut parented_fine_faces = 0usize;
    for (fverts, _, fblock) in &fine_faces {
        // Parent: the coarse face this fine face refines, if any.
        let mut parent: Option<((Vec<u32>, Vec<u32>), f64)> = None;
        for (sverts, cverts, _) in &coarse_faces {
            if sverts == fverts {
                parent = Some(((sverts.clone(), cverts.clone()), 1.0));
                break;
            }
            // Sub-face: all fine verts are verts / edge midpoints / (quad)
            // face center of the coarse face.  Edge midpoints come from the
            // CANONICAL perimeter order — the sorted order's consecutive
            // pairs are diagonals, not edges.
            let nv = cverts.len();
            let mut extended: std::collections::HashSet<[Qc; 3]> =
                cverts.iter().map(|&v| qc(&coarse_mesh, v)).collect();
            for i in 0..nv {
                let mid = mid_of(&coarse_mesh, cverts[i], cverts[(i + 1) % nv]);
                extended.insert(qc(&fine_mesh, mid));
            }
            if nv == 4 {
                let mut s = [0.0_f64; 3];
                for &v in cverts {
                    let c = coarse_mesh.node_coords(v);
                    for k in 0..3 {
                        s[k] += c[k];
                    }
                }
                extended.insert([q(s[0] / 4.0), q(s[1] / 4.0), q(s[2] / 4.0)]);
            }
            if fverts.iter().all(|v| extended.contains(&qc(&fine_mesh, *v))) {
                parent = Some(((sverts.clone(), cverts.clone()), 0.25));
                break;
            }
        }

        let kv = visits.get(fverts).copied().unwrap_or(0);
        match parent {
            Some(((sverts, _), ratio)) => {
                parented_fine_faces += 1;
                let cblock = &coarse_faces
                    .iter()
                    .find(|(v, _, _)| *v == sverts)
                    .unwrap()
                    .2;
                assert_eq!(
                    fblock.len(),
                    cblock.len(),
                    "{tag}: fine face {fverts:?} block len vs coarse parent's (shape-derived (k+1)^2 vs (k+1)(k+2)/2)"
                );
                expected_entries += fblock.len() * kv;
                for (m, &fd) in fblock.iter().enumerate() {
                    let row = p.row_ptr[fd as usize]..p.row_ptr[fd as usize + 1];
                    assert!(
                        !row.is_empty(),
                        "{tag}: fine face {fverts:?} dof {fd} (block offset {m}) has an empty P row — the walk missed this {} face block",
                        if fverts.len() == 4 { "QUAD" } else { "tri" }
                    );
                    assert_eq!(
                        row.len(),
                        1,
                        "{tag}: fine dof {fd} has {} entries, expected the single block-identity link",
                        row.len()
                    );
                    let k = row.start;
                    assert_eq!(
                        p.col_idx[k], cblock[m],
                        "{tag}: fine face {fverts:?} offset {m} links to coarse dof {}, expected the parent block's offset {m} ({})",
                        p.col_idx[k], cblock[m]
                    );
                    let want = ratio * kv as f64;
                    assert!(
                        (p.values[k] - want).abs() <= 1e-14,
                        "{tag}: P[{},{}] = {}, expected {want} (ratio {ratio} × {kv} visits)",
                        p.col_idx[k],
                        fd,
                        p.values[k]
                    );
                    assert!(
                        (row_sum[fd as usize] - want).abs() <= 1e-14,
                        "{tag}: P·1 at fine dof {fd} = {}, expected {want}",
                        row_sum[fd as usize]
                    );
                }
            }
            None => {
                // New flux information inside a coarse element: no coarse
                // counterpart — the rows must stay empty.
                for &fd in fblock {
                    let row = p.row_ptr[fd as usize]..p.row_ptr[fd as usize + 1];
                    assert!(
                        row.is_empty(),
                        "{tag}: unparented fine face dof {fd} received a coarse link"
                    );
                }
            }
        }
    }

    assert_eq!(
        stats.located_count, expected_entries,
        "{tag}: located_count (the walk's own block-size accounting)"
    );

    // Pᵀ coverage: every coarse FACE dof is claimed; coarse interior dofs
    // receive nothing (documented higher-order approximation).
    let mut col_claimed = vec![false; coarse.n_dofs()];
    for k in 0..p.col_idx.len() {
        col_claimed[p.col_idx[k] as usize] = true;
    }
    let mut covered_face_dofs = 0usize;
    for (_, _, cblock) in &coarse_faces {
        for &d in cblock {
            assert!(
                col_claimed[d as usize],
                "{tag}: coarse face dof {d} is claimed by no fine child (order {order})"
            );
            covered_face_dofs += 1;
        }
    }
    eprintln!("{tag}: covered coarse face dofs = {covered_face_dofs}, parented fine faces = {parented_fine_faces}, expected entries = {expected_entries}");

    // Pᵀ single-parent partition: every nonzero row's support lies inside a
    // single coarse element's dof set.
    for r in 0..fine.n_dofs() {
        let row = p.row_ptr[r]..p.row_ptr[r + 1];
        if row.is_empty() {
            continue;
        }
        let supported = (0..coarse_mesh.n_elements() as u32).any(|e| {
            let dofs = coarse.element_dofs(e);
            row.clone().all(|k| dofs.contains(&(p.col_idx[k] as u32)))
        });
        assert!(
            supported,
            "{tag}: fine dof {r} row support matches no single coarse element"
        );
    }
}

/// HEX hierarchy, RT1: the headline red case.  Quad faces carry (k+1)^2 = 4
/// face dofs — the old tri-only stride (3) left them orphaned and, on hex,
/// the whole P came back empty.
#[test]
fn hex_rt1_prolongation_walks_quad_face_blocks() {
    let coarse = Mesh::<3>::unit_cube_hex(2);
    let fine = fem_mesh::refine_uniform_3d(&coarse);
    check_hierarchy("hex rt1", coarse, fine, 1);
}

// HEX hierarchy, RT0: block size 1 hides a wrong *stride* but not a wrong
// *walk* — the old tet-only face enumeration still returned an empty P here.
// (D468, round 52: hex RT0 prolongation now uses the MFEM-exact
// `LocalInterpolation_RT` semantics — dense midline rows, single-write sub-face
// ratios — so the old structural walk pin no longer described the truth; the
// stronger bitwise pin lives in
// tests/d468_hdiv_prolongation_mfem_parity.rs::d468_hex_rt0_matches_mfem,
// which also asserts the sparse structure bidirectionally.  The superseded
// legacy pin was removed rather than ignored.)

/// TET hierarchy, RT1: the pre-existing simplex path must survive the
/// shape-driven rewrite bit-for-bit (same audit, triangular faces only).
#[test]
fn tet_rt1_prolongation_structure_unchanged() {
    let coarse = Mesh::<3>::unit_cube_tet(1);
    let fine = fem_mesh::refine_uniform_3d(&coarse);
    check_hierarchy("tet rt1", coarse, fine, 1);
}

// TET/PRISM hierarchy, RT0: (D481/D482, round 53) both geometries now use the
// MFEM-exact `LocalInterpolation_RT` semantics — mirrored-sliver frames on the
// tet, the Prism family slot rows on the wedge — so midline/interior sub-faces
// carry dense interpolation rows and the old "unparented ⇒ empty row" /
// block-identity structural pins no longer describe the truth.  The stronger
// bitwise pins live in tests/d468_hdiv_prolongation_mfem_parity.rs
// (tet_rt0 264/264 rows at 5.551e-17, prism_rt0 88/88 rows bitwise), together
// with the constant-field P·x_c ≡ x_f semantics tests.  The superseded legacy
// pins were removed rather than ignored.

