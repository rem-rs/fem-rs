//! D584 — skewed (top-face-sheared) prism RT0 prolongation: MFEM parity pin,
//! constant-field exactness, and the single-source slot-row convention.
//!
//! D572 doubled `PrismRT0`'s two triangular-face basis functions (the
//! `RT0WdgFiniteElement` convention, `fe_fixed_order.cpp:6403`), so the stored
//! dofs are physical face fluxes (dof = f·adj(J)·n̂|F|).  The interpolation
//! row tables — `crates/space/src/hdiv.rs::interp_rows(Prism6)` and
//! `crates/assembly/src/transfer.rs::hdiv_rt_slot_rows(Prism, 0)` — however
//! still carried the generic `RT_WedgeElement` nk (±1 on the triangular faces
//! = 2·n̂|F|, `fe_rt.cpp`), a hybrid convention matching no MFEM collection:
//! with them the reference dual reads W = diag(2,2,1,1,1) and the prolongation
//! rows P = B·W⁻¹ land on the *generic* convention's interpolation rows
//! instead of MFEM RT0Wdg's (`fe_fixed_order.cpp:6442`, whose DEBUG assert
//! documents that its own nk table is point-dual to its doubled basis).
//!
//! Round 58 predicted the defect would show on skewed prisms as a 2× error in
//! the "triangular fine rows × quadrilateral parent columns" cross block.  It
//! does not: a prism's uniform refinement subdivides the element **in its own
//! ENGINE frame** (the shear lives entirely in the parent→physical Jacobian),
//! so the child embedding A is diagonal and adj(A)ᵀ never mixes the layer axis
//! with the in-plane axes — every convention-sensitive entry (c_j·d_k ∉ {1})
//! is identically zero, on axis-aligned AND skewed prisms alike.  Probes:
//! `tmp/d584/d584_probe_prism_skew.cpp` dumps MFEM's P for the skewed wedge
//! from BOTH collections and they are entrywise identical (46/46, diff = 0,
//! `$HOME/work/d584/d584_prism_skew_{gen,rw}.txt`).
//!
//! The tables are nonetheless realigned to the RT0Wdg nk (±½ triangular-face
//! rows, `fe_fixed_order.cpp:6439`): the rows then state the same n̂|F|
//! semantics as the dofs they produce/consume, W = I and P = B reproduces
//! MFEM's RT0Wdg interpolation machinery directly instead of relying on the
//! diagonal-W cancellation.  All values are pinned below and must be unchanged.

use fem_assembly::transfer::build_prolongation_hdiv;
use fem_linalg::CsrMatrix;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_space::dof_manager::FaceKey;
use fem_space::HDivSpace;
use std::collections::{HashMap, HashSet};

/// A single prism with its top face translated by (0.3, 0.2) — the same
/// geometry as the MFEM probe's skew wedge.  The element stays affine (the
/// ENGINE frame absorbs the shear), so the exact-path prolongation builder
/// must serve the refinement.  A single element also keeps this fixture free
/// of the coincident-vertex pinch of the two-wedge cube (its crossing z = 0
/// diagonals refine to two distinct fine nodes at the same point — a
/// `HdivVertexMaps` ambiguity, exercised by the two-wedge test below).
fn skewed_prism_mesh() -> Mesh<3> {
    Mesh::<3> {
        coords: vec![
            0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.3, 0.2, 1.0, //
            1.3, 0.2, 1.0, //
            0.3, 1.2, 1.0, //
        ],
        conn: vec![0, 1, 2, 3, 4, 5],
        vertex_parents: vec![],
        elem_tags: vec![1],
        elem_type: ElementType::Prism6,
        face_conn: vec![],
        face_tags: vec![],
        face_type: ElementType::Tri3,
        elem_types: None,
        elem_offsets: None,
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        nc_vertex_view: None,
        geometry: None,
    }
}

/// MFEM `MakeCartesian3D(1, 1, 1, WEDGE)` (the d468 `mfem_prism_mesh`
/// connectivity — the wedges share the main-diagonal quad) with the whole
/// z = 1 layer translated by (0.3, 0.2).  Exercises the exact path across a
/// shared interior face on the sheared geometry; the builder's coarse→fine
/// vertex correlation must keep twin-resolution robust (its midpoint lookup
/// is nearest-node, so coincident fine vertices — as produced by pinched
/// diagonal splits — must be expanded by coordinate, not picked by scan
/// order).  The constant field must stay dof-exact.
fn skewed_two_wedge_mesh() -> Mesh<3> {
    Mesh::<3> {
        coords: vec![
            0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            1.0, 1.0, 0.0, //
            0.3, 0.2, 1.0, //
            1.3, 0.2, 1.0, //
            0.3, 1.2, 1.0, //
            1.3, 1.2, 1.0, //
        ],
        conn: vec![0, 1, 3, 4, 5, 7, 0, 3, 2, 4, 7, 6],
        vertex_parents: vec![],
        elem_tags: vec![1, 1],
        elem_type: ElementType::Prism6,
        face_conn: vec![],
        face_tags: vec![],
        face_type: ElementType::Tri3,
        elem_types: None,
        elem_offsets: None,
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        nc_vertex_view: None,
        geometry: None,
    }
}

fn q(v: f64) -> i64 {
    (v * 1e9).round() as i64
}

fn coord3(mesh: &Mesh<3>, v: u32) -> [f64; 3] {
    let c = mesh.node_coords(v);
    [c[0], c[1], c[2]]
}

/// Face identity key: the face's vertex coordinates, sorted lexicographically
/// and flattened to the 1e-9 grid (the d468/d482 join).
fn face_key(mut vs: Vec<[f64; 3]>) -> Vec<i64> {
    let mut s: Vec<[i64; 3]> = vs.drain(..).map(|c| [q(c[0]), q(c[1]), q(c[2])]).collect();
    s.sort();
    s.into_iter().flatten().collect()
}

/// Sorted-first-3 FaceKey of a quad's four global vertices.
fn super_key(_mesh: &Mesh<3>, quad: [u32; 4]) -> FaceKey {
    let mut v: Vec<u32> = quad.to_vec();
    v.sort_unstable();
    FaceKey::new(v[0], v[1], v[2])
}

/// fem-rs face table (3-D prism): faces = bottom tri (0,1,2), top tri (3,4,5),
/// quads (0,1,4,3), (1,2,5,4), (0,2,5,3); face vertex key → global RT0 dof.
fn face_dofs_prism(space: &HDivSpace<Mesh<3>>, mesh: &Mesh<3>) -> HashMap<Vec<i64>, u32> {
    const TRI: [(usize, usize, usize); 2] = [(0, 1, 2), (3, 4, 5)];
    const QUAD: [[usize; 4]; 3] = [[0, 1, 4, 3], [1, 2, 5, 4], [0, 2, 5, 3]];
    let mut out = HashMap::new();
    for e in 0..mesh.n_elements() as u32 {
        let nd = mesh.element_nodes(e);
        for &(a, b, c) in &TRI {
            let dof = space
                .tri_face_dof(FaceKey::new(nd[a], nd[b], nd[c]))
                .unwrap_or_else(|| panic!("missing prism tri dof"));
            out.insert(
                face_key(vec![coord3(mesh, nd[a]), coord3(mesh, nd[b]), coord3(mesh, nd[c])]),
                dof,
            );
        }
        for qv in &QUAD {
            let fk = super_key(mesh, [nd[qv[0]], nd[qv[1]], nd[qv[2]], nd[qv[3]]]);
            let dof = space
                .tri_face_dof(fk)
                .unwrap_or_else(|| panic!("missing prism quad dof"));
            out.insert(
                face_key(vec![
                    coord3(mesh, nd[qv[0]]),
                    coord3(mesh, nd[qv[1]]),
                    coord3(mesh, nd[qv[2]]),
                    coord3(mesh, nd[qv[3]]),
                ]),
                dof,
            );
        }
    }
    out
}

/// Every MFEM truth entry must appear in the fem-rs P with the same value, and
/// the fem-rs P must carry no extra entries (exact sparsity parity).  Prism
/// fine/coarse faces are tri (9 coords) or quad (12 coords), so the split of
/// each truth row is inferred from the row length and the ambiguous
/// tri/quad-length ties are resolved by the coarse key lookup.
fn assert_matches_mfem_prism(
    p: &CsrMatrix<f64>,
    fine_map: &HashMap<Vec<i64>, u32>,
    coarse_map: &HashMap<Vec<i64>, u32>,
    truth: &[&[f64]],
    label: &str,
) {
    let mut rs: HashMap<u32, HashMap<u32, f64>> = HashMap::new();
    for r in 0..p.nrows {
        for k in p.row_ptr[r]..p.row_ptr[r + 1] {
            rs.entry(r as u32).or_default().insert(p.col_idx[k] as u32, p.values[k]);
        }
    }
    let mut matched_rs = HashSet::new();
    let mut max_err = 0.0_f64;
    for row in truth {
        let n = row.len();
        let mut found = false;
        for (vf, vc) in [(3usize, 3usize), (3, 4), (4, 3), (4, 4)] {
            if 3 * vf + 3 * vc + 1 != n {
                continue;
            }
            let fine_key: Vec<i64> = row[..3 * vf].iter().map(|v| q(*v)).collect();
            let coarse_key: Vec<i64> = row[3 * vf..3 * vf + 3 * vc].iter().map(|v| q(*v)).collect();
            let v = row[n - 1];
            let (Some(&r), Some(&c)) = (fine_map.get(&fine_key), coarse_map.get(&coarse_key))
            else {
                continue;
            };
            let got = rs
                .get(&r)
                .and_then(|m| m.get(&c))
                .unwrap_or_else(|| panic!("{label}: fem-rs P[{r},{c}] missing for MFEM entry {v}"));
            max_err = max_err.max((got - v).abs());
            assert!((got - v).abs() <= 1e-12, "{label}: P[{r},{c}] = {got} vs MFEM {v}");
            matched_rs.insert((r, c));
            found = true;
            break;
        }
        assert!(found, "{label}: no join for truth row {row:?}");
    }
    let extra: Vec<(u32, u32)> = rs
        .iter()
        .flat_map(|(r, m)| m.keys().map(move |c| (*r, *c)))
        .filter(|k| !matched_rs.contains(k))
        .collect();
    assert!(
        extra.is_empty(),
        "{label}: fem-rs carries {} entries absent from MFEM P, e.g. {:?}",
        extra.len(),
        &extra[..extra.len().min(4)]
    );
    eprintln!("{label}: {} MFEM entries matched, max|delta| = {max_err:.3e}", truth.len());
}

include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tmp/d584/d584_prism_skew_truth.rs"));

/// MFEM RT0Wdg parity on the skewed single wedge: every P entry (46) equals
/// the `RT0_3DFECollection` probe value, and the sparsity matches exactly.
#[test]
fn d584_skew_prism_rt0_matches_mfem_rt0wdg() {
    let coarse_mesh = skewed_prism_mesh();
    let fine_mesh = fem_mesh::refine_uniform_3d(&coarse_mesh);
    let coarse_space = HDivSpace::new(coarse_mesh.clone(), 0);
    let fine_space = HDivSpace::new(fine_mesh.clone(), 0);
    let (p, stats) = build_prolongation_hdiv(&coarse_space, &fine_space);
    assert_eq!(
        stats.located_count,
        fine_space.n_dofs(),
        "exact path must serve the skewed affine prism"
    );
    let fm = face_dofs_prism(&fine_space, &fine_mesh);
    let cm = face_dofs_prism(&coarse_space, &coarse_mesh);
    assert_matches_mfem_prism(&p, &fm, &cm, MFEM_SKEW_PRISM_O0, "skew prism RT0 (RT0Wdg)");
}

/// A constant field is exactly representable in RT0 on every affine mesh, so
/// the prolongation of its coarse projection must equal the fine projection
/// dof-for-dof — on the skewed prism too.  (Round 58 predicted this would fail
/// with the generic-nk row tables; the probes show the cross block is
/// identically zero in every affine prism because the refinement children are
/// ENGINE-frame-aligned — the assertion documents the adjudicated behaviour.)
#[test]
fn d584_skew_prism_rt0_constant_field_prolongs_exactly() {
    let c3 = [0.9_f64, 0.4, -1.1];
    let coarse_mesh = skewed_prism_mesh();
    let fine_mesh = fem_mesh::refine_uniform_3d(&coarse_mesh);
    let coarse_space = HDivSpace::new(coarse_mesh.clone(), 0);
    let fine_space = HDivSpace::new(fine_mesh.clone(), 0);
    let (p, _) = build_prolongation_hdiv(&coarse_space, &fine_space);
    let x_c = coarse_space.interpolate_vector(&|_| c3.to_vec());
    let x_f = fine_space.interpolate_vector(&|_| c3.to_vec());
    let mut y = vec![0.0_f64; fine_space.n_dofs()];
    p.spmv(x_c.as_slice(), &mut y);
    let mut worst = (0usize, 0.0_f64);
    for i in 0..fine_space.n_dofs() {
        let d = (y[i] - x_f.as_slice()[i]).abs();
        if d > worst.1 {
            worst = (i, d);
        }
        assert!(
            d <= 1e-12,
            "skew prism: constant-flux dof {i}: P·x_c = {} vs fine projection {} (|res| {d})",
            y[i],
            x_f.as_slice()[i]
        );
    }
    eprintln!(
        "d584: constant-field prolongation exact on skew prism, worst |res| = {:.3e} at dof {}",
        worst.1, worst.0
    );
}

/// Fixture guard: the ENGINE frame is genuinely skewed — the layer column
/// pt(3)−pt(0) = (0.3, 0.2, 1) has a non-orthogonal in-plane projection (the
/// η component 0.3 is 30% of the layer height), and the whole refinement still
/// takes the exact path.  Guards the two tests above against silently
/// degenerating to an axis-aligned fixture.
#[test]
fn d584_skew_fixture_is_genuinely_skewed() {
    let mesh = skewed_prism_mesh();
    let p = |i: u32| mesh.node_coords(i);
    let (v0, v1, v2, v3) = (p(0), p(1), p(2), p(3));
    let vertical = [v3[0] - v0[0], v3[1] - v0[1], v3[2] - v0[2]];
    let eta = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    let zeta = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    assert!(
        dot(vertical, eta).abs() > 0.1 && dot(vertical, zeta).abs() > 0.1,
        "fixture degraded: layer column no longer sheared against the tri plane"
    );
    // and the exact path must still serve it
    let fine_mesh = fem_mesh::refine_uniform_3d(&mesh);
    let coarse_space = HDivSpace::new(mesh, 0);
    let fine_space = HDivSpace::new(fine_mesh, 0);
    let (_, stats) = build_prolongation_hdiv(&coarse_space, &fine_space);
    assert_eq!(stats.located_count, fine_space.n_dofs());
}

/// Two-wedge cube, sheared: the interior diagonal quad makes the z = 0
/// diagonals of the two wedges cross at (0.5, 0.5, 0), and their midpoints
/// refine to TWO distinct fine nodes at the same coordinates.  The
/// prolongation builder's coarse→fine vertex correlation must not confuse the
/// twins (a plain nearest-node lookup resolves the tie by scan order and hands
/// the wrong twin to the second wedge's extended vertex set, which used to
/// make the exact path decline to the approximate legacy builder).  With the
/// coordinate-twin correction the exact path serves the whole refinement and
/// the constant field stays dof-exact.
#[test]
fn d584_skew_two_wedge_rt0_constant_field_prolongs_exactly() {
    let c3 = [0.9_f64, 0.4, -1.1];
    let coarse_mesh = skewed_two_wedge_mesh();
    let fine_mesh = fem_mesh::refine_uniform_3d(&coarse_mesh);
    let coarse_space = HDivSpace::new(coarse_mesh, 0);
    let fine_space = HDivSpace::new(fine_mesh, 0);
    let (p, stats) = build_prolongation_hdiv(&coarse_space, &fine_space);
    assert_eq!(
        stats.located_count,
        fine_space.n_dofs(),
        "exact path must serve the sheared two-wedge mesh (twin-vertex correlation)"
    );
    let x_c = coarse_space.interpolate_vector(&|_| c3.to_vec());
    let x_f = fine_space.interpolate_vector(&|_| c3.to_vec());
    let mut y = vec![0.0_f64; fine_space.n_dofs()];
    p.spmv(x_c.as_slice(), &mut y);
    for i in 0..fine_space.n_dofs() {
        assert!(
            (y[i] - x_f.as_slice()[i]).abs() <= 1e-12,
            "two-wedge skew: constant-flux dof {i}: P·x_c = {} vs fine projection {}",
            y[i],
            x_f.as_slice()[i]
        );
    }
}
