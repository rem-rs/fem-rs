//! D598 — a *legitimate* twin-vertex (pinched-diagonal) fixture that actually
//! triggers the `coord_twins` coordinate-equivalence path of the HDiv
//! prolongation builder's coarse→fine vertex correlation
//! (`crates/assembly/src/transfer.rs`).
//!
//! Trigger scenario (the D584 adjudication's defence case): two adjacent hexes
//! are each decomposed by MFEM's `AddHexAsWedges` (`mesh.cpp:2262`,
//! `hex_to_wdg = {0,1,2,4,5,6}, {0,2,3,4,6,7}`) — but the upper hex carries a
//! *rotated* vertex numbering, so its wedge split cuts the shared z = 1 face
//! along the *anti*-diagonal where the lower hex's split leaves the *main*
//! diagonal.  The shared face then carries both diagonals as coarse edges
//! (a "pinched" diagonal cut, exactly the adjudicated
//! "相邻 hex 各自 AddHexAsWedges 的共享面上对角交叉" case):
//!
//! ```text
//!    (0,1) 7────────6 (1,1)          z=1 face, from above:
//!          │ ╲    ╱ │                A-side diagonal 4–6 (main)
//!          │  ╲  ╱  │                B-side diagonal 5–7 (anti)
//!          │   ╳    │                crossing at (0.5, 0.5, 1) —
//!          │  ╱  ╲  │                NOT a coarse mesh node
//!          │ ╱    ╲ │
//!    (0,0) 4────────5 (1,0)
//! ```
//!
//! Under uniform refinement the midpoints of the two crossing diagonals
//! refine to **two distinct fine nodes at the same coordinates** — coordinate
//! twins.  A plain nearest-node midpoint lookup resolves the tie by scan
//! order and hands both coarse edges the same twin, starving one wedge's
//! extended vertex set and declining the exact prolongation path (the
//! original D584 red).  The `coord_twins` expansion (1e-9 grid) is what keeps
//! the exact path alive here; no test exercised that path before D598.

use fem_assembly::transfer::build_prolongation_hdiv;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_space::HDivSpace;

/// The pinched two-hex wedge mesh: box [0,1]×[0,1]×[0,2], lower hex numbered
/// in the MFEM standard order, upper hex with labels rotated one position
/// around the vertex cycle (so `AddHexAsWedges`' diagonal plane crosses the
/// lower hex's on the shared face).  Each hex is split exactly per MFEM
/// `AddHexAsWedges`: wedges (0,1,2,4,5,6) and (0,2,3,4,6,7) of the hex's own
/// numbering.
///
/// Global nodes: 0..3 lower-hex bottom (z=0), 4..7 the z=1 square
/// (4=(0,0), 5=(1,0), 6=(1,1), 7=(0,1)), 8..11 upper-hex top (z=2).
fn pinched_two_hex_wedge_mesh() -> Mesh<3> {
    Mesh::<3> {
        coords: vec![
            0.0, 0.0, 0.0, // 0  A.v0
            1.0, 0.0, 0.0, // 1  A.v1
            1.0, 1.0, 0.0, // 2  A.v2
            0.0, 1.0, 0.0, // 3  A.v3
            0.0, 0.0, 1.0, // 4  A.v4 = B.v1  (z=1 square corner (0,0))
            1.0, 0.0, 1.0, // 5  A.v5 = B.v2  (z=1 square corner (1,0))
            1.0, 1.0, 1.0, // 6  A.v6 = B.v3  (z=1 square corner (1,1))
            0.0, 1.0, 1.0, // 7  A.v7 = B.v0  (z=1 square corner (0,1))
            0.0, 1.0, 2.0, // 8  B.v4
            0.0, 0.0, 2.0, // 9  B.v5
            1.0, 0.0, 2.0, // 10 B.v6
            1.0, 1.0, 2.0, // 11 B.v7
        ],
        conn: vec![
            // Lower hex (standard numbering) split by AddHexAsWedges:
            0, 1, 2, 4, 5, 6, // A0
            0, 2, 3, 4, 6, 7, // A1 — leaves main-diagonal edge 4–6 on z=1
            // Upper hex (labels rotated: v0=(0,1,1), v1=(0,0,1), v2=(1,0,1),
            // v3=(1,1,1), v4..v7 likewise) split by AddHexAsWedges:
            7, 4, 5, 8, 9, 10, // B0
            7, 5, 6, 8, 10, 11, // B1 — leaves anti-diagonal edge 5–7 on z=1
        ],
        vertex_parents: vec![],
        elem_tags: vec![1, 1, 1, 1],
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

/// Count fine-mesh coordinate-twin groups: sets of ≥2 distinct nodes whose
/// coordinates agree on the 1e-9 grid (the same key the builder's
/// `coord_twins` map uses).
fn twin_groups(mesh: &Mesh<3>) -> Vec<Vec<u32>> {
    use std::collections::HashMap;
    let mut by_key: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
    for n in 0..mesh.n_nodes() as u32 {
        let c = mesh.node_coords(n);
        let key = [
            (c[0] * 1e9).round() as i64,
            (c[1] * 1e9).round() as i64,
            (c[2] * 1e9).round() as i64,
        ];
        by_key.entry(key).or_default().push(n);
    }
    let mut groups: Vec<Vec<u32>> = by_key
        .into_values()
        .filter(|g| g.len() > 1)
        .collect();
    groups.sort();
    groups
}

/// Fixture guard: the pinch is real — the coarse mesh carries BOTH diagonals
/// of the shared z = 1 face (edges 4–6 and 5–7), and the refinement
/// materialises the coincident fine twins at (0.5, 0.5, 1).  Guards the main
/// test below against silently degenerating to an unpinched (twin-free)
/// fixture that would no longer exercise `coord_twins`.
#[test]
fn d598_fixture_pinch_materialises_fine_twins() {
    let coarse = pinched_two_hex_wedge_mesh();
    // Both crossing diagonals present as (tri-face) wedge edges.
    let has_tri_edge = |a: u32, b: u32| {
        (0..coarse.n_elements() as u32).any(|e| {
            let nd = coarse.element_nodes(e);
            let pairs = [
                (nd[0], nd[1]),
                (nd[1], nd[2]),
                (nd[2], nd[0]),
                (nd[3], nd[4]),
                (nd[4], nd[5]),
                (nd[5], nd[3]),
            ];
            pairs.iter().any(|&(x, y)| (x == a && y == b) || (x == b && y == a))
        })
    };
    assert!(has_tri_edge(4, 6), "main diagonal 4-6 (lower hex split) missing");
    assert!(has_tri_edge(5, 7), "anti diagonal 5-7 (upper hex split) missing");

    let fine = fem_mesh::refine_uniform_3d(&coarse);
    let groups = twin_groups(&fine);
    assert!(
        !groups.is_empty(),
        "refinement produced no coordinate twins — fixture lost its pinch"
    );
    // The pinch site: two distinct fine nodes at (0.5, 0.5, 1).
    let centre = groups
        .iter()
        .any(|g| {
            let c = fine.node_coords(g[0]);
            g.len() >= 2
                && (c[0] - 0.5).abs() < 1e-9
                && (c[1] - 0.5).abs() < 1e-9
                && (c[2] - 1.0).abs() < 1e-9
        });
    assert!(
        centre,
        "no twin pair at the diagonal crossing (0.5, 0.5, 1): {groups:?}"
    );
}

/// The point of D598: with the coincident twins present, the exact
/// prolongation path must still serve EVERY fine dof (the `coord_twins`
/// expansion defeats the scan-order twin confusion), and a constant field —
/// exactly representable in RT0 — must prolong dof-exactly.
#[test]
fn d598_pinched_wedge_prolongation_exact_path_and_constant_field() {
    let c3 = [0.7_f64, -0.3, 1.2];
    let coarse_mesh = pinched_two_hex_wedge_mesh();
    let fine_mesh = fem_mesh::refine_uniform_3d(&coarse_mesh);
    let coarse_space = HDivSpace::new(coarse_mesh, 0);
    let fine_space = HDivSpace::new(fine_mesh, 0);
    let (p, stats) = build_prolongation_hdiv(&coarse_space, &fine_space);
    assert_eq!(
        stats.located_count,
        fine_space.n_dofs(),
        "exact path must serve the pinched two-hex wedge refinement \
         (coord_twins must resolve the diagonal-midpoint twins)"
    );
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
            "pinched wedges: constant-flux dof {i}: P·x_c = {} vs fine \
             projection {} (|res| {d})",
            y[i],
            x_f.as_slice()[i]
        );
    }
    eprintln!(
        "d598: exact path served all {} fine dofs; constant-field worst |res| = {:.3e} at dof {}",
        fine_space.n_dofs(),
        worst.1,
        worst.0
    );
}

