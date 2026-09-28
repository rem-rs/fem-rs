//! D832-2 (round 89 Lane B): 3-D periodic merged-mesh refinement geometry —
//! the D832-1 recipe (periodic Quad9, round 88) ported to the hex paths.
//!
//! MFEM 4.10 oracle (`tmp/d89b/d832_hex_probe.cpp`, serial build against
//! `$HOME/mfem410_ser`), 3×3×3 hex cube `[0,3]³`, both refinement orders:
//!
//! * `MakePeriodic` **always materializes discontinuous `L2_T1_3D_Pk` nodes**
//!   (`Mesh::MakePeriodic` calls `SetCurvature(order, /*discont=*/true)`
//!   before renumbering — even a straight, node-less mesh gains
//!   `L2_T1_3D_P1` nodes), so every element keeps the coordinates of its own
//!   side of a seam; refining the folded mesh keeps the `L2_T1` layout with
//!   **fully per-element own-side rows** (straight: 27·8 = 216 coarse /
//!   216·8 = 1728 fine; order-2: 27·27 = 729 / 216·27 = 5832).
//! * Both branches agree on the endpoint: NE = NV = 216, nodes `L2_T1_3D_Pk`
//!   own-side, 216 (P1) / 1728 (P2) distinct nodal positions on the torus,
//!   **both** seam planes populated (x=0: 144, x=3: 144 for P1; 324/324 for
//!   P2), every element volume 1/8, torus volume 27, and H1(1/2/3) =
//!   216/1728/5832 (= n³·1, n³·8, n³·27 on the n³ torus — MFEM probe: coarse
//!   folded 27/216/729, refined 216/1728/5832).  The *unfolded* order-2
//!   refinement keeps shared `H1_3D_P2` nodes (2197 = 13³ dofs) — only the
//!   fold switches the nodal space to `L2_T1`.
//!
//! The mesh-level assertions below are independent of any space build (the
//! round-87 lesson): compactness, edge counts, per-element own-side rows and
//! Jacobian determinants are read straight off the `Mesh`; only the trailing
//! H1 counts touch `H1Space`, matching MFEM's probe output.
//!
//! Fixtures: the in-process `make_cartesian_3d` (+ `set_curvature(2)`, the
//! hex27-import equivalent) + `make_periodic` construction, and the two
//! tracked folded files `data/periodic-cube-p2.mesh` and (as the byte copy
//! `tests/data/d814_periodic_cube.mesh.txt`) `data/periodic-cube.mesh`.

use fem_element::quadrature::hex_rule;
use fem_io::mfem::read_mfem;
use fem_mesh::{ElementType, Mesh, MeshTopology, refine_uniform_3d};
use fem_space::{FESpace, H1Space};
use std::io::Cursor;

/// The tags `make_cartesian_3d` assigns to the box sides:
/// bottom `z=0` → 1, front `y=0` → 2, right `x=sx` → 3, back `y=sy` → 4,
/// left `x=0` → 5, top `z=sz` → 6 (see `periodic_hex_3d.rs`).
const PAIRS: [(i32, i32, [f64; 3]); 3] = [
    (5, 3, [3.0, 0.0, 0.0]), // left  → right, +x
    (2, 4, [0.0, 3.0, 0.0]), // front → back,  +y
    (1, 6, [0.0, 0.0, 3.0]), // bottom → top,  +z
];

fn straight_periodic() -> Mesh<3> {
    Mesh::<3>::make_cartesian_3d(3, 3, 3, ElementType::Hex8, 3.0, 3.0, 3.0, false)
        .make_periodic(&PAIRS, 1e-10)
        .expect("make periodic")
}

/// The hex27-import equivalent: straight hex8 + order-2 `nodes` (affine), the
/// form a Gmsh type-12 import or MFEM `SetCurvature(2, false)` produces.
fn curved_periodic() -> Mesh<3> {
    let mut base = Mesh::<3>::make_cartesian_3d(3, 3, 3, ElementType::Hex8, 3.0, 3.0, 3.0, false);
    base.set_curvature(2);
    base.make_periodic(&PAIRS, 1e-10).expect("make periodic")
}

fn folded_p2_file() -> Mesh<3> {
    read_mfem(Cursor::new(
        include_str!("../../../data/periodic-cube-p2.mesh").as_bytes(),
    ))
    .expect("read the folded p2 cube")
    .mesh3d
    .expect("3-D mesh")
}

fn folded_p1_file() -> Mesh<3> {
    read_mfem(Cursor::new(include_str!("data/d814_periodic_cube.mesh.txt").as_bytes()))
        .expect("read the folded p1 cube")
        .mesh3d
        .expect("3-D mesh")
}

/// Distinct positions of a coordinate table modulo the `[0,3)³` period on the
/// `1/ppu` lattice (the torus-quotient position count).
fn distinct_torus_positions(coords: &[f64], ppu: i64) -> usize {
    let mut keys = std::collections::BTreeSet::new();
    for c in coords.chunks_exact(3) {
        let k: [i64; 3] = std::array::from_fn(|d| {
            let k = (c[d] * ppu as f64).round() as i64 % (3 * ppu);
            if k < 0 { k + 3 * ppu } else { k }
        });
        keys.insert(k);
    }
    keys.len()
}

/// Number of geometry dofs whose `axis` coordinate sits at `level` (±1e-9).
fn count_at_plane(coords: &[f64], axis: usize, level: f64) -> usize {
    coords
        .chunks_exact(3)
        .filter(|c| (c[axis] - level).abs() < 1e-9)
        .count()
}

/// Unique undirected edges of the hex corner connectivity.
fn hex_edge_count(mesh: &Mesh<3>) -> usize {
    let mut edges = std::collections::BTreeSet::new();
    for e in 0..mesh.n_elems() as u32 {
        let ns = mesh.elem_nodes(e);
        for k in 0..4 {
            // bottom ring (0-1-2-3), top ring (4-5-6-7), vertical pillars
            for (a, b) in
                [(ns[k], ns[(k + 1) % 4]), (ns[4 + k], ns[4 + (k + 1) % 4]), (ns[k], ns[4 + k])]
            {
                edges.insert((a.min(b), a.max(b)));
            }
        }
    }
    edges.len()
}

/// Number of nodes referenced by at least one element.
fn referenced_nodes(mesh: &Mesh<3>) -> usize {
    let mut used = std::collections::BTreeSet::<u32>::new();
    for e in 0..mesh.n_elems() as u32 {
        used.extend(mesh.elem_nodes(e));
    }
    used.len()
}

/// Functional cell sanity: every geometry row entry in range, every child an
/// isoparametric cell of volume `cell` with positive Jacobian (integral over
/// a Gauss rule; affine fixtures except the coordinate-rounded folded files,
/// whose ~1e-6 cell spread the `tol` absorbs).
fn assert_cells(mesh: &Mesh<3>, cell: f64, tol: f64, what: &str) {
    let geo = mesh
        .geometry
        .as_ref()
        .unwrap_or_else(|| panic!("{what}: geometry table must survive refinement"));
    assert_eq!(
        geo.conn.len(),
        mesh.n_elems() as usize * geo.nodes_per_elem,
        "{what}: one full row per child"
    );
    for (i, &n) in geo.conn.iter().enumerate() {
        assert!(
            (n as usize) < geo.n_nodes,
            "{what}: geometry row entry {i} references {n} ≥ table size"
        );
    }
    let rule = hex_rule(4);
    let mut total = 0.0_f64;
    for e in 0..mesh.n_elems() as u32 {
        let mut vol = 0.0_f64;
        for (xi, w) in rule.points.iter().zip(rule.weights.iter()) {
            let (_jac, det, _x) = mesh.element_jacobian(e, xi);
            assert!(det > 0.0, "{what}: element {e} det {det} ≤ 0 at {xi:?}");
            vol += w * det;
        }
        assert!(
            (vol - cell).abs() < tol,
            "{what}: element {e} volume {vol} ≠ {cell}"
        );
        total += vol;
    }
    let torus = cell * mesh.n_elems() as f64;
    assert!(
        (total - torus).abs() < 1e-6,
        "{what}: torus volume {total} ≠ {torus}"
    );
}

/// The refined table's **fully discontinuous own-side layout**: element-major
/// rows of fresh, unshared dof ids (`n_nodes = NE · dpe`) — MFEM's refined
/// `L2_T1` nodes.  A shared-dof (wrapped-vertex-frame) table fails the
/// `n_nodes` pin; a dropped table fails the `geometry` pin in `assert_cells`.
fn assert_fresh_own_side(mesh: &Mesh<3>, dpe: usize, what: &str) {
    let geo = mesh
        .geometry
        .as_ref()
        .unwrap_or_else(|| panic!("{what}: geometry table must survive refinement"));
    assert_eq!(geo.conn.len(), mesh.n_elems() as usize * dpe, "{what}: row size");
    assert_eq!(
        geo.n_nodes,
        mesh.n_elems() as usize * dpe,
        "{what}: per-element own-side rows — a shared/wrapped table keys seam \
         dofs by the folded fine nodes"
    );
}

/// H1 counts on the n³ torus: n³, (2n)³, (3n)³ (MFEM probe: 27/216/729
/// coarse, 216/1728/5832 refined), with no orphan dofs.
fn assert_torus_h1(mesh: &Mesh<3>, n: usize, what: &str) {
    let counts: Vec<(u8, usize)> = [1u8, 2, 3]
        .iter()
        .map(|&o| (o, H1Space::new(mesh.clone(), o).n_dofs()))
        .collect();
    let cub = |k: usize| k * k * k;
    assert_eq!(counts[0], (1, cub(n)), "{what}: H1(1)");
    assert_eq!(counts[1], (2, cub(2 * n)), "{what}: H1(2)");
    assert_eq!(counts[2], (3, cub(3 * n)), "{what}: H1(3)");
    let space = H1Space::new(mesh.clone(), 2);
    let mut used = std::collections::BTreeSet::new();
    for e in 0..mesh.n_elems() as u32 {
        for &d in space.element_dofs(e) {
            assert!(d < space.n_dofs() as u32, "{what}: dof {d} out of range");
            used.insert(d);
        }
    }
    assert_eq!(used.len(), space.n_dofs(), "{what}: orphan dofs");
}

/// Straight hex torus, fold → refine (MFEM branch B).  The refined geometry
/// must stay the fully per-element own-side `L2_T1_3D_P1` table (1728 dofs),
/// not a dropped table (wrapped vertex frame) and not a shared-dof table.
#[test]
fn d832_straight_p1_fold_then_refine_matches_mfem() {
    let coarse = straight_periodic();
    assert_eq!(coarse.n_nodes(), 27, "folded 3×3×3 torus V");
    assert_eq!(coarse.n_elems(), 27);
    assert_eq!(hex_edge_count(&coarse), 81, "coarse torus E");
    assert_eq!(coarse.geom_order(), 1, "MakePeriodic materializes own-side nodes");
    assert_eq!(
        coarse.geom_n_nodes(),
        64,
        "the make_periodic snapshot rows address the pre-merge (4³) node table"
    );
    assert_cells(&coarse, 1.0, 1e-12, "coarse straight torus");

    let fine = refine_uniform_3d(&coarse);
    assert_eq!(fine.n_elems(), 216, "6×6×6 cells");
    assert_eq!(fine.element_type(0), ElementType::Hex8);
    assert_eq!(fine.n_nodes(), 216, "fine torus V (MFEM NV=216)");
    assert_eq!(referenced_nodes(&fine), fine.n_nodes(), "no orphan vertices");
    assert_eq!(hex_edge_count(&fine), 648, "fine torus E");
    assert_fresh_own_side(&fine, 8, "refined straight torus");
    assert_cells(&fine, 0.125, 1e-12, "refined straight torus");

    let geo = fine.geometry.as_ref().expect("geometry");
    assert_eq!(geo.order, 1);
    assert_eq!(
        distinct_torus_positions(&geo.coords, 2),
        216,
        "geometry dofs: 0.5-lattice torus positions (MFEM 216)"
    );
    assert_eq!(
        count_at_plane(&geo.coords, 0, 0.0),
        144,
        "own-side x=0 seam plane (MFEM 144)"
    );
    assert_eq!(
        count_at_plane(&geo.coords, 0, 3.0),
        144,
        "own-side x=3 seam plane (MFEM 144)"
    );
    assert_torus_h1(&fine, 6, "refined straight torus");
}

/// Order-2 hex torus, fold → refine (MFEM branch B, the D832-2 headline: the
/// hex27-import form).  The refined geometry must be 216·27 own-side rows
/// (MFEM `L2_T1_3D_P2` 5832), coherent across the seam — not the wrapped
/// shared-dof table.
#[test]
fn d832_curved_p2_fold_then_refine_matches_mfem() {
    let coarse = curved_periodic();
    assert_eq!(coarse.n_nodes(), 27);
    assert_eq!(coarse.geom_order(), 2, "carried order-2 nodes survive the fold");
    assert_cells(&coarse, 1.0, 1e-12, "coarse curved torus");

    let fine = refine_uniform_3d(&coarse);
    assert_eq!(fine.n_elems(), 216);
    assert_eq!(fine.n_nodes(), 216, "fine torus V");
    assert_eq!(referenced_nodes(&fine), fine.n_nodes(), "no orphan vertices");
    assert_eq!(hex_edge_count(&fine), 648, "fine torus E");
    assert_fresh_own_side(&fine, 27, "refined curved torus");
    assert_cells(&fine, 0.125, 1e-12, "refined curved torus");

    let geo = fine.geometry.as_ref().expect("geometry");
    assert_eq!(geo.order, 2);
    assert_eq!(
        distinct_torus_positions(&geo.coords, 4),
        1728,
        "geometry dofs: 0.25-lattice torus positions (MFEM 1728)"
    );
    assert_eq!(
        count_at_plane(&geo.coords, 0, 0.0),
        324,
        "own-side x=0 seam plane (MFEM 324)"
    );
    assert_eq!(
        count_at_plane(&geo.coords, 0, 3.0),
        324,
        "own-side x=3 seam plane (MFEM 324)"
    );
    // The fine vertex table keeps first-touch own-side picks: 0.5-lattice
    // positions only (the wrapped frame would invent 1.5-style positions).
    assert_eq!(distinct_torus_positions(&fine.coords, 2), 216, "vertex torus positions");
    assert_torus_h1(&fine, 6, "refined curved torus");
}

/// Order-2 hex, refine → fold (MFEM branch A): refining the *unfolded* curved
/// mesh keeps MFEM's shared `H1_3D_P2` nodes (2197 = 13³ dofs — probe
/// A-refined), and folding afterwards carries the geometry through unchanged.
#[test]
fn d832_curved_p2_refine_then_fold_matches_mfem() {
    let mut base = Mesh::<3>::make_cartesian_3d(3, 3, 3, ElementType::Hex8, 3.0, 3.0, 3.0, false);
    base.set_curvature(2);
    let fine = refine_uniform_3d(&base);
    assert_eq!(fine.n_elems(), 216);
    assert_eq!(fine.n_nodes(), 343, "unfolded 7³ vertex lattice (MFEM A-refined NV=343)");
    assert_eq!(
        fine.geom_n_nodes(),
        2197,
        "shared H1_3D_P2 dofs of the unfolded refinement (MFEM A-refined 2197)"
    );
    assert_cells(&fine, 0.125, 1e-12, "refined unfolded curved");

    let folded = fine.make_periodic(&PAIRS, 1e-10).expect("fold the refined mesh");
    assert_eq!(folded.n_elems(), 216);
    assert_eq!(folded.n_nodes(), 216, "folded torus V (MFEM A-periodic NV=216)");
    assert_eq!(referenced_nodes(&folded), folded.n_nodes(), "no orphan vertices");
    assert_eq!(folded.geom_order(), 2, "geometry survives the fold");
    assert_eq!(folded.geom_n_nodes(), 2197, "the fold does not touch the table");
    assert_cells(&folded, 0.125, 1e-12, "refined-then-folded torus");
    assert_torus_h1(&folded, 6, "refined-then-folded torus");
}

/// The tracked folded `L2_T1_3D_P2` file (MFEM's own `periodic-cube.geo`
/// product): refining it must keep 216·27 own-side rows and the exact cell
/// volumes (cube `[-1,1]³`, volume 8, cells 1/27 after refinement).
#[test]
fn d832_folded_p2_file_refine_matches_mfem() {
    let coarse = folded_p2_file();
    assert_eq!(coarse.n_nodes(), 27);
    assert_eq!(coarse.n_elems(), 27);
    assert_eq!(coarse.geom_order(), 2);
    assert_eq!(coarse.geom_n_nodes(), 27 * 27, "L2_T1_3D_P2 own-side dofs (MFEM 729)");
    assert_cells(&coarse, 8.0 / 27.0, 1e-6, "coarse folded p2 file");

    let fine = refine_uniform_3d(&coarse);
    assert_eq!(fine.n_elems(), 216);
    assert_eq!(fine.n_nodes(), 216);
    assert_fresh_own_side(&fine, 27, "refined folded p2 file");
    assert_cells(&fine, 1.0 / 27.0, 1e-6, "refined folded p2 file");
    let geo = fine.geometry.as_ref().expect("geometry");
    assert_eq!(geo.order, 2);
    assert_eq!(geo.n_nodes, 216 * 27, "own-side rows (MFEM L2_T1_3D_P2 5832)");
    assert_torus_h1(&fine, 6, "refined folded p2 file");
}

/// The tracked folded `L2_T1_3D_P1` file: control for the D814-2 own-side
/// path (already green before D832-2 — must stay green).  The file stores
/// 6-decimal coordinates, so the cell volumes spread ~1e-6 around 8/27
/// (MFEM probe: min 0.296295407, max 0.296296741).
#[test]
fn d832_folded_p1_file_refine_stays_own_side() {
    let coarse = folded_p1_file();
    assert_eq!(coarse.n_nodes(), 27);
    assert_eq!(coarse.geom_order(), 1);
    assert_eq!(coarse.geom_n_nodes(), 27 * 8, "L2_T1_3D_P1 own-side dofs (MFEM 216)");
    assert_cells(&coarse, 8.0 / 27.0, 2e-6, "coarse folded p1 file");

    let fine = refine_uniform_3d(&coarse);
    assert_eq!(fine.n_elems(), 216);
    assert_eq!(fine.n_nodes(), 216);
    assert_fresh_own_side(&fine, 8, "refined folded p1 file");
    assert_cells(&fine, 1.0 / 27.0, 2e-6, "refined folded p1 file");
    assert_torus_h1(&fine, 6, "refined folded p1 file");
}

/// Iterating the refinement keeps the torus quotient (12×12×12: V = E = F =
/// C = 1728, H1 = 1728/13824/46656, cell volume 1/64) with own-side rows.
#[test]
fn d832_curved_p2_second_refinement_stays_torus() {
    let fine = refine_uniform_3d(&curved_periodic());
    assert_fresh_own_side(&fine, 27, "first refinement");
    let fine2 = refine_uniform_3d(&fine);
    assert_eq!(fine2.n_elems(), 1728);
    assert_eq!(fine2.n_nodes(), 1728, "12³ torus V");
    assert_eq!(referenced_nodes(&fine2), fine2.n_nodes());
    assert_fresh_own_side(&fine2, 27, "second refinement");
    assert_cells(&fine2, 1.0 / 64.0, 1e-12, "second refinement");
    let geo = fine2.geometry.as_ref().expect("geometry");
    assert_eq!(geo.order, 2);
    assert_eq!(geo.n_nodes, 1728 * 27, "own-side rows");
    assert_eq!(
        distinct_torus_positions(&geo.coords, 8),
        13824,
        "geometry dofs: 0.125-lattice torus positions"
    );
    assert_torus_h1(&fine2, 12, "second refinement");
}
