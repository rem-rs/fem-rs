//! D835-2 (round 90 Lane B): the D832-2 own-side transport extended to the two
//! remaining public hex refinement entries — [`refine_hex8_uniform`] and
//! [`refine_nonconforming_hex_aniso`] — on periodically merged parents.
//!
//! ## The defect
//!
//! D832-2 (round 89) taught `refine_nonconforming_hex` (and hence
//! `refine_uniform_3d`) to keep the per-element **own-side** geometry of a
//! periodically merged parent (MFEM `MakePeriodic`'s discontinuous
//! `L2_T1_3D_Pk` nodes) across refinement.  Two public entries were not
//! covered:
//!
//! * `refine_hex8_uniform` handled the *curved* periodic parent (through
//!   `build_refined_hex_geometry`'s `is_periodic_merged_3d` branch) but
//!   dropped the *straight* periodic parent's order-1 own-side snapshot: the
//!   refined mesh came back with no geometry table at all and a vertex table
//!   in the merged (wrapped) frame — seam-crossing midpoints land at the
//!   wrong torus position and seam cells become inside-out;
//! * `refine_nonconforming_hex_aniso` carried no geometry transport at all —
//!   periodic or not, curved or straight.
//!
//! There is no MFEM oracle for anisotropic hex refinement (MFEM hexes only
//! refine isotropically), so the aniso pins below are mesh-level: own-side
//! rows, positive Jacobians, per-child volumes and own-side seam planes —
//! the same assertions the round-87 lesson requires (independent of any
//! space build).  The uniform straight-periodic case keeps the D832-2 MFEM
//! oracle numbers (216 cells, NV = 216, 1728 own-side dofs, seam planes
//! 144/144, H1 216/1728/5832).
//!
//! Fixtures: the in-process `make_cartesian_3d` + `make_periodic`
//! construction of `d832_hex_periodic_refine.rs`.

use fem_element::quadrature::hex_rule;
use fem_mesh::{
    refine_hex8_uniform, refine_nonconforming_hex_aniso, ElementType, HexRefineDir, Mesh,
    MeshTopology,
};
use fem_space::{FESpace, H1Space};

/// The tags `make_cartesian_3d` assigns to the box sides (see
/// `d832_hex_periodic_refine.rs` / `periodic_hex_3d.rs`).
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

fn curved_periodic() -> Mesh<3> {
    let mut base = Mesh::<3>::make_cartesian_3d(3, 3, 3, ElementType::Hex8, 3.0, 3.0, 3.0, false);
    base.set_curvature(2);
    base.make_periodic(&PAIRS, 1e-10).expect("make periodic")
}

/// Number of geometry dofs whose `axis` coordinate sits at `level` (±1e-9).
fn count_at_plane(coords: &[f64], axis: usize, level: f64) -> usize {
    coords
        .chunks_exact(3)
        .filter(|c| (c[axis] - level).abs() < 1e-9)
        .count()
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

/// Number of nodes referenced by at least one element.
fn referenced_nodes(mesh: &Mesh<3>) -> usize {
    let mut used = std::collections::BTreeSet::<u32>::new();
    for e in 0..mesh.n_elems() as u32 {
        used.extend(mesh.elem_nodes(e));
    }
    used.len()
}

/// The refined table's **fully discontinuous own-side layout**: element-major
/// rows of fresh, unshared dof ids (`n_nodes = NE · dpe`).
fn assert_fresh_own_side(mesh: &Mesh<3>, dpe: usize, what: &str) {
    let geo = mesh
        .geometry
        .as_ref()
        .unwrap_or_else(|| panic!("{what}: geometry table must survive refinement"));
    assert_eq!(geo.conn.len(), mesh.n_elems() as usize * dpe, "{what}: row size");
    assert_eq!(
        geo.n_nodes,
        mesh.n_elems() as usize * dpe,
        "{what}: per-element own-side rows — a dropped or shared/wrapped table \
         cannot represent both sides of a periodic seam"
    );
    assert!(
        geo.conn.iter().all(|&d| (d as usize) < geo.n_nodes),
        "{what}: geometry row entry out of range"
    );
}

/// Cell sanity for a uniformly sized refinement: positive Jacobian everywhere
/// and every element of volume `cell`.
fn assert_uniform_cells(mesh: &Mesh<3>, cell: f64, what: &str) {
    let rule = hex_rule(4);
    for e in 0..mesh.n_elems() as u32 {
        let mut vol = 0.0_f64;
        for (xi, w) in rule.points.iter().zip(rule.weights.iter()) {
            let (_jac, det, _x) = mesh.element_jacobian(e, xi);
            assert!(det > 0.0, "{what}: element {e} det {det} ≤ 0 at {xi:?}");
            vol += w * det;
        }
        assert!(
            (vol - cell).abs() < 1e-12,
            "{what}: element {e} volume {vol} ≠ {cell}"
        );
    }
}

/// H1 counts on the n³ torus (MFEM probe: refined straight torus
/// 216/1728/5832), with no orphan dofs.
fn assert_torus_h1(mesh: &Mesh<3>, n: usize, what: &str) {
    let cub = |k: usize| k * k * k;
    assert_eq!(H1Space::new(mesh.clone(), 1).n_dofs(), cub(n), "{what}: H1(1)");
    assert_eq!(H1Space::new(mesh.clone(), 2).n_dofs(), cub(2 * n), "{what}: H1(2)");
    assert_eq!(H1Space::new(mesh.clone(), 3).n_dofs(), cub(3 * n), "{what}: H1(3)");
}

/// First element whose merged-frame corner extent in x is 2 cells wide — a
/// seam cell spanning `[2, 3→0]` (only a seam cell folds that far).
fn first_x_seam_cell(mesh: &Mesh<3>) -> u32 {
    for e in 0..mesh.n_elems() as u32 {
        let ns = mesh.elem_nodes(e);
        let xs: Vec<f64> = ns.iter().map(|&n| mesh.coords_of(n)[0]).collect();
        let (lo, hi) = xs.iter().fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
        if hi - lo > 1.5 {
            return e;
        }
    }
    panic!("no seam cell found");
}

/// `refine_hex8_uniform` on the straight hex torus (all elements marked):
/// the refined geometry must stay the fully per-element own-side
/// `L2_T1_3D_P1` table (1728 dofs) — MFEM's fold→refine oracle, now on the
/// direct uniform entry instead of the `refine_uniform_3d` route.
#[test]
fn d835_hex8_uniform_straight_periodic_keeps_own_side_rows() {
    let coarse = straight_periodic();
    let all: Vec<u32> = (0..coarse.n_elems() as u32).collect();
    let (fine, _, _) = refine_hex8_uniform(&coarse, &all);
    assert_eq!(fine.n_elems(), 216, "6×6×6 cells");
    assert_eq!(fine.n_nodes(), 216, "fine torus V (MFEM NV=216)");
    assert_eq!(referenced_nodes(&fine), fine.n_nodes(), "no orphan vertices");
    assert_fresh_own_side(&fine, 8, "refined straight torus");
    assert_uniform_cells(&fine, 0.125, "refined straight torus");

    let geo = fine.geometry.as_ref().expect("geometry");
    assert_eq!(geo.order, 1);
    assert_eq!(
        distinct_torus_positions(&geo.coords, 2),
        216,
        "geometry dofs: 0.5-lattice torus positions (MFEM 216)"
    );
    assert_eq!(count_at_plane(&geo.coords, 0, 0.0), 144, "own-side x=0 seam plane (MFEM 144)");
    assert_eq!(count_at_plane(&geo.coords, 0, 3.0), 144, "own-side x=3 seam plane (MFEM 144)");
    assert_torus_h1(&fine, 6, "refined straight torus");
}

/// `refine_hex8_uniform` with a *single* seam cell marked: its children keep
/// own-side rows across the seam (the far child lives in `x ∈ [2.5, 3]`, not
/// the wrapped `[2.5, 0]`), and the unmarked elements are carried through
/// unchanged (`IDENTITY` rows).
#[test]
fn d835_hex8_uniform_partial_seam_cell_keeps_own_side_rows() {
    let coarse = straight_periodic();
    let seam = first_x_seam_cell(&coarse);
    let (fine, _, _) = refine_hex8_uniform(&coarse, &[seam]);
    assert_eq!(fine.n_elems(), 26 + 8);
    assert_fresh_own_side(&fine, 8, "partially refined straight torus");
    // children of the marked cell: volume 1/2; carried cells: volume 1.
    let rule = hex_rule(4);
    for e in 0..fine.n_elems() as u32 {
        let mut vol = 0.0_f64;
        for (xi, w) in rule.points.iter().zip(rule.weights.iter()) {
            let (_jac, det, _x) = fine.element_jacobian(e, xi);
            assert!(det > 0.0, "partial refine: element {e} det {det} ≤ 0");
            vol += w * det;
        }
        // The marked cell expands in place (isotropic octasection): its 8
        // children occupy the fine indices `seam .. seam + 8`, volume 1/8.
        let expect = if (seam..seam + 8).contains(&e) { 0.125 } else { 1.0 };
        assert!((vol - expect).abs() < 1e-12, "element {e} volume {vol} ≠ {expect}");
    }
    // Own-side values across the seam: the table knows x = 3.
    let geo = fine.geometry.as_ref().expect("geometry");
    assert!(
        count_at_plane(&geo.coords, 0, 3.0) > 0,
        "own-side x=3 dofs must exist on the far side of the seam"
    );
}

/// Guard (green since D832-2): the *curved* periodic parent through
/// `refine_hex8_uniform` rides `build_refined_hex_geometry`'s
/// `is_periodic_merged_3d` branch and must keep 216·27 own-side rows.
#[test]
fn d835_hex8_uniform_curved_periodic_keeps_own_side_rows() {
    let coarse = curved_periodic();
    let all: Vec<u32> = (0..coarse.n_elems() as u32).collect();
    let (fine, _, _) = refine_hex8_uniform(&coarse, &all);
    assert_eq!(fine.n_elems(), 216);
    assert_fresh_own_side(&fine, 27, "refined curved torus");
    assert_uniform_cells(&fine, 0.125, "refined curved torus");
    let geo = fine.geometry.as_ref().expect("geometry");
    assert_eq!(geo.order, 2);
    assert_eq!(
        distinct_torus_positions(&geo.coords, 4),
        1728,
        "geometry dofs: 0.25-lattice torus positions (MFEM 1728)"
    );
}

/// `refine_nonconforming_hex_aniso`, X split of every cell on the straight
/// torus: the geometry snapshot must survive as own-side rows (432 dofs),
/// every child a positive-Jacobian half cell — not the dropped table whose
/// wrapped vertex frame turns every seam child inside-out.
#[test]
fn d835_aniso_x_straight_periodic_keeps_own_side_rows() {
    let coarse = straight_periodic();
    let marked: Vec<(u32, HexRefineDir)> = (0..coarse.n_elems() as u32).map(|e| (e, HexRefineDir::X)).collect();
    let (fine, _) = refine_nonconforming_hex_aniso(&coarse, &marked, None);
    assert_eq!(fine.n_elems(), 54);
    assert_eq!(fine.n_nodes(), 54, "27 coarse + 27 x-midpoints (no orphans)");
    assert_eq!(referenced_nodes(&fine), fine.n_nodes());
    assert_fresh_own_side(&fine, 8, "aniso-X refined straight torus");
    assert_uniform_cells(&fine, 0.5, "aniso-X refined straight torus");

    // Own-side seam planes: 9 seam cells' far children carry 4 corners each
    // at x = 3, and 9 low children carry 4 corners each at x = 0.
    let geo = fine.geometry.as_ref().expect("geometry");
    assert_eq!(geo.order, 1);
    assert_eq!(count_at_plane(&geo.coords, 0, 3.0), 36, "own-side x=3 dofs");
    assert_eq!(count_at_plane(&geo.coords, 0, 0.0), 36, "own-side x=0 dofs");
    assert_eq!(
        distinct_torus_positions(&geo.coords, 2),
        54,
        "corner dofs on the 0.5×1×1 torus lattice"
    );
}

/// `refine_nonconforming_hex_aniso`, XY split of every cell on the *curved*
/// torus: 4 children per cell, each keeping its own order-2 row (216·27
/// own-side dofs), all positive-Jacobian quarter cells.
#[test]
fn d835_aniso_xy_curved_periodic_keeps_own_side_rows() {
    let coarse = curved_periodic();
    let marked: Vec<(u32, HexRefineDir)> =
         (0..coarse.n_elems() as u32).map(|e| (e, HexRefineDir::XY)).collect();
    let (fine, _) = refine_nonconforming_hex_aniso(&coarse, &marked, None);
    assert_eq!(fine.n_elems(), 108);
    assert_fresh_own_side(&fine, 27, "aniso-XY refined curved torus");
    assert_uniform_cells(&fine, 0.25, "aniso-XY refined curved torus");
    let geo = fine.geometry.as_ref().expect("geometry");
    assert_eq!(geo.order, 2);
    // Own-side dofs on both x seam planes (children of the x seam cells).
    assert!(count_at_plane(&geo.coords, 0, 3.0) > 0, "own-side x=3 dofs");
    assert!(count_at_plane(&geo.coords, 0, 0.0) > 0, "own-side x=0 dofs");
}
