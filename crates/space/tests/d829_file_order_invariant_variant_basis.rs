//! D829-1 / D829-2 (round 87, lane A) — file-order invariance of shared-entity
//! hp variant bases, and the hex × pyramid shared base quad face.
//!
//! D829-1: a SAME-order mixed-family shared entity holds ONE (entity, order)
//! variant whose basis used to be donated by the FIRST-ENCOUNTERING element
//! (round-86 state), so the variant's node positions drifted with the element
//! file order.  MFEM's ruling is collection-global: probe
//! `tmp/d86a/out_p33_gl.txt` vs `tmp/d87a/out_p33_cu.txt` (same tet p3 × pyr
//! p3 mesh, only the collection flipped) shows every single-variant dof —
//! including the tet's OWN edge dofs — at GLL 0.2764/0.7236 under the default
//! Gauss-Lobatto collection and at equispaced 1/3, 2/3 under
//! `BasisType::ClosedUniform`.  fem-rs is a per-family franken-collection
//! (TetPk equispaced cells, GLL elsewhere — the d176 divergence), so the
//! faithful translation is the GL-dominant sticky-OR ruling: any GLL-family
//! user makes the shared variant GLL; only an all-tet user list keeps it
//! equispaced.  `collect_edge_variants`/`collect_face_variants` fold the
//! flag with `|` now — the basis is a function of the user-family SET, not
//! of the file order.
//!
//! D829-2: the hex × pyramid shared BASE quad face (pyramid Fuentes
//! j-reversed layout vs hex ascending GLL tensor).  MFEM probe
//! `tmp/d87a/hex_pyr_face_probe.cpp` (hex p2 under a pyramid p3, shared base
//! face; outputs `out_h2p3_hexfirst.txt` / `out_h2p3_pyrfirst.txt`):
//!
//! ```text
//! hex-first: raw face block [43,44,45,46] = ASCENDING tensor slots;
//!            pyr element row lists [45,46,43,44]  (QuadDofOrd compensation)
//! pyr-first: raw face block [38..41]     = FUENTES slots;
//!            mesh-level face verts flip to (3,2,1,0)
//! constraint rows per PHYSICAL point identical across both file orders,
//! e.g. the p3 face dof at (cp1, cp1):
//!   <- (v0, +0.104721360) (v1, -0.040000000) (v2, +0.015278640)
//!      (v3, -0.040000000) (e(0,1), +0.258885438) (e(1,2), -0.098885438)
//!      (e(2,3), -0.098885438) (e(0,3), +0.258885438) (face-p2, +0.640000000)
//! ```
//!
//! So MFEM itself is file-order dependent at the SLOT level (the face's
//! layout follows the face-creating element) and invariant at the PHYSICS
//! level (one row per dof, consistent with the stored layout; each element
//! compensates through `QuadDofOrd`).  fem-rs mirrors this: the builder's
//! per-face first-encounter `reversed_y` flag = MFEM's face-creator layout,
//! and the detector's two-sided row emission dedups to the FIRST-ENCOUNTER
//! side's row (stable sort + push order), which is exactly the side the
//! builder coords used — self-consistent without any code change.  These
//! tests PIN that coupling (the convention pins and the interpolation
//! identities are the teeth: flipping the builder flag, the detector arm
//! flags, or the dedup order turns them red).
//!
//! Fixtures `data/d829_hexpyr_hex_first.msh` / `data/d829_hexpyr_pyr_first.msh`:
//! one Hex8 under one Pyramid5 sharing the base quad face {0,1,2,3}, same
//! physical mesh, element file order swapped; node ids identical.
//!
//! ```text
//! cargo test -p fem-space --test d829_file_order_invariant_variant_basis -- --nocapture
//! ```

use fem_io::gmsh::read_msh;
use fem_mesh::{Mesh, MeshTopology};
use fem_space::dof_manager::{DofManager, EdgeKey, FaceKey};
use fem_space::p_refine::{build_variable_order_dof_manager, detect_p_constraints};

const FIXTURE_TET_FIRST: &str = include_str!("../../../data/d827_mixed_tet_first.msh");
const FIXTURE_PYR_FIRST: &str = include_str!("../../../data/d827_mixed_pyr_first.msh");
const FIXTURE_HEX_FIRST: &str = include_str!("../../../data/d829_hexpyr_hex_first.msh");
const FIXTURE_PYR_FIRST_HEX: &str = include_str!("../../../data/d829_hexpyr_pyr_first.msh");

/// Shared edges of the d827 tet/pyr fixture (node ids identical in both file
/// orders).
const D827_SHARED_EDGES: [(u32, u32); 3] = [(0, 1), (0, 4), (1, 4)];

/// cp(1)/cp(2) of the closed Gauss-Lobatto p3 points on [0, 1].
fn cp1() -> f64 {
    0.5 * (1.0 - 1.0 / 5.0_f64.sqrt())
}
fn cp2() -> f64 {
    0.5 * (1.0 + 1.0 / 5.0_f64.sqrt())
}

fn tet_first_mesh() -> Mesh<3> {
    read_msh(FIXTURE_TET_FIRST.as_bytes()).expect("parse tet-first fixture").into_3d().expect("3-D")
}
fn pyr_first_mesh() -> Mesh<3> {
    read_msh(FIXTURE_PYR_FIRST.as_bytes()).expect("parse pyr-first fixture").into_3d().expect("3-D")
}
fn hex_first_mesh() -> Mesh<3> {
    read_msh(FIXTURE_HEX_FIRST.as_bytes()).expect("parse hex-first fixture").into_3d().expect("3-D")
}
fn pyr_first_hex_mesh() -> Mesh<3> {
    read_msh(FIXTURE_PYR_FIRST_HEX.as_bytes())
        .expect("parse pyr-first hex fixture").into_3d().expect("3-D")
}

/// Lagrange weights of `nodes` (ascending, endpoints included) at `t`.
fn lag(nodes: &[f64], t: f64) -> Vec<f64> {
    (0..nodes.len()).map(|j| {
        nodes.iter().enumerate()
            .filter(|&(m, _)| m != j)
            .fold(1.0, |acc, (_, &nm)| acc * (t - nm) / (nodes[j] - nm))
    }).collect()
}

/// The p2 Gauss-Lobatto master weights on {0, 1/2, 1}.
fn w2(t: f64) -> Vec<f64> {
    lag(&[0.0, 0.5, 1.0], t)
}

fn bits(x: f64) -> u64 {
    x.to_bits()
}

/// Exact coordinate key of a dof (bit-level: the builders compute every
/// coordinate from the same formula, so equal physics gives equal bits).
fn coord_key(dm: &DofManager, dof: u32) -> [u64; 3] {
    let c = dm.dof_coord(dof);
    [bits(c[0]), bits(c[1]), bits(c[2])]
}

/// Sorted (parent dof, weight) pairs of a constraint row.
fn row_of(constraints: &[fem_space::p_refine::PRefineConstraint], dof: u32)
    -> Vec<(u32, f64)>
{
    let c = constraints.iter().find(|c| c.constrained == dof)
        .unwrap_or_else(|| panic!("no constraint on dof {dof}"));
    let mut ps = c.parents.clone();
    ps.sort_by_key(|&(d, _)| d);
    ps
}

fn assert_weights(actual: &[(u32, f64)], expected: &[(u32, f64)], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: parent count");
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert_eq!(a.0, e.0, "{context}: parent dof");
        assert!((a.1 - e.1).abs() < 1e-12, "{context}: weight {} vs {}", a.1, e.1);
    }
}

/// Assert `u` (sampled at every dof coordinate) satisfies every constraint —
/// the teeth for the coords <-> constraint-rows convention coupling (D829-2):
/// a row computed in the wrong face layout evaluates the master trace at a
/// point other than the constrained dof's coordinate and breaks the identity.
fn assert_identities(
    dm: &DofManager,
    constraints: &[fem_space::p_refine::PRefineConstraint],
    u: &[f64],
) {
    for c in constraints {
        let lhs = u[c.constrained as usize];
        let rhs: f64 = c.parents.iter().map(|&(d, w)| w * u[d as usize]).sum();
        assert!(
            (lhs - rhs).abs() < 1e-12,
            "constrained dof {} (coord {:?}): {lhs} != {rhs}",
            c.constrained,
            dm.dof_coord(c.constrained)
        );
    }
}

// ═══════════════════════════ D829-1 ═══════════════════════════

/// Same-order tet p3 × pyramid p3 (single shared variants): after the
/// sticky-GLL ruling the shared variant's node set must be GLL in BOTH file
/// orders (red before the fix: the tet-first file placed it equispaced),
/// private variants must keep their own family's basis, and the whole dof
/// coordinate set must be bit-identical across the file order swap.
#[test]
fn d829_same_order_shared_variant_basis_independent_of_file_order() {
    let mut coord_sets: Vec<std::collections::BTreeSet<[u64; 3]>> = Vec::new();
    for (name, mesh) in [("tet-first", tet_first_mesh()), ("pyr-first", pyr_first_mesh())] {
        let orders: Vec<u8> = vec![3, 3];
        let dm = build_variable_order_dof_manager(&mesh, &orders);
        let constraints = detect_p_constraints(&dm, &mesh, &orders);

        assert_eq!(dm.n_dofs, 47, "{name}: probe out_p33_gl.txt NDofs");
        assert!(constraints.is_empty(), "{name}: no mixed-order rows at p=[3,3]");

        for &(a, b) in &D827_SHARED_EDGES {
            let variants = &dm.edge_variants[&EdgeKey::new(a, b)];
            assert_eq!(variants.len(), 1, "{name}: one variant per order on ({a},{b})");
            assert_eq!(variants[0].0, 3, "{name}: shared variant order");
            let dofs = &variants[0].1;
            assert_eq!(dofs.len(), 2, "{name}: shared edge dof count");
            // sticky-GLL: the shared variant sits at cp1/cp2 in BOTH files.
            let (ca, cb) = (dm.dof_coord(a), dm.dof_coord(b));
            for (k, &dof) in dofs.iter().enumerate() {
                let t = [cp1(), cp2()][k];
                for d in 0..3 {
                    assert!(
                        (dm.dof_coord(dof)[d] - ((1.0 - t) * ca[d] + t * cb[d])).abs() < 1e-12,
                        "{name}: shared edge ({a},{b}) dof {dof} must sit at the GLL \
                         position t={t} (sticky-GLL, D829-1)"
                    );
                }
            }
        }

        // The shared tri face (0,1,4): single p3 variant, one interior dof at
        // the GLL barycentric centroid (orientation-invariant point).
        let face_dofs = &dm.face_variants[&FaceKey::new(0, 1, 4)];
        assert_eq!(face_dofs.len(), 1, "{name}: one tri-face variant");
        assert_eq!(face_dofs[0].1.len(), 1, "{name}: p3 tri face interior dofs");
        let c = dm.dof_coord(face_dofs[0].1[0]);
        let expected: Vec<f64> = (0..3).map(|d| {
            (mesh.node_coords(0)[d] + mesh.node_coords(1)[d] + mesh.node_coords(4)[d]) / 3.0
        }).collect();
        for d in 0..3 {
            assert!((c[d] - expected[d]).abs() < 1e-12, "{name}: shared tri face centroid");
        }

        // Private variants keep their own family basis (the sticky rule must
        // not over-unify): the tet-private edge (0,5) stays equispaced, the
        // pyramid-private base edge (1,2) stays GLL.
        let (ca, cb) = (dm.dof_coord(0), dm.dof_coord(5));
        for (k, &dof) in dm.edge_variants[&EdgeKey::new(0, 5)][0].1.iter().enumerate() {
            let t = (k + 1) as f64 / 3.0;
            for d in 0..3 {
                assert!(
                    (dm.dof_coord(dof)[d] - ((1.0 - t) * ca[d] + t * cb[d])).abs() < 1e-12,
                    "{name}: tet-private edge (0,5) stays equispaced"
                );
            }
        }
        let (ca, cb) = (dm.dof_coord(1), dm.dof_coord(2));
        for (k, &dof) in dm.edge_variants[&EdgeKey::new(1, 2)][0].1.iter().enumerate() {
            let t = [cp1(), cp2()][k];
            for d in 0..3 {
                assert!(
                    (dm.dof_coord(dof)[d] - ((1.0 - t) * ca[d] + t * cb[d])).abs() < 1e-12,
                    "{name}: pyramid-private edge (1,2) stays GLL"
                );
            }
        }

        coord_sets.push((0..dm.n_dofs as u32)
            .map(|d| coord_key(&dm, d)).collect());
    }
    // The whole coordinate set is bit-identical across the file order swap.
    assert_eq!(coord_sets[0], coord_sets[1],
        "D829-1: the same-order single-variant dofs must not drift with the \
         element file order");
}

/// Mixed orders stay per-variant (control): tet p3 × pyr p2 keeps the tet
/// side equispaced and the pyramid side GLL in BOTH file orders — the
/// sticky-GLL rule only binds variants shared by both families at the SAME
/// order.
#[test]
fn d829_mixed_order_variants_still_per_family() {
    for (name, mesh, orders) in [
        ("tet-first", tet_first_mesh(), vec![3u8, 2]),
        ("pyr-first", pyr_first_mesh(), vec![2u8, 3]),
    ] {
        let dm = build_variable_order_dof_manager(&mesh, &orders);
        for &(a, b) in &D827_SHARED_EDGES {
            let variants = &dm.edge_variants[&EdgeKey::new(a, b)];
            assert_eq!(variants.len(), 2, "{name}: two variants on ({a},{b})");
            let (ca, cb) = (dm.dof_coord(a), dm.dof_coord(b));
            let tet_side = variants.iter().find(|(p, _)| *p == 3).unwrap();
            for (k, &dof) in tet_side.1.iter().enumerate() {
                let t = (k + 1) as f64 / 3.0;
                for d in 0..3 {
                    assert!(
                        (dm.dof_coord(dof)[d] - ((1.0 - t) * ca[d] + t * cb[d])).abs() < 1e-12,
                        "{name}: tet p3 variant of ({a},{b}) stays equispaced"
                    );
                }
            }
            let pyr_side = variants.iter().find(|(p, _)| *p == 2).unwrap();
            assert_eq!(pyr_side.1.len(), 1, "{name}: pyr p2 midpoint");
            for d in 0..3 {
                assert!(
                    (dm.dof_coord(pyr_side.1[0])[d] - (ca[d] + cb[d]) / 2.0).abs() < 1e-12,
                    "{name}: pyr p2 midpoint dof coordinate"
                );
            }
        }
    }
}

// ═══════════════════════════ D829-2 ═══════════════════════════

/// The 9 parents and 9 weights of a shared-base-face p3 row, per ascending
/// tensor slot (r, t) — the MFEM probe oracle (rows 43-46 of
/// `out_h2p3_hexfirst.txt`).  Parent dof ids: the 4 face vertices, the 4
/// base-edge p2 dofs, the base-face p2 dof.
#[allow(clippy::type_complexity)]
fn expected_face_row(
    dm: &DofManager,
    corners: &[u32; 4],
    e: &[u32; 4],
    f2: u32,
    r: f64,
    t: f64,
) -> Vec<(u32, f64)> {
    let w = w2(r);
    let u = w2(t);
    let grid: [(usize, usize, u32); 9] = [
        (0, 0, corners[0]), (1, 0, e[0]), (2, 0, corners[1]),
        (0, 1, e[3]), (1, 1, f2), (2, 1, e[1]),
        (0, 2, corners[3]), (1, 2, e[2]), (2, 2, corners[2]),
    ];
    let mut ps: Vec<(u32, f64)> = grid.iter()
        .map(|&(a, b, dof)| (dof, w[a] * u[b]))
        .collect();
    ps.sort_by_key(|&(d, _)| d);
    let _ = dm;
    ps
}

/// D829-2 negative pin, hex-first file (the shared base face's stored layout
/// = the hex's ascending GLL tensor, exactly MFEM's face-creator layout):
/// structure (60 dofs, 12 rows = 8 edge + 4 face), ascending slot
/// coordinates, and the MFEM probe's face rows verbatim.
#[test]
fn d829_hexpyr_shared_base_face_matches_mfem_probe() {
    let mesh = hex_first_mesh();
    assert_eq!(mesh.element_type(0), fem_mesh::ElementType::Hex8);
    let orders: Vec<u8> = vec![2, 3];
    let dm = build_variable_order_dof_manager(&mesh, &orders);
    let constraints = detect_p_constraints(&dm, &mesh, &orders);

    // MFEM NDofs=60 (probe out_h2p3_hexfirst.txt).
    assert_eq!(dm.n_dofs, 60);
    assert_eq!(constraints.len(), 12, "8 shared-edge rows + 4 base-face rows");

    // Shared base edges: p3 slave dofs interpolate the p2 master with the
    // GLL weights {0.3236, 0.8, -0.1236} (probe rows 10/11, 13/14, 16/17,
    // 19/20).
    let base_edges = [(0u32, 1u32), (1, 2), (2, 3), (0, 3)];
    for &(a, b) in &base_edges {
        let master = dm.edge_variants[&EdgeKey::new(a, b)].iter()
            .find(|(p, _)| *p == 2).unwrap().1[0];
        let slaves = dm.edge_variants[&EdgeKey::new(a, b)].iter()
            .find(|(p, _)| *p == 3).unwrap().1.clone();
        assert_eq!(slaves.len(), 2);
        for (k, &dof) in slaves.iter().enumerate() {
            let w = w2([cp1(), cp2()][k]);
            assert_weights(
                &row_of(&constraints, dof),
                &[(a as u32, w[0]), (b as u32, w[2]), (master, w[1])],
                &format!("base edge ({a},{b}) p3 dof {k}"),
            );
        }
    }

    // The base face {0,1,2,3}: p2 master dof + 4 p3 slots.  hex-first =>
    // the stored layout is the ASCENDING tensor (MFEM raw block [43..46]).
    let fkey = FaceKey::new(0, 1, 2);
    let fvariants = &dm.face_variants[&fkey];
    assert_eq!(fvariants.len(), 2, "p2 master + p3 slave variants");
    let f2 = fvariants.iter().find(|(p, _)| *p == 2).unwrap().1[0];
    let f3 = fvariants.iter().find(|(p, _)| *p == 3).unwrap().1.clone();
    assert_eq!(f3.len(), 4);

    let g = [cp1(), cp2()];
    let corners = [0u32, 1, 2, 3];
    let e = [
        dm.edge_variants[&EdgeKey::new(0, 1)].iter().find(|(p, _)| *p == 2).unwrap().1[0],
        dm.edge_variants[&EdgeKey::new(1, 2)].iter().find(|(p, _)| *p == 2).unwrap().1[0],
        dm.edge_variants[&EdgeKey::new(2, 3)].iter().find(|(p, _)| *p == 2).unwrap().1[0],
        dm.edge_variants[&EdgeKey::new(0, 3)].iter().find(|(p, _)| *p == 2).unwrap().1[0],
    ];
    for (j, &dof) in f3.iter().enumerate() {
        let (ix, iy) = (j % 2, j / 2);
        let (r, t) = (g[ix], g[iy]); // ascending: NOT j-reversed (hex first)
        // slot coordinate: bilinear over the canon corner cycle (0,1,2,3).
        let c: Vec<[f64; 3]> = (0..4).map(|n| {
            let c = mesh.node_coords(corners[n]); [c[0], c[1], c[2]]
        }).collect();
        let xyz = dm.dof_coord(dof);
        for d in 0..3 {
            let expect = (1.0 - r) * (1.0 - t) * c[0][d]
                + r * (1.0 - t) * c[1][d]
                + r * t * c[2][d]
                + (1.0 - r) * t * c[3][d];
            assert!((xyz[d] - expect).abs() < 1e-12,
                "hex-first face slot {j} must sit at the ascending tensor point");
        }
        // MFEM probe row (43-46): 9 parents, weights w2(r) (x) w2(t).
        assert_weights(
            &row_of(&constraints, dof),
            &expected_face_row(&dm, &corners, &e, f2, r, t),
            &format!("hex-first base face slot {j} (r={r}, t={t})"),
        );
    }

    // Teeth: the surviving rows must be consistent with the stored
    // coordinates — a quadratic field sampled at the dof coordinates
    // satisfies every constraint exactly (a convention mismatch would
    // evaluate the master trace at the wrong face point and break this).
    let u: Vec<f64> = (0..dm.n_dofs as u32)
        .map(|d| {
            let x = dm.dof_coord(d);
            2.0 * x[0] * x[0] - 3.0 * x[1] * x[2] + 0.5 * x[0]
        }).collect();
    assert_identities(&dm, &constraints, &u);
}

/// D829-2 negative pin, pyramid-first file: the stored base-face layout is
/// the Fuentes j-reversed tensor (MFEM's face-creator layout again — probe
/// rows 38-41), yet the PHYSICAL dof coordinate set and the constraint rows
/// (matched by coordinate) are bit-identical to the hex-first run.
#[test]
fn d829_hexpyr_face_rows_independent_of_element_file_order() {
    let mesh = pyr_first_hex_mesh();
    assert_eq!(mesh.element_type(0), fem_mesh::ElementType::Pyramid5);
    let orders: Vec<u8> = vec![3, 2]; // (pyr, hex) — the file swapped elements
    let dm = build_variable_order_dof_manager(&mesh, &orders);
    let constraints = detect_p_constraints(&dm, &mesh, &orders);

    assert_eq!(dm.n_dofs, 60);
    assert_eq!(constraints.len(), 12);

    // Fuentes pin: face slot 0 sits at (cp1, cp_{p-1}) = (cp1, cp2), slot 3
    // at (cp2, cp1) — the j-REVERSED layout (pyr first-encounter).  A
    // regression that flips the builder flag or the dedup order moves these.
    let fkey = FaceKey::new(0, 1, 2);
    let f3 = dm.face_variants[&fkey].iter().find(|(p, _)| *p == 3).unwrap().1.clone();
    let corners = [0u32, 1, 2, 3];
    let c: Vec<[f64; 3]> = (0..4).map(|n| {
        let c = mesh.node_coords(corners[n]); [c[0], c[1], c[2]]
    }).collect();
    let bilinear = |r: f64, t: f64, d: usize| {
        (1.0 - r) * (1.0 - t) * c[0][d]
            + r * (1.0 - t) * c[1][d]
            + r * t * c[2][d]
            + (1.0 - r) * t * c[3][d]
    };
    let at = |dof: u32| dm.dof_coord(dof);
    for d in 0..3 {
        assert!((at(f3[0])[d] - bilinear(cp1(), cp2(), d)).abs() < 1e-12,
            "pyr-first face slot 0 at Fuentes (cp1, cp2)");
        assert!((at(f3[3])[d] - bilinear(cp2(), cp1(), d)).abs() < 1e-12,
            "pyr-first face slot 3 at Fuentes (cp2, cp1)");
    }

    // Physics invariance: the constraint rows matched by dof coordinates and
    // the whole coordinate multiset must be bit-identical to the hex-first
    // run (rows keyed the round-86 way).
    let mesh_h = hex_first_mesh();
    let orders_h: Vec<u8> = vec![2, 3];
    let dm_h = build_variable_order_dof_manager(&mesh_h, &orders_h);
    let cs_h = detect_p_constraints(&dm_h, &mesh_h, &orders_h);

    let mut map_h = std::collections::BTreeMap::new();
    for row in &cs_h {
        let mut ps: Vec<([u64; 3], u64)> = row.parents.iter()
            .map(|&(d, w)| (coord_key(&dm_h, d), bits(w))).collect();
        ps.sort_unstable();
        map_h.insert(coord_key(&dm_h, row.constrained), ps);
    }
    let mut map_p = std::collections::BTreeMap::new();
    for row in &constraints {
        let mut ps: Vec<([u64; 3], u64)> = row.parents.iter()
            .map(|&(d, w)| (coord_key(&dm, d), bits(w))).collect();
        ps.sort_unstable();
        map_p.insert(coord_key(&dm, row.constrained), ps);
    }
    assert_eq!(map_h, map_p,
        "D829-2: face/edge rows must be physically identical across the file \
         order swap");

    let mut set_h: std::collections::BTreeSet<[u64; 3]> = (0..dm_h.n_dofs as u32)
        .map(|d| coord_key(&dm_h, d)).collect();
    let set_p: std::collections::BTreeSet<[u64; 3]> = (0..dm.n_dofs as u32)
        .map(|d| coord_key(&dm, d)).collect();
    set_h.extend(set_p.iter().copied());
    assert_eq!(set_h.len(), dm_h.n_dofs, "coordinate sets coincide (and no \
        duplicates mask a drift)");

    // Teeth: rows consistent with the stored (Fuentes) coordinates.
    let u: Vec<f64> = (0..dm.n_dofs as u32)
        .map(|d| {
            let x = dm.dof_coord(d);
            2.0 * x[0] * x[0] - 3.0 * x[1] * x[2] + 0.5 * x[0]
        }).collect();
    assert_identities(&dm, &constraints, &u);
}
