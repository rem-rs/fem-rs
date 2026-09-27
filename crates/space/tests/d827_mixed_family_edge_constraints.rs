//! D827-1 (round 86, lane A) — mixed-family hp EDGE constraints dispatch the
//! 1-D basis per (edge, order) VARIANT.
//!
//! `detect_p_constraints` used to take its 1-D node-set flag from
//! `elem_uses_gll(mesh.element_type(0))` (homogeneous-mesh assumption): on a
//! mixed tet × pyramid mesh every mixed-order edge constraint was computed
//! with element 0's family basis, misplacing the other family's half.  The
//! fix tracks the basis per (edge, order) VARIANT (D829-1, round 87: the
//! flag folds with a sticky OR over the users' family flags) in the
//! builder's edge DOF coordinates AND the constraint node positions.
//!
//! MFEM 4.10 ground truth (probe `tmp/d86a/mixed_edge_probe.cpp`, outputs
//! `out_p32_gl.txt` / `out_p32_cu.txt` / `out_p23_gl.txt` / `out_p43_gl.txt`
//! / `out_p33_gl.txt`; one Tet4 + one Pyramid5 sharing the tri face (0,1,4),
//! `EnsureNCMesh(true)` — the meshgen gate (mesh/mesh.cpp:11794) ignores the
//! tet+pyramid mesh without the simplices opt-in, self-verified: the
//! variable-order space then aborts at fem/fespace.cpp:2776):
//!
//! ```text
//! VariableOrderMinimumRule (fem/fespace.cpp:1094) masters a mixed edge at
//! the lowest ADJACENT order and evaluates
//!   slave_fe->GetTransferMatrix(*master_fe, ...)   with
//!   master/slave_fe = fec->GetFE(Geometry::SEGMENT, p/q)
//! — the COLLECTION-level segment FE for both sides, independent of the
//! adjacent cell families:
//!   GL collection:  constrained 7 <- (0, +0.323606798) (1, -0.123606798) (6, +0.800000000)
//!   CU collection:  constrained 7 <- (0, +0.222222222) (1, -0.111111111) (6, +0.888888889)
//! (identical mixed mesh; flipping only the collection basis flips every
//! edge row GLL <-> equispaced).  Shared edges hold ONE variant per order
//! (p33 probe: variant orders "3", no constraints, NDofs=47) — MFEM never
//! duplicates variants per family.
//! ```
//!
//! fem-rs's cell bases are per-family by long-standing convention (TetPk
//! equispaced, GLL elsewhere — the documented d176 TetPk divergence), so the
//! faithful translation is variant-level dispatch: each variant's node
//! positions follow the family that uses it (equispaced on the tet side,
//! GLL on the pyramid side of the same edge).  A variant shared by two
//! families at the SAME order takes the sticky-GLL basis (D829-1: any
//! GLL-family user makes it GLL; MFEM's own ruling is collection-global —
//! probe `tmp/d87a/out_p33_cu.txt`).
//!
//! Fixtures `data/d827_mixed_tet_first.msh` / `data/d827_mixed_pyr_first.msh`
//! are the same physical mesh with the element file order swapped (tet
//! first / pyramid first); node ids are identical, so shared edges are
//! (0,1), (0,4), (1,4) in both.
//!
//! ```text
//! cargo test -p fem-space --test d827_mixed_family_edge_constraints -- --nocapture
//! ```

use fem_io::gmsh::read_msh;
use fem_mesh::{Mesh, MeshTopology};
use fem_space::dof_manager::{DofManager, EdgeKey};
use fem_space::p_refine::{build_variable_order_dof_manager, detect_p_constraints,
    PRefineConstraint};

const FIXTURE_TET_FIRST: &str = include_str!("../../../data/d827_mixed_tet_first.msh");
const FIXTURE_PYR_FIRST: &str = include_str!("../../../data/d827_mixed_pyr_first.msh");

/// Node ids of the shared tri face (0,1,4) and its edges.
const SHARED_EDGES: [(u32, u32); 3] = [(0, 1), (0, 4), (1, 4)];

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

/// Lagrange weights of `nodes` (ascending, endpoints included) at `t`.
fn lag(nodes: &[f64], t: f64) -> Vec<f64> {
    (0..nodes.len()).map(|j| {
        nodes.iter().enumerate()
            .filter(|&(m, _)| m != j)
            .fold(1.0, |acc, (_, &nm)| acc * (t - nm) / (nodes[j] - nm))
    }).collect()
}

/// Weights of the order-`p` master Lagrange basis on the unit edge evaluated
/// at `t` (`p+1` equispaced or closed-GLL nodes).
fn master_weights(p: u8, t: f64, gll: bool) -> Vec<f64> {
    let nodes: Vec<f64> = (0..=p as usize)
        .map(|j| if gll {
            match p as usize {
                2 => [0.0, 0.5, 1.0][j],
                3 => [0.0, cp1(), cp2(), 1.0][j],
                _ => panic!("master_weights: p {p} not pinned"),
            }
        } else {
            j as f64 / p as f64
        })
        .collect();
    lag(&nodes, t)
}

/// Constraint map dof -> parents (sorted by dof id) for the given dofs.
fn rows_of(
    constraints: &[PRefineConstraint],
    dofs: &[u32],
) -> Vec<(u32, Vec<(u32, f64)>)> {
    dofs.iter().map(|&d| {
        let c = constraints.iter().find(|c| c.constrained == d)
            .unwrap_or_else(|| panic!("no constraint for dof {d}"));
        let mut parents = c.parents.clone();
        parents.sort_by_key(|&(p, _)| p);
        (d, parents)
    }).collect()
}

fn assert_weights(actual: &[(u32, f64)], expected: &[(u32, f64)], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: parent count");
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert_eq!(a.0, e.0, "{context}: parent dof");
        assert!((a.1 - e.1).abs() < 1e-12, "{context}: weight {} vs {}", a.1, e.1);
    }
}

/// The dofs of the order-`p` variant of a shared edge.
fn variant_dofs(dm: &DofManager, edge: (u32, u32), p: u8) -> Vec<u32> {
    dm.edge_variants[&EdgeKey::new(edge.0, edge.1)]
        .iter().find(|(q, _)| *q == p)
        .unwrap_or_else(|| panic!("no p{p} variant on edge {edge:?}"))
        .1.clone()
}

/// A quadratic test field (degree <= 2 masters).
fn quad_field(x: &[f64]) -> f64 {
    2.0 * x[0] * x[0] - 3.0 * x[1] * x[2] + 0.5 * x[0]
}

/// A cubic test field (degree <= 3 masters).
fn cubic_field(x: &[f64]) -> f64 {
    x[0] * x[0] * x[1] - 2.0 * x[2] * x[2] * x[0] + 0.5 * x[1] * x[2] - x[0]
}

/// Assert `u` (a field sampled at every dof coordinate) satisfies every
/// constraint (master-trace interpolation identity).
fn assert_identities(
    dm: &DofManager,
    constraints: &[PRefineConstraint],
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

/// Tet p3 x pyramid p2, tet first in the file: the tet-side p3 edge dofs sit
/// at EQUISPACED positions (TetPk family) and interpolate the p2 master at
/// {0, 1/2, 1}: weights 2/9, 8/9, -1/9 (both families' p2 node sets coincide
/// at the midpoint, so this pair pins the tet side without basis ambiguity).
#[test]
fn d827_tet_p3_pyr_p2_tet_slave_stays_equispaced() {
    let mesh = tet_first_mesh();
    assert_eq!(mesh.element_type(0), fem_mesh::ElementType::Tet4);
    let orders: Vec<u8> = vec![3, 2];
    let dm = build_variable_order_dof_manager(&mesh, &orders);
    let constraints = detect_p_constraints(&dm, &mesh, &orders);

    // 3 shared edges x 2 tet p3 edge dofs + 1 mixed tri face dof.
    assert_eq!(constraints.len(), 7, "6 edge rows + 1 tri-face row");

    for &(a, b) in &SHARED_EDGES {
        let dofs = variant_dofs(&dm, (a, b), 3);
        assert_eq!(dofs.len(), 2, "tet p3 variant of edge ({a},{b})");
        let mid = variant_dofs(&dm, (a, b), 2);
        assert_eq!(mid.len(), 1, "p2 variant midpoint dof");
        // tet-side equispaced DOF COORDINATES: 1/3 and 2/3 along (a -> b)
        // (canonical key direction; (a,b) here is already ascending).
        let (ca, cb) = (dm.dof_coord(a as u32), dm.dof_coord(b as u32));
        for (k, &dof) in dofs.iter().enumerate() {
            let t = (k + 1) as f64 / 3.0;
            let xyz = dm.dof_coord(dof);
            for d in 0..3 {
                assert!((xyz[d] - ((1.0 - t) * ca[d] + t * cb[d])).abs() < 1e-12,
                    "tet p3 edge dof {dof} coordinate along ({a},{b})");
            }
        }
        // constraint rows: equispaced slave positions on the p2 master
        for (k, &dof) in dofs.iter().enumerate() {
            let t = (k + 1) as f64 / 3.0;
            let w = master_weights(2, t, false);
            let expected = vec![(a, w[0]), (b, w[2]), (mid[0], w[1])];
            let rows = rows_of(&constraints, &[dof]);
            assert_weights(&rows[0].1, &expected, &format!("edge ({a},{b}) dof@t={t}"));
        }
    }

    // Identity: a quadratic field satisfies every constraint exactly.
    let n = dm.n_dofs as usize;
    let u: Vec<f64> = (0..n as u32)
        .map(|d| quad_field(dm.dof_coord(d))).collect();
    assert_identities(&dm, &constraints, &u);
}

/// Tet p2 x pyramid p3, TET FIRST in the file: the pyramid-side p3 edge dofs
/// sit at the closed-GLL positions cp1/cp2 (Fuentes family) and interpolate
/// the p2 master with GLL weights {L(0), L(1), L(1/2)} =
/// {0.3236.., -0.1236.., 0.8}.  Element 0 is the tet — the pre-D827-1 code
/// took `gll` from element 0 and computed equispaced slave positions
/// (weights {2/9, 8/9, -1/9}): this is the red test.
#[test]
fn d827_tet_p2_pyr_p3_pyr_slave_takes_gll_despite_tet_first() {
    let mesh = tet_first_mesh();
    let orders: Vec<u8> = vec![2, 3];
    let dm = build_variable_order_dof_manager(&mesh, &orders);
    let constraints = detect_p_constraints(&dm, &mesh, &orders);

    assert_eq!(constraints.len(), 7, "6 edge rows + 1 tri-face row");

    for &(a, b) in &SHARED_EDGES {
        let dofs = variant_dofs(&dm, (a, b), 3);
        assert_eq!(dofs.len(), 2, "pyr p3 variant of edge ({a},{b})");
        let mid = variant_dofs(&dm, (a, b), 2);
        assert_eq!(mid.len(), 1);
        // pyramid-side GLL DOF COORDINATES (per-variant basis in the builder)
        let (ca, cb) = (dm.dof_coord(a as u32), dm.dof_coord(b as u32));
        for (k, &dof) in dofs.iter().enumerate() {
            let t = [cp1(), cp2()][k];
            let xyz = dm.dof_coord(dof);
            for d in 0..3 {
                assert!((xyz[d] - ((1.0 - t) * ca[d] + t * cb[d])).abs() < 1e-12,
                    "pyr p3 edge dof {dof} GLL coordinate along ({a},{b}): got {xyz:?}");
            }
        }
        // constraint rows: GLL slave positions on the p2 master
        for (k, &dof) in dofs.iter().enumerate() {
            let t = [cp1(), cp2()][k];
            let w = master_weights(2, t, false);
            let expected = vec![(a, w[0]), (b, w[2]), (mid[0], w[1])];
            let rows = rows_of(&constraints, &[dof]);
            assert_weights(&rows[0].1, &expected, &format!("edge ({a},{b}) dof@t={t}"));
        }
    }

    // Identity: a quadratic field satisfies every constraint exactly.
    let n = dm.n_dofs as usize;
    let u: Vec<f64> = (0..n as u32)
        .map(|d| quad_field(dm.dof_coord(d))).collect();
    assert_identities(&dm, &constraints, &u);
}

/// The same physical mesh with the element file order swapped must produce
/// the same constraints (matched by dof coordinates).  Pre-fix, orders
/// [3, 2] with the pyramid first took `gll = true` from element 0 and
/// mis-placed the TET-side slave positions at the GLL points.
#[test]
fn d827_constraints_independent_of_element_file_order() {
    let mesh_t = tet_first_mesh();
    let mesh_p = pyr_first_mesh();
    assert_eq!(mesh_p.element_type(0), fem_mesh::ElementType::Pyramid5);

    for orders_tp in [[3u8, 2], [2, 3], [4, 3]] {
        // tet-first orders (tet, pyr); pyr-first file swaps the elements.
        let orders_pf = [orders_tp[1], orders_tp[0]];
        let dm_t = build_variable_order_dof_manager(&mesh_t, &orders_tp);
        let dm_p = build_variable_order_dof_manager(&mesh_p, &orders_pf);
        let cs_t = detect_p_constraints(&dm_t, &mesh_t, &orders_tp);
        let cs_p = detect_p_constraints(&dm_p, &mesh_p, &orders_pf);
        assert_eq!(cs_t.len(), cs_p.len(),
            "{orders_tp:?}: constraint count must not depend on the file order");

        // key: constrained dof coordinate rounded; value: parent
        // (coordinate, weight) pairs sorted.
        let key = |dm: &DofManager, d: u32| -> [i64; 3] {
            dm.dof_coord(d).iter()
                .map(|&x| (x * 1e9).round() as i64).collect::<Vec<_>>()
                .try_into().unwrap()
        };
        let mut map_t = std::collections::BTreeMap::new();
        for c in &cs_t {
            let mut ps: Vec<([i64; 3], f64)> = c.parents.iter()
                .map(|&(d, w)| (key(&dm_t, d), w)).collect();
            ps.sort_by(|a, b| a.0.cmp(&b.0));
            map_t.insert(key(&dm_t, c.constrained), ps);
        }
        let mut map_p = std::collections::BTreeMap::new();
        for c in &cs_p {
            let mut ps: Vec<([i64; 3], f64)> = c.parents.iter()
                .map(|&(d, w)| (key(&dm_p, d), w)).collect();
            ps.sort_by(|a, b| a.0.cmp(&b.0));
            map_p.insert(key(&dm_p, c.constrained), ps);
        }
        assert_eq!(map_t, map_p,
            "{orders_tp:?}: constraint rows must not depend on the file order");
    }
}

/// Tet p4 x pyramid p3, tet first: the master is the pyramid-side p3 variant
/// (GLL nodes {0, cp1, cp2, 1}) and the slaves are the tet-side p4 dofs at
/// the equispaced positions {1/4, 1/2, 3/4} — master and slave bases differ
/// ON THE SAME ROW SET.  Pre-fix the master nodes were taken equispaced
/// (element 0 = tet): red.
#[test]
fn d827_tet_p4_pyr_p3_master_gll_nodes_slave_equispaced() {
    let mesh = tet_first_mesh();
    let orders: Vec<u8> = vec![4, 3];
    let dm = build_variable_order_dof_manager(&mesh, &orders);
    let constraints = detect_p_constraints(&dm, &mesh, &orders);

    // 3 shared edges x 3 tet p4 edge dofs + mixed tri face (p4 tet side:
    // 3 interior dofs; p3 pyr side masters) + ... count first:
    let edge_rows: Vec<_> = constraints.iter()
        .filter(|c| c.parents.len() == 4).collect();
    assert_eq!(edge_rows.len(), 9, "3 shared edges x 3 tet p4 edge dofs");

    for &(a, b) in &SHARED_EDGES {
        let slave = variant_dofs(&dm, (a, b), 4);
        assert_eq!(slave.len(), 3);
        let master = variant_dofs(&dm, (a, b), 3);
        assert_eq!(master.len(), 2, "pyr p3 variant: cp1/cp2 dofs");
        // per-variant coordinates on one edge: tet p4 at equispaced
        // 1/4, 1/2, 3/4; pyr p3 at GLL cp1, cp2.
        let (ca, cb) = (dm.dof_coord(a as u32), dm.dof_coord(b as u32));
        for (k, &dof) in slave.iter().enumerate() {
            let t = (k + 1) as f64 / 4.0;
            for d in 0..3 {
                assert!((dm.dof_coord(dof)[d]
                    - ((1.0 - t) * ca[d] + t * cb[d])).abs() < 1e-12,
                    "tet p4 dof {dof} equispaced coordinate");
            }
        }
        for (k, &dof) in master.iter().enumerate() {
            let t = [cp1(), cp2()][k];
            for d in 0..3 {
                assert!((dm.dof_coord(dof)[d]
                    - ((1.0 - t) * ca[d] + t * cb[d])).abs() < 1e-12,
                    "pyr p3 dof {dof} GLL coordinate");
            }
        }
        // rows: equispaced slave positions evaluated on the GLL p3 master
        for (k, &dof) in slave.iter().enumerate() {
            let t = (k + 1) as f64 / 4.0;
            let w = master_weights(3, t, true);
            let expected = vec![(a, w[0]), (b, w[3]), (master[0], w[1]), (master[1], w[2])];
            let rows = rows_of(&constraints, &[dof]);
            assert_weights(&rows[0].1, &expected, &format!("edge ({a},{b}) p4 dof@t={t}"));
        }
    }

    // Identity: a cubic field satisfies every constraint exactly.
    let n = dm.n_dofs as usize;
    let u: Vec<f64> = (0..n as u32)
        .map(|d| cubic_field(dm.dof_coord(d))).collect();
    assert_identities(&dm, &constraints, &u);
}

/// Same-order shared edge (tet p3 x pyr p3): MFEM stores ONE variant per
/// order (probe out_p33_gl.txt: shared-edge variant orders "3", no
/// constraints, NDofs=47) and never per family.  fem-rs matches the
/// structure (one variant, no constraints, NDofs 47).  D829-1 (round 87)
/// replaced the first-encounter basis donation with the sticky-GLL ruling;
/// the file-order invariance of the shared variant's node set is pinned in
/// `d829_file_order_invariant_variant_basis.rs`.
#[test]
fn d827_same_order_shared_edge_single_variant_matches_mfem_ndofs() {
    let mesh = tet_first_mesh();
    let orders: Vec<u8> = vec![3, 3];
    let dm = build_variable_order_dof_manager(&mesh, &orders);
    let constraints = detect_p_constraints(&dm, &mesh, &orders);
    assert!(constraints.is_empty(), "no mixed-order constraints at p=[3,3]");

    // MFEM probe NDofs=47 (6 verts + 11x2 edge + 6 tri-face + 1 shared tri
    // + 4 base quad + 8 pyramid interior; linear rows, no gaps).
    assert_eq!(dm.n_dofs, 47, "probe out_p33_gl.txt NDofs");

    for &(a, b) in &SHARED_EDGES {
        let variants = &dm.edge_variants[&EdgeKey::new(a, b)];
        assert_eq!(variants.len(), 1, "one variant per order, not per family");
        assert_eq!(variants[0].0, 3);
        assert_eq!(variants[0].1.len(), 2);
    }
}

/// Negative control (homogeneous tet pair, orders [2, 3]): the tet-pair
/// edge rows must remain the equispaced {2/9, 8/9, -1/9} pattern — the
/// per-variant dispatch changes nothing on single-family meshes.
#[test]
fn d827_homogeneous_tet_pair_unchanged() {
    let mesh = Mesh::<3>::uniform(
        vec![
            0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 1.,
        ],
        vec![0, 1, 2, 3, 1, 2, 3, 4],
        vec![1, 1],
        fem_mesh::ElementType::Tet4,
        vec![],
        vec![],
        fem_mesh::ElementType::Tri3,
    );
    let orders = [2u8, 3];
    let dm = build_variable_order_dof_manager(&mesh, &orders);
    let constraints = detect_p_constraints(&dm, &mesh, &orders);

    // 3 shared edges x 2 p3 edge dofs + 1 face dof (d176 structure).
    assert_eq!(constraints.len(), 7);
    // the tet pair shares face {1,2,3}: edges (1,2), (1,3), (2,3)
    let shared_edges = [(1u32, 2u32), (1, 3), (2, 3)];
    for &(a, b) in &shared_edges {
        let dofs = variant_dofs(&dm, (a, b), 3);
        assert_eq!(dofs.len(), 2);
        let mid = variant_dofs(&dm, (a, b), 2);
        assert_eq!(mid.len(), 1);
        for (k, &dof) in dofs.iter().enumerate() {
            let t = (k + 1) as f64 / 3.0;
            let w = master_weights(2, t, false);
            let expected = vec![(a, w[0]), (b, w[2]), (mid[0], w[1])];
            let rows = rows_of(&constraints, &[dof]);
            assert_weights(&rows[0].1, &expected, &format!("tet pair edge ({a},{b}) t={t}"));
        }
    }
}

/// 2D NC (hanging-node) hp guard for the D827-1 propagation sharing: the
/// detector recomputes the per-variant basis map and MUST re-apply the 2D
/// NC propagation to it — the master edge (1,2) here gains the propagated
/// p1 variant only through
/// `propagate_nc_2d_edge_variants` (the coarse cell is p2 and one fine cell
/// is p1, so `collect_edge_variants` alone does not list that order on the
/// master edge, and the basis lookup panics without the shared propagation
/// — verified).  Same-family quads, so all bases stay GLL and the rows are
/// the classic 0.5/0.5 hanging-node weights.
#[test]
fn d827_nc_hanging_propagated_variant_basis_lookup() {
    // Quad A (coarse, left): (0,0) (1,0) (1,2) (0,2); quads B/C (fine,
    // right): hanging node 6 at (1,1) on master edge (1,2).
    let mesh = Mesh::<2>::uniform(
        vec![
            0., 0., 1., 0., 1., 2., 0., 2., // 0..3: quad A
            2., 0., 2., 1., 1., 1., 2., 2., // 4 (2,0), 5 (2,1), 6 (1,1), 7 (2,2)
        ],
        vec![
            0, 1, 2, 3, // A
            1, 4, 5, 6, // B = (1,0) (2,0) (2,1) (1,1)
            6, 5, 7, 2, // C = (1,1) (2,1) (2,2) (1,2)
        ],
        vec![1, 1, 1],
        fem_mesh::ElementType::Quad4,
        vec![],
        vec![],
        fem_mesh::ElementType::Line2,
    );
    assert_eq!(mesh.n_nodes(), 8);
    assert_eq!(mesh.n_elements(), 3);

    // A p2, B p1, C p2: master edge (1,2) collects only A's p2 variant, and
    // the propagation ADDS the slaves' p1 variant to it.
    let orders: Vec<u8> = vec![2, 1, 2];
    let dm = build_variable_order_dof_manager(&mesh, &orders);
    let constraints = detect_p_constraints(&dm, &mesh, &orders);

    // Master edge (1,2) now holds variants {1, 2}: its p2 variant dof is
    // constrained to the p1 endpoints with 0.5/0.5.
    let master_variants = &dm.edge_variants[&EdgeKey::new(1, 2)];
    let orders_on_master: Vec<u8> = master_variants.iter().map(|(p, _)| *p).collect();
    assert_eq!(orders_on_master, vec![1, 2], "propagated p1 variant present");
    let mid2 = &master_variants.iter().find(|(p, _)| *p == 2).unwrap().1;
    assert_eq!(mid2.len(), 1);
    let row = &constraints.iter().find(|c| c.constrained == mid2[0])
        .unwrap_or_else(|| panic!("master edge p2 dof unconstrained"))
        .parents;
    assert_weights(row, &[(1, 0.5), (2, 0.5)], "master (1,2) p2 <- p1");

    // Hanging node 6 <- master p1 trace at t = 0.5.
    let row = &constraints.iter().find(|c| c.constrained == 6)
        .unwrap_or_else(|| panic!("hanging node unconstrained"))
        .parents;
    assert_weights(row, &[(1, 0.5), (2, 0.5)], "hanging node 6");

    // Identity: a linear field satisfies every constraint exactly.
    let n = dm.n_dofs as usize;
    let u: Vec<f64> = (0..n as u32)
        .map(|d| {
            let x = dm.dof_coord(d);
            3.0 * x[0] - 2.0 * x[1] + 1.0
        }).collect();
    assert_identities(&dm, &constraints, &u);
}
